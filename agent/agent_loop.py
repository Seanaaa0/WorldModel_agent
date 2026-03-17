from encoder.state_encoder import StateEncoder
from memory.world_memory import WorldMemory
from monitor.monitor import Monitor
from planner.rule_planner import RulePlanner
from skills.skill_executor import SkillExecutor

import time


class AgentLoop:
    def __init__(self, env, planner=None, sleep_time=0.0, verbose=True):
        self.env = env
        self.verbose = verbose

        self.encoder = StateEncoder()
        self.memory = WorldMemory(recent_window=10)

        self.monitor = Monitor(
            stuck_unique_threshold=2,
            stuck_min_window=6,
            replan_on_wall_hit=True,
            replan_on_stuck=True,
            replan_on_prediction_mismatch=True,
            prediction_error_threshold=4.0,
        )

        # -----------------------------
        # Dual planner setup
        # -----------------------------
        self.fast_planner = RulePlanner()

        # If planner is provided and is NOT RulePlanner, treat it as slow planner
        # (typically LLMPlanner). If planner is None, slow planner stays disabled.
        if planner is None:
            self.slow_planner = None
        elif isinstance(planner, RulePlanner):
            self.fast_planner = planner
            self.slow_planner = None
        else:
            self.slow_planner = planner

        self.executor = SkillExecutor()

        # Predictor is optional for now because the old predictor may still
        # depend on the pre-PO latent schema.
        self.predictor = None
        self.predictor_enabled = False

        try:
            from predictor.mlp_predictor import MLPPredictor

            self.predictor = MLPPredictor(
                checkpoint_path="predictor/checkpoints/jepa_lite_mlp_po_v2.pt"
            )
            self.predictor_enabled = True

            if self.verbose:
                print("[AgentLoop] Predictor loaded successfully.")
        except Exception as e:
            self.predictor = None
            self.predictor_enabled = False
            if self.verbose:
                print(f"[AgentLoop] Predictor disabled: {e}")

        self.current_skill = None
        self.current_skill_steps = 0

        # Keep small because V3 macro skills already compress multiple steps.
        self.max_cached_skill_steps = 1

        self.sleep_time = sleep_time

        # experiment counters
        self.slow_planner_calls = 0
        self.fast_planner_calls = 0
        self.scan_count = 0

        # Routing state:
        # count consecutive local failures before escalating to slow planner.
        self.consecutive_local_failures = 0

    def _cached_skill_is_still_valid(self, z_t):
        if self.current_skill is None:
            return False

        skill_name = self.current_skill.get("skill")
        skill_args = self.current_skill.get("args", {})
        walls = z_t["local_walls"]

        # Primitive move validity
        if skill_name == "move":
            direction = skill_args.get("direction", "").lower()
            if direction not in walls:
                return False
            if walls[direction]:
                return False
            return True

        # Macro move validity:
        # only keep cached if the first direction is still open.
        if skill_name in {"move_k_steps", "move_until_blocked"}:
            direction = skill_args.get("direction", "").lower()
            if direction not in walls:
                return False
            if walls[direction]:
                return False
            return True

        # scan / escape_loop should usually be one-shot skills
        if skill_name in {"scan", "escape_loop"}:
            return False

        return True

    def _safe_abs_diff(self, a, b):
        if a is None or b is None:
            return 0.0
        return abs(a - b)

    def _compute_prediction_signal(self, z_hat, z_real):
        """
        PO-safe prediction comparison.

        Since PO latent may contain None for:
        - goal_pos
        - dx
        - dy
        - goal_distance

        we only compare fields when they are available.
        """
        pos_mismatch = int(z_hat.get("agent_pos") != z_real.get("agent_pos"))

        goal_distance_error = self._safe_abs_diff(
            z_hat.get("goal_distance"), z_real.get("goal_distance")
        )
        dx_error = self._safe_abs_diff(z_hat.get("dx"), z_real.get("dx"))
        dy_error = self._safe_abs_diff(z_hat.get("dy"), z_real.get("dy"))

        goal_visible_mismatch = int(
            z_hat.get("goal_visible", False) != z_real.get(
                "goal_visible", False)
        )

        wall_mismatch = 0
        comparable_wall_count = 0

        pred_walls = z_hat.get("local_walls", {})
        real_walls = z_real.get("local_walls", {})

        for k in ["up", "down", "left", "right"]:
            pred_v = pred_walls.get(k)
            real_v = real_walls.get(k)

            if pred_v is None or real_v is None:
                continue

            comparable_wall_count += 1
            if pred_v != real_v:
                wall_mismatch += 1

        total_error = (
            2.0 * pos_mismatch
            + 1.0 * goal_distance_error
            + 0.5 * dx_error
            + 0.5 * dy_error
            + 0.5 * goal_visible_mismatch
            + 0.5 * wall_mismatch
        )

        return {
            "pos_mismatch": pos_mismatch,
            "goal_distance_error": goal_distance_error,
            "dx_error": dx_error,
            "dy_error": dy_error,
            "goal_visible_mismatch": goal_visible_mismatch,
            "wall_mismatch": wall_mismatch,
            "comparable_wall_count": comparable_wall_count,
            "total_error": total_error,
        }

    def _should_use_slow_planner(
        self,
        z_t,
        memory_summary,
        decision,
        last_info,
    ):
        """
        Event-triggered slow planner.

        Important distinction:
        - REPLAN means "pick a new skill"
        - SLOW means "the situation is strategic enough to justify LLM cost"

        Therefore, local failures stay in FAST by default, while major
        information / phase changes escalate to SLOW.
        """
        if self.slow_planner is None:
            return False

        # 1) Just scanned: new information arrived.
        if last_info is not None and last_info.get("scan", False):
            return True

        # 2) First time goal becomes visible / known.
        if (
            z_t.get("goal_visible", False)
            and memory_summary.get("seen_goal_count", 0) <= 1
        ):
            return True

        # 3) Strong stuck / oscillation -> strategic reset.
        if self.memory.is_stuck_by_repetition():
            return True

        # 4) escape_loop is a meaningful high-level recovery skill.
        if (
            last_info is not None
            and last_info.get("macro_skill") == "escape_loop"
        ):
            return True

        # 5) Escalate only after repeated local failures.
        if self.consecutive_local_failures >= 2:
            return True

        # Default: local replans remain inside FAST.
        return False

    def _select_planner(
        self,
        z_t,
        memory_summary,
        decision,
        last_info,
    ):
        use_slow = self._should_use_slow_planner(
            z_t=z_t,
            memory_summary=memory_summary,
            decision=decision,
            last_info=last_info,
        )

        if use_slow:
            self.slow_planner_calls += 1
            if self.verbose:
                print("[Planner Routing] Using SLOW planner")
            return self.slow_planner, "slow"

        self.fast_planner_calls += 1
        if self.verbose:
            print("[Planner Routing] Using FAST planner")
        return self.fast_planner, "fast"

    def _build_planner_context(self, z_t):
        """
        Build V3 planner-facing structured memory context.
        """
        return self.memory.get_planner_context(
            agent_pos=z_t["agent_pos"],
            patch_radius=3,
            top_k_frontiers=5,
        )

    def _choose_skill_with_compat(
        self,
        planner,
        z_t,
        memory_summary,
        planner_context,
        decision,
        last_info,
    ):
        """
        Backward-compatible planner call.

        New V3 planners can accept:
        - memory_patch
        - frontier_candidates
        - loop_hints
        - planner_context

        Old V2 planners may only accept:
        - z_t
        - memory_summary
        - replan
        - last_info
        """
        try:
            return planner.choose_skill(
                z_t=z_t,
                memory_summary=memory_summary,
                memory_patch=planner_context["memory_patch"],
                frontier_candidates=planner_context["frontier_candidates"],
                loop_hints=planner_context["loop_hints"],
                planner_context=planner_context,
                replan=(decision == "REPLAN"),
                last_info=last_info,
            )
        except TypeError:
            return planner.choose_skill(
                z_t=z_t,
                memory_summary=memory_summary,
                replan=(decision == "REPLAN"),
                last_info=last_info,
            )

    def _should_invalidate_after_execution(
        self,
        skill_spec: dict,
        info: dict,
        z_tp1: dict,
        memory_summary_before_step: dict,
        monitor_prediction: dict,
    ) -> bool:
        """
        Decide whether cached skill should be dropped after execution.

        Main principle:
        - scan / escape_loop are one-shot
        - macro skills are one-shot by default
        - wall hit / OOB / monitor replan => invalidate
        - newly visible goal => invalidate
        """
        skill_name = skill_spec.get("skill", "")

        if monitor_prediction["decision"] == "REPLAN":
            return True

        if info.get("hit_wall") or info.get("out_of_bounds"):
            return True

        if skill_name in {"scan", "escape_loop"}:
            return True

        if skill_name in {"move_k_steps", "move_until_blocked"}:
            return True

        # If goal is newly seen after this transition, force replanning.
        if (
            z_tp1.get("goal_visible", False)
            and memory_summary_before_step.get("seen_goal_count", 0) == 0
        ):
            return True

        # If macro skill reported short execution / interrupted execution,
        # invalidate to let planner reconsider.
        if info.get("macro_skill") in {"move_k_steps", "move_until_blocked"}:
            requested_k = info.get("requested_k")
            requested_max_k = info.get("requested_max_k")
            actual_steps = info.get("actual_steps", 0)

            if requested_k is not None and actual_steps < requested_k:
                return True

            if requested_max_k is not None and actual_steps < requested_max_k:
                return True

        return False

    def run(self, max_steps=200):
        obs_t = self.env.reset()
        last_info = None
        self.consecutive_local_failures = 0

        if self.verbose:
            print("=== Agent Loop Start ===")
            print(f"Fast Planner: {self.fast_planner.__class__.__name__}")
            print(
                "Slow Planner:",
                self.slow_planner.__class__.__name__ if self.slow_planner is not None else "None",
            )
            print("Environment reset.\n")
            self.env.render()

        for step in range(max_steps):
            z_t = self.encoder.encode(obs_t)

            # Update memory from current real observation.
            self.memory.update(z_t, last_info)
            memory_summary = self.memory.get_summary()
            planner_context = self._build_planner_context(z_t)

            monitor_result = self.monitor.decide(
                z_t=z_t,
                memory=self.memory,
                last_info=last_info,
                prediction_signal=None,
            )

            decision = monitor_result["decision"]

            if self.verbose:
                print(f"[Step {step + 1}]")
                print("z_t =", z_t)
                print("memory_summary =", memory_summary)
                print("memory_patch =", planner_context["memory_patch"])
                print("frontier_candidates =",
                      planner_context["frontier_candidates"])
                print("loop_hints =", planner_context["loop_hints"])
                print("monitor =", monitor_result)

            if decision == "STOP":
                print("Monitor STOP:", monitor_result["reason"])
                break

            need_new_skill = (
                self.current_skill is None
                or decision == "REPLAN"
                or not self._cached_skill_is_still_valid(z_t)
                or self.current_skill_steps >= self.max_cached_skill_steps
            )

            planner_name = "cached"

            if need_new_skill:
                planner_to_use, planner_name = self._select_planner(
                    z_t=z_t,
                    memory_summary=memory_summary,
                    decision=decision,
                    last_info=last_info,
                )

                self.current_skill = self._choose_skill_with_compat(
                    planner=planner_to_use,
                    z_t=z_t,
                    memory_summary=memory_summary,
                    planner_context=planner_context,
                    decision=decision,
                    last_info=last_info,
                )
                self.current_skill_steps = 0

            skill_spec = self.current_skill

            if skill_spec["skill"] == "scan":
                self.scan_count += 1

            if self.verbose:
                print("planner_used =", planner_name)
                print("chosen_skill =", skill_spec)

            # Predictor branch (optional)
            z_hat_tp1 = None
            prediction_signal = None

            if self.predictor_enabled and self.predictor is not None:
                try:
                    z_hat_tp1 = self.predictor.predict_next_state(
                        z_t,
                        skill_spec,
                    )
                except Exception as e:
                    z_hat_tp1 = None
                    prediction_signal = None
                    if self.verbose:
                        print(f"[AgentLoop] Predictor step skipped: {e}")

            execution_result = self.executor.execute(
                self.env,
                skill_spec,
            )

            obs_tp1 = execution_result["obs"]
            done = execution_result["done"]
            info = execution_result["info"]

            z_tp1 = self.encoder.encode(obs_tp1)

            if z_hat_tp1 is not None:
                try:
                    prediction_signal = self._compute_prediction_signal(
                        z_hat_tp1,
                        z_tp1,
                    )
                except Exception as e:
                    prediction_signal = None
                    if self.verbose:
                        print(f"[AgentLoop] Prediction compare skipped: {e}")

            if self.verbose and prediction_signal is not None:
                print(
                    f"[Pred] z_hat={z_hat_tp1.get('agent_pos')} "
                    f"z_real={z_tp1.get('agent_pos')} "
                    f"err={prediction_signal}"
                )

            monitor_prediction = self.monitor.decide(
                z_t=z_tp1,
                memory=self.memory,
                last_info=info,
                prediction_signal=prediction_signal,
            )

            had_local_failure = bool(
                info.get("hit_wall")
                or info.get("out_of_bounds")
                or monitor_prediction.get("reason") == "prediction_mismatch"
            )
            if had_local_failure:
                self.consecutive_local_failures += 1
            else:
                self.consecutive_local_failures = 0

            invalidate_skill = self._should_invalidate_after_execution(
                skill_spec=skill_spec,
                info=info,
                z_tp1=z_tp1,
                memory_summary_before_step=memory_summary,
                monitor_prediction=monitor_prediction,
            )

            if invalidate_skill:
                self.current_skill = None

            if self.current_skill is None:
                self.current_skill_steps = 0
            else:
                self.current_skill_steps += 1

            if self.verbose:
                print("execution_result =", execution_result)
                print(
                    f"[Fast Layer] invalidate_skill = {invalidate_skill}, "
                    f"current_skill_steps = {self.current_skill_steps}, "
                    f"current_skill = {self.current_skill}, "
                    f"consecutive_local_failures = {self.consecutive_local_failures}"
                )
                self.env.render()

            obs_t = obs_tp1
            last_info = info

            if self.sleep_time > 0:
                time.sleep(self.sleep_time)

            if done:
                if last_info is not None and last_info.get("goal_reached", False):
                    print("Goal reached!")
                else:
                    print("Episode finished.")
                break

        if self.verbose:
            print("=== Agent Loop Finished ===")
            final_z = self.encoder.encode(obs_t)
            print("final_z =", final_z)

            if last_info is not None and last_info.get("goal_reached", False):
                print("Result: Goal reached!")
            elif last_info is not None and last_info.get("max_steps_reached", False):
                print("Result: Max steps reached.")
            else:
                print("Result: Stopped by monitor or loop ended.")

        print(f"SLOW_PLANNER_CALLS: {self.slow_planner_calls}")
        print(f"FAST_PLANNER_CALLS: {self.fast_planner_calls}")
        print(f"SCAN_COUNT: {self.scan_count}")

        if last_info is not None and last_info.get("goal_reached", False):
            return True, step + 1
        else:
            return False, step + 1

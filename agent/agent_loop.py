from encoder.state_encoder import StateEncoder
from memory.world_memory import WorldMemory
from monitor.monitor import Monitor
from planner.rule_planner import RulePlanner
from skills.skill_executor import SkillExecutor

import time


class PhaseController:
    """
    Thin adapter:
    - LLM decides current PHASE
    - RulePlanner executes that phase through BFS / frontier / local fallback
    """

    def __init__(self, executor_planner=None):
        self.executor_planner = executor_planner or RulePlanner()

    def choose_skill(
        self,
        phase_decision: dict | None,
        z_t: dict,
        memory_summary: dict,
        planner_context: dict,
        decision: str,
        last_info: dict | None,
    ) -> dict:
        phase = None
        phase_reason = None

        if phase_decision is not None:
            phase = phase_decision.get("phase", None)
            phase_reason = phase_decision.get("reason", None)

        # Hard recover stays outside RulePlanner
        if phase == "recover":
            return {"skill": "escape_loop", "args": {}}

        planner_context = dict(planner_context or {})
        planner_context["forced_phase"] = phase
        planner_context["phase_reason"] = phase_reason

        return self.executor_planner.choose_skill(
            z_t=z_t,
            memory_summary=memory_summary,
            memory_patch=planner_context.get("memory_patch"),
            frontier_candidates=planner_context.get("frontier_candidates"),
            loop_hints=planner_context.get("loop_hints"),
            planner_context=planner_context,
            replan=(decision == "REPLAN"),
            last_info=last_info,
        )


class AgentLoop:
    """
    V5-b phase-based agent loop.

    Core split:
    - slow planner (LLM): choose / update PHASE only
    - fast planner (RulePlanner): execute phase using BFS / frontier exploration
    - executor: still executes primitive skills

    Design goal:
    - remove LLM from primitive move selection
    - keep RulePlanner as robust execution backbone
    """

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
            replan_on_prediction_mismatch=False,
            prediction_error_threshold=9999.0,
        )

        # ---------------------------------
        # Planner split
        # ---------------------------------
        self.fast_planner = RulePlanner()

        if planner is None:
            self.slow_planner = None
        elif isinstance(planner, RulePlanner):
            self.fast_planner = planner
            self.slow_planner = None
        else:
            self.slow_planner = planner

        self.phase_controller = PhaseController(
            executor_planner=self.fast_planner)
        self.executor = SkillExecutor()

        # Predictor disabled in this version
        self.predictor = None
        self.predictor_enabled = False

        if self.verbose:
            print("[AgentLoop] Predictor disabled.")
            print("[AgentLoop] V5-b mode: LLM decides PHASE, RulePlanner executes.")

        self.current_skill = None
        self.current_skill_steps = 0
        self.max_cached_skill_steps = 1

        self.current_phase_decision = None

        self.sleep_time = sleep_time

        # experiment counters
        self.slow_planner_calls = 0
        self.fast_planner_calls = 0
        self.scan_count = 0

        # routing state
        self.consecutive_local_failures = 0

    # =========================================================
    # Skill cache validity
    # =========================================================

    def _cached_skill_is_still_valid(self, z_t):
        if self.current_skill is None:
            return False

        skill_name = self.current_skill.get("skill")
        skill_args = self.current_skill.get("args", {})
        walls = z_t["local_walls"]

        if skill_name == "move":
            direction = skill_args.get("direction", "").lower()
            if direction not in walls:
                return False
            if walls[direction]:
                return False
            return True

        if skill_name in {"scan", "escape_loop"}:
            return False

        return False

    # =========================================================
    # Slow planner routing (phase updates only)
    # =========================================================

    def _should_use_slow_planner(
        self,
        z_t,
        memory_summary,
        decision,
        last_info,
    ):
        """
        Tight routing for V5-b:
        slow planner only updates high-level phase when phase-relevant events happen.
        """

        if self.slow_planner is None:
            return False

        # 0) episode start / no phase yet
        if self.current_phase_decision is None:
            return True

        current_phase = self.current_phase_decision.get("phase", None)

        # 1) explicit task phase transitions
        if last_info is not None and last_info.get("picked_key", False):
            return True

        if last_info is not None and last_info.get("opened_door", False):
            return True

        # 2) only update to goal phase when goal becomes relevant
        #    goal visible alone is NOT enough before key / before open door
        has_key = bool(z_t.get("has_key", False))
        goal_visible = bool(z_t.get("goal_visible", False))
        known_door_open = bool(memory_summary.get("known_door_open", False))
        visible_door_open = bool(z_t.get("visible_door_open", False))
        door_open = known_door_open or visible_door_open

        if has_key and door_open and goal_visible and current_phase != "go_to_goal":
            return True

        # 3) hard stuck / oscillation
        if self.memory.is_stuck_by_repetition():
            return True

        loop_hints = self.memory.get_loop_hints()
        if loop_hints.get("repeat_count_current_pos", 0) >= 3:
            return True

        if loop_hints.get("oscillation_pair") is not None:
            return True

        # 4) after escape_loop, let slow planner update phase again
        if (
            last_info is not None
            and last_info.get("macro_skill") == "escape_loop"
        ):
            return True

        # 5) repeated local failures
        if self.consecutive_local_failures >= 2:
            return True

        # 6) DO NOT refresh phase on generic REPLAN by default.
        # local replans should usually stay inside the current phase.
        return False

    def _update_phase_if_needed(
        self,
        z_t,
        memory_summary,
        planner_context,
        decision,
        last_info,
    ):
        if not self._should_use_slow_planner(
            z_t=z_t,
            memory_summary=memory_summary,
            decision=decision,
            last_info=last_info,
        ):
            return

        if self.slow_planner is None:
            return

        self.slow_planner_calls += 1

        if self.verbose:
            print("[Planner Routing] Updating PHASE via SLOW planner")

        self.current_phase_decision = self.slow_planner.choose_phase(
            z_t=z_t,
            memory_summary=memory_summary,
            memory_patch=planner_context.get("memory_patch"),
            frontier_candidates=planner_context.get("frontier_candidates"),
            loop_hints=planner_context.get("loop_hints"),
            planner_context=planner_context,
            replan=(decision == "REPLAN"),
            last_info=last_info,
        )

        if self.verbose:
            print("[Phase Update] current_phase_decision =",
                  self.current_phase_decision)

    # =========================================================
    # Planner context
    # =========================================================

    def _build_planner_context(self, z_t):
        return self.memory.get_planner_context(
            agent_pos=z_t["agent_pos"],
            patch_radius=3,
            top_k_frontiers=5,
        )

    # =========================================================
    # Skill invalidation
    # =========================================================

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
        - wall hit / OOB / monitor replan => invalidate
        - key pickup / door open => invalidate (phase change)
        - goal visible newly discovered => invalidate
        """
        skill_name = skill_spec.get("skill", "")

        if monitor_prediction["decision"] == "REPLAN":
            return True

        if info.get("hit_wall") or info.get("out_of_bounds"):
            return True

        if skill_name in {"scan", "escape_loop"}:
            return True

        if info.get("picked_key", False):
            return True

        if info.get("opened_door", False):
            return True

        if (
            z_tp1.get("goal_visible", False)
            and memory_summary_before_step.get("seen_goal_count", 0) == 0
        ):
            return True

        return False

    # =========================================================
    # Main run loop
    # =========================================================

    def run(self, max_steps=200):
        obs_t = self.env.reset()
        self.memory.reset()

        last_info = None
        self.consecutive_local_failures = 0
        self.current_skill = None
        self.current_skill_steps = 0
        self.current_phase_decision = None

        self.slow_planner_calls = 0
        self.fast_planner_calls = 0
        self.scan_count = 0

        if self.verbose:
            print("=== Agent Loop Start ===")
            print(
                f"Fast Planner (executor): {self.fast_planner.__class__.__name__}")
            print(
                "Slow Planner (phase):",
                self.slow_planner.__class__.__name__ if self.slow_planner is not None else "None",
            )
            print("Environment reset.\n")
            self.env.render()

        final_step = 0

        for step in range(max_steps):
            final_step = step + 1

            z_t = self.encoder.encode(obs_t)

            # update memory from current real observation
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
                print("current_phase_decision =", self.current_phase_decision)

            if decision == "STOP":
                print("Monitor STOP:", monitor_result["reason"])
                break

            # update PHASE only when needed
            self._update_phase_if_needed(
                z_t=z_t,
                memory_summary=memory_summary,
                planner_context=planner_context,
                decision=decision,
                last_info=last_info,
            )

            need_new_skill = (
                self.current_skill is None
                or decision == "REPLAN"
                or not self._cached_skill_is_still_valid(z_t)
                or self.current_skill_steps >= self.max_cached_skill_steps
            )

            if need_new_skill:
                self.fast_planner_calls += 1

                self.current_skill = self.phase_controller.choose_skill(
                    phase_decision=self.current_phase_decision,
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
                print("chosen_skill =", skill_spec)

            execution_result = self.executor.execute(
                self.env,
                skill_spec,
            )

            obs_tp1 = execution_result["obs"]
            done = execution_result["done"]
            info = execution_result["info"]

            z_tp1 = self.encoder.encode(obs_tp1)

            monitor_prediction = self.monitor.decide(
                z_t=z_tp1,
                memory=self.memory,
                last_info=info,
                prediction_signal=None,
            )

            had_local_failure = bool(
                info.get("hit_wall")
                or info.get("out_of_bounds")
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
            print("final_phase_decision =", self.current_phase_decision)

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
            return True, final_step
        else:
            return False, final_step

from encoder.state_encoder import StateEncoder
from memory.world_memory import WorldMemory
from monitor.monitor import Monitor
from planner.rule_planner import RulePlanner
from skills.skill_executor import SkillExecutor
from planner.predictive_planner_v8 import PredictivePlannerV8

import time


class PhaseController:
    """
    Thin adapter:
    - LLM decides current PHASE
    - fast planner executes that phase
    - recover phase is handled here using loop-aware local escape
    """

    DIR_TO_DELTA = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
    }

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

        planner_context = dict(planner_context or {})
        planner_context["forced_phase"] = phase
        planner_context["phase_reason"] = phase_reason

        # -------------------------------------------------
        # Hard recover stays outside fast planner
        # -------------------------------------------------
        if phase == "recover":
            return self._choose_recover_skill(
                z_t=z_t,
                memory_summary=memory_summary,
                planner_context=planner_context,
                last_info=last_info,
            )

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

    def _choose_recover_skill(
        self,
        z_t: dict,
        memory_summary: dict,
        planner_context: dict,
        last_info: dict | None,
    ) -> dict:
        """
        Recovery policy:
        1. Avoid moving back into oscillation pair cells if possible
        2. Prefer legal neighbors with lower visit counts
        3. Prefer not to immediately reverse the last action
        4. If no safe move exists, fall back to scan
        """
        agent_pos = tuple(z_t["agent_pos"])
        walls = z_t["local_walls"]
        loop_hints = planner_context.get("loop_hints", {}) or {}

        oscillation_pair = loop_hints.get("oscillation_pair", None)
        recent_positions = loop_hints.get("recent_positions", []) or []

        banned_positions = set()
        if oscillation_pair is not None:
            for p in oscillation_pair:
                banned_positions.add(tuple(p))

        visit_counts = memory_summary.get("visit_counts", {}) or {}

        last_action = None
        if last_info is not None:
            last_action = last_info.get("action", None)
            if isinstance(last_action, str):
                last_action = last_action.upper()

        reverse_of_last = {
            "UP": "DOWN",
            "DOWN": "UP",
            "LEFT": "RIGHT",
            "RIGHT": "LEFT",
        }.get(last_action, None)

        candidates = []

        for direction, (dr, dc) in self.DIR_TO_DELTA.items():
            wall_key = direction.lower()
            if walls.get(wall_key, False):
                continue

            next_pos = (agent_pos[0] + dr, agent_pos[1] + dc)

            score = 0.0

            # Strong penalty: do not stay inside oscillation pair if avoidable
            if next_pos in banned_positions:
                score -= 100.0
            else:
                score += 20.0

            # Penalize immediate reverse of last action
            if reverse_of_last is not None and direction == reverse_of_last:
                score -= 15.0

            # Penalize revisits
            score -= 2.0 * float(visit_counts.get(next_pos, 0))

            # Prefer positions not in recent positions
            if next_pos not in [tuple(p) for p in recent_positions]:
                score += 8.0

            # Mild preference for unexplored / less visited area
            if visit_counts.get(next_pos, 0) == 0:
                score += 5.0

            candidates.append((score, direction, next_pos))

        if not candidates:
            return {"skill": "scan", "args": {}}

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_direction, best_next_pos = candidates[0]

        # If every legal move still stays inside the same oscillation trap, scan instead
        if best_next_pos in banned_positions and len(candidates) == 1:
            return {"skill": "scan", "args": {}}

        # If top choice is still inside banned pair but there exists an outside option, use outside option
        for score, direction, next_pos in candidates:
            if next_pos not in banned_positions:
                return {
                    "skill": "move",
                    "args": {"direction": direction},
                }

        # Otherwise use the best available move
        return {
            "skill": "move",
            "args": {"direction": best_direction},
        }


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

    def __init__(
        self,
        env,
        fast_planner=None,
        slow_planner=None,
        sleep_time=0.0,
        verbose=True,
    ):
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
        # Dual planner split
        # ---------------------------------
        self.fast_planner = fast_planner if fast_planner is not None else RulePlanner()
        self.slow_planner = slow_planner

        self.phase_controller = PhaseController(
            executor_planner=self.fast_planner
        )
        self.executor = SkillExecutor()

        # Predictor disabled in AgentLoop itself.
        # Predictor lives inside fast planner (e.g. PredictivePlannerV8).
        self.predictor = None
        self.predictor_enabled = False

        if self.verbose:
            print("[AgentLoop] Predictor disabled.")
            print("[AgentLoop] V5-b mode: LLM decides PHASE, fast planner executes.")

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
        Tight routing for V5-b / V8:
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

        # 6) generic REPLAN alone does not always mean phase change
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
            print(
                "[Phase Update]",
                {
                    "step": z_t["step_count"],
                    "agent_pos": z_t["agent_pos"],
                    "has_key": z_t["has_key"],
                    "decision": decision,
                    "last_info": {
                        "picked_key": last_info.get("picked_key", False) if last_info else False,
                        "opened_door": last_info.get("opened_door", False) if last_info else False,
                        "hit_wall": last_info.get("hit_wall", False) if last_info else False,
                        "blocked_by_locked_door": last_info.get("blocked_by_locked_door", False) if last_info else False,
                    },
                    "phase": self.current_phase_decision,
                },
            )
    # =========================================================
    # Planner context
    # =========================================================

    def _build_planner_context(self, z_t):
        ctx = self.memory.get_planner_context(
            agent_pos=z_t["agent_pos"],
            patch_radius=3,
            top_k_frontiers=5,
        )
        ctx["memory_obj"] = self.memory
        return ctx
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

            should_print_step_summary = (
                step < 8
                or step % 20 == 0
                or monitor_result["decision"] == "REPLAN"
                or (last_info is not None and last_info.get("picked_key", False))
                or (last_info is not None and last_info.get("opened_door", False))
                or step >= max_steps - 30
            )

            if self.verbose and should_print_step_summary:
                print(f"[Step {step + 1}]")
                print("agent_pos =", z_t["agent_pos"])
                print("has_key =", z_t["has_key"])
                print("visible_key_pos =", z_t.get("visible_key_pos"))
                print("visible_door_pos =", z_t.get("visible_door_pos"))
                print("visible_goal_pos =", z_t.get("visible_goal_pos"))
                print("memory_known_key =", memory_summary.get("known_key_pos"))
                print("memory_known_door =",
                      memory_summary.get("known_door_pos"))
                print("memory_known_door_open =",
                      memory_summary.get("known_door_open"))
                print("memory_known_goal =",
                      memory_summary.get("known_goal_pos"))
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

                if self.verbose and should_print_step_summary:
                    print("chosen_skill =", self.current_skill)

            execution_result = self.executor.execute(
                self.env, self.current_skill)
            self.current_skill_steps += 1

            obs_tp1 = execution_result["obs"]
            done = execution_result["done"]
            info = execution_result["info"]

            if self.current_skill.get("skill") == "scan":
                self.scan_count += 1

            if (
                info.get("hit_wall", False)
                or info.get("out_of_bounds", False)
                or info.get("blocked_by_locked_door", False)
            ):
                self.consecutive_local_failures += 1
            else:
                self.consecutive_local_failures = 0

            z_tp1 = self.encoder.encode(obs_tp1)

            invalidate_skill = self._should_invalidate_after_execution(
                skill_spec=self.current_skill,
                info=info,
                z_tp1=z_tp1,
                memory_summary_before_step=memory_summary,
                monitor_prediction=self.monitor.decide(
                    z_t=z_tp1,
                    memory=self.memory,
                    last_info=info,
                    prediction_signal=None,
                ),
            )

            if invalidate_skill:
                self.current_skill = None
                self.current_skill_steps = 0

            if self.verbose and should_print_step_summary:
                print(
                    "execution_info =",
                    {
                        "old_pos": info.get("old_pos"),
                        "new_pos": info.get("new_pos"),
                        "hit_wall": info.get("hit_wall"),
                        "out_of_bounds": info.get("out_of_bounds"),
                        "blocked_by_locked_door": info.get("blocked_by_locked_door"),
                        "picked_key": info.get("picked_key"),
                        "opened_door": info.get("opened_door"),
                        "goal_reached": info.get("goal_reached"),
                        "step_count": info.get("step_count"),
                    },
                )
                print(
                    "[Fast Layer] invalidate_skill =",
                    invalidate_skill,
                    ", current_skill_steps =",
                    self.current_skill_steps,
                    ", current_skill =",
                    self.current_skill,
                    ", consecutive_local_failures =",
                    self.consecutive_local_failures,
                    sep="",
                )
                print("has_key:", z_tp1.get("has_key", False))
                print(f"steps: {info.get('step_count', step + 1)}/{max_steps}")
                print()

            obs_t = obs_tp1
            last_info = info

            if done:
                if info.get("goal_reached", False):
                    if self.verbose:
                        print("[Episode End] Goal reached.")
                    return True, final_step

                if self.verbose:
                    print("[Episode End] Done without goal.")
                return False, final_step

            if self.sleep_time > 0:
                time.sleep(self.sleep_time)

        if self.verbose:
            print("[Episode End] Max steps reached.")
        return False, final_step

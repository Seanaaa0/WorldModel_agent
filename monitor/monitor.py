class Monitor:
    """
    V5-a task-world runtime monitor.

    Responsibilities:
    - stop if goal is reached
    - trigger replanning on failed movement
    - detect oscillation / getting stuck
    - optionally trigger replanning on prediction mismatch
    - remain compatible with key / door / goal task world schema
    """

    def __init__(
        self,
        stuck_unique_threshold=2,
        stuck_min_window=6,
        replan_on_wall_hit=True,
        replan_on_stuck=True,
        replan_on_prediction_mismatch=True,
        prediction_error_threshold=4.0,
    ):
        self.stuck_unique_threshold = stuck_unique_threshold
        self.stuck_min_window = stuck_min_window
        self.replan_on_wall_hit = replan_on_wall_hit
        self.replan_on_stuck = replan_on_stuck
        self.replan_on_prediction_mismatch = replan_on_prediction_mismatch
        self.prediction_error_threshold = prediction_error_threshold

    def decide(
        self,
        z_t,
        memory,
        last_info,
        prediction_signal=None,
    ):
        """
        Decide whether to continue current behavior, replan, or stop.
        """
        agent_pos = z_t["agent_pos"]
        goal_visible = z_t.get("goal_visible", False)
        visible_goal_pos = z_t.get("visible_goal_pos", None)

        memory_summary = memory.get_summary()
        known_goal_pos = memory_summary.get("known_goal_pos", None)
        has_key = bool(memory_summary.get("has_key", False))
        known_door_pos = memory_summary.get("known_door_pos", None)

        loop_hints = memory.get_loop_hints()
        oscillation_pair = loop_hints.get("oscillation_pair")
        repeat_count_current_pos = loop_hints.get(
            "repeat_count_current_pos", 0)

        # --------------------------------------------------
        # 1) Goal reached
        # --------------------------------------------------
        if goal_visible and visible_goal_pos is not None and agent_pos == visible_goal_pos:
            return {"decision": "STOP", "reason": "goal_reached_visible"}

        if known_goal_pos is not None:
            known_goal_pos = tuple(known_goal_pos)
            if agent_pos == known_goal_pos:
                return {"decision": "STOP", "reason": "goal_reached_memory"}

        # --------------------------------------------------
        # 2) Immediate phase-switch replans
        # --------------------------------------------------
        if last_info is not None:
            if last_info.get("picked_key", False):
                return {"decision": "REPLAN", "reason": "picked_key_phase_change"}

            if last_info.get("opened_door", False):
                return {"decision": "REPLAN", "reason": "opened_door_phase_change"}

        # --------------------------------------------------
        # 3) Last action failed
        # --------------------------------------------------
        if last_info and self.replan_on_wall_hit:
            if (
                last_info.get("hit_wall")
                or last_info.get("out_of_bounds")
                or last_info.get("blocked_by_locked_door")
            ):
                return {"decision": "REPLAN", "reason": "movement_failed"}

         # --------------------------------------------------
        # 4) Early oscillation detection
        # --------------------------------------------------
        if oscillation_pair is not None:
            if not has_key:
                return {"decision": "REPLAN", "reason": "oscillation_before_key"}

            if has_key and known_goal_pos is not None:
                return {"decision": "REPLAN", "reason": "oscillation_after_goal_known"}

            if has_key and known_door_pos is not None:
                return {"decision": "REPLAN", "reason": "oscillation_after_key"}

            return {"decision": "REPLAN", "reason": "oscillation_detected"}

        # Current cell repeated too much very recently
        if repeat_count_current_pos >= 3:
            return {"decision": "REPLAN", "reason": "high_repeat_current_pos"}
        # --------------------------------------------------
        # 5) Repeated movement / oscillation (coarser stuck detector)
        # --------------------------------------------------
        if (
            self.replan_on_stuck
            and memory.is_stuck_by_repetition(
                threshold_unique=self.stuck_unique_threshold,
                min_window=self.stuck_min_window,
            )
        ):
            if not has_key and memory_summary.get("known_key_pos") is not None:
                return {"decision": "REPLAN", "reason": "stuck_before_key"}

            if has_key and memory_summary.get("known_goal_pos") is not None:
                return {"decision": "REPLAN", "reason": "stuck_after_goal_known"}

            if has_key and memory_summary.get("known_door_pos") is not None:
                return {"decision": "REPLAN", "reason": "stuck_after_key"}

            return {"decision": "REPLAN", "reason": "stuck_repetition"}
        # --------------------------------------------------
        # 6) Prediction mismatch
        # --------------------------------------------------
        if (
            self.replan_on_prediction_mismatch
            and prediction_signal is not None
            and prediction_signal.get("total_error", 0.0) >= self.prediction_error_threshold
        ):
            return {
                "decision": "REPLAN",
                "reason": "prediction_mismatch",
            }

        return {"decision": "CONTINUE", "reason": "normal"}

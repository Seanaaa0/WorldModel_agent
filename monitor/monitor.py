class Monitor:
    """
    PO-friendly runtime monitor.

    Responsibilities:
    - stop if goal is reached
    - trigger replanning on failed movement
    - detect oscillation / getting stuck
    - optionally trigger replanning on prediction mismatch
    - avoid relying on always-available goal_distance
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

        Inputs:
        - z_t: current latent state
        - memory: WorldMemory instance
        - last_info: last env step info
        - prediction_signal: optional dict like {"total_error": ...}
        """
        agent_pos = z_t["agent_pos"]
        goal_visible = z_t["goal_visible"]
        visible_goal_pos = z_t["goal_pos"]

        memory_summary = memory.get_summary()
        known_goal_pos = memory_summary["known_goal_pos"]

        # 1) Goal reached
        # Case A: goal currently visible and agent is on it
        if goal_visible and visible_goal_pos is not None and agent_pos == visible_goal_pos:
            return {"decision": "STOP", "reason": "goal_reached_visible"}

        # Case B: goal was seen before and memory knows where it is
        if known_goal_pos is not None and agent_pos == known_goal_pos:
            return {"decision": "STOP", "reason": "goal_reached_memory"}

        # 2) Last action failed: wall hit / boundary hit
        if (
            last_info
            and self.replan_on_wall_hit
            and (last_info.get("hit_wall") or last_info.get("out_of_bounds"))
        ):
            return {"decision": "REPLAN", "reason": "movement_failed"}

        # 3) Repeated movement / oscillation
        if (
            self.replan_on_stuck
            and memory.is_stuck_by_repetition(
                threshold_unique=self.stuck_unique_threshold,
                min_window=self.stuck_min_window,
            )
        ):
            # If goal has been seen before, say so in the reason.
            if known_goal_pos is not None:
                return {"decision": "REPLAN", "reason": "stuck_after_goal_known"}
            return {"decision": "REPLAN", "reason": "stuck_repetition"}

        # 4) Prediction mismatch
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

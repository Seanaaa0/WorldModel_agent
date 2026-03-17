from planner.planner_base import BasePlanner
from predictor.mlp_predictor import MLPPredictor


class PredictiveRulePlanner(BasePlanner):
    """
    Model-based Rule Planner (v2)

    Idea:
    - keep scan / escape_loop logic simple and rule-based
    - for primitive move candidates, use predictor for one-step rollout
    - score predicted next state using BOTH:
        (a) predicted geometry / position
        (b) predicted goal_distance when available
    """

    DIR_TO_DELTA = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
    }

    def __init__(
        self,
        predictor_checkpoint: str = "predictor/checkpoints/jepa_lite_mlp_po_v2.pt",
        use_predictor: bool = True,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        self.use_predictor = use_predictor

        self.predictor = None
        if use_predictor:
            try:
                self.predictor = MLPPredictor(
                    checkpoint_path=predictor_checkpoint
                )
                if self.verbose:
                    print("[PredictiveRulePlanner] Predictor loaded.")
            except Exception as e:
                self.predictor = None
                if self.verbose:
                    print(f"[PredictiveRulePlanner] Predictor disabled: {e}")

    def choose_skill(
        self,
        z_t: dict,
        memory_summary: dict | None = None,
        memory_patch: list | None = None,
        frontier_candidates: list | None = None,
        loop_hints: dict | None = None,
        planner_context: dict | None = None,
        replan: bool = False,
        last_info: dict | None = None,
    ) -> dict:
        memory_summary = memory_summary or {}
        frontier_candidates = frontier_candidates or []
        loop_hints = loop_hints or {}

        step_count = z_t["step_count"]
        just_scanned = last_info is not None and last_info.get("scan", False)

        agent_pos = z_t["agent_pos"]
        walls = z_t["local_walls"]

        goal_pos = z_t.get("goal_pos", None)
        if goal_pos is None:
            remembered_goal = memory_summary.get("known_goal_pos", None)
            if remembered_goal is not None:
                goal_pos = tuple(remembered_goal)

        # 1) Hard stuck handling
        if loop_hints.get("is_stuck", False):
            return {"skill": "escape_loop", "args": {}}

        # 2) Replan after actual failure can trigger one scan
        if replan and not just_scanned:
            if last_info is not None and (
                last_info.get("hit_wall", False) or last_info.get(
                    "out_of_bounds", False)
            ):
                return {"skill": "scan", "args": {}}

        # 3) Periodic scan only during exploration
        if (
            goal_pos is None
            and step_count > 0
            and step_count % 12 == 0
            and not just_scanned
        ):
            return {"skill": "scan", "args": {}}

        # 4) Primitive move candidates only
        candidates = self._get_open_candidates(walls)
        if not candidates:
            return {"skill": "scan", "args": {}}

        best_action = None
        best_score = float("-inf")

        for action in candidates:
            skill_spec = {
                "skill": "move",
                "args": {"direction": action},
            }

            z_hat = self._predict_next_state(z_t, skill_spec)
            score = self._score_predicted_move(
                z_t=z_t,
                z_hat=z_hat,
                action=action,
                goal_pos=goal_pos,
                memory_summary=memory_summary,
                frontier_candidates=frontier_candidates,
                loop_hints=loop_hints,
                last_info=last_info,
            )

            if self.verbose:
                print(
                    f"[PredictiveRulePlanner] action={action} "
                    f"pred_pos={z_hat.get('agent_pos')} "
                    f"pred_goal_visible={z_hat.get('goal_visible')} "
                    f"pred_goal_distance={z_hat.get('goal_distance')} "
                    f"score={score:.3f}"
                )

            if score > best_score:
                best_score = score
                best_action = action

        if best_action is None:
            return {"skill": "scan", "args": {}}

        return {
            "skill": "move",
            "args": {"direction": best_action},
        }

    # =========================================================
    # Prediction
    # =========================================================

    def _predict_next_state(self, z_t: dict, skill_spec: dict) -> dict:
        """
        Use learned predictor if available; otherwise fallback to symbolic one-step move.
        """
        if self.predictor is not None:
            try:
                return self.predictor.predict_next_state(z_t, skill_spec)
            except Exception as e:
                if self.verbose:
                    print(
                        f"[PredictiveRulePlanner] Predictor fallback due to: {e}")

        return self._symbolic_predict_next_state(z_t, skill_spec)

    def _symbolic_predict_next_state(self, z_t: dict, skill_spec: dict) -> dict:
        r, c = z_t["agent_pos"]
        direction = skill_spec["args"]["direction"].upper()
        walls = z_t["local_walls"]

        if walls.get(direction.lower(), False):
            nr, nc = r, c
        else:
            dr, dc = self.DIR_TO_DELTA[direction]
            nr, nc = r + dr, c + dc

        goal_pos = z_t.get("goal_pos", None)
        pred_goal_distance = None
        if goal_pos is not None:
            pred_goal_distance = self._manhattan((nr, nc), goal_pos)

        return {
            "agent_pos": (nr, nc),
            "goal_visible": z_t.get("goal_visible", False),
            "goal_pos": goal_pos,
            "dx": None,
            "dy": None,
            "goal_distance": pred_goal_distance,
            "local_walls": {
                "up": None,
                "down": None,
                "left": None,
                "right": None,
            },
            "step_count": z_t["step_count"] + 1,
        }

    # =========================================================
    # Scoring
    # =========================================================

    def _score_predicted_move(
        self,
        z_t: dict,
        z_hat: dict,
        action: str,
        goal_pos: tuple[int, int] | None,
        memory_summary: dict,
        frontier_candidates: list,
        loop_hints: dict,
        last_info: dict | None,
    ) -> float:
        current_pos = z_t["agent_pos"]
        pred_pos = tuple(z_hat["agent_pos"])

        score = 0.0

        # A) Goal-directed value
        if goal_pos is not None:
            current_dist = self._manhattan(current_pos, goal_pos)

            pred_goal_distance = z_hat.get("goal_distance", None)
            if pred_goal_distance is None:
                pred_dist = self._manhattan(pred_pos, goal_pos)
            else:
                pred_dist = float(pred_goal_distance)

            # Main value: predicted reduction in goal distance
            score += 16.0 * (current_dist - pred_dist)

            # Still keep directional prior
            score += self._goal_axis_bias(current_pos, goal_pos, action)

        # B) Exploration value
        else:
            score += self._frontier_progress_bonus(
                current_pos=current_pos,
                pred_pos=pred_pos,
                frontier_candidates=frontier_candidates,
            )

            # If model predicts goal becomes visible, that's strong value.
            if not z_t.get("goal_visible", False) and z_hat.get("goal_visible", False):
                score += 8.0

        # C) Visit / novelty / anti-repeat
        score += self._novelty_bonus(pred_pos, memory_summary)
        score += self._visit_count_bonus(pred_pos, memory_summary)
        score += self._repeat_penalty(pred_pos, memory_summary)
        score += self._oscillation_penalty(pred_pos, loop_hints)
        score += self._failed_direction_penalty(action, last_info)

        return score

    def _frontier_progress_bonus(
        self,
        current_pos: tuple[int, int],
        pred_pos: tuple[int, int],
        frontier_candidates: list,
    ) -> float:
        if not frontier_candidates:
            return 0.0

        parsed = []
        for item in frontier_candidates:
            if "pos" not in item:
                continue
            pos = tuple(item["pos"])
            frontier_score = float(item.get("frontier_score", 0))
            parsed.append((pos, frontier_score))

        if not parsed:
            return 0.0

        best_target, best_score = max(parsed, key=lambda x: x[1])

        current_dist = self._manhattan(current_pos, best_target)
        pred_dist = self._manhattan(pred_pos, best_target)

        bonus = 0.0

        if pred_pos == best_target:
            bonus += 5.0

        if pred_dist < current_dist:
            bonus += 4.0
        elif pred_dist > current_dist:
            bonus -= 2.0

        bonus += min(best_score, 4.0) * 0.5
        return bonus

    def _goal_axis_bias(
        self,
        current_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        action: str,
    ) -> float:
        ar, ac = current_pos
        gr, gc = goal_pos

        dy = gr - ar
        dx = gc - ac

        score = 0.0

        if abs(dx) >= abs(dy):
            if dx > 0 and action == "RIGHT":
                score += 4.0
            elif dx < 0 and action == "LEFT":
                score += 4.0

            if dy > 0 and action == "DOWN":
                score += 2.0
            elif dy < 0 and action == "UP":
                score += 2.0
        else:
            if dy > 0 and action == "DOWN":
                score += 4.0
            elif dy < 0 and action == "UP":
                score += 4.0

            if dx > 0 and action == "RIGHT":
                score += 2.0
            elif dx < 0 and action == "LEFT":
                score += 2.0

        return score

    def _novelty_bonus(
        self,
        pred_pos: tuple[int, int],
        memory_summary: dict,
    ) -> float:
        visit_counts = memory_summary.get("visit_counts", {})
        if pred_pos not in visit_counts:
            return 6.0
        return 0.0

    def _visit_count_bonus(
        self,
        pred_pos: tuple[int, int],
        memory_summary: dict,
    ) -> float:
        visit_counts = memory_summary.get("visit_counts", {})
        count = visit_counts.get(pred_pos, 0)

        if count == 0:
            return 5.0
        if count == 1:
            return 2.0
        if count == 2:
            return -1.0
        return -2.5 * count

    def _repeat_penalty(
        self,
        pred_pos: tuple[int, int],
        memory_summary: dict,
    ) -> float:
        recent_positions = memory_summary.get("recent_positions", [])
        recent_tail = [tuple(p) for p in recent_positions[-8:]]

        if pred_pos in set(recent_tail):
            return -12.0
        return 0.0

    def _oscillation_penalty(
        self,
        pred_pos: tuple[int, int],
        loop_hints: dict,
    ) -> float:
        oscillation_pair = loop_hints.get("oscillation_pair")
        if not oscillation_pair:
            return 0.0

        pair_set = {tuple(p) for p in oscillation_pair}
        if pred_pos in pair_set:
            return -14.0

        return 0.0

    def _failed_direction_penalty(
        self,
        action: str,
        last_info: dict | None,
    ) -> float:
        if last_info is None:
            return 0.0

        if last_info.get("hit_wall") or last_info.get("out_of_bounds"):
            failed = str(last_info.get("action", "")).strip().upper()
            if action == failed:
                return -20.0

        return 0.0

    # =========================================================
    # Helpers
    # =========================================================

    def _get_open_candidates(self, walls: dict) -> list[str]:
        actions = ["UP", "RIGHT", "DOWN", "LEFT"]
        return [a for a in actions if not walls.get(a.lower(), False)]

    def _manhattan(
        self,
        a: tuple[int, int],
        b: tuple[int, int],
    ) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

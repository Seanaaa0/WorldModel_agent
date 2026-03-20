from planner.planner_base import BasePlanner

try:
    from predictor.mlp_predictor import MLPPredictor
except Exception:
    MLPPredictor = None


class PredictivePlannerV8(BasePlanner):
    """
    V8 Predictive Planner

    Design:
    - keep robust phase / mode resolution
    - keep scan / stuck / recover handling
    - keep BFS-to-known-target when memory is reliable
    - replace local action selection with predictor-driven one-step rollout

    Predictor role:
    - predictor is no longer just a bonus term
    - predictor directly evaluates candidate next states
    """

    DIR_TO_DELTA = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
    }

    ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]

    def __init__(
        self,
        predictor_checkpoint: str,
        verbose: bool = False,
    ):
        self.verbose = bool(verbose)
        self.predictor_checkpoint = predictor_checkpoint
        self.predictor = None

        if MLPPredictor is None:
            raise RuntimeError("MLPPredictor import failed.")

        self.predictor = MLPPredictor(checkpoint_path=predictor_checkpoint)

        if self.verbose:
            print(
                f"[PredictivePlannerV8] predictor loaded: {predictor_checkpoint}")

    # =========================================================
    # Main entry
    # =========================================================

    def choose_skill(
        self,
        z_t: dict,
        memory_summary: dict | None = None,
        memory_patch: list | None = None,
        frontier_candidates: list | None = None,
        loop_hints: dict | None = None,
        replan: bool = False,
        last_info: dict | None = None,
        planner_context: dict | None = None,
    ) -> dict:
        memory_summary = memory_summary or {}
        frontier_candidates = frontier_candidates or []
        loop_hints = loop_hints or {}
        planner_context = planner_context or {}

        step_count = z_t["step_count"]
        just_scanned = last_info is not None and last_info.get("scan", False)

        agent_pos = z_t["agent_pos"]
        walls = z_t["local_walls"]
        has_key = bool(z_t.get("has_key", False))

        visible_key_pos = z_t.get("visible_key_pos", None)
        visible_door_pos = z_t.get("visible_door_pos", None)
        visible_goal_pos = z_t.get("visible_goal_pos", None)
        visible_door_open = z_t.get("visible_door_open", None)

        known_key_pos = memory_summary.get("known_key_pos", None)
        known_door_pos = memory_summary.get("known_door_pos", None)
        known_goal_pos = memory_summary.get("known_goal_pos", None)
        known_door_open = memory_summary.get("known_door_open", None)

        if known_key_pos is not None:
            known_key_pos = tuple(known_key_pos)
        if known_door_pos is not None:
            known_door_pos = tuple(known_door_pos)
        if known_goal_pos is not None:
            known_goal_pos = tuple(known_goal_pos)

        # hard stuck handling
        if loop_hints.get("is_stuck", False):
            return {"skill": "escape_loop", "args": {}}

        forced_phase = planner_context.get("forced_phase", None)

        # replan after actual failure => allow one scan
        if replan and not just_scanned:
            if last_info is not None and (
                last_info.get("hit_wall", False)
                or last_info.get("out_of_bounds", False)
                or last_info.get("blocked_by_locked_door", False)
            ):
                return {"skill": "scan", "args": {}}

        # periodic scan under uncertainty
        if step_count > 0 and not just_scanned:
            if not has_key:
                if visible_key_pos is None and known_key_pos is None and step_count % 10 == 0:
                    return {"skill": "scan", "args": {}}
            else:
                if visible_goal_pos is None and known_goal_pos is None and step_count % 8 == 0:
                    return {"skill": "scan", "args": {}}

        target_pos, mode = self._resolve_target_and_mode(
            forced_phase=forced_phase,
            has_key=has_key,
            visible_key_pos=visible_key_pos,
            known_key_pos=known_key_pos,
            visible_door_pos=visible_door_pos,
            known_door_pos=known_door_pos,
            visible_door_open=visible_door_open,
            known_door_open=known_door_open,
            visible_goal_pos=visible_goal_pos,
            known_goal_pos=known_goal_pos,
        )

        if self.verbose:
            print(
                "[PredictivePlannerV8 DEBUG]",
                {
                    "forced_phase": forced_phase,
                    "mode": mode,
                    "target_pos": target_pos,
                    "has_key": has_key,
                    "known_key_pos": known_key_pos,
                    "known_door_pos": known_door_pos,
                    "known_goal_pos": known_goal_pos,
                },
            )

        # Keep BFS to known target: reliable shortcut if memory is already good
        memory_obj = planner_context.get("memory_obj", None)
        if memory_obj is not None and target_pos is not None:
            path = memory_obj.get_path_to_known_target(agent_pos, target_pos)
            action = memory_obj.first_action_from_path(path)

            if path is not None:
                path_len = len(path)
            else:
                path_len = None

            # 1) Normal BFS shortcut
            if action is not None and not walls.get(action.lower(), False):
                return {
                    "skill": "move",
                    "args": {"direction": action},
                }

            # 2) Small extra guard:
            #    if target is known and already quite close, do not immediately
            #    fall back to free-form predictive wandering.
            #    Instead, force one scan to refresh local geometry / visibility.
            if path_len is not None and path_len <= 5:
                return {"skill": "scan", "args": {}}

        # Otherwise: predictor-driven local selection
        chosen_direction = self._choose_predictive_direction(
            z_t=z_t,
            agent_pos=agent_pos,
            walls=walls,
            mode=mode,
            target_pos=target_pos,
            memory_summary=memory_summary,
            frontier_candidates=frontier_candidates,
            loop_hints=loop_hints,
            last_info=last_info,
        )

        if chosen_direction is None:
            return {"skill": "scan", "args": {}}

        return {
            "skill": "move",
            "args": {"direction": chosen_direction},
        }

    # =========================================================
    # Phase resolution
    # =========================================================

    def _resolve_target_and_mode(
        self,
        forced_phase,
        has_key: bool,
        visible_key_pos,
        known_key_pos,
        visible_door_pos,
        known_door_pos,
        visible_door_open,
        known_door_open,
        visible_goal_pos,
        known_goal_pos,
    ):
        if forced_phase is not None:
            phase = self._normalize_phase(forced_phase)

            if phase == "recover":
                return None, "recover"

            if phase == "find_key":
                target_pos = visible_key_pos or known_key_pos
                if target_pos is not None:
                    return target_pos, "to_key"
                return None, "pre_key_explore"

            if phase == "go_to_door":
                target_pos = visible_door_pos or known_door_pos
                if target_pos is not None:
                    return target_pos, "to_door"
                return None, "post_key_explore"

            if phase == "search_goal":
                return None, "post_door_explore"

            if phase == "go_to_goal":
                target_pos = visible_goal_pos or known_goal_pos
                if target_pos is not None:
                    return target_pos, "to_goal"
                return None, "post_door_explore"

        target_pos = None
        mode = "explore"

        if not has_key:
            target_pos = visible_key_pos or known_key_pos
            if target_pos is not None:
                mode = "to_key"
            else:
                mode = "pre_key_explore"
        else:
            goal_candidate = visible_goal_pos or known_goal_pos
            door_candidate = visible_door_pos or known_door_pos

            door_is_open = (
                (visible_door_open is True)
                or (known_door_open is True)
            )

            if (not door_is_open) and (door_candidate is not None):
                target_pos = door_candidate
                mode = "to_door"
            elif goal_candidate is not None:
                target_pos = goal_candidate
                mode = "to_goal"
            elif door_is_open:
                mode = "post_door_explore"
            else:
                mode = "post_key_explore"

        return target_pos, mode

    def _normalize_phase(self, phase: str | None) -> str | None:
        if phase is None:
            return None

        phase = str(phase).strip().lower()
        aliases = {
            "to_key": "find_key",
            "pre_key_explore": "find_key",
            "to_door": "go_to_door",
            "post_key_explore": "go_to_door",
            "post_door_explore": "search_goal",
            "to_goal": "go_to_goal",
        }
        return aliases.get(phase, phase)

    # =========================================================
    # Predictive local selection
    # =========================================================

    def _choose_predictive_direction(
        self,
        z_t: dict,
        agent_pos: tuple[int, int],
        walls: dict,
        mode: str,
        target_pos,
        memory_summary: dict,
        frontier_candidates: list,
        loop_hints: dict,
        last_info: dict | None,
    ) -> str | None:
        candidates = self._get_open_candidates(walls)
        if not candidates:
            return None

        scored = []
        for action in candidates:
            pred = self._predict_next_state_safe(z_t, action)
            if pred is None:
                continue

            score = self._evaluate_predicted_state(
                z_t=z_t,
                pred_state=pred,
                action=action,
                mode=mode,
                target_pos=target_pos,
                memory_summary=memory_summary,
                frontier_candidates=frontier_candidates,
                loop_hints=loop_hints,
                last_info=last_info,
            )
            scored.append((score, action))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)

        if self.verbose:
            print("[PredictiveDir Scores]", scored)

        return scored[0][1]

    def _evaluate_predicted_state(
        self,
        z_t: dict,
        pred_state: dict,
        action: str,
        mode: str,
        target_pos,
        memory_summary: dict,
        frontier_candidates: list,
        loop_hints: dict,
        last_info: dict | None,
    ) -> float:
        pred_pos = self._as_tuple(pred_state.get("agent_pos"))
        current_pos = self._as_tuple(z_t.get("agent_pos"))

        if pred_pos is None or current_pos is None:
            return -999.0

        score = 0.0

        # -----------------------------------------------------
        # hard penalties
        # -----------------------------------------------------
        if pred_pos == current_pos:
            score -= 12.0

        score += self._failed_direction_penalty(action, last_info)
        score += self._repeat_penalty(pred_pos, memory_summary)
        score += self._oscillation_penalty(pred_pos, loop_hints)
        score += self._visit_count_penalty(pred_pos, memory_summary)

        # -----------------------------------------------------
        # V7 information gain
        # -----------------------------------------------------
        view_gain_score = float(pred_state.get("view_gain_score", 0.0))
        score += 1.5 * view_gain_score

        if bool(pred_state.get("new_key_seen", False)):
            score += 6.0

        if bool(pred_state.get("new_door_seen", False)):
            score += 5.0

        if bool(pred_state.get("new_goal_seen", False)):
            score += 7.0

        # -----------------------------------------------------
        # local wall openness
        # -----------------------------------------------------
        pred_walls = pred_state.get("local_walls", {})
        if isinstance(pred_walls, dict):
            open_count = 0
            for k in ("up", "down", "left", "right"):
                if pred_walls.get(k) is False:
                    open_count += 1

            if open_count <= 1:
                score -= 3.0
            elif open_count == 2:
                score += 0.5
            elif open_count >= 3:
                score += 2.0

        # -----------------------------------------------------
        # target-mode evaluation
        # -----------------------------------------------------
        if target_pos is not None:
            current_dist = self._manhattan(current_pos, target_pos)
            pred_dist = self._manhattan(pred_pos, target_pos)

            score += 12.0 * (current_dist - pred_dist)
            score += self._target_axis_bias(current_pos, target_pos, action)

            if pred_pos == target_pos:
                if mode == "to_key":
                    score += 14.0
                elif mode == "to_door":
                    score += 8.0
                elif mode == "to_goal":
                    score += 12.0

        # -----------------------------------------------------
        # mode-specific signals
        # -----------------------------------------------------
        if mode == "to_key":
            score += self._object_seek_bonus(
                pred_pos,
                pred_state,
                object_pos_key="visible_key_pos",
                object_visible_key="key_visible",
                seen_bonus=6.0,
                near_bonus=5.0,
            )
            if bool(pred_state.get("has_key", False)):
                score += 10.0

        elif mode == "to_door":
            score += self._object_seek_bonus(
                pred_pos,
                pred_state,
                object_pos_key="visible_door_pos",
                object_visible_key="door_visible",
                seen_bonus=5.0,
                near_bonus=4.0,
            )
            if pred_state.get("visible_door_open", None) is True:
                score += 3.0

        elif mode == "to_goal":
            score += self._object_seek_bonus(
                pred_pos,
                pred_state,
                object_pos_key="visible_goal_pos",
                object_visible_key="goal_visible",
                seen_bonus=7.0,
                near_bonus=6.0,
            )

        elif mode == "pre_key_explore":
            score += self._predicted_novelty_bonus(pred_pos, memory_summary)
            if bool(pred_state.get("key_visible", False)):
                score += 8.0

        elif mode == "post_key_explore":
            score += self._predicted_novelty_bonus(pred_pos, memory_summary)
            if bool(pred_state.get("door_visible", False)):
                score += 7.0

        elif mode == "post_door_explore":
            score += self._predicted_novelty_bonus(pred_pos, memory_summary)
            if bool(pred_state.get("goal_visible", False)):
                score += 9.0

        else:
            score += self._predicted_novelty_bonus(pred_pos, memory_summary)

        # -----------------------------------------------------
        # mild frontier prior (soft only)
        # -----------------------------------------------------
        score += 0.3 * self._frontier_bonus(
            current_pos,
            pred_pos,
            memory_summary,
            frontier_candidates,
        )

        return score

    # =========================================================
    # Predictor helpers
    # =========================================================

    def _predict_next_state_safe(self, z_t: dict, action: str):
        if self.predictor is None:
            return None
        skill = {"skill": "move", "args": {"direction": action}}
        try:
            return self.predictor.predict_next_state(z_t, skill)
        except Exception as e:
            if self.verbose:
                print(f"[PredictivePlannerV8] predictor inference failed: {e}")
            return None

    def _object_seek_bonus(
        self,
        pred_agent_pos,
        pred: dict,
        object_pos_key: str,
        object_visible_key: str,
        seen_bonus: float,
        near_bonus: float,
    ) -> float:
        score = 0.0
        object_pos = self._as_tuple(pred.get(object_pos_key))
        object_visible = bool(pred.get(object_visible_key, False))

        if object_visible:
            score += seen_bonus

        if object_pos is not None and pred_agent_pos is not None:
            dist = self._manhattan(pred_agent_pos, object_pos)
            score += max(0.0, near_bonus - 0.8 * dist)
            if dist == 0:
                score += 4.0
            elif dist == 1:
                score += 2.0

        return score

    def _predicted_novelty_bonus(self, pred_agent_pos, memory_summary: dict) -> float:
        if pred_agent_pos is None:
            return 0.0
        visit_counts = memory_summary.get("visit_counts", {})
        count = visit_counts.get(pred_agent_pos, 0)
        if count == 0:
            return 3.0
        if count == 1:
            return 1.0
        return -1.0 * min(count, 3)

    # =========================================================
    # Shared scoring helpers
    # =========================================================

    def _frontier_bonus(
        self,
        agent_pos: tuple[int, int],
        next_pos: tuple[int, int],
        memory_summary: dict,
        frontier_candidates: list,
    ) -> float:
        bonus = 0.0

        if not frontier_candidates:
            return bonus

        parsed = []
        for item in frontier_candidates:
            if "pos" not in item:
                continue
            pos = tuple(item["pos"])
            frontier_score = float(item.get("frontier_score", 0))
            visit_count = float(item.get("visit_count", 0))
            parsed.append((pos, frontier_score, visit_count))

        if not parsed:
            return bonus

        frontier_positions = {pos for pos, _, _ in parsed}
        if next_pos in frontier_positions:
            bonus += 4.0

        best_target, best_frontier_score, _ = max(parsed, key=lambda x: x[1])

        current_dist = self._manhattan(agent_pos, best_target)
        next_dist = self._manhattan(next_pos, best_target)

        if next_dist < current_dist:
            bonus += 3.5
        elif next_dist > current_dist:
            bonus -= 1.5

        bonus += min(best_frontier_score, 4.0) * 0.5

        recent_positions = memory_summary.get("recent_positions", [])
        recent_set = set(tuple(p) for p in recent_positions[-8:])
        if next_pos in recent_set:
            bonus -= 2.0

        return bonus

    def _repeat_penalty(
        self,
        next_pos: tuple[int, int],
        memory_summary: dict,
    ) -> float:
        recent_positions = memory_summary.get("recent_positions", [])
        recent_tail = [tuple(p) for p in recent_positions[-8:]]
        if next_pos in set(recent_tail):
            return -12.0
        return 0.0

    def _oscillation_penalty(
        self,
        next_pos: tuple[int, int],
        loop_hints: dict,
    ) -> float:
        oscillation_pair = loop_hints.get("oscillation_pair")
        if not oscillation_pair:
            return 0.0

        pair_set = {tuple(p) for p in oscillation_pair}
        if next_pos in pair_set:
            return -20.0
        return 0.0

    def _failed_direction_penalty(
        self,
        action: str,
        last_info: dict | None,
    ) -> float:
        if last_info is None:
            return 0.0

        if (
            last_info.get("hit_wall")
            or last_info.get("out_of_bounds")
            or last_info.get("blocked_by_locked_door")
        ):
            failed = str(last_info.get("action", "")).strip().upper()
            if action == failed:
                return -20.0
        return 0.0

    def _visit_count_penalty(
        self,
        next_pos: tuple[int, int],
        memory_summary: dict,
    ) -> float:
        visit_counts = memory_summary.get("visit_counts", {})
        count = visit_counts.get(next_pos, 0)

        if count == 0:
            return 0.0
        if count == 1:
            return -1.0
        if count == 2:
            return -3.0
        return -5.0 - 1.5 * count

    def _target_axis_bias(
        self,
        agent_pos: tuple[int, int],
        target_pos: tuple[int, int],
        action: str,
    ) -> float:
        ar, ac = agent_pos
        tr, tc = target_pos

        dy = tr - ar
        dx = tc - ac

        score = 0.0

        if abs(dx) >= abs(dy):
            if dx > 0 and action == "RIGHT":
                score += 5.0
            elif dx < 0 and action == "LEFT":
                score += 5.0

            if dy > 0 and action == "DOWN":
                score += 2.5
            elif dy < 0 and action == "UP":
                score += 2.5
        else:
            if dy > 0 and action == "DOWN":
                score += 5.0
            elif dy < 0 and action == "UP":
                score += 5.0

            if dx > 0 and action == "RIGHT":
                score += 2.5
            elif dx < 0 and action == "LEFT":
                score += 2.5

        return score

    # =========================================================
    # Geometry helpers
    # =========================================================

    def _get_open_candidates(self, walls: dict) -> list[str]:
        return [a for a in self.ACTIONS if not walls.get(a.lower(), False)]

    def _manhattan(
        self,
        a: tuple[int, int],
        b: tuple[int, int],
    ) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _as_tuple(self, x):
        if x is None:
            return None
        try:
            return tuple(x)
        except Exception:
            return None

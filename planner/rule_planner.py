from planner.planner_base import BasePlanner

try:
    from predictor.mlp_predictor import MLPPredictor
except Exception:
    MLPPredictor = None


class RulePlanner(BasePlanner):
    """
    V5-b Rule Planner / Execution Engine

    Core role:
    - robust executor for phase-guided behavior
    - can still self-infer phase if forced_phase is not provided
    - supports BFS over known memory map + frontier exploration + local fallback
    - optional one-step predictor bonus for local move selection

    Supported forced phases:
    - find_key
    - go_to_door
    - search_goal
    - go_to_goal
    - recover
    """

    DIR_TO_DELTA = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
    }

    ROLLOUT_ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]

    def __init__(
        self,
        use_predictor: bool = False,
        predictor_checkpoint: str | None = None,
        predictor_weight: float = 1.0,
        verbose: bool = False,
    ):
        self.use_predictor = bool(use_predictor)
        self.predictor_checkpoint = predictor_checkpoint
        self.predictor_weight = float(predictor_weight)
        self.verbose = bool(verbose)

        self.predictor = None
        if self.use_predictor and predictor_checkpoint and MLPPredictor is not None:
            try:
                self.predictor = MLPPredictor(
                    checkpoint_path=predictor_checkpoint)
                if self.verbose:
                    print(
                        f"[RulePlanner] Predictor loaded: {predictor_checkpoint}")
            except Exception as e:
                self.predictor = None
                if self.verbose:
                    print(f"[RulePlanner] Predictor load failed: {e}")

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

        if loop_hints.get("is_stuck", False):
            return {"skill": "escape_loop", "args": {}}

        forced_phase = planner_context.get("forced_phase", None)

        if replan and not just_scanned:
            if last_info is not None and (
                last_info.get("hit_wall", False)
                or last_info.get("out_of_bounds", False)
                or last_info.get("blocked_by_locked_door", False)
            ):
                return {"skill": "scan", "args": {}}

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
                "[RulePlanner DEBUG]",
                {
                    "forced_phase": forced_phase,
                    "has_key": has_key,
                    "visible_key_pos": visible_key_pos,
                    "known_key_pos": known_key_pos,
                    "visible_door_pos": visible_door_pos,
                    "known_door_pos": known_door_pos,
                    "visible_door_open": visible_door_open,
                    "known_door_open": known_door_open,
                    "visible_goal_pos": visible_goal_pos,
                    "known_goal_pos": known_goal_pos,
                    "mode": mode,
                    "target_pos": target_pos,
                }
            )

        memory_obj = planner_context.get("memory_obj", None)
        frontier_action = None

        if memory_obj is not None:
            if target_pos is not None:
                path = memory_obj.get_path_to_known_target(
                    agent_pos, target_pos)
                action = memory_obj.first_action_from_path(path)
                if action is not None and not walls.get(action.lower(), False):
                    return {
                        "skill": "move",
                        "args": {"direction": action},
                    }

            path = memory_obj.get_path_to_best_frontier(
                agent_pos=agent_pos,
                top_k=10,
                mode=mode,
            )
            action = memory_obj.first_action_from_path(path)

            if action is not None and not walls.get(action.lower(), False):
                frontier_action = action
        if target_pos is not None:
            chosen_direction = self._choose_target_directed_direction(
                z_t=z_t,
                agent_pos=agent_pos,
                target_pos=target_pos,
                walls=walls,
                memory_summary=memory_summary,
                frontier_candidates=frontier_candidates,
                loop_hints=loop_hints,
                last_info=last_info,
                mode=mode,
                frontier_action=frontier_action,
            )
        else:
            chosen_direction = self._choose_exploration_direction(
                z_t=z_t,
                agent_pos=agent_pos,
                walls=walls,
                memory_summary=memory_summary,
                frontier_candidates=frontier_candidates,
                loop_hints=loop_hints,
                last_info=last_info,
                mode=mode,
                frontier_action=frontier_action,
            )

        if chosen_direction is None:
            return {"skill": "scan", "args": {}}

        return {
            "skill": "move",
            "args": {"direction": chosen_direction},
        }

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

            door_is_open = ((visible_door_open is True)
                            or (known_door_open is True))

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

    def _choose_target_directed_direction(
        self,
        z_t: dict,
        agent_pos: tuple[int, int],
        target_pos: tuple[int, int],
        walls: dict,
        memory_summary: dict,
        frontier_candidates: list,
        loop_hints: dict,
        last_info: dict | None,
        mode: str,
        frontier_action: str | None = None,
    ) -> str | None:
        candidates = self._get_open_candidates(walls)
        if not candidates:
            return None

        scored = []
        for action in candidates:
            score = self._score_target_directed_move(
                z_t=z_t,
                agent_pos=agent_pos,
                target_pos=target_pos,
                action=action,
                memory_summary=memory_summary,
                frontier_candidates=frontier_candidates,
                loop_hints=loop_hints,
                last_info=last_info,
                mode=mode,
            )

            # frontier path 不直接決定，只當 bonus
            if frontier_action is not None and action == frontier_action:
                score += 2.5

            scored.append((score, action))

        scored.sort(key=lambda x: x[0], reverse=True)

        if self.verbose:
            print("[TargetDir Scores]", scored)

        return scored[0][1]

    def _choose_exploration_direction(
        self,
        z_t: dict,
        agent_pos: tuple[int, int],
        walls: dict,
        memory_summary: dict,
        frontier_candidates: list,
        loop_hints: dict,
        last_info: dict | None,
        mode: str = "explore",
        frontier_action: str | None = None,
    ) -> str | None:
        if mode == "recover":
            return None

        candidates = self._get_open_candidates(walls)
        if not candidates:
            return None

        scored = []
        for action in candidates:
            score = self._score_exploration_move(
                z_t=z_t,
                agent_pos=agent_pos,
                action=action,
                memory_summary=memory_summary,
                frontier_candidates=frontier_candidates,
                loop_hints=loop_hints,
                last_info=last_info,
                mode=mode,
            )

            if frontier_action is not None and action == frontier_action:
                score += 2.5

            scored.append((score, action))

        scored.sort(key=lambda x: x[0], reverse=True)

        if self.verbose:
            print("[ExploreDir Scores]", scored)

        return scored[0][1]

    def _get_open_candidates(self, walls: dict) -> list[str]:
        actions = ["UP", "RIGHT", "DOWN", "LEFT"]
        return [a for a in actions if not walls.get(a.lower(), False)]

    def _score_target_directed_move(
        self,
        z_t: dict,
        agent_pos: tuple[int, int],
        target_pos: tuple[int, int],
        action: str,
        memory_summary: dict,
        frontier_candidates: list,
        loop_hints: dict,
        last_info: dict | None,
        mode: str,
    ) -> float:
        next_pos = self._simulate_move(agent_pos, action)

        current_dist = self._manhattan(agent_pos, target_pos)
        next_dist = self._manhattan(next_pos, target_pos)

        # =========================
        # Positive guidance terms
        # =========================
        distance_score = 15.0 * (current_dist - next_dist)
        axis_score = self._target_axis_bias(agent_pos, target_pos, action)
        novelty_score = 0.8 * self._novelty_bonus(next_pos, memory_summary)
        frontier_score = 0.3 * self._frontier_bonus(
            agent_pos, next_pos, memory_summary, frontier_candidates
        )

        arrival_score = 0.0
        if next_pos == target_pos:
            if mode == "to_key":
                arrival_score += 14.0
            elif mode == "to_door":
                arrival_score += 8.0
            elif mode == "to_goal":
                arrival_score += 12.0

        mode_bias_score = 0.0
        if mode == "to_key":
            mode_bias_score += 2.0 * (current_dist - next_dist)

        predictor_score = self._predictive_bonus(
            z_t=z_t,
            action=action,
            mode=mode,
            target_pos=target_pos,
            memory_summary=memory_summary,
            heuristic_next_pos=next_pos,
        )

        # =========================
        # Negative penalty terms
        # =========================
        visit_penalty = self._visit_count_penalty(next_pos, memory_summary)
        repeat_penalty = self._repeat_penalty(next_pos, memory_summary)
        failed_penalty = self._failed_direction_penalty(action, last_info)
        oscillation_penalty = self._oscillation_penalty(next_pos, loop_hints)

        # =========================
        # Final score
        # =========================
        score = (
            distance_score
            + axis_score
            + novelty_score
            + frontier_score
            + arrival_score
            + mode_bias_score
            + predictor_score
            + visit_penalty
            + repeat_penalty
            + failed_penalty
            + oscillation_penalty
        )

        if self.verbose:
            print(
                "[TargetScoreBreakdown]",
                {
                    "action": action,
                    "mode": mode,
                    "next_pos": next_pos,
                    "distance_score": round(distance_score, 3),
                    "axis_score": round(axis_score, 3),
                    "novelty_score": round(novelty_score, 3),
                    "frontier_score": round(frontier_score, 3),
                    "arrival_score": round(arrival_score, 3),
                    "mode_bias_score": round(mode_bias_score, 3),
                    "predictor_score": round(predictor_score, 3),
                    "visit_penalty": round(visit_penalty, 3),
                    "repeat_penalty": round(repeat_penalty, 3),
                    "failed_penalty": round(failed_penalty, 3),
                    "oscillation_penalty": round(oscillation_penalty, 3),
                    "total": round(score, 3),
                },
            )

        return score

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

    def _score_exploration_move(
        self,
        z_t: dict,
        agent_pos: tuple[int, int],
        action: str,
        memory_summary: dict,
        frontier_candidates: list,
        loop_hints: dict,
        last_info: dict | None,
        mode: str = "explore",
    ) -> float:
        next_pos = self._simulate_move(agent_pos, action)

        # =========================
        # Positive guidance terms
        # =========================
        novelty_score = self._novelty_bonus(next_pos, memory_summary)
        visit_bonus_score = self._visit_count_bonus(next_pos, memory_summary)
        frontier_score = self._frontier_bonus(
            agent_pos, next_pos, memory_summary, frontier_candidates
        )

        mode_bias_score = 0.0
        if mode == "pre_key_explore":
            mode_bias_score += self._pre_key_door_penalty(
                agent_pos=agent_pos,
                next_pos=next_pos,
                memory_summary=memory_summary,
            )

        if mode == "post_door_explore":
            mode_bias_score += self._post_door_bonus(
                agent_pos=agent_pos,
                next_pos=next_pos,
                memory_summary=memory_summary,
            )

        if mode == "post_key_explore":
            mode_bias_score += self._search_door_bias(
                agent_pos=agent_pos,
                next_pos=next_pos,
                memory_summary=memory_summary,
            )

        predictor_score = self._predictive_bonus(
            z_t=z_t,
            action=action,
            mode=mode,
            target_pos=None,
            memory_summary=memory_summary,
            heuristic_next_pos=next_pos,
        )

        # =========================
        # Negative penalty terms
        # =========================
        repeat_penalty = self._repeat_penalty(next_pos, memory_summary)
        failed_penalty = self._failed_direction_penalty(action, last_info)
        oscillation_penalty = self._oscillation_penalty(next_pos, loop_hints)

        # =========================
        # Final score
        # =========================
        score = (
            novelty_score
            + visit_bonus_score
            + frontier_score
            + mode_bias_score
            + predictor_score
            + repeat_penalty
            + failed_penalty
            + oscillation_penalty
        )

        if self.verbose:
            print(
                "[ExploreScoreBreakdown]",
                {
                    "action": action,
                    "mode": mode,
                    "next_pos": next_pos,
                    "novelty_score": round(novelty_score, 3),
                    "visit_bonus_score": round(visit_bonus_score, 3),
                    "frontier_score": round(frontier_score, 3),
                    "mode_bias_score": round(mode_bias_score, 3),
                    "predictor_score": round(predictor_score, 3),
                    "repeat_penalty": round(repeat_penalty, 3),
                    "failed_penalty": round(failed_penalty, 3),
                    "oscillation_penalty": round(oscillation_penalty, 3),
                    "total": round(score, 3),
                },
            )

        return score

    def _predictive_bonus(
        self,
        z_t: dict,
        action: str,
        mode: str,
        target_pos,
        memory_summary: dict,
        heuristic_next_pos: tuple[int, int],
    ) -> float:
        if self.predictor is None or not self.use_predictor:
            return 0.0

        pred1 = self._predict_next_state_safe(z_t, action)
        if pred1 is None:
            return 0.0

        pred1_pos = self._as_tuple(pred1.get("agent_pos"))
        if pred1_pos is None:
            return 0.0

        # 第一步 rollout 分數
        first_step_score = self._score_predicted_state(
            base_state=z_t,
            pred_state=pred1,
            action=action,
            mode=mode,
            target_pos=target_pos,
            memory_summary=memory_summary,
            heuristic_next_pos=heuristic_next_pos,
        )

        # 第二步 rollout：從 pred1 再試 4 個動作，取最佳值
        second_step_score = self._best_second_step_value(
            pred_state=pred1,
            mode=mode,
            target_pos=target_pos,
            memory_summary=memory_summary,
        )

        total = first_step_score + 0.6 * second_step_score

        if self.verbose:
            print(
                "[PredictiveBonus2Step]",
                {
                    "action": action,
                    "mode": mode,
                    "first_step_score": round(first_step_score, 3),
                    "second_step_score": round(second_step_score, 3),
                    "total": round(total, 3),
                },
            )

        return self.predictor_weight * total

    def _score_predicted_state(
        self,
        base_state: dict,
        pred_state: dict,
        action: str,
        mode: str,
        target_pos,
        memory_summary: dict,
        heuristic_next_pos: tuple[int, int],
    ) -> float:
        score = 0.0

        base_pos = self._as_tuple(base_state.get("agent_pos"))
        pred_pos = self._as_tuple(pred_state.get("agent_pos"))

        if base_pos is None or pred_pos is None:
            return 0.0

        # 預測這步沒動，強烈扣分
        if pred_pos == base_pos:
            score -= 8.0

        # predictor 與 heuristic move 不一致，略扣
        if pred_pos != heuristic_next_pos:
            score -= 2.0

        # 若有 target，優先看距離改善
        if target_pos is not None:
            current_dist = self._manhattan(base_pos, target_pos)
            pred_dist = self._manhattan(pred_pos, target_pos)
            score += 8.0 * (current_dist - pred_dist)

        # mode-specific prediction bonus
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
            if bool(pred_state.get("key_visible", False)):
                score += 8.0
            score += self._object_seek_bonus(
                pred_pos,
                pred_state,
                object_pos_key="visible_key_pos",
                object_visible_key="key_visible",
                seen_bonus=4.0,
                near_bonus=3.0,
            )
            score += self._predicted_novelty_bonus(pred_pos, memory_summary)

        elif mode == "post_key_explore":
            if bool(pred_state.get("door_visible", False)):
                score += 7.0
            score += self._object_seek_bonus(
                pred_pos,
                pred_state,
                object_pos_key="visible_door_pos",
                object_visible_key="door_visible",
                seen_bonus=4.0,
                near_bonus=3.0,
            )
            score += self._predicted_novelty_bonus(pred_pos, memory_summary)

        elif mode == "post_door_explore":
            if bool(pred_state.get("goal_visible", False)):
                score += 9.0
            score += self._object_seek_bonus(
                pred_pos,
                pred_state,
                object_pos_key="visible_goal_pos",
                object_visible_key="goal_visible",
                seen_bonus=5.0,
                near_bonus=4.0,
            )
            score += self._predicted_novelty_bonus(pred_pos, memory_summary)

        else:
            score += self._predicted_novelty_bonus(pred_pos, memory_summary)

        # =========================
        # V7: information gain bonus
        # =========================
        view_gain_score = float(pred_state.get("view_gain_score", 0.0))
        score += 1.5 * view_gain_score

        if bool(pred_state.get("new_key_seen", False)):
            score += 6.0

        if bool(pred_state.get("new_door_seen", False)):
            score += 5.0

        if bool(pred_state.get("new_goal_seen", False)):
            score += 7.0

        # =========================
        # V7: local wall openness bonus
        # =========================
        pred_walls = pred_state.get("local_walls", {})
        if isinstance(pred_walls, dict):
            open_count = 0
            for k in ("up", "down", "left", "right"):
                if pred_walls.get(k) is False:
                    open_count += 1

            # 太封閉略扣，較開闊略加
            if open_count <= 1:
                score -= 3.0
            elif open_count == 2:
                score += 0.5
            elif open_count >= 3:
                score += 2.0

        return score

    def _best_second_step_value(
        self,
        pred_state: dict,
        mode: str,
        target_pos,
        memory_summary: dict,
    ) -> float:

        pred1_pos = self._as_tuple(pred_state.get("agent_pos"))
        if pred1_pos is None:
            return 0.0
        scores = []

        for action2 in self.ROLLOUT_ACTIONS:
            heuristic_next_pos2 = self._simulate_move(pred1_pos, action2)

            pred2 = self._predict_next_state_safe(pred_state, action2)
            if pred2 is None:
                continue

            step2_score = self._score_predicted_state(
                base_state=pred_state,
                pred_state=pred2,
                action=action2,
                mode=mode,
                target_pos=target_pos,
                memory_summary=memory_summary,
                heuristic_next_pos=heuristic_next_pos2,
            )

            scores.append(step2_score)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def _predict_next_state_safe(self, z_t: dict, action: str):
        if self.predictor is None:
            return None
        skill = {"skill": "move", "args": {"direction": action}}
        try:
            return self.predictor.predict_next_state(z_t, skill)
        except Exception as e:
            if self.verbose:
                print(f"[RulePlanner] predictor inference failed: {e}")
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

    def _novelty_bonus(
        self,
        next_pos: tuple[int, int],
        memory_summary: dict,
    ) -> float:
        visit_counts = memory_summary.get("visit_counts", {})
        if next_pos not in visit_counts:
            return 7.0
        return 0.0

    def _visit_count_bonus(
        self,
        next_pos: tuple[int, int],
        memory_summary: dict,
    ) -> float:
        visit_counts = memory_summary.get("visit_counts", {})
        count = visit_counts.get(next_pos, 0)

        if count == 0:
            return 5.0
        if count == 1:
            return 2.0
        if count == 2:
            return -1.0
        return -2.5 * count

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

    def _pre_key_door_penalty(
        self,
        agent_pos: tuple[int, int],
        next_pos: tuple[int, int],
        memory_summary: dict,
    ) -> float:
        door_pos = memory_summary.get("known_door_pos", None)
        if door_pos is None:
            return 0.0

        door_pos = tuple(door_pos)
        current_dist = self._manhattan(agent_pos, door_pos)
        next_dist = self._manhattan(next_pos, door_pos)

        penalty = 0.0
        if next_dist < current_dist:
            penalty -= 4.0
        elif next_dist > current_dist:
            penalty += 1.5

        if next_dist <= 1:
            penalty -= 8.0
        elif next_dist == 2:
            penalty -= 3.0
        return penalty

    def _post_door_bonus(
        self,
        agent_pos: tuple[int, int],
        next_pos: tuple[int, int],
        memory_summary: dict,
    ) -> float:
        door_pos = memory_summary.get("known_door_pos", None)
        if door_pos is None:
            return 0.0

        door_pos = tuple(door_pos)
        current_dist = self._manhattan(agent_pos, door_pos)
        next_dist = self._manhattan(next_pos, door_pos)

        bonus = 0.0
        if next_dist > current_dist:
            bonus += 4.0
        elif next_dist < current_dist:
            bonus -= 3.0

        if next_dist <= 1:
            bonus -= 6.0
        return bonus

    def _search_door_bias(
        self,
        agent_pos: tuple[int, int],
        next_pos: tuple[int, int],
        memory_summary: dict,
    ) -> float:
        recent_positions = memory_summary.get("recent_positions", [])
        recent_tail = set(tuple(p) for p in recent_positions[-6:])
        if next_pos in recent_tail:
            return -2.0
        return 1.0

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

    def _simulate_move(
        self,
        agent_pos: tuple[int, int],
        action: str,
    ) -> tuple[int, int]:
        r, c = agent_pos
        dr, dc = self.DIR_TO_DELTA[action]
        return (r + dr, c + dc)

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

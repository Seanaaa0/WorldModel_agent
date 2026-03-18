from planner.planner_base import BasePlanner


class RulePlanner(BasePlanner):
    """
    V5-b Rule Planner / Execution Engine

    Core role:
    - robust executor for phase-guided behavior
    - can still self-infer phase if forced_phase is not provided
    - supports BFS over known memory map + frontier exploration + local fallback

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

        # Hard stuck handling
        if loop_hints.get("is_stuck", False):
            return {"skill": "escape_loop", "args": {}}

        forced_phase = planner_context.get("forced_phase", None)

        # Replan after actual failure can trigger one scan
        if replan and not just_scanned:
            if last_info is not None and (
                last_info.get("hit_wall", False)
                or last_info.get("out_of_bounds", False)
                or last_info.get("blocked_by_locked_door", False)
            ):
                return {"skill": "scan", "args": {}}

        # Periodic scan during uncertainty
        if step_count > 0 and not just_scanned:
            if not has_key:
                if visible_key_pos is None and known_key_pos is None and step_count % 10 == 0:
                    return {"skill": "scan", "args": {}}
            else:
                if visible_goal_pos is None and known_goal_pos is None and step_count % 8 == 0:
                    return {"skill": "scan", "args": {}}

        # =====================================================
        # Phase / mode selection
        # =====================================================

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

        # =====================================================
        # Global BFS first
        # =====================================================

        memory_obj = planner_context.get("memory_obj", None)

        if memory_obj is not None:
            # 1) known target path
            if target_pos is not None:
                path = memory_obj.get_path_to_known_target(
                    agent_pos, target_pos)
                action = memory_obj.first_action_from_path(path)
                if action is not None and not walls.get(action.lower(), False):
                    return {
                        "skill": "move",
                        "args": {"direction": action},
                    }

            # 2) frontier target path
            path = memory_obj.get_path_to_best_frontier(
                agent_pos=agent_pos,
                top_k=10,
                mode=mode,
            )
            action = memory_obj.first_action_from_path(path)
            if action is not None and not walls.get(action.lower(), False):
                return {
                    "skill": "move",
                    "args": {"direction": action},
                }

        # =====================================================
        # Local fallback
        # =====================================================

        if target_pos is not None:
            chosen_direction = self._choose_target_directed_direction(
                agent_pos=agent_pos,
                target_pos=target_pos,
                walls=walls,
                memory_summary=memory_summary,
                frontier_candidates=frontier_candidates,
                loop_hints=loop_hints,
                last_info=last_info,
                mode=mode,
            )
        else:
            chosen_direction = self._choose_exploration_direction(
                agent_pos=agent_pos,
                walls=walls,
                memory_summary=memory_summary,
                frontier_candidates=frontier_candidates,
                loop_hints=loop_hints,
                last_info=last_info,
                mode=mode,
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
        # if phase is externally forced by LLM, obey it first
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

        # =====================================================
        # default self-inferred behavior (V5-a / backward compatible)
        # =====================================================

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
                (visible_door_open is True) or
                (known_door_open is True)
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
    # Local fallback selection
    # =========================================================

    def _choose_target_directed_direction(
        self,
        agent_pos: tuple[int, int],
        target_pos: tuple[int, int],
        walls: dict,
        memory_summary: dict,
        frontier_candidates: list,
        loop_hints: dict,
        last_info: dict | None,
        mode: str,
    ) -> str | None:
        candidates = self._get_open_candidates(walls)
        if not candidates:
            return None

        scored = []
        for action in candidates:
            score = self._score_target_directed_move(
                agent_pos=agent_pos,
                target_pos=target_pos,
                action=action,
                memory_summary=memory_summary,
                frontier_candidates=frontier_candidates,
                loop_hints=loop_hints,
                last_info=last_info,
                mode=mode,
            )
            scored.append((score, action))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _choose_exploration_direction(
        self,
        agent_pos: tuple[int, int],
        walls: dict,
        memory_summary: dict,
        frontier_candidates: list,
        loop_hints: dict,
        last_info: dict | None,
        mode: str = "explore",
    ) -> str | None:
        if mode == "recover":
            return None

        candidates = self._get_open_candidates(walls)
        if not candidates:
            return None

        scored = []
        for action in candidates:
            score = self._score_exploration_move(
                agent_pos=agent_pos,
                action=action,
                memory_summary=memory_summary,
                frontier_candidates=frontier_candidates,
                loop_hints=loop_hints,
                last_info=last_info,
                mode=mode,
            )
            scored.append((score, action))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _get_open_candidates(self, walls: dict) -> list[str]:
        actions = ["UP", "RIGHT", "DOWN", "LEFT"]
        return [a for a in actions if not walls.get(a.lower(), False)]

    # =========================================================
    # Local target mode
    # =========================================================

    def _score_target_directed_move(
        self,
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

        score = 0.0
        score += 15.0 * (current_dist - next_dist)
        score += self._target_axis_bias(agent_pos, target_pos, action)
        score += 0.8 * self._novelty_bonus(next_pos, memory_summary)
        score += 0.3 * self._frontier_bonus(
            agent_pos, next_pos, memory_summary, frontier_candidates
        )

        if next_pos == target_pos:
            if mode == "to_key":
                score += 14.0
            elif mode == "to_door":
                score += 8.0
            elif mode == "to_goal":
                score += 12.0

        if mode == "to_key":
            score += 2.0 * (current_dist - next_dist)

        score += self._visit_count_penalty(next_pos, memory_summary)
        score += self._repeat_penalty(next_pos, memory_summary)
        score += self._failed_direction_penalty(action, last_info)
        score += self._oscillation_penalty(next_pos, loop_hints)

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

    # =========================================================
    # Local exploration mode
    # =========================================================

    def _score_exploration_move(
        self,
        agent_pos: tuple[int, int],
        action: str,
        memory_summary: dict,
        frontier_candidates: list,
        loop_hints: dict,
        last_info: dict | None,
        mode: str = "explore",
    ) -> float:
        next_pos = self._simulate_move(agent_pos, action)

        score = 0.0

        score += self._novelty_bonus(next_pos, memory_summary)
        score += self._visit_count_bonus(next_pos, memory_summary)
        score += self._frontier_bonus(
            agent_pos, next_pos, memory_summary, frontier_candidates
        )

        if mode == "pre_key_explore":
            score += self._pre_key_door_penalty(
                agent_pos=agent_pos,
                next_pos=next_pos,
                memory_summary=memory_summary,
            )

        if mode == "post_door_explore":
            score += self._post_door_bonus(
                agent_pos=agent_pos,
                next_pos=next_pos,
                memory_summary=memory_summary,
            )

        if mode == "post_key_explore":
            score += self._search_door_bias(
                agent_pos=agent_pos,
                next_pos=next_pos,
                memory_summary=memory_summary,
            )

        score += self._repeat_penalty(next_pos, memory_summary)
        score += self._failed_direction_penalty(action, last_info)
        score += self._oscillation_penalty(next_pos, loop_hints)

        return score

    # =========================================================
    # Scoring helpers
    # =========================================================

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
        # mild bias to keep exploring novel/frontier rather than circling key area
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

    # =========================================================
    # Geometry helpers
    # =========================================================

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

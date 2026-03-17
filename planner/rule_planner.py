from planner.planner_base import BasePlanner


class RulePlanner(BasePlanner):
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
    ) -> dict:
        memory_summary = memory_summary or {}
        memory_patch = memory_patch or []
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

        # Hard stuck handling
        if loop_hints.get("is_stuck", False):
            return {"skill": "escape_loop", "args": {}}

        # Replan after actual failure can trigger one scan
        if replan and not just_scanned:
            if last_info is not None and (
                last_info.get("hit_wall", False) or last_info.get(
                    "out_of_bounds", False)
            ):
                return {"skill": "scan", "args": {}}

        # Periodic scan only during exploration
        if (
            goal_pos is None
            and step_count > 0
            and step_count % 12 == 0
            and not just_scanned
        ):
            return {"skill": "scan", "args": {}}

        if goal_pos is not None:
            chosen_direction = self._choose_goal_directed_direction(
                agent_pos=agent_pos,
                goal_pos=goal_pos,
                walls=walls,
                memory_summary=memory_summary,
                frontier_candidates=frontier_candidates,
                loop_hints=loop_hints,
                last_info=last_info,
            )
        else:
            chosen_direction = self._choose_exploration_direction(
                agent_pos=agent_pos,
                walls=walls,
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
    # Direction selection
    # =========================================================

    def _choose_goal_directed_direction(
        self,
        agent_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        walls: dict,
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
            score = self._score_goal_directed_move(
                agent_pos=agent_pos,
                goal_pos=goal_pos,
                action=action,
                memory_summary=memory_summary,
                frontier_candidates=frontier_candidates,
                loop_hints=loop_hints,
                last_info=last_info,
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
    ) -> str | None:
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
            )
            scored.append((score, action))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _get_open_candidates(self, walls: dict) -> list[str]:
        actions = ["UP", "RIGHT", "DOWN", "LEFT"]
        return [a for a in actions if not walls.get(a.lower(), False)]

    # =========================================================
    # Goal mode
    # =========================================================

    def _score_goal_directed_move(
        self,
        agent_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        action: str,
        memory_summary: dict,
        frontier_candidates: list,
        loop_hints: dict,
        last_info: dict | None,
    ) -> float:
        next_pos = self._simulate_move(agent_pos, action)

        current_dist = self._manhattan(agent_pos, goal_pos)
        next_dist = self._manhattan(next_pos, goal_pos)

        score = 0.0

        # 1) Strong goal attraction
        score += 14.0 * (current_dist - next_dist)

        # 2) Explicit directional bias toward goal axis
        score += self._goal_axis_bias(agent_pos, goal_pos, action)

        # 3) Still prefer fresher ground, but weaker than goal bias
        score += 0.8 * self._novelty_bonus(next_pos, memory_summary)
        score += 0.3 * self._frontier_bonus(
            agent_pos, next_pos, memory_summary, frontier_candidates
        )

        # 4) Strong anti-loop / anti-repeat in goal mode
        score += self._visit_count_penalty(next_pos, memory_summary)
        score += self._repeat_penalty(next_pos, memory_summary)
        score += self._failed_direction_penalty(action, last_info)
        score += self._oscillation_penalty(next_pos, loop_hints)

        return score

    def _goal_axis_bias(
        self,
        agent_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        action: str,
    ) -> float:
        """
        Extra bias for moves aligned with the larger remaining axis error.
        """
        ar, ac = agent_pos
        gr, gc = goal_pos

        dy = gr - ar
        dx = gc - ac

        score = 0.0

        # larger-error axis gets stronger preference
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
    # Exploration mode
    # =========================================================

    def _score_exploration_move(
        self,
        agent_pos: tuple[int, int],
        action: str,
        memory_summary: dict,
        frontier_candidates: list,
        loop_hints: dict,
        last_info: dict | None,
    ) -> float:
        next_pos = self._simulate_move(agent_pos, action)

        score = 0.0

        # 1) Prefer unexplored / low-visit regions
        score += self._novelty_bonus(next_pos, memory_summary)
        score += self._visit_count_bonus(next_pos, memory_summary)

        # 2) Prefer moves that get closer to good frontier targets
        score += self._frontier_bonus(
            agent_pos, next_pos, memory_summary, frontier_candidates
        )

        # 3) Stronger anti-repeat
        score += self._repeat_penalty(next_pos, memory_summary)
        score += self._failed_direction_penalty(action, last_info)
        score += self._oscillation_penalty(next_pos, loop_hints)

        return score

    # =========================================================
    # Memory / frontier / repeat scoring
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
        """
        Goal mode: heavily dislike repeatedly revisiting high-count cells.
        """
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
        """
        Two-part frontier reward:
        1) direct bonus if next_pos itself is a frontier
        2) shaping reward if next_pos gets closer to a strong frontier target
        """
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

        # Use best frontier target as shaping anchor
        best_target, best_frontier_score, _ = max(parsed, key=lambda x: x[1])

        current_dist = self._manhattan(agent_pos, best_target)
        next_dist = self._manhattan(next_pos, best_target)

        if next_dist < current_dist:
            bonus += 3.5
        elif next_dist > current_dist:
            bonus -= 1.5

        # small preference for moving toward high-score frontier
        bonus += min(best_frontier_score, 4.0) * 0.5

        # prefer not to re-enter very recent area even if frontier-ish
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

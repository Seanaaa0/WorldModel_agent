from collections import deque, defaultdict
from typing import Deque, Dict, List, Optional, Set, Tuple


class WorldMemory:
    """
    PO-friendly world memory for the maze agent.

    Responsibilities:
    - track visited positions
    - track recent positions (for stuck / oscillation detection)
    - store known local wall information at visited positions
    - store discovered free cells from local observations
    - remember goal position once it has been seen
    - maintain simple exploration statistics
    - maintain visit counts for anti-loop / novelty heuristics

    V3 additions:
    - get_memory_patch()
    - get_frontier_candidates()
    - get_loop_hints()
    - get_planner_context()
    """

    DIR_TO_DELTA = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    def __init__(self, recent_window: int = 10) -> None:
        self.recent_window = recent_window

        self.visited_positions: Set[Tuple[int, int]] = set()
        self.recent_positions: Deque[Tuple[int, int]] = deque(
            maxlen=recent_window)

        # Example:
        # {(3,4): {"up": False, "down": True, "left": False, "right": True}}
        self.known_walls: Dict[Tuple[int, int], Dict[str, bool]] = {}

        # Cells that we have positively observed as traversable.
        self.known_free_positions: Set[Tuple[int, int]] = set()

        # Cells positively observed as walls from local_view.
        self.known_wall_positions: Set[Tuple[int, int]] = set()

        # All cells that have appeared in local observation and are in-bounds.
        self.observed_cells: Set[Tuple[int, int]] = set()

        # Goal memory: only known after first sighting.
        self.known_goal_pos: Optional[Tuple[int, int]] = None
        self.goal_last_seen_step: Optional[int] = None
        self.seen_goal_count: int = 0

        # Visit frequency for exploration heuristic.
        self.visit_counts: Dict[Tuple[int, int], int] = defaultdict(int)

        self.total_steps_observed: int = 0

    def reset(self) -> None:
        """
        Reset all memory for a new episode.
        """
        self.visited_positions.clear()
        self.recent_positions.clear()
        self.known_walls.clear()
        self.known_free_positions.clear()
        self.known_wall_positions.clear()
        self.observed_cells.clear()

        self.known_goal_pos = None
        self.goal_last_seen_step = None
        self.seen_goal_count = 0

        self.visit_counts.clear()
        self.total_steps_observed = 0

    def update(self, z_t: dict, info: Optional[dict] = None) -> None:
        """
        Update memory using current latent state and optional env step info.

        Expected z_t fields (PO v1):
        {
            "agent_pos": (r, c),
            "goal_visible": bool,
            "goal_pos": (gr, gc) or None,
            "local_walls": {...},
            "local_view": [[...], [...], [...]],
            "step_count": int,
            "view_radius": int,
        }
        """
        pos = z_t["agent_pos"]
        local_walls = z_t["local_walls"]
        local_view = z_t["local_view"]
        goal_visible = z_t["goal_visible"]
        goal_pos = z_t["goal_pos"]
        step_count = z_t["step_count"]
        view_radius = z_t["view_radius"]

        self.visited_positions.add(pos)
        self.recent_positions.append(pos)
        self.visit_counts[pos] += 1

        self.known_walls[pos] = dict(local_walls)
        self.known_free_positions.add(pos)

        self._update_from_local_walls(pos, local_walls)
        self._update_from_local_view(pos, local_view, view_radius)

        if goal_visible and goal_pos is not None:
            self.known_goal_pos = goal_pos
            self.goal_last_seen_step = step_count
            self.seen_goal_count += 1

        self.total_steps_observed += 1

    def _update_from_local_walls(
        self,
        pos: Tuple[int, int],
        local_walls: Dict[str, bool],
    ) -> None:
        """
        Infer directly adjacent free cells from local wall observations.

        If direction is not a wall, the adjacent cell is known traversable.
        """
        r, c = pos
        for direction, is_wall in local_walls.items():
            dr, dc = self.DIR_TO_DELTA[direction]
            nr, nc = r + dr, c + dc

            if is_wall:
                self.observed_cells.add((nr, nc))
                self.known_wall_positions.add((nr, nc))
            else:
                self.known_free_positions.add((nr, nc))
                self.observed_cells.add((nr, nc))

        self.observed_cells.add(pos)

    def _update_from_local_view(
        self,
        pos: Tuple[int, int],
        local_view: list,
        view_radius: int,
    ) -> None:
        """
        Parse local view patch and store observed / free / goal information.

        local_view tokens:
        - BOUNDARY
        - WALL
        - FREE
        - GOAL
        - AGENT
        """
        center_r, center_c = pos

        for i, row in enumerate(local_view):
            for j, token in enumerate(row):
                dr = i - view_radius
                dc = j - view_radius
                world_pos = (center_r + dr, center_c + dc)

                if token == "BOUNDARY":
                    continue

                self.observed_cells.add(world_pos)

                if token in ("FREE", "AGENT", "GOAL"):
                    self.known_free_positions.add(world_pos)

                if token == "WALL":
                    self.known_wall_positions.add(world_pos)

                if token == "GOAL":
                    self.known_goal_pos = world_pos

    def get_summary(self) -> dict:
        """
        Return a compact summary for planner / monitor.
        """
        return {
            "visited_count": len(self.visited_positions),
            "recent_positions": list(self.recent_positions),
            "known_goal_pos": self.known_goal_pos,
            "goal_last_seen_step": self.goal_last_seen_step,
            "seen_goal_count": self.seen_goal_count,
            "known_walls_count": len(self.known_walls),
            "known_free_count": len(self.known_free_positions),
            "known_wall_pos_count": len(self.known_wall_positions),
            "observed_cells_count": len(self.observed_cells),
            "visit_counts": dict(self.visit_counts),
            "total_steps_observed": self.total_steps_observed,
        }

    def has_visited(self, pos: Tuple[int, int]) -> bool:
        return pos in self.visited_positions

    def is_known_free(self, pos: Tuple[int, int]) -> bool:
        return pos in self.known_free_positions

    def is_observed(self, pos: Tuple[int, int]) -> bool:
        return pos in self.observed_cells

    def get_walls(self, pos: Tuple[int, int]) -> Optional[Dict[str, bool]]:
        return self.known_walls.get(pos)

    def has_seen_goal(self) -> bool:
        return self.known_goal_pos is not None

    def get_visit_count(self, pos: Tuple[int, int]) -> int:
        return self.visit_counts.get(pos, 0)

    def estimate_local_frontier_score(self, pos: Tuple[int, int]) -> int:
        """
        Heuristic frontier score:
        count how many 4-neighbors of pos are still unobserved.
        Larger score => this position is closer to unexplored territory.
        """
        r, c = pos
        score = 0
        for dr, dc in self.DIR_TO_DELTA.values():
            nbr = (r + dr, c + dc)
            if nbr not in self.observed_cells:
                score += 1
        return score

    def unique_recent_positions(self) -> int:
        """
        Number of unique positions inside the recent movement window.
        Small values may indicate oscillation / getting stuck.
        """
        return len(set(self.recent_positions))

    def is_stuck_by_repetition(
        self,
        threshold_unique: int = 2,
        min_window: int = 6,
    ) -> bool:
        """
        Heuristic stuck detector:
        if agent has enough recent history, but only visited very few unique positions,
        it is probably oscillating.
        """
        if len(self.recent_positions) < min_window:
            return False
        return self.unique_recent_positions() <= threshold_unique

    # =========================================================
    # V3 planner-facing interfaces
    # =========================================================

    def _memory_token_for_pos(
        self,
        pos: Tuple[int, int],
        agent_pos: Tuple[int, int],
    ) -> str:
        """
        Priority:
        A > G > # > V > . > ?
        """
        if pos == agent_pos:
            return "A"

        if self.known_goal_pos is not None and pos == self.known_goal_pos:
            return "G"

        if pos in self.known_wall_positions:
            return "#"

        if pos in self.visited_positions:
            return "V"

        if pos in self.known_free_positions:
            return "."

        return "?"

    def get_memory_patch(
        self,
        center_pos: Tuple[int, int],
        patch_radius: int = 3,
    ) -> List[List[str]]:
        """
        Return a planner-friendly memory patch centered on agent.

        Tokens:
        A = agent
        G = known goal
        # = known wall
        V = visited free cell
        . = known free but not visited
        ? = unknown
        """
        center_r, center_c = center_pos
        patch: List[List[str]] = []

        for dr in range(-patch_radius, patch_radius + 1):
            row_tokens: List[str] = []
            for dc in range(-patch_radius, patch_radius + 1):
                pos = (center_r + dr, center_c + dc)
                row_tokens.append(self._memory_token_for_pos(pos, center_pos))
            patch.append(row_tokens)

        return patch

    def _is_frontier_candidate(self, pos: Tuple[int, int]) -> bool:
        """
        A frontier candidate is a known traversable position with at least one
        unknown 4-neighbor.
        """
        if pos in self.known_wall_positions:
            return False

        if pos not in self.known_free_positions and pos not in self.visited_positions:
            return False

        return self.estimate_local_frontier_score(pos) > 0

    def _manhattan_distance(
        self,
        a: Tuple[int, int],
        b: Tuple[int, int],
    ) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_frontier_candidates(
        self,
        agent_pos: Tuple[int, int],
        top_k: int = 5,
    ) -> List[dict]:
        """
        Return top-k frontier candidates for planner.

        Candidate fields:
        - pos
        - frontier_score
        - visit_count
        - distance_from_agent
        - is_recent
        """
        candidates: List[dict] = []
        recent_set = set(self.recent_positions)

        # Use known traversable area as search base.
        candidate_space = self.known_free_positions | self.visited_positions

        for pos in candidate_space:
            if not self._is_frontier_candidate(pos):
                continue

            item = {
                "pos": pos,
                "frontier_score": self.estimate_local_frontier_score(pos),
                "visit_count": self.get_visit_count(pos),
                "distance_from_agent": self._manhattan_distance(agent_pos, pos),
                "is_recent": pos in recent_set,
            }
            candidates.append(item)

        # Sort by:
        # 1) higher frontier_score
        # 2) shorter distance
        # 3) lower visit count
        # 4) non-recent preferred
        candidates.sort(
            key=lambda x: (
                -x["frontier_score"],
                x["distance_from_agent"],
                x["visit_count"],
                x["is_recent"],
            )
        )

        return candidates[:top_k]

    def _detect_oscillation_pair(self) -> Optional[List[Tuple[int, int]]]:
        """
        Detect simple ABAB oscillation in recent positions.
        Example:
        [A, B, A, B] -> return [A, B]
        """
        recent = list(self.recent_positions)
        if len(recent) < 4:
            return None

        a, b, c, d = recent[-4], recent[-3], recent[-2], recent[-1]
        if a == c and b == d and a != b:
            return [a, b]

        return None

    def get_loop_hints(self) -> dict:
        """
        Return planner-facing anti-loop hints.
        """
        current_pos = self.recent_positions[-1] if self.recent_positions else None
        oscillation_pair = self._detect_oscillation_pair()

        return {
            "is_stuck": self.is_stuck_by_repetition(),
            "recent_positions": list(self.recent_positions),
            "repeat_count_current_pos": self.get_visit_count(current_pos) if current_pos is not None else 0,
            "oscillation_pair": oscillation_pair,
            "unique_recent_positions": self.unique_recent_positions(),
        }

    def get_planner_context(
        self,
        agent_pos: Tuple[int, int],
        patch_radius: int = 3,
        top_k_frontiers: int = 5,
    ) -> dict:
        """
        Convenience wrapper for V3 planner input.
        """
        return {
            "memory_summary": self.get_summary(),
            "memory_patch": self.get_memory_patch(
                center_pos=agent_pos,
                patch_radius=patch_radius,
            ),
            "frontier_candidates": self.get_frontier_candidates(
                agent_pos=agent_pos,
                top_k=top_k_frontiers,
            ),
            "loop_hints": self.get_loop_hints(),
        }

    def to_debug_dict(self) -> dict:
        """
        Verbose state dump for debugging.
        """
        return {
            "visited_positions": sorted(list(self.visited_positions)),
            "recent_positions": list(self.recent_positions),
            "known_walls": self.known_walls,
            "known_free_positions": sorted(list(self.known_free_positions)),
            "known_wall_positions": sorted(list(self.known_wall_positions)),
            "observed_cells": sorted(list(self.observed_cells)),
            "known_goal_pos": self.known_goal_pos,
            "goal_last_seen_step": self.goal_last_seen_step,
            "seen_goal_count": self.seen_goal_count,
            "visit_counts": dict(self.visit_counts),
            "total_steps_observed": self.total_steps_observed,
        }

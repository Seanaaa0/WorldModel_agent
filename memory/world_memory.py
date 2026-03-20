from collections import deque, defaultdict
from typing import Deque, Dict, List, Optional, Set, Tuple


class WorldMemory:
    """
    V5-b task-world memory with lightweight global planning support.

    Responsibilities:
    - track visited positions
    - track recent positions (for stuck / oscillation detection)
    - store known local wall information at visited positions
    - store discovered free cells / wall cells from local observations
    - remember seen object positions:
        - KEY
        - DOOR_LOCKED
        - DOOR_OPEN
        - GOAL
    - track simple inventory state (has_key)
    - maintain visit counts for exploration / anti-loop heuristics
    - provide BFS utilities over known map for planner
    """

    DIR_TO_DELTA = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    DELTA_TO_DIR = {
        (-1, 0): "UP",
        (1, 0): "DOWN",
        (0, -1): "LEFT",
        (0, 1): "RIGHT",
    }

    OBJECT_TOKENS = {
        "KEY",
        "DOOR_LOCKED",
        "DOOR_OPEN",
        "GOAL",
    }

    def __init__(self, recent_window: int = 10) -> None:
        self.recent_window = recent_window

        self.visited_positions: Set[Tuple[int, int]] = set()
        self.recent_positions: Deque[Tuple[int, int]] = deque(
            maxlen=recent_window)

        # {(r,c): {"up": bool, "down": bool, ...}}
        self.known_walls: Dict[Tuple[int, int], Dict[str, bool]] = {}

        self.known_free_positions: Set[Tuple[int, int]] = set()
        self.known_wall_positions: Set[Tuple[int, int]] = set()
        self.observed_cells: Set[Tuple[int, int]] = set()

        # Object memory
        self.known_key_pos: Optional[Tuple[int, int]] = None
        self.known_door_pos: Optional[Tuple[int, int]] = None
        self.known_door_open: Optional[bool] = None
        self.known_goal_pos: Optional[Tuple[int, int]] = None

        self.seen_key_count: int = 0
        self.seen_door_count: int = 0
        self.seen_goal_count: int = 0

        # Last seen steps
        self.key_last_seen_step: Optional[int] = None
        self.door_last_seen_step: Optional[int] = None
        self.goal_last_seen_step: Optional[int] = None

        # Agent-side task state
        self.has_key: bool = False

        # Visit frequency
        self.visit_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        self.total_steps_observed: int = 0

    def reset(self) -> None:
        self.visited_positions.clear()
        self.recent_positions.clear()
        self.known_walls.clear()
        self.known_free_positions.clear()
        self.known_wall_positions.clear()
        self.observed_cells.clear()

        self.known_key_pos = None
        self.known_door_pos = None
        self.known_door_open = None
        self.known_goal_pos = None

        self.seen_key_count = 0
        self.seen_door_count = 0
        self.seen_goal_count = 0

        self.key_last_seen_step = None
        self.door_last_seen_step = None
        self.goal_last_seen_step = None

        self.has_key = False

        self.visit_counts.clear()
        self.total_steps_observed = 0

    def update(self, z_t: dict, info: Optional[dict] = None) -> None:
        pos = z_t["agent_pos"]
        local_walls = z_t["local_walls"]
        local_view = z_t["local_view"]
        visible_objects = z_t["visible_objects"]
        has_key = bool(z_t["has_key"])
        step_count = z_t["step_count"]
        view_radius = z_t["view_radius"]

        self.visited_positions.add(pos)
        self.recent_positions.append(pos)
        self.visit_counts[pos] += 1

        self.known_walls[pos] = dict(local_walls)
        self.known_free_positions.add(pos)

        self.has_key = has_key

        self._update_from_local_walls(pos, local_walls)
        self._update_from_local_view(pos, local_view, view_radius)
        self._update_from_visible_objects(visible_objects, step_count)

        if has_key:
            self.known_key_pos = None

        if info is not None and info.get("opened_door", False):
            if info.get("new_pos") is not None:
                self.known_door_pos = tuple(info["new_pos"])
            self.known_door_open = True

        self.total_steps_observed += 1

    def _update_from_local_walls(
        self,
        pos: Tuple[int, int],
        local_walls: Dict[str, bool],
    ) -> None:
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
        center_r, center_c = pos

        for i, row in enumerate(local_view):
            for j, token in enumerate(row):
                dr = i - view_radius
                dc = j - view_radius
                world_pos = (center_r + dr, center_c + dc)

                if token == "BOUNDARY":
                    continue

                self.observed_cells.add(world_pos)

                if token in ("EMPTY", "AGENT", "KEY", "DOOR_OPEN", "GOAL"):
                    self.known_free_positions.add(world_pos)

                if token == "WALL":
                    self.known_wall_positions.add(world_pos)

                if token == "KEY":
                    if not self.has_key:
                        self.known_key_pos = world_pos

                if token == "DOOR_LOCKED":
                    self.known_door_pos = world_pos
                    self.known_door_open = False

                if token == "DOOR_OPEN":
                    self.known_door_pos = world_pos
                    self.known_door_open = True

                if token == "GOAL":
                    self.known_goal_pos = world_pos

    def _update_from_visible_objects(
        self,
        visible_objects: list,
        step_count: int,
    ) -> None:
        for item in visible_objects:
            obj_type = item["type"]
            obj_pos = tuple(item["pos"])

            if obj_type == "KEY":
                if not self.has_key:
                    self.known_key_pos = obj_pos
                    self.key_last_seen_step = step_count
                    self.seen_key_count += 1

            elif obj_type == "DOOR_LOCKED":
                self.known_door_pos = obj_pos
                self.known_door_open = False
                self.door_last_seen_step = step_count
                self.seen_door_count += 1

            elif obj_type == "DOOR_OPEN":
                self.known_door_pos = obj_pos
                self.known_door_open = True
                self.door_last_seen_step = step_count
                self.seen_door_count += 1

            elif obj_type == "GOAL":
                self.known_goal_pos = obj_pos
                self.goal_last_seen_step = step_count
                self.seen_goal_count += 1

    def get_summary(self) -> dict:
        return {
            "visited_count": len(self.visited_positions),
            "recent_positions": list(self.recent_positions),

            "known_key_pos": self.known_key_pos,
            "known_door_pos": self.known_door_pos,
            "known_door_open": self.known_door_open,
            "known_goal_pos": self.known_goal_pos,

            "has_key": self.has_key,

            "key_last_seen_step": self.key_last_seen_step,
            "door_last_seen_step": self.door_last_seen_step,
            "goal_last_seen_step": self.goal_last_seen_step,

            "seen_key_count": self.seen_key_count,
            "seen_door_count": self.seen_door_count,
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
        r, c = pos
        score = 0
        for dr, dc in self.DIR_TO_DELTA.values():
            nbr = (r + dr, c + dc)
            if nbr not in self.observed_cells:
                score += 1
        return score

    def unique_recent_positions(self) -> int:
        return len(set(self.recent_positions))

    def is_stuck_by_repetition(
        self,
        threshold_unique: int = 2,
        min_window: int = 6,
    ) -> bool:
        if len(self.recent_positions) < min_window:
            return False
        return self.unique_recent_positions() <= threshold_unique

    # =========================================================
    # Planner-facing interfaces
    # =========================================================

    def _memory_token_for_pos(
        self,
        pos: Tuple[int, int],
        agent_pos: Tuple[int, int],
    ) -> str:
        if pos == agent_pos:
            return "A"

        if self.known_key_pos is not None and pos == self.known_key_pos and not self.has_key:
            return "K"

        if self.known_door_pos is not None and pos == self.known_door_pos:
            if self.known_door_open is True:
                return "O"
            return "D"

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
        candidates: List[dict] = []
        recent_set = set(self.recent_positions)

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

        candidates.sort(
            key=lambda x: (
                -x["frontier_score"],
                x["distance_from_agent"],
                x["visit_count"],
                x["is_recent"],
            )
        )

        return candidates[:top_k]

    # =========================================================
    # NEW: global planning helpers
    # =========================================================

    def is_planning_passable(self, pos: Tuple[int, int]) -> bool:
        if pos in self.known_wall_positions:
            return False

        if pos == self.known_door_pos and self.known_door_open is False and not self.has_key:
            return False

        return pos in self.known_free_positions or pos in self.visited_positions or pos == self.known_door_pos

    def get_neighbors_for_planning(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        out = []
        r, c = pos

        for dr, dc in self.DIR_TO_DELTA.values():
            nbr = (r + dr, c + dc)
            if self.is_planning_passable(nbr):
                out.append(nbr)

        return out

    def find_path_bfs(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Return full path [start, ..., goal] on known-free map, or None.
        """
        if start == goal:
            return [start]

        if not self.is_planning_passable(start):
            return None

        # allow goal if it is known target cell even if not yet in known_free_positions
        if goal in self.known_wall_positions:
            return None

        q = deque([start])
        parent: Dict[Tuple[int, int],
                     Optional[Tuple[int, int]]] = {start: None}

        while q:
            cur = q.popleft()

            for nbr in self.get_neighbors_for_planning(cur):
                if nbr in parent:
                    continue
                parent[nbr] = cur
                if nbr == goal:
                    return self._reconstruct_path(parent, goal)
                q.append(nbr)

        return None

    def _reconstruct_path(
        self,
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    def first_action_from_path(
        self,
        path: Optional[List[Tuple[int, int]]],
    ) -> Optional[str]:
        if path is None or len(path) < 2:
            return None

        a = path[0]
        b = path[1]
        dr = b[0] - a[0]
        dc = b[1] - a[1]

        return self.DELTA_TO_DIR.get((dr, dc))

    def select_best_frontier_target(
        self,
        agent_pos: Tuple[int, int],
        top_k: int = 10,
        mode: str = "explore",
    ) -> Optional[Tuple[int, int]]:
        """
        Choose a frontier target, not just one-step move.

        Score intuition:
        - prefer high frontier score
        - prefer BFS-reachable targets
        - discourage heavily visited targets
        - discourage very recent targets
        - before key: penalize door-near targets
        - after open door: reward targets farther from door
        """
        candidates = self.get_frontier_candidates(
            agent_pos=agent_pos, top_k=top_k)
        if not candidates:
            return None

        best_pos = None
        best_score = -10**18

        for item in candidates:
            pos = tuple(item["pos"])
            path = self.find_path_bfs(agent_pos, pos)
            if path is None:
                continue

            path_len = len(path) - 1
            frontier_score = float(item["frontier_score"])
            visit_count = float(item["visit_count"])
            is_recent = bool(item["is_recent"])

            score = 0.0
            score += 4.0 * frontier_score
            score -= 1.0 * path_len
            score -= 2.0 * visit_count
            if is_recent:
                score -= 3.0

            # pre-key: avoid frontiers too close to closed door
            if mode == "pre_key_explore" and self.known_door_pos is not None:
                dist_to_door = self._manhattan_distance(
                    pos, self.known_door_pos)
                if dist_to_door <= 1:
                    score -= 8.0
                elif dist_to_door == 2:
                    score -= 3.0

            # post-door: encourage going away from door region
            if mode == "post_door_explore" and self.known_door_pos is not None:
                dist_to_door = self._manhattan_distance(
                    pos, self.known_door_pos)
                score += 0.8 * dist_to_door

            if score > best_score:
                best_score = score
                best_pos = pos

        return best_pos

    def get_path_to_best_frontier(
        self,
        agent_pos: Tuple[int, int],
        top_k: int = 10,
        mode: str = "explore",
    ) -> Optional[List[Tuple[int, int]]]:
        frontier_target = self.select_best_frontier_target(
            agent_pos=agent_pos,
            top_k=top_k,
            mode=mode,
        )
        if frontier_target is None:
            return None
        return self.find_path_bfs(agent_pos, frontier_target)

    def get_path_to_known_target(
        self,
        agent_pos: Tuple[int, int],
        target_pos: Optional[Tuple[int, int]],
    ) -> Optional[List[Tuple[int, int]]]:
        if target_pos is None:
            return None
        return self.find_path_bfs(agent_pos, tuple(target_pos))

    # =========================================================
    # Loop hints / planner context
    # =========================================================

    def _detect_oscillation_pair(self) -> Optional[List[Tuple[int, int]]]:
        recent = list(self.recent_positions)
        if len(recent) < 4:
            return None

        a, b, c, d = recent[-4], recent[-3], recent[-2], recent[-1]
        if a == c and b == d and a != b:
            return [a, b]

        return None

    def get_loop_hints(self) -> dict:
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
        return {
            "visited_positions": sorted(list(self.visited_positions)),
            "recent_positions": list(self.recent_positions),
            "known_walls": self.known_walls,
            "known_free_positions": sorted(list(self.known_free_positions)),
            "known_wall_positions": sorted(list(self.known_wall_positions)),
            "observed_cells": sorted(list(self.observed_cells)),
            "known_key_pos": self.known_key_pos,
            "known_door_pos": self.known_door_pos,
            "known_door_open": self.known_door_open,
            "known_goal_pos": self.known_goal_pos,
            "has_key": self.has_key,
            "visit_counts": dict(self.visit_counts),
            "total_steps_observed": self.total_steps_observed,
        }

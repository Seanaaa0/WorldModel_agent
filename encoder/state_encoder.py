class StateEncoder:
    """
    Convert raw environment observation into latent state representation.

    PO v1 encoder:
    - accepts partial observation from MazeEnv
    - does NOT assume global goal is always known
    - keeps structured latent state for planner / memory / predictor refactor

    Expected observation schema:
    {
        "pos": (r, c),
        "walls": {"up": bool, "down": bool, "left": bool, "right": bool},
        "local_view": [[...], [...], [...]],
        "goal_visible": bool,
        "visible_goal_pos": (gr, gc) or None,
        "step_count": int,
        "view_radius": int,
    }
    """

    def __init__(self):
        pass

    def encode(self, obs: dict) -> dict:
        """
        Convert environment observation -> latent state z_t
        """
        self._validate_obs(obs)

        r, c = obs["pos"]
        walls = obs["walls"]
        local_view = obs["local_view"]
        goal_visible = obs["goal_visible"]
        visible_goal_pos = obs["visible_goal_pos"]
        step_count = obs["step_count"]
        view_radius = obs["view_radius"]

        if goal_visible and visible_goal_pos is not None:
            gr, gc = visible_goal_pos
            dx = gc - c
            dy = gr - r
            goal_distance = abs(dx) + abs(dy)
            goal_pos = (gr, gc)
        else:
            dx = None
            dy = None
            goal_distance = None
            goal_pos = None

        z_t = {
            "agent_pos": (r, c),

            # PO-related goal fields
            "goal_visible": goal_visible,
            "goal_pos": goal_pos,

            # Only defined when goal is visible
            "dx": dx,
            "dy": dy,
            "goal_distance": goal_distance,

            # Local partial observation
            "local_walls": walls,
            "local_view": local_view,

            # Metadata
            "step_count": step_count,
            "view_radius": view_radius,
        }

        return z_t

    def _validate_obs(self, obs: dict) -> None:
        """
        Basic schema validation for PO observations.
        """
        required_keys = {
            "pos",
            "walls",
            "local_view",
            "goal_visible",
            "visible_goal_pos",
            "step_count",
            "view_radius",
        }

        missing = required_keys - set(obs.keys())
        if missing:
            raise KeyError(
                f"Missing required observation keys: {sorted(missing)}")

        if not isinstance(obs["pos"], tuple) or len(obs["pos"]) != 2:
            raise ValueError('obs["pos"] must be a tuple (r, c)')

        if not isinstance(obs["walls"], dict):
            raise ValueError('obs["walls"] must be a dict')

        for k in ("up", "down", "left", "right"):
            if k not in obs["walls"]:
                raise KeyError(f'obs["walls"] missing key: "{k}"')

        if not isinstance(obs["local_view"], list):
            raise ValueError('obs["local_view"] must be a 2D list')

        if not isinstance(obs["goal_visible"], bool):
            raise ValueError('obs["goal_visible"] must be a bool')

        if obs["visible_goal_pos"] is not None:
            if (
                not isinstance(obs["visible_goal_pos"], tuple)
                or len(obs["visible_goal_pos"]) != 2
            ):
                raise ValueError(
                    'obs["visible_goal_pos"] must be None or tuple (gr, gc)')

        if not isinstance(obs["step_count"], int):
            raise ValueError('obs["step_count"] must be an int')

        if not isinstance(obs["view_radius"], int):
            raise ValueError('obs["view_radius"] must be an int')

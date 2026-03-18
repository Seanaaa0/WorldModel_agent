class StateEncoder:
    """
    V5-a Task World Encoder

    Convert raw environment observation into structured latent state.

    Expected observation schema:
    {
        "pos": (r, c),
        "walls": {"up": bool, "down": bool, "left": bool, "right": bool},
        "local_view": [[...], [...], [...]],
        "visible_objects": [
            {"type": "KEY", "pos": (r, c)},
            {"type": "DOOR_LOCKED", "pos": (r, c)},
            {"type": "DOOR_OPEN", "pos": (r, c)},
            {"type": "GOAL", "pos": (r, c)},
        ],
        "inventory": {"has_key": bool},
        "step_count": int,
        "view_radius": int,
    }
    """

    def __init__(self):
        pass

    def encode(self, obs: dict) -> dict:
        self._validate_obs(obs)

        r, c = obs["pos"]
        walls = obs["walls"]
        local_view = obs["local_view"]
        visible_objects = obs["visible_objects"]
        inventory = obs["inventory"]
        step_count = obs["step_count"]
        view_radius = obs["view_radius"]

        has_key = bool(inventory.get("has_key", False))

        visible_key_pos = None
        visible_door_pos = None
        visible_door_open = None
        visible_goal_pos = None

        for item in visible_objects:
            obj_type = item["type"]
            obj_pos = tuple(item["pos"])

            if obj_type == "KEY" and visible_key_pos is None:
                visible_key_pos = obj_pos

            elif obj_type in ("DOOR_LOCKED", "DOOR_OPEN") and visible_door_pos is None:
                visible_door_pos = obj_pos
                visible_door_open = (obj_type == "DOOR_OPEN")

            elif obj_type == "GOAL" and visible_goal_pos is None:
                visible_goal_pos = obj_pos

        z_t = {
            "agent_pos": (r, c),

            # local geometry
            "local_walls": walls,
            "local_view": local_view,

            # planner-friendly object information
            "visible_objects": visible_objects,

            "has_key": has_key,

            "key_visible": visible_key_pos is not None,
            "visible_key_pos": visible_key_pos,

            "door_visible": visible_door_pos is not None,
            "visible_door_pos": visible_door_pos,
            "visible_door_open": visible_door_open,

            "goal_visible": visible_goal_pos is not None,
            "visible_goal_pos": visible_goal_pos,

            # metadata
            "step_count": step_count,
            "view_radius": view_radius,
        }

        return z_t

    def _validate_obs(self, obs: dict) -> None:
        required_keys = {
            "pos",
            "walls",
            "local_view",
            "visible_objects",
            "inventory",
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
            if not isinstance(obs["walls"][k], bool):
                raise ValueError(f'obs["walls"]["{k}"] must be a bool')

        if not isinstance(obs["local_view"], list):
            raise ValueError('obs["local_view"] must be a 2D list')

        if not isinstance(obs["visible_objects"], list):
            raise ValueError('obs["visible_objects"] must be a list')

        for item in obs["visible_objects"]:
            if not isinstance(item, dict):
                raise ValueError('each visible object must be a dict')
            if "type" not in item or "pos" not in item:
                raise KeyError(
                    'each visible object must contain "type" and "pos"')
            if not isinstance(item["type"], str):
                raise ValueError('visible object "type" must be a string')
            if not isinstance(item["pos"], tuple) or len(item["pos"]) != 2:
                raise ValueError('visible object "pos" must be a tuple (r, c)')

        if not isinstance(obs["inventory"], dict):
            raise ValueError('obs["inventory"] must be a dict')
        if "has_key" not in obs["inventory"]:
            raise KeyError('obs["inventory"] missing key: "has_key"')
        if not isinstance(obs["inventory"]["has_key"], bool):
            raise ValueError('obs["inventory"]["has_key"] must be a bool')

        if not isinstance(obs["step_count"], int):
            raise ValueError('obs["step_count"] must be an int')

        if not isinstance(obs["view_radius"], int):
            raise ValueError('obs["view_radius"] must be an int')

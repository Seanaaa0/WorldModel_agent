import torch
import torch.nn as nn

from predictor.base_predictor import BasePredictor


class DynamicsMLP(nn.Module):
    def __init__(self, input_dim=22, hidden_dim=128, output_dim=22):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class MLPPredictor(BasePredictor):
    """
    V5/V6-aligned one-step partial-observable dynamics predictor.

    Input x (22 dims):
    - agent_pos: r, c
    - has_key
    - key_visible, door_visible, goal_visible
    - door_open_known, door_open_value
    - local_walls up/down/left/right
    - rel_key dr/dc
    - rel_door dr/dc
    - rel_goal dr/dc
    - action onehot (UP/DOWN/LEFT/RIGHT)

    Output y (22 dims):
    - next_agent_pos: r, c
    - next_has_key
    - next_key_visible, next_door_visible, next_goal_visible
    - next_door_open_known, next_door_open_value
    - next_rel_key dr/dc
    - next_rel_door dr/dc
    - next_rel_goal dr/dc
    - next_local_walls up/down/left/right
    - view_gain_score
    - new_key_seen, new_door_seen, new_goal_seen
    """

    ACTION_TO_ONEHOT = {
        "UP": [1.0, 0.0, 0.0, 0.0],
        "DOWN": [0.0, 1.0, 0.0, 0.0],
        "LEFT": [0.0, 0.0, 1.0, 0.0],
        "RIGHT": [0.0, 0.0, 0.0, 1.0],
    }

    def __init__(self, checkpoint_path=None, device=None, hidden_dim=128):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = 22
        self.output_dim = 22
        self.hidden_dim = hidden_dim
        self.model = DynamicsMLP(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
        ).to(self.device)
        self.model.eval()
        self.target_schema = None

        if checkpoint_path is not None:
            ckpt = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
            ckpt_input_dim = int(ckpt.get("input_dim", self.input_dim))
            ckpt_output_dim = int(ckpt.get("output_dim", self.output_dim))
            if ckpt_input_dim != self.input_dim or ckpt_output_dim != self.output_dim:
                raise ValueError(
                    f"Checkpoint dims mismatch: got ({ckpt_input_dim}, {ckpt_output_dim}), "
                    f"expected ({self.input_dim}, {self.output_dim})"
                )
            self.target_schema = ckpt.get("target_schema")
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()

    @staticmethod
    def _bool01(x):
        return 1.0 if bool(x) else 0.0

    @staticmethod
    def _round_bool(x):
        return bool(round(float(x)))

    @staticmethod
    def _door_open_pair(value):
        if value is None:
            return 0.0, 0.0
        return 1.0, (1.0 if bool(value) else 0.0)

    @staticmethod
    def _relative_pos(agent_pos, target_pos):
        if target_pos is None:
            return 0.0, 0.0
        return float(target_pos[0] - agent_pos[0]), float(target_pos[1] - agent_pos[1])

    @staticmethod
    def _apply_rel(agent_pos, dr, dc, visible_flag):
        if not visible_flag:
            return None
        r = int(round(float(agent_pos[0] + dr)))
        c = int(round(float(agent_pos[1] + dc)))
        return (r, c)

    def _z_action_to_tensor(self, z_t, skill_spec):
        r, c = z_t["agent_pos"]
        walls = z_t["local_walls"]

        key_rel = self._relative_pos(
            z_t["agent_pos"], z_t.get("visible_key_pos"))
        door_rel = self._relative_pos(
            z_t["agent_pos"], z_t.get("visible_door_pos"))
        goal_rel = self._relative_pos(
            z_t["agent_pos"], z_t.get("visible_goal_pos"))
        door_open_known, door_open_value = self._door_open_pair(
            z_t.get("visible_door_open"))

        if skill_spec.get("skill") != "move":
            action_onehot = [0.0, 0.0, 0.0, 0.0]
        else:
            action = skill_spec["args"]["direction"].upper()
            action_onehot = self.ACTION_TO_ONEHOT[action]

        x = [
            float(r),
            float(c),
            self._bool01(z_t.get("has_key", False)),
            self._bool01(z_t.get("key_visible", False)),
            self._bool01(z_t.get("door_visible", False)),
            self._bool01(z_t.get("goal_visible", False)),
            float(door_open_known),
            float(door_open_value),
            self._bool01(walls["up"]),
            self._bool01(walls["down"]),
            self._bool01(walls["left"]),
            self._bool01(walls["right"]),
            float(key_rel[0]), float(key_rel[1]),
            float(door_rel[0]), float(door_rel[1]),
            float(goal_rel[0]), float(goal_rel[1]),
            *action_onehot,
        ]
        return torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _symbolic_noop(self, z_t):
        return {
            "agent_pos": tuple(z_t["agent_pos"]),
            "local_walls": {
                "up": z_t["local_walls"].get("up"),
                "down": z_t["local_walls"].get("down"),
                "left": z_t["local_walls"].get("left"),
                "right": z_t["local_walls"].get("right"),
            },
            "has_key": bool(z_t.get("has_key", False)),
            "key_visible": bool(z_t.get("key_visible", False)),
            "visible_key_pos": z_t.get("visible_key_pos", None),
            "door_visible": bool(z_t.get("door_visible", False)),
            "visible_door_pos": z_t.get("visible_door_pos", None),
            "visible_door_open": z_t.get("visible_door_open", None),
            "goal_visible": bool(z_t.get("goal_visible", False)),
            "visible_goal_pos": z_t.get("visible_goal_pos", None),
            "step_count": int(z_t.get("step_count", 0)),
            "view_radius": int(z_t.get("view_radius", 0)),
        }

    def predict_next_state(self, z_t, skill_spec):
        if skill_spec.get("skill") != "move":
            return self._symbolic_noop(z_t)

        x = self._z_action_to_tensor(z_t, skill_spec)
        with torch.no_grad():
            y = self.model(x).squeeze(0).cpu().tolist()

        pred_r = int(round(y[0]))
        pred_c = int(round(y[1]))
        next_agent_pos = (pred_r, pred_c)

        pred_has_key = self._round_bool(y[2])
        pred_key_visible = self._round_bool(y[3])
        pred_door_visible = self._round_bool(y[4])
        pred_goal_visible = self._round_bool(y[5])

        pred_door_open_known = self._round_bool(y[6])
        pred_door_open_value = self._round_bool(y[7])
        pred_visible_door_open = pred_door_open_value if pred_door_open_known else None

        pred_visible_key_pos = self._apply_rel(
            next_agent_pos, y[8], y[9], pred_key_visible)
        pred_visible_door_pos = self._apply_rel(
            next_agent_pos, y[10], y[11], pred_door_visible)
        pred_visible_goal_pos = self._apply_rel(
            next_agent_pos, y[12], y[13], pred_goal_visible)

        pred_local_walls = {
            "up": self._round_bool(y[14]),
            "down": self._round_bool(y[15]),
            "left": self._round_bool(y[16]),
            "right": self._round_bool(y[17]),
        }

        pred_view_gain_score = float(y[18])
        pred_new_key_seen = self._round_bool(y[19])
        pred_new_door_seen = self._round_bool(y[20])
        pred_new_goal_seen = self._round_bool(y[21])

        return {
            "agent_pos": next_agent_pos,
            "local_walls": pred_local_walls,
            "has_key": pred_has_key,
            "key_visible": pred_key_visible,
            "visible_key_pos": pred_visible_key_pos,
            "door_visible": pred_door_visible,
            "visible_door_pos": pred_visible_door_pos,
            "visible_door_open": pred_visible_door_open,
            "goal_visible": pred_goal_visible,
            "visible_goal_pos": pred_visible_goal_pos,
            "view_gain_score": pred_view_gain_score,
            "new_key_seen": pred_new_key_seen,
            "new_door_seen": pred_new_door_seen,
            "new_goal_seen": pred_new_goal_seen,
            "step_count": int(z_t.get("step_count", 0)) + 1,
            "view_radius": int(z_t.get("view_radius", 0)),
        }

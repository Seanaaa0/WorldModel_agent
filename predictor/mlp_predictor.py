import torch
import torch.nn as nn

from predictor.base_predictor import BasePredictor


class DynamicsMLP(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=128, output_dim=4):
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
    PO Dynamics Predictor v2

    Predict:
    - next agent_pos (r, c)
    - next goal_visible (0/1)
    - next goal_distance

    Input features:
    - current agent_pos (r, c)
    - current goal_visible
    - current local_walls (up, down, left, right)
    - action one-hot (UP, DOWN, LEFT, RIGHT)

    Still NOT predicting:
    - goal_pos
    - dx / dy
    - local_view / next local_walls
    """

    ACTION_TO_ONEHOT = {
        "UP":    [1.0, 0.0, 0.0, 0.0],
        "DOWN":  [0.0, 1.0, 0.0, 0.0],
        "LEFT":  [0.0, 0.0, 1.0, 0.0],
        "RIGHT": [0.0, 0.0, 0.0, 1.0],
    }

    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = DynamicsMLP().to(self.device)
        self.model.eval()

        if checkpoint_path is not None:
            ckpt = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()

    def _z_action_to_tensor(self, z_t, skill_spec):
        r, c = z_t["agent_pos"]
        goal_visible = float(bool(z_t.get("goal_visible", False)))

        walls = z_t["local_walls"]
        wall_up = float(bool(walls["up"]))
        wall_down = float(bool(walls["down"]))
        wall_left = float(bool(walls["left"]))
        wall_right = float(bool(walls["right"]))

        if skill_spec["skill"] != "move":
            action_onehot = [0.0, 0.0, 0.0, 0.0]
        else:
            action = skill_spec["args"]["direction"].upper()
            action_onehot = self.ACTION_TO_ONEHOT[action]

        x = [
            float(r),
            float(c),
            goal_visible,
            wall_up,
            wall_down,
            wall_left,
            wall_right,
            *action_onehot,
        ]
        return torch.tensor(
            x,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

    def predict_next_state(self, z_t, skill_spec):
        # Keep non-move skills as symbolic no-op
        if skill_spec["skill"] != "move":
            return {
                "agent_pos": z_t["agent_pos"],
                "goal_visible": z_t.get("goal_visible", False),
                "goal_pos": z_t.get("goal_pos", None),
                "dx": z_t.get("dx", None),
                "dy": z_t.get("dy", None),
                "goal_distance": z_t.get("goal_distance", None),
                "local_walls": {
                    "up": None,
                    "down": None,
                    "left": None,
                    "right": None,
                },
                "step_count": z_t["step_count"],
            }

        x = self._z_action_to_tensor(z_t, skill_spec)

        with torch.no_grad():
            y = self.model(x).squeeze(0).cpu().tolist()

        pred_r = int(round(y[0]))
        pred_c = int(round(y[1]))
        pred_goal_visible = bool(round(y[2]))
        pred_goal_distance = float(y[3])

        return {
            "agent_pos": (pred_r, pred_c),
            "goal_visible": pred_goal_visible,
            "goal_pos": None,
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

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from predictor.mlp_predictor import DynamicsMLP

ACTION_TO_ONEHOT = {
    "UP": [1.0, 0.0, 0.0, 0.0],
    "DOWN": [0.0, 1.0, 0.0, 0.0],
    "LEFT": [0.0, 0.0, 1.0, 0.0],
    "RIGHT": [0.0, 0.0, 0.0, 1.0],
}


def _bool01(x):
    return 1.0 if bool(x) else 0.0


def _door_open_pair(value):
    if value is None:
        return 0.0, 0.0
    return 1.0, (1.0 if bool(value) else 0.0)


def _relative_pos(agent_pos, target_pos):
    if target_pos is None:
        return 0.0, 0.0
    return float(target_pos[0] - agent_pos[0]), float(target_pos[1] - agent_pos[1])


def _wall01(v):
    return 1.0 if bool(v) else 0.0


class PredictorDataset(Dataset):
    """
    V2-aligned one-step dynamics dataset.

    Input x (22 dims):
    - agent_pos: r, c
    - has_key
    - key_visible, door_visible, goal_visible
    - door_open_known, door_open_value
    - local_walls up/down/left/right
    - rel_key dr/dc
    - rel_door dr/dc
    - rel_goal dr/dc
    - action onehot (4)

    Target y (22 dims):
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

    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        z_t = item["z_t"]
        z_tp1 = item["z_tp1"]
        action = item["action"]

        agent_pos = z_t["agent_pos"]
        key_rel = _relative_pos(agent_pos, z_t.get("visible_key_pos"))
        door_rel = _relative_pos(agent_pos, z_t.get("visible_door_pos"))
        goal_rel = _relative_pos(agent_pos, z_t.get("visible_goal_pos"))
        door_open_known, door_open_value = _door_open_pair(
            z_t.get("visible_door_open"))

        x = [
            float(agent_pos[0]),
            float(agent_pos[1]),
            _bool01(z_t.get("has_key", False)),
            _bool01(z_t.get("key_visible", False)),
            _bool01(z_t.get("door_visible", False)),
            _bool01(z_t.get("goal_visible", False)),
            door_open_known,
            door_open_value,
            _bool01(z_t["local_walls"]["up"]),
            _bool01(z_t["local_walls"]["down"]),
            _bool01(z_t["local_walls"]["left"]),
            _bool01(z_t["local_walls"]["right"]),
            key_rel[0], key_rel[1],
            door_rel[0], door_rel[1],
            goal_rel[0], goal_rel[1],
            *ACTION_TO_ONEHOT[action],
        ]

        next_agent_pos = z_tp1["agent_pos"]
        next_key_rel = _relative_pos(
            next_agent_pos, z_tp1.get("visible_key_pos"))
        next_door_rel = _relative_pos(
            next_agent_pos, z_tp1.get("visible_door_pos"))
        next_goal_rel = _relative_pos(
            next_agent_pos, z_tp1.get("visible_goal_pos"))
        next_door_open_known, next_door_open_value = _door_open_pair(
            z_tp1.get("visible_door_open"))

        aux = item.get("aux_targets", {})

        y = [
            float(next_agent_pos[0]),
            float(next_agent_pos[1]),
            _bool01(z_tp1.get("has_key", False)),
            _bool01(z_tp1.get("key_visible", False)),
            _bool01(z_tp1.get("door_visible", False)),
            _bool01(z_tp1.get("goal_visible", False)),
            next_door_open_known,
            next_door_open_value,
            next_key_rel[0], next_key_rel[1],
            next_door_rel[0], next_door_rel[1],
            next_goal_rel[0], next_goal_rel[1],
            _wall01(z_tp1["local_walls"]["up"]),
            _wall01(z_tp1["local_walls"]["down"]),
            _wall01(z_tp1["local_walls"]["left"]),
            _wall01(z_tp1["local_walls"]["right"]),
            float(aux.get("view_gain_score", 0.0)),
            _bool01(aux.get("new_key_seen", False)),
            _bool01(aux.get("new_door_seen", False)),
            _bool01(aux.get("new_goal_seen", False)),
        ]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train(
    dataset_path="outputs/predictor_dataset_v7_mixed.jsonl",
    checkpoint_path="predictor/checkpoints/mlp_predictor_v7.pt",
    epochs=20,
    batch_size=128,
    lr=1e-3,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PredictorDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = 22
    output_dim = 22
    model = DynamicsMLP(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"[epoch {epoch:03d}] loss = {avg_loss:.6f}")

    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "dataset_path": dataset_path,
            "target_schema": {
                "x": [
                    "agent_r", "agent_c", "has_key",
                    "key_visible", "door_visible", "goal_visible",
                    "door_open_known", "door_open_value",
                    "wall_up", "wall_down", "wall_left", "wall_right",
                    "rel_key_dr", "rel_key_dc",
                    "rel_door_dr", "rel_door_dc",
                    "rel_goal_dr", "rel_goal_dc",
                    "action_up", "action_down", "action_left", "action_right",
                ],
                "y": [
                    "next_agent_r", "next_agent_c", "next_has_key",
                    "next_key_visible", "next_door_visible", "next_goal_visible",
                    "next_door_open_known", "next_door_open_value",
                    "next_rel_key_dr", "next_rel_key_dc",
                    "next_rel_door_dr", "next_rel_door_dc",
                    "next_rel_goal_dr", "next_rel_goal_dc",
                    "next_wall_up", "next_wall_down", "next_wall_left", "next_wall_right",
                    "view_gain_score",
                    "new_key_seen", "new_door_seen", "new_goal_seen",
                ],
            },
        },
        ckpt_path,
    )

    print(f"checkpoint saved to: {ckpt_path}")
    print(f"input_dim = {input_dim}, output_dim = {output_dim}")


if __name__ == "__main__":
    train()

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from predictor.mlp_predictor import DynamicsMLP


ACTION_TO_ONEHOT = {
    "UP":    [1.0, 0.0, 0.0, 0.0],
    "DOWN":  [0.0, 1.0, 0.0, 0.0],
    "LEFT":  [0.0, 0.0, 1.0, 0.0],
    "RIGHT": [0.0, 0.0, 0.0, 1.0],
}


class PredictorDataset(Dataset):
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

        r, c = z_t["agent_pos"]
        x = [
            float(r),
            float(c),
            float(bool(z_t["goal_visible"])),
            float(bool(z_t["local_walls"]["up"])),
            float(bool(z_t["local_walls"]["down"])),
            float(bool(z_t["local_walls"]["left"])),
            float(bool(z_t["local_walls"]["right"])),
            *ACTION_TO_ONEHOT[action],
        ]

        nr, nc = z_tp1["agent_pos"]

        goal_distance_next = z_tp1.get("goal_distance", None)
        if goal_distance_next is None:
            goal_distance_next = 0.0

        y = [
            float(nr),
            float(nc),
            float(bool(z_tp1["goal_visible"])),
            float(goal_distance_next),
        ]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train(
    dataset_path="outputs/predictor_dataset_po_v2.jsonl",
    checkpoint_path="predictor/checkpoints/jepa_lite_mlp_po_v2.pt",
    epochs=20,
    batch_size=128,
    lr=1e-3,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PredictorDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DynamicsMLP(input_dim=11, output_dim=4).to(device)
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
            "input_dim": 11,
            "output_dim": 4,
        },
        ckpt_path,
    )

    print(f"checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    train()

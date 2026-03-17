import json
import random
from pathlib import Path

from env.maze_env import MazeEnv
from encoder.state_encoder import StateEncoder


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]


def z_to_record(z):
    return {
        "agent_pos": list(z["agent_pos"]),
        "goal_visible": bool(z["goal_visible"]),
        "goal_distance": z.get("goal_distance", None),
        "local_walls": z["local_walls"],
        "step_count": z["step_count"],
        "view_radius": z["view_radius"],
    }


def collect_dataset(
    output_path="outputs/predictor_dataset_po_v2.jsonl",
    num_episodes=300,
    max_steps_per_episode=80,
    maze_size=8,
    wall_prob=0.2,
    seed=42,
    view_radius=1,
):
    random.seed(seed)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = StateEncoder()
    total_samples = 0

    with out_path.open("w", encoding="utf-8") as f:
        for ep in range(num_episodes):
            env = MazeEnv(
                size=maze_size,
                wall_prob=wall_prob,
                seed=seed + ep,
                view_radius=view_radius,
            )
            obs_t = env.reset()
            done = False

            for _ in range(max_steps_per_episode):
                if done:
                    break

                z_t = encoder.encode(obs_t)

                action = random.choice(ACTIONS)

                obs_tp1, done, info = env.step(action)
                z_tp1 = encoder.encode(obs_tp1)

                record = {
                    "z_t": z_to_record(z_t),
                    "action": action,
                    "z_tp1": z_to_record(z_tp1),
                    "info": {
                        "move_success": info["move_success"],
                        "hit_wall": info["hit_wall"],
                        "out_of_bounds": info["out_of_bounds"],
                        "goal_reached": info["goal_reached"],
                    },
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_samples += 1

                obs_t = obs_tp1

    print(f"dataset saved to: {out_path}")
    print(f"total_samples = {total_samples}")


if __name__ == "__main__":
    collect_dataset()

import json
import random
from pathlib import Path
from typing import Optional

from env.maze_env import MazeEnv
from encoder.state_encoder import StateEncoder
from memory.world_memory import WorldMemory
from planner.rule_planner import RulePlanner
from skills.skill_executor import SkillExecutor

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]


def _pos_to_list(pos):
    if pos is None:
        return None
    return [int(pos[0]), int(pos[1])]


def _count_unknown_local_walls(z: dict) -> int:
    walls = z["local_walls"]
    cnt = 0
    for k in ("up", "down", "left", "right"):
        if walls.get(k) is None:
            cnt += 1
    return cnt


def _derive_aux_targets(z_t: dict, z_tp1: dict) -> dict:
    prev_unknown = _count_unknown_local_walls(z_t)
    next_unknown = _count_unknown_local_walls(z_tp1)

    view_gain_score = float(prev_unknown - next_unknown)

    new_key_seen = (not bool(z_t.get("key_visible", False))
                    ) and bool(z_tp1.get("key_visible", False))
    new_door_seen = (not bool(z_t.get("door_visible", False))
                     ) and bool(z_tp1.get("door_visible", False))
    new_goal_seen = (not bool(z_t.get("goal_visible", False))
                     ) and bool(z_tp1.get("goal_visible", False))

    return {
        "view_gain_score": view_gain_score,
        "new_key_seen": bool(new_key_seen),
        "new_door_seen": bool(new_door_seen),
        "new_goal_seen": bool(new_goal_seen),
    }


def z_to_record(z: dict) -> dict:
    return {
        "agent_pos": _pos_to_list(z["agent_pos"]),
        "local_walls": {
            "up": bool(z["local_walls"]["up"]),
            "down": bool(z["local_walls"]["down"]),
            "left": bool(z["local_walls"]["left"]),
            "right": bool(z["local_walls"]["right"]),
        },
        "has_key": bool(z.get("has_key", False)),
        "key_visible": bool(z.get("key_visible", False)),
        "visible_key_pos": _pos_to_list(z.get("visible_key_pos", None)),
        "door_visible": bool(z.get("door_visible", False)),
        "visible_door_pos": _pos_to_list(z.get("visible_door_pos", None)),
        "visible_door_open": (
            None
            if z.get("visible_door_open", None) is None
            else bool(z.get("visible_door_open"))
        ),
        "goal_visible": bool(z.get("goal_visible", False)),
        "visible_goal_pos": _pos_to_list(z.get("visible_goal_pos", None)),
        "step_count": int(z["step_count"]),
        "view_radius": int(z["view_radius"]),
    }


class DatasetCollector:
    def __init__(
        self,
        policy_mode: str = "mixed",
        mixed_rule_prob: float = 0.7,
        seed: int = 42,
    ) -> None:
        if policy_mode not in {"rule", "random", "mixed"}:
            raise ValueError("policy_mode must be one of: rule, random, mixed")
        if not (0.0 <= mixed_rule_prob <= 1.0):
            raise ValueError("mixed_rule_prob must be in [0, 1]")

        self.policy_mode = policy_mode
        self.mixed_rule_prob = mixed_rule_prob
        self.rng = random.Random(seed)

        self.encoder = StateEncoder()
        self.rule_planner = RulePlanner()
        self.executor = SkillExecutor()

    def _build_rule_move(
        self,
        z_t: dict,
        memory: WorldMemory,
        last_info: Optional[dict],
        replan: bool,
    ) -> Optional[dict]:
        planner_context = memory.get_planner_context(
            agent_pos=z_t["agent_pos"])
        planner_context["memory_obj"] = memory

        skill = self.rule_planner.choose_skill(
            z_t=z_t,
            memory_summary=planner_context["memory_summary"],
            memory_patch=planner_context["memory_patch"],
            frontier_candidates=planner_context["frontier_candidates"],
            loop_hints=planner_context["loop_hints"],
            replan=replan,
            last_info=last_info,
            planner_context=planner_context,
        )

        if skill.get("skill") == "move":
            return skill
        return None

    def _sample_skill(
        self,
        z_t: dict,
        memory: WorldMemory,
        last_info: Optional[dict],
        replan: bool,
    ):
        if self.policy_mode == "random":
            source = "random"
            return source, {"skill": "move", "args": {"direction": self.rng.choice(ACTIONS)}}

        if self.policy_mode == "rule":
            skill = self._build_rule_move(z_t, memory, last_info, replan)
            if skill is None:
                source = "rule_fallback_random"
                return source, {"skill": "move", "args": {"direction": self.rng.choice(ACTIONS)}}
            return "rule", skill

        # mixed
        use_rule = self.rng.random() < self.mixed_rule_prob
        if use_rule:
            skill = self._build_rule_move(z_t, memory, last_info, replan)
            if skill is not None:
                return "rule", skill
        return "random", {"skill": "move", "args": {"direction": self.rng.choice(ACTIONS)}}


def collect_dataset(
    output_path="outputs/predictor_dataset_v7_mixed.jsonl",
    num_episodes=300,
    max_steps_per_episode=300,
    maze_size=20,
    wall_prob=0.12,
    seed=42,
    view_radius=3,
    policy_mode="mixed",
    mixed_rule_prob=0.7,
):
    random.seed(seed)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    collector = DatasetCollector(
        policy_mode=policy_mode,
        mixed_rule_prob=mixed_rule_prob,
        seed=seed,
    )

    total_samples = 0
    source_counts = {"rule": 0, "random": 0, "rule_fallback_random": 0}

    with out_path.open("w", encoding="utf-8") as f:
        for ep in range(num_episodes):
            env = MazeEnv(
                size=maze_size,
                wall_prob=wall_prob,
                seed=seed + ep,
                max_steps=max_steps_per_episode,
                view_radius=view_radius,
            )
            memory = WorldMemory()
            obs_t = env.reset()
            z_t = collector.encoder.encode(obs_t)
            memory.update(z_t, info=None)

            done = False
            last_info = None
            replan = False

            for step_idx in range(max_steps_per_episode):
                if done:
                    break

                source, skill = collector._sample_skill(
                    z_t=z_t,
                    memory=memory,
                    last_info=last_info,
                    replan=replan,
                )
                source_counts[source] = source_counts.get(source, 0) + 1

                exec_result = collector.executor.execute(env, skill)
                obs_tp1 = exec_result["obs"]
                done = bool(exec_result["done"])
                info = exec_result["info"]
                z_tp1 = collector.encoder.encode(obs_tp1)
                aux_targets = _derive_aux_targets(z_t, z_tp1)

                record = {
                    "episode": ep,
                    "step": step_idx,
                    "source": source,
                    "z_t": z_to_record(z_t),
                    "action": skill["args"]["direction"],
                    "z_tp1": z_to_record(z_tp1),
                    "aux_targets": aux_targets,
                    "info": {
                        "move_success": bool(info.get("move_success", False)),
                        "hit_wall": bool(info.get("hit_wall", False)),
                        "out_of_bounds": bool(info.get("out_of_bounds", False)),
                        "blocked_by_locked_door": bool(info.get("blocked_by_locked_door", False)),
                        "picked_key": bool(info.get("picked_key", False)),
                        "opened_door": bool(info.get("opened_door", False)),
                        "goal_reached": bool(info.get("goal_reached", False)),
                        "has_key": bool(info.get("has_key", False)),
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_samples += 1

                memory.update(z_tp1, info=info)
                replan = bool(
                    info.get("hit_wall", False)
                    or info.get("out_of_bounds", False)
                    or info.get("blocked_by_locked_door", False)
                )
                last_info = info
                z_t = z_tp1

    print(f"dataset saved to: {out_path}")
    print(f"total_samples = {total_samples}")
    print(f"source_counts = {source_counts}")
    print(
        "config = "
        f"episodes={num_episodes}, max_steps={max_steps_per_episode}, "
        f"size={maze_size}, wall_prob={wall_prob}, view_radius={view_radius}, "
        f"policy_mode={policy_mode}, mixed_rule_prob={mixed_rule_prob}"
    )


if __name__ == "__main__":
    collect_dataset()

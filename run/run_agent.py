from env.maze_env import MazeEnv
from agent.agent_loop import AgentLoop
from planner.rule_planner import RulePlanner
from planner.predictive_rule_planner import PredictiveRulePlanner
from planner.llm_planner import LLMPlanner

import random
import os
import sys
import numpy as np


# =========================
# Experiment config
# =========================
PLANNER_MODE = "llm_slow"   # "rule" / "fast_predictive_legacy" / "llm_slow"

MAZE_SIZE = 15
WALL_PROB = 0.12
VIEW_RADIUS = 3
MAX_STEPS = 200
SLEEP_TIME = 0.0001

MODEL_PATH = r"D:\models\qwen\Qwen2.5-3B-Instruct"

# controlled seeds
SEEDS = list(range(20))

# =========================
# Logging / debug config
# =========================
DEBUG_VERBOSE = False
DEBUG_SEEDS = {0, 1, 2, 3}

# benchmark mode example:
# DEBUG_VERBOSE = False
# DEBUG_SEEDS = set()

# =========================
# Output folder config
# =========================
OUTPUT_SUBDIR = "llm_slow_v5a_15x15_2"


# =========================
# Tee logger
# =========================
class Tee:
    def __init__(self, file_path):
        self.file = open(file_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


# =========================
# Seed control
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# =========================
# Planner builder
# =========================
def build_planner():
    if PLANNER_MODE == "llm_slow":
        print("[Run] Using LLMPlanner as SLOW planner")
        return LLMPlanner(
            model_path=MODEL_PATH,
            max_new_tokens=24,
            temperature=0.0,
            do_sample=False,
            verbose=False,
        )

    if PLANNER_MODE == "fast_predictive_legacy":
        print("[Run] Using PredictiveRulePlanner")
        return PredictiveRulePlanner(
            predictor_checkpoint="predictor/checkpoints/jepa_lite_mlp_po_v2.pt",
            use_predictor=True,
            verbose=False,
        )

    print("[Run] Using RulePlanner")
    return RulePlanner()


# =========================
# Main
# =========================
def main() -> None:
    output_dir = os.path.join("outputs", OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    # build once, reuse across seeds
    planner = build_planner()

    for seed in SEEDS:
        set_seed(seed)

        log_path = os.path.join(output_dir, f"seed_{seed}.txt")
        tee = Tee(log_path)
        sys.stdout = tee

        episode_verbose = DEBUG_VERBOSE and (seed in DEBUG_SEEDS)

        try:
            print("=" * 40)
            print(f"[Run] SEED = {seed}")
            print(f"[Run] episode_verbose = {episode_verbose}")
            print(
                f"[Run] config: planner_mode={PLANNER_MODE}, "
                f"size={MAZE_SIZE}, wall_prob={WALL_PROB}, "
                f"view_radius={VIEW_RADIUS}, max_steps={MAX_STEPS}"
            )

            env = MazeEnv(
                size=MAZE_SIZE,
                wall_prob=WALL_PROB,
                seed=seed,
                max_steps=MAX_STEPS,
                view_radius=VIEW_RADIUS,
            )

            agent = AgentLoop(
                env=env,
                planner=planner,
                sleep_time=SLEEP_TIME,
                verbose=episode_verbose,
            )

            success, steps = agent.run(max_steps=MAX_STEPS)

            print(f"RESULT: {'SUCCESS' if success else 'FAIL'}")
            print(f"STEPS: {steps}")
            print(f"SEED: {seed}")

            all_results.append({
                "seed": seed,
                "success": success,
                "steps": steps,
            })

        finally:
            sys.stdout = tee.stdout
            tee.close()

    # =========================
    # Summary
    # =========================
    total = len(all_results)
    success_count = sum(1 for r in all_results if r["success"])
    fail_count = total - success_count

    success_steps = [r["steps"] for r in all_results if r["success"]]

    print("\n=================================")
    print("Experiment Summary")
    print("=================================")
    print(f"DIR:  outputs/{OUTPUT_SUBDIR}")
    print(f"Total episodes: {total}")
    print(f"Success: {success_count}")
    print(f"Fail: {fail_count}")
    print(f"Success rate: {success_count / total:.3f}")

    if success_steps:
        print(
            f"Average steps (success only): {sum(success_steps) / len(success_steps):.2f}")
        print(f"Min steps: {min(success_steps)}")
        print(f"Max steps: {max(success_steps)}")


if __name__ == "__main__":
    main()

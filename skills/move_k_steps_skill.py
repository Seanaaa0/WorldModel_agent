from skills.base_skill import BaseSkill


class MoveKStepsSkill(BaseSkill):
    """
    Move in one direction for up to k steps.
    Stops early if blocked or episode ends.
    """

    VALID_DIRECTIONS = {"UP", "DOWN", "LEFT", "RIGHT"}

    def execute(self, env, **kwargs) -> dict:
        direction = kwargs.get("direction", "").upper()
        k = int(kwargs.get("k", 1))

        if direction not in self.VALID_DIRECTIONS:
            raise ValueError(f"Invalid direction: {direction}")
        if k < 1:
            raise ValueError("k must be >= 1")

        last_obs = None
        last_done = False
        last_info = None
        actual_steps = 0

        for _ in range(k):
            obs, done, info = env.step(direction)
            last_obs = obs
            last_done = done
            last_info = info
            actual_steps += 1

            if done or info.get("hit_wall") or info.get("out_of_bounds"):
                break

        return {
            "skill_name": "move_k_steps",
            "skill_args": {"direction": direction, "k": k},
            "obs": last_obs,
            "done": last_done,
            "info": {
                **(last_info or {}),
                "macro_skill": "move_k_steps",
                "requested_k": k,
                "actual_steps": actual_steps,
            },
        }

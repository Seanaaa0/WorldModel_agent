from skills.base_skill import BaseSkill


class MoveUntilBlockedSkill(BaseSkill):
    """
    Move repeatedly in one direction until blocked or max_k reached.
    """

    VALID_DIRECTIONS = {"UP", "DOWN", "LEFT", "RIGHT"}

    def execute(self, env, **kwargs) -> dict:
        direction = kwargs.get("direction", "").upper()
        max_k = int(kwargs.get("max_k", 4))

        if direction not in self.VALID_DIRECTIONS:
            raise ValueError(f"Invalid direction: {direction}")
        if max_k < 1:
            raise ValueError("max_k must be >= 1")

        last_obs = None
        last_done = False
        last_info = None
        actual_steps = 0

        for _ in range(max_k):
            obs, done, info = env.step(direction)
            last_obs = obs
            last_done = done
            last_info = info

            if info.get("hit_wall") or info.get("out_of_bounds"):
                break

            actual_steps += 1

            if done:
                break

        return {
            "skill_name": "move_until_blocked",
            "skill_args": {"direction": direction, "max_k": max_k},
            "obs": last_obs,
            "done": last_done,
            "info": {
                **(last_info or {}),
                "macro_skill": "move_until_blocked",
                "requested_max_k": max_k,
                "actual_steps": actual_steps,
            },
        }

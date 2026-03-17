from skills.base_skill import BaseSkill


class MoveSkill(BaseSkill):
    """
    Primitive movement skill.
    """

    VALID_DIRECTIONS = {"UP", "DOWN", "LEFT", "RIGHT"}

    def execute(self, env, **kwargs) -> dict:
        direction = kwargs.get("direction", "").upper()

        if direction not in self.VALID_DIRECTIONS:
            raise ValueError(
                f"Invalid move direction: {direction}. "
                f"Valid directions: {sorted(self.VALID_DIRECTIONS)}"
            )

        obs, done, info = env.step(direction)

        return {
            "skill_name": "move",
            "skill_args": {"direction": direction},
            "obs": obs,
            "done": done,
            "info": info,
        }

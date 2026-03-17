from skills.base_skill import BaseSkill


class EscapeLoopSkill(BaseSkill):
    """
    Simple loop escape skill.
    Chooses the first safe open direction from a fixed priority.
    """

    PRIORITY = ["UP", "RIGHT", "DOWN", "LEFT"]

    def execute(self, env, **kwargs) -> dict:
        obs = env.get_obs()
        walls = obs["walls"]

        chosen = None
        for d in self.PRIORITY:
            if not walls.get(d.lower(), False):
                chosen = d
                break

        if chosen is None:
            return {
                "skill_name": "escape_loop",
                "skill_args": {},
                "obs": obs,
                "done": False,
                "info": {
                    "scan": True,
                    "move_success": False,
                    "hit_wall": False,
                    "out_of_bounds": False,
                    "goal_reached": False,
                    "max_steps_reached": False,
                    "escape_loop_failed": True,
                },
            }

        obs2, done, info = env.step(chosen)

        return {
            "skill_name": "escape_loop",
            "skill_args": {},
            "obs": obs2,
            "done": done,
            "info": {
                **info,
                "macro_skill": "escape_loop",
                "escape_direction": chosen,
            },
        }

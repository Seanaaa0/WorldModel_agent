from skills.base_skill import BaseSkill


class ScanSkill(BaseSkill):
    """
    Observation skill.
    Returns current observation without moving the agent.
    """

    def execute(self, env, **kwargs) -> dict:
        obs = env.get_obs()

        return {
            "skill_name": "scan",
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
            },
        }

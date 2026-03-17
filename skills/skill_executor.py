from skills.move_skill import MoveSkill
from skills.scan_skill import ScanSkill
from skills.move_k_steps_skill import MoveKStepsSkill
from skills.move_until_blocked_skill import MoveUntilBlockedSkill
from skills.escape_loop_skill import EscapeLoopSkill


class SkillExecutor:
    """
    Central dispatcher for agent skills.
    """

    def __init__(self) -> None:
        self.skills = {
            "move": MoveSkill(),
            "scan": ScanSkill(),
            "move_k_steps": MoveKStepsSkill(),
            "move_until_blocked": MoveUntilBlockedSkill(),
            "escape_loop": EscapeLoopSkill(),
        }

    def execute(self, env, skill_spec: dict) -> dict:
        """
        skill_spec example:
        {
            "skill": "move",
            "args": {"direction": "UP"}
        }
        """
        skill_name = skill_spec.get("skill")
        skill_args = skill_spec.get("args", {})

        if skill_name not in self.skills:
            raise ValueError(
                f"Unknown skill: {skill_name}. "
                f"Available skills: {list(self.skills.keys())}"
            )

        skill = self.skills[skill_name]
        return skill.execute(env, **skill_args)

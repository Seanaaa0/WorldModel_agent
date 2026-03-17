from abc import ABC, abstractmethod


class BaseSkill(ABC):
    """
    Base class for all executable skills.
    """

    @abstractmethod
    def execute(self, env, **kwargs) -> dict:
        """
        Execute the skill on the environment.

        Returns a standardized result dict.
        """
        raise NotImplementedError

from abc import ABC, abstractmethod
from typing import Optional


class BasePlanner(ABC):
    @abstractmethod
    def choose_skill(
        self,
        z_t: dict,
        memory_summary: Optional[dict] = None,
        replan: bool = False,
        last_info: dict | None = None,
    ) -> dict:
        raise NotImplementedError

"""SkillMemory abstract base."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Skill:
    id: str
    type: str
    source: str
    source_task_id: str
    description: str = ""
    source_subtask_idx: int | None = None
    extra: dict = field(default_factory=dict)


class SkillMemory(ABC):

    @abstractmethod
    def retrieve(self, query: str, extra_queries: list[str] | None = None) -> str:
        """Prompt-ready text of the top-k relevant skills."""

    @abstractmethod
    def apply_skill_delta(self, delta_M: dict, world=None) -> list[str]:
        """Append (or merge) the skill(s) in `delta_M` into memory; returns one human-readable message per touched entry."""

    def get_brief(self) -> str:
        """One-line-per-entry summary fed to the induction prompt."""
        return ""

    def get_stats(self) -> dict:
        return {}

    def remove_by_name(self, name: str) -> None:
        return None

    def get_active_skill_codes(self) -> list[str]:
        return []

    def get_active_skill_names(self) -> list[str]:
        return []

    @abstractmethod
    def snapshot(self) -> dict:
        """Full state, JSON-serialisable, for per-task logging."""

    @abstractmethod
    def load_snapshot(self, snapshot: dict) -> None:
        """Restore from a snapshot produced by `snapshot()`."""

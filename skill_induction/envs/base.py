"""World abstract base: the environment contract.

Implementations must:
- expose `task` (with `.instruction`, `.task_id`, optional `.group_key`)
- be used as a context manager (`__enter__` / `__exit__`)
- return observations from `execute(payload)` as strings, with the literal substring "Execution failed" on errors so agents can detect them
- expose `evaluate_run() -> (success, extras)` for logging only; the agent and memory must not consult this (fairness clause).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class World(ABC):
    task: object

    def __enter__(self) -> "World":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    @abstractmethod
    def execute(self, payload: str) -> str:
        ...

    @abstractmethod
    def evaluate_run(self) -> tuple[bool, dict]:
        ...

    def calculate_difficulty(self) -> int:
        """Optional task-complexity score, 0 by default; AppWorld overrides it with the gold solution-code line count from `ground_truth/metadata.json`."""
        return 0

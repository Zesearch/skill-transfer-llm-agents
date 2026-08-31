"""Linear (task-level) agent backbone.

For the "Linear, no memory" condition this is exactly the ReAct base class with no memory plugged in. Memory-equipped task-level variants (AWM / ASI) subclass `LinearAgent`, overriding `_build_initial_history` to inject retrieved skills and `_on_task_end` to write back.

This thin subclass exists so table-row labels are unambiguous: "linear_no_memory", "linear_awm", ...
"""

from __future__ import annotations

from skill_induction.agents.base import ReactAgent


class LinearAgent(ReactAgent):
    """Memory-free ReAct. Use as-is for the "Linear (no memory)" row.

    When `memory` is provided, the subclass should override `_build_initial_history` to call `memory.retrieve(...)` and `_on_task_end` to call `memory.add(...)`. The bare agent does neither.
    """
    pass

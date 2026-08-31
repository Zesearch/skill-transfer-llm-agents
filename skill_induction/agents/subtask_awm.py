"""SubtaskAWM: subtask agent with doc-memory skill induction.

After every `end_subgoal` the executor's reflected summary is followed by an induction call that appends one workflow note to `AWMMemory`. At the start of each subgoal, the top relevant workflow is retrieved via embedding similarity and injected into the executor's local history.
"""

from __future__ import annotations

from skill_induction.agents.subtask import SubtaskAgent
from skill_induction.memory._delta import parse_skill_update_json
from skill_induction.prompts.awm_induction import (
    create_skill_update_prompt,
    induction_demo_messages,
    skill_induction_tool,
)


class SubtaskAWM(SubtaskAgent):

    _memory_type = "doc"

    def _retrieve_for_subgoal(self, world, task, subgoal: str) -> str:
        if self.memory is None:
            return ""
        task_instr = getattr(task, "instruction", "") or ""
        return self.memory.retrieve(
            task_instr,
            extra_queries=[subgoal] if subgoal else None,
        ) or ""

    def _on_subtask_end(self, world, executor_history, subtask_idx):
        if self.memory is None:
            return
        # Splice the induction demo before the real induction prompt.
        demo = induction_demo_messages(
            self.benchmark, variant="subtask", level=self._induction_level)
        prompt = create_skill_update_prompt(
            memory_brief=self.memory.get_brief(),
            benchmark=self.benchmark,
            granularity="atomic",  # subtask-level: reusable primitives
            level=self._induction_level,
        )
        branch = list(executor_history) + list(demo) + [
            {"role": "user", "content": prompt}
        ]
        tool = skill_induction_tool(level=self._induction_level)
        from skill_induction.agents.base import _retry_until_tool_call
        resp = _retry_until_tool_call(
            lambda: self.model.chat(
                branch, tools=[tool],
                tool_choice="skill_induction", max_tokens=8192,
            ),
            tag="SubtaskAWM induction",
            on_attempt=self._account_tokens,
            require_tool="skill_induction",
            ledger=getattr(self.model, "ledger", None),
        )
        if resp is None or not resp.tool_calls:
            return
        delta_M = parse_skill_update_json(
            resp.tool_calls[0].arguments or {})
        if delta_M:
            self.memory.apply_skill_delta(delta_M, world=world)

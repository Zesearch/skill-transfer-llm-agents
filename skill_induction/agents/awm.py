"""LinearAWM: vanilla ReAct plus doc-memory skill induction at task end.

Cross-task retrieval runs at task start: the top workflow note from `AWMMemory` is appended as a user turn before the ReAct loop begins. At task end, the induction prompt is appended to the finished trajectory and the model is forced to emit a single `skill_induction` tool call, whose arguments go to `memory.apply_skill_delta`.

The induction demo (a user / assistant tool_call / tool ack triple) is spliced into the initial history so the model sees one full example of the induction step before producing its own.
"""

from __future__ import annotations

from skill_induction.agents.linear import LinearAgent
from skill_induction.memory._delta import parse_skill_update_json
from skill_induction.prompts.awm_induction import (
    create_skill_update_prompt,
    induction_demo_messages,
    skill_induction_tool,
)


class LinearAWM(LinearAgent):

    def _build_initial_history(self, task) -> list[dict]:
        history = super()._build_initial_history(task)
        # Consume-only arm (--subtask-skills): induction never happens, so drop the induction demo.
        if not self._subtask_skills:
            demo = induction_demo_messages(
                self.benchmark, variant="linear", level=self._induction_level)
            # Splice the induction demo before the trailer user message, keeping the trailer adjacent to the task it introduces.
            if (len(history) >= 2
                    and history[-1].get("role") == "user"
                    and history[-2].get("role") == "user"):
                user_task = history[-1]
                trailer = history[-2]
                history = history[:-2] + list(demo) + [trailer, user_task]
            elif history and history[-1].get("role") == "user":
                user_task = history[-1]
                history = history[:-1] + list(demo) + [user_task]
        if self.memory is None:
            return history
        retrieved = self.memory.retrieve(task.instruction)
        if retrieved:
            history.append({"role": "user", "content": retrieved})
            self._last_retrieved_text = retrieved
        return history

    def _on_task_end(self, world, history, completed: bool) -> None:
        # Induce regardless of `completed`; gating on it would skip induction after a forced finish.
        if self.memory is None or self._subtask_skills:
            return
        prompt = create_skill_update_prompt(
            memory_brief=self.memory.get_brief(),
            benchmark=self.benchmark,
            granularity="coarse",  # task-level: one whole-task workflow
            level=self._induction_level,
        )
        history.append({"role": "user", "content": prompt})
        tool = skill_induction_tool(level=self._induction_level)
        from skill_induction.agents.base import _retry_until_tool_call
        resp = _retry_until_tool_call(
            lambda: self.model.chat(
                history, tools=[tool],
                tool_choice="skill_induction", max_tokens=8192,
            ),
            tag="LinearAWM induction",
            require_tool="skill_induction",
            ledger=getattr(self.model, "ledger", None),
        )
        if resp is None or not resp.tool_calls:
            return
        delta_M = parse_skill_update_json(resp.tool_calls[0].arguments or {})
        if delta_M:
            self.memory.apply_skill_delta(delta_M, world=world)

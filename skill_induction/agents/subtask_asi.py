"""SubtaskASI: subtask agent with code-memory skill induction.

At the start of each subgoal the top relevant helper from `ASIMemory` is retrieved, exec()'d into the REPL, and injected into the executor's history. After every `end_subgoal` the executor's reflected summary is followed by an induction call that appends one or more helpers to memory. `check_and_remove_errored` drops helpers whose calls produced tracebacks during the just-finished subgoal.
"""

from __future__ import annotations

from skill_induction.agents.asi import (
    _PYTHON_REPL_BENCHES, _SIDECAR_PY_BENCHES,
    _split_def_blocks, preload_active_skills)
from skill_induction.agents.subtask import SubtaskAgent
from skill_induction.memory._delta import parse_code_skill_update_json
from skill_induction.prompts.asi_induction import (
    create_skill_update_prompt,
    induction_demo_messages,
    skill_induction_tool,
)


class SubtaskASI(SubtaskAgent):

    _memory_type = "code"
    _agent_kind = "asi"

    def _on_executor_start(self, world) -> None:
        preload_active_skills(self, world)

    def _retrieve_for_subgoal(self, world, task, subgoal: str) -> str:
        if self.memory is None:
            return ""
        task_instr = getattr(task, "instruction", "") or ""
        retrieved = self.memory.retrieve(
            task_instr,
            extra_queries=[subgoal] if subgoal else None,
        ) or ""
        if retrieved:
            names = set(getattr(
                self.memory, "_context_function_names", set()))
            picked: list[tuple[str, str]] = []
            for s in getattr(self.memory, "_skills", []):
                name = (s.extra or {}).get("name") or ""
                if name in names and s.source:
                    picked.append((name, s.source))
            if self.benchmark in _PYTHON_REPL_BENCHES:
                for name, src in picked:
                    out = world.execute(src)
                    if isinstance(out, str) and "Execution failed" in out:
                        if hasattr(self.memory, "remove_by_name"):
                            self.memory.remove_by_name(name)
            elif self.benchmark in _SIDECAR_PY_BENCHES:
                ns = getattr(self, "_py_ns", None)
                if ns is None:
                    ns = {}
                    self._py_ns = ns
                good: list[str] = []
                for name, src in picked:
                    try:
                        exec(compile(src, "<sidecar-preload>", "exec"),
                             ns)
                        good.append(src)
                    except Exception:
                        if hasattr(self.memory, "remove_by_name"):
                            self.memory.remove_by_name(name)
                # Also define the skills in the env's executor namespace, where officebench's run_code calls them; otherwise they raise NameError at call time.
                if good and hasattr(world, "preload_skills"):
                    world.preload_skills(good)
            for name, src in _split_def_blocks([s for _, s in picked]):
                self._register_function(name, src)
        return retrieved

    def _on_executor_run_code(self, world, code, observation) -> bool:
        if self.memory is None:
            return False
        if hasattr(self.memory, "check_and_remove_errored"):
            self.memory.check_and_remove_errored(code, observation)
        return False

    def _on_subtask_end(self, world, executor_history, subtask_idx):
        if self.memory is None:
            return
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
            tag="SubtaskASI induction",
            on_attempt=self._account_tokens,
            require_tool="skill_induction",
            ledger=getattr(self.model, "ledger", None),
        )
        if resp is None or not resp.tool_calls:
            return
        delta_M = parse_code_skill_update_json(
            resp.tool_calls[0].arguments or {})
        if delta_M:
            self.memory.apply_skill_delta(delta_M, world=world)

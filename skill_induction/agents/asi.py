"""LinearASI: vanilla ReAct plus code-memory skill induction at task end.

At task start the agent retrieves the top function from `ASIMemory`, exec()s it into the REPL so the name resolves, and appends the function source as a user turn; skills whose exec fails are marked poisoned. At task end the induction prompt is appended to the finished trajectory and the model is forced to emit a single `skill_induction` tool call, whose arguments go to `memory.apply_skill_delta`.

The induction demo (a user / assistant tool_call / tool ack triple) is spliced into the initial history so the model sees one full example of the induction step before producing its own.
"""

from __future__ import annotations

import ast

from skill_induction.agents.linear import LinearAgent
from skill_induction.memory._delta import parse_code_skill_update_json
from skill_induction.prompts.asi_induction import (
    create_skill_update_prompt,
    induction_demo_messages,
    skill_induction_tool,
)


class LinearASI(LinearAgent):

    _agent_kind = "asi"

    def _build_initial_history(self, task) -> list[dict]:
        history = super()._build_initial_history(task)
        # Consume-only arm (--subtask-skills): drop the induction demo, since induction never runs.
        if not self._subtask_skills:
            demo = induction_demo_messages(
                self.benchmark, variant="linear", level=self._induction_level)
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
            for name, src in _split_def_blocks(
                    self.memory.get_active_skill_codes()):
                self._register_function(name, src)
        return history

    def _run_loop_setup(self, world) -> None:
        preload_active_skills(self, world)

    def _on_task_end(self, world, history, completed: bool) -> None:
        # Induce regardless of `completed` (incl. forced completions); a final induction always runs at task end.
        if self.memory is None or self._subtask_skills:
            return
        prompt = create_skill_update_prompt(
            memory_brief=self.memory.get_brief(),
            benchmark=self.benchmark,
            granularity="coarse",  # task-level: one whole-task skill
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
            tag="LinearASI induction",
            require_tool="skill_induction",
            ledger=getattr(self.model, "ledger", None),
        )
        if resp is None or not resp.tool_calls:
            return
        delta_M = parse_code_skill_update_json(
            resp.tool_calls[0].arguments or {})
        if delta_M:
            self.memory.apply_skill_delta(delta_M, world=world)


def _split_def_blocks(sources: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for src in sources or []:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                out.append((node.name, src))
    return out


_PYTHON_REPL_BENCHES = {"appworld", "kramabench"}
_SIDECAR_PY_BENCHES = {"officebench"}


def preload_active_skills(agent, world) -> None:
    """Exec each active skill source into the REPL so future calls can refer to them by name. Sources whose exec fails are poisoned.

    appworld / kramabench: `world.execute` is a Python kernel, so preload by running the source through it. officebench: its main action surface (dicts) is not Python, so preload into the agent's in-process `_py_ns` sidecar, where the model's `python_code` tool argument routes.
    """
    mem = getattr(agent, "memory", None)
    if mem is None:
        return
    bench = agent.benchmark
    names = mem.get_active_skill_names() if hasattr(
        mem, "get_active_skill_names") else []
    codes = mem.get_active_skill_codes() if hasattr(
        mem, "get_active_skill_codes") else []
    if bench in _PYTHON_REPL_BENCHES:
        for fn_name, src in zip(names, codes):
            if not src:
                continue
            out = world.execute(src)
            if isinstance(out, str) and "Execution failed" in out:
                if hasattr(mem, "remove_by_name"):
                    mem.remove_by_name(fn_name)
        return
    if bench in _SIDECAR_PY_BENCHES:
        ns = getattr(agent, "_py_ns", None)
        if ns is None:
            ns = {}
            agent._py_ns = ns
        good: list[str] = []
        for fn_name, src in zip(names, codes):
            if not src:
                continue
            try:
                exec(compile(src, "<sidecar-preload>", "exec"), ns)
                good.append(src)
            except Exception:
                if hasattr(mem, "remove_by_name"):
                    mem.remove_by_name(fn_name)
        # The skills must also be defined in the env's executor namespace, where officebench run_code's `_run_python` calls them; otherwise a retrieved skill raises NameError at call time because it only exists in the agent sidecar `ns`.
        if good and hasattr(world, "preload_skills"):
            world.preload_skills(good)
        return

"""Subtask agent: planner + executor with periodic reflection.

Architecture:

  plan_option (LLM) → first subgoal
  → executor ReAct loop until `end_subgoal`
  → reflect on the executor's live history → summary
  → (optional) skill induction in the same conversation → memory write
  → plan_option (LLM) with the new summary → next subgoal
  → executor loop again ...
  until `apis.supervisor.complete_task()` / `system.finish_task` / task-specific completion signal is reached, or `max_iter` is hit.

The executor's history is rebuilt fresh each subgoal so its context stays bounded: planner system + planner demos + identity + the current subgoal + current progress + retrieved memory.

Subclasses (`SubtaskAWM`, `SubtaskASI`) override `_memory_type` to enable cross-task memory and skill induction.
"""

from __future__ import annotations

import copy
import os as _os
import time
from typing import Optional

from tqdm import tqdm

from skill_induction.agents.base import ReactAgent
from skill_induction.caching.call_ledger import set_phase
from skill_induction.prompts.react import react_tools
from skill_induction.prompts.subtask import (
    INITIAL_PROGRESS,
    PLANNER_NEXT_SUBTASK_NUDGE,
    SUMMARIZE_USER,
    build_executor_system_prompt,
    build_planner_system_prompt,
    build_summarizer_system_prompt,
    identity_line,
    plan_demo_messages,
    plan_option_tool,
    subagent_demo_messages,
    write_progress_summary_tool,
)


_COMPLETION_MARKERS = {
    "appworld": "complete_task",
    "officebench": "finish_task",
    "kramabench": "complete_task",
}

import re as _re

# The executor ends a subgoal by writing `end_subgoal()` inside its run_code; there is no end_subgoal tool. This matches the token in the run_code payload.
_END_SUBGOAL_RE = _re.compile(r"\bend_subgoal\b")


def _split_end_subgoal(payload: str) -> tuple[str, bool]:
    """Return (code_without_end_signal, ended). Lines referencing `end_subgoal` are stripped so the remaining real code can run cleanly; the model may do real work and call end_subgoal() in the same block."""
    if not payload or not _END_SUBGOAL_RE.search(payload):
        return payload, False
    kept = [ln for ln in payload.splitlines()
            if not _END_SUBGOAL_RE.search(ln)]
    return "\n".join(kept).strip(), True


def _is_comment_or_blank_only(payload: str) -> bool:
    """True if the payload has no executable content: only blank lines and `#` comments (e.g. `# Subgoal complete.`). Weak models often signal subgoal completion with such a comment instead of writing `end_subgoal()`; treating it as an end_subgoal avoids a bogus 'did not yield an action' step."""
    if not payload or not payload.strip():
        return False
    for ln in payload.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        return False
    return True


class SubtaskAgent(ReactAgent):

    # Subclasses set "doc" or "code" to enable cross-task induction.
    _memory_type: Optional[str] = None

    # Cap on subgoals per task (loop guard). There is no per-subgoal step cap and no forced end_subgoal: the executor runs until it naturally calls end_subgoal / complete_task, bounded only by the remaining total task budget; the only forced action is complete_task, on budget exhaustion or retry failure.
    MAX_SUBGOALS = 15

    def __init__(self, model, *, benchmark: str = "appworld",
                 memory=None, verbose: bool = False,
                 induction_level: int = 3,
                 max_subtasks: int = 15,
                 max_iter_per_subtask: int = 15):
        super().__init__(model, benchmark=benchmark, memory=memory,
                         verbose=verbose, induction_level=induction_level)
        all_tools = list(react_tools(benchmark, agent_kind=self._agent_kind))
        # Executor tools: run_code only, no `end_subgoal` tool. With `tool_choice="required"` some models reflexively emitted the no-arg end_subgoal to satisfy the requirement, ending the subgoal with zero work; instead the executor writes `end_subgoal()` inside its run_code (and finishes the whole task via the bench completion action), so ending requires writing code like any other action.
        self.executor_tools = [t for t in all_tools
                               if (t.get("function") or {}).get("name")
                               != "finish"]
        # Planner tools: only plan_option, no `finish` tool; the task terminates from the environment side when the bench completion action returns success, and giving the planner `finish` led weaker models to short-circuit tasks after over-optimistic reflect summaries.
        self.planner_tools = [plan_option_tool()]
        # The ReactAgent.run_task path is unused for this subclass, but some base helpers read `self.tools`, so keep it sane.
        self.tools = self.executor_tools

        # OfficeBench tasks are short; the iterative planner ignores the prompt's "<=5" guidance and over-decomposes, adding handoff overhead and non-actionable steps, so hard-cap it.
        self.MAX_SUBGOALS = max_subtasks
        # `max_iter_per_subtask` is accepted for signature compatibility but intentionally not used as a cap (no per-subtask limit).
        _ = max_iter_per_subtask

        self._current_summary: str = ""
        self._task_completed: bool = False
        self._planner_finished: bool = False
        self._subtask_plans: list[str] = []
        self._subtask_summaries: list[str] = []
        self._subtask_traces: list[list[dict]] = []
        # Combined display history for inspector / debugging.
        self._display_history: list[dict] = []
        # Planner growing chat history (built lazily in `_plan_subgoal`).
        self._planner_history: list[dict] = []

    # public API

    def run_task(self, world, max_iter: int = 50) -> dict:
        self.stats = self._init_stats()
        self.stats["start_time"] = time.time()
        self.trace = []
        self._current_summary = ""
        self._task_completed = False
        self._planner_finished = False
        self._subtask_plans = []
        self._subtask_summaries = []
        self._subtask_traces = []
        self._display_history = []
        self._last_executor_history = None
        self._planner_history = []
        self._func_sources = {}
        # Per-task set of retrieved-skill function names actually called by the executor; both the `code` path (_execute_action) and the `python_code` sidecar feed it via `_scan_invocations`, accumulated at agent level so the task-level record sees the union across subgoals.
        self._invoked_skill_names = set()
        # Surface agent_kind on the world so per-bench prompts (e.g. OfficeBench's per-turn action menu) can adapt their wording; ASI exposes a Python-function path the non-ASI tools don't.
        try:
            world.agent_kind = self._agent_kind
            world.llm_fn = self._officebench_llm_complete
        except AttributeError:
            pass

        task = world.task

        # Planner state: built once, grows each subgoal cycle.
        self._planner_system = build_planner_system_prompt(self.benchmark)
        self._planner_demos = plan_demo_messages(self.benchmark)
        # Executor state: rebuilt each subgoal.
        self._executor_system = build_executor_system_prompt(self.benchmark)
        self._executor_demos = subagent_demo_messages(self.benchmark)

        completed = False
        step_budget = max_iter

        # First subgoal.
        set_phase(self.model, "planner")
        subgoal = self._plan_subgoal(task, summary="")
        if subgoal is None:
            return self._wrap_metrics(world, completed=False)

        for subtask_idx in range(self.MAX_SUBGOALS):
            if step_budget <= 0:
                break
            self._subtask_plans.append(subgoal)

            set_phase(self.model, f"executor[{subtask_idx}]")
            cross_task = self._retrieve_for_subgoal(world, task, subgoal)
            # Carry the live env state (OfficeBench's per-turn action menu for the current app) into the fresh executor so it resumes where the previous subgoal left off instead of re-discovering.
            env_state = getattr(world, "current_action_menu", lambda: "")()
            executor_history = self._build_executor_history(
                task, subgoal=subgoal,
                summary=self._current_summary,
                cross_task=cross_task,
                env_state=env_state,
            )
            self._on_executor_start(world)
            self._last_executor_history = executor_history

            # No per-subtask step cap: the executor runs until it naturally calls end_subgoal / complete_task, bounded only by the remaining total task budget; if it burns the whole remaining budget in one subgoal, the outer loop's force-complete-task catches it.
            subgoal_completed, intra_trace, steps_used = self._run_executor(
                world, executor_history,
                step_budget=step_budget,
                subtask_idx=subtask_idx,
            )
            step_budget -= max(steps_used, 1)
            self._subtask_traces.append(intra_trace)
            # Append the executor's history into the display history for inspector visibility, excluding the system / demo prefix so the inspector doesn't double-render demos.
            prefix_len = len(self._executor_demos) + 1  # system + demos
            self._display_history.extend(executor_history[prefix_len:])

            # Summarize (reflect) then induce, uniformly after every subgoal that ended normally, including the final task-completing one, so the final subgoal's skill is induced on a summarized history like all the others. A subgoal "ended normally" when the executor wrote end_subgoal() or ran the completion action.
            if subgoal_completed:
                set_phase(self.model, f"reflect[{subtask_idx}]")
                summary = self._reflect(executor_history)
                if summary:
                    self._current_summary = summary
                    self._subtask_summaries.append(summary)

            # Induce (AWM/ASI) after reflect, sharing the same conversation for KV-cache reuse; covers a normal subgoal end and a stuck / force-completed subgoal (task done but subgoal_completed False).
            if ((subgoal_completed or self._task_completed)
                    and self._memory_type is not None
                    and self.memory is not None):
                set_phase(self.model, f"induction[{subtask_idx}]")
                self._on_subtask_end(world, executor_history, subtask_idx)

            # Task done: the executor's run_code triggered the bench completion marker (or a forced completion fired), so end the loop (the summarizer + induction for this final subgoal already ran).
            if self._task_completed:
                completed = True
                break

            if not subgoal_completed and step_budget <= 0:
                break

            # Plan next subgoal.
            if step_budget <= 0 or subtask_idx + 1 >= self.MAX_SUBGOALS:
                break
            set_phase(self.model, "planner")
            subgoal = self._plan_subgoal(task, summary=self._current_summary)
            if self._planner_finished:
                completed = self._task_completed
                break
            if subgoal is None:
                completed = self._task_completed
                break

        # Force a terminal action when the loop exited without the env signal firing (max_iter hit, planner gave up, MAX_SUBGOALS exhausted); recorded as a normal complete_task step. Then still induce the last subgoal's skill for awm/asi, because a forced finish doesn't mean there's nothing to learn from the work.
        if not completed:
            forced_hist: list[dict] = []
            if self._force_terminal_action(world, forced_hist):
                completed = True
            self._display_history.extend(forced_hist)
            last_hist = getattr(self, "_last_executor_history", None)
            if (last_hist is not None and self._memory_type is not None
                    and self.memory is not None):
                set_phase(self.model, "induction[final]")
                self._on_subtask_end(world, last_hist, -1)

        difficulty = world.calculate_difficulty()
        success, eval_extras = world.evaluate_run()
        return self._wrap_metrics(
            world, completed=completed, success=success,
            difficulty=difficulty, eval_extras=eval_extras,
        )

    # planner

    def _plan_subgoal(self, task, summary: str) -> Optional[str]:
        # Rebuild the planner conversation fresh on every call: the model sees [system, *demos, user(task + current progress)] each time, with no accumulated past plan_options or summaries, which stops the planner from drifting into "re-plan complete_task" loops after a few cycles.
        ident = identity_line(task, self.benchmark)
        progress = summary.strip() or INITIAL_PROGRESS
        user_parts = []
        if ident:
            user_parts.append(ident)
            user_parts.append("")
        user_parts.append(f"Task: {task.instruction}")
        user_parts.append("")
        user_parts.append(f"Current Progress: {progress}")
        # Planner-level procedural memory (opt-in via SKILL_INDUCTION_PLANNER_MEMORY=1): give the planner a menu of reusable steps distilled from past tasks so it decomposes to leverage them, knowledge the live environment menu does not provide; the executor still receives the full skill bodies.
        if (self.memory is not None and _os.environ.get(
                "SKILL_INDUCTION_PLANNER_MEMORY") in ("1", "true", "True")):
            brief = (self.memory.get_brief()
                     if hasattr(self.memory, "get_brief") else "") or ""
            if brief.strip() and "empty" not in brief[:40].lower():
                user_parts.append("")
                user_parts.append(
                    "Reusable steps learned from past tasks (prefer "
                    "decomposing so each subgoal maps to one of these "
                    "when applicable):")
                user_parts.append(brief)
        user_msg = "\n".join(user_parts)
        history: list[dict] = [
            {"role": "system", "content": self._planner_system},
            *[dict(m) for m in self._planner_demos],
            {"role": "user", "content": user_msg},
        ]
        self._display_history.append({"role": "user", "content": user_msg})

        # 10-attempt tight retry: only successful tool-call responses land in history, text-only failures are silently dropped, and `on_attempt=self._account_tokens` counts every retry's tokens in the agent-loop stats.
        from skill_induction.agents.base import _retry_until_tool_call
        resp = _retry_until_tool_call(
            lambda: self.model.chat(
                history, tools=self.planner_tools,
                tool_choice="required", max_tokens=8192,
            ),
            tag="Subtask planner",
            on_attempt=self._account_tokens,
            require_tool="plan_option",
            ledger=getattr(self.model, "ledger", None),
        )
        if resp is None:
            return None
        tc = resp.tool_calls[0]
        args = tc.arguments or {}
        assistant_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": args},
            ],
        }
        if tc.name == "plan_option":
            subgoal = (args.get("subgoal") or "").strip()
            if subgoal:
                history.append(assistant_msg)
                self._display_history.append(assistant_msg)
                return subgoal
        # Unknown tool or empty subgoal: fall back to the task instruction.
        return task.instruction[:200]

    # executor

    def _build_executor_history(self, task, *, subgoal: str,
                                summary: str, cross_task: str,
                                env_state: str = "") -> list[dict]:
        ident = identity_line(task, self.benchmark)
        progress = summary.strip() or INITIAL_PROGRESS
        parts = []
        if ident:
            parts.append(ident)
            parts.append("")
        parts.append(f"Your current subgoal: {subgoal}")
        parts.append("")
        parts.append(f"Overall task (for context only): {task.instruction}")
        parts.append("")
        parts.append(f"Current Progress: {progress}")
        if env_state.strip():
            parts.append("")
            parts.append(env_state.rstrip())
        # Retrieved skills arrive already framed by the memory class (hints, not ground truth).
        framed = (cross_task or "").strip()
        if framed:
            parts.append("")
            parts.append(framed)
        parts.append("")
        parts.append("Proceed with execution. When the subgoal is complete, "
                     "write `end_subgoal()` in a run_code call.")
        user_msg = "\n".join(parts)
        return [
            {"role": "system", "content": self._executor_system},
            *[dict(m) for m in self._executor_demos],
            {"role": "user", "content": user_msg},
        ]

    def _on_executor_start(self, world) -> None:
        """Hook for subclasses (e.g. preload skills into the REPL)."""
        return None

    def _run_executor(self, world, history: list[dict], *,
                      step_budget: int, subtask_idx: int
                      ) -> tuple[bool, list[dict], int]:
        intra: list[dict] = []
        steps = 0          # run_code actions executed this subgoal
        completed = False
        _last_fail_obs = None   # break identical-failure retry loops
        _fail_streak = 0
        for _ in tqdm(range(step_budget), desc="Subtask exec",
                      leave=False, unit="step"):
            response = self._call_executor(history, tools=self.executor_tools)
            # `_call_executor` retries 10x on no-tool-call; if it still returns None the model is stuck, so force complete_task (recorded as a normal model action, no "forced" label) and signal task done.
            if response is None:
                self._force_terminal_action(world, history)
                self._task_completed = True
                return False, intra, steps
            history.append(self._assistant_message_from_response(response))
            # The executor's only tool is run_code; ending the subgoal and finishing the task are both expressed inside the code.
            tc = response.tool_calls[0]
            args = tc.arguments or {}
            py_src = (args.get("python_code") or "").strip()
            payload = self._action_payload(tc)
            # Detect an `end_subgoal` signal written into the code and split it off from any real code in the same block.
            real_payload, ended = _split_end_subgoal(payload)
            real_py_src, ended_py = _split_end_subgoal(py_src)
            ended = ended or ended_py
            payload, py_src = real_payload, real_py_src

            # OfficeBench only: a comment-only payload (e.g. "# Subgoal complete.", no real action) is the model signalling completion via a comment instead of end_subgoal(); treat it as end_subgoal so the bogus 'did not yield an OfficeBench action' error step never wastes a step or pollutes history.
            if not ended and self.benchmark == "officebench":
                _combined = "\n".join(
                    x for x in (payload, py_src) if x).strip()
                if _combined and _is_comment_or_blank_only(_combined):
                    ended = True
                    payload, py_src = "", ""

            ran_real = False
            observation = ""
            if payload or py_src:
                sidecar_out = ""
                if py_src and self.benchmark == "officebench":
                    sidecar_out = self._run_python_sidecar(world, py_src)
                main_out = (self._execute_action(world, payload)
                            if payload else "")
                if sidecar_out and main_out:
                    observation = (
                        f"[python_code]\n{sidecar_out}\n\n"
                        f"[code]\n{main_out}"
                    )
                else:
                    observation = sidecar_out or main_out
                ran_real = True
                steps += 1

            obs_text = (observation if isinstance(observation, str)
                        else str(observation))
            if ended:
                obs_text = (obs_text + "\n" if obs_text else "") + \
                    "(subgoal complete — control returned to the planner)"
            history.append({
                "role": "tool", "tool_call_id": tc.id,
                "name": tc.name, "content": obs_text,
            })
            if ran_real:
                intra.append({
                    "type": "run_code",
                    "content": payload or py_src,
                    "output": observation,
                })
                self._on_executor_run_code(world, payload, observation)
                if self._check_task_completed(payload, observation):
                    self._task_completed = True
                    return True, intra, steps
                # OfficeBench only: weak models retry the identical failing action many times (e.g. a malformed llm prompt or a guessed path); after 3 identical consecutive failures, inject one strong nudge to change approach so the step budget isn't burned looping.
                if self.benchmark == "officebench" and \
                        "Execution failed" in obs_text:
                    if obs_text == _last_fail_obs:
                        _fail_streak += 1
                    else:
                        _last_fail_obs, _fail_streak = obs_text, 1
                    if _fail_streak >= 3:
                        history.append({
                            "role": "user",
                            "content": (
                                "STOP — that exact action has failed 3× "
                                "with the same error. Do NOT repeat it. "
                                "Change approach: fix the key names against "
                                "the ##Instruction menu, switch to the "
                                "right app, run `ls /testbed/data` to get "
                                "exact filenames, or use a different "
                                "action. If this subgoal genuinely cannot "
                                "proceed, write `end_subgoal()`."),
                        })
                        _fail_streak = 0
                else:
                    _last_fail_obs, _fail_streak = None, 0
            if ended:
                completed = True
                break
            continue
            # Unknown tool: return an error.
            history.append({
                "role": "tool", "tool_call_id": tc.id, "name": tc.name,
                "content": (
                    f"ERROR: '{tc.name}' is not available. Call "
                    f"`run_code` or `end_subgoal`."),
            })
        # Loop ran out of total budget without the executor calling end_subgoal / complete_task. Do not force end_subgoal (no per-subtask forcing); return completed=False and let the outer loop's force-complete-task handle the task end.
        return completed, intra, steps

    def _call_executor(self, history, tools=None):
        # 10-attempt tight retry on no-tool-call responses; failed responses never enter history (see `_retry_until_tool_call`), and `on_attempt=self._account_tokens` counts every retry's tokens in agent-loop stats so the cost reflects all attempts.
        from skill_induction.agents.base import _retry_until_tool_call
        tools = tools if tools is not None else self.executor_tools
        return _retry_until_tool_call(
            lambda: self.model.chat(
                history, tools=tools,
                tool_choice="required", max_tokens=8192,
            ),
            tag="Subtask executor",
            on_attempt=self._account_tokens,
            ledger=getattr(self.model, "ledger", None),
        )

    def _on_executor_run_code(self, world, code, observation) -> bool:
        """Hook for subclasses; called after every real run_code execution."""
        return False

    def _check_task_completed(self, code, observation) -> bool:
        # Delegate to the universal env-signal detector on ReactAgent.
        return self._run_code_completed(code, observation)

    # reflect

    def _reflect(self, history: list[dict]) -> str:
        # Build the reflect branch: summarizer system + the real subgoal turn (which carries the accumulated "Current Progress") + the real run_code turns, keeping the (swapped) system slot plus everything from real_user onward of the [system, *executor_demos, real_user, *real_run] layout. The executor's few-shot demos are stripped because leaving them in made the summary mix demo content and drop the accumulated progress.
        summarizer_system = build_summarizer_system_prompt(self.benchmark)
        demo_count = len(getattr(self, "_executor_demos", []) or [])
        if history and history[0].get("role") == "system":
            tail = history[1 + demo_count:]
        else:
            tail = history[demo_count:]
        branch: list[dict] = [
            {"role": "system", "content": summarizer_system},
            *[dict(m) for m in tail],
            {"role": "user", "content": SUMMARIZE_USER},
        ]
        tool = write_progress_summary_tool()
        # Retry until the model returns the write_progress_summary tool specifically: Bedrock does not reliably enforce specific-tool toolChoice for the open models, which otherwise emit a stray run_code here and the summary silently empties (and stops accumulating across subgoals).
        from skill_induction.agents.base import _retry_until_tool_call
        resp = _retry_until_tool_call(
            lambda: self.model.chat(
                branch, tools=[tool],
                tool_choice="required",
                max_tokens=8192,
            ),
            tag="Subtask reflect",
            on_attempt=self._account_tokens,
            require_tool="write_progress_summary",
            ledger=getattr(self.model, "ledger", None),
        )
        if resp is None:
            return ""
        tcs = list(resp.tool_calls or [])
        summary = ""
        if tcs:
            args = tcs[0].arguments or {}
            summary = (args.get("summary") or "").strip()
        if not summary:
            summary = (resp.content or "").strip()
        if summary:
            # Mirror the reflect call into the live history so a later induction step shares the same conversation.
            history.append({"role": "user", "content": SUMMARIZE_USER})
            if tcs:
                # Drop content on tool-call turns (see base.py _assistant_message_from_response); the summary lives in the tool args, not in content.
                history.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": tcs[0].id, "name": tcs[0].name,
                         "arguments": tcs[0].arguments or {}},
                    ],
                })
                history.append({
                    "role": "tool",
                    "tool_call_id": tcs[0].id,
                    "name": tcs[0].name,
                    "content": "(summary recorded)",
                })
            else:
                history.append({"role": "assistant", "content": summary})
        return summary

    # induction (overridden by AWM / ASI)

    def _retrieve_for_subgoal(self, world, task, subgoal: str) -> str:
        """Cross-task retrieval; overridden by AWM / ASI subclasses."""
        return ""

    def _on_subtask_end(self, world, executor_history: list[dict],
                        subtask_idx: int) -> None:
        """Optional skill-induction hook; overridden by AWM / ASI."""
        return None

    # plumbing

    def _account_tokens(self, resp) -> None:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        self.stats["total_prompt_tokens"] += getattr(
            usage, "prompt_tokens", 0) or 0
        self.stats["total_completion_tokens"] += getattr(
            usage, "completion_tokens", 0) or 0
        self.stats["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
        peak = max(self.stats["peak_prompt_tokens"],
                   getattr(usage, "prompt_tokens", 0) or 0)
        self.stats["peak_prompt_tokens"] = peak

    def _wrap_metrics(self, world, *, completed: bool,
                      success: bool = False, difficulty: int = 0,
                      eval_extras: Optional[dict] = None) -> dict:
        self.stats["end_time"] = time.time()
        # KramaBench reports a continuous per-task score (RAE/F1/0-1); record it as tgc_score so subtask and linear cells use the same metric (the ReactAgent base does this too), otherwise subtask's binary success rate would inflate its mean vs linear's continuous score.
        if (self.benchmark == "kramabench" and isinstance(eval_extras, dict)
                and "score" in eval_extras):
            tgc = float(eval_extras["score"])
        else:
            tgc = 1.0 if success else 0.0
        return {
            "completed": completed,
            "success": success,
            "tgc_score": tgc,
            "subtask_count": len(self._subtask_plans),
            "env_steps": self.stats["env_steps"],
            "wall_clock_seconds": (
                self.stats["end_time"] - self.stats["start_time"]),
            "total_prompt_tokens": self.stats["total_prompt_tokens"],
            "total_completion_tokens": (
                self.stats["total_completion_tokens"]),
            "peak_prompt_tokens": self.stats["peak_prompt_tokens"],
            "difficulty": difficulty,
            "eval_extras": eval_extras or {},
            "invoked_skill_names": sorted(
                getattr(self, "_invoked_skill_names", set()) or set()),
            "trace": self.trace,
            "subtask_plans": list(self._subtask_plans),
            "subtask_summaries": list(self._subtask_summaries),
            "final_history": copy.deepcopy(self._display_history),
        }

"""ReactAgent base: tool-call loop, memory-free by design.

Built on the `ChatGenerator` + `World` + `SkillMemory` contracts.

Subclasses extend behavior via hooks rather than overriding `run_task`:

    _build_initial_history(task)         # inject retrieved skills / system msg
    _on_task_end(world, history, completed)  # write skills back to memory
    _dispatch_tool_call(tc, world)       # custom tool dispatch
    _action_payload(tc)                  # custom action-payload extraction
    _task_completed(observation, payload)  # custom completion detector
"""

from __future__ import annotations

import ast
import copy
import json
import re
import time

from tqdm import tqdm

from skill_induction.caching.call_ledger import set_phase
from skill_induction.models.base import ChatGenerator
from skill_induction.runtime_limits import CodeTimeout, time_limit
from skill_induction.prompts.react import (
    ACTION_ARG_KEY,
    create_react_chat,
    react_tools,
)


_COMPLETION_PATTERNS = {
    "appworld": "apis.supervisor.complete_task(",
    "kramabench": "complete_task(",
}


# Bench-specific terminal `run_code` payload. When the loop exhausts retries or its budget, we synthesize this run_code so the env records a completion signal and the evaluator runs. It is recorded as a normal run_code action, with no "forced" label.
_TERMINAL_RUN_CODE: dict[str, str] = {
    "appworld": "apis.supervisor.complete_task()",
    "officebench": "{'app': 'system', 'action': 'finish_task', 'answer': 'None'}",
    "kramabench": "complete_task(answer=None)",
}


def _retry_until_tool_call(call_fn, *, tag: str = "LLM",
                            max_attempts: int = 10,
                            sleep_s: float = 1.0,
                            on_attempt=None,
                            require_tool: str | None = None,
                            ledger=None):
    """Call `call_fn` up to `max_attempts` times until it returns a response with at least one tool_call; returns that response, or None if all attempts fail.

    Failed responses (None or no tool_calls) are discarded and never appended to chat history, so the model only sees its own successful tool calls.

    `on_attempt(response)` runs for every attempt so the caller can accumulate token usage across retries.

    `require_tool` accepts only a response whose first tool_call has this name. Bedrock does not reliably enforce specific-tool `toolChoice` for the open models, which sometimes emit a different tool on forced-tool steps (induction, planner); without this check the wrong tool's args would be recorded and parse to an empty skill.

    Failed attempts' entries are popped from `ledger.entries` so they don't show up as bogus text cells in the pkl / HTML inspector; their token usage is still counted via `on_attempt`.
    """
    import time as _time
    for attempt in range(max_attempts):
        n_before = len(ledger.entries) if ledger is not None else 0
        try:
            resp = call_fn()
        except Exception as e:
            print(f"[{tag}] attempt {attempt+1}/{max_attempts} error: {e}")
            resp = None
        if on_attempt is not None and resp is not None:
            try:
                on_attempt(resp)
            except Exception as e:
                print(f"[{tag}] on_attempt callback raised: {e}")
        ok = bool(resp is not None and resp.tool_calls)
        if ok and require_tool is not None:
            ok = resp.tool_calls[0].name == require_tool
        if ok:
            return resp
        # Drop any ledger entries the failed attempt produced so the inspector doesn't show a text-only cell or a rejected wrong-tool step.
        if ledger is not None:
            while len(ledger.entries) > n_before:
                ledger.entries.pop()
        if attempt < max_attempts - 1:
            if resp is not None and resp.tool_calls and require_tool:
                why = f"wrong tool {resp.tool_calls[0].name!r}!={require_tool!r}"
            else:
                why = f"content={'<empty>' if resp is None else 'text-only'}"
            print(f"[{tag}] attempt {attempt+1}/{max_attempts}: no usable "
                  f"tool call ({why}), retrying in {sleep_s:.1f}s")
            _time.sleep(sleep_s)
    print(f"[{tag}] giving up after {max_attempts} attempts.")
    return None


class ReactAgent:
    """Tool-call loop. Memory-free by design."""

    # ASI subclasses override to "asi" so the officebench run_code schema exposes the `python_code` sidecar field.
    _agent_kind: str = "non-asi"

    def __init__(
        self,
        model: ChatGenerator,
        *,
        benchmark: str = "appworld",
        memory=None,
        verbose: bool = False,
        induction_level: int = 3,
        skills_replay: bool = False,
        skills_half: str = "all",
        skills_demo: str = "linear",
    ):
        self.model = model
        self.benchmark = benchmark
        self.memory = memory
        self.tools = react_tools(benchmark, agent_kind=self._agent_kind)
        self.verbose = verbose
        # Prompt-component ablation level for skill induction (memory agents only): 1=instruction-only, 2=+demonstration, 3=+rules (deployed).
        self._induction_level = induction_level
        # Skill-transfer arm (linear memory agents only): memory is a frozen, externally supplied library; the agent retrieves from it but never induces into it (no induction demo, no task-end induction call). `skills_half` ('all'|'high'|'low') records which utility half of the donor library the runner loaded; `skills_demo` ('linear'|'subtask') which demonstration granularity was requested. Neither changes the prompt here (see LinearAgent._apply_frozen_skills_prompt).
        self._skills_replay = skills_replay
        self._skills_half = skills_half
        self._skills_demo = skills_demo
        self.trace: list[dict] = []
        self.stats = self._init_stats()
        # Per-task function-source registry, populated by ASI subclasses to detect indirect completion through wrappers.
        self._func_sources: dict[str, str] = {}
        # In-process Python sidecar namespace for ASI on benches without a Python REPL (officebench); preloaded helper defs live here and `python_code` args are exec'd in this dict.
        self._py_ns: dict = {}

    # public API

    def run_task(self, world, max_iter: int = 50) -> dict:
        """Returns a metrics dict (one row per task in the result pkl)."""
        self.stats = self._init_stats()
        self.stats["start_time"] = time.time()
        self.trace = []
        self._func_sources = {}
        self._py_ns = {}
        # Retrieved-skill function names actually called in executed code (invoked, not merely retrieved); populated by `_scan_invocations`.
        self._invoked_skill_names = set()
        # Surface agent_kind on the world so per-bench prompts (e.g. OfficeBench's action menu) can adapt their wording to the ASI Python-function path.
        try:
            world.agent_kind = self._agent_kind
            world.llm_fn = self._officebench_llm_complete
        except AttributeError:
            pass

        task = world.task
        set_phase(self.model, "retrieval")   # memory.retrieve in _build_initial_history
        history = self._build_initial_history(task)
        self._run_loop_setup(world)
        set_phase(self.model, "react")

        completed = False
        iters_used = 0

        for _ in tqdm(range(max_iter), desc="React", leave=False, unit="step"):
            iters_used += 1
            self._saw_end_subgoal = False
            # `_step` retries internally on no-tool-call; if it still comes back empty, force the bench-specific terminal action so the task ends gracefully.
            response = self._step(history)
            if response is None or not (response.tool_calls):
                forced = self._force_terminal_action(world, history)
                if forced:
                    completed = True
                break
            history.append(self._assistant_message_from_response(response))

            # Strict one-tool-per-step: `_step` truncated to first.
            tc = response.tool_calls[0]
            observation = self._dispatch_tool_call(tc, world)
            obs_text = (observation if isinstance(observation, str)
                        else str(observation))
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": obs_text,
            })
            if tc.name == "run_code" and self._run_code_completed(
                    self._action_payload(tc), obs_text):
                completed = True
                break
            if getattr(self, "_saw_end_subgoal", False):
                # end_subgoal at task level ends the episode; run the bench completion action so the env still records a completion signal.
                if self._force_terminal_action(world, history):
                    completed = True
                break

        # max_iter exhausted without success: force the terminal action so the env records a completion signal (the evaluator still decides success).
        if not completed and iters_used >= max_iter:
            if self._force_terminal_action(world, history):
                completed = True

        difficulty = world.calculate_difficulty()
        success, eval_extras = world.evaluate_run()
        set_phase(self.model, "induction")
        self._on_task_end(world, history, completed)
        self.stats["end_time"] = time.time()
        return self._finalize_metrics(
            success=success,
            completed=completed,
            difficulty=difficulty,
            eval_extras=eval_extras,
            history=history,
        )

    # extension hooks

    def _build_initial_history(self, task) -> list[dict]:
        # `create_react_chat` returns the full message list: system, few-shot demo as structured tool_calls, final user task message.
        return create_react_chat(task, benchmark=self.benchmark)

    def _run_loop_setup(self, world) -> None:
        """Hook invoked once after `_build_initial_history`, before the first model step. ASI variants exec retrieved skill sources into the environment here so their names resolve. Default: no-op."""
        return None

    def _on_task_end(self, world, history, completed: bool) -> None:
        return None

    def _action_payload(self, tc) -> str:
        arg_key = ACTION_ARG_KEY.get(self.benchmark, "code")
        raw = (tc.arguments or {}).get(arg_key, "")
        # The executor user template invites end_subgoal() inside run_code. At task level the whole task is the single subgoal: strip the signal so the remaining code runs, and remember it so run_task can end the episode.
        from skill_induction.agents.subtask import _split_end_subgoal
        code, ended = _split_end_subgoal(raw)
        if ended:
            self._saw_end_subgoal = True
        return code

    def _run_code_completed(self, code: str, observation: str) -> bool:
        """True when a run_code payload contains the bench-specific completion marker and the environment's observation acknowledges it. There is no separate finish tool; termination is purely an environment signal."""
        if not code or not isinstance(observation, str):
            return False
        marker = {
            "appworld":    "complete_task",
            "officebench": "finish_task",
            "kramabench":  "complete_task",
        }.get(self.benchmark)
        if marker is None:
            return False
        return marker in str(code).lower() and (
            "execution successful" in observation.lower()
            or "task finished" in observation.lower()
            or "marked the active task" in observation.lower()
        )

    def _ledger_event(self, name: str, detail: str = "") -> None:
        """Record a synthetic (non-LLM) agent action in the call ledger. The inspector renders from `llm_calls`, so forced terminal / end_subgoal actions would otherwise be invisible in its timeline."""
        led = getattr(self.model, "ledger", None)
        if led is not None:
            try:
                led.record_event(name=name, detail=detail)
            except Exception:
                pass

    def _force_terminal_action(self, world, history: list[dict]) -> bool:
        """Run the bench's completion action when the loop is out of budget or the model stopped emitting tool calls. Recorded as an ordinary run_code step in history and in the ledger, with no "forced" label. Returns True iff the env completion signal fired."""
        payload = _TERMINAL_RUN_CODE.get(self.benchmark)
        if payload is None:
            return False
        import uuid as _uuid
        from skill_induction.models.base import ChatResponse, ToolCall, Usage
        tc_id = f"runcode_{_uuid.uuid4().hex[:8]}"
        # Snapshot the pre-action messages for the ledger entry.
        msgs_snapshot = [dict(m) for m in history]
        history.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": tc_id, "name": "run_code",
                 "arguments": {"code": payload}},
            ],
        })
        observation = self._execute_action(world, payload)
        obs_text = (observation if isinstance(observation, str)
                    else str(observation))
        history.append({
            "role": "tool", "tool_call_id": tc_id, "name": "run_code",
            "content": obs_text,
        })
        # Record as a normal run_code call in the ledger so the inspector shows a clean run_code cell, not a synthetic "forced" event.
        led = getattr(self.model, "ledger", None)
        if led is not None:
            try:
                led.record(
                    model_id=getattr(self.model, "model_id", ""),
                    messages=msgs_snapshot,
                    tools=getattr(self, "tools", None),
                    tool_choice="required",
                    kwargs={"max_tokens": 8192},
                    response=ChatResponse(
                        content=None,
                        tool_calls=[ToolCall(id=tc_id, name="run_code",
                                             arguments={"code": payload})],
                        finish_reason="tool_use",
                        usage=Usage(0, 0, 0),
                    ),
                    cache_hit=False,
                    latency_s=0.0,
                )
            except Exception:
                pass
        return self._run_code_completed(payload, obs_text)

    def _officebench_llm_complete(self, prompt: str) -> str:
        """Back the OfficeBench `llm.complete_text` action with the agent's own model, injected onto the world as `llm_fn` (the upstream llm app shells out to OpenAI GPT-3.5, unavailable here). The call goes through the ledger like any other chat(), so its cost is counted."""
        prompt = (prompt or "").strip()
        if not prompt:
            return ("Execution failed: llm.complete_text needs a non-empty "
                    "'prompt'.")
        try:
            resp = self.model.chat(
                [{"role": "user", "content": prompt}], max_tokens=2048)
        except Exception as e:
            return f"Execution failed: llm app error: {e!r}"
        return (resp.content or "").strip() or "(the llm returned no text)"

    def _dispatch_tool_call(self, tc, world) -> str:
        if tc.name == "run_code":
            args = tc.arguments or {}
            py_src = (args.get("python_code") or "").strip()
            payload = self._action_payload(tc)
            # ASI sidecar: officebench `python_code` runs in self._py_ns and a dict result is forwarded as a native officebench action; the main `code` still runs after the sidecar when both are given.
            sidecar_out = ""
            if py_src and self.benchmark == "officebench":
                sidecar_out = self._run_python_sidecar(world, py_src)
            main_out = self._execute_action(world, payload) if payload else ""
            if sidecar_out and main_out:
                return f"[python_code]\n{sidecar_out}\n\n[code]\n{main_out}"
            return sidecar_out or main_out
        valid = [t["function"]["name"] for t in (self.tools or [])]
        return (
            f"ERROR: '{tc.name}' is not a registered tool. The only "
            f"tools available right now are: {valid}. If you meant to "
            f"call a Python function or app API named `{tc.name}`, put "
            f"the call inside `run_code`'s `code` argument (the model "
            f"is not allowed to invent top-level tool names)."
        )

    # completion-pattern call graph

    def _register_function(self, name: str, source: str) -> None:
        if not name or not source:
            return
        self._func_sources[name] = source

    def _bare_callees(self, source: str) -> set[str]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return set()
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                names.add(node.func.id)
        return names

    def _reaches_completion(self, name: str, visited: set[str]) -> bool:
        if name in visited:
            return False
        visited.add(name)
        src = self._func_sources.get(name)
        if not src:
            return False
        pat = _COMPLETION_PATTERNS.get(self.benchmark)
        if pat and pat in src:
            return True
        for callee in self._bare_callees(src):
            if self._reaches_completion(callee, visited):
                return True
        return False

    def _payload_reaches_completion(self, code: str) -> bool:
        pat = _COMPLETION_PATTERNS.get(self.benchmark)
        if not pat:
            return False
        if pat in code:
            return True
        try:
            roots = self._bare_callees(code)
        except SyntaxError:
            roots = set(re.findall(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*\(", code))
        if not roots:
            roots = set(re.findall(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*\(", code))
        visited: set[str] = set()
        for name in roots:
            if self._reaches_completion(name, visited):
                return True
        return False

    # internals

    def _step(self, history):
        # Retry no-tool-call responses; failures never enter history, so text-only output doesn't contaminate the conversation. If all attempts fail, return None and let the outer loop force a terminal action. `on_attempt` charges every attempt's tokens to agent-loop stats.
        response = _retry_until_tool_call(
            lambda: self.model.chat(
                history, tools=self.tools,
                tool_choice="required", max_tokens=8192,
            ),
            tag="React",
            on_attempt=self._account_usage,
            ledger=getattr(self.model, "ledger", None),
        )
        if response is None:
            return None

        # One tool call per step: truncate batched responses to the first call. Dropped calls never enter history, so the model doesn't reinforce batching; a batched plan just takes N turns instead of one.
        tcs = list(response.tool_calls or [])
        if len(tcs) > 1:
            response.tool_calls = tcs[:1]
            if self.verbose:
                print(f"[LLM] truncating {len(tcs)} batched calls → kept="
                      f"{tcs[0].name} dropped={[t.name for t in tcs[1:]]}")

        if self.verbose:
            preview_text = response.content or ""
            preview_calls = ", ".join(
                f"{tc.name}({json.dumps(tc.arguments)[:60]}...)"
                for tc in response.tool_calls
            )
            print(f"\n[LLM] tool_calls=[{preview_calls}]")
            if preview_text:
                print(f"  text: {preview_text[:200]}")
        return response

    def _run_python_sidecar(self, world, source: str) -> str:
        """Exec `source` in `self._py_ns`. If it sets `action` (or its last statement is an expression yielding a dict / str), forward that value to the bench's native action surface and append the env's observation to the captured stdout."""
        import ast
        import contextlib
        import io
        import traceback

        # The sidecar is where retrieved helpers get called on officebench; scan it for invocations like the Python-REPL `code` path does.
        self._scan_invocations(source)

        buf = io.StringIO()
        result_value = None
        try:
            tree = ast.parse(source, mode="exec")
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                final_expr = tree.body[-1].value
                preamble = ast.Module(
                    body=tree.body[:-1], type_ignores=[])
                ast.fix_missing_locations(preamble)
                expr_module = ast.Expression(body=final_expr)
                ast.fix_missing_locations(expr_module)
                with contextlib.redirect_stdout(buf), \
                        contextlib.redirect_stderr(buf), time_limit():
                    exec(compile(preamble, "<sidecar>", "exec"),
                         self._py_ns)
                    result_value = eval(
                        compile(expr_module, "<sidecar>", "eval"),
                        self._py_ns)
            else:
                with contextlib.redirect_stdout(buf), \
                        contextlib.redirect_stderr(buf), time_limit():
                    exec(compile(tree, "<sidecar>", "exec"), self._py_ns)
        except CodeTimeout as e:
            return f"Execution failed: {e}"
        except Exception:
            tb = traceback.format_exc()
            return f"Execution failed (sidecar traceback):\n{tb}"
        out = buf.getvalue().rstrip()

        # Forward `action` variable, then fall back to last-expr value.
        dispatched = self._py_ns.get("action", None)
        if dispatched is None:
            dispatched = result_value
        # Don't re-dispatch the same value twice on consecutive calls.
        if "action" in self._py_ns:
            self._py_ns.pop("action", None)

        if isinstance(dispatched, dict) and self.benchmark == "officebench":
            obs = self._execute_action(world, str(dispatched))
            return f"{out}\n[dispatched action] {obs}" if out else (
                f"[dispatched action] {obs}")
        return out or "(no output)"

    def _execute_action(self, world, payload: str) -> str:
        self.stats["env_steps"] += 1
        self._scan_invocations(payload)
        output = world.execute(payload)
        self.trace.append({
            "step": len(self.trace),
            "type": "run_code",
            "content": payload,
            "output": output,
            "token_usage": self._snap_token_usage(),
        })
        return output

    def _scan_invocations(self, code: str) -> None:
        """Record which retrieved skill functions are actually called in `code`. Candidates come from the memory's `_context_function_names` (names returned by the latest retrieve()); a `name(` pattern marks one invoked. No-op without memory or retrieved names; invocation is an ASI / Python-REPL concept."""
        if not code:
            return
        mem = getattr(self, "memory", None)
        names = getattr(mem, "_context_function_names", None) if mem else None
        if not names:
            return
        invoked = self._invoked_skill_names
        for name in names:
            if name in invoked:
                continue
            if re.search(rf"\b{re.escape(name)}\s*\(", code):
                invoked.add(name)

    @staticmethod
    def _init_stats() -> dict:
        return {
            "start_time": 0.0,
            "end_time": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "peak_prompt_tokens": 0,
            "env_steps": 0,
        }

    def _snap_token_usage(self) -> dict:
        return {
            "input": self.stats["total_prompt_tokens"],
            "output": self.stats["total_completion_tokens"],
            "total": self.stats["total_tokens"],
        }

    def _account_usage(self, response) -> None:
        """Charge one response's token usage into the agent's stats. Called for every attempt, success or fail, so retries count in agent-loop totals."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        total = getattr(usage, "total_tokens", 0) or 0
        self.stats["total_prompt_tokens"] += prompt
        self.stats["total_completion_tokens"] += completion
        self.stats["total_tokens"] += total
        if prompt > self.stats["peak_prompt_tokens"]:
            self.stats["peak_prompt_tokens"] = prompt

    @staticmethod
    def _assistant_message_from_response(response) -> dict:
        # Keep only the first tool call and blank the content, so batched calls and planning narration don't echo back into later turns.
        tcs = list(response.tool_calls or [])
        if tcs:
            first = tcs[0]
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": first.id, "name": first.name,
                     "arguments": first.arguments},
                ],
            }
        return {"role": "assistant", "content": response.content or ""}

    def _finalize_metrics(
        self,
        *,
        success: bool,
        completed: bool,
        difficulty: int,
        eval_extras: dict,
        history: list[dict],
    ) -> dict:
        # KramaBench reports a continuous per-task score (RAE / F1 / 0-1), surfaced as tgc_score; other benches stay binary.
        if (self.benchmark == "kramabench" and isinstance(eval_extras, dict)
                and "score" in eval_extras):
            tgc = float(eval_extras["score"])
        else:
            tgc = 1.0 if success else 0.0
        return {
            "completed": completed,
            "success": success,
            "tgc_score": tgc,
            "subtask_count": 1,
            "env_steps": self.stats["env_steps"],
            "wall_clock_seconds": self.stats["end_time"] - self.stats["start_time"],
            "total_prompt_tokens": self.stats["total_prompt_tokens"],
            "total_completion_tokens": self.stats["total_completion_tokens"],
            "peak_prompt_tokens": self.stats["peak_prompt_tokens"],
            "difficulty": difficulty,
            "eval_extras": eval_extras,
            "invoked_skill_names": sorted(
                getattr(self, "_invoked_skill_names", set()) or set()),
            "trace": self.trace,
            "final_history": copy.deepcopy(history),
        }

from typing import Callable

REACT_SYSTEM_PROMPT = {
    "appworld": """You are an AI Assistant that completes tasks autonomously by interacting with apps through their APIs in a Python REPL environment.

<APPWORLD_APIS>
Key APIs to discover available functionality:
```python
# To get a list of apps that are available to you.
print(apis.api_docs.show_app_descriptions())

# To get the list of apis under any app listed above, e.g. supervisor
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

# To get the specification of a particular api, e.g. supervisor app's show_account_passwords
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))
```
Each code execution will produce an output that you can use in subsequent calls. Using these APIs, you can generate code to interact with apps and solve the task.
</APPWORLD_APIS>

<APPWORLD_RULES>
Key instructions:

1. The example variables (email, passwords, etc.) are only for demonstration. Obtain the correct information by calling relevant APIs.
2. Only generate valid code blocks — no ```...``` or extra formatting. Thoughts should be code comments.
3. Variables from previous code blocks persist in subsequent blocks.
4. Write small chunks of code, one chunk per step. Verify before making irreversible changes.
5. The environment only has the Python standard library. OS/filesystem modules are disabled.
6. "file system" in task instructions means the file system *app* via APIs, not the actual OS.
7. Use only provided APIs, not Python packages (e.g., do NOT `import spotify`).
8. API documentation includes input arguments and output JSON schemas. Follow them exactly.
9. For paginated APIs, make sure to consider all pages.
10. Use `datetime.now()` or the phone app for current date/time.
11. For temporal requests, use proper time boundaries (00:00:00 to 23:59:59). Single timezone.
12. References to friends/family mean people in your phone's contacts list.
13. Personal info and credentials are in the "supervisor" app.
14. Once you have completed the task, call `apis.supervisor.complete_task()`. If the task asks for information (e.g., entities, numbers) as an answer, pass it via the `answer` argument: `apis.supervisor.complete_task(answer=<answer>)`. For action-only tasks (e.g., place an order, like all songs), call `apis.supervisor.complete_task()` with no arguments. The agent loop ends automatically when this call returns successfully — there is no separate `finish` tool.
15. Answers should be entities or numbers, not full sentences (e.g., `answer=10` not `answer="ten"`).
16. The task is guaranteed to be completable. DO NOT END until you complete it.
17. Make all decisions autonomously — no clarifications or confirmations needed.
18. **Prefer one tool call per turn.** Emit a single tool call and wait for its result before deciding the next step. If you do emit several calls in one turn they are ALL executed in order, but the later calls are written before you can see the earlier calls' outputs, so they often act on stale assumptions — when in doubt, emit ONE call, read the result, then continue.
</APPWORLD_RULES>""",

    "officebench": """You are an AI Assistant that completes office-productivity tasks by emitting JSON dict actions.

<AVAILABLE_APPS>
You have following apps installed in the system:
- calendar: an app to manage daily events on calendar.
- excel: an app to manipulate excel files, including reading, writing, etc.
- ocr: an app to recognize text from images.
- pdf: an app to manipulate pdf files, including format conversion and file reading.
- shell: an app to run shell commands in the system.
- word: an app to manipulate word files, including reading, writing, converting, etc.
- email: an app to manage emails, such as sending and reading emails.
- llm: an app to interact with the large language model to answer questions, generate text, etc.
</AVAILABLE_APPS>

<OFFICEBENCH_RULES>
1. You can help solve the task step by step.
2. You can interact with an operation system and use apps to solve the task.
3. You must follow the instructions and use the given json format to call APIs.
4. You can only generate one action at a time.
5. You can find files for your task in `/testbed/data`. If you don't know the filenames, please switch to shell app and call commands to list the directory.
</OFFICEBENCH_RULES>""",

    "kramabench": """You are an AI Assistant that solves end-to-end data-science questions by writing Python in a persistent REPL. Each `run_code` call executes Python in a PERSISTENT process — variables stay in memory across calls.

<KRAMABENCH_ENV>
- The question is answered by building a small data pipeline over real files: locate the right file(s), load, clean / type, filter, transform, and compute the final value.
- The domain's raw data lake is mounted read-only at `./data/` (your working directory is a scratch dir; `./data/` is a symlink to the files). Explore it with `os.listdir('data')`, `glob.glob('data/**', recursive=True)`, and `pd.read_csv` / `pd.read_excel`.
- Available libraries: `pandas`, `numpy`, `scipy`, plus the standard library (`os`, `glob`, `json`, `csv`, `re`, `math`, `pathlib`). `np` and `pd` are pre-imported.
- The `run_code` tool takes a single `code` argument (Python source). Output (stdout + tracebacks) is returned as text, truncated at ~4 KB. The kernel does NOT auto-display the last expression — use `print()`.
</KRAMABENCH_ENV>

<KRAMABENCH_RULES>
Key instructions:

1. ALWAYS inspect a file before computing on it: list `data/`, then `df.shape`, `df.columns.tolist()`, `df.dtypes`, `df.head()`. Column / sheet-name mismatches are the #1 silent failure (xlsx files often have multiple sheets — check `pd.ExcelFile(path).sheet_names`).
2. Variables PERSIST across `run_code` calls — once `df = pd.read_csv(...)` is loaded it stays. Build the pipeline incrementally and `print()` intermediate values to verify each step.
3. Read the question precisely: respect requested rounding, units, filters ("exclude missing values", "tumor samples only"), and the exact quantity asked for. Compute the literal value — do not approximate unless asked.
4. Tracebacks come back through `run_code`'s output. READ them and fix the real problem — don't shotgun-retry the same code.
5. The question is solvable from the provided files with the available libraries. DO NOT give up; if one file/column is wrong, explore others in `./data/`.
6. When you have the final answer, submit it by calling `complete_task(answer=<value>)` inside `run_code`. Pass the ACTUAL computed value (a number, string, or list) — NOT a sentence describing it, and NOT a variable you never printed. Match the expected answer type (e.g. a single rounded float for a numeric question, a Python list for a list question). This is the SOLE completion signal — there is no separate `finish` tool, and the loop ends when this call returns.
7. Make all decisions autonomously — no clarifications or confirmations needed.
8. **Prefer one tool call per turn.** Emit a single `run_code` call, read its result, then decide the next step.
</KRAMABENCH_RULES>""",
}

def _appworld_identity(task) -> dict:
    """AppWorld identity → dict consumed both by `_appworld_identity_line`
    (for prompt injection) and by subtask.py / orchestrator scaffolding
    that needs the raw fields. Keep the dict shape stable."""
    sup = task.supervisor
    return {
        "first_name": sup["first_name"],
        "last_name": sup["last_name"],
        "email": sup["email"],
        "phone_number": sup["phone_number"],
        "instruction": task.instruction,
    }


def _appworld_identity_line(task) -> str:
    f = _appworld_identity(task)
    return (
        f"My name is: {f['first_name']} {f['last_name']}. "
        f"My personal email is {f['email']} and phone number is "
        f"{f['phone_number']}."
    )


def _officebench_identity(task) -> dict:
    ctx = getattr(task, "context", {}) or {}
    return {
        "instruction": task.instruction,
        "username": ctx.get("username", "User"),
        "date": ctx.get("date", ""),
        "weekday": ctx.get("weekday", ""),
        "time": ctx.get("time", ""),
    }


def _officebench_identity_line(task) -> str:
    d = _officebench_identity(task)
    return (
        f"Today is {d['date']} ({d['weekday']}). "
        f"The current time is {d['time']}. "
        f"You are an AI assistant for user {d['username']}."
    )


def _kramabench_identity(task) -> dict:
    ctx = getattr(task, "context", {}) or {}
    return {
        "instruction": task.instruction,
        "domain": ctx.get("domain", ""),
        "answer_type": ctx.get("answer_type", ""),
        "data_listing": ctx.get("data_listing", ""),
    }


def _kramabench_identity_line(task) -> str:
    d = _kramabench_identity(task)
    atype_hint = {
        "numeric_exact": "a single number (respect any requested rounding)",
        "numeric_approximate": "a single number",
        "string_exact": "a short string",
        "list_exact": "a Python list of values",
    }.get(d["answer_type"], d["answer_type"] or "the requested value")
    return (
        f"Domain: {d['domain']}. The expected answer is {atype_hint}; pass "
        f"it to complete_task(answer=...).\n{d['data_listing']}"
    )


_IDENTITY_LINE_FN: dict[str, Callable] = {
    "appworld": _appworld_identity_line,
    "officebench": _officebench_identity_line,
    "kramabench": _kramabench_identity_line,
}


# Tool schemas.

_ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "run_code",
        "description": (
            "Execute Python code in the AppWorld REPL environment. "
            "Use print() to see outputs. Variables persist across "
            "calls. Interact with apps via the `apis` object (e.g., "
            "apis.spotify.login(...)). The environment only has access "
            "to the Python standard library; OS/filesystem modules are "
            "disabled."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python code to execute. Write small chunks of "
                        "code. Any thoughts should be put as code "
                        "comments."
                    ),
                },
            },
            "required": ["code"],
        },
    },
}


def _run_code_tool(benchmark: str, *, agent_kind: str = "non-asi") -> dict:
    """`run_code` tool variant for non-AppWorld benchmarks (officebench, kramabench).

    `agent_kind` selects whether the description / schema exposes ASI's Python-helper sidecar:

    - non-asi: single `code` field; description covers the bench's native action surface only (no Python helper text).
    - asi: adds an optional `python_code` field on officebench for the in-process sidecar where preloaded skill defs live. For kramabench `code` is already Python, so no extra field.
    """
    is_asi = agent_kind == "asi"
    python_code_desc: str | None = None
    if benchmark == "officebench":
        if is_asi:
            desc = (
                "Execute one OfficeBench action. The `code` parameter is a "
                "bare action dict (e.g. "
                "`{'app': 'shell', 'action': 'execute', 'command': 'ls /testbed/data'}`)."
                " Apps: calendar, email, excel, word, pdf, ocr, shell, "
                "llm, system. You may also pass `python_code`: Python "
                "executed in a sidecar namespace where preloaded skill "
                "`def`s persist; if it assigns to a variable named "
                "`action` (or its last expression is a dict), that dict "
                "is dispatched as the OfficeBench action."
            )
            code_desc = (
                "A bare OfficeBench action dict using single quotes "
                "(dict-literal style)."
            )
            python_code_desc = (
                "Optional. Python to run in the in-process sidecar where "
                "preloaded helper `def`s live. If it sets `action = "
                "{...}` (or its last expression is a dict), that dict is "
                "dispatched as the action."
            )
        else:
            desc = (
                "Execute one OfficeBench action. The `code` parameter is a "
                "bare action dict (e.g. "
                "`{'app': 'shell', 'action': 'execute', 'command': 'ls /testbed/data'}`)."
                " Apps: calendar, email, excel, word, pdf, ocr, shell, "
                "llm, system."
            )
            code_desc = (
                "A bare OfficeBench action dict using single quotes "
                "(dict-literal style)."
            )
    elif benchmark == "kramabench":
        desc = (
            "Execute Python code in the KramaBench data-science REPL. A "
            "persistent process holds your variables across calls; "
            "pandas / numpy / scipy are available (np, pd pre-imported). "
            "The domain's raw data lake is mounted read-only at `./data/` "
            "— explore it with os.listdir('data') / glob. Use print() to "
            "inspect (the last expression is NOT auto-displayed). When you "
            "have the final answer, call `complete_task(answer=<value>)` "
            "inside the code to finish — this is the SOLE completion "
            "signal (there is no separate `finish` tool)."
        )
        code_desc = (
            "Python code to execute. Variables persist across calls. Read "
            "files from ./data/. Submit the result with "
            "complete_task(answer=<value>). Any thoughts should be put as "
            "# comments."
        )
    else:
        raise ValueError(f"_run_code_tool: unknown benchmark {benchmark!r}")
    properties = {
        "code": {"type": "string", "description": code_desc},
    }
    if python_code_desc is not None:
        properties["python_code"] = {
            "type": "string",
            "description": python_code_desc,
        }
    return {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["code"],
            },
        },
    }


_THINK_TOOL = {
    "type": "function",
    "function": {
        "name": "think",
        "description": (
            "Record a private chain-of-thought / planning note. The thought "
            "is NOT executed and produces no observation; use it to reason "
            "before calling `run_code`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Your reasoning text.",
                },
            },
            "required": ["content"],
        },
    },
}


def _finish_tool(benchmark: str) -> dict:
    """Unified terminal `finish` tool, registered for all benchmarks.

    Signals to the agent loop that the task is complete. For benchmarks with a dedicated grader-facing completion action (AppWorld's `apis.supervisor.complete_task()`, OfficeBench's `system.finish_task`), the agent must execute the dedicated action via `run_code` first so the upstream grader records the completion; the loop's completion guard rejects `finish` if no such prior action is found in the trace and asks the model to retry.
    """
    if benchmark == "appworld":
        nudge = (
            "AppWorld: BEFORE calling `finish`, run "
            "`apis.supervisor.complete_task(answer=<answer>)` "
            "(or `apis.supervisor.complete_task()` for action-only "
            "tasks) inside a `run_code` call — that's what the grader "
            "watches. THEN call `finish` to end the loop."
        )
    elif benchmark == "officebench":
        nudge = (
            "OfficeBench: BEFORE calling `finish`, emit "
            "`{'app': 'system', 'action': 'finish_task', 'answer': "
            "<answer or 'None'>}` inside a `run_code` call so the "
            "OfficeBench runtime records the completion. THEN call "
            "`finish` to end the loop."
        )
    else:
        nudge = "Call once when the task is done."
    desc = (
        "Signal the task is complete and end the agent loop. " + nudge
    )
    return {
        "type": "function",
        "function": {
            "name": "finish",
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": (
                            "Optional answer value when the task asks "
                            "for one (entity / number / short string). "
                            "Omit for action-only tasks."
                        ),
                    },
                    "message": {
                        "type": "string",
                        "description": (
                            "Optional brief summary of what was "
                            "accomplished. Logged but not graded."
                        ),
                    },
                },
                "required": [],
            },
        },
    }


# Subagent stop signal: the executor calls this when the current subtask is satisfied so the per-subtask loop ends deterministically.
END_SUBGOAL_TOOL = {
    "type": "function",
    "function": {
        "name": "end_subgoal",
        "description": (
            "Call this when the CURRENT subtask is complete (or cannot "
            "progress). Ends the per-subtask loop and returns control to "
            "the orchestrator. Takes no arguments."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


# Per-benchmark tool lists for the linear row. There is no `finish` tool: task termination is detected from the environment's response to the bench-specific completion action (see `ReactAgent._check_task_completed`).
REACT_TOOLS: dict[str, list[dict]] = {
    "appworld":      [_ACTION_TOOL],
    "officebench":   [_run_code_tool("officebench")],
    "kramabench":    [_run_code_tool("kramabench")],
}


def _build_react_tools(benchmark: str, agent_kind: str) -> list[dict]:
    """Per-benchmark tool list parameterized by agent_kind.

    For appworld / kramabench the run_code surface is already Python, so agent_kind does not change the schema. For officebench the ASI variant exposes an extra `python_code` field routed to the in-process sidecar where preloaded helper `def`s live; the non-ASI variant exposes the bench-native action surface only.
    """
    if benchmark == "appworld":
        action_tool = _ACTION_TOOL
    else:
        action_tool = _run_code_tool(benchmark, agent_kind=agent_kind)
    # No `finish` tool: task termination is detected from the env response to the bench-specific completion action.
    return [action_tool]


# Per-benchmark argument key for the executable payload of the row's `run_code` tool; the agent dispatcher uses it to pull the code/command string out of `tool_call.arguments`.
ACTION_ARG_KEY: dict[str, str] = {
    "appworld": "code",
    "officebench": "code",
    "kramabench": "code",
}


# Action-vocabulary description fed into the AWM/ASI induction prompts so the inductor knows what counts as an action for the benchmark; the trajectory itself is not re-serialised.
ACTION_SURFACE: dict[str, str] = {
    "appworld": (
        "Actions are Python statements run in a stateful REPL — chiefly "
        "calls to the provided app APIs via `apis.<app>.<func>(...)` "
        "(plus ordinary Python). Reusable functions should wrap coherent "
        "sequences of these `apis.*` calls; take plain inputs (strings, "
        "ids, lists); and never hard-code task-specific values."
    ),
    "officebench": (
        "Actions are JSON-style dicts of the form "
        "`{\"app\": <name>, \"action\": <verb>, **app_specific_args}` "
        "executed inside a sandboxed Office environment (calendar / email "
        "/ word / excel / pdf / ocr / shell + a `system` app for "
        "switch_app and finish_task). Reusable functions should produce "
        "these dicts (assign to `action` or return it as the trailing "
        "expression); take plain inputs; never hard-code task-specific "
        "values."
    ),
    "kramabench": (
        "Actions are Python statements in a persistent REPL with pandas / "
        "numpy / scipy; the domain data lake is read-only under `./data/`. "
        "Reusable routines wrap common data-science steps (locate the "
        "right file, load, clean / type, filter, group / aggregate, "
        "compute) with plain parameters (paths, column names); never "
        "hard-code task-specific values. The final answer is submitted "
        "via `complete_task(answer=...)`."
    ),
}


def action_surface(benchmark: str) -> str:
    """The benchmark's action vocabulary description for inductors. Falls back to a generic line for unregistered benchmarks."""
    return ACTION_SURFACE.get(
        benchmark,
        "Actions are the environment's available operations; wrap common "
        "sequences into reusable, parameterised functions.")


# Tool getters.

def react_tools(benchmark: str = "appworld", *,
                agent_kind: str = "non-asi") -> list[dict]:
    """Return the canonical tool list for a benchmark.

    `agent_kind` ∈ {"non-asi", "asi"} switches the run_code schema for officebench (ASI exposes a `python_code` field for the in-process sidecar; non-ASI exposes only the native action surface). For appworld / kramabench the schema is identical.
    """
    if benchmark not in REACT_TOOLS:
        raise NotImplementedError(f"Benchmark not registered: {benchmark}")
    return _build_react_tools(benchmark, agent_kind)


def subagent_tools(benchmark: str = "appworld", *,
                   agent_kind: str = "non-asi") -> list[dict]:
    """Executor (subtask) tools: the row's normal tools minus `finish` plus `end_subgoal`. The subagent ends its scoped subtask via `end_subgoal`; task-level completion is decided by the planner."""
    base = react_tools(benchmark, agent_kind=agent_kind)
    filtered = [t for t in base
                if (t.get("function", {}) or {}).get("name") != "finish"]
    return list(filtered) + [END_SUBGOAL_TOOL]


def _identity_line_for(task, benchmark: str) -> str:
    fn = _IDENTITY_LINE_FN.get(benchmark)
    return fn(task) if fn else ""

_identity_line = _identity_line_for

def _stitched_exec_demo(benchmark: str) -> list[dict]:
    """Task-level demo derived from the subtask executor demo, so both levels show the same example. The per-subgoal user prompts and intermediate end_subgoal signals drop away; one user message carries the demo's overall task as the single subgoal, and the final end_subgoal stays as the terminal signal."""
    from skill_induction.prompts.subtask import (
        _EXEC_DEMOS, _codeify_end_subgoal, INITIAL_PROGRESS)
    demos = _codeify_end_subgoal(_EXEC_DEMOS.get(benchmark, []))
    if not demos:
        return []
    first = demos[0].get("content") or ""
    sub_m = "\n\nYour current subgoal: "
    task_m = "Overall task (for context only): "
    prog_m = "\n\nCurrent Progress:"
    if task_m not in first or prog_m not in first:
        return []
    identity = first.split(sub_m)[0] if sub_m in first else ""
    task_text = first.split(task_m, 1)[1].split(prog_m, 1)[0]
    parts = []
    if identity:
        parts += [identity, ""]
    parts += [
        f"Your current subgoal: {task_text}",
        "",
        f"{task_m}{task_text}",
        "",
        f"Current Progress: {INITIAL_PROGRESS}",
        "",
        "Proceed with execution. When the subgoal is complete, "
        "write `end_subgoal()` in a run_code call.",
    ]
    stitched = [{"role": "user", "content": "\n".join(parts)}]
    body = [dict(m) for m in demos if m.get("role") != "user"]

    def _is_end_call(m):
        if m.get("role") != "assistant":
            return False
        return any("end_subgoal" in str((tc or {}).get("arguments", {}))
                   for tc in (m.get("tool_calls") or []))

    end_idxs = [i for i, m in enumerate(body) if _is_end_call(m)]
    drop = set()
    for i in end_idxs[:-1]:
        drop.add(i)
        if i + 1 < len(body) and body[i + 1].get("role") == "tool":
            drop.add(i + 1)
    return stitched + [m for i, m in enumerate(body) if i not in drop]


def create_react_chat(task, benchmark: str = "appworld", *,
                      variant: str = "linear") -> list[dict]:
    """Build the initial chat-message history for a ReAct row.

    Returns a `list[dict]` of structured messages ready to pass into `model.chat(...)`: [system, *demo messages, user (identity + task)].
    """
    if benchmark in REACT_SYSTEM_PROMPT:
        system_text = REACT_SYSTEM_PROMPT[benchmark]

        # Both levels show the same execution example: the subtask executor demo stitched into one whole-task episode, wrapped in the same EXAMPLE boundaries.
        from skill_induction.prompts.subtask import (
            _wrap_demos_with_separators,
        )
        demo_msgs = _wrap_demos_with_separators(_stitched_exec_demo(benchmark))

        # The task-level agent is the executor special case: same user template as the subtask executor, with the whole task as its single subgoal and the fixed no-progress line.
        from skill_induction.prompts.subtask import INITIAL_PROGRESS
        identity = _identity_line_for(task, benchmark)
        parts = []
        if identity:
            parts += [identity, ""]
        parts += [
            f"Your current subgoal: {task.instruction}",
            "",
            f"Overall task (for context only): {task.instruction}",
            "",
            f"Current Progress: {INITIAL_PROGRESS}",
            "",
            "Proceed with execution. When the subgoal is complete, "
            "write `end_subgoal()` in a run_code call.",
        ]
        user_msg = {"role": "user", "content": "\n".join(parts)}
        return [{"role": "system", "content": system_text}] + demo_msgs + [user_msg]

    raise NotImplementedError(f"Benchmark not registered: {benchmark}")



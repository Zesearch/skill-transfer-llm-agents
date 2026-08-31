"""Subtask agent prompts (planner + executor + reflector)."""

from __future__ import annotations

from skill_induction.prompts.react import (
    REACT_SYSTEM_PROMPT,
    _appworld_identity_line,
    _officebench_identity_line,
    _kramabench_identity_line,
)


# Identity (shared).

# Maps a benchmark to the "USER: ..." identity line that precedes the task statement in both the planner and executor user messages (see `_plan_subgoal` / `_build_executor_history` in agents/subtask.py). For kramabench this line carries the data-lake file preview (task.context['data_listing']) via `_kramabench_identity_line`, so the planner can decide which files to open.
_IDENTITY_LINE_FN = {
    "appworld": _appworld_identity_line,
    "officebench": _officebench_identity_line,
    "kramabench": _kramabench_identity_line,
}


def identity_line(task, benchmark: str) -> str:
    fn = _IDENTITY_LINE_FN.get(benchmark)
    return fn(task) if fn else ""


# Tool schemas.

def plan_option_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "plan_option",
            "description": (
                "State your next subgoal — one focused, bounded unit of "
                "work. The executor will work ONLY on this subgoal, "
                "then summarize."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subgoal": {
                        "type": "string",
                        "description": (
                            "The next subgoal."
                        ),
                    },
                },
                "required": ["subgoal"],
            },
        },
    }


def end_subgoal_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "end_subgoal",
            "description": (
                "Signal that the current subgoal is complete."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }


# Executor role.

SUBTASK_EXECUTION_ROLE: dict[str, str] = {
    "appworld": """

<ROLE>
You are executing a specific subgoal. Focus ONLY on the stated subgoal — do not work on anything else.

Always discover APIs before using them: call show_api_descriptions(app_name) to list available APIs in an app, then show_api_doc(app_name, api_name) to read their parameters and return schemas. ALWAYS verify exact parameter names via show_api_doc before calling an API — do not guess from memory (e.g., login uses 'username', not 'email'). If skills are offered, verify their code matches current API docs before using them.

When (and only when) the subgoal is complete, end it by writing `end_subgoal()` inside a run_code call — there is NO separate end_subgoal tool. You may do real work and then call `end_subgoal()` in the SAME run_code block, or call it alone after your last action. Always do the subgoal's actual work FIRST — never end an empty subgoal. To finish the WHOLE task, run the benchmark's completion action (e.g. apis.supervisor.complete_task()) instead.
</ROLE>""",

    "officebench": """

<ROLE>
You are executing a specific subgoal. Focus ONLY on the stated subgoal — do not work on anything else.

Each `run_code` step is one OfficeBench action: either a bare action dict (e.g. `{'app': 'calendar', 'action': 'create_event', 'user': ..., 'summary': ..., 'time_start': ..., 'time_end': ...}`) or Python that assigns the action dict to `action`. Use the exact key names each app expects — sibling actions sometimes disagree (e.g. calendar `create_event` uses `user`, `list_events` uses `username`). There is NO general Python execution — run_code only emits ONE action dict (or `end_subgoal()`); you cannot compute, loop, or store variables.

CRITICAL — never extract / filter / compare / clean / reformat data in your own head or in Python. OfficeBench has an `llm` app for exactly that: `{'app': 'llm', 'action': 'complete_text', 'prompt': <one string that BOTH pastes the raw content you read AND states precisely what to produce>}`. It returns the processed result, which you then write with the relevant app. Use it whenever a step needs reasoning over content, e.g.:
- "From this OCR table <paste it>, return ONLY the rows where Buyer is 'Alice', as JSON, keeping ALL columns (Date, Car Model, Buyer, Seller, Price)."
- "From this PDF text <paste it>, extract the name, date, AND address as a contact card."
Doing the extraction/filtering yourself loses precision (wrong rows, dropped fields, hallucinated values) and fails the task. Read the raw content, hand it to the `llm` app with an exact instruction, then write the llm's output verbatim. Keep EVERY field/row the task could need — do not silently narrow to a subset.

When (and only when) the subgoal is complete, end it by writing `end_subgoal()` inside a run_code call — there is NO separate end_subgoal tool. You may do real work and then call `end_subgoal()` in the SAME run_code block, or call it alone after your last action. Always do the subgoal's actual work FIRST — never end an empty subgoal. To finish the WHOLE task, run the benchmark's completion action (e.g. `{'app': 'system', 'action': 'finish_task', 'answer': ...}`) instead.
</ROLE>""",

    "kramabench": """

<ROLE>
You are executing a specific subgoal of a data-science question. Focus ONLY on the stated subgoal — do not work on anything else.

Variables persist across `run_code` calls (a persistent Python kernel; np / pd pre-imported). The domain data lake is read-only under `./data/`. Inspect before computing: list `data/`, then `df.shape`, `df.columns.tolist()`, `df.head()` (for xlsx, check `pd.ExcelFile(p).sheet_names` first). Build the pipeline incrementally and `print()` only small intermediate values. NEVER print an entire DataFrame / array / raw file — use `.head(n)`, `.shape`, `.columns`, `.describe()`, or a small slice (output is truncated at ~4 KB and large dumps waste the step).

When (and only when) the subgoal is complete, end it by writing `end_subgoal()` inside a run_code call — there is NO separate end_subgoal tool. Always do the subgoal's actual work FIRST — never end an empty subgoal. To finish the WHOLE task (only on the final, completion subgoal), submit the answer with `complete_task(answer=<value>)` instead of `end_subgoal()`.
</ROLE>""",
}


# Planner role.

SUBTASK_PLAN_SYSTEM: dict[str, str] = {
    "appworld": """<ROLE>
You are planning the next subgoal for a task. You see the full task and current progress.

Break the task into NO MORE THAN 5 focused subgoals. Each subgoal except the final one should not be a single API call, but a coherent step (e.g., "Login and discover relevant Spotify APIs").

CRITICAL — preserve the task's constraints in EVERY subgoal. The executor only sees the subgoal text you write (not the original task), so each subgoal must carry the task's exact entities, filters, quantities, time windows, and conditions. NEVER broaden or drop them. For example, if the task is to order *the weightlifting benches* from a cart that also holds other items, the subgoal must be "remove the non-bench items, then order ONLY the weightlifting benches" — NOT "place the order for all items in the cart". Likewise keep "all/only/each", specific names, counts, and date ranges verbatim in the subgoal.

Always dedicate the final subgoal to completing the task via apis.supervisor.complete_task(). If the task really asks for some information (e.g., entities, numbers) as answer, return it as the answer argument, i.e. call `apis.supervisor.complete_task(answer=<answer>)`. For tasks that only require actions and do not require an answer (e.g., like placing an order), call `apis.supervisor.complete_task()`.

Call plan_option(subgoal) when ready. Once the executor reports that apis.supervisor.complete_task(...) was run, the agent loop ends automatically (there is no separate finish tool — keep proposing subgoals until the executor runs the completion action).
</ROLE>""",

    "officebench": """<ROLE>
You are planning the next subgoal for a task. You see the full task and current progress.

Break the task into NO MORE THAN 5 focused subgoals. Each subgoal except the final one should not be a single app command (JSON dict), but a coherent step (e.g., "Create the meeting on Bob's calendar").

OfficeBench actions are concrete app commands (JSON dicts). The ONLY way to reason over content (extract / filter / compare / clean / reformat) is the `llm` app — so a step that needs reasoning is phrased as "use the llm app to ...", never as a standalone non-action subgoal.

CRITICAL — preserve the task's constraints in EVERY subgoal. The executor only sees the subgoal text you write (not the original task), so each subgoal must carry the task's exact entities, filters, quantities, time windows, and conditions. NEVER broaden or drop them. For example, if the task is to schedule a meeting with *the marketing team* between 2-4pm, the subgoal must be "check Bob's calendar and schedule the meeting with ONLY the marketing team between 2-4pm" — NOT "schedule the meeting with whoever is available". Likewise keep "all/only/each", specific names, counts, and date ranges verbatim in the subgoal.

Always dedicate the final subgoal to completing the task via `{'app': 'system', 'action': 'finish_task', 'answer': ...}`. If the task really asks for some information (e.g., entities, numbers) as answer, return it as the answer argument, i.e. call `{'app': 'system', 'action': 'finish_task', 'answer': <answer>}`. For tasks that only require actions and do not require an answer (e.g., like scheduling a meeting), call `{'app': 'system', 'action': 'finish_task', 'answer': None}`.

Call plan_option(subgoal=<the structure above>). Once the executor reports the finish_task action in the system app was emitted, the loop ends automatically — keep proposing subgoals until the executor runs the completion action.
</ROLE>""",

    "kramabench": """<ROLE>
You are planning the next subgoal for a task. You see the full task and current progress.

Break the task into NO MORE THAN 5 focused subgoals. Each subgoal except the final one should not be a single API call, but a coherent step (e.g., "Load and inspect the relevant data file").

CRITICAL — preserve the task's constraints in EVERY subgoal. The executor only sees the subgoal text you write (not the original task), so each subgoal must carry the task's exact entities, filters, quantities, time windows, and conditions. NEVER broaden or drop them. For example, if the task is to compute the average of *the 'Price' column for all rows where 'Category' is 'Electronics'*, the subgoal must be "filter to rows where Category == 'Electronics', then compute the mean of the Price column" — NOT "compute the mean of the Price column". Likewise keep "all/only/each", specific names, counts, and date ranges verbatim in the subgoal.

Always dedicate the final subgoal to completing the task via `complete_task(answer=<value>)`. If the task really asks for some information (e.g., entities, numbers) as answer, return it as the answer argument, i.e. call `complete_task(answer=<answer>)`. For tasks that only require actions and do not require an answer (e.g., like generating a plot), call `complete_task(answer=None)`.

Call plan_option(subgoal) when ready. Once the executor reports that `complete_task(...)` was run, the agent loop ends automatically (there is no separate finish tool — keep proposing subgoals until the executor runs the completion action).
</ROLE>""",
}


# Summarizer role + tool.

SUMMARIZER_ROLE: dict[str, str] = {
    "appworld": """<ROLE>
You produce a progress summary after each subgoal. This summary is the ONLY context that persists — the execution log is discarded after this.

Your summary MUST include:
1. Completed steps and key outcomes
2. Environment variables: list every Python variable name that was ACTUALLY ASSIGNED in the execution log, with its type/content (e.g., "Env vars: access_token (str), artists (list of dicts with 'artist_id': int, 'name': str)"). Only include variables you can see being assigned (=) in the code. The next executor can only use variables if you mention them here.
3. APIs used: list which APIs were called (app.method), and ONLY note parameter gotchas or surprises (e.g., "login uses 'username' not 'email'", "access_token required as explicit parameter"). Do NOT try to reproduce full signatures — the next executor will read API docs directly.
4. Issues/gotchas: parameter name mismatches, pagination behavior, type constraints, failed approaches

Format: dense paragraph or compact bullet list. REPLACE the previous summary entirely — carry forward all relevant info from both the old progress and the new execution.

Submit your summary via the `write_progress_summary` tool. Respond with ONLY the tool call, no other text.
</ROLE>""",

    "officebench": """<ROLE>
You produce a progress summary after each subgoal. This summary is the ONLY context that persists — the execution log is discarded after this.

Your summary MUST include:
1. Completed steps and key outcomes
2. Discovered values: list every runtime value a later subgoal might need: file paths found, email addresses, event IDs, cell values, counts, computed results. Copy each `value` VERBATIM character-for-character (do not reformat / normalize / approximate). This is the planner's source of truth for grounding the next subgoal — be exhaustive.
3. APPs used: list which apps/actions were called, and ONLY note gotchas or surprises in their expected input or output (e.g., "the llm app returns a string, so I had to json.loads it to get the dict I needed for the calendar app"). Do NOT try to reproduce full schemas — the next executor will read API docs directly.
4. Issues/gotchas: short reminders to prevent repeat errors. ONLY record a guardrail for an action that produced an EXPLICIT error from the environment — NEVER second-guess an action the environment reported as successful. Cover: do-not-re-read (value is already in the Discovered values above), wrong field/key corrections, and failed approaches not to retry.

Format: dense paragraph or compact bullet list. REPLACE the previous summary entirely — carry forward all relevant info from both the old progress and the new execution.

Submit your summary via the `write_progress_summary` tool. Respond with ONLY the tool call, no other text.
</ROLE>""",

    "kramabench": """<ROLE>
You produce a progress summary after each subgoal. This summary is the ONLY context that persists — the execution log is discarded after this.

Your summary MUST include:
1. Completed steps and key outcomes (which file(s) under ./data/ were identified and loaded, which sheet, any cleaning/filtering applied)
2. Persistent Python variables still in scope with type / shape (e.g., "df (DataFrame, shape=(120, 4), columns=[...])", "cohort_ids (list of str)"). The next executor can only use variables you mention here.
3. Pandas / numpy / scipy patterns + gotchas (multi-sheet xlsx, header/skiprows, dtype coercion, NaN handling, scipy result attributes)
4. Any partial/intermediate numeric results already computed (copy verbatim) so the final subgoal can submit them

Format: dense paragraph or compact bullet list. REPLACE the previous summary entirely — carry forward all relevant info from both the old progress and the new execution.

Submit your summary via the `write_progress_summary` tool. Respond with ONLY the tool call, no other text.
</ROLE>""",
}


SUMMARIZE_USER = (
    "Write a comprehensive progress summary that REPLACES the previous "
    "progress. Carry forward all relevant information."
)


def write_progress_summary_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "write_progress_summary",
            "description": (
                "Submit the post-subgoal progress summary that will "
                "REPLACE the prior progress text. Include completed "
                "steps, env-variable names with types, APIs used (with "
                "gotchas), and any issues encountered."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": (
                            "Dense paragraph or compact bullet list. "
                            "Carries forward all relevant info from "
                            "both the old progress and the new "
                            "execution."
                        ),
                    },
                },
                "required": ["summary"],
            },
        },
    }


def build_summarizer_system_prompt(benchmark: str) -> str:
    """System prompt for the summarizer; it replaces the executor's system prompt. The benchmark's API / env context is preserved by reusing the underlying ReAct system prompt as a preamble."""
    base = REACT_SYSTEM_PROMPT.get(benchmark, "")
    role = SUMMARIZER_ROLE.get(benchmark, "")
    return base + "\n\n" + role if base else role


def build_executor_system_prompt(benchmark: str) -> str:
    """System prompt for the executor (subagent)."""
    base = REACT_SYSTEM_PROMPT.get(benchmark, "")
    role = SUBTASK_EXECUTION_ROLE.get(benchmark, "")
    return base + role if base else role


def build_planner_system_prompt(benchmark: str) -> str:
    """System prompt for the planner (orchestrator)."""
    base = REACT_SYSTEM_PROMPT.get(benchmark, "")
    role = SUBTASK_PLAN_SYSTEM.get(benchmark, "")
    return base + "\n\n" + role if base else role


# Few-shot demos.

_PLAN_DEMOS: dict[str, list[dict]] = {
    "appworld": [
        {
            "role": "user",
            "content": (
                "My name is: Glenn Smith. My personal email is "
                "glenn@gmail.com and phone number is 123456789.\n\n"
                "Task: Like all the songs from the artists I follow on "
                "Spotify."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_plan_appworld_0", "name": "plan_option",
                 "arguments": {
                     "subgoal": (
                         "Login to Spotify and discover relevant APIs "
                         "for following artists and liking songs"),
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_plan_appworld_0",
            "name": "plan_option",
            "content": (
                "Current Progress: Logged into Spotify. Env vars: "
                "access_token (str, Bearer token). APIs: "
                "apis.supervisor.show_account_passwords() → "
                "[{'account_name': str, 'password': str}]; "
                "apis.spotify.login(username: str, password: str) → "
                "{'access_token': str} — param is 'username' not "
                "'email'; apis.spotify.show_following_artists("
                "access_token: str, page_index: int=0, "
                "page_limit: int=5) → [{'artist_id': int, 'name': str, "
                "'genre': str}], empty list = end; "
                "apis.spotify.search_songs(artist_id: int, page_index: "
                "int=0, page_limit: int=5) → [{'song_id': int, "
                "'title': str}]; apis.spotify.like_song(song_id: int, "
                "access_token: str) → {'message': str}. Issues: login "
                "uses 'username' not 'email'; IDs are ints not strings;"
                " paginated APIs return empty list at end."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_plan_appworld_1", "name": "plan_option",
                 "arguments": {
                     "subgoal": (
                         "Get all followed artists on Spotify and like "
                         "all their songs"),
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_plan_appworld_1",
            "name": "plan_option",
            "content": (
                "Current Progress: Logged into Spotify, found 2 "
                "followed artists: The Weeknd (artist_id=12), Daft Punk "
                "(artist_id=47). Liked all their songs (5 + 3 = 8 "
                "total). Env vars: access_token (str), artists (list of "
                "2 dicts with 'artist_id': int, 'name': str). APIs: "
                "apis.spotify.login(username: str, password: str) → "
                "{'access_token': str} — 'username' not 'email'; "
                "apis.spotify.show_following_artists(access_token: str, "
                "page_index: int=0, page_limit: int=5) → "
                "[{'artist_id': int, 'name': str}], empty = end; "
                "apis.spotify.search_songs(artist_id: int, page_index: "
                "int=0, page_limit: int=5) → [{'song_id': int, "
                "'title': str}]; apis.spotify.like_song(song_id: int, "
                "access_token: str) → {'message': str}. Issues: IDs "
                "are ints not strings; paginated APIs return empty "
                "list at end."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_plan_appworld_2", "name": "plan_option",
                 "arguments": {
                     "subgoal": (
                         "Call apis.supervisor.complete_task() to "
                         "finish the task"),
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_plan_appworld_2",
            "name": "plan_option",
            "content": (
                "Current Progress: apis.supervisor.complete_task() was "
                "executed and the runtime returned 'Marked the active "
                "task complete.' All followed artists' songs are liked "
                "(5 + 3 = 8). Env vars: access_token (str), artists "
                "(list of 2 dicts). The agent loop ends automatically "
                "now — there is NO separate finish tool, so stop here."
            ),
        },
        {
            "role": "user",
            "content": (
                "Do NOT assume the environment is the same as in the "
                "example above. The example variables (email, "
                "passwords, tokens, etc.) were only for demonstration. "
                "Always discover the correct information via API calls."
            ),
        },
    ],
    "officebench": [
        {
            "role": "user",
            "content": (
                "Today is 2020-05-01 (Friday). The current time is "
                "10:00 AM. You are an AI assistant for user Bob.\n\n"
                "Task: Add a meeting to Bob's calendar at 5/17/2024 "
                "10:30 a.m to 11:00 a.m"
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_plan_officebench_0", "name": "plan_option",
                 "arguments": {
                     "subgoal": (
                         "Create the meeting on Bob's calendar."),
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_plan_officebench_0",
            "name": "plan_option",
            "content": (
                "Current Progress: Created Bob's calendar event via the "
                "calendar app; the only remaining step is to finish the "
                "task. Discovered values: event summary='Meeting', "
                "start='2024-05-17 10:30:00', end='2024-05-17 11:00:00'. "
                "APPs used: calendar create_event — keys are 'user' (NOT "
                "'username'), summary, time_start, time_end; time format "
                "'YYYY-MM-DD HH:MM:SS'; active app is now calendar. "
                "Issues: none."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_plan_officebench_1", "name": "plan_option",
                 "arguments": {
                     "subgoal": (
                         "Finish the task."),
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_plan_officebench_1",
            "name": "plan_option",
            "content": (
                "Current Progress: system.finish_task was emitted and the "
                "runtime returned 'Task finished'; Bob's calendar event "
                "('Meeting') was created earlier, so the task is complete. "
                "Discovered values: event summary='Meeting'. APPs used: "
                "calendar create_event, system finish_task. Issues: none. "
                "The agent loop ends automatically now — there is NO "
                "separate finish tool, so stop here."
            ),
        },
        {
            "role": "user",
            "content": (
                "Do NOT assume the environment / user / date is the "
                "same as in the example above. The example values were "
                "only for demonstration."
            ),
        },
    ],
}


def plan_demo_messages(benchmark: str) -> list[dict]:
    """Few-shot planner demo as a structured message list, wrapped in explicit separator user turns so the model treats it as a reference example rather than its own past behavior."""
    return _wrap_demos_with_separators(_PLAN_DEMOS.get(benchmark, []))


_EXEC_DEMOS: dict[str, list[dict]] = {
    "appworld": [
        {
            "role": "user",
            "content": (
                "My name is: Glenn Smith. My personal email is "
                "glenn@gmail.com and phone number is 123456789.\n\n"
                "Your current subgoal: Login to Spotify and discover "
                "relevant APIs for following artists and liking songs\n\n"
                "Overall task (for context only): Like all the songs "
                "from the artists I follow on Spotify.\n\n"
                "Current Progress: No progress yet — this is the start "
                "of the task.\n\n"
                "Proceed with execution. When the subgoal is complete, "
                "write `end_subgoal()` in a run_code call."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_exec_appworld_0", "name": "run_code",
                 "arguments": {
                     "code": (
                         "print(apis.supervisor.show_account_passwords())"
                     ),
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_exec_appworld_0",
            "name": "run_code",
            "content": (
                "[{'account_name': 'spotify', 'password': 'pass123'}, "
                "{'account_name': 'venmo', 'password': 'venmo456'}]"
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_exec_appworld_1", "name": "run_code",
                 "arguments": {
                     "code": (
                         "print(apis.api_docs.show_api_doc("
                         "app_name='spotify', api_name='login'))"
                     ),
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_exec_appworld_1",
            "name": "run_code",
            "content": """{{
 "app_name": "spotify",
 "api_name": "login",
 "parameters": [
  {{"name": "username", "type": "string", "required": true, "description": "Your account email"}},
  {{"name": "password", "type": "string", "required": true}}
 ],
 "response_schemas": {{"success": {{"access_token": "string", "token_type": "string"}}}}
}}""",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_exec_appworld_2", "name": "run_code",
                 "arguments": {
                     "code": (
                         "result = apis.spotify.login("
                         "username='glenn@gmail.com', "
                         "password='pass123')\n"
                         "access_token = result['access_token']\n"
                         "print('access_token:', access_token[:20], '...')"
                     ),
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_exec_appworld_2",
            "name": "run_code",
            "content": "access_token: eyJhbGciOiJIUzI1Ni ...",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_exec_appworld_3", "name": "run_code",
                 "arguments": {
                     "code": (
                         "print(apis.api_docs.show_api_descriptions("
                         "app_name='spotify'))"
                     ),
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_exec_appworld_3",
            "name": "run_code",
            "content": """[
 {{"name": "login", "description": "Login to your account."}},
 {{"name": "logout", "description": "Logout from your account."}},
 {{"name": "show_following_artists", "description": "Show the artists you follow."}},
 {{"name": "follow_artist", "description": "Follow an artist."}},
 {{"name": "unfollow_artist", "description": "Unfollow an artist."}},
 {{"name": "search_songs", "description": "Search for songs."}},
 {{"name": "like_song", "description": "Like a song."}},
 {{"name": "unlike_song", "description": "Unlike a song."}},
 {{"name": "show_liked_songs", "description": "Show your liked songs."}}
]""",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_exec_appworld_3_docs", "name": "run_code",
                 "arguments": {
                     "code": (
                         "print(apis.api_docs.show_api_doc("
                         "app_name='spotify', "
                         "api_name='show_following_artists'))\n"
                         "print('---')\n"
                         "print(apis.api_docs.show_api_doc("
                         "app_name='spotify', "
                         "api_name='search_songs'))\n"
                         "print('---')\n"
                         "print(apis.api_docs.show_api_doc("
                         "app_name='spotify', api_name='like_song'))"
                     ),
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_exec_appworld_3_docs",
            "name": "run_code",
            "content": """{{
 "app_name": "spotify",
 "api_name": "show_following_artists",
 "parameters": [
  {{"name": "access_token", "type": "string", "required": true}},
  {{"name": "page_index", "type": "integer", "required": false, "default": 0}},
  {{"name": "page_limit", "type": "integer", "required": false, "default": 5}}
 ],
 "response_schemas": {{"success": [{{"artist_id": 1, "name": "string", "genre": "string", "follower_count": 1}}]}}
}}
---
{{
 "app_name": "spotify",
 "api_name": "search_songs",
 "parameters": [
  {{"name": "query", "type": "string", "required": false}},
  {{"name": "artist_id", "type": "integer", "required": false}},
  {{"name": "page_index", "type": "integer", "required": false, "default": 0}},
  {{"name": "page_limit", "type": "integer", "required": false, "default": 5}}
 ],
 "response_schemas": {{"success": [{{"song_id": 1, "title": "string", "artists": [{{"id": 1, "name": "string"}}]}}]}}
}}
---
{{
 "app_name": "spotify",
 "api_name": "like_song",
 "parameters": [
  {{"name": "song_id", "type": "integer", "required": true}},
  {{"name": "access_token", "type": "string", "required": true}}
 ],
 "response_schemas": {{"success": {{"message": "string"}}}}
}}""",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_exec_appworld_4", "name": "end_subgoal",
                 "arguments": {
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_exec_appworld_4",
            "name": "end_subgoal",
            "content": "(subgoal complete — planner will summarize next)",
        },
        {
            "role": "user",
            "content": (
                "ANOTHER EXAMPLE (same task, later subgoal — has "
                "relevant experience from memory)"
            ),
        },
        {
            "role": "user",
            "content": (
                "My name is: Glenn Smith. My personal email is "
                "glenn@gmail.com and phone number is 123456789.\n\n"
                "Your current subgoal: Get all followed artists on "
                "Spotify and like all their songs\n\n"
                "Overall task (for context only): Like all the songs "
                "from the artists I follow on Spotify.\n\n"
                "Current Progress: Logged into Spotify. Env vars: "
                "access_token (str, Bearer token). APIs: "
                "apis.spotify.login(username: str, password: str) → "
                "{'access_token': str} — param is 'username' not "
                "'email'; apis.spotify.show_following_artists("
                "access_token: str, page_index: int=0, page_limit: "
                "int=5) → [{'artist_id': int, 'name': str, 'genre': "
                "str}], empty list = end; apis.spotify.search_songs("
                "artist_id: int, page_index: int=0, page_limit: int=5) "
                "→ [{'song_id': int, 'title': str, 'artists': [{'id': "
                "int, 'name': str}]}]; apis.spotify.like_song(song_id: "
                "int, access_token: str) → {'message': str}. Issues: "
                "login uses 'username' not 'email'; artist_id/song_id "
                "are ints not strings; paginated APIs return empty "
                "list at end.\n\n"
                "Experience notes from PREVIOUS tasks in this domain — "
                "HINTS, not ground truth. Their column / file / key names "
                "may not match this task; verify against the data listing, "
                "and do not let a note change this task's scope (filters, "
                "time range, entity, units, granularity). Adapt or "
                "ignore.\n\n"
                "[Login to an app and discover its APIs]\n"
                "- Procedure: apis.supervisor.show_account_passwords() "
                "→ filter by account_name → apis.api_docs.show_api_doc"
                "() to check param names → apis.<app>.login() → store "
                "access_token\n"
                "- apis.<app>.login() param is often 'username' not "
                "'email' — always check apis.api_docs.show_api_doc() "
                "first\n"
                "- Paginated APIs (page_index, page_limit) return "
                "empty list at end — loop with page_index until empty\n\n"
                "Proceed with execution. When the subgoal is complete, "
                "write `end_subgoal()` in a run_code call."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_exec_appworld_5", "name": "run_code",
                 "arguments": {
                     "code": (
                         "artists = []\n"
                         "i = 0\n"
                         "while True:\n"
                         "    page = apis.spotify.show_following_artists"
                         "(access_token=access_token, page_index=i)\n"
                         "    if not page:\n"
                         "        break\n"
                         "    artists.extend(page)\n"
                         "    i += 1\n"
                         "print(f\"Found {len(artists)} followed "
                         "artists:\")\n"
                         "for a in artists:\n"
                         "    print(f\"  artist_id={a['artist_id']}, "
                         "name={a['name']}\")"
                     ),
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_exec_appworld_5",
            "name": "run_code",
            "content": (
                "Found 2 followed artists:\n"
                "  artist_id=12, name=The Weeknd\n"
                "  artist_id=47, name=Daft Punk"
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_exec_appworld_6", "name": "run_code",
                 "arguments": {
                     "code": (
                         "for artist in artists:\n"
                         "    songs = []\n"
                         "    j = 0\n"
                         "    while True:\n"
                         "        page = apis.spotify.search_songs("
                         "artist_id=artist['artist_id'], page_index=j)\n"
                         "        if not page:\n"
                         "            break\n"
                         "        songs.extend(page)\n"
                         "        j += 1\n"
                         "    for song in songs:\n"
                         "        apis.spotify.like_song(song_id="
                         "song['song_id'], access_token=access_token)\n"
                         "    print(f\"Liked {len(songs)} songs by "
                         "{artist['name']}\")"
                     ),
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_exec_appworld_6",
            "name": "run_code",
            "content": (
                "Liked 5 songs by The Weeknd\n"
                "Liked 3 songs by Daft Punk"
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_exec_appworld_7", "name": "end_subgoal",
                 "arguments": {
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_exec_appworld_7",
            "name": "end_subgoal",
            "content": "(subgoal complete — planner will summarize next)",
        },
        {
            "role": "user",
            "content": (
                "ANOTHER EXAMPLE (same task, final subgoal — "
                "completing the task)"
            ),
        },
        {
            "role": "user",
            "content": (
                "My name is: Glenn Smith. My personal email is "
                "glenn@gmail.com and phone number is 123456789.\n\n"
                "Your current subgoal: Call "
                "apis.supervisor.complete_task() to finish the task\n\n"
                "Overall task (for context only): Like all the songs "
                "from the artists I follow on Spotify.\n\n"
                "Current Progress: Logged into Spotify. Found 2 followed artists: The Weeknd (artist_id=12), Daft Punk (artist_id=47). Liked all their songs (5 + 3 = 8 total). Env vars: access_token (str), artists (list of 2 dicts with 'artist_id': int, 'name': str). APIs: apis.spotify.login(username: str, password: str) → {{\"access_token\": str}} — 'username' not 'email'; apis.spotify.show_following_artists(access_token: str, page_index: int=0, page_limit: int=5) → [{{\"artist_id\": int, \"name\": str}}], empty = end; apis.spotify.search_songs(artist_id: int, page_index: int=0, page_limit: int=5) → [{{\"song_id\": int, \"title\": str}}]; apis.spotify.like_song(song_id: int, access_token: str) → {{\"message\": str}}. Issues: IDs are ints not strings; paginated APIs return empty list at end.\n\n"
                "Proceed with execution. When the subgoal is complete, "
                "write `end_subgoal()` in a run_code call."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_exec_appworld_8", "name": "run_code",
                 "arguments": {
                     "code": (
                         "apis.supervisor.complete_task()"
                     ),
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_exec_appworld_8",
            "name": "run_code",
            "content": "Marked the active task complete.",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_exec_appworld_9", "name": "end_subgoal",
                 "arguments": {
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_exec_appworld_9",
            "name": "end_subgoal",
            "content": "(subgoal complete — planner will summarize next)",
        },
        {
            "role": "user",
            "content": (
                "Do NOT assume the environment is the same as in the "
                "example above. The example variables (email, "
                "passwords, tokens, etc.) were only for demonstration. "
                "Always check API documentation (show_api_descriptions, "
                "show_api_doc) before calling any unfamiliar API."
            ),
        },
    ],
    "officebench": [
        {
            "role": "user",
            "content": (
                "Today is 2020-05-01 (Friday). The current time is "
                "10:00 AM. You are an AI assistant for user Bob.\n\n"
                "Your current subgoal: Create the meeting on Bob's "
                "calendar.\n\n"
                "Overall task (for context only): Add a meeting to "
                "Bob's calendar at 5/17/2024 10:30 a.m to 11:00 a.m.\n\n"
                "Current Progress: No progress yet — this is the start "
                "of the task.\n\n"
                "Proceed with execution. When the subgoal is complete, "
                "write `end_subgoal()` in a run_code call."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_exec_office_0", "name": "run_code",
                 "arguments": {
                     "code": (
                         "{'app': 'system', 'action': 'switch_app', "
                         "'target_app': 'calendar'}"
                     ),
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_exec_office_0",
            "name": "run_code",
            "content": (
                "Successfully switched to app: calendar\n"
                "\n"
                "##Current apps: calendar\n"
                "##Instruction: Choose one action from the list as the "
                "next step.\n"
                " - create a new event to a user's calendar where the "
                "time format is '%Y-%m-%d %H:%M:%S':{'app': 'calendar', "
                "'action': 'create_event', 'user': [USER_NAME], "
                "'summary': [EVENT_SUMMARY], 'time_start': "
                "[EVENT_START_TIME], 'time_end': [EVENT_END_TIME]}\n"
                " - delete an event from a user's calendar given the "
                "event summary:{'app': 'calendar', 'action': "
                "'delete_event', 'user': [USER_NAME], 'summary': "
                "[EVENT_SUMMARY]}\n"
                " - list all events from a user's calendar: "
                "{'app': 'calendar', 'action': 'list_events', "
                "'username': [USER_NAME]}\n"
                " - switch to another app among ['excel', 'ocr', 'pdf', "
                "'shell', 'word', 'email', 'llm']: {'app': 'system', "
                "'action': 'switch_app', 'target_app': "
                "[THE_APP_YOU_CHOOSE]}\n"
                " - finish the task with your answer as None if the "
                "task is not a question: {'app': 'system', 'action': "
                "'finish_task', 'answer': 'None'}\n"
                " - finish the task with your answer if the task is a "
                "question: {'app': 'system', 'action': 'finish_task', "
                "'answer': [ANSWER]}\n"
                "##Command:"
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_exec_office_1", "name": "run_code",
                 "arguments": {
                     "code": (
                         "{'app': 'calendar', 'action': "
                         "'create_event', 'user': 'Bob', "
                         "'summary': 'Meeting', "
                         "'time_start': '2024-05-17 10:30:00', "
                         "'time_end': '2024-05-17 11:00:00'}"
                     ),
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_exec_office_1",
            "name": "run_code",
            "content": (
                "Successfully create a new event to Bob's calendar.\n"
                "\n"
                "##Current apps: calendar\n"
                "##Instruction: Choose one action from the list as the "
                "next step.\n"
                " - create a new event to a user's calendar where the "
                "time format is '%Y-%m-%d %H:%M:%S':{'app': 'calendar', "
                "'action': 'create_event', 'user': [USER_NAME], "
                "'summary': [EVENT_SUMMARY], 'time_start': "
                "[EVENT_START_TIME], 'time_end': [EVENT_END_TIME]}\n"
                " - delete an event from a user's calendar given the "
                "event summary:{'app': 'calendar', 'action': "
                "'delete_event', 'user': [USER_NAME], 'summary': "
                "[EVENT_SUMMARY]}\n"
                " - list all events from a user's calendar: "
                "{'app': 'calendar', 'action': 'list_events', "
                "'username': [USER_NAME]}\n"
                " - switch to another app among ['excel', 'ocr', 'pdf', "
                "'shell', 'word', 'email', 'llm']: {'app': 'system', "
                "'action': 'switch_app', 'target_app': "
                "[THE_APP_YOU_CHOOSE]}\n"
                " - finish the task with your answer as None if the "
                "task is not a question: {'app': 'system', 'action': "
                "'finish_task', 'answer': 'None'}\n"
                " - finish the task with your answer if the task is a "
                "question: {'app': 'system', 'action': 'finish_task', "
                "'answer': [ANSWER]}\n"
                "##Command:"
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_exec_office_2", "name": "end_subgoal",
                 "arguments": {
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_exec_office_2",
            "name": "end_subgoal",
            "content": "(subgoal complete — planner will summarize next)",
        },
        {
            "role": "user",
            "content": (
                "ANOTHER EXAMPLE (same task, final subgoal — emitting "
                "system.finish_task)"
            ),
        },
        {
            "role": "user",
            "content": (
                "Today is 2020-05-01 (Friday). The current time is "
                "10:00 AM. You are an AI assistant for user Bob.\n\n"
                "Your current subgoal: Finish the task.\n\n"
                "Overall task (for context only): Add a meeting to "
                "Bob's calendar at 5/17/2024 10:30 a.m to 11:00 a.m.\n\n"
                "Current Progress: Created Bob's calendar event (summary "
                "'Meeting', 2024-05-17 10:30:00 to 11:00:00); the only "
                "remaining step is to finish the task. Discovered values: "
                "event summary='Meeting', start='2024-05-17 10:30:00', "
                "end='2024-05-17 11:00:00'. APPs used: calendar "
                "create_event — keys 'user'/summary/time_start/time_end, "
                "time 'YYYY-MM-DD HH:MM:SS'; active app is calendar. "
                "Issues: none.\n\n"
                "##Current apps: calendar\n"
                "##Instruction: Choose one action from the list as the "
                "next step.\n"
                " - create a new event to a user's calendar where the "
                "time format is '%Y-%m-%d %H:%M:%S':{'app': 'calendar', "
                "'action': 'create_event', 'user': [USER_NAME], "
                "'summary': [EVENT_SUMMARY], 'time_start': "
                "[EVENT_START_TIME], 'time_end': [EVENT_END_TIME]}\n"
                " - delete an event from a user's calendar given the "
                "event summary:{'app': 'calendar', 'action': "
                "'delete_event', 'user': [USER_NAME], 'summary': "
                "[EVENT_SUMMARY]}\n"
                " - list all events from a user's calendar: {'app': "
                "'calendar', 'action': 'list_events', 'username': "
                "[USER_NAME]}\n"
                " - switch to another app among ['excel', 'ocr', 'pdf', "
                "'shell', 'word', 'email', 'llm']: {'app': 'system', "
                "'action': 'switch_app', 'target_app': "
                "[THE_APP_YOU_CHOOSE]}\n"
                " - finish the task with your answer as None if the task "
                "is not a question: {'app': 'system', 'action': "
                "'finish_task', 'answer': 'None'}\n"
                " - finish the task with your answer if the task is a "
                "question: {'app': 'system', 'action': 'finish_task', "
                "'answer': [ANSWER]}\n"
                "##Command:\n\n"
                "Proceed with execution. When the subgoal is complete, "
                "write `end_subgoal()` in a run_code call."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_exec_office_3", "name": "run_code",
                 "arguments": {
                     "code": (
                         "{'app': 'system', 'action': 'finish_task', "
                         "'answer': 'None'}"
                     ),
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_exec_office_3",
            "name": "run_code",
            "content": "Task finished",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_exec_office_4", "name": "end_subgoal",
                 "arguments": {
                 }},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_exec_office_4",
            "name": "end_subgoal",
            "content": "(subgoal complete — planner will summarize next)",
        },
        {
            "role": "user",
            "content": (
                "Do NOT assume the environment / user / date is the "
                "same as in the example above. The example values were "
                "only for demonstration."
            ),
        },
    ],
}


def _wrap_demos_with_separators(demos: list[dict]) -> list[dict]:
    """Insert standalone separator user-turns around each demo example.

    Without strong boundaries the structured tool_call history of the demos blends into the model's perceived past actions (DeepSeek in particular pattern-matches the recent `end_subgoal` calls and ends real subgoals prematurely). Wrapping each example in explicit `=== EXAMPLE n ===` / `=== END OF EXAMPLE n ===` user messages gives the model a clear semantic boundary the chat template cannot collapse.

    Example boundaries are detected by:
      - the first user turn (start of example 1), and
      - any user turn that doesn't carry a real subgoal, namely the "ANOTHER EXAMPLE (..." separators and the trailing "Do NOT assume the environment..." line that closes the demo block.
    """
    if not demos:
        return []
    out: list[dict] = []
    ex_idx = 0
    began = False

    def open_example():
        nonlocal ex_idx, began
        ex_idx += 1
        began = True
        out.append({
            "role": "user",
            "content": (
                f"=== EXAMPLE {ex_idx} — illustrative trajectory from a "
                f"prior session. The assistant turns inside this block "
                f"are NOT your past actions; do not pattern-match them. "
                f"Read for guidance only."
            ),
        })

    def close_example():
        nonlocal began
        if began:
            out.append({
                "role": "user",
                "content": f"=== END OF EXAMPLE {ex_idx} ===",
            })
            began = False

    i = 0
    n = len(demos)
    while i < n:
        m = demos[i]
        role = m.get("role")
        content = (m.get("content") or "")
        is_legacy_marker = (
            role == "user"
            and (content.startswith("ANOTHER EXAMPLE")
                 or content.startswith("Do NOT assume"))
        )
        if is_legacy_marker:
            close_example()
            i += 1
            continue
        if not began and role == "user":
            open_example()
        out.append(dict(m))
        i += 1

    close_example()
    out.append({
        "role": "user",
        "content": (
            "=== END OF ALL EXAMPLES — the messages above are reference "
            "examples, not actions YOU previously took. Your real task "
            "starts in the NEXT user message."
        ),
    })
    return out


def _codeify_end_subgoal(demos: list[dict]) -> list[dict]:
    """Convert each structured `end_subgoal` tool-call demo turn into a `run_code` turn whose code is `end_subgoal()`.

    The executor has no end_subgoal tool; it ends a subgoal by writing `end_subgoal()` inside run_code. The demos must show exactly that: every assistant turn is a run_code call, and the subgoal-ending turn is `run_code(code="end_subgoal()")`. This keeps the demo consistent with the single-tool (run_code) surface the model actually has, so weak models stop reflexively emitting a phantom no-arg end_subgoal tool.
    """
    out: list[dict] = []
    for m in demos:
        tcs = m.get("tool_calls") or []
        if m.get("role") == "assistant" and any(
                (tc or {}).get("name") == "end_subgoal" for tc in tcs):
            tc = tcs[0]
            out.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": tc.get("id", "demo_end"),
                    "name": "run_code",
                    "arguments": {"code": "# Subgoal complete.\nend_subgoal()"},
                }],
            })
        elif m.get("role") == "tool" and m.get("name") == "end_subgoal":
            # The matching tool-ack must echo run_code (the call that wrote end_subgoal()), not a phantom `end_subgoal` tool: the prompts state there is no separate end_subgoal tool.
            nm = dict(m)
            nm["name"] = "run_code"
            out.append(nm)
        else:
            out.append(dict(m))
    return out
                                                                                
_KRAMA_SUMMARY_1 = (
    "Current Progress:\n"
    "- Completed: loaded data/water-body-testing-2018.csv into `df` (the file "
    "the question needs); confirmed the required columns exist.\n"
    "- Persistent variables: `df` (DataFrame, shape=(15102, 11); columns "
    "include 'Year', 'Beach Type Description', 'Violation').\n"
    "- Patterns/gotchas: 'Beach Type Description' values are {'Marine', "
    "'Freshwater'}; 'Violation' is {'no', 'yes'} (lowercase strings — compare "
    "to 'yes', not True/1).\n"
    "- Partial results: none computed yet."
)
_KRAMA_SUMMARY_2 = (
    "Current Progress:\n"
    "- Completed: filtered `df` to Marine beaches and computed the violation "
    "rate = (Violation=='yes').mean()*100, rounded to 2 dp.\n"
    "- Persistent variables: `df` (DataFrame, shape=(15102, 11)), `marine` "
    "(DataFrame, shape=(11876, 11)), `pct` = 5.99 (float).\n"
    "- Patterns/gotchas: 'Violation' is lowercase {'no', 'yes'} — compared to "
    "'yes', not True/1.\n"
    "- Partial results: 711 of 11876 Marine samples in violation; pct = 5.99 "
    "(the final rounded answer, ready to submit)."
)

_PLAN_DEMOS["kramabench"] = [
    {"role": "user", "content": (
        "Domain: environment. The expected answer is a single number "
        "(respect any requested rounding); pass it to "
        "complete_task(answer=...).\n"
        "The data lake `./data/` contains 37 file(s). Files for THIS task "
        "(read them at `./data/<name>`):\n\n"
        "### data/water-body-testing-2018.csv\n"
        "  columns (11): Community Code:int64, Community:object, County "
        "Code:int64, County Description:object, Year:int64, Sample "
        "Date:object, Beach Name:object, Beach Type Description:object, "
        "Organism:object, Indicator Level:int64, Violation:object\n\n"
        "Task: In 2018, what percentage of marine beach samples were in "
        "violation? Round to 2 decimal places."
    )},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "demo_plan_krama_0", "name": "plan_option", "arguments": {
            "subgoal": (
                "Load data/water-body-testing-2018.csv into df and confirm "
                "the columns needed ('Beach Type Description', "
                "'Violation').")}}]},
    {"role": "tool", "tool_call_id": "demo_plan_krama_0", "name": "plan_option",
     "content": _KRAMA_SUMMARY_1},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "demo_plan_krama_1", "name": "plan_option", "arguments": {
            "subgoal": (
                "Filter df to rows where 'Beach Type Description' == "
                "'Marine', then compute the percentage of those samples "
                "with Violation == 'yes', rounded to 2 decimal "
                "places.")}}]},
    {"role": "tool", "tool_call_id": "demo_plan_krama_1", "name": "plan_option",
     "content": _KRAMA_SUMMARY_2},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "demo_plan_krama_2", "name": "plan_option", "arguments": {
            "subgoal": (
                "Submit the final answer: call complete_task(answer=pct) "
                "reusing the computed percentage variable.")}}]},
    {"role": "tool", "tool_call_id": "demo_plan_krama_2", "name": "plan_option",
     "content": "Current Progress: complete_task(answer=5.99) executed and the "
     "runtime recorded the answer; the task is complete. The agent loop ends "
     "automatically now — there is NO separate finish tool, so stop here."},
    {"role": "user", "content": (
        "Do NOT assume the domain, file names, columns, or values are the "
        "same as in the example above. The example values were only for "
        "demonstration."
    )},
]

# The executor sees one subgoal at a time. This demo walks all three subgoals of the planner's decomposition (inspect -> compute -> submit) so the executor learns to do only the current subgoal's work, end ordinary subgoals with end_subgoal(), and submit with complete_task() only on the final subgoal. The "Current Progress" blocks reuse _KRAMA_SUMMARY_1/2 so they are byte-identical to the planner-demo summaries and the summarizer's real output.
_EXEC_DEMOS["kramabench"] = [
    {"role": "user", "content": (
        "Domain: environment. The expected answer is a single number.\n\n"
        "Your current subgoal: Load data/water-body-testing-2018.csv into df "
        "and confirm the columns needed ('Beach Type Description', "
        "'Violation').\n\n"
        "Overall task (for context only): In 2018, what percentage of marine "
        "beach samples were in violation? Round to 2 decimal places.\n\n"
        "Current Progress: No progress yet — this is the start of the task.\n\n"
        "Proceed with execution. When the subgoal is complete, write "
        "`end_subgoal()` in a run_code call."
    )},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "demo_exec_krama_0", "name": "run_code", "arguments": {
            "code": "# Inspect the 2018 file to confirm the columns I need "
            "(Beach Type Description, Violation).\n"
            "import pandas as pd\n"
            "df = pd.read_csv('data/water-body-testing-2018.csv')\n"
            "print('shape:', df.shape)\n"
            "print('columns:', df.columns.tolist())\n"
            "print('Beach Type values:', df['Beach Type Description'].unique()[:5])\n"
            "print('Violation values:', df['Violation'].unique()[:5])"}}]},
    {"role": "tool", "tool_call_id": "demo_exec_krama_0", "name": "run_code",
     "content": "shape: (15102, 11)\ncolumns: ['Community Code', 'Community', "
     "'County Code', 'County Description', 'Year', 'Sample Date', 'Beach Name', "
     "'Beach Type Description', 'Organism', 'Indicator Level', 'Violation']\n"
     "Beach Type values: ['Marine' 'Freshwater']\nViolation values: ['no' 'yes']"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "demo_exec_krama_1", "name": "end_subgoal", "arguments": {}}]},
    {"role": "tool", "tool_call_id": "demo_exec_krama_1", "name": "end_subgoal",
     "content": "(subgoal complete — planner will summarize next)"},

    {"role": "user", "content": (
        "ANOTHER EXAMPLE (same task, SECOND subgoal — compute the value; still "
        "an ordinary subgoal, so end it with end_subgoal())"
    )},
    {"role": "user", "content": (
        "Domain: environment. The expected answer is a single number.\n\n"
        "Your current subgoal: Filter df to rows where 'Beach Type "
        "Description' == 'Marine', then compute the percentage of those "
        "samples with Violation == 'yes', rounded to 2 decimal places.\n\n"
        "Overall task (for context only): In 2018, what percentage of marine "
        "beach samples were in violation? Round to 2 decimal places.\n\n"
        + _KRAMA_SUMMARY_1 + "\n\n"
        "Proceed with execution. When the subgoal is complete, write "
        "`end_subgoal()` in a run_code call."
    )},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "demo_exec_krama_2", "name": "run_code", "arguments": {
            "code": "# Filter to marine beaches, then % of samples with "
            "Violation == 'yes'; reuse `df` already in scope.\n"
            "marine = df[df['Beach Type Description'] == 'Marine']\n"
            "pct = round((marine['Violation'] == 'yes').mean() * 100, 2)\n"
            "print('marine samples:', len(marine), 'violations:', "
            "int((marine['Violation'] == 'yes').sum()))\n"
            "print('percentage:', pct)"}}]},
    {"role": "tool", "tool_call_id": "demo_exec_krama_2", "name": "run_code",
     "content": "marine samples: 11876 violations: 711\npercentage: 5.99"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "demo_exec_krama_3", "name": "end_subgoal", "arguments": {}}]},
    {"role": "tool", "tool_call_id": "demo_exec_krama_3", "name": "end_subgoal",
     "content": "(subgoal complete — planner will summarize next)"},

    {"role": "user", "content": (
        "ANOTHER EXAMPLE (same task, FINAL subgoal — submit the answer with "
        "complete_task, NOT end_subgoal)"
    )},
    {"role": "user", "content": (
        "Domain: environment. The expected answer is a single number.\n\n"
        "Your current subgoal: Submit the final answer: call "
        "complete_task(answer=pct) reusing the computed percentage "
        "variable.\n\n"
        "Overall task (for context only): In 2018, what percentage of marine "
        "beach samples were in violation? Round to 2 decimal places.\n\n"
        + _KRAMA_SUMMARY_2 + "\n\n"
        "Proceed with execution. When the subgoal is complete, write "
        "`end_subgoal()` in a run_code call."
    )},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "demo_exec_krama_4", "name": "run_code", "arguments": {
            "code": "# This is the final/submit subgoal: the work IS the "
            "submission — call complete_task(answer=pct) reusing the computed "
            "variable (do NOT recompute it).\n"
            "complete_task(answer=pct)"}}]},
    {"role": "tool", "tool_call_id": "demo_exec_krama_4", "name": "run_code",
     "content": "Execution successful. Task finished; answer recorded: 5.99."},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "demo_exec_krama_5", "name": "end_subgoal", "arguments": {}}]},
    {"role": "tool", "tool_call_id": "demo_exec_krama_5", "name": "end_subgoal",
     "content": "(subgoal complete — planner will summarize next)"},
    {"role": "user", "content": (
        "Do NOT assume the domain, file names, columns, or values are the "
        "same as in the example above. The example values were only for "
        "demonstration."
    )},
]


def subagent_demo_messages(benchmark: str) -> list[dict]:
    demos = _codeify_end_subgoal(_EXEC_DEMOS.get(benchmark, []))
    return _wrap_demos_with_separators(demos)


# Shared phrasing.

PLANNER_NEXT_SUBTASK_NUDGE = (
    "Please call plan_option to state your next subgoal."
)

INITIAL_PROGRESS = (
    "No progress yet — this is the start of the task."
)

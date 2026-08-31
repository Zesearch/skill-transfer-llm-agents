"""Code-memory (ASI) skill-induction prompts.

`CODE_SKILL_UPDATE_PROMPT[bench]` is the user-turn appended to the agent's finished trajectory (or, in the subtask agent, to the post-reflect conversation) when it is time to extract reusable Python helpers.

`skill_induction_tool()` returns the canonical OpenAI tool schema the model uses to submit one or more helper functions.

`induction_demo_messages(bench, *, variant)` returns a structured (user → assistant tool_call → tool ack) triple that the agent splices into the chat history before the real induction prompt, so the model sees one full example of the induction step before producing its own.
"""

from __future__ import annotations

import re

CODE_SKILL_UPDATE_PROMPT: dict[str, str] = {
    "appworld": '''Current function memory:
{memory_brief}

Look at the trajectory above — the work you JUST carried out. Write ONE reusable Python function capturing exactly that procedure (no more, no less), generalized for similar future work. You almost always have a procedure worth capturing — emit that one function; an empty functions list is a RARE exception.

<RULES>
- Emit EXACTLY ONE top-level function that reproduces the procedure you just carried out end-to-end. Do NOT split it into several separate top-level functions — if you need sub-steps, define them as NESTED inner functions INSIDE that one function.
- Its scope MUST MATCH what you just did: if you just carried out a whole multi-step goal, the one function performs that whole goal inline (log in, fetch, decide, act — all inside it); if you just did a single focused step, the one function is exactly that step.
- TRANSFERABLE: parametrize ALL instance-specific values (user IDs, emails, tokens, names, product IDs) as parameters. Do NOT hardcode emails, passwords, tokens, IDs, or task-specific names.
- Use full API paths with the apis. prefix: apis.app_name.api_name(param_names). Handle pagination where applicable (loop until an empty page).
- Do NOT emit a function that OVERLAPS one already in memory above — if a listed function already does this procedure (even under a different name or wording), REUSE it; never add a near-duplicate.
- Submit an EMPTY functions list ONLY in the rare case you did nothing reusable at all — i.e. the trajectory was essentially just the final apis.supervisor.complete_task() call with no real work before it. Even a single meaningful API call or lookup counts: wrap it as the one function. In every other case, emit the one function.
</RULES>

Example tool call (call the `skill_induction` tool with these JSON arguments):
{{
  "functions": [
    {{
      "name": "like_all_songs_from_followed_artists",
      "description": "Page through followed artists, then find and like every song by each",
      "implementation": "def like_all_songs_from_followed_artists(access_token):\\n    artists = []\\n    i = 0\\n    while True:\\n        page = apis.spotify.show_following_artists(page_index=i, access_token=access_token)\\n        if not page:\\n            break\\n        artists.extend(page)\\n        i += 1\\n    liked = 0\\n    for a in artists:\\n        j = 0\\n        while True:\\n            res = apis.spotify.search_songs(page_index=j, query=a['name'], artist_id=a['id'], access_token=access_token)\\n            if not res:\\n                break\\n            for s in res:\\n                apis.spotify.like_song(access_token=access_token, song_id=s['song_id'])\\n                liked += 1\\n            j += 1\\n    return liked"
    }}
  ]
}}

Respond with ONLY the function call, no other text.''',

    "officebench": '''Current function memory:
{memory_brief}

Look at the trajectory above — the work you JUST carried out. Write ONE reusable Python function capturing exactly that procedure (no more, no less), generalized for similar future work. You almost always have a procedure worth capturing — emit that one function; an empty functions list is a RARE exception.

<RULES>
- Emit EXACTLY ONE top-level function that reproduces the procedure you just carried out end-to-end. Do NOT split it into several separate top-level functions — if you need sub-steps, define them as NESTED inner functions INSIDE that one function.
- Its scope MUST MATCH what you just did: a whole multi-step goal → one function that performs the whole goal inline; a single focused step → one function that does exactly that step.
- The function RETURNS the OfficeBench JSON-dict action(s) to execute — a single action dict, or a list of them in order. Instance-specific values are PARAMETERS.
- Use the exact key names each app requires (e.g., calendar `create_event` uses `user`, `list_events` uses `username`).
- Do NOT hardcode names, dates, paths, or file contents from the current task.
- Do NOT emit a function that OVERLAPS one already in memory above — if a listed function already builds this procedure (even under a different name or wording), REUSE it; never add a near-duplicate.
- Submit an EMPTY functions list ONLY in the rare case you did nothing reusable at all — i.e. the trajectory was essentially just a switch_app + finish_task with no real action before it. Even a single meaningful action counts: wrap it as the one function. In every other case, emit the one function.
</RULES>

Example tool call (call the `skill_induction` tool with these JSON arguments):
{{
  "functions": [
    {{
      "name": "create_calendar_event_and_verify",
      "description": "Build the action sequence to create a calendar event from natural-language times, then list events to verify",
      "implementation": "def create_calendar_event_and_verify(user, summary, time_start, time_end):\\n    # time_start / time_end must be 'YYYY-MM-DD HH:MM:SS' 24-hour strings.\\n    # create_event uses key 'user'; list_events uses key 'username'.\\n    return [\\n        {{'app': 'calendar', 'action': 'create_event', 'user': user, 'summary': summary, 'time_start': time_start, 'time_end': time_end}},\\n        {{'app': 'calendar', 'action': 'list_events', 'username': user}},\\n    ]"
    }}
  ]
}}

Respond with ONLY the function call, no other text.''',

    "kramabench": '''Current function memory:
{memory_brief}

Look at the trajectory above — the work you JUST carried out. Write ONE reusable Python function capturing exactly that procedure (no more, no less), generalized for similar future data-science questions over this domain's data lake. You almost always have a procedure worth capturing — emit that one function; an empty functions list is a RARE exception.

<RULES>
- Emit EXACTLY ONE top-level function that reproduces the procedure you just carried out end-to-end. Do NOT split it into several separate top-level functions — if you need sub-steps (locate file, load sheet, clean, filter, merge, aggregate), define them as NESTED inner functions INSIDE that one function.
- Its scope MUST MATCH what you just did: a whole multi-step analysis → one function that performs that whole analysis inline; a single focused step → one function that does exactly that step.
- TRANSFERABLE: parametrize file paths, sheet / column names, group keys, filter predicates, aggregation functions. Do NOT hardcode column names, file names, or values from the current task.
- Available libraries: pandas, numpy, scipy, and the standard library. Data files are under `./data/`. Do NOT call `complete_task` inside the function (it is the agent's final action, not a reusable step).
- Write the `description` to name the concrete operation AND the kind of data it expects (e.g. "Spearman correlation between two proteomics-abundance columns"), so retrieval matches genuinely-similar future steps and is not surfaced for unrelated domains.
- Do NOT emit a function that OVERLAPS one already in memory above (even if worded differently) — REUSE it; never add a near-duplicate.
- Submit an EMPTY functions list ONLY in the rare case you did nothing reusable at all — i.e. the trajectory was essentially just submitting the answer via complete_task(answer=...) with no real data work before it. Even a single meaningful data step counts: wrap it as the one function. In every other case, emit the one function.
</RULES>

Example tool call (call the `skill_induction` tool with these JSON arguments):
{{
  "functions": [
    {{
      "name": "spearman_between_columns_in_file",
      "description": "Locate a data file by keyword, load it, and compute the Spearman correlation between two named numeric columns (dropping rows missing either).",
      "implementation": "def spearman_between_columns_in_file(root, file_keyword, col_a, col_b):\\n    import os, pandas as pd\\n    from scipy.stats import spearmanr\\n    def _find(root, kw):\\n        for dp, _, fns in os.walk(root):\\n            for fn in fns:\\n                if kw.lower() in fn.lower():\\n                    return os.path.join(dp, fn)\\n        return None\\n    path = _find(root, file_keyword)\\n    df = pd.read_csv(path)\\n    sub = df[[col_a, col_b]].dropna()\\n    return spearmanr(sub[col_a], sub[col_b]).correlation"
    }}
  ]
}}

Respond with ONLY the function call, no other text.''',
}


# Induction-prompt component ablation ladder: level 1 = minimal one-sentence instruction with no abstraction-flavored wording; level 2 = adds the one inline worked example extracted verbatim from the level-3 template (no few-shot demo splice); level 3 = full deployed prompt (long instruction + <RULES> + inline example + few-shot demo).
_EXAMPLE_RE = re.compile(r"\n\nExample tool call.*?(?=\n\nRespond with ONLY)", re.S)

_MIN_INSTRUCTION = (
    "Look at the trajectory above — the work you JUST carried out. "
    "Write ONE Python function capturing that procedure."
)


def _apply_level(template: str, level: int) -> str:
    if level >= 3:
        return template
    example = ""
    if level == 2:
        m = _EXAMPLE_RE.search(template)
        if m:
            example = m.group(0)  # includes leading blank line
    return ("Current function memory:\n{memory_brief}\n\n"
            + _MIN_INSTRUCTION + example
            + "\n\nRespond with ONLY the function call, no other text.")


def skill_induction_tool(level: int = 3) -> dict:
    """Tool schema for the code-memory induction call.

    Level-aware to match the prompt-component ablation: the field descriptions strip down in lockstep with the prose prompt, but the tool structure (field names, types, `required`) is identical across levels, so the parser is unaffected. Level 3 (default) is the deployed schema, unchanged.

    - level 1 = bare structural roles (no examples, no rules)
    - level 2 = adds concrete examples (`def <name>(...)` shape)
    - level 3 = adds abstraction/format rules (deployed; nested-inner-fn rule)
    """
    if level <= 1:
        # L1: bare structural roles; no reuse/abstraction wording, no `def <name>(...)` shape example.
        tool_desc = (
            "Submit Python function(s) from the trajectory above. Each "
            "entry has a `name` (the Python identifier), a one-line "
            "`description`, and a full `implementation` source. Submit "
            "an empty list if there is nothing to submit."
        )
        name_desc = "Python identifier for the function."
        desc_desc = "One sentence: what the function does."
        impl_desc = "Full function source code."
        arr_desc = "List of Python functions."
    elif level == 2:
        # L2: + the concrete `def <name>(...)` shape (an example, not a rule).
        tool_desc = (
            "Submit Python function(s) from the trajectory above. Each "
            "entry has a `name` (the Python identifier), a one-line "
            "`description`, and a full `implementation` source "
            "(`def <name>(...):\\n ...`). Submit an empty list if there "
            "is nothing to submit."
        )
        name_desc = "Python identifier — what `def <name>(...)` produces."
        desc_desc = "One sentence: what the function does."
        impl_desc = "Full `def <name>(...): ...` source."
        arr_desc = "List of Python functions."
    else:
        # L3: deployed schema, unchanged.
        tool_desc = (
            "Submit reusable Python helper function(s) distilled from the "
            "trajectory above. Each entry has a `name` (the Python "
            "identifier), a one-line `description` (the retrieval-match "
            "text for similar future tasks), and a full `implementation` "
            "source (`def <name>(...):\\n ...`). Submit an empty list if "
            "nothing here is reusable."
        )
        name_desc = "Python identifier — what `def <name>(...)` produces."
        desc_desc = ("One sentence: what the function does and when it "
                     "applies.")
        impl_desc = ("Full `def <name>(...): ...` source. Inline any helper "
                     "sub-functions needed.")
        arr_desc = "List of reusable Python helper functions."
    return {
        "type": "function",
        "function": {
            "name": "skill_induction",
            "description": tool_desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "functions": {
                        "type": "array",
                        "description": arr_desc,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": name_desc,
                                },
                                "description": {
                                    "type": "string",
                                    "description": desc_desc,
                                },
                                "implementation": {
                                    "type": "string",
                                    "description": impl_desc,
                                },
                            },
                            "required": [
                                "name", "description", "implementation"
                            ],
                        },
                    },
                },
                "required": ["functions"],
            },
        },
    }


# Few-shot demo args (per bench).

_DEMO_ARGS_LINEAR: dict[str, dict] = {
    "appworld": {
        "functions": [
            {
                "name": "like_all_songs_from_followed_artists",
                "description": (
                    "Page through followed artists, then find and like "
                    "every song by each."
                ),
                "implementation": (
                    "def like_all_songs_from_followed_artists("
                    "access_token):\n"
                    "    artists = []\n"
                    "    i = 0\n"
                    "    while True:\n"
                    "        page = apis.spotify.show_following_artists("
                    "page_index=i, access_token=access_token)\n"
                    "        if not page:\n"
                    "            break\n"
                    "        artists.extend(page)\n"
                    "        i += 1\n"
                    "    liked = 0\n"
                    "    for a in artists:\n"
                    "        j = 0\n"
                    "        while True:\n"
                    "            res = apis.spotify.search_songs("
                    "page_index=j, query=a['name'], artist_id=a['id'], "
                    "access_token=access_token)\n"
                    "            if not res:\n"
                    "                break\n"
                    "            for s in res:\n"
                    "                apis.spotify.like_song("
                    "access_token=access_token, song_id=s['song_id'])\n"
                    "                liked += 1\n"
                    "            j += 1\n"
                    "    return liked"
                ),
            },
        ],
    },
    "officebench": {
        "functions": [
            {
                "name": "make_calendar_create_event",
                "description": (
                    "Build the OfficeBench JSON action for creating a "
                    "calendar event."
                ),
                "implementation": (
                    "def make_calendar_create_event(user, summary, "
                    "time_start, time_end):\n"
                    "    return {'app': 'calendar', 'action': "
                    "'create_event', 'user': user, 'summary': summary, "
                    "'time_start': time_start, 'time_end': time_end}"
                ),
            },
            {
                "name": "make_finish_task",
                "description": (
                    "Build the OfficeBench `system.finish_task` action."
                ),
                "implementation": (
                    "def make_finish_task(answer='None'):\n"
                    "    return {'app': 'system', 'action': "
                    "'finish_task', 'answer': answer}"
                ),
            },
        ],
    },
}

_DEMO_ARGS_SUBTASK: dict[str, dict] = {
    "appworld": {
        "functions": [
            {
                "name": "get_followed_artists",
                "description": (
                    "Get all followed artists from Spotify with "
                    "pagination."
                ),
                "implementation": (
                    "def get_followed_artists(access_token):\n"
                    "    artists = []\n"
                    "    i = 0\n"
                    "    while True:\n"
                    "        page = apis.spotify.show_following_artists("
                    "page_index=i, access_token=access_token)\n"
                    "        if not page:\n"
                    "            break\n"
                    "        artists.extend(page)\n"
                    "        i += 1\n"
                    "    return artists"
                ),
            },
            {
                "name": "like_all_songs_by_artist",
                "description": (
                    "Search and like all songs by a given artist on "
                    "Spotify."
                ),
                "implementation": (
                    "def like_all_songs_by_artist(access_token, "
                    "artist_name, artist_id):\n"
                    "    songs = []\n"
                    "    i = 0\n"
                    "    while True:\n"
                    "        res = apis.spotify.search_songs("
                    "page_index=i, query=artist_name, "
                    "artist_id=artist_id, "
                    "access_token=access_token)\n"
                    "        if not res:\n"
                    "            break\n"
                    "        songs.extend(res)\n"
                    "        i += 1\n"
                    "    for s in songs:\n"
                    "        apis.spotify.like_song("
                    "access_token=access_token, song_id=s['song_id'])\n"
                    "    return len(songs)"
                ),
            },
        ],
    },
}


# KramaBench induction few-shot args (data-science, not the spotify fallback); used by both linear and subtask variants.
_DEMO_ARGS_LINEAR["kramabench"] = {
    "functions": [
        {
            "name": "pct_rows_equal",
            "description": "Percentage of a DataFrame's rows where a column "
            "equals a value, rounded to ndigits.",
            "implementation": "def pct_rows_equal(df, col, value, ndigits=2):\n"
            "    return round((df[col] == value).mean() * 100, ndigits)",
        },
        {
            "name": "load_csv_and_inspect",
            "description": "Read a CSV and print its shape, columns and head "
            "for inspection; returns the DataFrame.",
            "implementation": "def load_csv_and_inspect(path, n=3):\n"
            "    import pandas as pd\n    df = pd.read_csv(path)\n"
            "    print(df.shape); print(df.columns.tolist()); print(df.head(n))\n"
            "    return df",
        },
    ]
}


def induction_demo_messages(benchmark: str, *,
                            variant: str = "linear",
                            level: int = 3) -> list[dict]:
    """Structured (user, assistant tool_call, tool ack) triple spliced into the chat history before the real induction prompt.

    The demo is identical for linear and subtask: the induction prompt must be byte-for-byte the same across granularities so the comparison is honest, with granularity coming only from input scope. `variant` is accepted but ignored; both variants use the linear demo args.

    `level` is the prompt-component ablation level. The few-shot splice belongs to the full deployed prompt only (level 2's single demonstration is the inline example inside the prompt text), so this returns an empty list for levels 1 and 2 and the full triple at level 3.
    """
    del variant  # intentionally ignored, see docstring
    if level <= 2:
        return []  # L1: no demonstration; L2: inline example only (no splice)
    args = _DEMO_ARGS_LINEAR.get(benchmark) or _DEMO_ARGS_LINEAR["appworld"]
    tc_id = f"demo_induction_code_{benchmark}_0"
    demo_user = _apply_level(CODE_SKILL_UPDATE_PROMPT[benchmark], level).format(
        memory_brief="(empty — no functions yet)")
    return [
        {"role": "user", "content": demo_user},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": tc_id, "name": "skill_induction",
                 "arguments": {"functions": list(args["functions"])}},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id,
            "name": "skill_induction",
            "content": "(functions recorded)",
        },
    ]


def create_skill_update_prompt(*, memory_brief: str,
                               benchmark: str = "appworld",
                               granularity: str = "atomic",
                               level: int = 3) -> str:
    # `granularity` is intentionally ignored: both lanes share one byte-identical template, and granularity comes only from input scope. `level` selects the prompt-component ablation (see `_apply_level`); level 3 (default) is the deployed prompt.
    del granularity
    return _apply_level(
        CODE_SKILL_UPDATE_PROMPT[benchmark], level).format(
            memory_brief=memory_brief)

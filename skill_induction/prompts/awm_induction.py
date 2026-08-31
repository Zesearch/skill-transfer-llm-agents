"""Doc-memory (AWM) skill-induction prompts.

`SKILL_UPDATE_PROMPT[bench]` is the user-turn appended to the agent's finished trajectory (or, in the subtask agent, to the post-reflect conversation) when it is time to write a reusable experience note.

`skill_induction_tool()` returns the canonical OpenAI tool schema the model uses to submit its single doc-memory entry.

`induction_demo_messages(bench, *, variant)` returns a structured (user → assistant tool_call → tool ack) triple that the agent splices into the chat history before the real induction prompt, so the model sees one full example of the induction step before producing its own.
"""

from __future__ import annotations

import re

SKILL_UPDATE_PROMPT: dict[str, str] = {
    "appworld": '''Current skill memory:
{memory_brief}

Look at the trajectory above — the work you JUST carried out. Write ONE reusable experience note capturing exactly that procedure (no more, no less), generalized for similar future work.

<RULES>
- The `description` names the generalized procedure you just carried out (e.g. "Place an order on Amazon", NOT "Order Toshiba hard drive for Grant"); its scope = what you just did
- The FIRST bullet of `content` MUST be "Procedure: step1 → step2 → step3 → ..." summarizing exactly the procedure you just carried out. Remaining bullets are individual API facts, gotchas, and patterns (unordered, appendable)
- ALWAYS use full API paths with apis. prefix: apis.app_name.api_name(param_names) — NEVER write just app_name.api_name
- Bullets should record SURPRISING or NON-OBVIOUS behaviors only: parameter name mismatches (e.g., 'username' not 'email'), unexpected return key names, type constraints, edge cases. Do NOT state what an API normally does or restate its documentation
- All bullets should be specific and reusable
- Do NOT include: specific emails, passwords, token values, product IDs, names
- Do NOT add a note that OVERLAPS one already in experience memory above — if a listed note already covers this procedure (even under different wording), do NOT repeat it; at most add a genuinely new fact to the existing one
- If what you just did was trivially simple with no surprises or gotchas, submit the tool call with description="" and content="none"
</RULES>

Example tool call (call the `skill_induction` tool with these JSON arguments):
{{
  "description": "Place an order on Amazon",
  "content": "- Procedure: apis.amazon.show_cart → remove unwanted items → apis.amazon.show_payment_cards and filter expired → apis.amazon.show_addresses → apis.amazon.place_order\\n- apis.amazon.show_cart(access_token) → key is 'cart_items' NOT 'products'\\n- apis.amazon.delete_product_from_cart(access_token, product_id) — NOT 'remove_product_from_cart'\\n- apis.amazon.show_payment_cards(access_token) → 'payment_card_id' NOT 'card_id'\\n- Filter expired cards: expiry_year > now.year or (same year and expiry_month >= now.month)\\n- apis.amazon.place_order(payment_card_id, address_id, access_token) — all 3 required, return code 422 if expired\\n- If insufficient balance, try next valid card"
}}

Respond with ONLY the function call, no other text.''',

    "officebench": '''Current skill memory:
{memory_brief}

Look at the trajectory above — the work you JUST carried out. Write ONE reusable experience note capturing exactly that procedure (no more, no less), generalized for similar future work.

<RULES>
- The `description` names the generalized procedure you just carried out (e.g. "Create a calendar event from a natural-language time", NOT "Add Bob's meeting at 10:30 a.m on 5/17/2024"); its scope = what you just did
- The FIRST bullet of `content` MUST be "Procedure: step1 → step2 → step3 → ..." summarizing exactly the procedure you just carried out. Remaining bullets are individual action-shape facts, gotchas, and patterns (unordered, appendable)
- ALWAYS reference actions by `(app, action)` pair with their key arguments — never write just the action name
- Bullets should record SURPRISING or NON-OBVIOUS behaviors only: key-name mismatches between sibling actions (e.g., calendar `create_event` uses `user` but `list_events` uses `username`), time-format constraints, file-path conventions inside /testbed, format-conversion footguns. Do NOT restate normal action behavior
- Do NOT include: specific user names, dates, file paths, or task-specific strings
- Do NOT add a note that OVERLAPS one already in experience memory above — if a listed note already covers this procedure (even under different wording), do NOT repeat it; at most add a genuinely new fact to the existing one
- If what you just did was trivially simple with no surprises or gotchas, submit the tool call with description="" and content="none"
</RULES>

Example tool call (call the `skill_induction` tool with these JSON arguments):
{{
  "description": "Create a calendar event from a natural-language date/time",
  "content": "- Procedure: switch_app calendar → calendar.create_event with full ISO datetimes → calendar.list_events to verify\\n- calendar.create_event keys: user, summary, time_start, time_end (NOT username for create)\\n- calendar.list_events key: username (NOT user — different from create_event)\\n- time_start / time_end format: 'YYYY-MM-DD HH:MM:SS' (24-hour). Convert \\"10:30 a.m\\" → \\"10:30:00\\", \\"5:00 p.m\\" → \\"17:00:00\\", \\"5/17/2024\\" → \\"2024-05-17\\"\\n- Calendar files land at /testbed/calendar/<user>.ics — verify with shell ls /testbed/calendar"
}}

Respond with ONLY the function call, no other text.''',

    "kramabench": '''Current skill memory:
{memory_brief}

Look at the trajectory above — the work you JUST carried out. Write ONE reusable experience note capturing exactly that procedure (no more, no less), generalized for similar future data-science questions over this domain's data lake.

<RULES>
- The `description` names the generalized procedure you just carried out AND the kind of data it applies to (e.g. "Locate and load the correct sheet of a proteomics xlsx", NOT "Read mmc1.xlsx in the biomedical task"); its scope = what you just did, so retrieval surfaces it for genuinely-similar steps and not for unrelated domains
- The FIRST bullet of `content` MUST be "Procedure: step1 → step2 → step3 → ..." summarizing exactly the procedure you just carried out. Remaining bullets are pandas / numpy / scipy patterns, gotchas, and techniques (unordered, appendable)
- Bullets should record SURPRISING or NON-OBVIOUS behaviors only: multi-sheet xlsx (`pd.ExcelFile(p).sheet_names`), header/skiprows quirks, dtype coercion, NaN / missing-value handling, join-key normalization, scipy result attributes (`spearmanr(...).correlation`). Do NOT restate normal API behavior
- Data files are under `./data/`; the final answer is submitted via `complete_task(answer=...)` — that is the agent's action, not a note-worthy step
- Do NOT include specific file paths, column values, or numbers from this task
- Do NOT add a note that OVERLAPS one already in experience memory above — if a listed note already covers this procedure (even under different wording), do NOT repeat it; at most add a genuinely new fact to the existing one
- If what you just did was trivially simple with no surprises or gotchas, submit the tool call with description="" and content="none"
</RULES>

Example tool call (call the `skill_induction` tool with these JSON arguments):
{{
  "description": "Locate a file in the data lake and load the right table",
  "content": "- Procedure: `os.listdir('data')` / `glob.glob('data/**', recursive=True)` → pick the file whose name matches the question's entity → for xlsx check `pd.ExcelFile(<PATH>).sheet_names` then `pd.read_excel(<PATH>, sheet_name=<SHEET>)` → inspect `.shape` / `.columns` / `.head()`\\n- xlsx files frequently hold several sheets; never assume the first sheet is the data — list sheet names first\\n- Normalize join / id columns (strip, lower, dtype) before merging or filtering, or rows silently fail to match\\n- Drop rows missing the relevant columns BEFORE a correlation / aggregation, matching the question's 'exclude missing values' wording"
}}

Respond with ONLY the function call, no other text.''',
}


# Induction-prompt component ablation ladder: level 1 = minimal one-sentence instruction with no abstraction-flavored wording; level 2 = adds the one inline worked example extracted verbatim from the level-3 template (no few-shot demo splice); level 3 = full deployed prompt (long instruction + <RULES> + inline example + few-shot demo).
_EXAMPLE_RE = re.compile(r"\n\nExample tool call.*?(?=\n\nRespond with ONLY)", re.S)

_MIN_INSTRUCTION = (
    "Look at the trajectory above — the work you JUST carried out. "
    "Write ONE experience note about that work."
)


def _apply_level(template: str, level: int) -> str:
    if level >= 3:
        return template
    example = ""
    if level == 2:
        m = _EXAMPLE_RE.search(template)
        if m:
            example = m.group(0)  # includes leading blank line
    return ("Current skill memory:\n{memory_brief}\n\n"
            + _MIN_INSTRUCTION + example
            + "\n\nRespond with ONLY the function call, no other text.")


def skill_induction_tool(level: int = 3) -> dict:
    """Tool schema for the doc-memory induction call.

    Level-aware to match the prompt-component ablation: the field descriptions strip down in lockstep with the prose prompt, but the tool structure (field names, types, `required`) is identical across levels, so the parser is unaffected. Level 3 (default) is the deployed schema, unchanged.

    - level 1 = bare structural roles (no examples, no rules)
    - level 2 = adds concrete examples ('e.g. ...')
    - level 3 = adds abstraction/format rules (deployed)
    """
    if level >= 3:
        tool_desc = (
            "Submit one reusable experience note distilled from the "
            "trajectory above. Provide a one-line `description` (the "
            "retrieval-match text for similar future subgoals) and a "
            "`content` body whose first bullet starts with `Procedure:` "
            "followed by the unordered fact / gotcha bullets. If nothing "
            "here is reusable, submit description=\"\" and content=\"none\"."
        )
        desc = (
            "One-line generalized subgoal description (e.g. 'Place an "
            "order on Amazon' — not specific to this task)."
        )
        content = (
            "The reusable experience note. First bullet MUST be `- "
            "Procedure: step1 → step2 → ...`. Remaining bullets are "
            "unordered facts / gotchas / patterns. Use \"none\" if "
            "nothing is reusable."
        )
    else:
        # L1/L2: strip all abstraction/reuse-flavored wording ("reusable", "distilled", "retrieval-match ... future") from every field.
        tool_desc = (
            "Submit one experience note about the trajectory above. "
            "Provide a one-line `description` and a `content` body. "
            "If there is nothing to record, submit description=\"\" "
            "and content=\"none\"."
        )
        content = "The experience note. Use \"none\" if there is nothing to record."
        # L2 re-adds the concrete example; L1 keeps only the bare role.
        desc = ("One-line description of the note (e.g. 'Place an order on Amazon')."
                if level == 2 else "One-line description of the note.")
    return {
        "type": "function",
        "function": {
            "name": "skill_induction",
            "description": tool_desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": desc},
                    "content": {"type": "string", "description": content},
                },
                "required": ["description", "content"],
            },
        },
    }


# Few-shot demo args (per bench).

# Each entry is the {description, content} payload a model would naturally induce from the linear ReAct demo trajectory it has just seen in its initial history.
_DEMO_ARGS_LINEAR: dict[str, dict[str, str]] = {
    "appworld": {
        "description": "Look up a stored account password by app name",
        "content": (
            "- Procedure: list apps via apis.api_docs.show_app_descriptions"
            " → confirm the credential lives on the `supervisor` app via"
            " apis.api_docs.show_api_descriptions(app_name='supervisor')"
            " → read apis.api_docs.show_api_doc(app_name='supervisor',"
            " api_name='show_account_passwords') for the return shape →"
            " call apis.supervisor.show_account_passwords() once and"
            " index by app name → submit via"
            " apis.supervisor.complete_task(answer=<password>)\n"
            "- apis.supervisor.show_account_passwords() → list of"
            " {{'account_name': str, 'password': str}}; filter by"
            " account_name to find the one you want\n"
            "- Cross-service credentials live ONLY on the supervisor app"
            " — do NOT search file_system / gmail / etc. for them\n"
            "- complete_task(answer=<password>) returns the raw value as"
            " the task answer (not a sentence)"
        ),
    },
    "officebench": {
        "description": (
            "Create a calendar event from a natural-language date/time"
        ),
        "content": (
            "- Procedure: shell ls /testbed/data → switch_app calendar →"
            " calendar.create_event with full ISO datetimes →"
            " calendar.list_events to verify → finish_task\n"
            "- calendar.create_event keys: user, summary, time_start,"
            " time_end (NOT username for create)\n"
            "- calendar.list_events key: username (NOT user — different"
            " from create_event)\n"
            "- time_start / time_end format: 'YYYY-MM-DD HH:MM:SS'"
            " (24-hour). Convert \"10:30 a.m\" → \"10:30:00\","
            " \"5:00 p.m\" → \"17:00:00\", \"5/17/2024\" →"
            " \"2024-05-17\""
        ),
    },
}

# Subtask-variant demo args. For appworld only, the subagent's worked example is a multi-step Spotify task; the other benches share the Linear demo task so their args fall through to _DEMO_ARGS_LINEAR.
_DEMO_ARGS_SUBTASK: dict[str, dict[str, str]] = {
    "appworld": {
        "description": (
            "Login to an app and discover its APIs"
        ),
        "content": (
            "- Procedure: apis.supervisor.show_account_passwords() →"
            " filter by account_name → apis.api_docs.show_api_doc() to"
            " check the login signature → apis.<app>.login() → store"
            " the returned access_token → apis.api_docs"
            ".show_api_descriptions(app_name=<app>) to list the rest of"
            " the app's APIs\n"
            "- apis.<app>.login() param is often 'username' not 'email'"
            " — always check apis.api_docs.show_api_doc() first\n"
            "- Paginated APIs (page_index, page_limit) return an empty"
            " list at end — loop with page_index until empty\n"
            "- IDs returned by these APIs (artist_id, song_id, ...) are"
            " ints not strings — pass them through without quoting"
        ),
    },
}


# KramaBench induction few-shot args (data-science note, not the spotify fallback); used by both linear and subtask variants.
_DEMO_ARGS_LINEAR["kramabench"] = {
    "description": "Compute a percentage over filtered rows of a table",
    "content": (
        "- Procedure: `pd.read_csv(<PATH>)` → inspect `.shape` / `.columns` / "
        "`.head()` and the value set of the filter/label columns → filter to "
        "the target subset → `(col == value).mean() * 100` → round as the "
        "question requests\n"
        "- Check a label column's actual values (e.g. 'yes'/'no' vs True/False, "
        "case) before comparing — string labels are the #1 silent mismatch\n"
        "- Respect the question's exact scope (year, entity, beach type) in the "
        "filter; do not aggregate over rows it excludes"
    ),
}


def induction_demo_messages(benchmark: str, *,
                            variant: str = "linear",
                            level: int = 3) -> list[dict]:
    """Structured (user, assistant tool_call, tool ack) triple spliced into the chat history before the real induction prompt.

    The demo user message is the same text as the real induction prompt (built from `SKILL_UPDATE_PROMPT[bench]` with an empty memory brief), so the model pattern-matches the format it is about to be asked to produce.

    The demo is identical for linear and subtask: the induction prompt must be byte-for-byte the same across granularities so the comparison is honest, with granularity coming only from input scope. `variant` is accepted but ignored; both variants use the linear demo args.

    `level` is the prompt-component ablation level. The few-shot splice belongs to the full deployed prompt only (level 2's single demonstration is the inline example inside the prompt text), so this returns an empty list for levels 1 and 2 and the full triple at level 3.
    """
    del variant  # intentionally ignored, see docstring
    if level <= 2:
        return []  # L1: no demonstration; L2: inline example only (no splice)
    args = _DEMO_ARGS_LINEAR.get(benchmark) or _DEMO_ARGS_LINEAR["appworld"]
    tc_id = f"demo_induction_doc_{benchmark}_0"
    demo_user = _apply_level(SKILL_UPDATE_PROMPT[benchmark], level).format(
        memory_brief="(empty — no skills yet)")
    return [
        {"role": "user", "content": demo_user},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": tc_id, "name": "skill_induction",
                 "arguments": dict(args)},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id,
            "name": "skill_induction",
            "content": "(skill recorded)",
        },
    ]


def create_skill_update_prompt(*, memory_brief: str,
                               benchmark: str = "appworld",
                               granularity: str = "atomic",
                               level: int = 3) -> str:
    # `granularity` is intentionally ignored: both lanes share one byte-identical template, and granularity comes only from input scope. `level` selects the prompt-component ablation (see `_apply_level`); level 3 (default) is the deployed prompt.
    del granularity
    return _apply_level(
        SKILL_UPDATE_PROMPT[benchmark], level).format(memory_brief=memory_brief)

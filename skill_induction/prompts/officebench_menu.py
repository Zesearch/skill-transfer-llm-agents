"""OfficeBench per-turn action-menu block.

Mirrors upstream `_port/officebench/utils/policies.py::LLMPolicy.build_prompt` templates `prompt_decided_app` / `prompt_undecided_app`. Used in two places:

- runtime: `OfficeBenchWorld.execute()` appends the rendered block to every observation so the agent sees the same per-app action surface upstream OfficeBench prints between turns.
- demos: the linear / subtask few-shot demos paste the rendered block into their tool-result strings so demos and runtime show the same shape.

`agent_kind="asi"` appends a Python-function note to the instruction line (ASI's `run_code` exposes a `python_code` sidecar where helper `def`s live; a function returning a dict becomes the next action).
"""

from __future__ import annotations

# Verbatim from `_port/officebench/apps/<app>/<action>.py::DEMO`, except we restore the closing `}` that a few upstream DEMOs drop (excel.read_file, excel.delete_cell, word.read_file) so the rendered menu is valid JSON-style.
_APP_DEMOS: dict[str, list[tuple[str, str]]] = {
    "calendar": [
        ("create_event",
         "create a new event to a user's calendar where the time format is "
         "'%Y-%m-%d %H:%M:%S':{'app': 'calendar', 'action': 'create_event', "
         "'user': [USER_NAME], 'summary': [EVENT_SUMMARY], "
         "'time_start': [EVENT_START_TIME], 'time_end': [EVENT_END_TIME]}"),
        ("delete_event",
         "delete an event from a user's calendar given the event summary:"
         "{'app': 'calendar', 'action': 'delete_event', 'user': [USER_NAME], "
         "'summary': [EVENT_SUMMARY]}"),
        ("list_events",
         "list all events from a user's calendar: "
         "{'app': 'calendar', 'action': 'list_events', "
         "'username': [USER_NAME]}"),
    ],
    "excel": [
        ("read_file",
         "read the excel file to see the existing contents: "
         "{'app': 'excel', 'action': 'read_file', "
         "'file_path': [THE_PATH_TO_THE_EXCEL_FILE]}"),
        ("set_cell",
         "write text to a cell in the excel file (index starts from 1): "
         "{'app': 'excel', 'action': 'set_cell', "
         "'file_path': [THE_PATH_TO_THE_EXCEL_FILE], "
         "'row_idx': [THE_ROW_INDEX], 'column_idx': [THE_COLUMN_INDEX], "
         "'text': [THE_TEXT_TO_WRITE]}"),
        ("delete_cell",
         "delete text in a cell of the excel file (index starts from 1, "
         "delete means set empty): {'app': 'excel', 'action': "
         "'delete_cell', 'file_path': [THE_PATH_TO_THE_EXCEL_FILE], "
         "'row_idx': [THE_ROW_INDEX], 'column_idx': [THE_COLUMN_INDEX]}"),
        ("create_new_file",
         "create a new excel file: {'app': 'excel', 'action': "
         "'create_new_file', "
         "'file_path': [THE_PATH_TO_THE_NEW_EXCEL_FILE]}"),
        ("convert_to_pdf",
         "convert an excel document to a pdf: {'app': 'excel', 'action': "
         "'convert_to_pdf', 'excel_file_path': [THE_PATH_TO_THE_EXCEL_FILE], "
         "'pdf_file_path': [THE_PATH_TO_THE_PDF_FILE]}"),
    ],
    "ocr": [
        ("recognize_file",
         "recognize the text from an image file: {'app': 'ocr', 'action': "
         "'recognize_file', 'file_path': [THE_PATH_TO_THE_IMAGE_FILE]}"),
    ],
    "pdf": [
        ("convert_to_image",
         "convert a pdf file to an image file: {'app': 'pdf', 'action': "
         "'convert_to_image', 'pdf_file_path': [THE_PATH_TO_THE_PDF_FILE], "
         "'image_file_path': [THE_PATH_TO_THE_IMAGE_FILE]}"),
        ("convert_to_word",
         "convert a pdf file to a word file: {'app': 'pdf', 'action': "
         "'convert_to_word', 'pdf_file_path': [THE_PATH_TO_THE_PDF_FILE], "
         "'word_file_path': [THE_PATH_TO_THE_WORD_FILE]}"),
        ("read_file",
         "read a pdf file: {'app': 'pdf', 'action': 'read_file', "
         "'pdf_file_path': [THE_PATH_TO_THE_PDF_FILE]}"),
    ],
    "shell": [
        # Our env routes shell through `action`='execute' with a `command` field (see `OfficeBenchWorld._dispatch_action`); the upstream `command.py` DEMO uses `action`='command', which our env does not accept, so we surface the accepted form here.
        ("execute",
         "run a shell command: {'app': 'shell', 'action': 'execute', "
         "'command': [THE_COMMAND_YOU_WISH_TO_RUN]}"),
    ],
    "word": [
        ("convert_to_pdf",
         "convert a word document to a pdf: {'app': 'word', 'action': "
         "'convert_to_pdf', 'word_file_path': [THE_PATH_TO_THE_WORD_FILE], "
         "'pdf_file_path': [THE_PATH_TO_THE_PDF_FILE]}"),
        ("create_new_file",
         "create a new word file: {'app': 'word', 'action': "
         "'create_new_file', 'file_path': [THE_PATH_TO_THE_NEW_WORD_FILE]}"),
        ("read_file",
         "read the content of the word file: {'app': 'word', 'action': "
         "'read_file', 'file_path': [THE_PATH_TO_THE_WORD_FILE]}"),
        ("write_to_file",
         "write text to a word file: {'app': 'word', 'action': "
         "'write_to_file', 'file_path': [THE_PATH_TO_THE_WORD_FILE], "
         "'contents': [THE_CONTENTS_YOU_WISH_TO_WRITE]}"),
    ],
    "email": [
        ("send_email",
         "Send an email to a recipient: {'app': 'email', 'action': "
         "'send_email', 'sender': [SENDER], 'recipient': [RECIPIENT], "
         "'subject': [SUBJECT], 'content': [CONTENT]}"),
        ("list_emails",
         "List emails for a given username: {'app': 'email', 'action': "
         "'list_emails', 'username': [USER_NAME]}"),
        ("read_email",
         "Read a user's email by the given Email ID: {'app': 'email', "
         "'action': 'read_email', 'username': [USERNAME], "
         "'email_id': [EMAIL_ID]}"),
    ],
    "llm": [
        ("complete_text",
         "Query an LLM model for an answer to a given prompt: "
         "{'app': 'llm', 'action': 'complete_text', 'prompt': [PROMPT]}"),
    ],
}

ALL_APPS: list[str] = list(_APP_DEMOS.keys())


def _instruction_suffix(agent_kind: str) -> str:
    if agent_kind == "asi":
        return (
            " (or call a Python function whose returned dict becomes the "
            "next action)"
        )
    return ""


def format_action_menu(
    current_app: str | None,
    *,
    agent_kind: str = "non-asi",
    available_apps: list[str] | None = None,
) -> str:
    """Render the OfficeBench per-turn action menu.

    `current_app=None` → `prompt_undecided_app` shape: lists available apps and the `switch_app` instruction only.

    `current_app` set → `prompt_decided_app` shape: lists that app's actions, the `switch_app` line (with `current_app` excluded), and the two `finish_task` lines.

    `agent_kind="asi"` appends a Python-function note to the instruction. `available_apps` defaults to all 8 registered apps.
    """
    apps = list(available_apps if available_apps is not None else ALL_APPS)
    suffix = _instruction_suffix(agent_kind)

    if current_app is None:
        lines = [
            f"##Available apps: {apps}",
            f"##Instruction: Choose an app to switch to{suffix}.",
            " - choose an app from the available apps: "
            "{'app': 'system', 'action': 'switch_app', "
            "'target_app': [THE_APP_YOU_CHOOSE]}",
            "##Command:",
        ]
        return "\n".join(lines)

    others = [a for a in apps if a != current_app]
    lines = [
        f"##Current apps: {current_app}",
        "##Instruction: Choose one action from the list as the next step"
        f"{suffix}.",
    ]
    for _, demo in _APP_DEMOS.get(current_app, []):
        lines.append(f" - {demo}")
    lines.append(
        f" - switch to another app among {others}: "
        "{'app': 'system', 'action': 'switch_app', "
        "'target_app': [THE_APP_YOU_CHOOSE]}"
    )
    lines.append(
        " - finish the task with your answer as None if the task is not a "
        "question: {'app': 'system', 'action': 'finish_task', "
        "'answer': 'None'}"
    )
    lines.append(
        " - finish the task with your answer if the task is a question: "
        "{'app': 'system', 'action': 'finish_task', 'answer': [ANSWER]}"
    )
    lines.append("##Command:")
    return "\n".join(lines)

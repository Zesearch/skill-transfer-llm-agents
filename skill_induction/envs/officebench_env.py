"""OfficeBench adapter: actions run inside an apptainer container with bind-mounted `/testbed` (per-task host scratch) and `/apps` (vendored OfficeBench apps).

The vendored OfficeBench scripts (`_port/officebench/apps/**`) hard-code absolute paths like `/testbed/calendar/{user}.ics` and `/apps/word_app/word_read_file.py`. Rather than patching every script, we bind-mount those exact paths into a small ubuntu SIF and exec inside it, so the scripts run unmodified. Python deps live in the host conda env `officebench-env`; we bind-mount the user HOME so the conda python (with its RPATHs intact) works inside the container.

Why apptainer instead of a plain host subprocess: the upstream scripts hard-code `/testbed/` and `/apps/` at filesystem root, paths we can't create on the host without privileges. Bind-mounts give us those roots inside the container with zero source modifications.

Why apptainer instead of Docker: the host has no Docker daemon; apptainer is available.

Action format (verbatim from upstream `utils/env.py::exec_action`)::

    "{'app': 'calendar', 'action': 'create_event', 'user': 'Bob', ...}"
    "{'app': 'shell',    'action': 'execute', 'command': 'ls'}"
    "{'app': 'system',   'action': 'switch_app', 'target_app': 'word'}"
    "{'app': 'system',   'action': 'finish_task', 'answer': '42'}"

Per the `World` contract: errors include the substring `Execution failed`; `evaluate_run()` is the official success signal, used only for logging.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from skill_induction.envs.base import World
from skill_induction.paths import (
    OFFICEBENCH_PORT_ROOT,
    OFFICEBENCH_SANDBOX_SIF,
    OFFICEBENCH_SCRATCH_ROOT,
)
from skill_induction.prompts.officebench_menu import format_action_menu
from skill_induction.runtime_limits import CodeTimeout, time_limit


# Make the externally-vendored OfficeBench code importable (it lives outside the repo, not under the `skill_induction` package).
_EXTERNAL_PORT_PARENT = str(OFFICEBENCH_PORT_ROOT.parent)
if _EXTERNAL_PORT_PARENT not in sys.path:
    sys.path.insert(0, _EXTERNAL_PORT_PARENT)


# Paths.

# Vendored OfficeBench code lives outside the repo at `<EXTERNAL_ROOT>/_port/officebench/` (defaults in skill_induction.paths).
OFFICEBENCH_APPS_DIR = OFFICEBENCH_PORT_ROOT / "apps"
OFFICEBENCH_UTILS_DIR = OFFICEBENCH_PORT_ROOT / "utils"
OFFICEBENCH_TASKS_DIR = OFFICEBENCH_PORT_ROOT / "tasks"

OFFICEBENCH_CONTAINER_PYTHON = os.path.expanduser(os.environ.get(
    "OFFICEBENCH_CONTAINER_PYTHON",
    "~/miniconda3/envs/officebench-env/bin/python3",
))


# No-op server.

class OfficeBenchNoopServer:
    """OfficeBench has no HTTP server (actions run as apptainer-exec subprocesses); satisfies the runner's `ServerCls(**server_kwargs)` contract by returning an empty URL string."""

    def __init__(self, **kwargs):
        self.base_url = ""

    def __enter__(self) -> str:
        return self.base_url

    def __exit__(self, *exc) -> None:
        return None


# Task handle.

class _OfficeBenchTaskHandle:
    def __init__(self, task_id: str, config: dict, group_key: str):
        self.task_id = task_id
        self.id = task_id
        self.instruction = config["task"]
        self.group_key = group_key
        self.context = {
            "username": config.get("username", "User"),
            "date": config.get("date", ""),
            "weekday": config.get("weekday", ""),
            "time": config.get("time", ""),
        }
        self._evaluation = config.get("evaluation", [])


# World.

class OfficeBenchWorld(World):
    """Per-task OfficeBench environment.

    `task_id` may be `1-1/0`, `1-1__0`, or `tasks/1-1/subtasks/0.json`.
    """

    def __init__(
        self,
        task_id: str,
        experiment_name: str = "",
        *,
        server_url: str = "",
        data_root: str | Path | None = None,
        group_key: str | None = None,
        scratch_root: Path | None = None,
    ):
        self.task_id = self._normalize_task_id(task_id)
        self.experiment_name = experiment_name
        self._group_key = group_key or self._derive_group_key(self.task_id)
        self.scratch_root = Path(scratch_root or OFFICEBENCH_SCRATCH_ROOT)
        self.task: _OfficeBenchTaskHandle | None = None
        self.workdir = "/testbed/data"
        self.current_app: str | None = None
        self.history: list[tuple] = []
        self._testbed_host: Path | None = None
        # Injected by the agent (Linear / Subtask) before run: a callable `prompt -> str` backed by the agent's own model. The upstream `llm` app shells out to OpenAI GPT-3.5 (dead here), so llm actions route to this instead; None means the llm app is unavailable.
        self.llm_fn = None
        # Set by the agent (Linear / Subtask) before run; switches the `##Instruction` line in `format_action_menu` between dict-only and dict-or-python-function variants. Default matches the upstream OfficeBench prompt (dict-only).
        self.agent_kind: str = "non-asi"

    @staticmethod
    def _normalize_task_id(tid: str) -> str:
        tid = tid.replace(".json", "")
        if "tasks/" in tid:
            tid = tid.split("tasks/", 1)[1]
        if "subtasks/" in tid:
            bucket, sub = tid.split("/subtasks/")
            return f"{bucket}/{sub}"
        if "__" in tid:
            bucket, sub = tid.rsplit("__", 1)
            return f"{bucket}/{sub}"
        return tid

    @staticmethod
    def _derive_group_key(task_id: str) -> str:
        """`1-1/0` → `1app`, `2-5/3` → `2app`, `3-12/0` → `3app`."""
        prefix = task_id.split("-", 1)[0]
        try:
            return f"{int(prefix)}app"
        except ValueError:
            return "unknown"

    def _config_path(self) -> Path:
        bucket, sub = self.task_id.split("/", 1)
        return OFFICEBENCH_TASKS_DIR / bucket / "subtasks" / f"{sub}.json"

    def __enter__(self) -> "OfficeBenchWorld":
        cfg_path = self._config_path()
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"OfficeBench task config not found: {cfg_path}. Vendor the "
                f"upstream tasks/ tree under {OFFICEBENCH_TASKS_DIR}."
            )
        if not OFFICEBENCH_SANDBOX_SIF.exists():
            raise FileNotFoundError(
                f"Apptainer sandbox SIF missing: {OFFICEBENCH_SANDBOX_SIF}. "
                "Build it with `apptainer build <path> docker://ubuntu:24.04`."
            )
        with open(cfg_path) as f:
            config = json.load(f)
        self.task = _OfficeBenchTaskHandle(self.task_id, config, self._group_key)

        # Per-cell, per-task scratch testbed, fresh each __enter__. The path must include `experiment_name` (which carries stage+agent+task); otherwise parallel cells running the same task share one directory and one cell's rmtree races another's copytree.
        safe_tid = self.task_id.replace("/", "__")
        cell_tag = (self.experiment_name or "cell").replace("/", "__")
        self._testbed_host = (
            self.scratch_root / cell_tag / safe_tid / "testbed"
        )
        if self._testbed_host.exists():
            shutil.rmtree(self._testbed_host, ignore_errors=True)
        for sub in ("data", "emails", "calendar"):
            (self._testbed_host / sub).mkdir(parents=True, exist_ok=True)

        # Task-persistent Python REPL namespace for code-skill agents; helper `def`s survive across steps.
        self._py_namespace = {}

        # Seed task-supplied starter files if the upstream task ships any.
        bucket = self.task_id.split("/")[0]
        template = OFFICEBENCH_TASKS_DIR / bucket / "testbed"
        if template.exists():
            for item in template.iterdir():
                dst = self._testbed_host / item.name
                if item.is_dir():
                    shutil.copytree(item, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dst)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Remove this task's scratch testbed (evaluate_run() already ran before __exit__, so the score is captured); otherwise per-task testbeds accumulate and exhaust the filesystem inode quota, which blocks all writes and hangs runs. Set SKILL_INDUCTION_KEEP_SCRATCH=1 to keep them for debugging.
        if os.environ.get("SKILL_INDUCTION_KEEP_SCRATCH") in (
                "1", "true", "True"):
            return None
        tb = getattr(self, "_testbed_host", None)
        if tb is not None:
            shutil.rmtree(tb, ignore_errors=True)
        return None

    # Action dispatch.

    @staticmethod
    def _escape_raw_newlines_in_strings(s: str) -> str:
        """Escape literal newlines/tabs/CRs inside quoted strings so eval/json can parse the dict.

        Models routinely emit an action dict whose free-text value (e.g. an `llm` complete_text `prompt` or a `contents` field) contains real line breaks, which makes `eval`/`json.loads` raise 'unterminated string literal'. This walks the text tracking quote state and replaces only the in-string control chars; structure outside strings is untouched.
        """
        out = []
        in_str = False
        q = ""
        esc = False
        for ch in s:
            if in_str:
                if esc:
                    out.append(ch); esc = False; continue
                if ch == "\\":
                    out.append(ch); esc = True; continue
                if ch == q:
                    out.append(ch); in_str = False; continue
                if ch == "\n":
                    out.append("\\n"); continue
                if ch == "\r":
                    out.append("\\r"); continue
                if ch == "\t":
                    out.append("\\t"); continue
                out.append(ch)
            else:
                if ch in ("'", '"'):
                    in_str = True; q = ch
                out.append(ch)
        return "".join(out)

    def _parse_action(self, payload: str) -> Optional[dict]:
        """Try to read the payload as a bare action dict (the form linear / subtask agents emit). Returns the dict or None.

        Robust to action dicts whose string values contain real line breaks (a frequent model failure mode for `llm` complete_text prompts): on the first parse failure we retry with in-string newlines/tabs escaped.
        """
        s = payload.strip()
        if "{" in s and "}" in s:
            s = s[s.find("{") : s.rfind("}") + 1]
        for candidate in (s, self._escape_raw_newlines_in_strings(s)):
            try:
                return eval(candidate, {"__builtins__": {}}, {})
            except Exception:
                try:
                    return json.loads(candidate)
                except Exception:
                    continue
        # Tolerant regex fallback: the remaining failures are action dicts whose free-text value (an `llm` `prompt`, a word `contents`, a shell `command`) has broken inner quote/backslash escaping that neither eval nor json can parse. Extract app+action and the one free-text field by position so a malformed prompt doesn't zero out the whole action.
        import re as _re
        m_app = _re.search(r"""['"]app['"]\s*:\s*['"]([^'"]+)['"]""", s)
        m_act = _re.search(r"""['"]action['"]\s*:\s*['"]([^'"]+)['"]""", s)
        if m_app and m_act:
            d = {"app": m_app.group(1), "action": m_act.group(1)}
            # other simple scalar string keys (user, file_path, target_app…)
            for mk in _re.finditer(
                    r"""['"](\w+)['"]\s*:\s*['"]([^'"]*)['"]""", s):
                k, v = mk.group(1), mk.group(2)
                if k not in d:
                    d[k] = v
            # The one big free-text field, grabbed positionally (tolerant of any inner quoting) from its key to the closing brace.
            for key in ("prompt", "contents", "content", "command",
                        "text", "query", "answer", "code"):
                mk = _re.search(r"""['"]%s['"]\s*:\s*""" % key, s)
                if not mk:
                    continue
                val = s[mk.end():].rstrip()
                if val.endswith("}"):
                    val = val[:-1].rstrip()
                val = val.strip().strip(",").strip()
                if len(val) >= 2 and val[0] in "'\"" and val[-1] == val[0]:
                    val = val[1:-1]
                val = (val.replace("\\n", "\n").replace("\\t", "\t")
                          .replace('\\"', '"').replace("\\'", "'"))
                d[key] = val
                break
            return d
        return None

    def _run_python(self, code: str):
        """Run `code` in the task-persistent Python REPL namespace and return the OfficeBench action dict it produced.

        This is the path code-skill (ASI) agents use: they `def` helper functions and call them, and the returned dict is the action. Convention: the code assigns the action to a variable named `action` (or ends with a bare dict expression). Helper `def`s persist across steps.

        Returns (kind, value):
          - ('action', dict): an action dict to dispatch
          - ('error', str): the code raised; str is the message
          - ('noact', str): code ran but produced no action dict
        """
        import ast
        ns = self._py_namespace
        # Clear any stale `action` from a previous step so we only dispatch this step's action (helper `def`s still persist).
        ns.pop("action", None)
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ("error", f"Execution failed: Python syntax error: {e}")
        # If the last statement is a bare expression, capture its value.
        last_expr_val = None
        body = list(tree.body)
        tail_expr = None
        if body and isinstance(body[-1], ast.Expr):
            tail_expr = body.pop()
        try:
            with time_limit():
                if body:
                    exec(compile(ast.Module(body=body, type_ignores=[]),
                                 "<officebench_skill>", "exec"), ns)
                if tail_expr is not None:
                    last_expr_val = eval(compile(
                        ast.Expression(tail_expr.value),
                        "<officebench_skill>", "eval"), ns)
        except CodeTimeout as e:
            return ("error", f"Execution failed: {e}")
        except SystemExit as e:
            # Agent-written code may call sys.exit()/exit(); SystemExit is not an Exception, and uncaught it kills the whole runner silently.
            return ("error", f"Execution failed: code called sys.exit({e.code!r})")
        except Exception as e:
            return ("error", f"Execution failed: Python raised {e!r}")
        # Prefer an explicit `action` variable; else the last expression.
        act = ns.get("action")
        if not (isinstance(act, dict) and "app" in act):
            act = last_expr_val
        if isinstance(act, dict) and "app" in act:
            return ("action", act)
        return ("noact",
                "Execution failed: the Python code did not yield an "
                "OfficeBench action. Assign the action dict to a variable "
                "named `action`, e.g. "
                "`action = {'app': 'shell', 'action': 'execute', "
                "'command': 'ls /testbed/data'}`.")

    def preload_skills(self, srcs) -> None:
        """Exec reusable skill source into the persistent REPL namespace so retrieved code skills are callable by name in later `run_code` steps. Best-effort: a skill that fails to compile/exec is skipped silently."""
        for src in srcs or []:
            if not src or not str(src).strip():
                continue
            try:
                exec(compile(str(src), "<officebench_skill>", "exec"),
                     self._py_namespace)
            except Exception:
                pass

    def execute(self, payload: str) -> str:
        # Two accepted payload forms, auto-detected: a bare action dict (linear / subtask agents) or Python code that yields an `action` dict (code-skill / ASI agents).
        action = self._parse_action(payload)
        if not (isinstance(action, dict) and "app" in action):
            kind, val = self._run_python(payload)
            if kind == "action":
                action = val
            else:
                return self._with_action_menu(val, action_dict=None)
        out = self._dispatch_action(action)
        return self._with_action_menu(out, action_dict=action)

    def _with_action_menu(
        self, observation: str, *, action_dict: dict | None
    ) -> str:
        """Append the upstream OfficeBench per-turn action menu after each observation; suppressed once `system.finish_task` is emitted, because no next action is expected."""
        if (action_dict
                and action_dict.get("app") == "system"
                and action_dict.get("action") in ("finish_task", "got_stuck")):
            return observation
        menu = format_action_menu(
            self.current_app, agent_kind=self.agent_kind)
        if observation:
            return f"{observation}\n\n{menu}"
        return menu

    def _dispatch_action(self, action: dict) -> str:
        from officebench import apps as ob_apps  # type: ignore
        app = action["app"]
        op = action.get("action", "")

        # System actions: no subprocess needed.
        if app == "system":
            if op == "switch_app":
                target = action.get("target_app")
                if target not in ob_apps.AVAILABLE_APPS:
                    return f"Execution failed: unknown target_app {target!r}."
                self.current_app = target
                self.history.append((action, f"Switched to {target}"))
                return f"Successfully switched to app: {target}"
            if op == "finish_task":
                answer = str(action.get("answer", "None")).replace('"', '').replace("'", '')
                (self._testbed_host / "data" / "answer.txt").write_text(answer + "\n")
                self.history.append((action, "finish"))
                return "Task finished"
            if op == "got_stuck":
                self.history.append((action, "stuck"))
                return "Task failed"
            return f"Execution failed: unknown system action {op!r}."

        # Shell: handle cd statefully on the host side; everything else runs inside the container.
        if app == "shell":
            cmd = action.get("command", "")
            if isinstance(cmd, list):
                cmd = " ".join(cmd)
            cd_flag = cmd.strip().startswith("cd ")
            if cd_flag:
                target = cmd.strip()[3:].strip()
                if not target.startswith("/"):
                    target = os.path.normpath(os.path.join(self.workdir, target))
                # Verify the target dir exists inside the container.
                probe = self._exec_in_container(f"test -d '{target}' && echo OK")
                if "OK" not in probe:
                    return f"Execution failed: cd: no such directory: {target}"
                self.workdir = target
                return f"Successfully cd to {target}"
            return self._exec_in_container(cmd)

        # LLM app: upstream shells out to OpenAI GPT-3.5 via /openai_key.txt (dead here, and its triple-quoted prompt mis-parses in bash), so route to the in-process agent model (self.llm_fn). Accept any action name: complete_text and the synonyms models tend to guess all mean "answer this".
        if app == "llm":
            prompt = action.get("prompt") or action.get("text") or ""
            if not str(prompt).strip():
                return ("Execution failed: the llm app needs a non-empty "
                        "'prompt'. Usage: {'app': 'llm', 'action': "
                        "'complete_text', 'prompt': <text>}.")
            if self.llm_fn is None:
                return ("Execution failed: the llm app is unavailable here. "
                        "Reason it out yourself, then write the result with "
                        "the relevant app (word / excel / shell).")
            try:
                out = str(self.llm_fn(str(prompt)))
            except Exception as e:
                return f"Execution failed: llm app error: {e!r}"
            self.history.append((action, out))
            return out

        # App-routed action: construct the shell command on the host, exec in the container.
        try:
            action_module = ob_apps.AVAILABLE_ACTIONS[app][op]
        except KeyError:
            return f"Execution failed: unknown action {app}.{op}."
        try:
            command = action_module.construct_action(self.workdir, args=action)
        except Exception as e:
            return f"Execution failed: construct_action raised {e!r}."
        out = self._exec_in_container(command)
        self.history.append((action, out))
        return out

    # Apptainer-exec backend.

    def _exec_in_container(self, shell_cmd: str) -> str:
        """Run a shell command inside the apptainer sandbox.

        Binds:
          - host scratch testbed → /testbed
          - vendored apps dir → /apps
          - the user HOME → same path inside (so the conda env's python with its hardcoded RPATHs + user-site deps works in-container)

        Sets `--pwd` to the agent's tracked workdir.
        """
        home = os.path.expanduser("~")
        # Point `python3` invocations at the conda env's python, which has the OfficeBench deps installed.
        cmd_with_python = shell_cmd.replace(
            "python3 ", f"{OFFICEBENCH_CONTAINER_PYTHON} ", 1)
        # Give LibreOffice (and any X-less GUI tool) writable runtime/temp dirs under the per-task /testbed bind: on some hosts /run/user/<uid> is read-only, so the default XDG_RUNTIME_DIR is unwritable and `convert-to pdf` silently produces no pdf while the app reports success, false-negativing every conversion task. (Do not override $HOME: the apps import `fire` from the user-site $HOME/.local, so moving HOME silently breaks ocr/calendar/email/excel.)
        cmd_with_python = (
            'export XDG_RUNTIME_DIR=/testbed/.xdg TMPDIR=/testbed/.tmp; '
            'mkdir -p "$XDG_RUNTIME_DIR" "$TMPDIR" 2>/dev/null; '
            + cmd_with_python)
        argv = [
            "apptainer", "exec",
            "--bind", f"{self._testbed_host}:/testbed",
            "--bind", f"{OFFICEBENCH_APPS_DIR}:/apps",
            "--bind", f"{home}:{home}",
            "--pwd", self.workdir,
            str(OFFICEBENCH_SANDBOX_SIF),
            "bash", "-c", cmd_with_python,
        ]
        try:
            # errors="replace": container commands can emit non-UTF-8 bytes (e.g. reading a binary/latin-1 office doc), which would otherwise raise UnicodeDecodeError inside subprocess and kill the task.
            r = subprocess.run(
                argv, capture_output=True, text=True, errors="replace",
                timeout=120)
        except subprocess.TimeoutExpired:
            return "Execution failed: command timed out after 120s."
        output = (r.stdout or "").strip()
        if "OBSERVATION:" in output:
            output = output.split("OBSERVATION:")[-1].strip()
        if r.returncode != 0:
            stderr_tail = (r.stderr or "").strip().splitlines()[-5:]
            return ("Execution failed: returncode=" + str(r.returncode) + ". "
                    + ("\n".join(stderr_tail) or output))[:4000]
        return output[:4000] if output else f"Successfully executed: {shell_cmd[:200]}"

    # Evaluation.

    def evaluate_run(self) -> tuple[bool, dict]:
        from officebench.utils import evaluate as ob_eval  # type: ignore
        details = []
        ok = True
        for item in (self.task._evaluation if self.task else []):
            fn_name = item["function"]
            args = item.get("args", {})
            fn = getattr(ob_eval, fn_name, None)
            if fn is None:
                details.append({"function": fn_name, "result": None,
                                "error": "unknown evaluator"})
                ok = False
                continue
            try:
                result = bool(fn(str(self._testbed_host), args))
            except Exception as e:
                details.append({"function": fn_name, "result": None, "error": repr(e)})
                ok = False
                continue
            details.append({"function": fn_name, "result": result})
            if not result:
                ok = False
        return ok, {"checks": details}

    def current_action_menu(self) -> str:
        """The per-turn action menu for the current app.

        A subtask executor is rebuilt fresh each subgoal; without this, the app state the previous subgoal left (the menu the runtime normally shows every turn) is lost and the next executor guesses action names/keys. Empty before any app is active (matches the first-subgoal demo, which shows no menu).
        """
        if not self.current_app:
            return ""
        return format_action_menu(
            self.current_app, agent_kind=self.agent_kind)

    # System-prompt context.

    def system_prompt_context(self) -> dict:
        from officebench import apps as ob_apps  # type: ignore
        intros = [ob_apps.AVAILABLE_APPS[a].INTRO for a in ob_apps.AVAILABLE_APPS]
        return {
            **(self.task.context if self.task else {}),
            "available_apps": list(ob_apps.AVAILABLE_APPS.keys()),
            "app_intros": intros,
        }

    def calculate_difficulty(self) -> int:
        return len(self.task._evaluation) if self.task else 0

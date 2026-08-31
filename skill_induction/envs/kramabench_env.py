"""KramaBench adapter: local persistent-kernel Python REPL.

KramaBench (https://github.com/mitdbg/Kramabench) is an end-to-end data-science benchmark: each task asks for a final answer computed from a domain "data lake" of raw files (csv / xlsx / ...) through a Python pipeline. The upstream repo ships an evaluation harness but no agent environment, so we build one here that mirrors its code-ReAct baseline (DS-GURU) and lets our Linear / Subtask × AWM / ASI agents run unchanged.

Design:
- Action space: Python code in a persistent kernel (variables survive across `run_code` calls), the same surface as AppWorld, so ASI skills exec straight into the REPL and AWM notes carry over.
- The kernel is a plain local subprocess (no apptainer, no root server; KramaBench runs locally). Its cwd is a per-task scratch workspace with the domain data lake symlinked in read-only at `./data/`.
- Completion: the agent calls `complete_task(answer=...)` in the REPL. The kernel records the answer and prints a completion marker; the agent loop detects it like AppWorld's `apis.supervisor.complete_task()`.
- Verifier: the upstream deterministic per-task metric (Success for numeric/string-exact, F1 for list-exact, RAE for numeric-approximate), ported inline here so the env has no LLM-judge / litellm dependency. We only run the 92 deterministic tasks; the 12 LLM-judge tasks (string_approximate / list_approximate) are excluded. `evaluate_run()` is used for logging only (§2 fairness clause).
- Scenario / group_key: the domain (e.g. `biomedical-easy-9` → `biomedical`), used for cross-scenario skill-transfer analysis.

Per the `World` contract (§3.3): errors include the substring `Execution failed`.
"""

from __future__ import annotations

import base64
import json as _json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from skill_induction.envs.base import World
from skill_induction.paths import (
    KRAMABENCH_DATA_ROOT,
    KRAMABENCH_PORT_ROOT,
    KRAMABENCH_SCRATCH_ROOT,
)


# Config.

KRAMABENCH_WORKLOAD_DIR = KRAMABENCH_PORT_ROOT / "workload"
KRAMABENCH_DOMAINS = (
    "archeology", "astronomy", "biomedical",
    "environment", "legal", "wildfire",
)
# Answer types whose metric is deterministic (no LLM judge): the 92/104 tasks we evaluate. string_approximate / list_approximate use an LLM paraphrase judge and are excluded.
DETERMINISTIC_ANSWER_TYPES = frozenset({
    "numeric_exact", "string_exact", "list_exact", "numeric_approximate",
})

# Kernel interpreter: default to the runner's own interpreter, which already has pandas/numpy/scipy/openpyxl/lxml.
KRAMABENCH_PYTHON = os.path.expanduser(
    os.environ.get("KRAMABENCH_PYTHON", sys.executable)
)
KRAMABENCH_EXEC_TIMEOUT = float(os.environ.get("KRAMABENCH_EXEC_TIMEOUT", "120"))
# A continuous score at or above this counts as a binary "success" for TGC-style reporting. Exact answer types already score 0/1, so it only bites list_exact (F1) / numeric_approximate (RAE).
KRAMABENCH_SUCCESS_THRESHOLD = float(
    os.environ.get("KRAMABENCH_SUCCESS_THRESHOLD", "0.5")
)
# Max files to enumerate in the identity-line data-lake listing.
_MAX_LISTED_FILES = 40
# Max xlsx sheets to expand (schema + sample) per shown workbook.
_MAX_XLSX_SHEETS = 4


# Task index.

_TASK_INDEX: dict[str, dict] | None = None


def _load_task_index() -> dict[str, dict]:
    """task_id → task record (query/answer/answer_type/data_sources/domain), built once from all vendored domain workloads."""
    global _TASK_INDEX
    if _TASK_INDEX is not None:
        return _TASK_INDEX
    index: dict[str, dict] = {}
    for domain in KRAMABENCH_DOMAINS:
        wl = KRAMABENCH_WORKLOAD_DIR / f"{domain}.json"
        if not wl.exists():
            continue
        for task in _json.loads(wl.read_text()):
            tid = task["id"]
            rec = dict(task)
            rec["domain"] = domain
            index[tid] = rec
    _TASK_INDEX = index
    return index


def kramabench_task_ids(deterministic_only: bool = True) -> list[str]:
    """All KramaBench task ids in a stable order (domain order, then the workload's own order). With `deterministic_only`, drop the 12 LLM-judge tasks, leaving the 92 no-judge tasks."""
    index = _load_task_index()
    ids: list[str] = []
    for domain in KRAMABENCH_DOMAINS:
        for tid, rec in index.items():
            if rec["domain"] != domain:
                continue
            if deterministic_only and rec.get("answer_type") not in (
                    DETERMINISTIC_ANSWER_TYPES):
                continue
            ids.append(tid)
    return ids


def _domain_of(task_id: str) -> str:
    return task_id.split("-", 1)[0]


# Deterministic scoring: faithful ports of KramaBench `benchmark/metrics.py` for the four deterministic answer types, kept inline so the env needs no nltk / rouge / litellm. The continuous score is the primary metric (matches the KramaBench leaderboard); a binary success is thresholded from it.

def _str_to_float(num_string):
    if isinstance(num_string, (int, float)):
        return float(num_string)
    s = str(num_string).strip()
    if s.endswith("%"):
        return float(s[:-1]) / 100
    return float(s)


def _score_success(predicted, target) -> float:
    """numeric_exact / string_exact: exact match with float tolerance."""
    if predicted is None:
        return 0.0
    def _num(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            try:
                return _str_to_float(v)
            except Exception:
                return v
        return v
    p, t = _num(predicted), _num(target)
    if isinstance(p, (int, float)) and not isinstance(p, bool) and \
            isinstance(t, (int, float)) and not isinstance(t, bool):
        if p == t:
            return 1.0
        if isinstance(t, float) or isinstance(p, float):
            if t == 0:
                return float(abs(p) < 1e-6)
            rea = abs(p - t) / abs(t)
            abs_diff = abs(p - t)
            larger = max(abs(p), abs(t))
            return float(rea < 1e-6 or abs_diff < 1e-6
                         or abs_diff < larger * 1e-4)
        return 0.0
    if isinstance(t, str) and isinstance(p, str):
        return float(p.strip().lower() == t.strip().lower())
    return float(p == t)


def _coerce_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            parsed = _json.loads(v)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [x.strip() for x in v.split(",") if x.strip()]
    return [v]


def _score_f1(predicted, target) -> float:
    """list_exact: order-insensitive F1 with string/numeric matching."""
    try:
        tgt = _coerce_list(target)
        pred = _coerce_list(predicted)
        if len(tgt) == 0:
            return float(len(pred) == 0)
        matched: set[int] = set()
        recall_cnt = 0
        for t in tgt:
            found = False
            for j, p in enumerate(pred):
                if isinstance(t, str):
                    if str(p).strip().lower() == t.strip().lower():
                        found = True
                        matched.add(j)
                        break
                elif isinstance(t, (int, float)) and not isinstance(t, bool):
                    try:
                        pp = p if isinstance(p, (int, float)) else _str_to_float(p)
                    except Exception:
                        continue
                    denom = max(abs(t), 1e-12)
                    if abs(pp - t) / denom < 1e-6:
                        found = True
                        matched.add(j)
                        break
                else:
                    if p == t:
                        found = True
                        matched.add(j)
                        break
            if found:
                recall_cnt += 1
        recall = recall_cnt / len(tgt)
        precision = (len(matched) / len(pred)) if pred else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    except Exception:
        return 0.0


def _score_rae(predicted, target) -> float:
    """numeric_approximate: RAE score 1/(1+relative_abs_error)."""
    try:
        p = _str_to_float(predicted)
        t = _str_to_float(target)
        # A NaN predicted (or gold) answer is unusable, so score 0. NaN arithmetic does not raise, so without this guard 1/(1+nan)=nan would silently leak and poison means.
        if p != p or t != t:
            return 0.0
        rae = abs(p - t) / abs(t) if t != 0 else (0.0 if p == 0 else float("inf"))
        return 1.0 / (1.0 + rae)
    except Exception:
        return 0.0


def score_answer(predicted, target, answer_type: str) -> float:
    """Continuous KramaBench score in [0, 1] for a deterministic task."""
    if answer_type in ("numeric_exact", "string_exact"):
        return _score_success(predicted, target)
    if answer_type == "list_exact":
        return _score_f1(predicted, target)
    if answer_type == "numeric_approximate":
        return _score_rae(predicted, target)
    # LLM-judge types are excluded from our runs; score 0 defensively.
    return 0.0


# No-op server.

class KramaBenchNoopServer:
    """KramaBench has no HTTP server; the 'environment' is the per-task Python kernel held by `KramaBenchWorld`."""

    def __init__(self, **kwargs):
        self.base_url = ""

    def __enter__(self) -> str:
        return self.base_url

    def __exit__(self, *exc) -> None:
        return None


# Task handle.

class _KramaTaskHandle:
    def __init__(self, task_id: str, rec: dict, group_key: str,
                 data_listing: str):
        self.task_id = task_id
        self.id = task_id
        self.instruction = rec["query"]
        self.group_key = group_key
        self.answer_type = rec.get("answer_type")
        self.data_sources = rec.get("data_sources", [])
        self.context = {
            "domain": rec["domain"],
            "answer_type": rec.get("answer_type"),
            "data_sources": rec.get("data_sources", []),
            "data_listing": data_listing,
        }
        self._gold_answer = rec.get("answer")


# Persistent Python kernel: a sentinel-framed stdin/stdout protocol plus a `complete_task()` builtin injected into the kernel globals. On completion the daemon prints <<<KRAMA_COMPLETE>>> and <<<KRAMA_ANSWER <base64-of-json>>> so the host can recover the (possibly multi-line / non-string) answer.

_KERNEL_DAEMON_SRC = r"""
import sys, os, io, traceback, re, json, base64

os.environ.setdefault('MPLBACKEND', 'Agg')

_GLOBALS = {}
_STATE = {'answer': None, 'has_answer': False, 'done': False}

def complete_task(answer=None, *args, **kwargs):
    # Record the FIRST real answer; a later complete_task(None) (e.g. the
    # forced terminal action) must not clobber a genuine answer.
    if answer is not None or not _STATE['has_answer']:
        _STATE['answer'] = answer
        if answer is not None:
            _STATE['has_answer'] = True
    _STATE['done'] = True
    try:
        payload = base64.b64encode(
            json.dumps(answer, default=str).encode('utf-8')).decode('ascii')
    except Exception:
        payload = base64.b64encode(
            json.dumps(str(answer)).encode('utf-8')).decode('ascii')
    print('<<<KRAMA_COMPLETE>>>')
    print('<<<KRAMA_ANSWER ' + payload + '>>>')
    return 'Task marked complete.'

_GLOBALS['complete_task'] = complete_task

def _read_frame():
    line = sys.stdin.readline()
    if not line:
        return None, None
    m = re.match(r'^<<<EXEC ([0-9a-fA-F]+)>>>\s*$', line)
    if not m:
        return _read_frame()
    sentinel = m.group(1)
    end_tag = '<<<END ' + sentinel + '>>>'
    lines = []
    while True:
        ln = sys.stdin.readline()
        if not ln:
            return sentinel, ''.join(lines)
        if ln.rstrip('\n') == end_tag:
            return sentinel, ''.join(lines)
        lines.append(ln)

def _run(sentinel, code):
    buf = io.StringIO()
    so, se = sys.stdout, sys.stderr
    sys.stdout = buf
    sys.stderr = buf
    status = 'ok'
    try:
        exec(compile(code, '<agent>', 'exec'), _GLOBALS)
    except SystemExit:
        status = 'exit'
    except BaseException:
        traceback.print_exc()
        status = 'error'
    finally:
        sys.stdout = so
        sys.stderr = se
    out = buf.getvalue()
    so.write(out)
    if not out.endswith('\n'):
        so.write('\n')
    so.write('<<<DONE ' + sentinel + ' ' + status + '>>>\n')
    so.flush()

def main():
    sys.stdout.write('<<<READY>>>\n')
    sys.stdout.flush()
    while True:
        sentinel, code = _read_frame()
        if sentinel is None:
            return
        _run(sentinel, code)

if __name__ == '__main__':
    main()
"""


class _Kernel:
    READY_TAG = "<<<READY>>>"
    DONE_RE = re.compile(r"^<<<DONE ([0-9a-fA-F]+) (\w+)>>>\s*$")

    def __init__(self, python_bin: str, cwd: str):
        self.python_bin = python_bin
        self.cwd = cwd
        self.proc: subprocess.Popen | None = None
        self._out_lines: list[str] = []
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["MPLBACKEND"] = "Agg"
        # Keep the kernel from spawning huge BLAS thread pools per task.
        env.setdefault("OMP_NUM_THREADS", "4")
        self.proc = subprocess.Popen(
            [self.python_bin, "-u", "-c", _KERNEL_DAEMON_SRC],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, env=env,
            cwd=self.cwd, bufsize=1,
        )
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()
        deadline = time.time() + 30.0
        idx = 0
        while time.time() < deadline:
            with self._lock:
                while idx < len(self._out_lines):
                    line = self._out_lines[idx]
                    idx += 1
                    if line.rstrip("\n") == self.READY_TAG:
                        self._out_lines = self._out_lines[idx:]
                        return
            time.sleep(0.02)
            if self.proc.poll() is not None:
                raise RuntimeError(
                    "kramabench kernel died before READY: "
                    + "".join(self._out_lines[:50]))
        raise TimeoutError("kramabench kernel did not become READY in 30s")

    def _reader_loop(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            with self._lock:
                self._out_lines.append(line)

    def run(self, code: str) -> tuple[str, str]:
        if self.proc is None or self.proc.poll() is not None:
            return "Execution failed: kernel is not running.", "error"
        sentinel = uuid.uuid4().hex
        frame = f"<<<EXEC {sentinel}>>>\n{code}\n<<<END {sentinel}>>>\n"
        try:
            assert self.proc.stdin
            self.proc.stdin.write(frame)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            return f"Execution failed: kernel pipe closed: {e!r}", "error"
        deadline = time.time() + KRAMABENCH_EXEC_TIMEOUT
        idx = 0
        while time.time() < deadline:
            with self._lock:
                while idx < len(self._out_lines):
                    line = self._out_lines[idx]
                    idx += 1
                    m = self.DONE_RE.match(line)
                    if m and m.group(1) == sentinel:
                        collected = "".join(self._out_lines[:idx - 1])
                        self._out_lines = self._out_lines[idx:]
                        return collected, m.group(2)
            time.sleep(0.02)
            if self.proc.poll() is not None:
                with self._lock:
                    tail = "".join(self._out_lines[:idx])
                    self._out_lines = self._out_lines[idx:]
                return tail + "\n[kernel exited unexpectedly]\n", "error"
        with self._lock:
            tail = "".join(self._out_lines[:idx])
            self._out_lines = self._out_lines[idx:]
        try:
            self.proc.kill()
        except Exception:
            pass
        return (tail + f"\nExecution failed: kernel timed out after "
                f"{KRAMABENCH_EXEC_TIMEOUT:.0f}s.\n"), "timeout"

    def stop(self) -> None:
        if self.proc is not None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None


# World.

_COMPLETE_RE = re.compile(r"^<<<KRAMA_ANSWER (\S+)>>>\s*$", re.MULTILINE)


class KramaBenchWorld(World):
    """Per-task KramaBench environment (one persistent Python kernel)."""

    def __init__(
        self,
        task_id: str,
        experiment_name: str = "",
        *,
        server_url: str = "",
        group_key: str | None = None,
        data_root: str | Path | None = None,
        scratch_root: Path | None = None,
    ):
        self.task_id = task_id
        self.experiment_name = experiment_name
        self.data_root = Path(data_root or KRAMABENCH_DATA_ROOT)
        self.scratch_root = Path(scratch_root or KRAMABENCH_SCRATCH_ROOT)
        self._group_key = group_key
        self.task: _KramaTaskHandle | None = None
        self._kernel: _Kernel | None = None
        self._workspace: Path | None = None
        self._task_finished = False
        self._final_answer = None
        self._has_answer = False

    # Lifecycle.

    def __enter__(self) -> "KramaBenchWorld":
        index = _load_task_index()
        if self.task_id not in index:
            raise FileNotFoundError(
                f"KramaBench task not found: {self.task_id!r}. Vendored "
                f"workloads under {KRAMABENCH_WORKLOAD_DIR}.")
        rec = index[self.task_id]
        domain = rec["domain"]
        input_dir = self.data_root / domain / "input"
        if not input_dir.is_dir():
            raise FileNotFoundError(
                f"KramaBench data lake missing: {input_dir}. Point "
                f"KRAMABENCH_DATA_ROOT at the cloned `data/` dir.")

        # Per-task scratch workspace with the domain lake symlinked in. Key the dir by experiment_name (which carries stage, agent, model, and task) so concurrent cells of the same agent on different models never collide on one workspace path (a race on rmtree / symlink); ignore_errors tolerates a concurrent removal as a final safety net.
        cell_tag = (self.experiment_name or self.task_id).replace("/", "__")
        self._workspace = self.scratch_root / cell_tag
        if self._workspace.exists():
            shutil.rmtree(self._workspace, ignore_errors=True)
        self._workspace.mkdir(parents=True, exist_ok=True)
        (self._workspace / "data").symlink_to(input_dir, target_is_directory=True)

        listing = _build_data_listing(input_dir, rec.get("data_sources", []))
        group_key = self._group_key or domain
        # The runner passes the benchmark name as group_key; prefer the domain so scenario grouping is correct for transfer analysis.
        if group_key == "kramabench":
            group_key = domain
        self.task = _KramaTaskHandle(self.task_id, rec, group_key, listing)

        self._kernel = _Kernel(KRAMABENCH_PYTHON, str(self._workspace))
        self._kernel.start()
        # Friendly preamble: common imports + cwd note (best-effort).
        self._kernel.run(
            "import os, json, math\n"
            "try:\n    import numpy as np, pandas as pd\nexcept Exception:\n    pass\n"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._kernel is not None:
            self._kernel.stop()
            self._kernel = None
        # Leave the workspace for post-hoc inspection; it is tiny (just a symlink + any files the agent wrote).

    # Action dispatch.

    def execute(self, payload: str) -> str:
        code = payload.rstrip()
        if not code:
            return "Execution failed: empty code."
        if self._kernel is None:
            return "Execution failed: kernel not started."
        out, status = self._kernel.run(code)

        # Detect completion: the kernel printed the answer marker.
        m = _COMPLETE_RE.search(out)
        if m is not None:
            self._task_finished = True
            try:
                decoded = base64.b64decode(m.group(1)).decode("utf-8")
                answer = _json.loads(decoded)
            except Exception:
                answer = None
            if answer is not None or not self._has_answer:
                self._final_answer = answer
                if answer is not None:
                    self._has_answer = True
            # Strip the protocol markers from what the model sees.
            out = _COMPLETE_RE.sub("", out)
            out = out.replace("<<<KRAMA_COMPLETE>>>", "")
            shown = out.strip()
            msg = "Execution successful. Task finished; answer recorded"
            msg += f": {self._final_answer!r}." if self._has_answer else "."
            return (shown + "\n" + msg).strip() if shown else msg

        out = out.rstrip()
        if status in ("error", "timeout"):
            if "Execution failed" not in out:
                out = f"Execution failed (Python traceback):\n{out}"
        if not out:
            out = "(no output)"
        # Live state footer after every action (mirrors OfficeBench's per-turn menu): the data-lake files available + anything the agent has written.
        return (out[:3600] + "\n\n" + self._lake_footer()).strip()

    def _lake_footer(self) -> str:
        srcs = (self.task.data_sources if self.task else None) or []
        try:
            n = sum(1 for _ in (self._workspace / "data").iterdir())
        except Exception:
            n = 0
        try:
            written = sorted(p.name for p in self._workspace.iterdir()
                             if p.name != "data")
        except Exception:
            written = []
        s = f"[state] data lake ./data/: {n} file(s)"
        if srcs:
            s += f"; this task's files: {', '.join(str(x) for x in srcs)}"
        if written:
            s += f"; you have written: {', '.join(written[:10])}"
        s += ". Finish with complete_task(answer=...)."
        return s

    def preload_skills(self, srcs) -> None:
        """Exec reusable skill source into the persistent REPL so retrieved skill names resolve in later steps (used by ASI). Best-effort."""
        for src in srcs or []:
            if src and str(src).strip():
                try:
                    self.execute(str(src))
                except Exception:
                    pass

    # Evaluation (logging only).

    def evaluate_run(self) -> tuple[bool, dict]:
        if self.task is None:
            return False, {"error": "no task"}
        gold = self.task._gold_answer
        atype = self.task.answer_type
        score = score_answer(self._final_answer, gold, atype) \
            if self._task_finished else 0.0
        success = bool(score >= KRAMABENCH_SUCCESS_THRESHOLD)
        return success, {
            "score": score,
            "answer_type": atype,
            "predicted": self._final_answer,
            "gold": gold,
            "completed_signal": self._task_finished,
        }

    # KramaBench's own continuous metric is more informative than 0/1, so surface it via eval_extras["score"]; downstream analysis reads it from there.

    def system_prompt_context(self) -> dict:
        return self.task.context if self.task else {}

    def calculate_difficulty(self) -> int:
        """Number of gold subtasks = a proxy for pipeline complexity."""
        rec = _load_task_index().get(self.task_id, {})
        return len(rec.get("subtasks", []) or [])


def _build_data_listing(input_dir: Path, data_sources: list[str]) -> str:
    """Present the task's data files the way upstream KramaBench does (`--use_truth_subset` + DS-GURU `get_input_data` / `supply_data_snippet`).

    For the gold `data_sources`, show each file's full schema (columns + dtypes and sample rows for CSV, per-sheet columns + sample for xlsx, or a text preview) so the agent has the schema upfront instead of discovering the right file among up to ~1.5k. The listing states the total file count, marks the shown files as the ones this task needs (full schema below), then lists the remaining lake files by name as reference only.
    """
    import fnmatch
    import os as _os

    import pandas as pd

    all_files = sorted(p for p in input_dir.rglob("*") if p.is_file())
    rel_all = [str(p.relative_to(input_dir)) for p in all_files]

    # Resolve gold data_sources to real files (fnmatch on full rel-path and basename, mirroring DS-GURU.generate_prompt); "./" or empty means the whole lake.
    pats = [str(s) for s in (data_sources or []) if s and str(s) != "./"]
    resolved: list[str] = []
    for pat in pats:
        ms = [r for r in rel_all
              if r == pat or fnmatch.fnmatch(r, pat)
              or fnmatch.fnmatch(_os.path.basename(r), _os.path.basename(pat))]
        resolved.extend(ms if ms else [pat])   # keep unresolved pat as a note
    # de-dup preserving order
    seen = set(); resolved = [r for r in resolved if not (r in seen or seen.add(r))]
    whole_lake = not resolved
    if whole_lake:
        resolved = rel_all[:_MAX_LISTED_FILES]

    # Explanation header.
    n_total = len(rel_all)
    if whole_lake:
        lines = [
            f"The data lake `./data/` contains {n_total} file(s) and this "
            f"question may draw on any of them. The {len(resolved)} file(s) "
            f"below (full schema shown) are a sample to get you started; use "
            f"os.listdir('data') / glob to explore the rest."]
    else:
        lines = [
            f"The data lake `./data/` contains {n_total} file(s) total. The "
            f"{len(resolved)} file(s) below are the ones THIS task needs — "
            f"their full schema is shown; read them at `./data/<name>`. (The "
            f"remaining files are listed by name afterwards, for reference "
            f"only.)"]

    for r in resolved:
        path = input_dir / r
        lines.append(f"\n### data/{r}")
        if not path.exists():
            lines.append("  (named by the task — locate it under ./data/)")
            continue
        low = r.lower()
        if low.endswith(".csv"):
            df = None
            for kw in ({}, {"encoding": "latin-1"}):
                try:
                    df = pd.read_csv(path, engine="python", on_bad_lines="skip",
                                     nrows=200, **kw)
                    break
                except Exception:
                    continue
            if df is not None:
                dtypes = ", ".join(f"{c}:{t}" for c, t
                                   in df.dtypes.astype(str).items())
                lines.append(f"  columns ({df.shape[1]}): {dtypes[:1500]}")
                lines.append("  sample rows:")
                lines.append(df.head(10).to_csv(index=False)[:1800])
            else:
                lines.append("  (CSV unreadable here — inspect it in the REPL)")
        elif low.endswith((".xlsx", ".xls", ".xlsm")):
            # Multi-sheet workbook: list sheet names, then each sheet's columns + a bounded sample; xlsx files frequently hold several sheets.
            try:
                xl = pd.ExcelFile(path)
                sheets = list(xl.sheet_names)
                shown_sheets = sheets[:_MAX_XLSX_SHEETS]
                more = (f" … (+{len(sheets) - len(shown_sheets)} more)"
                        if len(sheets) > len(shown_sheets) else "")
                lines.append(f"  xlsx with {len(sheets)} sheet(s): "
                             + ", ".join(repr(s) for s in shown_sheets) + more)
                for s in shown_sheets:
                    try:
                        sdf = xl.parse(s, nrows=200)
                    except Exception:
                        lines.append(f"  - sheet {s!r}: (unreadable here — "
                                     "inspect it in the REPL)")
                        continue
                    dtypes = ", ".join(f"{c}:{t}" for c, t
                                       in sdf.dtypes.astype(str).items())
                    lines.append(f"  - sheet {s!r} columns ({sdf.shape[1]}): "
                                 f"{dtypes[:1200]}")
                    lines.append("    sample rows:")
                    lines.append(sdf.head(10).to_csv(index=False)[:1500])
            except Exception:
                lines.append("  (xlsx unreadable here — inspect it in the "
                             "REPL with pd.ExcelFile(path).sheet_names then "
                             "pd.read_excel(path, sheet_name=...))")
        elif low.endswith((".txt", ".md", ".json", ".html", ".htm", ".tle",
                           ".lst", ".dat", ".text")):
            try:
                txt = path.read_text(errors="ignore")
                lines.append("  preview (first lines):")
                lines.append("\n".join(txt.splitlines()[:15])[:1500])
            except Exception:
                lines.append("  (text unreadable here — inspect it in the REPL)")
        else:
            ext = r.rsplit(".", 1)[-1] if "." in r else "?"
            lines.append(f"  (binary/complex `.{ext}` — load it in the REPL, "
                         "e.g. cdflib / netCDF4 / numpy.load)")
    if whole_lake:
        lines.append(f"\n(This task's data spans the whole lake; {n_total} "
                     "files total — use os.listdir('data') / glob to explore.)")
    else:
        others = [r for r in rel_all if r not in set(resolved)]
        if others:
            shown = others[:60]
            tail = f" … (+{len(others) - 60} more)" if len(others) > 60 else ""
            lines.append("\nOther files in the lake (not required for this "
                         "task, for reference; names only):\n  "
                         + ", ".join(shown) + tail)
    return "\n".join(lines)

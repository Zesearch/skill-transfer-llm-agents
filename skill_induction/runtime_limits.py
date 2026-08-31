"""Wall-clock guard for in-process `exec()` of model-generated code.

The ASI (code-skill) paths exec model-written Python in a persistent in-process REPL namespace that must survive across steps, so it can't run in a subprocess; without a guard, a generated infinite loop would run forever.

SIGALRM fires in the main thread and CPython checks for pending signals between bytecode instructions, so a pure-Python loop is interrupted; code stuck inside a C extension only stops once it returns to the interpreter. `CodeTimeout` subclasses BaseException so a generated `except Exception` can't swallow it (the timer is one-shot and would never re-fire). No-op off the main thread, where signal handlers can't be installed.
"""

from __future__ import annotations

import contextlib
import os
import signal
import threading


class CodeTimeout(BaseException):
    """Raised inside exec'd code when it exceeds the wall-clock limit."""


def exec_timeout_seconds() -> int:
    try:
        return int(os.environ.get("SKILL_INDUCTION_PY_EXEC_TIMEOUT", "120"))
    except ValueError:
        return 120


@contextlib.contextmanager
def time_limit(seconds: int | None = None):
    seconds = exec_timeout_seconds() if seconds is None else seconds
    if (seconds <= 0
            or threading.current_thread() is not threading.main_thread()):
        yield
        return

    def _handler(signum, frame):
        raise CodeTimeout(
            f"code exceeded {seconds}s wall-clock limit "
            f"(possible infinite loop)")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)

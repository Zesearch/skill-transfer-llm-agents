"""AppWorld adapter: talks to a local `appworld serve environment` subprocess.

The server runs inside the dedicated `appworld-env` conda env (heavy deps, Python 3.11); the runtime (this process) lives in its own env and talks to it via HTTP.

The server writes its per-task scratch (DB snapshots, API-call logs, version files) under `<CWD>/experiments/outputs/<exp_name>/`, so we run the subprocess with `cwd = APPWORLD_SERVER_CWD` (defined in `skill_induction.paths`) to keep those files out of the repo.

Usage::

    with AppWorldServer(data_root=APPWORLD_DATA_ROOT) as server_url:
        with RemoteAppWorld(task_id, "exp_name", server_url=server_url,
                            data_root=APPWORLD_DATA_ROOT) as world:
            obs = world.execute("print(apis.api_docs.show_app_descriptions())")
            success, _ = world.evaluate_run()
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import requests

from skill_induction.envs.base import World
from skill_induction.paths import APPWORLD_SERVER_CWD


# Server lifecycle.

def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class AppWorldServer:
    """Context manager that spawns `appworld serve environment` and tears it down on exit. The server runs in the `appworld-env` conda env so the project runtime doesn't need AppWorld's heavy deps installed."""

    def __init__(
        self,
        data_root: str | Path,
        conda_env_name: str = "appworld-env",
        host: str = "127.0.0.1",
        # Cold start can exceed 60s (the first activation pages the conda env in over a networked filesystem, and parallel cells contend); warm starts are ~7s.
        startup_timeout: float = 180.0,
    ):
        # AppWorld's CLI `--root` must be the dir that contains `data/{api_docs,base_dbs,datasets,tasks}`: appworld appends `/data` itself, and `APPWORLD_DATA_ROOT` is exactly that parent.
        self.data_root = str(Path(data_root).resolve())
        self.conda_env_name = conda_env_name
        self.host = host
        self.port = _get_free_port()
        self.base_url = f"http://{self.host}:{self.port}"
        self.process: subprocess.Popen | None = None
        self.startup_timeout = startup_timeout

    def __enter__(self) -> str:
        print(f"[AppWorldServer] starting on port {self.port} (env={self.conda_env_name})...")
        self._spawn()
        self._wait_for_server()
        return self.base_url

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._terminate()

    def _spawn(self) -> None:
        """Launch the `appworld serve environment` subprocess on self.port."""
        # AppWorld writes per-task scratch relative to its CWD; point that outside the repo.
        server_cwd = APPWORLD_SERVER_CWD
        server_cwd.mkdir(parents=True, exist_ok=True)
        # `conda` is not on PATH inside a `conda run` subprocess, so call it by absolute path via CONDA_EXE (exported by `conda run`).
        conda_exe = os.environ.get("CONDA_EXE", "conda")
        cmd = [
            conda_exe, "run", "-n", self.conda_env_name, "--no-capture-output",
            "appworld", "serve", "environment",
            "--port", str(self.port), "--root", self.data_root,
        ]
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            preexec_fn=os.setsid, cwd=str(server_cwd),
        )

    def _terminate(self) -> None:
        if self.process is not None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except Exception:
                pass
            self.process = None

    def _healthy(self) -> bool:
        """True iff the subprocess is alive and `/docs` answers 200."""
        if self.process is None or self.process.poll() is not None:
            return False
        try:
            r = requests.get(f"{self.base_url}/docs", timeout=5)
            return r.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def ensure_alive(self) -> str:
        """Restart the server if it died (e.g. OOM-killed on local-model runs) so one crash doesn't cascade the rest of the task stream into connection-refused errors. Returns the current base_url (the port may change after a restart)."""
        if self._healthy():
            return self.base_url
        print("[AppWorldServer] server not responding — restarting...")
        self._terminate()
        # Fresh port avoids a TIME_WAIT bind race against the dead process.
        self.port = _get_free_port()
        self.base_url = f"http://{self.host}:{self.port}"
        self._spawn()
        self._wait_for_server()
        print(f"[AppWorldServer] restarted on port {self.port}.")
        return self.base_url

    def _wait_for_server(self) -> None:
        start = time.time()
        while time.time() - start < self.startup_timeout:
            try:
                r = requests.get(f"{self.base_url}/docs", timeout=5)
                if r.status_code == 200:
                    return
            except (requests.ConnectionError, requests.Timeout):
                # Connection refused (not up yet) and ReadTimeout (up but `/docs` slow under parallel cold-start contention) are both transient; keep polling until startup_timeout.
                pass
            if self.process is not None and self.process.poll() is not None:
                stderr = (self.process.stderr.read() if self.process.stderr else b"").decode(errors="replace")
                raise RuntimeError(
                    f"AppWorld server exited before becoming ready.\n{stderr[:2000]}"
                )
            time.sleep(1)
        raise RuntimeError(
            f"AppWorld server did not become ready within {self.startup_timeout}s."
        )


# World adapter.

class _TaskHandle:
    """Lightweight task object exposing what the agent needs."""

    def __init__(self, task_id: str, task_data: dict, group_key: str | None):
        self.task_id = task_id
        self.id = task_id
        self.instruction = task_data["instruction"]
        self.supervisor = task_data.get("supervisor", {})
        self.group_key = group_key


class RemoteAppWorld(World):
    def __init__(
        self,
        task_id: str,
        experiment_name: str,
        *,
        server_url: str,
        data_root: str | Path,
        group_key: str | None = None,
    ):
        self.task_id = task_id
        self.experiment_name = experiment_name
        self.server_url = server_url.rstrip("/")
        self.data_root = str(Path(data_root).resolve())
        self._group_key = group_key
        self.task: _TaskHandle | None = None  # populated in __enter__

    def __enter__(self) -> "RemoteAppWorld":
        # Initialize the task session on the server.
        r = requests.post(
            f"{self.server_url}/initialize",
            json={"task_id": self.task_id,
                  "experiment_name": self.experiment_name},
            timeout=60,
        )
        r.raise_for_status()
        r_task = requests.get(f"{self.server_url}/tasks/{self.task_id}", timeout=30)
        r_task.raise_for_status()
        task_data = r_task.json()["output"]
        self.task = _TaskHandle(self.task_id, task_data, self._group_key)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            requests.post(f"{self.server_url}/close",
                          json={"task_id": self.task_id}, timeout=10)
        except Exception:
            pass
        # Remove this task's on-disk scratch: nothing reads the per-task DB snapshots back after evaluate_run(), and leaving them accumulates inodes without bound. Set SKILL_INDUCTION_KEEP_SCRATCH=1 to keep them for debugging.
        if os.environ.get("SKILL_INDUCTION_KEEP_SCRATCH") != "1":
            import shutil
            exp_out = (Path(self.data_root) / "experiments" / "outputs"
                       / self.experiment_name)
            try:
                shutil.rmtree(exp_out / "tasks" / self.task_id, ignore_errors=True)
            except Exception:
                pass
            # Drop the now-empty per-experiment shell too (per-task exp names).
            try:
                tasks_dir = exp_out / "tasks"
                if tasks_dir.is_dir() and not any(tasks_dir.iterdir()):
                    shutil.rmtree(exp_out, ignore_errors=True)
            except Exception:
                pass

    # World API.

    def execute(self, payload: str) -> str:
        try:
            r = requests.post(
                f"{self.server_url}/execute",
                json={"task_id": self.task_id, "code": payload},
                timeout=120,
            )
        except requests.RequestException as e:
            return f"Execution failed: transport error: {e}"
        if r.status_code != 200:
            return f"Execution failed: Server Error {r.status_code}: {r.text[:500]}"
        output = r.json().get("output", "")
        return output if isinstance(output, str) else json.dumps(output)

    def preload_skills(self, srcs) -> None:
        """Exec reusable skill source into the persistent AppWorld REPL so retrieved code skills are callable by name in later steps. Best-effort; not recorded as a trace step."""
        for src in srcs or []:
            if not src or not str(src).strip():
                continue
            try:
                self.execute(str(src))
            except Exception:
                pass

    def evaluate_run(self) -> tuple[bool, dict]:
        try:
            r = requests.post(
                f"{self.server_url}/evaluate",
                json={
                    "task_id": self.task_id,
                    "experiment_name": self.experiment_name,
                    "suppress_errors": True,
                },
                timeout=60,
            )
        except requests.RequestException as e:
            return False, {"error": f"transport: {e}"}
        if r.status_code == 200:
            data = r.json().get("output", {}) or {}
            return bool(data.get("success", False)), data
        if r.status_code == 500:
            return False, {"error": "server 500", "raw": r.text[:2000]}
        return False, {"error": f"http {r.status_code}", "raw": r.text[:2000]}

    def calculate_difficulty(self) -> int:
        """Return the gold solution-code line count from local metadata; 0 if the file is missing. Test-normal tasks have an empty solution_code but the stat still records `num_solution_code_lines` from generation."""
        meta = Path(self.data_root) / "data" / "tasks" / self.task_id / "ground_truth" / "metadata.json"
        if not meta.exists():
            # Some checkouts point data_root directly at the `data/` dir, so also try data_root/tasks/.../metadata.json.
            meta = Path(self.data_root) / "tasks" / self.task_id / "ground_truth" / "metadata.json"
        if not meta.exists():
            return 0
        try:
            with open(meta) as f:
                return int(json.load(f).get("num_solution_code_lines", 0))
        except Exception:
            return 0

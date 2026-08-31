"""Project paths, defined in one place.

Small artifacts (result/memory pkls, pilot lists) live in the in-repo `data/` directory; heavy outputs and benchmark data live under configurable roots declared here.

Every path defaults to a location derived from the repo root (or a scratch/temp dir) and can be overridden via an environment variable, so the project is portable without code changes.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _env_path(var: str, default: str | Path) -> Path:
    return Path(os.environ.get(var, str(default))).expanduser()


# Repo root.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Small in-repo artifacts. Result and memory pkls live under data/results/.
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = DATA_DIR / "results"
PILOTS_DIR: Path = DATA_DIR / "pilots"

# Heavy external assets; default under the repo, override to relocate.
EXTERNAL_ROOT: Path = _env_path(
    "SKILL_INDUCTION_EXTERNAL",
    PROJECT_ROOT / "external",
)
# Scratch dirs, per-task testbeds, and server CWDs should live off the home filesystem: home often has a tight inode quota, and per-task officebench testbeds create hundreds of files each, enough to block all writes when the quota fills. Defaults to $SCRATCH or a temp dir; set SKILL_INDUCTION_WORK for a persistent location. Static small things (vendored benchmark code, the SIF image) stay under EXTERNAL_ROOT.
WORK_ROOT: Path = _env_path(
    "SKILL_INDUCTION_WORK",
    os.environ.get("SCRATCH")
    or os.path.join(tempfile.gettempdir(), "skill_induction_work"),
)
# AppWorld server CWD; its `experiments/outputs/<exp_name>/` artifacts land here, not in the repo.
APPWORLD_SERVER_CWD: Path = _env_path(
    "SKILL_INDUCTION_APPWORLD_CWD", WORK_ROOT / "appworld_cwd",
)
# Vendored OfficeBench upstream (`apps/`, `utils/`, `tasks/`); kept outside the repo because it's 21 MB of third-party code.
OFFICEBENCH_PORT_ROOT: Path = _env_path(
    "OFFICEBENCH_PORT_ROOT", EXTERNAL_ROOT / "_port" / "officebench",
)
# OfficeBench apptainer sandbox SIF (ubuntu + libreoffice + tesseract).
OFFICEBENCH_SANDBOX_SIF: Path = _env_path(
    "OFFICEBENCH_SANDBOX_SIF",
    EXTERNAL_ROOT / "officebench-sandbox" / "officebench.sif",
)
# Per-task scratch testbed root.
OFFICEBENCH_SCRATCH_ROOT: Path = _env_path(
    "OFFICEBENCH_SCRATCH_ROOT", WORK_ROOT / "officebench_scratch",
)
# Vendored KramaBench upstream: `workload/<domain>.json` task specs plus `data/<domain>/input/` per-domain data lakes (~1.3 GB, usually symlinked from bulk storage). Real test set; the upstream deterministic per-task metric is ported in the env.
KRAMABENCH_PORT_ROOT: Path = _env_path(
    "KRAMABENCH_PORT_ROOT", EXTERNAL_ROOT / "_port" / "kramabench",
)
# Per-domain data lake (read-only). Defaults to the symlink under the port root; override to point straight at a clone.
KRAMABENCH_DATA_ROOT: Path = _env_path(
    "KRAMABENCH_DATA_ROOT", KRAMABENCH_PORT_ROOT / "data",
)
# Per-task scratch workspace root (writable).
KRAMABENCH_SCRATCH_ROOT: Path = _env_path(
    "KRAMABENCH_SCRATCH_ROOT", WORK_ROOT / "kramabench_scratch",
)

# AppWorld's `--root`: the dir that contains `data/`, not `data/` itself. appworld appends `/data` on its own, so pointing this at `data/` yields a double `data/data` and `verify_data_exists()` fails.
APPWORLD_DATA_ROOT: Path = _env_path(
    "APPWORLD_DATA_ROOT",
    EXTERNAL_ROOT / "appworld-data",
)

# LLM disk cache.
LLM_CACHE_DIR: Path = _env_path(
    "SKILL_INDUCTION_CACHE_DIR",
    Path.home() / ".cache" / "skill_induction" / "llm_cache",
)


def ensure_dirs() -> None:
    """Create the writable dirs if missing."""
    for p in (DATA_DIR, RESULTS_DIR, PILOTS_DIR, APPWORLD_SERVER_CWD, LLM_CACHE_DIR):
        p.mkdir(parents=True, exist_ok=True)


def cell_result_path(
    benchmark: str, agent: str, model: str, stage: str = "phase0",
    suffix: str = "",
) -> Path:
    """Canonical result-pkl path; the memory companion is the same path with a `_memory.pkl` suffix."""
    name = f"{benchmark}_{agent}_{model}_{stage}{suffix}.pkl"
    return RESULTS_DIR / name


def cell_memory_path(
    benchmark: str, agent: str, model: str, stage: str = "phase0",
    suffix: str = "",
) -> Path:
    name = f"{benchmark}_{agent}_{model}_{stage}{suffix}_memory.pkl"
    return RESULTS_DIR / name

#!/usr/bin/env python3
"""Share of induced skills whose source task was not solved.

For each of the four memory conditions (induction level x skill format), every
skill in a run record's `skills_induced` counts toward that record's task, and
the task counts as solved iff the record's eval score is > 0 (KramaBench keeps
its continuous native score in `eval_extras['score']`; AppWorld and OfficeBench
fall back to the binary `success` flag). The four percentages are pooled over
the 3 benchmarks x 11 models of the paper grid, on the same stage basis as the
paper table: `final_v1` everywhere except subtask+code, which uses `noprune_v1`.

Usage: python3 scripts/unsolved_skill_ratio.py (from the repo root, once the
results pipeline is idle; --one <pkl> checks the counting on a single file).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"

BENCHES = ("appworld", "officebench", "kramabench")

MODELS = (
    "gemini-3.1-pro-gcp",
    "gemma-3-4b-local",
    "gemma-3-12b-local",
    "gemma-3-27b-local",
    "gpt-oss-120b-amazon",
    "nemotron-3-super-120b-amazon",
    "qwen-3-4b-local",
    "qwen-3-8b-local",
    "qwen-3-14b-local",
    "qwen-3-32b-local",
    "qwen-3-235b-a22b-amazon",
)

# (label pair, filename condition, stage): the paper-table basis.
CONDS = (
    (("task", "text"), "linear_awm", "final_v1"),
    (("task", "code"), "linear_asi", "final_v1"),
    (("subtask", "text"), "subtask_awm", "final_v1"),
    (("subtask", "code"), "subtask_asi", "noprune_v1"),
)


def score(record: dict) -> float:
    """Continuous native score, falling back to the binary success flag."""
    extras = record.get("eval_extras") or {}
    return extras.get("score", 1.0 if record.get("success") else 0.0)


def count_pkl(path: Path) -> tuple[int, int]:
    """(induced skills from unsolved tasks, total induced skills) in one run."""
    with open(path, "rb") as f:
        records = pickle.load(f)
    if isinstance(records, dict):
        records = list(records.values())
    unsolved = total = 0
    for r in records:
        if r.get("errored"):
            continue
        n = len(r.get("skills_induced") or [])
        if not n:
            continue
        total += n
        if score(r) <= 0:
            unsolved += n
    return unsolved, total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--one", type=Path, default=None,
                    help="count a single result pkl and exit")
    args = ap.parse_args()

    if args.one:
        unsolved, total = count_pkl(args.one)
        pct = 100.0 * unsolved / total if total else float("nan")
        print(f"{args.one.name}: {unsolved}/{total} induced skills "
              f"from unsolved tasks ({pct:.1f}%)")
        return 0

    missing = []
    print("Share of induced skills whose source task was not solved")
    print(f"(pooled over {len(BENCHES)} benchmarks x {len(MODELS)} models)\n")
    for (level, fmt), cond, stage in CONDS:
        unsolved = total = 0
        for bench in BENCHES:
            for model in MODELS:
                path = args.results_dir / f"{bench}_{cond}_{model}_{stage}.pkl"
                if not path.exists():
                    missing.append(path.name)
                    continue
                u, t = count_pkl(path)
                unsolved += u
                total += t
        pct = 100.0 * unsolved / total if total else float("nan")
        print(f"{level:8s} {fmt:4s} ({cond}, {stage}): "
              f"{pct:.1f}% of {total} induced skills ({unsolved} unsolved)")
    if missing:
        print(f"\nwarning: {len(missing)} cell files missing:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

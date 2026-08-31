"""Unified experiment runner.

One CLI; flip `--agent` and `--benchmark` to select any cell of the paper's condition grid.

Outputs:

  data/results/<benchmark>_<agent>_<model>_<stage>.pkl         (list of per-task dicts)
  data/results/<benchmark>_<agent>_<model>_<stage>_memory.pkl  (memory snapshot)

On startup the runner reads the result pkl and skips task_ids already present; pass `--override` to force a clean rerun.

Example::

    python -m skill_induction.experiments.run \
        --agent subtask_asi --benchmark appworld \
        --model qwen-3-235b-a22b-amazon --stage final_v1
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from skill_induction.caching.call_ledger import CallLedger
from skill_induction.caching.llm_cache import LLMCache
from skill_induction.experiments._components import (
    AGENT_NAMES, get_components, get_server_cls, get_world_cls,
)
from skill_induction.models.api_client import (
    BEDROCK_MODEL_CONFIG, BedrockChatGenerator,
)
from skill_induction.models.vllm_local import (
    LOCAL_MODEL_CONFIG, VLLMChatGenerator,
)
from skill_induction.models.gemini_gcp import (
    GEMINI_MODEL_CONFIG, GeminiChatGenerator,
)
from skill_induction.paths import (
    APPWORLD_DATA_ROOT, LLM_CACHE_DIR, PILOTS_DIR, RESULTS_DIR,
    cell_memory_path, cell_result_path, ensure_dirs,
)


def _load_api_config() -> dict:
    with open("api_key/config.json") as f:
        return json.load(f)["api_key"]


def _load_pilot(path: Path, num: int | None) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Pilot file missing: {path}. Curate one — one task_id per "
            "line; `#` comments and blank lines OK.")
    ids = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            ids.append(ln)
    return ids if num is None else ids[:num]


def _load_default_taskset(
    benchmark: str, num: int | None, appworld_split: str | None = None
) -> list[str]:
    """Default task set (no --pilot): every id in the benchmark's test split. For AppWorld, `appworld_split` picks 'test_normal' (168 tasks, the default) or 'test_challenge' (417). `num` caps the list; None means all."""
    if benchmark == "appworld":
        from skill_induction.paths import APPWORLD_DATA_ROOT
        # APPWORLD_DATA_ROOT is the dir that contains `data/`; the split id-lists live under `data/datasets/`.
        path = (APPWORLD_DATA_ROOT / "data" / "datasets"
                / f"{appworld_split or 'test_normal'}.txt")
        ids = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    elif benchmark == "officebench":
        from skill_induction.envs.officebench_env import OFFICEBENCH_TASKS_DIR
        ids = []
        for bucket in sorted(OFFICEBENCH_TASKS_DIR.glob("*-*"), key=lambda p: p.name):
            sub_dir = bucket / "subtasks"
            if not sub_dir.is_dir():
                continue
            for sub in sorted(sub_dir.glob("*.json")):
                ids.append(f"{bucket.name}/{sub.stem}")
    elif benchmark == "kramabench":
        # The 92 deterministic (no-LLM-judge) tasks, in stable domain order.
        from skill_induction.envs.kramabench_env import kramabench_task_ids
        ids = kramabench_task_ids(deterministic_only=True)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return ids if num is None else ids[:num]


def _shuffle_order_path(label: str, seed: int) -> Path:
    return RESULTS_DIR / f"shuffle_order_{label}_seed{seed}.json"


def _apply_shuffle(task_ids: list[str], label: str, seed: int) -> list[str]:
    """Permute the full task list deterministically and persist the order, so every condition walks the same order and task position is decorrelated from difficulty. The first run for a (label, seed) writes the JSON; later runs read it back."""
    path = _shuffle_order_path(label, seed)
    if path.exists():
        saved = json.loads(path.read_text())
        cur = set(task_ids)
        if set(saved) == cur:
            return saved
        # Taskset changed since the order was saved: keep the saved order for surviving tasks, append new ones shuffled by seed, and rewrite, so started runs stay order-consistent.
        kept = [t for t in saved if t in cur]
        extra = [t for t in task_ids if t not in set(saved)]
        random.Random(seed).shuffle(extra)
        order = kept + extra
        path.write_text(json.dumps(order, indent=0))
        print(f"[run] shuffle: taskset changed; refreshed {path.name} "
              f"({len(kept)} kept + {len(extra)} new)")
        return order
    order = list(task_ids)
    random.Random(seed).shuffle(order)
    path.write_text(json.dumps(order, indent=0))
    print(f"[run] shuffle: generated {path.name} ({len(order)} tasks, seed={seed})")
    return order


def _save_pkl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)


def _load_pkl(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        return default


# Labels that select a split of a benchmark: label -> (canonical benchmark, appworld split stem). Labels affect only output-file naming; behaviour uses the canonical benchmark.
_BENCHMARK_ALIASES = {
    "appworld":        ("appworld", "test_challenge"),
    "appworld-normal": ("appworld", "test_normal"),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", required=True, choices=AGENT_NAMES)
    p.add_argument("--benchmark", default="appworld",
                   choices=("appworld", "appworld-normal",
                            "officebench", "kramabench"))
    p.add_argument("--model", default="gpt-oss-120b-amazon",
                   choices=list(BEDROCK_MODEL_CONFIG) + list(LOCAL_MODEL_CONFIG)
                           + list(GEMINI_MODEL_CONFIG))
    p.add_argument("--num-tasks", "-N", type=int, default=None,
                   help="Number of tasks to run. With `--pilot`, defaults "
                        "to the full pilot file. Without `--pilot`, defaults "
                        "to the full benchmark test split.")
    p.add_argument("--max-iter", type=int, default=50,
                   help="Global per-task iteration cap.")
    p.add_argument("--stage", default="phase1_smoke",
                   help="Used in the pkl filename. e.g. 'phase0', "
                        "'phase1_smoke', 'table1', 'table2'.")
    p.add_argument("--pilot", action="store_true",
                   help="Use the curated pilot list at "
                        "data/pilots/<benchmark>.txt. Without this flag, the "
                        "runner uses the benchmark's full test split.")
    p.add_argument("--pilot-file", type=Path, default=None,
                   help=f"Override the pilot path "
                        f"(default: {PILOTS_DIR}/<benchmark>.txt).")
    p.add_argument("--override", action="store_true",
                   help="Discard the existing result pkl before running.")
    p.add_argument("--induction-level", type=int, default=3, choices=(1, 2, 3),
                   help="Prompt-component ablation for skill induction "
                        "(memory agents only): 1=instruction-only, "
                        "2=instruction+demonstration, 3=+abstraction-rules "
                        "(the deployed prompt; default). No effect on the "
                        "memory-free linear/subtask agents.")
    p.add_argument("--shuffle-seed", type=int, default=None,
                   help="Deterministically permute the FULL task order and "
                        "persist it to data/results/shuffle_order_<label>_seed<seed>.json "
                        "(shared by all conditions), so task position is "
                        "decorrelated from difficulty for learning-curve "
                        "analysis. -N then takes the first N of the shuffled order.")
    p.add_argument("--subtask-skills", action="store_true",
                   help="Skill-transfer arm: run the LINEAR agent on the "
                        "matching SUBTASK agent's skill library (linear_awm "
                        "← subtask_awm, linear_asi ← subtask_asi; same model/"
                        "benchmark). Online replay: entering each task, the "
                        "memory is set to the exact library the subtask donor "
                        "run had entering that task (no self- or future-skill "
                        "leakage). Induction is frozen. Legacy alias of "
                        "`--skills-level subtask`.")
    p.add_argument("--skills-level", choices=("task", "subtask"), default=None,
                   help="Skill-transfer arm: granularity of the donor "
                        "library the LINEAR agent consumes. 'subtask' = the "
                        "matching subtask agent's atomic library (linear_awm "
                        "← subtask_awm, linear_asi ← subtask_asi). 'task' = "
                        "the linear agent's OWN coarse library from the "
                        "donor stage. Either way the library is replayed per "
                        "task (donor's entering snapshot, no self- or "
                        "future-skill leakage) and induction is frozen.")
    p.add_argument("--skills-half", choices=("all", "high", "low"),
                   default="all",
                   help="Which half of the donor library to load, split at "
                        "the median skill utility (specificity x "
                        "abstractness; see analysis/skill_utility.py). "
                        "'high' keeps the top half, 'low' the bottom half; "
                        "'all' (default) is the whole library. Requires "
                        "--skills-level.")
    p.add_argument("--skills-stage", "--subtask-skills-stage",
                   dest="donor_stage", default="final_v1",
                   help="Stage of the donor run whose per-task memory "
                        "snapshots are replayed (default: final_v1).")
    p.add_argument("--skills-no-threshold", action="store_true",
                   help="Replay arms only: drop the retrieval similarity "
                        "floor (MIN_SIMILARITY) so retrieval is pure top-k. "
                        "Needed for the --skills-half comparison: "
                        "high-utility skills are broad (abstractness rewards "
                        "spread), so their nearest-task similarity sits "
                        "below the floor and the high arm silently starves; "
                        "this equalizes delivery so the halves differ only "
                        "in the skills themselves. Requires --skills-level.")
    args = p.parse_args()
    if args.subtask_skills:
        if args.skills_level not in (None, "subtask"):
            p.error("--subtask-skills is the legacy alias of "
                    "`--skills-level subtask`; don't combine it with "
                    "--skills-level task")
        args.skills_level = "subtask"
    if args.skills_level and args.agent not in ("linear_awm", "linear_asi"):
        p.error("--skills-level requires --agent linear_awm or linear_asi")
    if args.skills_half != "all" and not args.skills_level:
        p.error("--skills-half requires --skills-level")
    if args.skills_no_threshold and not args.skills_level:
        p.error("--skills-no-threshold requires --skills-level")

    ensure_dirs()
    # Output files are named by the label; everything downstream uses the canonical benchmark, so resolve once here.
    label = args.benchmark
    args.benchmark, appworld_split = _BENCHMARK_ALIASES.get(label, (label, None))
    args.label = label   # stored in records / used for output naming (matches pkl)
    out_path = cell_result_path(label, args.agent, args.model, args.stage)
    mem_path = cell_memory_path(label, args.agent, args.model, args.stage)

    components = get_components(args.agent)
    AgentCls = components["agent_cls"]
    memory_factory = components["memory_factory"]
    WorldCls = get_world_cls(args.benchmark)
    ServerCls = get_server_cls(args.benchmark)

    # SKILL_INDUCTION_NO_CACHE=1 disables disk caching (avoids stale replays across runs).
    cache = None if os.environ.get("SKILL_INDUCTION_NO_CACHE") in (
        "1", "true", "True") else LLMCache(root=LLM_CACHE_DIR)
    if args.model in LOCAL_MODEL_CONFIG:
        model = VLLMChatGenerator(model_id=args.model, cache=cache)
        print(f"[run] backend=vLLM  served={model.served_model}  "
              f"url={model.base_url}")
    elif args.model in GEMINI_MODEL_CONFIG:
        # Gemini via Vertex AI: auth is ADC; config.json supplies only project/location, no secrets.
        api_cfg = _load_api_config()
        model = GeminiChatGenerator(
            model_id=args.model, cache=cache,
            project=api_cfg.get("gcp_project"),
            location=api_cfg.get("gcp_location", "global"),
        )
        print(f"[run] backend=Gemini/Vertex  model={model.gemini_model_id}  "
              f"project={model.project}  location={model.location}")
    else:
        api_cfg = _load_api_config()
        model = BedrockChatGenerator(
            model_id=args.model, cache=cache,
            region=api_cfg.get("bedrock_region", "us-west-2"),
            bearer_token=api_cfg.get("bedrock_api_key"),
        )

    if args.override and out_path.exists():
        out_path.unlink()
        if mem_path.exists():
            mem_path.unlink()
    results: list[dict] = _load_pkl(out_path, [])
    # Errored records are persisted but excluded from resume so the next run retries them. SKILL_INDUCTION_SKIP_ERRORED=1 treats them as done, for tasks whose env deterministically crashes the process (officebench "File not exist" faults).
    _skip_err = os.environ.get("SKILL_INDUCTION_SKIP_ERRORED") in (
        "1", "true", "True")
    completed_ids = {r["task_id"] for r in results
                     if _skip_err or not r.get("errored")}
    errored_ids = {r["task_id"] for r in results if r.get("errored")}

    memory = memory_factory(model) if memory_factory else None
    # Skill-transfer arm: donor_snapshots[task_id] is the library the donor run had entering that task, so the consumer sees no self-induced or future skills. With --skills-half, each snapshot is then cut down to one median-utility half of the donor's skill pool.
    donor_snapshots = None
    # Record fields (None / 'all' outside the arm).
    args.skills_from = args.skills_stage = None
    if not args.skills_level:
        args.skills_half = "all"
    # 'subtask' would match the demonstration's granularity to the library's; in this codebase the linear demo already IS the stitched executor trajectory, so the knob is a no-op kept for interface parity (see LinearAgent._apply_frozen_skills_prompt).
    skills_demo = ("subtask" if args.skills_level == "subtask"
                   and not args.subtask_skills else "linear")
    if args.skills_level:
        donor_agent = ({"linear_awm": "subtask_awm",
                        "linear_asi": "subtask_asi"}[args.agent]
                       if args.skills_level == "subtask" else args.agent)
        args.skills_from = donor_agent
        args.skills_stage = args.donor_stage
        donor_path = cell_result_path(
            label, donor_agent, args.model, args.donor_stage)
        if donor_path == out_path:
            print(f"FATAL: donor and output are the same cell "
                  f"({donor_path.name}); give this run a --stage distinct "
                  f"from --skills-stage.", file=sys.stderr)
            return 2
        donor_recs = [r for r in _load_pkl(donor_path, [])
                      if not r.get("errored")]
        if not donor_recs:
            print(f"FATAL: --skills-level donor pkl missing or empty: "
                  f"{donor_path}", file=sys.stderr)
            return 2
        by_idx = {r["task_idx"]: r for r in donor_recs}
        donor_snapshots = {}
        for r in donor_recs:
            j = r["task_idx"] - 1
            # Walk past errored-donor gaps to the nearest earlier completed record; empty library before task 0.
            while j >= 0 and j not in by_idx:
                j -= 1
            prev = by_idx.get(j) if j >= 0 else None
            donor_snapshots[r["task_id"]] = (
                (prev or {}).get("memory_snapshot")
                or {"skills": [], "provenance_log": []})
        if args.skills_no_threshold and memory is not None:
            # Pure top-k delivery: disarm the similarity floor. Both memories gate on the instance attribute (ASI `>=`, AWM `>`), and best_scores bottoms out at -1.0, so -2.0 passes always.
            memory.MIN_SIMILARITY = -2.0
        print(f"[run] skills replay: level={args.skills_level} "
              f"half={args.skills_half} "
              f"threshold={'OFF (pure top-k)' if args.skills_no_threshold else 'on'} "
              f"donor={donor_path.name} ({len(donor_recs)} records) — "
              f"per-task entering library, induction frozen")
        if args.skills_half != "all":
            donor_snapshots = _apply_utility_half(
                donor_snapshots, donor_recs, args.skills_half,
                cell_result_path(label, args.agent, args.model, args.stage,
                                 suffix="_utilsplit"))
    # The skill-transfer arms all share an --agent, so only the --stage keeps their pkls apart. Refuse to append to a pkl written by a DIFFERENT arm (which would also make resume skip its tasks).
    prior_arms = ({(r.get("skills_from"), r.get("skills_half") or "all",
                    bool(r.get("skills_no_threshold"))) for r in results}
                  - {(args.skills_from, args.skills_half,
                      args.skills_no_threshold)})
    if prior_arms:
        print(f"FATAL: {out_path.name} already holds records from another "
              f"arm {sorted(map(str, prior_arms))}; this run is "
              f"({args.skills_from}, {args.skills_half}). Give it its own "
              f"--stage.", file=sys.stderr)
        return 2
    if memory is not None and mem_path.exists() and not args.skills_level:
        snap = _load_pkl(mem_path, None)
        if isinstance(snap, dict):
            payload = snap.get("snapshot") if "snapshot" in snap else snap
            if payload:
                memory.load_snapshot(payload)

    # Build the full task list, optionally shuffle and persist, then apply -N, so -N takes the first N of the shuffled order and every cell agrees on the order.
    if args.pilot:
        pilot_path = args.pilot_file or (PILOTS_DIR / f"{args.benchmark}.txt")
        task_ids = _load_pilot(pilot_path, None)
        task_source = f"pilot={pilot_path}"
    else:
        task_ids = _load_default_taskset(args.benchmark, None, appworld_split)
        split_note = f" [{appworld_split}]" if appworld_split else ""
        task_source = f"benchmark test split{split_note}"
    if args.shuffle_seed is not None:
        task_ids = _apply_shuffle(task_ids, args.label, args.shuffle_seed)
        task_source += f" · shuffled(seed={args.shuffle_seed})"
    if args.num_tasks is not None:
        task_ids = task_ids[:args.num_tasks]
    task_source += f" (N={args.num_tasks or 'all'})"
    if donor_snapshots is not None:
        missing = [t for t in task_ids if t not in donor_snapshots]
        if missing:
            print(f"FATAL: skills donor lacks {len(missing)} of "
                  f"the selected tasks (first: {missing[:3]}); finish the "
                  f"donor run first.", file=sys.stderr)
            return 2
    print(f"[run] agent={args.agent}  benchmark={args.benchmark}  model={args.model}")
    print(f"[run] stage={args.stage}  {task_source}  tasks={len(task_ids)}")
    print(f"[run] resume: {len(completed_ids)} task(s) already in {out_path}")
    print(f"[run] LLM cache: {cache.root if cache else '(disabled — SKILL_INDUCTION_NO_CACHE=1)'}")

    if args.benchmark == "appworld":
        server_kwargs = {"data_root": APPWORLD_DATA_ROOT}
    else:
        server_kwargs = {}

    n_total = len(task_ids)
    cell = f"{args.benchmark}/{args.agent}/{args.model}"
    n_skip = n_run = n_succ = n_comp = n_err = 0
    server = ServerCls(**server_kwargs)
    with server as server_url:
        for idx, task_id in enumerate(task_ids):
            pos = f"{idx + 1}/{n_total}"
            if task_id in completed_ids:
                n_skip += 1
                print(f"  [{pos}] {task_id}: skipped (already done) "
                      f"· progress {idx + 1}/{n_total}")
                continue
            # Restart a dead env server (e.g. OOM-killed) before each task so one crash doesn't cascade; no-op for benchmarks whose server is a noop.
            if hasattr(server, "ensure_alive"):
                try:
                    server_url = server.ensure_alive()
                except Exception as _e:
                    print(f"    WARN: could not (re)start env server: {_e}")
            print(f"  [{pos}] {task_id}: running...")
            ledger = CallLedger(str(task_id))
            if donor_snapshots is not None:
                # Replay: load the donor's entering library; drop its provenance log so this run records only its own retrievals.
                memory.load_snapshot(donor_snapshots[task_id])
                memory.provenance_log = []
            pre_skill_ids = _skill_ids(memory)
            if memory is not None and hasattr(memory, "set_current_task"):
                memory.set_current_task(task_id)
            model.ledger = ledger
            # Scope the cache so only an exact same-cell+task replay can hit. The skill-transfer arms share an --agent with the plain cell, and early tasks (empty library) can produce a byte-identical prompt, so the arm is part of the scope too.
            arm = (f"|{args.skills_level}-{args.skills_half}"
                   f"{'-nt' if args.skills_no_threshold else ''}"
                   if args.skills_level else "")
            model.cache_namespace = (
                f"{args.benchmark}|{args.agent}{arm}|{args.model}|{task_id}")
            try:
                world_kwargs = dict(
                    task_id=task_id,
                    experiment_name=f"{args.stage}_{args.agent}_{args.model}_{task_id}",
                    server_url=server_url,
                    group_key=args.benchmark,
                )
                if args.benchmark == "appworld":
                    world_kwargs["data_root"] = APPWORLD_DATA_ROOT
                # Retry transient env/server errors (e.g. appworld HTTP 500 on /initialize) after an env restart; SKILL_INDUCTION_TASK_RETRIES tunes it (default 2 retries = 3 attempts).
                _tries = int(os.environ.get("SKILL_INDUCTION_TASK_RETRIES", "2")) + 1
                for _attempt in range(_tries):
                    try:
                        world_kwargs["server_url"] = server_url
                        # The skills-replay kwargs are only passed when the arm is on; the subtask agents' __init__ doesn't accept them.
                        agent_kwargs = (
                            {"skills_replay": True,
                             "skills_half": args.skills_half,
                             "skills_demo": skills_demo}
                            if args.skills_level else {})
                        with WorldCls(**world_kwargs) as world:
                            agent = AgentCls(
                                model=model, benchmark=args.benchmark,
                                memory=memory, verbose=False,
                                induction_level=args.induction_level,
                                **agent_kwargs,
                            )
                            metrics = agent.run_task(world, max_iter=args.max_iter)
                        break
                    except Exception as _te:
                        if _attempt + 1 >= _tries:
                            raise
                        print(f"    [{pos}] {task_id}: attempt {_attempt + 1}/{_tries} "
                              f"failed ({_te}); restarting env + retrying")
                        if hasattr(server, "ensure_alive"):
                            try:
                                server_url = server.ensure_alive()
                            except Exception as _e2:
                                print(f"      env restart failed: {_e2}")
                        time.sleep(3)
            # SystemExit included: agent-generated code run in-process may call sys.exit(), which would otherwise kill the run with no error record; KeyboardInterrupt still propagates.
            except (Exception, SystemExit) as e:
                n_run += 1
                n_err += 1
                tb = traceback.format_exc()
                print(f"    ERROR: {e}\n{tb}")
                err_record = _error_record(
                    args, task_id, idx, ledger, str(e), tb)
                # Replace any prior errored row for this task_id, so retries don't duplicate rows.
                results = [r for r in results if r["task_id"] != task_id]
                results.append(err_record)
                _save_pkl(out_path, results)
                was_retry = task_id in errored_ids
                print(f"  [{pos}] {task_id}: ERRORED{'  (retry)' if was_retry else ''} · "
                      f"ok {n_succ}/{n_run} ran · errors {n_err} · {n_total} total")
                continue
            finally:
                model.ledger = None
                model.cache_namespace = ""

            record = _record(args, task_id, idx, metrics, memory,
                              ledger, pre_skill_ids)
            # Replace any prior errored row for this task_id, so a retry that succeeded overwrites the failure.
            results = [r for r in results if r["task_id"] != task_id]
            results.append(record)
            _save_pkl(out_path, results)
            # No memory file for the replay arm; its library is the donor's, never evolved.
            if memory is not None and donor_snapshots is None:
                _save_pkl(mem_path, memory.snapshot())
            n_run += 1
            n_succ += int(bool(metrics["success"]))
            n_comp += int(bool(metrics["completed"]))
            mark = "✓" if metrics["success"] else "✗"
            print(
                f"  [{pos}] {task_id} {mark} "
                f"success={metrics['success']} completed={metrics['completed']} "
                f"k={metrics.get('subtask_count', 1)} "
                f"steps={metrics['env_steps']} "
                f"ptok={metrics['total_prompt_tokens']} "
                f"│ cell so far: ✓{n_succ} / ran {n_run} "
                f"(+{n_skip} skipped) / {n_total} total"
            )

    if cache is not None:
        stats = cache.stats()
        print(f"\n[run] LLM cache hits={stats['hits']} misses={stats['misses']} "
              f"({stats['hit_rate']:.1%})")
    else:
        print("\n[run] LLM cache disabled")
    print(f"[run] results → {out_path} ({len(results)} records)")
    if memory is not None and donor_snapshots is None:
        print(f"[run] memory  → {mem_path} ({len(memory.get_active_skill_names())} skills)")
    # Grep-friendly per-cell summary line.
    print(f"[CELL DONE] {cell} :: success {n_succ}/{n_total} · "
          f"completed {n_comp}/{n_total} · ran {n_run} · "
          f"skipped {n_skip} · errors {n_err}")
    return 0


def _apply_utility_half(donor_snapshots: dict, donor_recs: list[dict],
                        half: str, split_path: Path) -> dict:
    """Cut every replayed snapshot down to one median-utility half of the donor's skill pool.

    The pool is the union of every skill that appears in any entering snapshot (i.e. everything the arm could ever retrieve), scored against the donor cell's task instructions, so a skill's half is fixed for the whole run rather than re-decided as the library grows. The two halves are exact complements. The split is written to `split_path` for the analysis to join on.
    """
    from skill_induction.analysis.skill_utility import (
        split_library, task_instruction,
    )
    pool: dict = {}
    for snap in donor_snapshots.values():
        for s in (snap or {}).get("skills") or []:
            if s.get("id") is not None:
                pool.setdefault(s["id"], s)
    instructions = [t for t in (task_instruction(r) for r in donor_recs) if t]
    split = split_library(pool.values(), instructions)
    keep = split[half]
    out = {}
    for task_id, snap in donor_snapshots.items():
        skills = [s for s in (snap or {}).get("skills") or []
                  if s.get("id") in keep]
        out[task_id] = {**(snap or {}), "skills": skills}
    _save_pkl(split_path, {
        "half": half, "median_utility": split["median"],
        "utility": split["utility"],
        "high_ids": sorted(map(str, split["high"])),
        "low_ids": sorted(map(str, split["low"])),
    })
    sizes = [len(s["skills"]) for s in out.values()]
    print(f"[run] utility split: pool={len(pool)} skills scored over "
          f"{len(instructions)} task instructions · median="
          f"{split['median']:.4f} · keeping the {half} half "
          f"({len(keep)} skills; per-task library "
          f"{min(sizes or [0])}-{max(sizes or [0])}) → {split_path.name}")
    return out


def _skill_objs(memory) -> list[dict]:
    """Full skill dicts from the memory snapshot ({} if no memory)."""
    if memory is None:
        return []
    try:
        return list((memory.snapshot() or {}).get("skills", []))
    except Exception:
        return []


def _skill_ids(memory) -> set:
    return {s.get("id") for s in _skill_objs(memory)}


def _error_record(args, task_id, idx, ledger, error_msg: str, tb: str) -> dict:
    """Failure-path counterpart to `_record`: keeps one pkl row per attempted task and preserves the partial ledger; `errored=True` makes resume retry the task."""
    lsum = ledger.summary() if ledger else {}
    return {
        "task_id": task_id,
        "task_idx": idx,
        "benchmark": args.label,
        "agent": args.agent,
        "model": args.model,
        "stage": args.stage,
        # Skill-transfer arm provenance (None / 'all' outside the arm).
        "skills_from": args.skills_from,
        "skills_stage": args.skills_stage,
        "skills_half": args.skills_half,
        "skills_no_threshold": args.skills_no_threshold,
        "errored": True,
        "error_message": error_msg,
        "traceback": tb,
        "completed": False,
        "success": False,
        "tgc_score": 0.0,
        "subtask_count": 0,
        "total_prompt_tokens": lsum.get("ledger_prompt_tokens", 0) or 0,
        "total_completion_tokens": lsum.get("ledger_completion_tokens", 0) or 0,
        "agentloop_prompt_tokens": 0,
        "agentloop_completion_tokens": 0,
        "peak_prompt_tokens": 0,
        "env_steps": 0,
        "wall_clock_seconds": 0.0,
        "difficulty": 0,
        "library_size_at_start": 0,
        "retrieved_skill_ids": [],
        "invoked_skill_ids": [],
        "skills_induced": [],
        "subtask_plans": [],
        "subtask_summaries": [],
        "trace": [],
        "final_history": [],
        "eval_extras": {},
        "llm_calls": ledger.entries if ledger else [],
        "llm_call_summary": lsum,
        "memory_snapshot": None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def _record(args, task_id, idx, metrics, memory, ledger, pre_skill_ids) -> dict:
    post = _skill_objs(memory)
    # Skills induced this task = snapshot skills whose id wasn't present before run_task.
    induced = [
        {
            "id": s.get("id"),
            "type": s.get("type"),
            "description": s.get("description"),
            "source": s.get("source"),
            "source_task_id": s.get("source_task_id"),
            "source_subtask_idx": s.get("source_subtask_idx"),
        }
        for s in post if s.get("id") not in pre_skill_ids
    ]
    retrieved = []
    retrieved_full = []
    if memory is not None:
        log = getattr(memory, "provenance_log", None) or []
        if log:
            retrieved = list(log[-1].get("retrieved", []) or [])
            # {id, name, source_task_id} per retrieved skill, captured at retrieval time; the analysis joins invoked names to ids via this.
            retrieved_full = list(log[-1].get("retrieved_full", []) or [])
    # Map invoked function names to retrieved-skill ids. Only retrieved skills count as invoked; AWM notes aren't callable, so this stays empty for AWM.
    invoked_names = set(metrics.get("invoked_skill_names", []) or [])
    name2id = {}
    for d in retrieved_full:
        name2id.setdefault(d.get("name"), d.get("id"))
    invoked_skill_ids = [name2id[n] for n in invoked_names
                         if n in name2id and name2id[n] is not None]
    # The ledger (every chat() call) is the authoritative cost total; the agent-loop counter omits the induction call, so it under-counts for awm/asi cells.
    lsum = ledger.summary() if ledger else {}
    tot_p = lsum.get("ledger_prompt_tokens") or metrics["total_prompt_tokens"]
    tot_c = lsum.get("ledger_completion_tokens") or metrics["total_completion_tokens"]
    rec = {
        "task_id": task_id,
        "task_idx": idx,
        "benchmark": args.label,
        "agent": args.agent,
        "model": args.model,
        "stage": args.stage,
        # Skill-transfer arm provenance (None / 'all' outside the arm).
        "skills_from": args.skills_from,
        "skills_stage": args.skills_stage,
        "skills_half": args.skills_half,
        "skills_no_threshold": args.skills_no_threshold,
        "completed": metrics["completed"],
        "success": metrics["success"],
        "tgc_score": metrics["tgc_score"],
        "subtask_count": metrics.get("subtask_count", 1),
        "total_prompt_tokens": tot_p,
        "total_completion_tokens": tot_c,
        # agent-loop subset (excludes induction) kept for transparency
        "agentloop_prompt_tokens": metrics["total_prompt_tokens"],
        "agentloop_completion_tokens": metrics["total_completion_tokens"],
        "peak_prompt_tokens": metrics["peak_prompt_tokens"],
        "env_steps": metrics["env_steps"],
        "wall_clock_seconds": metrics["wall_clock_seconds"],
        "difficulty": metrics.get("difficulty", 0),
        "library_size_at_start": len(pre_skill_ids),
        "retrieved_skill_ids": retrieved,
        # Per-retrieval provenance; source_task_id excludes within-task self-retrieval and disambiguates name collisions across domains.
        "retrieved_skills": retrieved_full,
        # Retrieved skills actually called in executed code this task (ASI only; [] for AWM).
        "invoked_skill_ids": invoked_skill_ids,
        "skills_induced": induced,
        "subtask_plans": metrics.get("subtask_plans", []),
        "subtask_summaries": metrics.get("subtask_summaries", []),
        "trace": metrics["trace"],
        "final_history": metrics.get("final_history", []),
        "eval_extras": metrics.get("eval_extras", {}),
        # Full per-call LLM ledger: a superset of final_history that preserves pre-summarisation content.
        "llm_calls": ledger.entries if ledger else [],
        "llm_call_summary": ledger.summary() if ledger else {},
        # Full skill library after this task.
        "memory_snapshot": (memory.snapshot() if memory else None),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return rec


if __name__ == "__main__":
    sys.exit(main())

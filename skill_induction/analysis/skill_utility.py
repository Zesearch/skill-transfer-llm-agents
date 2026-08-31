"""Skill utility: the library-quality metric, and the high/low split of a donor library built from it.

utility(skill) = specificity x abstractness. Both components are parameter-free and bounded in [0, 1], and both are computed from the cosine similarities c_1..c_N between the skill's description and the N task instructions of the benchmark/model cell:

  specificity  = P[ max_j c_j >= cos(t_i, t_k) ] over all task pairs (i, k): how much closer the skill sits to its nearest task than two tasks sit to each other;
  abstractness = exp(H(softmax(c / TAU))) / N: the effective fraction of tasks the skill's relevance spreads over.

Formula, embedder (all-MiniLM-L6-v2, the same one the memories retrieve with) and 500-char description truncation match the paper's skill-quality analysis, so a split computed here lines up with the reported numbers.

Used by `experiments/run.py --skills-half {high,low}` to hand a linear agent one half of a donor library and measure what library quality is worth, holding the tasks and the agent fixed.
"""

from __future__ import annotations

import math

import numpy as np

TAU = 0.1
DESC_TRUNC = 500          # descriptions are compared truncated, as in the paper
# Substrings that mark a user message as demo scaffolding rather than the real task (the example separators around the few-shot block).
_EXAMPLE_MARKERS = ("EXAMPLE", "illustrative trajectory", "reference examples")
# Linear rows frame the task as the executor's single subgoal; subtask planner rows carry a `Task:` line instead.
_SUBGOAL_HDR = "Your current subgoal:"
_OVERALL_HDR = "\n\nOverall task (for context only):"


def _encode(texts: list[str]) -> np.ndarray:
    """L2-normalised embeddings from the shared sentence-transformer."""
    from skill_induction.memory._embed import _model
    m = _model()
    if m == "lexical":
        raise RuntimeError(
            "skill utility needs sentence-transformers (all-MiniLM-L6-v2); "
            "the lexical retrieval fallback cannot produce embeddings.")
    return np.asarray(m.encode(list(texts), normalize_embeddings=True),
                      dtype=np.float32)


def task_instruction(record: dict) -> str | None:
    """The task instruction of a result record, read back out of its `final_history`: the first real user message after the few-shot demo block, stripped of the identity preamble and of any trailing scaffolding (overall-task echo, progress summary, tool preamble)."""
    users = [m.get("content") for m in (record.get("final_history") or [])
             if m.get("role") == "user" and m.get("content")]
    real = None
    seen_examples = False
    for u in users:
        if "END OF ALL EXAMPLES" in u:
            seen_examples = True
            continue
        if seen_examples and not any(k in u for k in _EXAMPLE_MARKERS):
            real = u
            break
    if real is None:
        for u in users:
            if not any(k in u for k in _EXAMPLE_MARKERS):
                real = u
                break
    if real is None:
        return None
    text = real
    if _SUBGOAL_HDR in text:
        # Linear rows: the instruction sits between the subgoal header and its overall-task echo.
        text = text.split(_SUBGOAL_HDR, 1)[1]
        text = text.split(_OVERALL_HDR, 1)[0]
    elif "Task:" in text:
        text = text.split("Task:", 1)[1]
    for stop in ("Current Progress", "\n\nYou have access",
                 "Available tools", "In `run_code`"):
        if stop in text:
            text = text.split(stop, 1)[0]
    return text.strip()[:DESC_TRUNC] or None


def utility_scores(descriptions, task_instructions) -> dict[str, tuple]:
    """{truncated description: (specificity, abstractness, utility)}."""
    descs = list(dict.fromkeys(d[:DESC_TRUNC] for d in descriptions if d))
    tasks = list(dict.fromkeys(t for t in task_instructions if t))
    if not descs or not tasks:
        return {}
    T = _encode(tasks)
    if len(tasks) > 1:
        pairs = np.sort((T @ T.T)[np.triu_indices(len(tasks), 1)])
    else:
        pairs = np.array([0.3], dtype=np.float32)
    C = _encode(descs) @ T.T
    out = {}
    for desc, c in zip(descs, C):
        z = c / TAU
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        entropy = -float(np.sum(p * np.log(np.maximum(p, 1e-12))))
        abstractness = math.exp(entropy) / len(c)
        specificity = float(
            np.searchsorted(pairs, max(float(c.max()), 0.0), side="right")
        ) / len(pairs)
        out[desc] = (specificity, abstractness, specificity * abstractness)
    return out


def split_library(skills, task_instructions) -> dict:
    """Split a pool of skills at the median utility.

    `skills` is any iterable of skill dicts (snapshot form: `id`, `description`). Returns::

        {"high": {id, ...}, "low": {id, ...}, "utility": {id: float},
         "median": float}

    Ranked, not thresholded: the skills are ordered by utility (ties broken by id, so the split is deterministic) and cut in the middle, which makes the two halves exact complements of equal size; an odd skill count puts the extra skill in the high half. A skill whose description cannot be scored (empty description) is dropped from both halves.
    """
    by_id: dict[str, str] = {}
    for s in skills:
        sid = s.get("id")
        desc = (s.get("description") or "")[:DESC_TRUNC]
        if sid is None or not desc:
            continue
        by_id.setdefault(sid, desc)
    scores = utility_scores(by_id.values(), task_instructions)
    util = {sid: scores[desc][2] for sid, desc in by_id.items()
            if desc in scores}
    order = sorted(util, key=lambda sid: (-util[sid], str(sid)))
    n_high = (len(order) + 1) // 2
    high, low = set(order[:n_high]), set(order[n_high:])
    median = float(np.median([util[s] for s in order])) if order else float("nan")
    return {"high": high, "low": low, "utility": util, "median": median}

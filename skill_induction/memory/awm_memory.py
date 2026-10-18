"""AWMMemory: workflow-text skill memory."""

from __future__ import annotations

import uuid
from typing import Optional

import numpy as np

from skill_induction.memory.base import Skill, SkillMemory
from skill_induction.memory._embed import _model


class AWMMemory(SkillMemory):
    MAX_ENTRIES = 30
    MAX_LINES = 15
    # Retrieval returns the top-k skills above this cosine similarity
    MIN_SIMILARITY = 0.30
    MERGE_THRESHOLD = 0.85

    def __init__(self, model=None, *, benchmark: str = "appworld",
                 top_k: int = 5, max_tokens: int = 8192):
        self.model = model
        self.benchmark = benchmark
        self.top_k = top_k
        self.max_tokens = max_tokens
        self._skills: list[Skill] = []
        # Cached embedding for each skill's description (parallel list, populated only at apply_skill_delta / load_snapshot).
        self._entry_embeddings: list = []
        # Provenance log, one row per task: {task, retrieved, created}.
        self.provenance_log: list[dict] = []
        self._current_task_id: Optional[str] = None

    # per-task hook

    def set_current_task(self, task_id) -> None:
        """Call at the start of each task to enable provenance tracking."""
        self._current_task_id = task_id
        self.provenance_log.append(
            {"task": task_id, "retrieved": [], "created": []})

    # embedder helper

    @property
    def embedder(self):
        m = _model()
        return None if m == "lexical" else m

    # retrieve

    def _current_domain(self) -> str | None:
        t = self._current_task_id or ""
        return t.split("-")[0] if t else None

    def _eligible_indices(self) -> list[int]:
        """Candidate pool.

        For KramaBench, gate to notes induced in the same domain (group_key): domains share no data, so a cross-domain note is noise. Other benchmarks are ungated.
        """
        idxs = list(range(len(self._skills)))
        if self.benchmark == "kramabench":
            d = self._current_domain()
            if d:
                idxs = [i for i in idxs
                        if (self._skills[i].source_task_id or "").split("-")[0] == d]
        return idxs

    def retrieve(self, query: str,
                 extra_queries: list[str] | None = None) -> str:
        if not self._skills:
            return ""
        top_k = self.top_k
        eligible = self._eligible_indices()
        if not eligible:
            return ""

        if len(eligible) <= top_k:
            selected = [self._skills[i] for i in eligible]
        elif self.embedder and self._entry_embeddings:
            emb_matrix = np.array(self._entry_embeddings)
            all_queries = [query] + (extra_queries or [])
            best_scores = np.full(len(self._skills), -1.0)
            for q in all_queries:
                q_vec = self.embedder.encode(q)
                norms = (np.linalg.norm(emb_matrix, axis=1)
                         * np.linalg.norm(q_vec) + 1e-8)
                scores = (emb_matrix @ q_vec) / norms
                best_scores = np.maximum(best_scores, scores)
            ranked = sorted(eligible, key=lambda i: best_scores[i], reverse=True)
            selected = [self._skills[i] for i in ranked[:top_k]
                        if best_scores[i] > self.MIN_SIMILARITY]
        else:
            selected = [self._skills[i] for i in eligible[-top_k:]]

        if not selected:
            return ""

        if self.provenance_log and self._current_task_id:
            self.provenance_log[-1]["retrieved"].extend(
                s.description for s in selected)
            # Provenance is captured at retrieval time, so source_task_id is the note's true origin: within-task self-retrieval is excluded and names are disambiguated by id. AWM notes are not callable, so only retrieval is logged.
            full = self.provenance_log[-1].setdefault("retrieved_full", [])
            for s in selected:
                full.append({
                    "id": s.id,
                    "name": s.description,
                    "source_task_id": s.source_task_id or "",
                })

        parts = [f"[{s.description}]\n{s.source}" for s in selected]
        body = "\n\n".join(parts)
        # Fallible-hint framing wrapped around every retrieval (linear and subtask agents both call retrieve): notes are hints, not ground truth.
        header = (
            "Experience notes from PREVIOUS tasks in this domain — HINTS, "
            "not ground truth. Their column / file / key names may not match "
            "this task; verify against the data listing, and do not let a note "
            "change this task's scope (filters, time range, entity, units, "
            "granularity). Adapt or ignore.\n\n"
        )
        return header + body

    # brief

    def get_brief(self) -> str:
        if not self._skills:
            return "(empty — no skills yet)"
        return "\n".join(f"- {s.description}" for s in self._skills)

    # apply delta

    def apply_skill_delta(self, delta_M: dict, world=None) -> list[str]:
        """Store one workflow note.

        If a near-duplicate description already exists, merge by appending non-duplicate lines (replacing the procedure line if the new note has one).
        """
        desc = delta_M.get("description", "").strip()
        content = delta_M.get("content", "").strip()
        if not desc or not content:
            return ["[AWM] Nothing to save"]

        if self.embedder:
            desc_vec = self.embedder.encode(desc)
            for i, emb in enumerate(self._entry_embeddings):
                sim = float(np.dot(desc_vec, emb)
                            / (np.linalg.norm(desc_vec)
                               * np.linalg.norm(emb) + 1e-8))
                if sim > self.MERGE_THRESHOLD:
                    s = self._skills[i]
                    old_lines = s.source.strip().split("\n")
                    new_lines = content.strip().split("\n")
                    new_proc = [
                        ln for ln in new_lines
                        if ln.strip().lower().startswith("- procedure:")]
                    new_other = [
                        ln for ln in new_lines
                        if not ln.strip().lower().startswith("- procedure:")]
                    if new_proc:
                        old_lines = [
                            ln for ln in old_lines
                            if not ln.strip().lower().startswith(
                                "- procedure:")]
                    seen = set(old_lines)
                    added = [ln for ln in new_other
                             if ln.strip() and ln not in seen]
                    s.source = "\n".join(new_proc + old_lines + added)
                    self._cap_lines(s)
                    self._entry_embeddings[i] = self.embedder.encode(
                        s.description)
                    return [f"[AWM] Merged into existing: {s.description}"]

        skill = Skill(
            id=f"awm_{uuid.uuid4().hex[:10]}",
            type="workflow",
            source=content,
            source_task_id=(self._current_task_id or ""),
            description=desc,
        )
        self._skills.append(skill)
        if self.embedder:
            self._entry_embeddings.append(self.embedder.encode(desc))
        self._cap_lines(skill)
        if self.provenance_log and self._current_task_id:
            self.provenance_log[-1]["created"].append(desc)
        return [f"[AWM] New entry: {desc}"]

    def _cap_lines(self, skill: Skill) -> None:
        lines = skill.source.strip().split("\n")
        if len(lines) <= self.MAX_LINES:
            return
        proc = [ln for ln in lines
                if ln.strip().lower().startswith("- procedure:")]
        other = [ln for ln in lines
                 if not ln.strip().lower().startswith("- procedure:")]
        keep = self.MAX_LINES - len(proc)
        skill.source = "\n".join(proc + other[-keep:])

    def _estimate_tokens(self) -> int:
        """Rough token estimate (4 chars ≈ 1 token)."""
        total = sum(len(s.description) + len(s.source)
                    for s in self._skills)
        return total // 4

    # stats / snapshot

    def get_stats(self) -> dict:
        return {
            "total_entries": len(self._skills),
            "total_content_lines": sum(
                len(s.source.strip().split("\n")) for s in self._skills),
            "estimated_tokens": self._estimate_tokens(),
        }

    def get_active_skill_names(self) -> list[str]:
        # AWM entries have no function names; ids stand in.
        return [s.id for s in self._skills]

    def snapshot(self) -> dict:
        return {
            "skills": [s.__dict__ for s in self._skills],
            "provenance_log": list(self.provenance_log),
        }

    def load_snapshot(self, snapshot: dict) -> None:
        self._skills = [Skill(**d) for d in snapshot.get("skills", [])]
        self._entry_embeddings = []
        if self.embedder:
            # Cache description embeddings across load_snapshot calls: the transfer replay reloads a mostly overlapping snapshot before every task, so re-encoding the whole library each time is waste.
            cache = getattr(self, "_desc_embed_cache", None)
            if cache is None:
                cache = self._desc_embed_cache = {}
            for s in self._skills:
                vec = cache.get(s.description)
                if vec is None:
                    vec = self.embedder.encode(s.description)
                    cache[s.description] = vec
                self._entry_embeddings.append(vec)
        self.provenance_log = list(snapshot.get("provenance_log", []))

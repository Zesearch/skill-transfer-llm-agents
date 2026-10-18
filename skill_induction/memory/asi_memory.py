"""ASIMemory: code-function skill memory."""

from __future__ import annotations

import re
import uuid
from typing import Optional

import numpy as np

from skill_induction.memory.base import Skill, SkillMemory
from skill_induction.memory._embed import _model


class ASIMemory(SkillMemory):
    MAX_ENTRIES = 30
    # Retrieval returns the top-k skills above this cosine similarity.
    MIN_SIMILARITY = 0.30
    # New functions whose description near-duplicates an existing skill are dropped instead of growing the library.
    DEDUP_THRESHOLD = 0.85

    def __init__(self, model=None, *, benchmark: str = "appworld",
                 top_k: int = 5, max_tokens: int = 8192):
        self.model = model
        self.benchmark = benchmark
        self.top_k = top_k
        self.max_tokens = max_tokens
        self._skills: list[Skill] = []
        self._entry_embeddings: list = []
        # Function names returned by the latest retrieve() (per-task scope); the agents read it to preload / register the just-retrieved functions and to detect their invocation in executed code.
        self._context_function_names: set[str] = set()
        self.provenance_log: list[dict] = []
        self._current_task_id: Optional[str] = None

    # per-task hook

    def set_current_task(self, task_id) -> None:
        self._current_task_id = task_id
        self._context_function_names = set()
        self.provenance_log.append(
            {"task": task_id, "retrieved": [], "created": []})

    # embedder helper

    @property
    def embedder(self):
        m = _model()
        return None if m == "lexical" else m

    # retrieve

    def retrieve(self, query: str,
                 extra_queries: list[str] | None = None) -> str:
        self._context_function_names = set()
        if not self._skills:
            return ""

        selected_indices = self._select_by_similarity(query, extra_queries)
        if not selected_indices:
            return ""

        all_indices = self._resolve_dependencies(selected_indices)
        sorted_indices = self._topological_sort(all_indices)

        for idx in sorted_indices:
            self._context_function_names.add(self._name(self._skills[idx]))
        if self.provenance_log and self._current_task_id:
            self.provenance_log[-1]["retrieved"].extend(
                self._name(self._skills[idx]) for idx in sorted_indices)
            # Provenance is captured at retrieval time, so source_task_id is the skill's true origin: within-task self-retrieval can be excluded and cross-domain name collisions are disambiguated by id.
            full = self.provenance_log[-1].setdefault("retrieved_full", [])
            for idx in sorted_indices:
                s = self._skills[idx]
                full.append({
                    "id": s.id,
                    "name": (s.extra or {}).get("name") or s.description,
                    "source_task_id": s.source_task_id or "",
                })

        # Reference-not-ground-truth framing wrapped around every retrieval (linear and subtask agents both call retrieve).
        header = (
            "The following Python helper functions were induced from "
            "PREVIOUS, DIFFERENT tasks. Treat them as REFERENCE, not "
            "ground truth — they may NOT fit this task. Use them to "
            "recall the relevant APIs and overall approach, but write "
            "your OWN code for THIS task. Do NOT blindly call a stored "
            "function assuming it works; only reuse one (by copying its "
            "definition into your code) if it genuinely matches what THIS "
            "task needs, and never let a helper change this task's goal, "
            "inputs, or scope."
        )
        parts = []
        for idx in sorted_indices:
            s = self._skills[idx]
            parts.append(f"# {s.description}\n{s.source}")
        return header + "\n\n" + "\n\n".join(parts)

    def _current_domain(self) -> str | None:
        t = self._current_task_id or ""
        return t.split("-")[0] if t else None

    def _eligible_indices(self) -> list[int]:
        """Candidate pool for retrieval.

        For KramaBench, gate to skills induced in the same domain (group_key): domains share no data, so a cross-domain helper is noise. Other benchmarks are ungated (their scenarios share an action surface).
        """
        idxs = list(range(len(self._skills)))
        if self.benchmark == "kramabench":
            d = self._current_domain()
            if d:
                idxs = [i for i in idxs
                        if (self._skills[i].source_task_id or "").split("-")[0] == d]
        return idxs

    def _select_by_similarity(self, query: str,
                              extra_queries: list[str] | None = None
                              ) -> list[int]:
        eligible = self._eligible_indices()
        if not eligible:
            return []
        if len(eligible) <= self.top_k:
            return eligible

        if self.embedder and self._entry_embeddings:
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
            k = min(self.top_k, len(ranked))
            return [int(i) for i in ranked[:k]
                    if best_scores[i] >= self.MIN_SIMILARITY]
        else:
            return eligible[-self.top_k:]

    def _resolve_dependencies(self, seed_indices: list[int]) -> list[int]:
        all_indices = set(seed_indices)
        queue = list(seed_indices)
        name_to_idx = {self._name(s): i
                       for i, s in enumerate(self._skills)}
        while queue:
            idx = queue.pop(0)
            impl = self._skills[idx].source
            for other_name, other_idx in name_to_idx.items():
                if other_idx in all_indices:
                    continue
                if re.search(r"\b" + re.escape(other_name) + r"\s*\(", impl):
                    all_indices.add(other_idx)
                    queue.append(other_idx)
        return list(all_indices)

    def _topological_sort(self, indices: list[int]) -> list[int]:
        if len(indices) <= 1:
            return list(indices)
        name_to_idx = {self._name(self._skills[i]): i for i in indices}
        deps: dict[int, set[int]] = {}
        for i in indices:
            fn_deps: set[int] = set()
            for other_name, other_idx in name_to_idx.items():
                if other_idx != i:
                    if re.search(r"\b" + re.escape(other_name) + r"\s*\(",
                                 self._skills[i].source):
                        fn_deps.add(other_idx)
            deps[i] = fn_deps
        in_degree = {i: len(d) for i, d in deps.items()}
        queue = sorted(i for i, deg in in_degree.items() if deg == 0)
        ordered: list[int] = []
        while queue:
            current = queue.pop(0)
            ordered.append(current)
            for i in indices:
                if current in deps.get(i, set()):
                    deps[i].discard(current)
                    in_degree[i] -= 1
                    if in_degree[i] == 0:
                        queue.append(i)
                        queue.sort()
        remaining = sorted(i for i in indices if i not in ordered)
        ordered.extend(remaining)
        return ordered

    # brief

    def get_brief(self) -> str:
        if not self._skills:
            return "(empty — no functions yet)"
        return "\n".join(
            f"- {self._name(s)}: {s.description}" for s in self._skills)

    # apply delta

    def apply_skill_delta(self, delta_M: dict, world=None) -> list[str]:
        functions = delta_M.get("functions", [])
        # Tolerate a single function dict emitted bare (not wrapped in a list).
        if isinstance(functions, dict):
            functions = [functions]
        if not isinstance(functions, list) or not functions:
            return ["[ASI] Nothing to save"]
        messages: list[str] = []
        for fn in functions:
            if not isinstance(fn, dict):
                # The model occasionally returns a bare string instead of a {name, description, implementation} dict; skip it rather than crash.
                messages.append("[ASI] Skipped malformed function entry "
                                f"(expected dict, got {type(fn).__name__})")
                continue
            name = fn.get("name", "").strip()
            desc = fn.get("description", "").strip()
            impl = fn.get("implementation", "").strip()
            if not name or not impl:
                continue
            if not desc:
                desc = name
            existing_idx = self._find_by_name(name)
            if existing_idx is not None:
                self._skills[existing_idx].description = desc
                self._skills[existing_idx].source = impl
                self._skills[existing_idx].extra = {
                    **(self._skills[existing_idx].extra or {}),
                    "name": name,
                }
                if self.embedder:
                    self._entry_embeddings[existing_idx] = (
                        self.embedder.encode(desc))
                messages.append(f"[ASI] Updated: {name}")
            else:
                # Drop near-duplicate descriptions (different name, same job) so the library stays compact and retrieval precise.
                if self.embedder and self._entry_embeddings:
                    new_vec = self.embedder.encode(desc)
                    dup = False
                    for emb in self._entry_embeddings:
                        sim = float(np.dot(new_vec, emb)
                                    / (np.linalg.norm(new_vec)
                                       * np.linalg.norm(emb) + 1e-8))
                        if sim >= self.DEDUP_THRESHOLD:
                            dup = True
                            break
                    if dup:
                        messages.append(
                            f"[ASI] Skipped near-duplicate: {name}")
                        continue
                skill = Skill(
                    id=f"asi_{uuid.uuid4().hex[:10]}",
                    type="code",
                    source=impl,
                    source_task_id=(self._current_task_id or ""),
                    description=desc,
                    extra={"name": name},
                )
                self._skills.append(skill)
                if self.embedder:
                    self._entry_embeddings.append(
                        self.embedder.encode(desc))
                if self.provenance_log and self._current_task_id:
                    self.provenance_log[-1]["created"].append(name)
                messages.append(f"[ASI] New function: {name}")
        return messages if messages else ["[ASI] Nothing to save"]

    # helpers

    @staticmethod
    def _name(s: Skill) -> str:
        # extra['name'] is always set by apply_skill_delta; load_snapshot preserves it via Skill(**d).
        return s.extra["name"]

    def _find_by_name(self, name: str) -> int | None:
        """Find entry index by function name."""
        for i, s in enumerate(self._skills):
            if self._name(s) == name:
                return i
        return None

    def _resolve_dependents(self, fn_name: str) -> set:
        """Find all functions that transitively depend on fn_name."""
        dependents: set[str] = set()
        queue = [fn_name]
        while queue:
            target = queue.pop(0)
            for s in self._skills:
                name = self._name(s)
                if name in dependents or name == fn_name:
                    continue
                if re.search(r"\b" + re.escape(target) + r"\s*\(",
                             s.source):
                    dependents.add(name)
                    queue.append(name)
        return dependents

    # stats / snapshot

    def get_stats(self) -> dict:
        return {
            "total_entries": len(self._skills),
            "total_code_lines": sum(
                len(s.source.strip().split("\n")) for s in self._skills),
        }

    def get_active_skill_codes(self) -> list[str]:
        return [s.source for s in self._skills]

    def get_active_skill_names(self) -> list[str]:
        return [self._name(s) for s in self._skills]

    def remove_by_name(self, fn_name: str) -> list[str]:
        """Remove a function and everything that transitively depends on it; returns all removed names."""
        idx = self._find_by_name(fn_name)
        if idx is None:
            return []
        dependents = self._resolve_dependents(fn_name)
        dependents.add(fn_name)
        removed: list[str] = []
        for i in sorted(
            [i for i, s in enumerate(self._skills)
             if self._name(s) in dependents],
            reverse=True,
        ):
            name = self._name(self._skills[i])
            self._skills.pop(i)
            if self._entry_embeddings:
                self._entry_embeddings.pop(i)
            self._context_function_names.discard(name)
            removed.append(name)
        return removed

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

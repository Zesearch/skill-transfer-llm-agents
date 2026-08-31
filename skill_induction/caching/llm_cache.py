"""Disk-backed LLM call cache, process-safe.

Key: SHA-256 of (model_id, messages, tools, generation_kwargs) after canonical serialization. Values are pickled ChatResponses, sharded by hash prefix under the cache root.

Reads are lock-free; writes hold a per-key filelock so concurrent processes don't corrupt each other.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

from filelock import FileLock

DEFAULT_CACHE_ROOT = Path(os.environ.get(
    "SKILL_INDUCTION_CACHE_DIR",
    str(Path.home() / ".cache" / "skill_induction" / "llm_cache"),
))


class LLMCache:
    def __init__(self, root: Path | str = DEFAULT_CACHE_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def make_key(
        self,
        model_id: str,
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice,
        gen_kwargs: dict,
        namespace: str = "",
    ) -> str:
        # namespace scopes the key to one cell+task (<benchmark>|<agent>|<model>|<task_id>) so byte-identical messages from different cells never share a cached response; empty means legacy content-addressed behaviour.
        payload = {
            "namespace": namespace,
            "model": model_id,
            "messages": _serialize_messages(messages),
            "tools": tools or [],
            "tool_choice": _serialize_tool_choice(tool_choice),
            "kwargs": _serialize_kwargs(gen_kwargs),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def get(self, key: str):
        path = self._path_for(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            with open(path, "rb") as f:
                resp = pickle.load(f)
            self.hits += 1
            return resp
        except (EOFError, pickle.UnpicklingError):
            # Corrupt entry: treat as a miss; the next put overwrites it.
            self.misses += 1
            return None

    def put(self, key: str, value) -> None:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(path) + ".lock")
        with lock:
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(value, f)
            os.replace(tmp, path)

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses,
                "total": self.hits + self.misses,
                "hit_rate": (self.hits / (self.hits + self.misses))
                            if (self.hits + self.misses) else 0.0}

    def _path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key[2:]}.pkl"


def _serialize_messages(messages):
    out = []
    for m in messages:
        d = {"role": m.get("role"), "content": _str_or_none(m.get("content"))}
        if m.get("tool_calls"):
            d["tool_calls"] = [
                {
                    "id": tc["id"] if isinstance(tc, dict) else tc.id,
                    "name": tc["name"] if isinstance(tc, dict) else tc.name,
                    "arguments": _canonicalize(
                        tc["arguments"] if isinstance(tc, dict) else tc.arguments
                    ),
                }
                for tc in m["tool_calls"]
            ]
        if "tool_call_id" in m:
            d["tool_call_id"] = m["tool_call_id"]
        if "name" in m:
            d["name"] = m["name"]
        out.append(d)
    return out


def _serialize_tool_choice(choice):
    if isinstance(choice, dict):
        return json.dumps(choice, sort_keys=True)
    return str(choice)


def _serialize_kwargs(kwargs: dict) -> dict:
    # Drop entries that don't affect the model response (e.g. timeouts).
    blocklist = {"timeout", "stream"}
    return {k: _canonicalize(v) for k, v in sorted(kwargs.items())
            if k not in blocklist}


def _canonicalize(v):
    if isinstance(v, dict):
        return {k: _canonicalize(v[k]) for k in sorted(v)}
    if isinstance(v, (list, tuple)):
        return [_canonicalize(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _str_or_none(content):
    if content is None:
        return None
    if isinstance(content, str):
        return content
    return _canonicalize(content)

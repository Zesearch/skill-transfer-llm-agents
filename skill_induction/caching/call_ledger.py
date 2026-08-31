"""Per-call LLM ledger: records the request and response of every ChatGenerator.chat() call in a task.

Unlike the result pkl's env-level `trace`, entries are LLM-call level, so subtask-agent entries keep the full pre-summarisation history. Messages are deep-copied before the call so later history mutation can't alias them.

The runner creates one CallLedger per task, attaches it as `model.ledger`, and stores `.entries` on the task record as `llm_calls`.
"""

from __future__ import annotations

import contextlib
import copy
import time
from typing import Any


def _jsonable(v: Any):
    """Best-effort conversion to pickle/JSON-stable primitives."""
    if isinstance(v, dict):
        return {str(k): _jsonable(v[k]) for k in v}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    # ToolCall / dataclasses / misc
    if hasattr(v, "__dict__"):
        return {str(k): _jsonable(x) for k, x in vars(v).items()}
    return str(v)


def _response_dict(resp) -> dict:
    if resp is None:
        return {"content": None, "tool_calls": [], "finish_reason": "",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                          "total_tokens": 0}}
    u = getattr(resp, "usage", None)
    return {
        "content": getattr(resp, "content", None),
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": _jsonable(tc.arguments)}
            for tc in (getattr(resp, "tool_calls", None) or [])
        ],
        "finish_reason": getattr(resp, "finish_reason", ""),
        "usage": {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            "total_tokens": getattr(u, "total_tokens", 0) or 0,
        },
    }


class CallLedger:
    """Append-only log of every LLM call in one task."""

    def __init__(self, task_id: str = ""):
        self.task_id = task_id
        self.phase: str = "init"
        self.entries: list[dict] = []
        self._seq = 0

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    @contextlib.contextmanager
    def phase_scope(self, phase: str):
        """Switch phase, restoring the previous one on exit so nested scopes compose."""
        prev = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = prev

    def record(self, *, model_id, messages, tools, tool_choice, kwargs,
               response, cache_hit, latency_s) -> None:
        self._seq += 1
        self.entries.append({
            "seq": self._seq,
            "phase": self.phase,
            "model_id": model_id,
            "cache_hit": bool(cache_hit),
            "latency_s": round(latency_s, 4),
            "timestamp": time.time(),
            "messages": copy.deepcopy(_jsonable(messages)),
            "tools": _jsonable(tools) if tools else None,
            "tool_choice": _jsonable(tool_choice),
            "kwargs": _jsonable(kwargs),
            "response": _response_dict(response),
        })

    def record_event(self, *, name: str, detail: str = "",
                     phase: str | None = None) -> None:
        """Record a non-LLM agent event (e.g. the deterministic auto-finish); marked `synthetic: True` so the inspector can label it as not a model call."""
        self._seq += 1
        self.entries.append({
            "seq": self._seq,
            "phase": phase or self.phase,
            "synthetic": True,
            "event": name,
            "model_id": "(deterministic agent action — no model call)",
            "cache_hit": False,
            "latency_s": 0.0,
            "timestamp": time.time(),
            "messages": [],
            "tools": None,
            "tool_choice": None,
            "kwargs": {},
            "response": {
                "content": detail,
                "tool_calls": [{
                    "id": "synthetic",
                    "name": name,
                    "arguments": {"detail": detail} if detail else {},
                }],
                "finish_reason": "synthetic_event",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                          "total_tokens": 0},
            },
        })

    def summary(self) -> dict:
        by_phase: dict[str, int] = {}
        prompt = completion = hits = 0
        for e in self.entries:
            by_phase[e["phase"]] = by_phase.get(e["phase"], 0) + 1
            prompt += e["response"]["usage"]["prompt_tokens"]
            completion += e["response"]["usage"]["completion_tokens"]
            hits += int(e["cache_hit"])
        return {
            "n_calls": len(self.entries),
            "n_cache_hits": hits,
            "calls_by_phase": by_phase,
            "ledger_prompt_tokens": prompt,
            "ledger_completion_tokens": completion,
        }


def set_phase(model, phase: str) -> None:
    """No-op if the model has no ledger attached (e.g. cache-only runs)."""
    led = getattr(model, "ledger", None)
    if led is not None:
        led.set_phase(phase)


@contextlib.contextmanager
def phase(model, phase: str):
    led = getattr(model, "ledger", None)
    if led is None:
        yield
        return
    with led.phase_scope(phase):
        yield

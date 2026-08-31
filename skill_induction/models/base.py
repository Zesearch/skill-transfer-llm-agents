"""ChatGenerator base + canonical message/tool/response dataclasses.

Trimmed to the pieces this project actually uses (Bedrock + canonical/OpenAI adapters; vLLM and other providers handled separately).
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable


# canonical types

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    """Backend-agnostic chat completion result."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: Usage = field(default_factory=Usage)
    raw: Any = None


# base class

class ChatGenerator(ABC):
    """Single chat entry point. Subclasses implement `_chat()`; the public `chat()` adds disk-backed caching."""

    SUPPORTS_TOOLS_NATIVE: bool = True

    def __init__(self, model_id: str, cache=None):
        self.model_id = model_id
        self._cache = cache
        # Optional per-task CallLedger (set by the runner). When present, every chat(), cache hit or fresh, is logged verbatim.
        self.ledger = None
        # Cache scope, set by the runner per task to "<benchmark>|<agent>|<model>|<task_id>" so byte-identical messages from different cells (e.g. subtask_awm vs subtask_asi on an empty library) never collide; only an identical replay of the same cell+task reuses cache. Empty means legacy content-only key.
        self.cache_namespace = ""

    @abstractmethod
    def _chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        **kwargs,
    ) -> ChatResponse:
        """Provider-specific call; must not consult the cache."""

    def chat(
        self,
        messages,
        *,
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        **kwargs,
    ) -> ChatResponse:
        messages = normalize_messages(messages)
        t0 = time.perf_counter()
        cache_hit = False
        if self._cache is not None:
            key = self._cache.make_key(
                self.model_id, messages, tools, tool_choice, kwargs,
                namespace=self.cache_namespace,
            )
            hit = self._cache.get(key)
            if hit is not None:
                resp, cache_hit = hit, True
            else:
                resp = self._chat(
                    messages, tools=tools, tool_choice=tool_choice, **kwargs,
                )
                self._cache.put(key, resp)
        else:
            resp = self._chat(
                messages, tools=tools, tool_choice=tool_choice, **kwargs,
            )
        if self.ledger is not None:
            self.ledger.record(
                model_id=self.model_id, messages=messages, tools=tools,
                tool_choice=tool_choice, kwargs=kwargs, response=resp,
                cache_hit=cache_hit, latency_s=time.perf_counter() - t0,
            )
        return resp


# message normalization

def normalize_messages(messages: Iterable) -> list[dict]:
    """Accept tuples or dicts; return canonical message dicts.

    Canonical message: {"role", "content"} plus optional "tool_calls" (assistant) or "tool_call_id"/"name" (tool result).
    """
    out: list[dict] = []
    for m in messages:
        if isinstance(m, tuple):
            role, content = m
            out.append({"role": role, "content": content})
        elif isinstance(m, dict):
            out.append(dict(m))
        else:
            raise TypeError(f"Unsupported message type: {type(m)}")
    return out


# Bedrock conversion

def to_bedrock_payload(
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_choice: str | dict = "auto",
) -> dict:
    """Canonical → Bedrock Converse API kwargs (system, messages, toolConfig)."""
    system_blocks: list[dict] = []
    converse: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            text = _stringify(m.get("content", ""))
            if text:
                system_blocks.append({"text": text})
            continue

        if role == "tool":
            block = {
                "toolResult": {
                    "toolUseId": m["tool_call_id"],
                    "content": [{"text": _stringify(m.get("content", ""))}],
                }
            }
            if (
                converse
                and converse[-1]["role"] == "user"
                and all("toolResult" in b for b in converse[-1]["content"])
            ):
                converse[-1]["content"].append(block)
            else:
                converse.append({"role": "user", "content": [block]})
            continue

        if role == "assistant":
            blocks: list[dict] = []
            text = m.get("content")
            if text:
                blocks.append({"text": text})
            for tc in m.get("tool_calls") or []:
                tc_id = tc["id"] if isinstance(tc, dict) else tc.id
                tc_name = tc["name"] if isinstance(tc, dict) else tc.name
                tc_args = tc["arguments"] if isinstance(tc, dict) else tc.arguments
                blocks.append({
                    "toolUse": {
                        "toolUseId": tc_id,
                        "name": _sanitize_tool_name(tc_name),
                        "input": tc_args if isinstance(tc_args, dict) else json.loads(tc_args or "{}"),
                    }
                })
            if not blocks:
                blocks = [{"text": "(no output)"}]
            converse.append({"role": "assistant", "content": blocks})
            continue

        text = _stringify(m.get("content", ""))
        if not text.strip():
            text = "(no output)"
        converse.append({"role": "user", "content": [{"text": text}]})

    payload: dict = {"messages": converse}
    if system_blocks:
        payload["system"] = system_blocks

    if tools:
        bedrock_tools = []
        for t in tools:
            fn = t["function"] if t.get("type") == "function" else t
            bedrock_tools.append({
                "toolSpec": {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "inputSchema": {"json": fn.get("parameters", {"type": "object"})},
                }
            })
        payload["toolConfig"] = {
            "tools": bedrock_tools,
            "toolChoice": _bedrock_tool_choice(tool_choice),
        }
    return payload


def parse_bedrock_response(resp) -> ChatResponse:
    """Bedrock Converse response → canonical ChatResponse.

    No text on tool-call turns: when the model emits a tool call we set `content=None` and discard all accompanying text, including `reasoningContent` and any `text` blocks. The model still reasons server-side (reasoning_effort is on), which improves the tool call, but the reasoning text is never stored, so it never appears as a "text cell" in the ledger / pkl / HTML inspector and never leaks into history. Pure-text responses (no tool call, which should not happen under tool_choice=required) keep their text so the retry loop can detect them.

    One tool call per step: only the first `toolUse` block is kept. The raw content list (in `raw`) is rewritten to drop extras so the ledger / pkl / HTML inspector see exactly one tool call.
    """
    output_msg = resp["output"]["message"]
    content_blocks = output_msg.get("content", [])
    text_parts: list[str] = []
    tool_use_blocks: list[dict] = []
    for block in content_blocks:
        if "text" in block:
            text_parts.append(block.get("text") or "")
        elif "toolUse" in block:
            tool_use_blocks.append(block)
    tool_calls: list[ToolCall] = []
    for block in (tool_use_blocks[:1] if tool_use_blocks else []):
        tu = block["toolUse"]
        tool_calls.append(ToolCall(
            id=tu["toolUseId"],
            name=_sanitize_tool_name(tu.get("name", "")),
            arguments=tu.get("input", {}) or {},
        ))
    # When a tool call exists, rewrite raw content to keep only the single tool_use block (drops reasoning, text, and extra tool_use blocks) so the inspector renders one clean tool call; with no tool call, leave raw untouched so the text response stays visible for retry.
    if tool_calls:
        output_msg["content"] = tool_use_blocks[:1]
        content = None
    else:
        content = "".join(text_parts) or None
    usage = resp.get("usage", {}) or {}
    return ChatResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=resp.get("stopReason", ""),
        usage=Usage(
            prompt_tokens=usage.get("inputTokens", 0) or 0,
            completion_tokens=usage.get("outputTokens", 0) or 0,
            total_tokens=(usage.get("inputTokens", 0) or 0)
                         + (usage.get("outputTokens", 0) or 0),
        ),
        raw=resp,
    )


# helpers

_TOOLCALL_TEMPLATE_MARKERS = (
    "<｜DSML｜function_calls",
    "<｜tool▁calls▁begin｜>",
    "<｜tool▁calls▁end｜>",
    "<｜tool▁call▁begin｜>",
    "<｜tool▁call▁end｜>",
    "<｜tool▁sep｜>",
    "<tool_call>",
    "</tool_call>",
    "<|python_tag|>",
)
_DEEPSEEK_SPECIAL_RE = re.compile(r"<｜[^｜]*?｜[^>]*?>")
_PIPE_SPECIAL_RE = re.compile(r"<\|[^|]+\|>")
_BEDROCK_TOOL_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _strip_toolcall_template_leakage(text: str) -> str:
    if not text:
        return text
    for marker in _TOOLCALL_TEMPLATE_MARKERS:
        text = text.replace(marker, "")
    text = _DEEPSEEK_SPECIAL_RE.sub("", text)
    text = _PIPE_SPECIAL_RE.sub("", text)
    return text.rstrip()


def _sanitize_tool_name(name: str) -> str:
    if not name:
        return "unknown_tool"
    if "." in name:
        name = name.rsplit(".", 1)[-1]
    name = _BEDROCK_TOOL_NAME_RE.sub("_", name)
    return name or "unknown_tool"


def _bedrock_tool_choice(choice):
    # Bedrock Converse expects one of {"auto":{}}, {"any":{}}, or {"tool":{"name":<name>}}; OpenAI-style choices (e.g. {"type":"function","function":{"name":"X"}}) are also accepted and translated.
    if isinstance(choice, dict):
        # OpenAI-style forced-tool: {"type":"function","function":{"name":"X"}}
        if choice.get("type") == "function" and isinstance(
                choice.get("function"), dict
        ) and choice["function"].get("name"):
            return {"tool": {"name": choice["function"]["name"]}}
        # already a Bedrock-shaped dict
        return choice
    if choice == "auto":
        return {"auto": {}}
    if choice == "required" or choice == "any":
        return {"any": {}}
    if isinstance(choice, str):
        return {"tool": {"name": choice}}
    return {"auto": {}}


def _stringify(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and "text" in c:
                parts.append(c["text"])
            else:
                parts.append(str(c))
        return "".join(parts)
    return str(content) if content is not None else ""

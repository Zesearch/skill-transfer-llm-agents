"""Local dense models via vLLM's OpenAI-compatible server.

`vllm serve Qwen/Qwen3-14B --enable-auto-tool-choice --tool-call-parser hermes` exposes `/v1/chat/completions`. This wraps it under the same `ChatGenerator` contract as `BedrockChatGenerator`, so caching + the per-call ledger (both in `ChatGenerator.chat`) work unchanged.

No-hidden-thinking policy (project rule: reasoning is the explicit `think` tool, never internal CoT): Qwen3 dense models are hybrid-thinking and think by default; we disable it per request via vLLM's `chat_template_kwargs={"enable_thinking": False}` extra body field.
"""

from __future__ import annotations

import json
import os
import time

import requests

from skill_induction.models.base import (
    ChatGenerator,
    ChatResponse,
    ToolCall,
    Usage,
)


# Short alias → served model name (must match `vllm serve <name>`).
LOCAL_MODEL_CONFIG: dict[str, str] = {
    "qwen-3-4b-local": "Qwen/Qwen3-4B",
    "qwen-3-8b-local": "Qwen/Qwen3-8B",
    "qwen-3-14b-local": "Qwen/Qwen3-14B",
    "qwen-3-32b-local": "Qwen/Qwen3-32B",
    "gemma-3-4b-local": "google/gemma-3-4b-it",
    "gemma-3-12b-local": "google/gemma-3-12b-it",
    "gemma-3-27b-local": "google/gemma-3-27b-it",
}

# Models that think-by-default and need explicit disabling.
_NO_THINK_PREFIXES = ("Qwen/Qwen3", "Qwen3")


def vllm_serve_args(alias: str) -> str:
    """Extra `vllm serve` flags for a local model (for the serving launcher).

    Gemma-3's chat template has no native tool support, so it needs vLLM's pythonic tool parser plus the gemma3-pythonic template (vendored from vLLM v0.21.0 examples/). Context is capped at 40960 (Qwen3's native window) so both families run under the same limit and the 27B KV cache fits alongside the weights on one 96 GB GH200.
    """
    served = LOCAL_MODEL_CONFIG[alias]
    if served.startswith("google/gemma-3"):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        template = os.path.join(repo_root, "scripts", "gemma3_pythonic.jinja")
        return (f"--enable-auto-tool-choice --tool-call-parser pythonic "
                f"--chat-template {template} --max-model-len 40960")
    return "--enable-auto-tool-choice --tool-call-parser hermes"


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """Canonical → OpenAI chat messages (assistant tool_calls + tool results need the OpenAI shape)."""
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id", ""),
                "content": m.get("content") if isinstance(m.get("content"), str)
                           else str(m.get("content", "")),
            })
            continue
        if role == "assistant" and m.get("tool_calls"):
            out.append({
                "role": "assistant",
                "content": m.get("content") or "",
                "tool_calls": [
                    {
                        "id": tc["id"] if isinstance(tc, dict) else tc.id,
                        "type": "function",
                        "function": {
                            "name": tc["name"] if isinstance(tc, dict) else tc.name,
                            "arguments": json.dumps(
                                tc["arguments"] if isinstance(tc, dict)
                                else tc.arguments
                            ),
                        },
                    }
                    for tc in m["tool_calls"]
                ],
            })
            continue
        content = m.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        out.append({"role": role, "content": content})
    return out


class VLLMChatGenerator(ChatGenerator):
    """OpenAI-compatible client for a local vLLM server."""

    SUPPORTS_TOOLS_NATIVE = True

    def __init__(
        self,
        model_id: str,
        *,
        cache=None,
        base_url: str | None = None,
        served_model: str | None = None,
        max_retries: int = 6,
        base_sleep_s: float = 3.0,
        max_sleep_s: float = 96.0,
        enable_thinking: bool = False,
        repetition_penalty: float | None = None,
        timeout_s: float = 600.0,
    ):
        super().__init__(model_id=model_id, cache=cache)
        self.served_model = (
            served_model or LOCAL_MODEL_CONFIG.get(model_id, model_id))
        self.base_url = (
            base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        ).rstrip("/")
        self.max_retries = max_retries
        self.base_sleep_s = base_sleep_s
        self.max_sleep_s = max_sleep_s
        self.timeout_s = timeout_s
        # Env override (SKILL_INDUCTION_VLLM_THINK=0 disables Qwen3 thinking): non-thinking is dramatically faster, which matters for local models and is effectively required for RL rollouts.
        _think_env = os.environ.get("SKILL_INDUCTION_VLLM_THINK")
        if _think_env is not None:
            enable_thinking = _think_env not in ("0", "false", "False", "")
        self.enable_thinking = enable_thinking
        # Repetition control: small local dense models (Qwen3-14B especially) fall into degenerate verbatim loops far more than frontier models, and `repetition_penalty` > 1.0 suppresses that. Env-overridable (VLLM_REPETITION_PENALTY), default 1.1; temperature is untouched, so run-to-run sampling variance is preserved. Local vLLM only, the Bedrock client is unaffected.
        self.repetition_penalty = (
            repetition_penalty if repetition_penalty is not None
            else float(os.environ.get("VLLM_REPETITION_PENALTY", "1.1"))
        )

    def _chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        **kwargs,
    ) -> ChatResponse:
        body: dict = {
            "model": self.served_model,
            "messages": _to_openai_messages(messages),
        }
        if "max_tokens" in kwargs:
            body["max_tokens"] = int(kwargs["max_tokens"])
        if "temperature" in kwargs:
            body["temperature"] = float(kwargs["temperature"])
        if "top_p" in kwargs:
            body["top_p"] = float(kwargs["top_p"])
        # Suppress degenerate repetition loops (a local-model failure mode); vLLM honours `repetition_penalty` as a sampling extra.
        if self.repetition_penalty and self.repetition_penalty != 1.0:
            body["repetition_penalty"] = self.repetition_penalty
        if tools:
            body["tools"] = tools          # already OpenAI tool schema
            # Convention (matches Bedrock `_bedrock_tool_choice`): a bare tool-name string forces that specific tool; translate to OpenAI's forced-function object so vLLM honours it.
            tc = tool_choice
            if isinstance(tc, str) and tc not in (
                    "auto", "none", "required"):
                tc = {"type": "function", "function": {"name": tc}}
            body["tool_choice"] = tc
        # No-hidden-thinking: disable Qwen3 thinking unless explicitly on.
        if any(self.served_model.startswith(p) for p in _NO_THINK_PREFIXES):
            body["chat_template_kwargs"] = {
                "enable_thinking": bool(self.enable_thinking)
            }

        url = f"{self.base_url}/chat/completions"
        sleep_s = self.base_sleep_s
        last_exc = None
        for _ in range(self.max_retries):
            try:
                r = requests.post(url, json=body, timeout=self.timeout_s)
                if r.status_code >= 500 or r.status_code == 429:
                    raise RuntimeError(f"vLLM HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                return _parse_openai_response(r.json())
            except (requests.ConnectionError, requests.Timeout,
                    RuntimeError) as e:
                last_exc = e
                print(f"[vLLM] retryable: {e}; sleep {sleep_s}s")
                time.sleep(sleep_s)
                sleep_s = min(sleep_s * 2, self.max_sleep_s)
                continue
        raise RuntimeError(
            f"vLLM call failed after {self.max_retries} retries: {last_exc}")


def _parse_openai_response(data: dict) -> ChatResponse:
    """vLLM/OpenAI chat-completion response → canonical ChatResponse.

    No text on tool-call turns: when the model emits a tool call we set `content=None` and discard all text including `reasoning_content`. The model still reasons during generation (enable_thinking on) but the reasoning text isn't stored, so there is no "text cell" in the inspector and no leak into history. Pure-text responses keep their text for the retry loop.

    One tool call per step: keep only the first tool_call and rewrite the raw message so the ledger/pkl/HTML see exactly one.
    """
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {}) or {}
    raw_tool_calls = list(msg.get("tool_calls") or [])
    if len(raw_tool_calls) > 1:
        msg["tool_calls"] = raw_tool_calls[:1]
    tool_calls: list[ToolCall] = []
    for tc in (raw_tool_calls[:1] if raw_tool_calls else []):
        fn = tc.get("function", {}) or {}
        raw_args = fn.get("arguments", "")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
        tool_calls.append(ToolCall(
            id=tc.get("id", "") or "",
            name=fn.get("name", "") or "",
            arguments=args if isinstance(args, dict) else {"_value": args},
        ))
    if tool_calls:
        # Drop content + reasoning from the stored message so the inspector renders a clean tool call.
        msg["content"] = None
        msg.pop("reasoning_content", None)
        content = None
    else:
        content = msg.get("content") or None
    usage = data.get("usage", {}) or {}
    return ChatResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=choice.get("finish_reason", "") or "",
        usage=Usage(
            prompt_tokens=usage.get("prompt_tokens", 0) or 0,
            completion_tokens=usage.get("completion_tokens", 0) or 0,
            total_tokens=usage.get("total_tokens", 0) or 0,
        ),
        raw=data,
    )

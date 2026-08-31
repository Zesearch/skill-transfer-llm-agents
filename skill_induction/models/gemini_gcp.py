"""Google Gemini via Vertex AI, under the same `ChatGenerator` contract.

Auth is Application Default Credentials (ADC) tied to the user's own GCP identity: the GCP project sits under an org policy that blocks API keys bound to a service account, so `gcloud auth application-default login` (user creds) + `vertexai=True` is the only working path. `project` and `location` come from `api_key/config.json` (`gcp_project` / `gcp_location`, default `global`), mirroring how the Bedrock creds are loaded in `run.py`.

Parity with the Bedrock/vLLM backends:
  - one tool call per step (only the first `functionCall` part is kept);
  - no text on tool-call turns (content=None when a tool call is emitted);
  - no hidden thinking stored: `include_thoughts=False` and any `thought` parts are dropped, so reasoning happens server-side but never lands in history, the pkl, or the inspector (matches the reasoning_effort=low policy on the Bedrock reasoning models).
"""

from __future__ import annotations

import os
import time
import uuid

from skill_induction.models.base import (
    ChatGenerator,
    ChatResponse,
    ToolCall,
    Usage,
)


# Short alias → Vertex model id. gemini-3.1-pro is preview as of 2026-02.
GEMINI_MODEL_CONFIG: dict[str, str] = {
    "gemini-3.1-pro-gcp": "gemini-3.1-pro-preview",
}


# Gemini API HTTP codes we retry (rate limit + transient server errors).
_RETRY_CODES = (429, 500, 502, 503, 504)


# Gemini-3 thinking models 400 if a functionCall in the input history lacks a `thought_signature` (an opaque token the model returns with each call and expects back verbatim to resume its private reasoning). We deliberately do not round-trip the real signatures: reusing them would give Gemini a cross-turn thinking memory the other reasoning models don't get, breaking the controlled comparison. Vertex does not validate the signature bytes, so a fixed placeholder satisfies the check while forcing Gemini to re-reason each turn from the visible history alone; it also covers synthetic history calls (few-shot demos, the terminal run_code) that never had a real signature.
_PLACEHOLDER_THOUGHT_SIGNATURE = b"skill_induction/no-persisted-reasoning"


def _to_gemini_contents(messages: list[dict]):
    """Canonical messages → (system_instruction, contents).

    Roles: system → merged system_instruction; user → role "user"; assistant → role "model" (text part and/or functionCall parts); tool → functionResponse part grouped into a role "user" content (Gemini sends tool results as user-side function responses). Gemini matches a response to its call by name, not id, so we resolve each tool_call_id → name from the preceding assistant tool_calls.
    """
    from google.genai import types

    system_parts: list[str] = []
    contents: list = []
    id_to_name: dict[str, str] = {}

    for m in messages:
        role = m.get("role")

        if role == "system":
            text = m.get("content")
            if isinstance(text, str) and text.strip():
                system_parts.append(text)
            elif text:
                system_parts.append(str(text))
            continue

        if role == "tool":
            name = m.get("name") or id_to_name.get(m.get("tool_call_id", ""), "") \
                or "unknown_tool"
            content = m.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            part = types.Part.from_function_response(
                name=name, response={"result": content})
            # Merge into a trailing user function-response content if present, else open a new one.
            if (contents and contents[-1].role == "user"
                    and contents[-1].parts
                    and contents[-1].parts[-1].function_response is not None):
                contents[-1].parts.append(part)
            else:
                contents.append(types.Content(role="user", parts=[part]))
            continue

        if role == "assistant":
            parts: list = []
            text = m.get("content")
            if isinstance(text, str) and text:
                parts.append(types.Part.from_text(text=text))
            for tc in m.get("tool_calls") or []:
                tc_id = tc["id"] if isinstance(tc, dict) else tc.id
                tc_name = tc["name"] if isinstance(tc, dict) else tc.name
                tc_args = tc["arguments"] if isinstance(tc, dict) else tc.arguments
                if tc_id:
                    id_to_name[tc_id] = tc_name
                fcpart = types.Part.from_function_call(
                    name=tc_name, args=tc_args if isinstance(tc_args, dict) else {})
                # Required by Gemini-3; placeholder by design (see constant).
                fcpart.thought_signature = _PLACEHOLDER_THOUGHT_SIGNATURE
                parts.append(fcpart)
            if not parts:
                parts = [types.Part.from_text(text="(no output)")]
            contents.append(types.Content(role="model", parts=parts))
            continue

        # user (or any other) → plain text user turn
        text = m.get("content", "")
        if not isinstance(text, str):
            text = str(text)
        if not text.strip():
            text = "(no output)"
        contents.append(types.Content(
            role="user", parts=[types.Part.from_text(text=text)]))

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def _to_gemini_tools(tools: list[dict]):
    """OpenAI tool schema → [types.Tool]. Uses `parameters_json_schema` so the project's JSON Schemas pass through verbatim (no lossy Schema coercion)."""
    from google.genai import types

    decls = []
    for t in tools:
        fn = t["function"] if t.get("type") == "function" else t
        decls.append(types.FunctionDeclaration(
            name=fn["name"],
            description=fn.get("description", ""),
            parameters_json_schema=fn.get("parameters") or {"type": "object"},
        ))
    return [types.Tool(function_declarations=decls)]


def _to_gemini_tool_config(tool_choice):
    """Canonical tool_choice → types.ToolConfig.

    "auto"→AUTO, "required"/"any"→ANY, "none"→NONE, a bare tool-name string or OpenAI forced-function dict → ANY restricted to that name (matches the Bedrock `_bedrock_tool_choice` convention).
    """
    from google.genai import types
    Mode = types.FunctionCallingConfigMode

    mode = Mode.AUTO
    allowed = None
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") if tool_choice.get("type") == "function" else None
        if isinstance(fn, dict) and fn.get("name"):
            mode, allowed = Mode.ANY, [fn["name"]]
    elif tool_choice in ("required", "any"):
        mode = Mode.ANY
    elif tool_choice == "none":
        mode = Mode.NONE
    elif isinstance(tool_choice, str) and tool_choice != "auto":
        mode, allowed = Mode.ANY, [tool_choice]

    fcc = types.FunctionCallingConfig(mode=mode, allowed_function_names=allowed)
    return types.ToolConfig(function_calling_config=fcc)


def _parse_gemini_response(resp) -> ChatResponse:
    """Vertex Gemini response → canonical ChatResponse (one tool call, no text on tool-call turns, thought parts dropped)."""
    candidates = getattr(resp, "candidates", None) or []
    parts = []
    if candidates:
        content = getattr(candidates[0], "content", None)
        parts = (getattr(content, "parts", None) or []) if content else []

    text_parts: list[str] = []
    fn_calls: list = []
    for p in parts:
        if getattr(p, "thought", None):          # server-side reasoning, drop
            continue
        if getattr(p, "function_call", None) is not None:
            fn_calls.append(p.function_call)
        elif getattr(p, "text", None):
            text_parts.append(p.text)

    tool_calls: list[ToolCall] = []
    for fc in fn_calls[:1]:                       # first call only
        tool_calls.append(ToolCall(
            id=(getattr(fc, "id", None) or f"call_{uuid.uuid4().hex[:16]}"),
            name=getattr(fc, "name", "") or "",
            arguments=dict(getattr(fc, "args", None) or {}),
        ))

    content = None if tool_calls else ("".join(text_parts) or None)

    um = getattr(resp, "usage_metadata", None)
    prompt_toks = getattr(um, "prompt_token_count", 0) or 0 if um else 0
    out_toks = getattr(um, "candidates_token_count", 0) or 0 if um else 0
    total_toks = getattr(um, "total_token_count", 0) or 0 if um else 0

    finish = ""
    if candidates:
        fr = getattr(candidates[0], "finish_reason", None)
        finish = getattr(fr, "value", None) or (str(fr) if fr else "")

    try:
        raw = resp.model_dump(exclude_none=True)   # picklable dict for pkl/inspector
    except Exception:
        raw = None
    return ChatResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish,
        usage=Usage(prompt_tokens=prompt_toks, completion_tokens=out_toks,
                    total_tokens=total_toks or (prompt_toks + out_toks)),
        raw=raw,
    )


class GeminiChatGenerator(ChatGenerator):
    """Vertex-AI-backed Gemini generator (ADC user creds)."""

    SUPPORTS_TOOLS_NATIVE = True

    def __init__(
        self,
        model_id: str,
        *,
        cache=None,
        project: str | None = None,
        location: str | None = None,
        thinking_level: str | None = "low",
        max_retries: int = 8,
        base_sleep_s: float = 4.0,
        max_sleep_s: float = 512.0,
    ):
        super().__init__(model_id=model_id, cache=cache)
        self.gemini_model_id = GEMINI_MODEL_CONFIG.get(model_id, model_id)
        self.project = project or os.environ.get("GCP_PROJECT")
        self.location = location or os.environ.get("GCP_LOCATION", "global")
        # Low thinking level mirrors the reasoning_effort="low" policy on the Bedrock reasoning models; env-overridable, set to "" / "none" to omit.
        self.thinking_level = os.environ.get("GEMINI_THINKING_LEVEL", thinking_level)
        self.max_retries = max_retries
        self.base_sleep_s = base_sleep_s
        self.max_sleep_s = max_sleep_s
        # Lazy import so unit tests / docs don't require google-genai.
        from google import genai
        self.client = genai.Client(
            vertexai=True, project=self.project, location=self.location)

    def _chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        **kwargs,
    ) -> ChatResponse:
        from google.genai import types
        from google.genai import errors as genai_errors

        system_instruction, contents = _to_gemini_contents(messages)
        cfg_kwargs: dict = {}
        if system_instruction:
            cfg_kwargs["system_instruction"] = system_instruction
        if "max_tokens" in kwargs:
            cfg_kwargs["max_output_tokens"] = int(kwargs["max_tokens"])
        if "temperature" in kwargs:
            cfg_kwargs["temperature"] = float(kwargs["temperature"])
        if "top_p" in kwargs:
            cfg_kwargs["top_p"] = float(kwargs["top_p"])
        if tools:
            cfg_kwargs["tools"] = _to_gemini_tools(tools)
            cfg_kwargs["tool_config"] = _to_gemini_tool_config(tool_choice)
        if self.thinking_level and self.thinking_level not in ("none", ""):
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                include_thoughts=False, thinking_level=self.thinking_level)
        else:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                include_thoughts=False)
        config = types.GenerateContentConfig(**cfg_kwargs)

        sleep_s = self.base_sleep_s
        last_exc = None
        for _ in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.gemini_model_id, contents=contents, config=config)
                parsed = _parse_gemini_response(resp)
                # An empty response (no text, no tool call) means Gemini flaked, most often FinishReason MALFORMED_FUNCTION_CALL (a known intermittent failure of the thinking models: an unparseable functionCall makes Vertex return zero parts); temperature has variance, so re-sampling almost always recovers.
                if parsed.content is None and not parsed.tool_calls:
                    last_exc = f"empty response (finish={parsed.finish_reason})"
                    print(f"[Gemini] retryable ({last_exc}); resample")
                    time.sleep(1.0)
                    continue
                return parsed
            except genai_errors.APIError as e:
                last_exc = e
                if getattr(e, "code", None) in _RETRY_CODES:
                    print(f"[Gemini] retryable ({e.code}): {e}; sleep {sleep_s}s")
                    time.sleep(sleep_s)
                    sleep_s = min(sleep_s * 2, self.max_sleep_s)
                    continue
                raise
            except (ConnectionError, TimeoutError) as e:
                last_exc = e
                print(f"[Gemini] retryable ({e.__class__.__name__}): {e}; "
                      f"sleep {sleep_s}s")
                time.sleep(sleep_s)
                sleep_s = min(sleep_s * 2, self.max_sleep_s)
                continue
        raise RuntimeError(
            f"Gemini call failed after {self.max_retries} retries: {last_exc}")

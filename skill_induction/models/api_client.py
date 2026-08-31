"""Bedrock-backed ChatGenerator using boto3's Converse API.

The MoE API models in the paper (Qwen3-235B-A22B, GPT-OSS-120B, Nemotron-3-Super-120B) are served through Amazon Bedrock (boto3, not the OpenAI HTTP path).

Retry semantics: exponential backoff on throttling, connection, and model errors.
"""

from __future__ import annotations

import os
import time

from skill_induction.models.base import (
    ChatGenerator,
    ChatResponse,
    parse_bedrock_response,
    to_bedrock_payload,
)


# Short alias to Bedrock model identifier; limited to the API models used in the paper's experiments.
BEDROCK_MODEL_CONFIG: dict[str, str] = {
    "qwen-3-235b-a22b-amazon":   "qwen.qwen3-235b-a22b-2507-v1:0",
    "nemotron-3-super-120b-amazon": "nvidia.nemotron-super-3-120b",
    "gpt-oss-120b-amazon":       "openai.gpt-oss-120b-1:0",
}


# Models that reject `temperature` / `top_p` / `top_k`: those kwargs are stripped from the Bedrock `inferenceConfig` before the request goes out.
_NO_SAMPLING_PARAMS: set[str] = set()


# Per-model reasoning policy: models that support a `reasoning_effort` parameter on Bedrock are opted into "low" effort, so the chain-of-thought arrives in a separate `reasoningContent` block (folded into `content` by `parse_bedrock_response` to persist across turns). gpt-oss-120b's floor effort is "low"; nemotron-3-super-120b supports the param; qwen-3-235b-a22b-2507 (Instruct release) does not, and Bedrock silently ignores unknown additionalModelRequestFields.
BEDROCK_REASONING: dict[str, str] = {
    "gpt-oss-120b-amazon": "low",
    "nemotron-3-super-120b-amazon": "low",
}


_RETRY_CODES = (
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelErrorException",
    "InternalServerException",
    "ModelTimeoutException",
)


class BedrockChatGenerator(ChatGenerator):
    """ChatGenerator backed by Bedrock Converse.

    `model_id` accepts either a short alias from `BEDROCK_MODEL_CONFIG` (preferred, keeps logs human-readable) or a literal Bedrock model id.
    """

    SUPPORTS_TOOLS_NATIVE = True

    def __init__(
        self,
        model_id: str,
        *,
        cache=None,
        region: str | None = None,
        bearer_token: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        max_retries: int = 8,
        base_sleep_s: float = 4.0,
        max_sleep_s: float = 512.0,
    ):
        super().__init__(model_id=model_id, cache=cache)
        self.bedrock_model_id = BEDROCK_MODEL_CONFIG.get(model_id, model_id)
        self.region = region or os.environ.get("AWS_REGION", "us-west-2")
        # Two auth paths: explicit IAM access-key/secret or a Bedrock bearer token. Explicit keys win and skip the bearer env var so the two never conflict.
        creds: dict[str, str] = {}
        if aws_access_key_id and aws_secret_access_key:
            creds = {
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key,
            }
        elif bearer_token:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = bearer_token
        # Lazy import so unit tests / docs don't require boto3.
        import boto3
        from botocore.config import Config as _BotoConfig
        # Without a read/connect timeout a hung Bedrock connection never raises, so the retry loop never fires and the whole cell freezes indefinitely; with read_timeout the stuck call raises ReadTimeoutError, which the retry loop below handles. boto's own retries are disabled (max_attempts=0) since we run our own backoff loop.
        self.client = boto3.client(
            "bedrock-runtime", region_name=self.region, **creds,
            config=_BotoConfig(
                read_timeout=int(os.environ.get("BEDROCK_READ_TIMEOUT", "300")),
                connect_timeout=int(os.environ.get("BEDROCK_CONNECT_TIMEOUT", "30")),
                retries={"max_attempts": 0},
            ))
        self.max_retries = max_retries
        self.base_sleep_s = base_sleep_s
        self.max_sleep_s = max_sleep_s

    def _chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        **kwargs,
    ) -> ChatResponse:
        payload = to_bedrock_payload(messages, tools=tools, tool_choice=tool_choice)
        inference_config: dict = {}
        if "max_tokens" in kwargs:
            inference_config["maxTokens"] = int(kwargs["max_tokens"])
        accepts_sampling = self.model_id not in _NO_SAMPLING_PARAMS
        if accepts_sampling and "temperature" in kwargs:
            inference_config["temperature"] = float(kwargs["temperature"])
        if accepts_sampling and "top_p" in kwargs:
            inference_config["topP"] = float(kwargs["top_p"])

        params = {
            "modelId": self.bedrock_model_id,
            "messages": payload["messages"],
            "inferenceConfig": inference_config,
        }
        if "system" in payload:
            params["system"] = payload["system"]
        if "toolConfig" in payload:
            params["toolConfig"] = payload["toolConfig"]
        # Centralised no-hidden-thinking policy (see BEDROCK_REASONING); an explicit kwarg, if ever passed, still wins.
        reasoning = kwargs.get(
            "reasoning_effort", BEDROCK_REASONING.get(self.model_id))
        if reasoning is not None:
            params["additionalModelRequestFields"] = {
                "reasoning_effort": reasoning
            }

        sleep_s = self.base_sleep_s
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.converse(**params)
                return parse_bedrock_response(resp)
            except Exception as e:
                last_exc = e
                code = ""
                response_obj = getattr(e, "response", None)
                if isinstance(response_obj, dict):
                    code = response_obj.get("Error", {}).get("Code", "")
                cls_name = e.__class__.__name__
                if code in _RETRY_CODES or cls_name in (
                    "EndpointConnectionError",
                    "ConnectionClosedError",
                    "ReadTimeoutError",
                ):
                    print(f"[Bedrock] retryable ({code or cls_name}): {e}; sleep {sleep_s}s")
                    time.sleep(sleep_s)
                    sleep_s = min(sleep_s * 2, self.max_sleep_s)
                    continue
                raise
        raise RuntimeError(
            f"Bedrock call failed after {self.max_retries} retries: {last_exc}"
        )

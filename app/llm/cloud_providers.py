"""Optional cloud providers — Bedrock and OpenAI.

Neither is needed to run the demo. They exist to show the adapter pattern
scales to production backends (and because the author uses Bedrock at work).
Both import their SDKs lazily so the package works without them installed.
"""
from __future__ import annotations

import json
import os

from .base import LLMError, LLMProvider


class BedrockProvider(LLMProvider):
    """AWS Bedrock via boto3. Requires `pip install boto3` + AWS creds."""

    name = "bedrock"

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
        region: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.region = region or os.getenv("AWS_REGION", "eu-west-1")

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        try:
            import boto3  # noqa: PLC0415 (lazy import keeps boto3 optional)
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError as exc:
            raise LLMError("boto3 not installed. Run `pip install boto3`.") from exc

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system

        try:
            client = boto3.client("bedrock-runtime", region_name=self.region)
            resp = client.invoke_model(modelId=self.model_id, body=json.dumps(body))
        except (BotoCoreError, ClientError) as exc:
            raise LLMError(f"Bedrock call failed: {exc}") from exc

        payload = json.loads(resp["body"].read())
        try:
            return payload["content"][0]["text"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Bedrock response shape: {payload}") from exc


class OpenAIProvider(LLMProvider):
    """OpenAI via the official SDK. Requires `pip install openai` + API key."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        try:
            from openai import OpenAI, OpenAIError  # noqa: PLC0415
        except ImportError as exc:
            raise LLMError("openai not installed. Run `pip install openai`.") from exc

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            client = OpenAI()  # reads OPENAI_API_KEY from env
            resp = client.chat.completions.create(model=self.model, messages=messages)
        except OpenAIError as exc:
            raise LLMError(f"OpenAI call failed: {exc}") from exc

        return resp.choices[0].message.content.strip()


class HuggingFaceProvider(LLMProvider):
    """Hugging Face Inference API — free tier, ideal for public demos.

    Reads ``HF_TOKEN`` from env (create one at huggingface.co/settings/tokens).
    """

    name = "huggingface"

    # Comma-separated list of fallbacks. HF's free `hf-inference` tier prunes
    # models regularly, so we try several non-gated chat models in order.
    DEFAULT_MODELS = (
        "Qwen/Qwen2.5-7B-Instruct,"
        "mistralai/Mistral-7B-Instruct-v0.3,"
        "HuggingFaceH4/zephyr-7b-beta,"
        "microsoft/Phi-3-mini-4k-instruct"
    )

    def __init__(self, model: str | None = None) -> None:
        raw = model or os.getenv("HF_MODEL") or self.DEFAULT_MODELS
        self.models = [m.strip() for m in raw.split(",") if m.strip()]

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        try:
            from huggingface_hub import InferenceClient  # noqa: PLC0415
        except ImportError as exc:
            raise LLMError(
                "huggingface_hub not installed. Run `pip install huggingface_hub`."
            ) from exc

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        token = os.getenv("HF_TOKEN")
        errors: list[str] = []
        for model in self.models:
            try:
                # Force the free HF-hosted provider; otherwise the router may
                # pick a paid one (Together, Fireworks, ...) the user hasn't
                # enabled.
                client = InferenceClient(
                    model=model, token=token, provider="hf-inference"
                )
                resp = client.chat_completion(messages=messages, max_tokens=512)
            except Exception as exc:
                errors.append(f"{model}: {exc}")
                continue

            try:
                return resp.choices[0].message.content.strip()
            except (AttributeError, IndexError):
                errors.append(f"{model}: unexpected response shape {resp}")
                continue

        raise LLMError(
            "All Hugging Face models failed. Tried:\n- " + "\n- ".join(errors)
        )

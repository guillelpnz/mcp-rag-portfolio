"""Provider selection from a single env var: LLM_PROVIDER.

Defaults to the free local Ollama backend so the demo runs out of the box.
"""
from __future__ import annotations

import os

from .base import LLMError, LLMProvider
from .cloud_providers import BedrockProvider, OpenAIProvider
from .ollama_provider import OllamaProvider

_PROVIDERS = {
    "ollama": OllamaProvider,
    "bedrock": BedrockProvider,
    "openai": OpenAIProvider,
}


def get_provider() -> LLMProvider:
    """Build the provider named by ``LLM_PROVIDER`` (default: ollama)."""
    choice = os.getenv("LLM_PROVIDER", "ollama").lower()
    cls = _PROVIDERS.get(choice)
    if cls is None:
        raise LLMError(
            f"Unknown LLM_PROVIDER '{choice}'. Valid: {', '.join(_PROVIDERS)}."
        )
    return cls()


__all__ = ["get_provider", "LLMProvider", "LLMError"]

"""LLM provider abstraction.

A single Protocol every provider implements, so the rest of the app never
cares which backend is running. Swapping Ollama -> Bedrock -> OpenAI is a
one-line config change, not a code change.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


class LLMError(RuntimeError):
    """Raised when a provider fails to return a completion.

    Wraps provider-specific errors (network, auth, rate limit) into one type
    the API layer can catch and turn into a clean HTTP response.
    """


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal contract for a text-generation backend."""

    name: str

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Return a completion for ``prompt``.

        Implementations MUST raise :class:`LLMError` on any failure rather
        than leaking provider-specific exceptions.
        """
        ...

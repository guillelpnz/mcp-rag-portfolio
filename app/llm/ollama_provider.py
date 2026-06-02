"""Ollama provider — the free default.

Runs fully local models (llama3.2, qwen2.5, etc.) with no API key and no
billing. This is what makes the whole demo runnable by anyone with
`docker compose up`.
"""
from __future__ import annotations

import os

import httpx

from .base import LLMError, LLMProvider


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        payload: dict = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system

        try:
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"Ollama timed out after {self.timeout}s. Is the model '{self.model}' "
                "pulled and the server warm?"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"Ollama returned {exc.response.status_code}. "
                f"Check that model '{self.model}' exists (`ollama pull {self.model}`)."
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

        data = resp.json()
        text = data.get("response", "").strip()
        if not text:
            raise LLMError("Ollama returned an empty completion.")
        return text

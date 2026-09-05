"""Narrow native Ollama HTTP adapter."""

from __future__ import annotations

import math
from typing import Any

import httpx


class ProviderError(RuntimeError):
    """A configured local model provider failed or returned an invalid payload."""


class OllamaClient:
    def __init__(
        self, base_url: str, timeout: float = 10.0, transport: httpx.BaseTransport | None = None
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with httpx.Client(
                base_url=self.base_url, timeout=self.timeout, transport=self.transport
            ) as client:
                response = client.post(path, json=payload)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("response is not an object")
                return data
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(f"Ollama request to {path} failed: {exc}") from exc

    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        data = self._post("/api/embed", {"model": model, "input": inputs})
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise ProviderError("Ollama /api/embed must return a nonempty matrix")
        if len(embeddings) != len(inputs):
            raise ProviderError("Ollama /api/embed must return one embedding row per input")

        converted: list[list[float]] = []
        for row in embeddings:
            if not isinstance(row, list) or not row:
                raise ProviderError("Ollama /api/embed must return nonempty rows")
            try:
                converted_row = [float(value) for value in row]
            except (TypeError, ValueError) as exc:
                raise ProviderError(
                    "Ollama /api/embed rows must contain only finite numeric values"
                ) from exc
            if not all(math.isfinite(value) for value in converted_row):
                raise ProviderError(
                    "Ollama /api/embed rows must contain only finite numeric values"
                )
            converted.append(converted_row)

        if len({len(row) for row in converted}) != 1:
            raise ProviderError("Ollama /api/embed rows must have equal dimensionality")
        return converted

    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        data = self._post("/api/chat", {"model": model, "messages": messages, "stream": False})
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ProviderError("Ollama /api/chat returned no message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("Ollama /api/chat returned empty message.content")
        return content.strip()

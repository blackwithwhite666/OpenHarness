"""Client for the telegent-style inference service used by eval mining."""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_INFERENCE_URL = "https://inference.worfalomey.top"


class InferenceClient:
    """Small async client for dense embeddings and reranking."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    @classmethod
    def from_env(
        cls,
        *,
        default_url: str = DEFAULT_INFERENCE_URL,
        timeout: float = 120.0,
    ) -> "InferenceClient":
        """Build a client from ``INFERENCE_URL`` or the shared hosted endpoint."""
        return cls(os.getenv("INFERENCE_URL") or default_url, timeout=timeout)

    async def health(self) -> dict[str, Any]:
        """Return inference service health payload."""
        response = await self._client.get("/health")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("inference /health response must be a JSON object")
        return payload

    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool = True,
        return_sparse: bool = False,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Embed texts through ``POST /embed``."""
        payload: dict[str, Any] = {
            "texts": texts,
            "return_dense": return_dense,
            "return_sparse": return_sparse,
        }
        if batch_size is not None:
            payload["batch_size"] = batch_size
        response = await self._client.post("/embed", json=payload)
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise RuntimeError("inference /embed response must be a JSON object")
        return result

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        max_length: int = 1024,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Rerank documents through ``POST /rerank``."""
        payload: dict[str, Any] = {
            "query": query,
            "documents": documents,
            "max_length": max_length,
        }
        if batch_size is not None:
            payload["batch_size"] = batch_size
        response = await self._client.post("/rerank", json=payload)
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise RuntimeError("inference /rerank response must be a JSON object")
        return result

    async def aclose(self) -> None:
        """Close the owned underlying HTTP client."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "InferenceClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.aclose()

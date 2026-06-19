from __future__ import annotations

import json

import httpx
import pytest

from openharness.evals.inference import DEFAULT_INFERENCE_URL, InferenceClient


@pytest.mark.asyncio
async def test_inference_client_uses_telegent_embed_and_rerank_contract():
    seen_requests: list[tuple[str, str, dict[str, object] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8")) if request.content else None
        seen_requests.append((request.method, request.url.path, payload))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/embed":
            assert payload == {
                "texts": ["hello", "world"],
                "return_dense": True,
                "return_sparse": False,
                "batch_size": 2,
            }
            return httpx.Response(
                200,
                json={
                    "model": "BAAI/bge-m3",
                    "count": 2,
                    "dense": [[0.1, 0.2], [0.3, 0.4]],
                    "lexical_weights": None,
                },
            )
        if request.url.path == "/rerank":
            assert payload == {
                "query": "hello",
                "documents": ["a", "b"],
                "max_length": 1024,
                "batch_size": 2,
            }
            return httpx.Response(200, json={"model": "reranker", "scores": [0.9, 0.1]})
        return httpx.Response(404)

    http_client = httpx.AsyncClient(
        base_url="https://inference.test",
        transport=httpx.MockTransport(handler),
    )
    client = InferenceClient("https://inference.test/", client=http_client)

    assert client.base_url == "https://inference.test"
    assert await client.health() == {"status": "ok"}
    assert await client.embed(["hello", "world"], batch_size=2) == {
        "model": "BAAI/bge-m3",
        "count": 2,
        "dense": [[0.1, 0.2], [0.3, 0.4]],
        "lexical_weights": None,
    }
    assert await client.rerank("hello", ["a", "b"], batch_size=2) == {
        "model": "reranker",
        "scores": [0.9, 0.1],
    }

    assert seen_requests == [
        ("GET", "/health", None),
        (
            "POST",
            "/embed",
            {
                "texts": ["hello", "world"],
                "return_dense": True,
                "return_sparse": False,
                "batch_size": 2,
            },
        ),
        (
            "POST",
            "/rerank",
            {
                "query": "hello",
                "documents": ["a", "b"],
                "max_length": 1024,
                "batch_size": 2,
            },
        ),
    ]
    await http_client.aclose()


@pytest.mark.asyncio
async def test_inference_client_from_env_uses_hosted_default(monkeypatch):
    monkeypatch.delenv("INFERENCE_URL", raising=False)

    client = InferenceClient.from_env()

    assert client.base_url == DEFAULT_INFERENCE_URL
    await client.aclose()

"""Ohmo entrypoints for eval/data-flywheel embedding indexes."""

from __future__ import annotations

from pathlib import Path

from openharness.evals.embeddings import (
    EmbeddingClient,
    EvalEmbeddingIndexWrite,
    write_embedding_index,
)
from openharness.evals.inference import InferenceClient

from ohmo.evals.adapter import get_eval_store


async def write_ohmo_embedding_index(
    *,
    workspace: str | Path | None = None,
    client: EmbeddingClient | None = None,
    inference_url: str | None = None,
    batch_size: int = 32,
) -> EvalEmbeddingIndexWrite:
    """Build an embedding index for captured Ohmo eval episodes."""
    store = get_eval_store(workspace)
    if client is not None and inference_url is not None:
        raise ValueError("client and inference_url are mutually exclusive")
    inference_client = client or (
        InferenceClient(inference_url) if inference_url else InferenceClient.from_env()
    )
    try:
        return await write_embedding_index(
            store=store,
            client=inference_client,
            batch_size=batch_size,
        )
    finally:
        if client is None:
            await inference_client.aclose()

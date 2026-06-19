"""Dense embedding materialization for eval/data-flywheel facets."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from openharness.evals.facets import EvalTextFacetInput, collect_text_facets
from openharness.evals.models import EvalEmbeddingManifest, EvalEmbeddingRecord
from openharness.evals.store import EvalStore
from openharness.utils.fs import atomic_write_text


class EmbeddingClient(Protocol):
    """Protocol implemented by telegent-style inference clients."""

    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool = True,
        return_sparse: bool = False,
        batch_size: int | None = None,
    ) -> dict[str, object]:
        """Return dense vectors for texts."""


@dataclass(frozen=True)
class EvalEmbeddingIndexWrite:
    """Summary returned after writing a dense embedding index."""

    manifest: EvalEmbeddingManifest
    manifest_path: Path
    records_path: Path
    manifest_relative_path: str
    records_relative_path: str


async def write_embedding_index(
    *,
    store: EvalStore,
    client: EmbeddingClient,
    facets: Sequence[EvalTextFacetInput] | None = None,
    batch_size: int = 32,
    records_filename: str = "embedding_records.jsonl",
    manifest_filename: str = "embedding_manifest.json",
) -> EvalEmbeddingIndexWrite:
    """Embed text facets, write JSONL records, and refresh SQLite lookup rows."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    facet_inputs = list(facets) if facets is not None else collect_text_facets(store)
    records_path = _embedding_output_path(store, records_filename)
    manifest_path = _embedding_output_path(store, manifest_filename)

    records = await _embed_facets(client, facet_inputs, batch_size=batch_size)
    records_relative_path = records_path.relative_to(store.root).as_posix()
    manifest = _build_manifest(
        records=records,
        records_relative_path=records_relative_path,
        facet_count=len(facet_inputs),
    )

    _write_records(records_path, records)
    atomic_write_text(manifest_path, manifest.model_dump_json(indent=2) + "\n")
    store.replace_embedding_index(records, jsonl_path=records_relative_path)

    return EvalEmbeddingIndexWrite(
        manifest=manifest,
        manifest_path=manifest_path,
        records_path=records_path,
        manifest_relative_path=manifest_path.relative_to(store.root).as_posix(),
        records_relative_path=records_relative_path,
    )


async def _embed_facets(
    client: EmbeddingClient,
    facet_inputs: Sequence[EvalTextFacetInput],
    *,
    batch_size: int,
) -> list[EvalEmbeddingRecord]:
    records: list[EvalEmbeddingRecord] = []
    expected_dimensions: int | None = None
    expected_model: str | None = None

    for batch_start in range(0, len(facet_inputs), batch_size):
        batch = list(facet_inputs[batch_start : batch_start + batch_size])
        response = await client.embed(
            [item.text for item in batch],
            return_dense=True,
            return_sparse=False,
            batch_size=batch_size,
        )
        model = _response_model(response)
        if expected_model is None:
            expected_model = model
        elif model != expected_model:
            raise RuntimeError("embedding response model changed within one index build")

        count = response.get("count")
        if count is not None and count != len(batch):
            raise RuntimeError("embedding response count does not match input texts")

        dense = response.get("dense")
        if not isinstance(dense, list):
            raise RuntimeError("embedding response must include dense vectors")
        if len(dense) != len(batch):
            raise RuntimeError("embedding response dense vector count does not match input texts")

        for item, vector_value in zip(batch, dense):
            vector = _vector(vector_value)
            dimensions = len(vector)
            if expected_dimensions is None:
                expected_dimensions = dimensions
            elif dimensions != expected_dimensions:
                raise RuntimeError("embedding vector dimensions changed within one index build")
            records.append(
                EvalEmbeddingRecord(
                    facet=item.facet,
                    model=model,
                    dimensions=dimensions,
                    vector=vector,
                    metadata={
                        "provider": "telegent_inference",
                        "return_dense": True,
                        "return_sparse": False,
                    },
                )
            )

    return records


def _build_manifest(
    *,
    records: Sequence[EvalEmbeddingRecord],
    records_relative_path: str,
    facet_count: int,
) -> EvalEmbeddingManifest:
    if not records:
        return EvalEmbeddingManifest(
            model="",
            dimensions=0,
            records_path=records_relative_path,
            facet_count=facet_count,
            embedding_count=0,
            skipped_count=facet_count,
            metadata={"provider": "telegent_inference"},
        )
    first = records[0]
    return EvalEmbeddingManifest(
        model=first.model,
        dimensions=first.dimensions,
        records_path=records_relative_path,
        facet_count=facet_count,
        embedding_count=len(records),
        skipped_count=facet_count - len(records),
        metadata={"provider": "telegent_inference"},
    )


def _write_records(path: Path, records: Sequence[EvalEmbeddingRecord]) -> None:
    lines = [record.model_dump_json() for record in records]
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    atomic_write_text(path, payload)


def _response_model(response: dict[str, object]) -> str:
    model = response.get("model")
    if not isinstance(model, str) or not model:
        raise RuntimeError("embedding response must include a non-empty model")
    return model


def _vector(value: object) -> list[float]:
    if not isinstance(value, list) or not value:
        raise RuntimeError("embedding vector must be a non-empty list")
    vector: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise RuntimeError("embedding vector values must be finite numbers")
        vector.append(float(item))
    return vector


def _embedding_output_path(store: EvalStore, filename: str) -> Path:
    embeddings_dir = store.root / "embeddings"
    output_path = (embeddings_dir / filename).resolve()
    if not _is_relative_to(output_path, embeddings_dir.resolve()):
        raise ValueError("embedding output filename must stay under store.root/embeddings")
    return output_path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True

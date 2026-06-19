from __future__ import annotations

import json
from pathlib import Path

import pytest

from openharness.evals import EvalEpisode, EvalEvent, EvalStore, write_embedding_index


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool = True,
        return_sparse: bool = False,
        batch_size: int | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "texts": texts,
                "return_dense": return_dense,
                "return_sparse": return_sparse,
                "batch_size": batch_size,
            }
        )
        return {
            "model": "BAAI/bge-m3",
            "count": len(texts),
            "dense": [[float(index), float(len(text))] for index, text in enumerate(texts)],
            "lexical_weights": None,
        }


class MismatchedEmbeddingClient(FakeEmbeddingClient):
    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool = True,
        return_sparse: bool = False,
        batch_size: int | None = None,
    ) -> dict[str, object]:
        await super().embed(
            texts,
            return_dense=return_dense,
            return_sparse=return_sparse,
            batch_size=batch_size,
        )
        return {
            "model": "BAAI/bge-m3",
            "count": len(texts),
            "dense": [[1.0, 2.0]],
            "lexical_weights": None,
        }


@pytest.mark.asyncio
async def test_write_embedding_index_materializes_dense_records_without_raw_text(
    tmp_path: Path,
):
    store = EvalStore(tmp_path / "evals")
    _add_text_episode(store)
    client = FakeEmbeddingClient()

    result = await write_embedding_index(store=store, client=client, batch_size=1)

    assert result.manifest_path == store.root / "embeddings" / "embedding_manifest.json"
    assert result.records_path == store.root / "embeddings" / "embedding_records.jsonl"
    assert result.manifest.records_path == "embeddings/embedding_records.jsonl"
    assert result.manifest.model == "BAAI/bge-m3"
    assert result.manifest.dimensions == 2
    assert result.manifest.facet_count == 3
    assert result.manifest.embedding_count == 3
    assert result.manifest.skipped_count == 0
    assert store.count_embedding_records() == 3

    assert client.calls == [
        {
            "texts": ["Secret launch goal"],
            "return_dense": True,
            "return_sparse": False,
            "batch_size": 1,
        },
        {
            "texts": ["Please compare private vendors"],
            "return_dense": True,
            "return_sparse": False,
            "batch_size": 1,
        },
        {
            "texts": ["private tool summary"],
            "return_dense": True,
            "return_sparse": False,
            "batch_size": 1,
        },
    ]

    record_rows = [
        json.loads(line)
        for line in result.records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["dimensions"] for row in record_rows] == [2, 2, 2]
    assert [row["model"] for row in record_rows] == ["BAAI/bge-m3"] * 3
    assert {row["facet"]["facet_kind"] for row in record_rows} == {
        "user_goal",
        "user_request",
        "tool_input",
    }

    serialized_outputs = (
        result.records_path.read_text(encoding="utf-8")
        + result.manifest_path.read_text(encoding="utf-8")
    )
    for sensitive_fragment in (
        "Secret launch goal",
        "Please compare private vendors",
        "private tool summary",
        "private full tool input",
    ):
        assert sensitive_fragment not in serialized_outputs


@pytest.mark.asyncio
async def test_write_embedding_index_rejects_response_shape_mismatch(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_text_episode(store)

    with pytest.raises(RuntimeError, match="dense vector count"):
        await write_embedding_index(
            store=store,
            client=MismatchedEmbeddingClient(),
            batch_size=32,
        )

    assert store.count_embedding_records() == 0
    assert not (store.root / "embeddings" / "embedding_records.jsonl").exists()


@pytest.mark.asyncio
async def test_write_embedding_index_rejects_escaped_output_paths(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")

    with pytest.raises(ValueError, match="store.root/embeddings"):
        await write_embedding_index(
            store=store,
            client=FakeEmbeddingClient(),
            records_filename="../escaped.jsonl",
        )


def _add_text_episode(store: EvalStore) -> None:
    store.append_episode(
        EvalEpisode(
            episode_id="ep-1",
            source="gateway",
            app="ohmo",
            session_id="session-1",
            user_goal="Secret launch goal",
            user_text="Please compare private vendors",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id="ep-1",
            kind="tool_started",
            tool_name="web_fetch",
            tool_call_id="tool-1",
            payload={
                "input_summary": "private tool summary",
                "input": "private full tool input",
            },
        )
    )

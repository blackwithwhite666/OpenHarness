from __future__ import annotations

from pathlib import Path

import pytest

from openharness.evals import EvalEpisode
from ohmo.evals import get_eval_store, write_ohmo_embedding_index


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool = True,
        return_sparse: bool = False,
        batch_size: int | None = None,
    ) -> dict[str, object]:
        self.texts.extend(texts)
        return {
            "model": "BAAI/bge-m3",
            "count": len(texts),
            "dense": [[0.1, 0.2, 0.3] for _ in texts],
            "lexical_weights": None,
        }


@pytest.mark.asyncio
async def test_write_ohmo_embedding_index_uses_workspace_eval_store(tmp_path: Path):
    workspace = tmp_path / "workspace"
    store = get_eval_store(workspace)
    store.append_episode(
        EvalEpisode(
            episode_id="ep-1",
            source="gateway",
            app="ohmo",
            session_id="session-1",
            user_text="private ohmo request",
        )
    )
    client = FakeEmbeddingClient()

    result = await write_ohmo_embedding_index(
        workspace=workspace,
        client=client,
        batch_size=8,
    )

    assert result.manifest_path == workspace.resolve() / "evals" / "embeddings" / (
        "embedding_manifest.json"
    )
    assert result.manifest.embedding_count == 1
    assert result.manifest.dimensions == 3
    assert store.count_embedding_records() == 1
    assert client.texts == ["private ohmo request"]
    assert "private ohmo request" not in result.records_path.read_text(encoding="utf-8")

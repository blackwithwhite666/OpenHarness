"""Tests for per-turn catalog-vs-Honcho shadow comparison."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from ohmo.memory_backend import ShadowMemoryBackend


class _FakeBase:
    async def search(self, query: str, top_k: int) -> list[object]:
        del query, top_k
        return [
            SimpleNamespace(name="fact-1", rank=1, snippet="wife is Marina"),
            SimpleNamespace(name="fact-2", rank=2, snippet="likes concise replies"),
        ]


class _FakeHoncho:
    def __init__(self) -> None:
        self.calls = 0

    async def query_conclusions(self, query: str, **kwargs: object) -> list[object]:
        del query, kwargs
        self.calls += 1
        return [
            SimpleNamespace(content="my wife is Marina"),
            SimpleNamespace(content="I like concise replies"),
        ]

    async def create_messages(
        self,
        session: str,
        messages: list[dict[str, object]],
    ) -> list[object]:
        del session, messages
        return []


async def test_derived_recall_logs_shadow_comparison_without_second_honcho_query(
    tmp_path: Path,
) -> None:
    fake_honcho = _FakeHoncho()
    comparison_log_path = tmp_path / "shadow.jsonl"
    backend = ShadowMemoryBackend(
        _FakeBase(),  # type: ignore[arg-type]
        honcho_client=fake_honcho,  # type: ignore[arg-type]
        comparison_log_path=comparison_log_path,
        is_owner=True,
    )

    block = await backend.derived_recall_block(
        "who is my wife?",
        budget=4_000,
        timeout=1.0,
    )
    await backend.await_pending()

    assert block is not None
    assert fake_honcho.calls == 1
    assert comparison_log_path.is_file()
    record = json.loads(comparison_log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["query"] == "who is my wife?"
    assert record["catalog_hits"]
    assert record["honcho_hits"]
    assert all(hit["id"] is None for hit in record["honcho_hits"])

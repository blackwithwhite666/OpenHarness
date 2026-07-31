"""Tests for catalog-authoritative, off-path Honcho shadow recall."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from ohmo.gateway.models import GatewayConfig
from ohmo.memory_backend import (
    CatalogMemoryBackend,
    FileMemoryBackend,
    ShadowMemoryBackend,
    make_memory_backend,
    shadow_report,
)
from ohmo.memory_catalog import MemoryCatalog


@dataclass(frozen=True)
class FakeConclusion:
    id: str
    content: str


class FakeHoncho:
    def __init__(
        self,
        conclusions: list[FakeConclusion] | None = None,
        *,
        wait: bool = False,
        fail: bool = False,
    ) -> None:
        self.conclusions = conclusions or []
        self.wait = wait
        self.fail = fail
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.queries: list[tuple[str, str, str, int]] = []
        self.create_messages_calls = 0
        self.dream_calls = 0

    async def query_conclusions(
        self,
        query: str,
        *,
        observer: str,
        observed: str,
        top_k: int,
    ) -> list[FakeConclusion]:
        self.queries.append((query, observer, observed, top_k))
        self.started.set()
        if self.wait:
            await self.release.wait()
        if self.fail:
            raise OSError("fake Honcho is unavailable")
        return self.conclusions

    async def create_messages(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.create_messages_calls += 1
        raise AssertionError("shadow mode must never ingest messages")

    async def dialectic(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.dream_calls += 1
        raise AssertionError("shadow mode must never call a dream/deriver route")


class FakeHonchoExchange(FakeHoncho):
    def __init__(
        self,
        *args: object,
        fail_exchange: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fail_exchange = fail_exchange
        self.exchange_calls: list[tuple[str, list[dict[str, object]]]] = []

    async def create_messages(self, session: str, messages: object) -> list[object]:
        self.create_messages_calls += 1
        self.exchange_calls.append((session, [dict(message) for message in messages]))
        if self.fail_exchange:
            raise OSError("fake Honcho exchange ingestion failed")
        return []


def _catalog_backend(workspace: Path) -> CatalogMemoryBackend:
    catalog = MemoryCatalog(workspace)
    assert catalog.add("owner", "Timezone", "User prefers Europe/Moscow.").ok
    assert catalog.add("owner", "Editor", "User prefers Vim for quick edits.").ok
    return CatalogMemoryBackend(catalog, workspace)


async def test_shadow_model_surface_is_byte_identical_to_catalog(tmp_path: Path):
    plain = _catalog_backend(tmp_path)
    fake = FakeHoncho(
        [
            FakeConclusion("honcho-extra", "A learned fact absent from the catalog."),
            FakeConclusion("honcho-different", "A conflicting learned preference."),
        ]
    )
    shadow = ShadowMemoryBackend(
        plain,
        honcho_client=fake,  # type: ignore[arg-type]
        comparison_log_path=tmp_path / "comparison.jsonl",
    )

    plain_prompt = await plain.render_prompt()
    shadow_prompt = await shadow.render_prompt()
    plain_hits = await plain.search("prefers", 10)
    shadow_hits = await shadow.search("prefers", 10)

    assert shadow_prompt == plain_prompt
    assert shadow_hits == plain_hits
    assert "learned fact" not in shadow_prompt
    assert all("learned fact" not in hit.snippet for hit in shadow_hits)
    await shadow.await_pending()


async def test_shadow_query_is_off_path_and_writes_comparison_after_drain(tmp_path: Path):
    base = _catalog_backend(tmp_path)
    fake = FakeHoncho(
        [
            FakeConclusion("matching", "User prefers Europe/Moscow."),
            FakeConclusion("extra", "Only Honcho knows this."),
        ],
        wait=True,
    )
    log_path = tmp_path / "memory" / "comparison.jsonl"
    shadow = ShadowMemoryBackend(
        base,
        honcho_client=fake,  # type: ignore[arg-type]
        comparison_log_path=log_path,
    )

    expected = await base.search("Europe", 5)
    actual = await asyncio.wait_for(shadow.search("Europe", 5), timeout=0.2)

    assert actual == expected
    await asyncio.wait_for(fake.started.wait(), timeout=0.2)
    assert not log_path.exists()
    fake.release.set()
    await shadow.await_pending()

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record.keys() == {
        "ts",
        "query",
        "catalog_hits",
        "honcho_hits",
        "catalog_latency_ms",
        "honcho_latency_ms",
        "overlap",
        "catalog_only",
        "honcho_only",
        "rank_corr",
    }
    assert record["query"] == "Europe"
    assert record["catalog_hits"] == [{"name": "timezone.md", "rank": 1}]
    assert record["honcho_hits"] == [
        {"id": "matching", "snippet": "User prefers Europe/Moscow."},
        {"id": "extra", "snippet": "Only Honcho knows this."},
    ]
    assert record["overlap"] == 1
    assert record["catalog_only"] == 0
    assert record["honcho_only"] == 1
    assert record["rank_corr"] == 1.0
    assert record["catalog_latency_ms"] >= 0
    assert record["honcho_latency_ms"] >= 0
    assert fake.queries == [("Europe", "ohmo-curated", "owner", 5)]


async def test_shadow_swallows_honcho_errors_and_never_changes_result(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
):
    base = _catalog_backend(tmp_path)
    fake = FakeHoncho(fail=True)
    log_path = tmp_path / "failed.jsonl"
    shadow = ShadowMemoryBackend(
        base,
        honcho_client=fake,  # type: ignore[arg-type]
        comparison_log_path=log_path,
    )

    expected = await base.search("Vim", 3)
    actual = await shadow.search("Vim", 3)
    await shadow.await_pending()

    assert actual == expected
    assert not log_path.exists()
    assert "shadow recall comparison failed" in caplog.text


async def test_shadow_turn_is_read_only_and_never_ingests_or_dreams(tmp_path: Path):
    base = _catalog_backend(tmp_path)
    fake = FakeHoncho()
    shadow = ShadowMemoryBackend(
        base,
        honcho_client=fake,  # type: ignore[arg-type]
        comparison_log_path=tmp_path / "comparison.jsonl",
    )

    await shadow.render_prompt()
    await shadow.search("timezone", 2)
    await shadow.append_turn("user", "Please remember this conversation.")
    await shadow.await_pending()

    assert fake.queries == [("timezone", "ohmo-curated", "owner", 2)]
    assert fake.create_messages_calls == 0
    assert fake.dream_calls == 0


async def test_shadow_append_exchange_creates_ordered_batch_with_forced_roles(tmp_path: Path):
    base = _catalog_backend(tmp_path)
    fake = FakeHonchoExchange()
    shadow = ShadowMemoryBackend(
        base,
        honcho_client=fake,  # type: ignore[arg-type]
        observed="friend",
        assistant_peer="ohmo-friend",
        conversation_learning=True,
        comparison_log_path=tmp_path / "exchange.jsonl",
    )

    await shadow.append_exchange(
        "User says hi.",
        "Assistant answers.",
        user_metadata={"role": "not-user", "logical_turn_id": "turn-1"},
        assistant_metadata={"role": "not-asst", "logical_turn_id": "turn-1", "client_op_id": "op-assistant"},
    )
    await shadow.await_pending()

    assert fake.create_messages_calls == 1
    assert fake.exchange_calls
    session, messages = fake.exchange_calls[0]
    assert session == "ohmo"
    assert len(messages) == 2
    assert messages[0]["content"] == "User says hi."
    assert messages[0]["peer_id"] == "friend"
    assert messages[0]["metadata"]["role"] == "user"
    assert messages[0]["metadata"]["logical_turn_id"] == "turn-1"
    assert messages[1]["content"] == "Assistant answers."
    assert messages[1]["peer_id"] == "ohmo-friend"
    assert messages[1]["metadata"]["role"] == "assistant"
    assert messages[1]["metadata"]["client_op_id"] == "op-assistant"


async def test_shadow_append_exchange_delegates_when_learning_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    base = _catalog_backend(tmp_path)
    calls: list[tuple[str, str]] = []

    async def append_turn(role: str, text: str) -> None:
        calls.append((role, text))

    monkeypatch.setattr(base, "append_turn", append_turn)
    fake = FakeHonchoExchange()
    shadow = ShadowMemoryBackend(
        base,
        honcho_client=fake,  # type: ignore[arg-type]
        conversation_learning=False,
        comparison_log_path=tmp_path / "exchange-disabled.jsonl",
    )

    await shadow.append_exchange(
        "User turns off.",
        "Assistant turns off.",
        user_metadata={"role": "override"},
        assistant_metadata={"role": "override"},
    )

    assert calls == [("user", "User turns off."), ("assistant", "Assistant turns off.")]
    assert fake.create_messages_calls == 0


async def test_shadow_append_exchange_failure_isolated_from_runtime(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    base = _catalog_backend(tmp_path)
    fake = FakeHonchoExchange(fail_exchange=True)
    shadow = ShadowMemoryBackend(
        base,
        honcho_client=fake,  # type: ignore[arg-type]
        conversation_learning=True,
        comparison_log_path=tmp_path / "exchange-failed.jsonl",
    )

    await shadow.append_exchange(
        "User fails.",
        "Assistant fails.",
        user_metadata={},
        assistant_metadata={},
    )
    await shadow.await_pending()

    assert fake.create_messages_calls == 1
    assert "ohmo conversation learning exchange ingestion failed" in caplog.text


async def test_shadow_append_turn_remains_single_message_ingest_batch(tmp_path: Path):
    base = _catalog_backend(tmp_path)
    fake = FakeHonchoExchange()
    shadow = ShadowMemoryBackend(
        base,
        honcho_client=fake,  # type: ignore[arg-type]
        comparison_log_path=tmp_path / "turn-regression.jsonl",
        conversation_learning=True,
    )

    await shadow.append_turn("assistant", "Assistant follow-up.")
    await shadow.await_pending()

    assert fake.create_messages_calls == 1
    assert fake.exchange_calls
    session, messages = fake.exchange_calls[0]
    assert session == "ohmo"
    assert len(messages) == 1
    assert messages[0]["peer_id"] == "ohmo"
    assert messages[0]["metadata"]["role"] == "assistant"


@pytest.mark.parametrize("is_owner", [True, False])
async def test_shadow_without_enabled_owner_client_is_pure_pass_through(
    is_owner: bool,
    tmp_path: Path,
):
    base = _catalog_backend(tmp_path)
    fake = FakeHoncho()
    log_path = tmp_path / "disabled.jsonl"
    shadow = ShadowMemoryBackend(
        base,
        honcho_client=fake if not is_owner else None,  # type: ignore[arg-type]
        comparison_log_path=log_path,
        is_owner=is_owner,
    )

    expected = await base.search("prefers", 2)
    actual = await shadow.search("prefers", 2)
    await shadow.await_pending()

    assert actual == expected
    assert fake.queries == []
    assert not log_path.exists()


def test_shadow_factory_default_and_unbuilt_honcho_kinds(tmp_path: Path):
    shadow = make_memory_backend(
        GatewayConfig(memory_backend="shadow"),
        tmp_path / "shadow",
    )
    default = make_memory_backend(GatewayConfig(), tmp_path / "default")

    assert GatewayConfig().memory_backend == "file"
    assert isinstance(shadow, ShadowMemoryBackend)
    assert isinstance(default, FileMemoryBackend)
    with pytest.raises(
        NotImplementedError,
        match="honcho memory backend not built in Phase 0",
    ):
        make_memory_backend(GatewayConfig(memory_backend="honcho"), tmp_path / "honcho")


def test_shadow_factory_builds_honcho_only_for_configured_tenant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from ohmo.memory_service import honcho_client as honcho_client_module

    created: list[tuple[str, str, str]] = []

    class StubHonchoClient:
        def __init__(self, base_url: str, jwt: str, workspace: str) -> None:
            created.append((base_url, jwt, workspace))

    monkeypatch.setattr(honcho_client_module, "HonchoClient", StubHonchoClient)
    owner_backend = make_memory_backend(
        GatewayConfig(
            memory_backend="shadow",
            owner_principals=("owner-id",),
            honcho_base_url="https://honcho.test",
            honcho_api_key="secret",
            honcho_workspace="workspace-one",
        ),
        tmp_path / "owner",
    )
    non_owner_backend = make_memory_backend(
        GatewayConfig(
            memory_backend="shadow",
            honcho_base_url="https://honcho.test",
            honcho_api_key="secret",
            honcho_workspace="workspace-one",
        ),
        tmp_path / "non-owner",
    )
    partial_backend = make_memory_backend(
        GatewayConfig(
            memory_backend="shadow",
            owner_principals=("owner-id",),
            honcho_base_url="https://honcho.test",
            honcho_api_key="secret",
        ),
        tmp_path / "partial",
    )
    family_backend = make_memory_backend(
        GatewayConfig(
            memory_backend="shadow",
            owner_principals=("owner-id",),
            honcho_base_url="https://honcho.test",
            tenant_honcho={
                "marina": {
                    "workspace": "workspace-marina",
                    "api_key": "marina-secret",
                    "observed_peer": "marina-person",
                }
            },
        ),
        tmp_path / "family",
        tenant_id="marina",
    )

    assert isinstance(owner_backend, ShadowMemoryBackend)
    assert isinstance(non_owner_backend, ShadowMemoryBackend)
    assert isinstance(partial_backend, ShadowMemoryBackend)
    assert isinstance(family_backend, ShadowMemoryBackend)
    assert created == [
        ("https://honcho.test", "secret", "workspace-one"),
        ("https://honcho.test", "marina-secret", "workspace-marina"),
    ]
    assert isinstance(owner_backend._honcho_client, StubHonchoClient)
    assert non_owner_backend._honcho_client is None
    assert partial_backend._honcho_client is None
    assert isinstance(family_backend._honcho_client, StubHonchoClient)
    assert family_backend._base._tenant_id == "marina"
    assert family_backend._observed == "marina-person"


async def test_non_owner_disables_configured_shadow_client(tmp_path: Path):
    fake = FakeHoncho()
    base = _catalog_backend(tmp_path)
    shadow = ShadowMemoryBackend(
        base,
        honcho_client=fake,  # type: ignore[arg-type]
        is_owner=False,
        comparison_log_path=tmp_path / "non-owner.jsonl",
    )

    assert await shadow.search("Europe", 1) == await base.search("Europe", 1)
    await shadow.await_pending()

    assert fake.queries == []
    assert not (tmp_path / "non-owner.jsonl").exists()


async def test_shadow_report_aggregates_jsonl_records(tmp_path: Path):
    base = _catalog_backend(tmp_path)
    fake = FakeHoncho(
        [
            FakeConclusion("matching", "User prefers Europe/Moscow."),
            FakeConclusion("extra", "Only Honcho knows this."),
        ]
    )
    log_path = tmp_path / "comparison.jsonl"
    shadow = ShadowMemoryBackend(
        base,
        honcho_client=fake,  # type: ignore[arg-type]
        comparison_log_path=log_path,
    )

    await shadow.search("Europe", 5)
    await shadow.await_pending()

    report = shadow_report(log_path)
    assert "Records: 1" in report
    assert "Mean overlap: 50.0%" in report
    assert "Mean completeness: 100.0%" in report
    assert "Catalog-only rate: 0.0%" in report
    assert "Honcho-only rate: 50.0%" in report
    assert "Mean rank correlation: 1.000" in report

"""Tests for opt-in, gate-confined Honcho derived recall."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from ohmo.gateway.models import GatewayConfig
from ohmo.gateway.turn_context import TurnContext
from ohmo.memory_backend import CatalogMemoryBackend, ShadowMemoryBackend
from ohmo.memory_catalog import MemoryCatalog
from ohmo.prompt_seam import compose_runtime_prompt, prepare_turn


@dataclass(frozen=True)
class FakeConclusion:
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
        self.queries: list[tuple[str, str, str, int]] = []
        self.release = asyncio.Event()

    async def query_conclusions(
        self,
        query: str,
        *,
        observer: str,
        observed: str,
        top_k: int,
    ) -> list[FakeConclusion]:
        self.queries.append((query, observer, observed, top_k))
        if self.wait:
            await self.release.wait()
        if self.fail:
            raise OSError("fake Honcho is unavailable")
        return self.conclusions


def _owner_private_context() -> TurnContext:
    return TurnContext(
        principal="owner",
        is_owner=True,
        is_private=True,
        channel="telegram",
        chat_id="owner",
        session_id="session-owner",
    )


def _backend(
    workspace: Path,
    fake: FakeHoncho,
) -> tuple[CatalogMemoryBackend, ShadowMemoryBackend]:
    catalog = MemoryCatalog(workspace)
    assert catalog.add("Timezone", "User prefers Europe/Moscow.").ok
    base = CatalogMemoryBackend(catalog, workspace)
    shadow = ShadowMemoryBackend(
        base,
        honcho_client=fake,  # type: ignore[arg-type]
        comparison_log_path=workspace / "comparison.jsonl",
    )
    return base, shadow


@pytest.mark.parametrize(
    ("visible_recall", "tools_confined"),
    ((False, True), (True, False)),
)
async def test_visible_recall_is_dormant_unless_opted_in_and_gate_allowed(
    visible_recall: bool,
    tools_confined: bool,
    tmp_path: Path,
) -> None:
    config = GatewayConfig(
        visible_recall=visible_recall,
        tools_confined=tools_confined,
    )
    fake = FakeHoncho([FakeConclusion("Honcho-only secret preference.")])
    base, shadow = _backend(tmp_path, fake)
    catalog_only = await base.render_prompt()

    snapshot = await prepare_turn(
        shadow,
        turn_ctx=_owner_private_context(),
        tools_confined=config.tools_confined,
        principal_isolated=True,
        visible_recall=config.visible_recall,
        latest_user_prompt="What do I prefer?",
    )

    assert GatewayConfig().visible_recall is False
    assert str(snapshot) == catalog_only
    assert compose_runtime_prompt("memory-free base", snapshot) == compose_runtime_prompt(
        "memory-free base", catalog_only
    )
    assert "Honcho-only secret preference." not in snapshot
    assert fake.queries == []


async def test_visible_recall_surfaces_labelled_derived_hits_under_composite_budget(
    tmp_path: Path,
) -> None:
    fake = FakeHoncho(
        [
            FakeConclusion("User likes concise status updates."),
            FakeConclusion("A" * 2_000),
        ]
    )
    _, shadow = _backend(tmp_path, fake)
    budget = 1_200

    snapshot = await prepare_turn(
        shadow,
        budget=budget,
        turn_ctx=_owner_private_context(),
        tools_confined=True,
        principal_isolated=True,
        visible_recall=True,
        latest_user_prompt="How should you update me?",
    )

    assert snapshot.gate_decision.allowed is True
    assert "## Recalled (honcho, derived — may be imperfect)" in snapshot
    assert "catalog memory above is curated" in snapshot
    assert "- User likes concise status updates." in snapshot
    assert len(snapshot) <= budget
    assert fake.queries == [("How should you update me?", "ohmo", "owner", 10)]


@pytest.mark.parametrize("failure", ["timeout", "error"])
async def test_visible_recall_failure_falls_open_to_catalog_only(
    failure: str,
    tmp_path: Path,
) -> None:
    fake = FakeHoncho(wait=failure == "timeout", fail=failure == "error")
    base, shadow = _backend(tmp_path, fake)
    catalog_only = await base.render_prompt()

    snapshot = await prepare_turn(
        shadow,
        turn_ctx=_owner_private_context(),
        tools_confined=True,
        principal_isolated=True,
        visible_recall=True,
        latest_user_prompt="Recall something useful.",
        derived_recall_timeout=0.01,
    )

    assert str(snapshot) == catalog_only
    assert "honcho, derived" not in snapshot
    assert fake.queries == [("Recall something useful.", "ohmo", "owner", 10)]


@pytest.mark.parametrize(
    ("context_changes", "principal_isolated", "tools_confined"),
    (
        ({"is_owner": False}, True, True),
        ({"is_private": False}, True, True),
        ({}, False, True),
        ({}, True, False),
    ),
)
async def test_visible_recall_gate_denies_each_single_false_conjunct(
    context_changes: dict[str, bool],
    principal_isolated: bool,
    tools_confined: bool,
    tmp_path: Path,
) -> None:
    fake = FakeHoncho([FakeConclusion("This must stay out of the prompt.")])
    base, shadow = _backend(tmp_path, fake)
    catalog_only = await base.render_prompt()

    snapshot = await prepare_turn(
        shadow,
        turn_ctx=replace(_owner_private_context(), **context_changes),
        tools_confined=tools_confined,
        principal_isolated=principal_isolated,
        visible_recall=True,
        latest_user_prompt="What do you recall?",
    )

    assert snapshot.gate_decision.allowed is False
    assert str(snapshot) == catalog_only
    assert "This must stay out of the prompt." not in snapshot
    assert fake.queries == []

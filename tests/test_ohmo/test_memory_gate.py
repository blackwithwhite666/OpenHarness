"""Tests for the fail-closed memory confidentiality gate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ohmo.gateway.memory_gate import (
    evaluate_memory_gate,
    principal_isolated_session,
)
from ohmo.gateway.models import GatewayConfig
from ohmo.gateway.turn_context import TurnContext
from ohmo.memory_backend import FileMemoryBackend
from ohmo.memory_store import MemoryStore
from ohmo.prompt_seam import compose_runtime_prompt, prepare_turn
from ohmo.prompts import build_ohmo_system_prompt


def _owner_private_context() -> TurnContext:
    return TurnContext(
        principal="owner",
        is_owner=True,
        is_private=True,
        channel="telegram",
        chat_id="owner",
        session_id="session-owner",
    )


@pytest.mark.parametrize(
    ("context_changes", "principal_isolated", "reason"),
    (
        ({"is_owner": False}, True, "is_canonical_owner"),
        ({"is_private": False}, True, "is_trusted_private_chat"),
        ({}, False, "principal_isolated_session"),
    ),
)
def test_memory_gate_denies_each_single_false_conjunct(
    context_changes: dict[str, bool],
    principal_isolated: bool,
    reason: str,
) -> None:
    turn_ctx = replace(_owner_private_context(), **context_changes)

    decision = evaluate_memory_gate(
        turn_ctx,
        principal_isolated=principal_isolated,
    )

    assert decision.allowed is False
    assert decision.reasons == (reason,)


def test_memory_gate_allows_only_when_all_three_conjuncts_are_true() -> None:
    decision = evaluate_memory_gate(
        _owner_private_context(),
        principal_isolated=True,
    )

    assert decision.allowed is True
    assert decision.reasons == ()


def test_memory_gate_treats_missing_inputs_as_false() -> None:
    decision = evaluate_memory_gate(
        None,
        principal_isolated=None,
    )

    assert decision.allowed is False
    assert decision.reasons == (
        "is_canonical_owner",
        "is_trusted_private_chat",
        "principal_isolated_session",
    )


async def test_owner_private_isolated_turn_opens_gate_without_visible_recall(
    tmp_path: Path,
) -> None:
    config = GatewayConfig()
    backend = FileMemoryBackend(MemoryStore(tmp_path / "workspace"))

    snapshot = await prepare_turn(
        backend,
        turn_ctx=_owner_private_context(),
        principal_isolated=True,
        visible_recall=config.visible_recall,
    )

    assert config.visible_recall is False
    assert snapshot.gate_decision.allowed is True
    assert snapshot.gate_decision.reasons == ()


def test_principal_isolated_session_requires_exact_known_owner() -> None:
    turn_ctx = _owner_private_context()

    assert principal_isolated_session(turn_ctx, "owner") is True
    assert principal_isolated_session(turn_ctx, "shared") is False
    assert principal_isolated_session(turn_ctx, None) is False
    assert principal_isolated_session(None, "owner") is False


async def test_prepare_turn_gate_metadata_does_not_change_memory_injection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OHMO_MEMORY_AUTOINDEX", "0")
    workspace = tmp_path / ".ohmo-home"
    store = MemoryStore(workspace)
    assert store.add("Timezone", "User prefers UTC.").ok
    backend = FileMemoryBackend(store)
    memory_free = build_ohmo_system_prompt(
        tmp_path,
        workspace=workspace,
        include_ohmo_memory=False,
    )
    legacy = build_ohmo_system_prompt(
        tmp_path,
        workspace=workspace,
        include_ohmo_memory=True,
    )

    snapshot = await prepare_turn(
        backend,
        turn_ctx=_owner_private_context(),
        principal_isolated=True,
    )

    assert snapshot.gate_decision.allowed is True
    assert compose_runtime_prompt(memory_free, snapshot) == legacy

"""Tests for the fail-closed memory confidentiality gate."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openharness.engine.messages import ConversationMessage
from openharness.tools.base import ToolRegistry

from ohmo.gateway.memory_gate import (
    evaluate_memory_gate,
    memory_engaged,
    principal_isolated_session,
)
from ohmo.gateway.models import GatewayConfig
from ohmo.gateway.runtime import OhmoSessionRuntimePool
from ohmo.gateway.turn_context import TurnContext
from ohmo.memory_backend import FileMemoryBackend
from ohmo.memory_judge import JudgeOutcome
from ohmo.memory_store import MemoryStore
from ohmo.memory_tool import OhmoMemoryTool
from ohmo.prompt_seam import compose_runtime_prompt, prepare_turn
from ohmo.prompts import build_ohmo_system_prompt
from ohmo.workspace import initialize_workspace


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


def test_memory_engaged_preserves_legacy_and_requires_gate_when_owners_exist() -> None:
    denied = evaluate_memory_gate(None, principal_isolated=False)
    allowed = evaluate_memory_gate(_owner_private_context(), principal_isolated=True)

    assert memory_engaged((), denied) is True
    assert memory_engaged(("owner",), denied) is False
    assert memory_engaged(("owner",), allowed) is True


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


@pytest.mark.parametrize(
    ("case", "owner_principals", "turn_ctx", "principal_isolated", "expected"),
    (
        (
            "legacy",
            (),
            replace(_owner_private_context(), is_owner=False),
            False,
            True,
        ),
        ("owner-allowed", ("owner",), _owner_private_context(), True, True),
        (
            "non-owner-denied",
            ("owner",),
            replace(_owner_private_context(), is_owner=False),
            True,
            False,
        ),
    ),
)
async def test_prepare_turn_gates_catalog_only_when_owners_are_configured(
    case: str,
    owner_principals: tuple[str, ...],
    turn_ctx: TurnContext,
    principal_isolated: bool,
    expected: bool,
) -> None:
    backend = SimpleNamespace(
        render_prompt=AsyncMock(
            return_value=(
                "# ohmo Memory\n"
                "- Personal memory directory: /owner/workspace/memory\n\n"
                "Owner curated secret"
            )
        )
    )

    snapshot = await prepare_turn(
        backend,
        turn_ctx=turn_ctx,
        principal_isolated=principal_isolated,
        owner_principals=owner_principals,
    )

    assert snapshot.memory_engaged is expected, case
    assert bool(snapshot) is expected, case
    assert backend.render_prompt.await_count == int(expected), case


def _surface_bundle(pool: OhmoSessionRuntimePool) -> SimpleNamespace:
    registry = ToolRegistry()
    registry.register(OhmoMemoryTool(pool._prompt_memory_backend))
    return SimpleNamespace(
        tool_registry=registry,
        autodream_context=pool._autodream_context(),
        engine=SimpleNamespace(
            tool_metadata={"autodream_context": pool._autodream_context()},
            messages=[ConversationMessage.from_user_text("a durable owner fact")],
            api_client=object(),
        ),
        current_settings=lambda: SimpleNamespace(model="model", timeout=30.0),
    )


@pytest.mark.parametrize(
    ("context_changes", "session_owner"),
    (
        ({"is_owner": False}, "owner"),
        ({"is_private": False}, "owner"),
        ({}, "different-principal"),
    ),
)
async def test_configured_owner_denial_removes_every_authoritative_surface(
    context_changes: dict[str, bool],
    session_owner: str,
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OHMO_MEMORY_AUTOINDEX", "0")
    monkeypatch.setenv("OHMO_MEMORY_JUDGE", "1")
    monkeypatch.setenv("OHMO_MEMORY_JUDGE_INTERVAL", "1")
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    store = MemoryStore(workspace)
    assert store.add("Owner only", "Owner curated secret").ok
    pool = OhmoSessionRuntimePool(
        cwd=tmp_path,
        workspace=workspace,
        provider_profile="codex",
    )
    pool._gateway_config = pool._gateway_config.model_copy(
        update={"owner_principals": ("owner",)}
    )
    turn_ctx = replace(_owner_private_context(), **context_changes)
    pool._session_owner_principals[turn_ctx.session_id] = session_owner
    original_render = pool._prompt_memory_backend.render_prompt
    pool._prompt_memory_backend.render_prompt = AsyncMock(wraps=original_render)

    prompt = await pool._runtime_system_prompt(
        SimpleNamespace(cwd=str(tmp_path)),
        "tell me about the owner",
        turn_ctx=turn_ctx,
    )
    bundle = _surface_bundle(pool)
    engaged = pool._configure_turn_memory_surfaces(bundle, turn_ctx)
    pool._maybe_schedule_memory_judge(bundle, "denied", turn_ctx=turn_ctx)

    assert engaged is False
    assert pool._prompt_memory_backend.render_prompt.await_count == 0
    assert "Owner curated secret" not in prompt
    assert "# ohmo Memory" not in prompt
    assert "# ohmo Workspace" not in prompt
    assert "Personal workspace root:" not in prompt
    assert "Personal memory directory:" not in prompt
    assert bundle.tool_registry.get("memory") is None
    assert bundle.autodream_context is None
    assert "autodream_context" not in bundle.engine.tool_metadata
    assert pool._judge_turn_counts == {}
    assert pool._judge_tasks == {}


@pytest.mark.parametrize(
    ("case", "owner_principals", "turn_ctx"),
    (
        (
            "legacy",
            (),
            replace(_owner_private_context(), is_owner=False, is_private=False),
        ),
        ("owner-allowed", ("owner",), _owner_private_context()),
    ),
)
async def test_legacy_and_configured_owner_keep_all_memory_surfaces(
    case: str,
    owner_principals: tuple[str, ...],
    turn_ctx: TurnContext,
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OHMO_MEMORY_AUTOINDEX", "0")
    monkeypatch.setenv("OHMO_MEMORY_JUDGE", "1")
    monkeypatch.setenv("OHMO_MEMORY_JUDGE_INTERVAL", "1")
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    store = MemoryStore(workspace)
    assert store.add("Owner only", "Owner curated secret").ok
    pool = OhmoSessionRuntimePool(
        cwd=tmp_path,
        workspace=workspace,
        provider_profile="codex",
    )
    pool._gateway_config = pool._gateway_config.model_copy(
        update={"owner_principals": owner_principals}
    )
    pool._session_owner_principals[turn_ctx.session_id] = turn_ctx.principal
    judge_calls: list[dict] = []

    async def fake_judge(**kwargs):
        judge_calls.append(kwargs)
        return JudgeOutcome()

    monkeypatch.setattr("ohmo.gateway.runtime.run_memory_judge", fake_judge)

    prompt = await pool._runtime_system_prompt(
        SimpleNamespace(cwd=str(tmp_path)),
        "what do you remember?",
        turn_ctx=turn_ctx,
    )
    bundle = _surface_bundle(pool)
    engaged = pool._configure_turn_memory_surfaces(bundle, turn_ctx)
    pool._maybe_schedule_memory_judge(bundle, case, turn_ctx=turn_ctx)
    task = pool._judge_tasks[case]
    await task
    await asyncio.sleep(0)

    assert engaged is True, case
    assert "Owner curated secret" in prompt, case
    assert "# ohmo Memory" in prompt, case
    assert "# ohmo Workspace" in prompt, case
    assert bundle.tool_registry.get("memory") is not None, case
    assert bundle.autodream_context is not None, case
    assert "autodream_context" in bundle.engine.tool_metadata, case
    assert len(judge_calls) == 1, case


def test_denied_composition_strips_precomposed_memory_and_workspace(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OHMO_MEMORY_AUTOINDEX", "0")
    workspace = tmp_path / ".ohmo-home"
    store = MemoryStore(workspace)
    assert store.add("Owner only", "Owner curated secret").ok
    legacy = build_ohmo_system_prompt(
        tmp_path,
        workspace=workspace,
        include_ohmo_memory=True,
    )

    denied = compose_runtime_prompt(legacy, "", memory_engaged=False)

    assert "Owner curated secret" not in denied
    assert "# ohmo Memory" not in denied
    assert "# ohmo Workspace" not in denied
    assert "Personal workspace root:" not in denied

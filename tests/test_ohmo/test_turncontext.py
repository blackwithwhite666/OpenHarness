"""Tests for gateway-authenticated per-turn identity plumbing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from openharness.channels.bus.events import InboundMessage

from ohmo.gateway.config import load_gateway_config
from ohmo.gateway.models import GatewayConfig
from ohmo.gateway.runtime import OhmoSessionRuntimePool
from ohmo.gateway.turn_context import (
    TurnContext,
    build_turn_context,
    canonical_principal,
    is_private_message,
)
from ohmo.workspace import initialize_workspace


def _message(*, sender_id: str = "12345|alice", metadata: dict | None = None) -> InboundMessage:
    return InboundMessage(
        channel="telegram",
        sender_id=sender_id,
        chat_id="12345",
        content="hello",
        metadata=metadata or {},
    )


def test_telegram_canonical_principal_ignores_mutable_username() -> None:
    original = canonical_principal("telegram", "12345|alice")
    renamed = canonical_principal("telegram", "12345|alice_renamed")

    assert original == renamed == "12345"


def test_owner_status_uses_only_configured_canonical_principal() -> None:
    config = GatewayConfig(owner_principals=("12345",))

    owner = build_turn_context(
        _message(sender_id="12345|alice"),
        session_id="session-owner",
        owner_principals=config.owner_principals,
    )
    non_owner = build_turn_context(
        _message(sender_id="67890|bob"),
        session_id="session-other",
        owner_principals=config.owner_principals,
    )

    assert owner.is_owner is True
    assert non_owner.is_owner is False


def test_load_gateway_config_parses_owner_principals(tmp_path: Path) -> None:
    workspace = tmp_path / ".ohmo-home"
    workspace.mkdir()
    (workspace / "gateway.json").write_text(
        json.dumps({"owner_principals": ["12345", "ou_owner"]}) + "\n",
        encoding="utf-8",
    )

    assert load_gateway_config(workspace).owner_principals == ("12345", "ou_owner")


def test_private_status_uses_destination_scope_and_tracks_forward_provenance() -> None:
    group = _message(metadata={"is_group": True})
    forwarded = _message(metadata={"is_group": False, "is_forwarded": True})
    forwarded_group = _message(metadata={"is_group": True, "is_forwarded": True})
    unknown = _message(metadata={})
    private = _message(metadata={"is_group": False})

    assert is_private_message(group) is False
    assert is_private_message(forwarded) is True
    assert is_private_message(forwarded_group) is False
    assert is_private_message(unknown) is False
    assert is_private_message(private) is True

    private_forward_context = build_turn_context(forwarded, session_id="forward-private")
    group_forward_context = build_turn_context(forwarded_group, session_id="forward-group")
    assert private_forward_context.is_private is True
    assert private_forward_context.is_forwarded is True
    assert group_forward_context.is_private is False
    assert group_forward_context.is_forwarded is True
    assert build_turn_context(private, session_id="plain").is_forwarded is False


async def test_runtime_prompt_threads_turn_context_into_prepare_turn(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    pool = OhmoSessionRuntimePool(
        cwd=tmp_path,
        workspace=workspace,
        provider_profile="codex",
    )
    turn_ctx = TurnContext(
        principal="12345",
        is_owner=True,
        is_private=True,
        channel="telegram",
        chat_id="12345",
        session_id="session-owner",
    )
    captured: dict[str, object] = {}

    async def fake_prepare_turn(
        backend,
        *,
        budget=None,
        turn_ctx=None,
        principal_isolated=None,
        visible_recall=False,
        latest_user_prompt=None,
        derived_recall_timeout=0.5,
        owner_principals=(),
        memory_engaged_override=None,
        derived_backend=None,
        derived_recall_allowed_override=None,
    ):
        captured["backend"] = backend
        captured["budget"] = budget
        captured["turn_ctx"] = turn_ctx
        captured["principal_isolated"] = principal_isolated
        captured["visible_recall"] = visible_recall
        captured["latest_user_prompt"] = latest_user_prompt
        captured["owner_principals"] = owner_principals
        captured["memory_engaged_override"] = memory_engaged_override
        captured["derived_backend"] = derived_backend
        captured["derived_recall_allowed_override"] = derived_recall_allowed_override
        return ""

    monkeypatch.setattr("ohmo.gateway.runtime.prepare_turn", fake_prepare_turn)

    await pool._runtime_system_prompt(
        SimpleNamespace(cwd=str(tmp_path)),
        "hello",
        turn_ctx=turn_ctx,
    )

    assert captured["turn_ctx"] is turn_ctx
    assert captured["principal_isolated"] is False
    assert captured["visible_recall"] is False
    assert captured["latest_user_prompt"] == "hello"
    assert captured["owner_principals"] == ()
    assert captured["memory_engaged_override"] is True
    assert captured["derived_backend"] is None
    assert captured["derived_recall_allowed_override"] is False

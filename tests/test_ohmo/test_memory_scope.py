"""Tests for fail-closed memory audience authorization."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from pydantic import ValidationError

from ohmo.gateway.memory_gate import MemoryScope, resolve_memory_scope
from ohmo.gateway.models import GatewayConfig
from ohmo.gateway.turn_context import TurnContext


def _turn_context(principal: str = "100") -> TurnContext:
    return TurnContext(
        principal=principal,
        is_owner=principal == "100",
        is_private=True,
        channel="telegram",
        chat_id=principal,
        session_id=f"session-{principal}",
    )


def _family_config() -> GatewayConfig:
    return GatewayConfig(
        owner_principals=("100",),
        family_principals={"200": "marina", "300": "zoya"},
        shared_tenants=("family-shared",),
        enabled_memory_tenants=("owner", "marina"),
    )


def test_owner_private_isolated_scope_includes_shared_tenants() -> None:
    scope = resolve_memory_scope(
        _family_config(),
        _turn_context(),
        principal_isolated=True,
    )

    assert scope == MemoryScope("owner", ("family-shared",))


def test_enabled_family_private_isolated_scope_includes_shared_tenants() -> None:
    scope = resolve_memory_scope(
        _family_config(),
        _turn_context("200"),
        principal_isolated=True,
    )

    assert scope == MemoryScope("marina", ("family-shared",))


def test_family_tenant_without_consent_has_no_memory_scope() -> None:
    scope = resolve_memory_scope(
        _family_config(),
        _turn_context("300"),
        principal_isolated=True,
    )

    assert scope is None


@pytest.mark.parametrize(
    ("turn_ctx", "principal_isolated"),
    (
        (_turn_context("999"), True),
        (replace(_turn_context("200"), is_private=False), True),
        (_turn_context("200"), False),
    ),
)
def test_unrecognized_or_untrusted_turn_has_no_memory_scope(
    turn_ctx: TurnContext,
    principal_isolated: bool,
) -> None:
    assert (
        resolve_memory_scope(
            _family_config(),
            turn_ctx,
            principal_isolated=principal_isolated,
        )
        is None
    )


def test_username_spoof_does_not_change_canonical_numeric_identity() -> None:
    scope = resolve_memory_scope(
        _family_config(),
        _turn_context("200|different_handle"),
        principal_isolated=True,
    )

    assert scope == MemoryScope("marina", ("family-shared",))


@pytest.mark.parametrize(
    "config_values",
    (
        {
            "owner_principals": ("200",),
            "family_principals": {"200": "marina"},
        },
        {"family_principals": {"not-numeric": "marina"}},
        {"family_principals": {"200": "Marina"}},
        {"family_principals": {"200": "owner"}},
        {"shared_tenants": ("owner",)},
    ),
)
def test_invalid_memory_tenant_config_is_rejected(
    config_values: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        GatewayConfig(**config_values)


def test_legacy_owner_keeps_private_isolated_scope() -> None:
    cfg = GatewayConfig(owner_principals=("100",))

    assert resolve_memory_scope(
        cfg,
        _turn_context(),
        principal_isolated=True,
    ) == MemoryScope("owner", ())
    assert (
        resolve_memory_scope(
            cfg,
            _turn_context("999"),
            principal_isolated=True,
        )
        is None
    )

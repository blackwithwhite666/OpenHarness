"""Focused coverage for gateway-owned speaker knowledge references."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ohmo.gateway.models import GatewayConfig
from ohmo.gateway.profile_context import render_profile_context
from ohmo.gateway.runtime import OhmoSessionRuntimePool
from ohmo.gateway.turn_context import TurnContext, build_turn_context
from ohmo.workspace import initialize_workspace
from openharness.config.settings import Settings
from openharness.channels.bus.events import InboundMessage


def _readme(root: Path, slug: str) -> Path:
    path = root / "projects" / slug / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Project\n", encoding="utf-8")
    return path


def _context(principal: str, *, channel: str = "telegram") -> TurnContext:
    return TurnContext(
        principal=principal,
        is_owner=False,
        is_private=True,
        channel=channel,
        chat_id=principal,
        session_id=f"session-{principal}",
    )


def _config(root: Path) -> GatewayConfig:
    return GatewayConfig(
        knowledge_base_root=root,
        principal_knowledge_projects={
            "200": ("marina-health", "marina-nutrition"),
            "300": ("zoe-school-calendar",),
        },
    )


def test_profile_context_keeps_order_and_isolates_principals(tmp_path: Path) -> None:
    root = tmp_path / "knowledge-base"
    marina_health = _readme(root, "marina-health")
    marina_nutrition = _readme(root, "marina-nutrition")
    zoe = _readme(root, "zoe-school-calendar")
    config = _config(root)

    marina = render_profile_context(config, _context("200"))
    zoe_context = render_profile_context(config, _context("300"))

    assert marina.index(str(marina_health.resolve())) < marina.index(str(marina_nutrition.resolve()))
    assert "marina-health" in marina and "marina-nutrition" in marina
    assert str(zoe.resolve()) not in marina
    assert "zoe-school-calendar" in zoe_context
    assert str(marina_health.resolve()) not in zoe_context


def test_profile_context_omits_untrusted_or_missing_contexts(tmp_path: Path) -> None:
    root = tmp_path / "knowledge-base"
    _readme(root, "marina-health")
    config = GatewayConfig(
        knowledge_base_root=root,
        principal_knowledge_projects={"200": ("marina-health",)},
    )

    assert render_profile_context(config, None) == ""
    assert render_profile_context(config, _context("999")) == ""
    assert render_profile_context(config, _context("200", channel="whatsapp")) == ""


def test_profile_context_uses_canonical_telegram_principal(tmp_path: Path) -> None:
    root = tmp_path / "knowledge-base"
    _readme(root, "marina-health")
    config = GatewayConfig(
        knowledge_base_root=root,
        principal_knowledge_projects={"200": ("marina-health",)},
    )
    message = InboundMessage(
        channel="telegram",
        sender_id="200|spoofed-handle",
        chat_id="200",
        content="hello",
        metadata={"is_group": False},
    )

    assert build_turn_context(message, session_id="session").principal == "200"
    assert "marina-health" in render_profile_context(config, build_turn_context(message, session_id="session"))


def test_profile_context_fails_closed_for_missing_or_escaping_readmes(tmp_path: Path) -> None:
    root = tmp_path / "knowledge-base"
    _readme(root, "valid-project")
    config = GatewayConfig(
        knowledge_base_root=root,
        principal_knowledge_projects={"200": ("valid-project", "missing-project")},
    )
    assert render_profile_context(config, _context("200")) == ""

    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    escaped = root / "projects" / "escaped-project" / "README.md"
    escaped.parent.mkdir(parents=True)
    escaped.symlink_to(outside)
    escaped_config = GatewayConfig(
        knowledge_base_root=root,
        principal_knowledge_projects={"200": ("escaped-project",)},
    )
    assert render_profile_context(escaped_config, _context("200")) == ""
    assert render_profile_context(
        GatewayConfig(
            knowledge_base_root=tmp_path / "missing-root",
            principal_knowledge_projects={"200": ("valid-project",)},
        ),
        _context("200"),
    ) == ""


@pytest.mark.parametrize(
    "values",
    (
        {"principal_knowledge_projects": {"200": ("valid-project",)}},
        {
            "knowledge_base_root": Path("relative"),
            "principal_knowledge_projects": {"200": ("valid-project",)},
        },
        {
            "knowledge_base_root": Path("/tmp/kb"),
            "principal_knowledge_projects": {"0200": ("valid-project",)},
        },
        {
            "knowledge_base_root": Path("/tmp/kb"),
            "principal_knowledge_projects": {"1" * 21: ("valid-project",)},
        },
        {
            "knowledge_base_root": Path("/tmp/kb"),
            "principal_knowledge_projects": {"200": ("Invalid_Slug",)},
        },
        {
            "knowledge_base_root": Path("/tmp/kb"),
            "principal_knowledge_projects": {"200": ("valid-project", "valid-project")},
        },
    ),
)
def test_profile_context_config_rejects_unsafe_values(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        GatewayConfig(**values)


async def test_runtime_recomposes_profile_context_without_duplication(tmp_path: Path) -> None:
    root = tmp_path / "knowledge-base"
    _readme(root, "marina-health")
    workspace = tmp_path / ".ohmo"
    initialize_workspace(workspace)
    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    pool._gateway_config = GatewayConfig(
        knowledge_base_root=root,
        principal_knowledge_projects={"200": ("marina-health",)},
    )
    bundle = SimpleNamespace(
        cwd=str(tmp_path),
        current_settings=lambda: Settings(model="test", system_prompt="custom base"),
    )

    first = await pool._runtime_system_prompt(bundle, "hello", turn_ctx=_context("200"))
    second = await pool._runtime_system_prompt(bundle, "again", turn_ctx=_context("200"))

    assert first.count("# Speaker knowledge-base project references") == 1
    assert second.count("# Speaker knowledge-base project references") == 1

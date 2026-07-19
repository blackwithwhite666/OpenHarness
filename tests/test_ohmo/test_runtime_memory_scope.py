"""Runtime coverage for per-turn catalog audience scopes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from openharness.engine.messages import ConversationMessage
from openharness.tools.base import ToolExecutionContext, ToolRegistry

from ohmo.gateway.memory_gate import MemoryScope
from ohmo.gateway.models import GatewayConfig
from ohmo.gateway.runtime import OhmoSessionRuntimePool
from ohmo.gateway.turn_context import TurnContext
from ohmo.memory_backend import CatalogMemoryBackend, ShadowMemoryBackend
from ohmo.memory_catalog import MemoryCatalog
from ohmo.memory_judge import JudgeOutcome
from ohmo.memory_tool import OhmoMemoryTool, OhmoMemoryToolInput
from ohmo.workspace import initialize_workspace


class _FakeHoncho:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.messages: list[list[dict[str, object]]] = []

    async def query_conclusions(self, query: str, **kwargs: object) -> list[object]:
        del kwargs
        self.queries.append(query)
        return [SimpleNamespace(content="owner-derived fact")]

    async def create_messages(
        self,
        session: str,
        messages: list[dict[str, object]],
    ) -> list[object]:
        del session
        self.messages.append(messages)
        return []


def _context(principal: str, *, owner: bool) -> TurnContext:
    return TurnContext(
        principal=principal,
        is_owner=owner,
        is_private=True,
        channel="telegram",
        chat_id=principal,
        session_id=f"session-{principal}",
    )


def _family_config(**changes: object) -> GatewayConfig:
    values: dict[str, object] = {
        "memory_backend": "file",
        "owner_principals": ("100",),
        "family_principals": {"200": "marina"},
        "enabled_memory_tenants": ("owner", "marina"),
        "shared_tenants": ("family-shared",),
    }
    values.update(changes)
    return GatewayConfig(**values)


def _surface_bundle() -> SimpleNamespace:
    return SimpleNamespace(
        tool_registry=ToolRegistry(),
        autodream_context=None,
        engine=SimpleNamespace(
            tool_metadata={},
            messages=[ConversationMessage.from_user_text("a durable fact")],
            api_client=object(),
        ),
        current_settings=lambda: SimpleNamespace(model="model", timeout=30.0),
    )


def _seed_catalog(workspace: Path) -> MemoryCatalog:
    catalog = MemoryCatalog(workspace)
    catalog.ensure_tenant("marina", "private")
    catalog.ensure_tenant("family-shared", "shared")
    assert catalog.add("owner", "Owner note", "owner-only row").ok
    assert catalog.add("marina", "Marina note", "marina-only row").ok
    assert catalog.add("family-shared", "Family note", "shared-family row").ok
    return catalog


async def _assert_scoped_surfaces(
    *,
    pool: OhmoSessionRuntimePool,
    catalog: MemoryCatalog,
    turn_ctx: TurnContext,
    expected_tenant: str,
    expected_private: str,
    excluded_private: str,
    monkeypatch,
    tmp_path: Path,
) -> None:
    pool._session_owner_principals[turn_ctx.session_id] = turn_ctx.principal
    scope = pool._resolve_turn_memory_scope(turn_ctx)
    assert scope == MemoryScope(expected_tenant, ("family-shared",))

    prompt = await pool._runtime_system_prompt(
        SimpleNamespace(cwd=str(tmp_path)),
        "what do you remember?",
        turn_ctx=turn_ctx,
        memory_scope=scope,
    )
    assert expected_private in prompt
    assert "shared-family row" in prompt
    assert excluded_private not in prompt

    bundle = _surface_bundle()
    assert pool._configure_turn_memory_surfaces(
        bundle,
        turn_ctx,
        memory_scope=scope,
    )
    tool = bundle.tool_registry.get("memory")
    assert isinstance(tool, OhmoMemoryTool)
    assert isinstance(tool._store, CatalogMemoryBackend)
    assert tool._store._tenant_id == expected_tenant
    assert tool._store._shared_tenant_id == "family-shared"

    listed = await tool.execute(
        OhmoMemoryToolInput(action="list"),
        ToolExecutionContext(cwd=tmp_path),
    )
    expected_title = "Owner note" if expected_tenant == "owner" else "Marina note"
    excluded_title = "Marina note" if expected_tenant == "owner" else "Owner note"
    assert expected_title in listed.output
    assert excluded_title not in listed.output
    assert "[shared] Family note" in listed.output

    added = await tool.execute(
        OhmoMemoryToolInput(
            action="add",
            title=f"{expected_tenant} tool write",
            content=f"tool write for {expected_tenant}",
        ),
        ToolExecutionContext(cwd=tmp_path),
    )
    assert added.is_error is False
    assert catalog.get(expected_tenant, f"{expected_tenant}_tool_write") is not None

    judge_calls: list[dict[str, object]] = []

    async def fake_judge(**kwargs: object) -> JudgeOutcome:
        judge_calls.append(kwargs)
        store = kwargs["store"]
        assert getattr(store, "_tenant_id") == expected_tenant
        assert expected_private in {entry.content for entry in store.list()}
        assert excluded_private not in {entry.content for entry in store.list()}
        assert "shared-family row" in {entry.content for entry in store.list()}
        result = store.add(
            f"{expected_tenant} judge write",
            f"judge write for {expected_tenant}",
        )
        assert result.ok
        return JudgeOutcome()

    monkeypatch.setattr("ohmo.gateway.runtime.run_memory_judge", fake_judge)
    pool._maybe_schedule_memory_judge(
        bundle,
        f"judge-{expected_tenant}",
        turn_ctx=turn_ctx,
        memory_scope=scope,
    )
    await pool._judge_tasks[f"judge-{expected_tenant}"]
    await asyncio.sleep(0)
    assert len(judge_calls) == 1
    assert catalog.get(expected_tenant, f"{expected_tenant}_judge_write") is not None
    assert bundle.autodream_context is not None


async def test_owner_and_family_turns_bind_all_catalog_surfaces_to_their_tenant(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OHMO_MEMORY_JUDGE", "1")
    monkeypatch.setenv("OHMO_MEMORY_JUDGE_INTERVAL", "1")
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    catalog = _seed_catalog(workspace)
    pool = OhmoSessionRuntimePool(
        cwd=tmp_path,
        workspace=workspace,
        provider_profile="codex",
    )
    pool._gateway_config = _family_config()

    await _assert_scoped_surfaces(
        pool=pool,
        catalog=catalog,
        turn_ctx=_context("100", owner=True),
        expected_tenant="owner",
        expected_private="owner-only row",
        excluded_private="marina-only row",
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )
    await _assert_scoped_surfaces(
        pool=pool,
        catalog=catalog,
        turn_ctx=_context("200", owner=False),
        expected_tenant="marina",
        expected_private="marina-only row",
        excluded_private="owner-only row",
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )


async def test_family_turn_never_uses_owner_honcho_recall_or_ingest(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    catalog = _seed_catalog(workspace)
    honcho = _FakeHoncho()
    shadow = ShadowMemoryBackend(
        CatalogMemoryBackend(catalog, workspace),
        honcho_client=honcho,  # type: ignore[arg-type]
        conversation_learning=True,
    )
    pool = OhmoSessionRuntimePool(
        cwd=tmp_path,
        workspace=workspace,
        provider_profile="codex",
    )
    pool._gateway_config = _family_config(
        memory_backend="shadow",
        visible_recall=True,
        conversation_learning=True,
    )
    pool._prompt_memory_backend = shadow
    turn_ctx = _context("200", owner=False)
    pool._session_owner_principals[turn_ctx.session_id] = turn_ctx.principal
    scope = pool._resolve_turn_memory_scope(turn_ctx)

    prompt = await pool._runtime_system_prompt(
        SimpleNamespace(cwd=str(tmp_path)),
        "recall family history",
        turn_ctx=turn_ctx,
        memory_scope=scope,
    )
    await pool._append_conversation_turn(
        turn_ctx=turn_ctx,
        memory_scope=scope,
        user_text="family user turn",
        assistant_text="family assistant turn",
    )
    await shadow.await_pending()

    assert "marina-only row" in prompt
    assert "owner-derived fact" not in prompt
    assert "honcho, derived" not in prompt
    assert honcho.queries == []
    assert honcho.messages == []

    owner_ctx = _context("100", owner=True)
    pool._session_owner_principals[owner_ctx.session_id] = owner_ctx.principal
    owner_scope = pool._resolve_turn_memory_scope(owner_ctx)
    owner_prompt = await pool._runtime_system_prompt(
        SimpleNamespace(cwd=str(tmp_path)),
        "recall owner history",
        turn_ctx=owner_ctx,
        memory_scope=owner_scope,
    )
    await pool._append_conversation_turn(
        turn_ctx=owner_ctx,
        memory_scope=owner_scope,
        user_text="owner user turn",
        assistant_text="owner assistant turn",
    )
    await shadow.await_pending()

    assert "owner-derived fact" in owner_prompt
    assert "honcho, derived" in owner_prompt
    assert honcho.queries == ["recall owner history"]
    assert len(honcho.messages) == 2


@pytest.mark.parametrize(
    ("turn_ctx", "session_principal", "enabled_tenants"),
    (
        (_context("999", owner=False), "999", ("owner", "marina")),
        (
            TurnContext(
                principal="200",
                is_owner=False,
                is_private=False,
                channel="telegram",
                chat_id="200",
                session_id="session-200",
            ),
            "200",
            ("owner", "marina"),
        ),
        (_context("200", owner=False), "different-principal", ("owner", "marina")),
        (_context("200", owner=False), "200", ("owner",)),
    ),
    ids=("unknown", "non-private", "non-isolated", "not-enabled"),
)
async def test_denied_family_turn_has_no_authoritative_memory_surface(
    turn_ctx: TurnContext,
    session_principal: str,
    enabled_tenants: tuple[str, ...],
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OHMO_MEMORY_JUDGE", "1")
    monkeypatch.setenv("OHMO_MEMORY_JUDGE_INTERVAL", "1")
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    _seed_catalog(workspace)
    pool = OhmoSessionRuntimePool(
        cwd=tmp_path,
        workspace=workspace,
        provider_profile="codex",
    )
    pool._gateway_config = _family_config(enabled_memory_tenants=enabled_tenants)
    pool._session_owner_principals[turn_ctx.session_id] = session_principal

    assert pool._resolve_turn_memory_scope(turn_ctx) is None
    prompt = await pool._runtime_system_prompt(
        SimpleNamespace(cwd=str(tmp_path)),
        "what do you remember?",
        turn_ctx=turn_ctx,
        memory_scope=None,
    )
    bundle = _surface_bundle()
    engaged = pool._configure_turn_memory_surfaces(
        bundle,
        turn_ctx,
        memory_scope=None,
    )
    pool._maybe_schedule_memory_judge(
        bundle,
        "denied",
        turn_ctx=turn_ctx,
        memory_scope=None,
    )

    assert engaged is False
    assert "# ohmo Memory" not in prompt
    assert bundle.tool_registry.get("memory") is None
    assert bundle.autodream_context is None
    assert pool._judge_turn_counts == {}
    assert pool._judge_tasks == {}


async def test_empty_identity_registries_preserve_the_single_backend_for_every_turn(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    pool = OhmoSessionRuntimePool(
        cwd=tmp_path,
        workspace=workspace,
        provider_profile="codex",
    )
    assert pool._gateway_config.owner_principals == ()
    assert pool._gateway_config.family_principals == {}
    owner = _context("100", owner=True)
    non_owner = _context("999", owner=False)

    owner_prompt = await pool._runtime_system_prompt(
        SimpleNamespace(cwd=str(tmp_path)),
        "same prompt",
        turn_ctx=owner,
    )
    non_owner_prompt = await pool._runtime_system_prompt(
        SimpleNamespace(cwd=str(tmp_path)),
        "same prompt",
        turn_ctx=non_owner,
    )
    owner_bundle = _surface_bundle()
    non_owner_bundle = _surface_bundle()
    assert pool._configure_turn_memory_surfaces(owner_bundle, owner)
    assert pool._configure_turn_memory_surfaces(non_owner_bundle, non_owner)

    assert owner_prompt == non_owner_prompt
    assert owner_bundle.tool_registry.get("memory")._store is pool._prompt_memory_backend
    assert non_owner_bundle.tool_registry.get("memory")._store is pool._prompt_memory_backend

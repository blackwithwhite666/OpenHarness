"""Runtime coverage for per-turn catalog audience scopes."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from openharness.channels.bus.events import InboundMessage
from openharness.engine.messages import ConversationMessage
from openharness.evals import DecisionTraceValidationError, TRACE_FINALIZATION
from openharness.tools.base import ToolExecutionContext, ToolRegistry
from ohmo.evals import GatewayEvalRecorder

from ohmo.gateway.config import save_gateway_config
from ohmo.gateway.memory_gate import MemoryScope
from ohmo.gateway.models import GatewayConfig
from ohmo.gateway.runtime import (
    OhmoSessionRuntimePool,
    _build_conversation_turn_metadata,
    _logical_turn_id_for_conversation,
    _message_identity_for_turn,
)
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


class _BoundFakeHoncho:
    instances: dict[str, _BoundFakeHoncho] = {}

    def __init__(self, base_url: str, jwt: str, workspace: str) -> None:
        self.base_url = base_url
        self.jwt = jwt
        self.workspace = workspace
        self.queries: list[tuple[str, dict[str, object]]] = []
        self.messages: list[tuple[str, list[dict[str, object]]]] = []
        self.instances[workspace] = self

    async def query_conclusions(self, query: str, **kwargs: object) -> list[object]:
        self.queries.append((query, kwargs))
        return [SimpleNamespace(content=f"{self.workspace} derived fact")]

    async def create_messages(
        self,
        session: str,
        messages: list[dict[str, object]],
    ) -> list[object]:
        self.messages.append((session, messages))
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


def _message() -> InboundMessage:
    return InboundMessage(
        channel="telegram",
        sender_id="100",
        chat_id="100",
        content="What did I eat today?",
        metadata={"message_id": 0, "chat_type": "p2p"},
        timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )


def _metadata(
    *,
    turn_ctx: TurnContext,
    scope: MemoryScope,
    message: InboundMessage,
    recorder: GatewayEvalRecorder | None,
) -> tuple[str, dict[str, object], dict[str, object]]:
    return _build_conversation_turn_metadata(
        turn_ctx=turn_ctx,
        message=message,
        scope=scope,
        recorder=recorder,
    )


def _assert_expected_metadata_keys(metadata: dict[str, object], *, recorder: bool = False) -> None:
    expected = {
        "tenant_id",
        "source_principal",
        "gateway_session_id",
        "logical_turn_id",
        "client_op_id",
        "decision_trace_status",
        "nutrition_annotation_status",
        "decision_trace_episode_id",
    }
    if recorder:
        assert set(metadata.keys()) == expected | {"decision_trace"}
    else:
        assert set(metadata.keys()) == expected


def _assert_metadata_fields(
    metadata: dict[str, object],
    *,
    turn_ctx: TurnContext,
    scope: MemoryScope,
    recorder: GatewayEvalRecorder | None,
) -> None:
    _assert_expected_metadata_keys(
        metadata,
        recorder="decision_trace" in metadata,
    )
    assert metadata["tenant_id"] == scope.private_tenant
    assert metadata["source_principal"] == f"{turn_ctx.channel}:{turn_ctx.principal}"
    assert metadata["gateway_session_id"] == turn_ctx.session_id
    expected_decision_trace_episode_id = None if recorder is None else recorder.episode_id
    assert metadata["decision_trace_episode_id"] == expected_decision_trace_episode_id
    assert metadata["client_op_id"].startswith(metadata["logical_turn_id"] + ":")

    if "decision_trace" in metadata:
        assert metadata["client_op_id"].endswith(":assistant")


def _build_recorder(tmp_path: Path, message: InboundMessage) -> GatewayEvalRecorder:
    bundle = SimpleNamespace(
        session_id="session-100",
        current_settings=lambda: SimpleNamespace(model="test-model"),
        cwd=str(tmp_path),
    )
    return GatewayEvalRecorder.start(
        workspace=tmp_path,
        bundle=bundle,
        message=message,
        session_key="telegram:100",
        user_text=message.content or "",
        user_goal="track nutrition",
    )


def _assert_honcho_metadata_depth(value: object, depth: int = 1) -> int:
    max_depth = depth
    if isinstance(value, dict):
        for nested in value.values():
            if isinstance(nested, dict):
                max_depth = max(max_depth, _assert_honcho_metadata_depth(nested, depth + 1))
            elif isinstance(nested, list):
                for item in nested:
                    if isinstance(item, dict):
                        max_depth = max(
                            max_depth,
                            _assert_honcho_metadata_depth(item, depth + 1),
                        )
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                max_depth = max(max_depth, _assert_honcho_metadata_depth(item, depth + 1))
    return max_depth


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
        message=_message(),
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
        message=_message(),
        user_text="owner user turn",
        assistant_text="owner assistant turn",
    )
    await shadow.await_pending()

    assert "owner-derived fact" in owner_prompt
    assert "honcho, derived" in owner_prompt
    assert honcho.queries == ["recall owner history"]
    assert len(honcho.messages) == 1
    owner_exchange = honcho.messages[0]
    assert isinstance(owner_exchange, list)
    assert len(owner_exchange) == 2
    assert owner_exchange[0]["content"] == "owner user turn"
    assert owner_exchange[0]["peer_id"] == "owner"
    assert owner_exchange[0]["metadata"]["role"] == "user"
    assert owner_exchange[1]["content"] == "owner assistant turn"
    assert owner_exchange[1]["peer_id"] == "ohmo"
    assert owner_exchange[1]["metadata"]["role"] == "assistant"


async def test_owner_and_marina_honcho_recall_and_ingest_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from ohmo.memory_service import honcho_client as honcho_client_module

    _BoundFakeHoncho.instances = {}
    monkeypatch.setattr(honcho_client_module, "HonchoClient", _BoundFakeHoncho)
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    _seed_catalog(workspace)
    save_gateway_config(
        _family_config(
            memory_backend="shadow",
            visible_recall=True,
            conversation_learning=True,
            honcho_base_url="https://honcho.test",
            honcho_api_key="owner-jwt",
            honcho_workspace="owner-workspace",
            tenant_honcho={
                "marina": {
                    "workspace": "marina-workspace",
                    "api_key": "marina-jwt",
                }
            },
        ),
        workspace,
    )
    pool = OhmoSessionRuntimePool(
        cwd=tmp_path,
        workspace=workspace,
        provider_profile="codex",
    )

    owner_ctx = _context("100", owner=True)
    marina_ctx = _context("200", owner=False)
    for turn_ctx in (owner_ctx, marina_ctx):
        pool._session_owner_principals[turn_ctx.session_id] = turn_ctx.principal
    owner_scope = pool._resolve_turn_memory_scope(owner_ctx)
    marina_scope = pool._resolve_turn_memory_scope(marina_ctx)
    assert owner_scope is not None
    assert marina_scope is not None

    owner_prompt = await pool._runtime_system_prompt(
        SimpleNamespace(cwd=str(tmp_path)),
        "owner recall",
        turn_ctx=owner_ctx,
        memory_scope=owner_scope,
    )
    marina_prompt = await pool._runtime_system_prompt(
        SimpleNamespace(cwd=str(tmp_path)),
        "marina recall",
        turn_ctx=marina_ctx,
        memory_scope=marina_scope,
    )
    await pool._append_conversation_turn(
        turn_ctx=owner_ctx,
        memory_scope=owner_scope,
        message=_message(),
        user_text="owner user turn",
        assistant_text="owner assistant turn",
    )
    await pool._append_conversation_turn(
        turn_ctx=marina_ctx,
        memory_scope=marina_scope,
        message=_message(),
        user_text="marina user turn",
        assistant_text="marina assistant turn",
    )
    owner_shadow = pool._shadow_backend_for_scope(owner_scope)
    marina_shadow = pool._shadow_backend_for_scope(marina_scope)
    assert owner_shadow is not None
    assert marina_shadow is not None
    await asyncio.gather(owner_shadow.await_pending(), marina_shadow.await_pending())

    owner_honcho = _BoundFakeHoncho.instances["owner-workspace"]
    marina_honcho = _BoundFakeHoncho.instances["marina-workspace"]
    assert (owner_honcho.base_url, owner_honcho.jwt) == (
        "https://honcho.test",
        "owner-jwt",
    )
    assert (marina_honcho.base_url, marina_honcho.jwt) == (
        "https://honcho.test",
        "marina-jwt",
    )
    assert owner_honcho.queries == [
        (
            "owner recall",
            {"observer": "ohmo", "observed": "owner", "top_k": 10},
        )
    ]
    assert marina_honcho.queries == [
        (
            "marina recall",
            {"observer": "ohmo", "observed": "marina", "top_k": 10},
        )
    ]
    assert "owner-workspace derived fact" in owner_prompt
    assert "marina-workspace derived fact" not in owner_prompt
    assert "marina-workspace derived fact" in marina_prompt
    assert "owner-workspace derived fact" not in marina_prompt
    assert len(owner_honcho.messages) == 1
    owner_session, owner_exchange = owner_honcho.messages[0]
    assert owner_session == "ohmo"
    assert len(owner_exchange) == 2
    assert owner_exchange[0]["content"] == "owner user turn"
    assert owner_exchange[0]["peer_id"] == "owner"
    assert owner_exchange[0]["metadata"]["role"] == "user"
    assert owner_exchange[1]["content"] == "owner assistant turn"
    assert owner_exchange[1]["peer_id"] == "ohmo"
    assert owner_exchange[1]["metadata"]["role"] == "assistant"

    assert len(marina_honcho.messages) == 1
    marina_session, marina_exchange = marina_honcho.messages[0]
    assert marina_session == "ohmo"
    assert len(marina_exchange) == 2
    assert marina_exchange[0]["content"] == "marina user turn"
    assert marina_exchange[0]["peer_id"] == "marina"
    assert marina_exchange[0]["metadata"]["role"] == "user"
    assert marina_exchange[1]["content"] == "marina assistant turn"
    assert marina_exchange[1]["peer_id"] == "ohmo"
    assert marina_exchange[1]["metadata"]["role"] == "assistant"


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


def test_runtime_memory_turn_metadata_statuses_cover_every_recorder_state(tmp_path: Path) -> None:
    scope = MemoryScope("owner", ("family-shared",))
    turn_ctx = _context("100", owner=True)
    message_with_id = _message()
    expected_turn_id = _logical_turn_id_for_conversation(
        turn_ctx=turn_ctx,
        message=message_with_id,
    )
    logical_turn_id, user_metadata, assistant_metadata = _metadata(
        turn_ctx=turn_ctx,
        scope=scope,
        message=message_with_id,
        recorder=None,
    )
    assert _metadata(turn_ctx=turn_ctx, scope=scope, message=message_with_id, recorder=None)[
        0
    ] == expected_turn_id
    assert logical_turn_id == expected_turn_id == assistant_metadata["logical_turn_id"] == user_metadata[
        "logical_turn_id"
    ]
    assert _assert_honcho_metadata_depth(user_metadata) == 1
    assert _assert_honcho_metadata_depth(assistant_metadata) == 1
    assert user_metadata["client_op_id"] == f"{expected_turn_id}:user"
    assert assistant_metadata["client_op_id"] == f"{expected_turn_id}:assistant"
    assert user_metadata["decision_trace_status"] == "disabled"
    assert user_metadata["nutrition_annotation_status"] == "disabled"
    assert assistant_metadata["decision_trace_status"] == "disabled"
    assert assistant_metadata["nutrition_annotation_status"] == "disabled"
    assert user_metadata["decision_trace_episode_id"] is None
    assert assistant_metadata["decision_trace_episode_id"] is None
    assert _message_identity_for_turn(message_with_id) == "0"
    _assert_metadata_fields(
        user_metadata,
        turn_ctx=turn_ctx,
        scope=scope,
        recorder=None,
    )
    _assert_metadata_fields(
        assistant_metadata,
        turn_ctx=turn_ctx,
        scope=scope,
        recorder=None,
    )
    _assert_expected_metadata_keys(user_metadata)
    _assert_expected_metadata_keys(assistant_metadata)
    assert not isinstance(logical_turn_id, dict)
    assert user_metadata["client_op_id"] != assistant_metadata["client_op_id"]


def test_runtime_memory_turn_metadata_status_recorded_with_nutrition(tmp_path: Path) -> None:
    scope = MemoryScope("owner", ("family-shared",))
    turn_ctx = _context("100", owner=True)
    message_with_id = _message()
    recorder = _build_recorder(tmp_path / ".ohmo-home", message_with_id)
    recorder.decision_trace_recorder.trace_requirement_signals("сколько калорий в супе")
    recorder.decision_trace_recorder.record(
        TRACE_FINALIZATION,
        {
            "schema_version": 1,
            "trace_event_id": "trace-1",
            "annotations": {
                "nutrition": {
                    "energy_kcal_min": 10.0,
                    "energy_kcal_max": 12.5,
                    "items": [
                        {
                            "name": "apple",
                            "quantity_text": "1",
                            "energy_kcal_min": 5,
                            "energy_kcal_max": 10,
                        }
                    ],
                }
            },
        },
    )
    _, user_metadata, assistant_metadata = _metadata(
        turn_ctx=turn_ctx,
        scope=scope,
        message=message_with_id,
        recorder=recorder,
    )

    assert user_metadata["decision_trace_status"] == "recorded"
    assert user_metadata["nutrition_annotation_status"] == "recorded"
    assert "decision_trace" not in user_metadata
    assert assistant_metadata["decision_trace_status"] == "recorded"
    assert assistant_metadata["nutrition_annotation_status"] == "recorded"
    assert "decision_trace" in assistant_metadata
    decision_trace = assistant_metadata["decision_trace"]
    assert isinstance(decision_trace, dict)
    assert decision_trace["kind"] == "trace_finalization"
    assert decision_trace["episode_id"] == recorder.episode_id
    assert decision_trace["schema_version"] == 1
    assert decision_trace["trace_event_id"] == "trace-1"
    assert decision_trace["annotations"]["nutrition"]["energy_kcal_min"] == 10.0
    assert decision_trace["annotations"]["nutrition"]["record_type"] == "meal_estimate"
    assert "payload" not in decision_trace
    assert isinstance(decision_trace["timestamp"], str)
    _assert_expected_metadata_keys(assistant_metadata, recorder=True)
    _assert_expected_metadata_keys(user_metadata)
    assert _assert_honcho_metadata_depth(assistant_metadata) <= 5
    assert _assert_honcho_metadata_depth(user_metadata) <= 1


def test_runtime_memory_turn_metadata_status_applicable_but_missing_finalization(tmp_path: Path) -> None:
    scope = MemoryScope("owner", ("family-shared",))
    turn_ctx = _context("100", owner=True)
    message_with_id = _message()
    recorder = _build_recorder(tmp_path / ".ohmo-home", message_with_id)
    recorder.decision_trace_recorder.trace_requirement_signals("сколько калорий в ужине?")
    _, user_metadata, assistant_metadata = _metadata(
        turn_ctx=turn_ctx,
        scope=scope,
        message=message_with_id,
        recorder=recorder,
    )

    assert user_metadata["decision_trace_status"] == "missing"
    assert user_metadata["nutrition_annotation_status"] == "missing"
    assert assistant_metadata["decision_trace_status"] == "missing"
    assert assistant_metadata["nutrition_annotation_status"] == "missing"
    assert "decision_trace" not in assistant_metadata
    _assert_expected_metadata_keys(user_metadata)
    _assert_expected_metadata_keys(assistant_metadata)


def test_runtime_memory_turn_metadata_status_generic_finalization_without_nutrition(tmp_path: Path) -> None:
    scope = MemoryScope("owner", ("family-shared",))
    turn_ctx = _context("100", owner=True)
    message_with_id = _message()
    recorder = _build_recorder(tmp_path / ".ohmo-home", message_with_id)
    recorder.decision_trace_recorder.record(
        TRACE_FINALIZATION,
        {
            "schema_version": 1,
            "trace_event_id": "trace-2",
            "outcome": "assistant responded",
        },
    )
    _, user_metadata, assistant_metadata = _metadata(
        turn_ctx=turn_ctx,
        scope=scope,
        message=message_with_id,
        recorder=recorder,
    )
    assert user_metadata["decision_trace_status"] == "recorded"
    assert user_metadata["nutrition_annotation_status"] == "missing"
    assert assistant_metadata["decision_trace_status"] == "recorded"
    assert assistant_metadata["nutrition_annotation_status"] == "missing"
    assert "decision_trace" in assistant_metadata
    _assert_expected_metadata_keys(user_metadata)
    _assert_expected_metadata_keys(assistant_metadata, recorder=True)


def test_runtime_memory_turn_metadata_status_invalid_nutrition_finalization(tmp_path: Path) -> None:
    scope = MemoryScope("owner", ("family-shared",))
    turn_ctx = _context("100", owner=True)
    message_with_id = _message()
    recorder = _build_recorder(tmp_path / ".ohmo-home", message_with_id)
    recorder.decision_trace_recorder.trace_requirement_signals("сколько калорий в салате")
    with pytest.raises(DecisionTraceValidationError):
        recorder.decision_trace_recorder.record(
            TRACE_FINALIZATION,
            {
                "schema_version": 1,
                "trace_event_id": "trace-3",
                "annotations": {
                    "nutrition": {
                        "energy_kcal_min": -1,
                    }
                },
            },
        )
    _, user_metadata, assistant_metadata = _metadata(
        turn_ctx=turn_ctx,
        scope=scope,
        message=message_with_id,
        recorder=recorder,
    )
    assert user_metadata["decision_trace_status"] == "invalid"
    assert user_metadata["nutrition_annotation_status"] == "invalid"
    assert assistant_metadata["decision_trace_status"] == "invalid"
    assert assistant_metadata["nutrition_annotation_status"] == "invalid"
    assert "decision_trace" not in assistant_metadata
    _assert_expected_metadata_keys(user_metadata)
    _assert_expected_metadata_keys(assistant_metadata)


def test_runtime_memory_turn_metadata_status_not_applicable(tmp_path: Path) -> None:
    scope = MemoryScope("owner", ("family-shared",))
    turn_ctx = _context("100", owner=True)
    message_with_id = _message()
    recorder = _build_recorder(tmp_path / ".ohmo-home", message_with_id)
    _, user_metadata, assistant_metadata = _metadata(
        turn_ctx=turn_ctx,
        scope=scope,
        message=message_with_id,
        recorder=recorder,
    )
    assert user_metadata["decision_trace_status"] == "missing"
    assert user_metadata["nutrition_annotation_status"] == "not_applicable"
    assert assistant_metadata["decision_trace_status"] == "missing"
    assert assistant_metadata["nutrition_annotation_status"] == "not_applicable"
    assert "decision_trace" not in assistant_metadata
    _assert_expected_metadata_keys(user_metadata)
    _assert_expected_metadata_keys(assistant_metadata)


def test_runtime_memory_turn_metadata_timestamp_fallback_is_isoformat(tmp_path: Path) -> None:
    scope = MemoryScope("owner", ("family-shared",))
    turn_ctx = _context("100", owner=True)
    message = InboundMessage(
        channel="telegram",
        sender_id="100",
        chat_id="100",
        content="What did I eat today?",
        metadata={"chat_type": "p2p"},
        timestamp=datetime(2026, 2, 1, 2, 3, 4, tzinfo=timezone.utc),
    )
    assert _message_identity_for_turn(message) == "2026-02-01T02:03:04+00:00"

    recorder = _build_recorder(tmp_path / ".ohmo-home", message)
    logical_turn_id_first, _, _ = _metadata(
        turn_ctx=turn_ctx,
        scope=scope,
        message=message,
        recorder=recorder,
    )
    logical_turn_id_second, _, _ = _metadata(
        turn_ctx=turn_ctx,
        scope=scope,
        message=message,
        recorder=recorder,
    )
    assert logical_turn_id_first == logical_turn_id_second

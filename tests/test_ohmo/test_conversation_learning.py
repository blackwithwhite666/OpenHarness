"""Tests for default-off, owner-scoped Honcho conversation learning."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openharness.api.usage import UsageSnapshot
from openharness.channels.bus.events import InboundMessage
from openharness.engine.stream_events import AssistantTextDelta
from openharness.evals import TRACE_FINALIZATION

from ohmo.gateway.config import save_gateway_config
from ohmo.gateway.models import GatewayConfig
from ohmo.gateway.runtime import (
    OhmoSessionRuntimePool,
    _logical_turn_id_for_conversation,
)
from ohmo.gateway.turn_context import TurnContext
from ohmo.memory_backend import CatalogMemoryBackend, ShadowMemoryBackend, make_memory_backend
from ohmo.memory_catalog import MemoryCatalog
from ohmo.memory_service.outbox import DrainReport, drain_once
from ohmo.workspace import initialize_workspace


class FakeHoncho:
    def __init__(self, *, wait: bool = False, fail: bool = False) -> None:
        self.wait = wait
        self.fail = fail
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.messages: list[tuple[str, list[dict[str, object]]]] = []
        self.conclusions: list[list[dict[str, object]]] = []

    async def create_messages(
        self,
        session: str,
        messages: list[dict[str, object]],
    ) -> list[Any]:
        self.messages.append((session, messages))
        self.started.set()
        if self.wait:
            await self.release.wait()
        if self.fail:
            raise OSError("fake Honcho is unavailable")
        return []

    async def create_conclusions(
        self,
        conclusions: list[dict[str, object]],
    ) -> list[Any]:
        self.conclusions.append(conclusions)
        return [SimpleNamespace(id=f"conclusion-{len(self.conclusions)}")]

    async def delete_conclusion(self, conclusion_id: str) -> None:
        del conclusion_id


def _shadow_backend(
    workspace: Path,
    honcho: FakeHoncho,
    *,
    conversation_learning: bool,
) -> ShadowMemoryBackend:
    return ShadowMemoryBackend(
        CatalogMemoryBackend(MemoryCatalog(workspace), workspace),
        honcho_client=honcho,  # type: ignore[arg-type]
        conversation_learning=conversation_learning,
        comparison_log_path=workspace / "comparison.jsonl",
    )


def _owner_context(*, is_owner: bool = True, is_private: bool = True) -> TurnContext:
    return TurnContext(
        principal="owner",
        is_owner=is_owner,
        is_private=is_private,
        channel="feishu",
        chat_id="owner-chat",
        session_id="owner-session",
    )


def _message(
    *,
    message_id: int | str = 0,
    content: str = "User request.",
    timestamp: datetime | None = None,
) -> InboundMessage:
    return InboundMessage(
        channel="feishu",
        sender_id="owner",
        chat_id="owner-chat",
        content=content,
        metadata={"message_id": message_id, "chat_type": "p2p"},
        timestamp=timestamp or datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
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


def _pool_for_gate(
    backend: ShadowMemoryBackend,
    turn_ctx: TurnContext,
    *,
    conversation_learning: bool,
    session_owner_principal: str | None = "owner",
) -> OhmoSessionRuntimePool:
    pool = object.__new__(OhmoSessionRuntimePool)
    pool._gateway_config = GatewayConfig(conversation_learning=conversation_learning)
    pool._prompt_memory_backend = backend
    pool._session_owner_principals = {
        turn_ctx.session_id: session_owner_principal,
    }
    return pool


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    decision_trace_payload: dict[str, Any] | None = None,
) -> None:
    async def fake_build_runtime(**kwargs: object) -> SimpleNamespace:
        del kwargs

        class FakeEngine:
            messages: list[object] = []
            total_usage = UsageSnapshot()
            tool_metadata: dict[str, object] = {}
            _decision_trace_recorder: object | None = None

            def set_system_prompt(self, prompt: str) -> None:
                del prompt

            def set_decision_trace_recorder(self, recorder: object | None) -> None:
                self._decision_trace_recorder = recorder

            async def submit_message(self, content: object):
                del content
                if (
                    self._decision_trace_recorder is not None
                    and decision_trace_payload is not None
                ):
                    self._decision_trace_recorder.record(TRACE_FINALIZATION, decision_trace_payload)
                yield AssistantTextDelta(text="Assistant response.")

        return SimpleNamespace(
            engine=FakeEngine(),
            cwd=str(tmp_path),
            session_id="owner-session",
            current_settings=lambda: SimpleNamespace(model="fake-model"),
            commands=SimpleNamespace(lookup=lambda raw: None),
        )

    async def fake_start_runtime(bundle: object) -> None:
        del bundle

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)


async def test_owner_private_isolated_turn_ingests_without_delaying_reply(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    save_gateway_config(
        GatewayConfig(
            evals_capture=False,
            memory_backend="shadow",
            conversation_learning=True,
            owner_principals=("owner",),
        ),
        workspace,
    )
    _install_fake_runtime(monkeypatch, tmp_path)
    honcho = FakeHoncho(wait=True)
    backend = _shadow_backend(workspace, honcho, conversation_learning=True)
    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    pool._prompt_memory_backend = backend
    message = _message()

    async def collect_updates() -> list[object]:
        return [update async for update in pool.stream_message(message, "feishu:owner-chat")]

    updates = await asyncio.wait_for(collect_updates(), timeout=0.2)

    assert updates[-1].kind == "final"
    assert updates[-1].text == "Assistant response."
    await asyncio.wait_for(honcho.started.wait(), timeout=0.2)
    honcho.release.set()
    await backend.await_pending()
    assert len(honcho.messages) == 1
    [session, messages] = honcho.messages[0]
    assert session == "ohmo"
    assert len(messages) == 2
    assert messages[0]["content"] == "User request."
    assert messages[0]["peer_id"] == "owner"
    assert messages[0]["metadata"]["role"] == "user"
    assert messages[1]["content"] == "Assistant response."
    assert messages[1]["peer_id"] == "ohmo"
    assert messages[1]["metadata"]["role"] == "assistant"


async def test_learning_off_never_calls_create_messages(tmp_path: Path) -> None:
    honcho = FakeHoncho()
    backend = _shadow_backend(tmp_path, honcho, conversation_learning=True)
    turn_ctx = _owner_context()
    pool = _pool_for_gate(backend, turn_ctx, conversation_learning=False)

    await pool._append_conversation_turn(
        turn_ctx=turn_ctx,
        message=_message(),
        user_text="User request.",
        assistant_text="Assistant response.",
    )
    await backend.await_pending()

    assert GatewayConfig().conversation_learning is False
    assert honcho.messages == []


@pytest.mark.parametrize(
    ("turn_ctx", "session_owner_principal"),
    (
        (_owner_context(is_owner=False), "owner"),
        (_owner_context(is_private=False), "owner"),
        (_owner_context(), "different-principal"),
    ),
    ids=("non-owner", "non-private", "non-isolated"),
)
async def test_runtime_gate_rejects_ineligible_turns(
    turn_ctx: TurnContext,
    session_owner_principal: str | None,
    tmp_path: Path,
) -> None:
    honcho = FakeHoncho()
    backend = _shadow_backend(tmp_path, honcho, conversation_learning=True)
    pool = _pool_for_gate(
        backend,
        turn_ctx,
        conversation_learning=True,
        session_owner_principal=session_owner_principal,
    )

    await pool._append_conversation_turn(
        turn_ctx=turn_ctx,
        message=_message(),
        user_text="User request.",
        assistant_text="Assistant response.",
    )
    await backend.await_pending()

    assert honcho.messages == []


async def test_honcho_ingestion_error_is_swallowed_and_turn_remains_complete(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    save_gateway_config(
        GatewayConfig(
            evals_capture=False,
            memory_backend="shadow",
            conversation_learning=True,
            owner_principals=("owner",),
        ),
        workspace,
    )
    _install_fake_runtime(monkeypatch, tmp_path)
    honcho = FakeHoncho(fail=True)
    backend = _shadow_backend(workspace, honcho, conversation_learning=True)
    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    pool._prompt_memory_backend = backend
    message = _message()

    updates = [update async for update in pool.stream_message(message, "feishu:owner-chat")]
    await backend.await_pending()

    assert updates[-1].kind == "final"
    assert updates[-1].text == "Assistant response."
    assert len(honcho.messages) == 1
    assert "conversation learning exchange ingestion failed" in caplog.text


async def test_stream_message_owner_private_turn_emits_flattened_trace_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    save_gateway_config(
        GatewayConfig(
            evals_capture=True,
            memory_backend="shadow",
            conversation_learning=True,
            owner_principals=("owner",),
        ),
        workspace,
    )
    _install_fake_runtime(
        monkeypatch,
        tmp_path,
        decision_trace_payload={
            "schema_version": 1,
            "trace_event_id": "trace-1",
            "outcome": "assistant responded",
            "annotations": {
                "nutrition": {
                    "energy_kcal_min": 10.0,
                    "energy_kcal_max": 12.0,
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
    honcho = FakeHoncho()
    backend = _shadow_backend(workspace, honcho, conversation_learning=True)
    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    pool._prompt_memory_backend = backend
    turn_ctx = _owner_context()
    message = _message(content="What did I eat today?")

    updates = [update async for update in pool.stream_message(message, "feishu:owner-chat")]
    await backend.await_pending()

    assert updates[-1].kind == "final"
    assert updates[-1].text == "Assistant response."
    assert len(honcho.messages) == 1
    session, messages = honcho.messages[0]
    assert session == "ohmo"
    assert len(messages) == 2
    assert messages[0]["content"] == "What did I eat today?"
    assert messages[0]["peer_id"] == "owner"
    assert messages[0]["metadata"]["role"] == "user"
    assert messages[1]["content"] == "Assistant response."
    assert messages[1]["peer_id"] == "ohmo"
    assert messages[1]["metadata"]["role"] == "assistant"

    user_metadata = messages[0]["metadata"]
    assistant_metadata = messages[1]["metadata"]
    assert user_metadata["tenant_id"] == "owner"
    assert assistant_metadata["tenant_id"] == "owner"
    assert user_metadata["source_principal"] == "feishu:owner"
    assert assistant_metadata["source_principal"] == "feishu:owner"
    assert user_metadata["gateway_session_id"] == "owner-session"
    assert assistant_metadata["gateway_session_id"] == "owner-session"
    assert user_metadata["decision_trace_status"] == "recorded"
    assert assistant_metadata["decision_trace_status"] == "recorded"
    assert user_metadata["nutrition_annotation_status"] == "recorded"
    assert assistant_metadata["nutrition_annotation_status"] == "recorded"
    assert user_metadata["decision_trace_episode_id"] == assistant_metadata["decision_trace_episode_id"]

    expected_turn_id = _logical_turn_id_for_conversation(
        turn_ctx=turn_ctx,
        message=message,
    )
    assert user_metadata["logical_turn_id"] == expected_turn_id
    assert assistant_metadata["logical_turn_id"] == expected_turn_id
    assert user_metadata["client_op_id"] == f"{expected_turn_id}:user"
    assert assistant_metadata["client_op_id"] == f"{expected_turn_id}:assistant"
    assert user_metadata["role"] == "user"
    assert assistant_metadata["role"] == "assistant"
    assert "decision_trace" not in user_metadata
    assert "decision_trace" in assistant_metadata

    decision_trace = assistant_metadata["decision_trace"]
    assert isinstance(decision_trace, dict)
    assert decision_trace["kind"] == "trace_finalization"
    assert decision_trace["episode_id"] == user_metadata["decision_trace_episode_id"]
    assert decision_trace["schema_version"] == 1
    assert decision_trace["trace_event_id"] == "trace-1"
    assert decision_trace["outcome"] == "assistant responded"
    assert isinstance(decision_trace["timestamp"], str)
    assert decision_trace["timestamp"].endswith("+00:00")
    assert "payload" not in decision_trace
    assert "raw_episode" not in decision_trace
    assert "events" not in decision_trace

    nutrition = decision_trace["annotations"]["nutrition"]
    assert nutrition["record_type"] == "meal_estimate"
    assert nutrition["energy_kcal_min"] == 10.0
    assert nutrition["energy_kcal_max"] == 12.0
    nutrition_item = nutrition["items"][0]
    assert nutrition_item["name"] == "apple"
    assert nutrition_item["quantity_text"] == "1"
    assert nutrition_item["energy_kcal_min"] == 5.0
    assert nutrition_item["energy_kcal_max"] == 10.0
    assert json.dumps(user_metadata)
    assert json.dumps(assistant_metadata)
    assert _assert_honcho_metadata_depth(user_metadata) == 1
    assert _assert_honcho_metadata_depth(assistant_metadata) <= 5


def test_shadow_factory_passes_learning_flag_and_preserves_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from ohmo.memory_service import honcho_client as honcho_client_module

    honcho = FakeHoncho()
    monkeypatch.setattr(honcho_client_module, "HonchoClient", lambda *args: honcho)

    backend = make_memory_backend(
        GatewayConfig(
            memory_backend="shadow",
            conversation_learning=True,
            owner_principals=("owner",),
            honcho_base_url="https://honcho.test",
            honcho_api_key="secret",
            honcho_workspace="workspace-one",
        ),
        tmp_path,
    )

    assert isinstance(backend, ShadowMemoryBackend)
    assert backend._conversation_learning is True
    assert GatewayConfig().memory_backend == "file"
    with pytest.raises(NotImplementedError, match="honcho memory backend not built"):
        make_memory_backend(GatewayConfig(memory_backend="honcho"), tmp_path / "honcho")


async def test_curated_mirror_uses_direct_conclusions_not_messages(tmp_path: Path) -> None:
    catalog = MemoryCatalog(db_path=tmp_path / "catalog.sqlite3")
    honcho = FakeHoncho()
    assert catalog.add("owner", "Timezone", "User lives in Moscow.").ok

    report = await drain_once(catalog, honcho)  # type: ignore[arg-type]

    assert report == DrainReport(mirrored=1)
    assert honcho.conclusions == [
        [
            {
                "content": "User lives in Moscow.",
                "observer_id": "ohmo-curated",
                "observed_id": "owner",
            }
        ]
    ]
    assert honcho.messages == []

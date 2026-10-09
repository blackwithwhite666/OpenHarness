"""Ordinary nutrition stream, source provenance, and durable append regressions."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from openharness.channels.bus.events import InboundMessage, OutboundMessage
from openharness.evals import TRACE_FINALIZATION
from openharness.engine.stream_events import AssistantTextDelta
from openharness.engine.messages import ConversationMessage
from openharness.tools.base import ToolExecutionContext, ToolRegistry

from ohmo.gateway.memory_gate import MemoryScope
from ohmo.gateway.models import GatewayConfig
from ohmo.gateway.runtime import OhmoSessionRuntimePool, _build_inbound_user_message
from ohmo.memory_backend import ShadowMemoryBackend
from ohmo.memory_service.honcho_client import Message
from ohmo.workspace import initialize_workspace
from ohmo.attachment_store import AttachmentStore
from ohmo.conversation_image_tool import LoadConversationImageInput, LoadConversationImageTool
from tests.test_ohmo.test_conversation_attachments import PNG_BYTES

from tests.test_ohmo.test_camera_ingress import _ingress


class _BaseMemory:
    def __init__(self, root: Path) -> None:
        self._memory_dir = root

    async def append_turn(self, role: str, text: str) -> None:
        del role, text


class _Honcho:
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.fail_before_append = False
        self.timeout_after_append = False
        self.wrong_assistant_operation = False

    async def find_messages_by_client_op_id(self, session: str, operation: str):
        del session
        return [m for m in self.messages if m.metadata.get("client_op_id") == operation]

    async def create_messages(self, session: str, values: list[dict]):
        if self.fail_before_append:
            raise OSError("synthetic append unavailable")
        result = []
        for value in values:
            metadata = dict(value["metadata"])
            if self.wrong_assistant_operation and metadata.get("role") == "assistant":
                metadata["client_op_id"] = "synthetic-wrong-operation"
            message = Message(
                id=f"honcho-{len(self.messages) + 1}",
                content=value["content"],
                peer_id=value["peer_id"],
                session_id=session,
                metadata=metadata,
                created_at=datetime.now(timezone.utc),
                workspace_id="fixture",
                token_count=1,
            )
            self.messages.append(message)
            result.append(message)
        if self.timeout_after_append:
            raise TimeoutError("synthetic response lost after authoritative append")
        return result


class _Engine:
    def __init__(self) -> None:
        self.decision_trace_recorder = None
        self.tool_metadata: dict = {}
        self.messages: list = []
        self.turns: list[tuple[str, list[str], datetime]] = []
        self.pool = None
        self.wellness_actors = []
        self.scripted_payload = None
        self.scripted_answer = "Спасибо."

    def set_decision_trace_recorder(self, recorder) -> None:
        self.decision_trace_recorder = recorder

    def set_system_prompt(self, prompt: str) -> None:
        del prompt

    async def submit_message(self, _user_message, *, wellness_actor=None):
        self.wellness_actors.append(wellness_actor)
        self.messages.append(_user_message)
        message = self.pool._active_message
        self.turns.append((message.content, list(message.media), message.timestamp))
        recorder = self.decision_trace_recorder
        if self.scripted_payload is not None:
            recorder.record(TRACE_FINALIZATION, {
                **self.scripted_payload,
                "trace_event_id": f"synthetic-trace-{message.metadata.get('message_id')}",
            })
        yield AssistantTextDelta(text=self.scripted_answer)


def _consumed_payload() -> dict:
    return {
        "schema_version": 1,
        "trace_event_id": "synthetic-meal-finalization",
        "annotations": {
            "nutrition": {
                "schema_version": 2,
                "record_type": "meal_observation",
                "consumption_status": "consumed",
                "basis": ["image", "owner_statement"],
                "energy_kcal_best": 120,
                "items": [{"name": "pears", "quantity_text": "2 pieces"}],
            }
        },
    }


def _correction_payload() -> dict:
    return {
        "schema_version": 1,
        "trace_event_id": "synthetic-meal-correction",
        "annotations": {"nutrition": {
            "schema_version": 2,
            "record_type": "meal_correction",
            "changed_fields": ["meal_date"],
            "meal_date": "2026-09-28",
        }},
    }


def _script_finalization(
    pool, payload: dict | None, answer: str = "Записано."
) -> None:
    """Model fixture emits an explicit annotation independent of message words."""
    engine = pool._test_bundle.engine

    async def submit(user_message, *, wellness_actor=None):
        del wellness_actor
        engine.messages.append(user_message)
        recorder = engine.decision_trace_recorder
        recorder.trace_requirement_signals(pool._active_message.content)
        if payload is not None:
            recorder.record(TRACE_FINALIZATION, {
                **payload,
                "trace_event_id": f"synthetic-trace-{pool._active_message.metadata.get('message_id')}",
            })
        yield AssistantTextDelta(text=answer)

    engine.submit_message = submit


def _pool(tmp_path: Path, ingress, honcho: _Honcho, monkeypatch) -> OhmoSessionRuntimePool:
    workspace = tmp_path / "workspace"
    initialize_workspace(workspace)
    cfg = GatewayConfig(
        enabled_channels=["telegram"],
        conversation_learning=True,
        evals_capture=True,
        memory_backend="shadow",
        honcho_base_url="https://honcho.fixture.invalid",
        family_principals={"123": "marina"},
        enabled_memory_tenants=("marina",),
        tenant_honcho={"marina": {"workspace": "fixture", "api_key": "fixture", "observed_peer": "owner"}},
        camera_ingress=ingress.config,
    )
    pool = object.__new__(OhmoSessionRuntimePool)
    pool._gateway_config = cfg
    pool._workspace = workspace
    pool._cwd = workspace
    pool._attachment_store = None
    pool._session_backend = SimpleNamespace()
    pool._bundles = {}
    pool._camera_ingress = ingress
    pool._session_owner_principals = {"camera-session": "123"}
    pool._cwd_for_message = lambda *_: workspace
    pool._bind_session_owner = lambda *_: None
    pool._resolve_turn_memory_scope = lambda *_: MemoryScope("marina", ())
    pool._configure_turn_memory_surfaces = lambda *_args, **_kwargs: None
    pool._maybe_schedule_memory_judge = lambda *_args, **_kwargs: None
    pool._todo_cleanup_update = lambda **_kwargs: None

    async def pass_todo_guard(**kwargs):
        kwargs["state"]["reply"] = "".join(kwargs["reply_parts"])
        if False:
            yield None

    pool._guard_todo_final = pass_todo_guard
    backend = ShadowMemoryBackend(
        _BaseMemory(workspace), honcho, conversation_learning=True
    )
    pool._shadow_backend_for_scope = lambda _scope: backend
    engine = _Engine()
    engine.pool = pool
    engine.scripted_payload = _consumed_payload()
    bundle = SimpleNamespace(
        session_id="camera-session",
        engine=engine,
        commands=SimpleNamespace(lookup=lambda _text: None),
        tool_registry=None,
        cwd=str(workspace),
    )
    pool._test_bundle = bundle

    async def get_bundle(*_args, **_kwargs):
        return bundle

    async def save_snapshot(*_args, **_kwargs):
        return None

    async def runtime_prompt(*_args, **_kwargs):
        return "synthetic system prompt"

    pool.get_bundle = get_bundle
    pool._save_snapshot = save_snapshot
    pool._runtime_system_prompt = runtime_prompt
    pool._register_conversation_image_tool = lambda *_args, **_kwargs: None
    pool._set_group_request_context = lambda *_args: None
    pool._restore_group_request_context = lambda *_args: None
    pool._clear_reminder_context = lambda *_args: None
    return pool


async def _turn(pool, message, ingress):
    pool._active_message = message
    updates = [
        update
        async for update in pool.stream_message(message, ingress.config.session_key)
    ]
    final = next((update for update in updates if update.kind == "final"), None)
    assert final is not None, (updates, pool._test_bundle.engine.turns)
    outbound = OutboundMessage(
        channel="telegram",
        chat_id="123",
        content=final.text,
        metadata=final.metadata,
    )
    from openharness.channels.bus.events import OutboundDeliveryReceipt

    candidate_id = message.metadata.get("_camera_candidate_id")
    if isinstance(candidate_id, str):
        native_id = 500 + len(ingress._attempts[candidate_id]["reply_ids"])
        ingress.note_assistant_receipt(
            outbound,
            OutboundDeliveryReceipt(
                channel="telegram", chat_id="123", native_message_ids=(native_id,)
            ),
        )
    return final


@pytest.mark.asyncio
async def test_ordinary_meal_append_is_receipt_bound_and_replay_safe(tmp_path, monkeypatch):
    ingress, _root, _bus, _ = _ingress(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    _script_finalization(pool, _consumed_payload())
    first = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Я съела два кусочка, запиши завтрак",
        metadata={"message_id": 801, "is_group": False, "_synthetic": True},
        timestamp=datetime(2026, 9, 28, 8, 30, tzinfo=timezone.utc),
    )
    result = await _turn(pool, first, ingress)
    assert result.text.startswith(honcho.messages[1].content)
    assert "Записано; баланс обновляется." in result.text
    assert result.metadata["nutrition_sync_status"] == "pending"
    event_id = result.metadata["nutrition_append_event_id"]
    assert event_id == "honcho-2"
    assert len(honcho.messages) == 2
    assert honcho.messages[1].metadata["decision_trace"]["annotations"]["nutrition"][
        "record_type"
    ] == "meal_observation"

    replay = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content=first.content,
        metadata={"message_id": 801, "is_group": False, "_synthetic": True},
        timestamp=first.timestamp,
    )
    engine = pool._test_bundle.engine
    turns_before = tuple(engine.turns)
    messages_before = tuple(engine.messages)
    stored_before = tuple(
        (row.id, row.content, repr(row.metadata), row.created_at)
        for row in honcho.messages
    )
    updates = [update async for update in pool.stream_message(replay, ingress.config.session_key)]
    assert updates == []
    assert tuple(engine.turns) == turns_before
    assert tuple(engine.messages) == messages_before
    assert tuple(
        (row.id, row.content, repr(row.metadata), row.created_at)
        for row in honcho.messages
    ) == stored_before
    assert result.metadata["nutrition_append_event_id"] == event_id
    assert honcho.messages[1].id == event_id
    assert honcho.messages[1].metadata["received_at"] == first.timestamp.isoformat()

    next_day = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Я съела два кусочка, запиши завтрак",
        metadata={"message_id": 802, "is_group": False, "_synthetic": True},
        timestamp=datetime(2026, 9, 29, 8, 30, tzinfo=timezone.utc),
    )
    next_result = await _turn(pool, next_day, ingress)
    assert next_result.metadata["nutrition_append_event_id"] == "honcho-4"
    assert next_result.metadata["nutrition_append_event_id"] != event_id
    assert len(honcho.messages) == 4
    assert honcho.messages[3].metadata["received_at"] == next_day.timestamp.isoformat()

    await ingress.close()


@pytest.mark.asyncio
async def test_text_only_observation_can_be_corrected_by_authenticated_native_reply(
    tmp_path, monkeypatch,
):
    ingress, _root, _bus, _ = _ingress(tmp_path)
    try:
        honcho = _Honcho()
        pool = _pool(tmp_path, ingress, honcho, monkeypatch)
        observation = InboundMessage(
            channel="telegram", sender_id="123", chat_id="123",
            content="I ate one bowl of oatmeal for breakfast.",
            metadata={"message_id": "text-meal-901", "is_group": False},
            timestamp=datetime(2026, 9, 28, 8, 30, tzinfo=timezone.utc),
        )
        _script_finalization(pool, _consumed_payload())
        first = await _turn(pool, observation, ingress)
        assert first.metadata.get("nutrition_sync_status") == "pending", (
            first.metadata, len(honcho.messages)
        )
        original = honcho.messages[1]
        assert original.metadata["ingest_source"] == "telegram"
        assert original.metadata["source_message_id"] == "text-meal-901"

        correction = InboundMessage(
            channel="telegram", sender_id="123", chat_id="123",
            content="The oatmeal was yesterday, please fix its date.",
            metadata={
                "message_id": "text-correction-902",
                "reply_to_message_id": "text-meal-901",
                "is_group": False,
            },
            timestamp=datetime(2026, 9, 29, 8, 30, tzinfo=timezone.utc),
        )
        _script_finalization(pool, _correction_payload(), "I corrected the date.")
        result = await _turn(pool, correction, ingress)
        saved = honcho.messages[-1].metadata
        assert result.metadata.get("nutrition_sync_status") == "pending", (
            result.metadata, len(honcho.messages),
            {
                key: honcho.messages[-1].metadata.get(key)
                for key in ("selected_source", "target_meal_id", "reply_to_source_message_id", "decision_trace")
            } if honcho.messages else None,
        )
        assert saved["ingest_source"] == "telegram"
        assert saved["source_message_id"] == "text-correction-902"
        assert saved["reply_to_source_message_id"] == "text-meal-901"
        assert saved["selected_source"]["source_message_id"] == "text-meal-901"
        assert saved["selected_source"]["original_receipt_event_id"] == original.id
        assert saved["target_meal_id"]
        assert len(honcho.messages) == 4
    finally:
        await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("target_kind", ["missing", "foreign", "forged"])
async def test_text_only_native_reply_correction_requires_owned_original_receipt(
    tmp_path, monkeypatch, target_kind,
):
    ingress, _root, _bus, _ = _ingress(tmp_path)
    try:
        honcho = _Honcho()
        pool = _pool(tmp_path, ingress, honcho, monkeypatch)
        observation = InboundMessage(
            channel="telegram", sender_id="123", chat_id="123",
            content="I ate one bowl of oatmeal for breakfast.",
            metadata={"message_id": "text-meal-911", "is_group": False},
            timestamp=datetime(2026, 9, 28, 8, 30, tzinfo=timezone.utc),
        )
        _script_finalization(pool, _consumed_payload())
        await _turn(pool, observation, ingress)
        if target_kind == "foreign":
            honcho.messages[1].metadata["tenant_id"] = "another-owner"
        elif target_kind == "forged":
            honcho.messages[1].metadata["source_message_id"] = "forged-source-id"
        reply_target = "missing-meal-999" if target_kind == "missing" else "text-meal-911"
        correction = InboundMessage(
            channel="telegram", sender_id="123", chat_id="123",
            content="Please correct that meal date.",
            metadata={
                "message_id": "text-correction-912",
                "reply_to_message_id": reply_target,
                "is_group": False,
            },
            timestamp=datetime(2026, 9, 29, 8, 30, tzinfo=timezone.utc),
        )
        _script_finalization(pool, _correction_payload())
        result = await _turn(pool, correction, ingress)
        assert len(honcho.messages) == 2
        assert "nutrition_append_event_id" not in result.metadata
    finally:
        await ingress.close()


@pytest.mark.asyncio
async def test_late_ordinary_correction_after_runtime_reconstruction_uses_original_receipt(
    tmp_path, monkeypatch,
):
    """A rotated runtime selects untouched old Camera provenance and appends normally."""
    ingress, _root, _bus, _ = _ingress(tmp_path)
    try:
        honcho = _Honcho()
        original_runtime = _pool(tmp_path, ingress, honcho, monkeypatch)
        original_runtime._test_bundle.session_id = "runtime-session-original"
        pool = _pool(tmp_path, ingress, honcho, monkeypatch)
        # Simulate a newly reconstructed runtime. The durable historical ref
        # remains byte-for-byte in its original session; only this runtime's
        # active bundle is newer.
        pool._test_bundle.session_id = "runtime-session-current"
        monkeypatch.setattr(
            "ohmo.gateway.runtime._build_inbound_user_message",
            _build_inbound_user_message,
        )
        store = AttachmentStore(pool._workspace)
        pool._attachment_store = store
        ref = store.ingest_bytes(PNG_BYTES, media_type="image/png")
        provenance = {
            "schema_version": 1, "channel": "telegram",
            "principal": "telegram:__camera__", "chat_id": "123",
            "session_key": "telegram:123",
            "gateway_session_id": original_runtime._test_bundle.session_id,
            "received_at": "2026-10-01T20:59:58+00:00",
            "timestamp_authority": "camera_capture_time", "is_group": False,
            "is_forwarded": False, "source_message_id": "71",
            "photo_source_message_id": "71", "source_origin": "dropbox_camera",
            "origin_principal": "telegram:__camera__", "owner_principal": "telegram:123",
            "camera_candidate_id": "dropbox-camera-v1-" + "a" * 64,
            "native_photo_message_id": "71",
            "consumed_occurrences": [{
                "append_source_message_id": "original-meal-append",
                "receipt_event_id": "original-assistant-event",
                "client_op_id": "original-observation:assistant",
                "gateway_session_id": original_runtime._test_bundle.session_id,
            }],
        }
        camera_candidate_id = "dropbox-camera-v1-" + "a" * 64
        newer_candidate_id = "dropbox-camera-v1-" + "b" * 64
        ingress._attempts[camera_candidate_id] = {
            "state": "photo_sent", "photo_id": 71,
            "photo_delivery_confirmed": True, "attention_active": True,
        }
        ingress._attempts[newer_candidate_id] = {
            "state": "photo_sent", "photo_id": 72,
            "photo_delivery_confirmed": True, "attention_active": True,
        }
        provenance["camera_candidate_id"] = camera_candidate_id
        original_source = ConversationMessage(
            role="user", event_id="original-photo-event",
            content=[ref.model_copy(update={"source_provenance": provenance})],
        )
        original_provenance = dict(original_source.content[0].source_provenance)
        engine = pool._test_bundle.engine
        engine.messages = [original_source]
        pool._test_bundle.tool_registry = ToolRegistry()

        def register_image_tool(bundle, *, on_attachment_load_started=None,
                                on_attachment_loaded=None, on_source_selected=None,
                                on_source_selection_started=None,
                                **_kwargs):
            bundle.tool_registry.register(LoadConversationImageTool(
                store,
                is_attachment_allowed=lambda attachment_id: pool._conversation_attachment_allowed(
                    bundle, attachment_id
                ),
                on_load_started=on_attachment_load_started,
                on_loaded=on_attachment_loaded,
                on_source_selected=on_source_selected,
                on_source_selection_started=on_source_selection_started,
            ))

        pool._register_conversation_image_tool = register_image_tool

        async def submit_correction(user_message, *, wellness_actor=None):
            del wellness_actor
            engine.messages.append(user_message)
            recorder = engine.decision_trace_recorder
            message = pool._active_message
            recorder.trace_requirement_signals(message.content)
            tool = pool._test_bundle.tool_registry.get("load_conversation_image")
            loaded = await tool.execute(
                LoadConversationImageInput(
                    attachment_id=ref.attachment_id, select_as_nutrition_source=True,
                ),
                ToolExecutionContext(cwd=pool._workspace),
            )
            assert loaded.is_error is False
            recorder.record(TRACE_FINALIZATION, {
                "schema_version": 1, "trace_event_id": "late-correction-trace",
                "annotations": {"nutrition": {
                    "schema_version": 2, "record_type": "meal_correction",
                    "changed_fields": ["meal_date"], "meal_date": "2026-10-01",
                }},
            })
            yield AssistantTextDelta(text="Исправила дату записи.")

        engine.submit_message = submit_correction
        late = InboundMessage(
            channel="telegram", sender_id="123", chat_id="123",
            content="Please correct the date on that old meal.",
            metadata={"message_id": "late-correction-901", "is_group": False},
            timestamp=datetime(2026, 10, 9, 9, 0, tzinfo=timezone.utc),
        )
        result = await _turn(pool, late, ingress)
        assert len(honcho.messages) == 2
        stored = honcho.messages[1].metadata
        assert stored["gateway_session_id"] == "runtime-session-current"
        selected = stored["selected_source"]
        assert selected["gateway_session_id"] == "runtime-session-original"
        assert selected["source_message_id"] == "original-meal-append"
        assert selected["original_receipt_event_id"] == "original-assistant-event"
        assert stored["target_meal_id"]
        assert stored["ingest_source"] == "dropbox_camera"
        assert result.metadata["nutrition_append_event_id"] == honcho.messages[1].id
        assert original_source.content[0].source_provenance == original_provenance
        assert ingress._attempts[camera_candidate_id]["attention_active"] is False
        assert ingress._attempts[newer_candidate_id]["attention_active"] is True
    finally:
        await ingress.close()


@pytest.mark.asyncio
async def test_ordinary_meal_append_reconciles_timeout_and_never_claims_unknown(
    tmp_path, monkeypatch
):
    ingress, _root, _bus, _ = _ingress(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    honcho.timeout_after_append = True
    _script_finalization(pool, _consumed_payload())
    message = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Я съела два кусочка, запиши завтрак",
        metadata={"message_id": 811, "is_group": False, "_synthetic": True},
    )
    result = await _turn(pool, message, ingress)
    assert result.text.startswith(honcho.messages[1].content)
    assert "Записано; баланс обновляется." in result.text
    assert result.metadata["nutrition_append_event_id"] == "honcho-2"
    assert len(honcho.messages) == 2

    honcho2 = _Honcho()
    honcho2.fail_before_append = True
    pool2 = _pool(tmp_path / "failed", ingress, honcho2, monkeypatch)
    _script_finalization(pool2, _consumed_payload())
    _script_finalization(pool2, _consumed_payload())
    failed_message = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Я съела два кусочка, запиши завтрак",
        metadata={"message_id": 812, "is_group": False, "_synthetic": True},
    )
    pool2._active_message = failed_message
    failed_updates = []
    with pytest.raises(OSError, match="synthetic append unavailable"):
        async for update in pool2.stream_message(failed_message, ingress.config.session_key):
            failed_updates.append(update)
    assert not any(update.kind == "final" for update in failed_updates)
    assert all("Записано" not in update.text for update in failed_updates)
    assert honcho2.messages == []

    honcho3 = _Honcho()
    honcho3.wrong_assistant_operation = True
    pool3 = _pool(tmp_path / "mismatched", ingress, honcho3, monkeypatch)
    _script_finalization(pool3, _consumed_payload())
    _script_finalization(pool3, _consumed_payload())
    mismatched_message = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Я съела два кусочка, запиши завтрак",
        metadata={"message_id": 813, "is_group": False, "_synthetic": True},
    )
    pool3._active_message = mismatched_message
    mismatched_updates = []
    with pytest.raises(Exception, match="(?i)(receipt|reconcil|operation)"):
        async for update in pool3.stream_message(mismatched_message, ingress.config.session_key):
            mismatched_updates.append(update)
    assert not any(update.kind == "final" for update in mismatched_updates)
    assert all("Записано" not in update.text for update in mismatched_updates)
    assert len(honcho3.messages) == 2
    assert honcho3.messages[1].metadata["client_op_id"] == "synthetic-wrong-operation"
    await ingress.close()


@pytest.mark.asyncio
async def test_ordinary_nonfood_reply_without_finalizer_creates_no_nutrition_annotation(
    tmp_path, monkeypatch,
):
    ingress, _root, _bus, _ = _ingress(tmp_path)
    try:
        honcho = _Honcho()
        pool = _pool(tmp_path, ingress, honcho, monkeypatch)
        _script_finalization(pool, None, "That is a plastic bowl, not food.")
        message = InboundMessage(
            channel="telegram", sender_id="123", chat_id="123",
            content="Can you identify this object?",
            metadata={"message_id": "nonfood-301", "is_group": False},
        )
        result = await _turn(pool, message, ingress)
        await asyncio.gather(*pool._shadow_backend_for_scope(None)._pending)
        assert result.text == "That is a plastic bowl, not food."
        assert len(honcho.messages) == 2
        assistant_metadata = honcho.messages[1].metadata
        assert "decision_trace" not in assistant_metadata
        assert assistant_metadata["nutrition_annotation_status"] == "not_applicable"
    finally:
        await ingress.close()


@pytest.mark.asyncio
async def test_structured_record_is_not_vetoed_by_old_uncertainty_words(tmp_path, monkeypatch):
    ingress, _root, _bus, _ = _ingress(tmp_path)
    try:
        honcho = _Honcho()
        pool = _pool(tmp_path, ingress, honcho, monkeypatch)
        _script_finalization(pool, _consumed_payload())
        message = InboundMessage(
            channel="telegram", sender_id="123", chat_id="123",
            content="I am unsure and this is only an estimate, please log it.",
            metadata={"message_id": "structured-record-302", "is_group": False},
        )
        result = await _turn(pool, message, ingress)
        assert result.metadata["nutrition_append_event_id"] == honcho.messages[1].id
        assert honcho.messages[1].metadata["decision_trace"]["annotations"]["nutrition"][
            "record_type"
        ] == "meal_observation"
    finally:
        await ingress.close()

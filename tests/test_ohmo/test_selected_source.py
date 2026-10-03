from datetime import datetime, timezone
from types import SimpleNamespace

from openharness.channels.bus.events import InboundMessage
from openharness.engine.messages import AttachmentRefBlock, ConversationMessage
from ohmo.evals.nutrition_persistence import derive_meal_id
from ohmo.evals.nutrition_trace import NutritionAnnotationV2
from ohmo.gateway.runtime import (
    _SELECTED_SOURCE_AUTHORITY,
    _event_id_for_inbound_message,
    _build_conversation_turn_metadata,
    _committed_nutrition_reply,
)
from ohmo.gateway.selected_source import resolve_selected_photo_source
from ohmo.gateway.turn_context import TurnContext
from ohmo.gateway.memory_gate import MemoryScope


def _history_ref(*, source_id: str, append_id: str | None, event_id: str = "source-event"):
    provenance = {
        "schema_version": 1,
        "channel": "telegram",
        "principal": "telegram:123",
        "chat_id": "123",
        "session_key": "telegram:123",
        "gateway_session_id": "gateway-session",
        "received_at": "2026-10-01T20:59:58+00:00",
        "timestamp_authority": "inbound_event_timestamp",
        "is_group": False,
        "is_forwarded": False,
        "source_message_id": source_id,
    }
    if append_id is not None:
        provenance["append_source_message_id"] = append_id
    return ConversationMessage(
        role="user",
        event_id=event_id,
        content=[AttachmentRefBlock(
            attachment_id="a" * 64, media_type="image/jpeg", byte_size=4,
            source_provenance=provenance,
        )],
    )


def test_loaded_photo_binds_verified_append_identity_not_coalesced_photo_id():
    inbound = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content="correct this",
        timestamp=datetime(2026, 10, 2, tzinfo=timezone.utc),
        metadata={"is_group": False, "message_id": "correction-message"},
    )
    turn = TurnContext(
        principal="123", is_owner=True, is_private=True, channel="telegram",
        chat_id="123", session_id="gateway-session",
    )
    binding = resolve_selected_photo_source(
        attachment_id="a" * 64,
        history=[_history_ref(source_id="photo-message", append_id="burst-append")],
        message=inbound,
        turn_ctx=turn,
        session_key="telegram:123",
        gateway_session_id="gateway-session",
        tenant_id="tenant-123",
        authorized_participant=True,
    )
    assert binding is not None
    assert binding["source_message_id"] == "photo-message"
    assert binding["append_source_message_id"] == "burst-append"

    annotation = NutritionAnnotationV2.model_validate({
        "schema_version": 2,
        "record_type": "meal_correction",
        "changed_fields": ["meal_date"],
        "meal_date": "2026-10-01",
    })
    recorder = SimpleNamespace(
        episode_id="episode-synthetic",
        decision_trace_status="recorded",
        nutrition_annotation_status="recorded",
        decision_trace_envelope=None,
        validated_nutrition_envelope=annotation.model_dump(
            mode="json", exclude_unset=True
        ),
    )
    inbound.metadata["_selected_source_binding"] = (
        _SELECTED_SOURCE_AUTHORITY, binding
    )
    _, user_metadata, _ = _build_conversation_turn_metadata(
        turn_ctx=turn, message=inbound,
        scope=MemoryScope(private_tenant="tenant-123", shared_tenants=()),
        recorder=recorder,
    )
    assert user_metadata["selected_source"]["source_message_id"] == "photo-message"
    assert user_metadata["selected_source"]["append_source_message_id"] == "burst-append"
    assert user_metadata["target_meal_id"] == derive_meal_id(
        tenant_id="tenant-123", source_principal="telegram:123",
        gateway_session_id="gateway-session", source_message_id="burst-append",
    )


def test_selected_photo_binding_rejects_ambiguous_and_unmapped_legacy_sources():
    inbound = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content="correct this",
        metadata={"is_group": False},
    )
    turn = TurnContext(
        principal="123", is_owner=True, is_private=True, channel="telegram",
        chat_id="123", session_id="gateway-session",
    )
    common = dict(
        attachment_id="a" * 64, message=inbound, turn_ctx=turn,
        session_key="telegram:123", gateway_session_id="gateway-session",
        tenant_id="tenant-123", authorized_participant=True,
    )
    original = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content="",
        metadata={"is_group": False, "message_id": "legacy-photo"},
    )
    legacy_id = _event_id_for_inbound_message(original)
    legacy_binding = resolve_selected_photo_source(
        **common,
        history=[_history_ref(
            source_id="legacy-photo", append_id=None, event_id=legacy_id,
        )],
    )
    assert legacy_binding is not None
    assert legacy_binding["append_source_message_id"] == "legacy-photo"
    assert resolve_selected_photo_source(
        **common,
        history=[
            _history_ref(source_id="photo-1", append_id="append-1"),
            _history_ref(source_id="photo-2", append_id="append-2"),
        ],
    ) is None
    assert resolve_selected_photo_source(
        **common,
        history=[_history_ref(source_id="old-photo", append_id=None)],
    ) is None


def test_receipt_proven_consumed_occurrence_selects_unique_append_and_rejects_multiple():
    inbound = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content="correct this",
        metadata={"is_group": False},
    )
    turn = TurnContext(
        principal="123", is_owner=True, is_private=True, channel="telegram",
        chat_id="123", session_id="gateway-session",
    )
    ref = _history_ref(source_id="uncertain-photo", append_id=None)
    ref.content[0].source_provenance["consumed_occurrences"] = [{
        "append_source_message_id": "portion-answer",
        "receipt_event_id": "honcho-event-synthetic",
        "client_op_id": "turn-synthetic:assistant",
    }]
    args = dict(
        attachment_id="a" * 64, history=[ref], message=inbound, turn_ctx=turn,
        session_key="telegram:123", gateway_session_id="gateway-session",
        tenant_id="tenant-123", authorized_participant=True,
    )
    binding = resolve_selected_photo_source(**args)
    assert binding is not None
    assert binding["source_message_id"] == "uncertain-photo"
    assert binding["append_source_message_id"] == "portion-answer"

    ref.content[0].source_provenance["consumed_occurrences"].append({
        "append_source_message_id": "another-portion-answer",
        "receipt_event_id": "another-honcho-event",
        "client_op_id": "another-turn:assistant",
    })
    assert resolve_selected_photo_source(**args) is None

    foreign = TurnContext(
        principal="456", is_owner=False, is_private=True, channel="telegram",
        chat_id="456", session_id="gateway-session",
    )
    assert resolve_selected_photo_source(**{**args, "turn_ctx": foreign}) is None


def test_committed_reply_uses_receipt_nutrients_and_plain_correction_status():
    correction = NutritionAnnotationV2(
        record_type="meal_correction", changed_fields=["meal_date"],
        meal_date="2026-10-02",
    )
    assert _committed_nutrition_reply(
        correction, status="Изменение сохранено; баланс обновляется."
    ) == "Изменение сохранено; баланс обновляется."

    observation = NutritionAnnotationV2(
        record_type="meal_observation", consumption_status="consumed",
        energy_kcal_best=750,
    )
    assert _committed_nutrition_reply(
        observation, status="Записано; приём пищи пока не привязан к дате. Баланс обновляется."
    ) == "Примерно 750 ккал.\nЗаписано; приём пищи пока не привязан к дате. Баланс обновляется."

"""Focused regressions for a completed native Camera option replay."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from openharness.channels.bus.events import InboundMessage, OutboundDeliveryReceipt, OutboundMessage

from ohmo.gateway.camera import (
    CAMERA_AUTHORITY, CameraIngress, _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY,
)
from ohmo.evals.nutrition_persistence import derive_meal_id
from ohmo.memory_backend import ShadowMemoryBackend
from tests.camera_e2e_probe.camera_runtime_support import _validate_completed_photo_replay
from tests.test_ohmo.test_memory_backend import _Base, _Honcho, _metadata
from tests.test_ohmo.test_camera_ingress import (
    _admit,
    _candidate,
    _ingress,
    _native_callback,
    _observed_camera_meal_receipt,
)
from tests.test_ohmo.test_nutrition_dialogue_stream import (
    FakeTelegram,
    _Honcho as RuntimeHoncho,
    _pool as runtime_pool,
    _turn as runtime_turn,
)


async def _save_callback_portion(tmp_path, monkeypatch):
    ingress, root, bus, _ = _ingress(tmp_path, FakeTelegram())
    request = _candidate(root)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    await bus.consume_inbound()
    honcho = RuntimeHoncho()
    pool = runtime_pool(tmp_path, ingress, honcho, monkeypatch)
    photo_id = ingress._attempts[request["candidate_id"]]["photo_id"]

    first = await _native_callback(
        bus, label="Да, я съела только часть", target=photo_id,
        options=["Да, я съела только часть", "Нет, не ела"],
        prompt="Съели ли вы это? Фото сделано 2026-10-01.",
    )
    first.metadata["callback_query_id"] = "callback-1"
    ingress.process_real_inbound(first)
    assert first.metadata["_camera_answer"] == "yes"
    first_final = await runtime_turn(pool, first, ingress)
    assert "Сколько" in first_final.text
    assert ingress._attempts[request["candidate_id"]]["state"] == "clarifying"

    clarification_id = ingress._attempts[request["candidate_id"]]["reply_ids"][-1]
    portion = await _native_callback(
        bus, label="2 кусочка", target=clarification_id,
        options=["1 кусочек", "2 кусочка", "Половину порции"],
        prompt="Вы съели это? Сколько примерно вы съели?",
    )
    portion.metadata["callback_query_id"] = "callback-portion-1"
    ingress.process_real_inbound(portion)
    assert portion.metadata["_camera_answer"] == "yes"
    saved_final = await runtime_turn(pool, portion, ingress)
    assert saved_final.metadata["nutrition_append_event_id"] == "honcho-4"
    assert ingress._attempts[request["candidate_id"]]["camera_commit"]["event_id"] == "honcho-4"
    assert len(honcho.messages) == 4
    return ingress, bus, request, pool, honcho, portion, saved_final


async def _save_typed_confirmation(
    tmp_path, monkeypatch, label="Да, я это съела", *, reply_to_photo=True,
):
    ingress, root, bus, _ = _ingress(tmp_path, FakeTelegram())
    request = _candidate(root)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    await bus.consume_inbound()
    honcho = RuntimeHoncho()
    pool = runtime_pool(tmp_path, ingress, honcho, monkeypatch)
    photo_id = ingress._attempts[request["candidate_id"]]["photo_id"]
    answer = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=label,
        metadata={"message_id": "typed-camera-first-confirmation",
                  **({"reply_to_message_id": str(photo_id)} if reply_to_photo else {}),
                  "_telegram_raw_text": label, "is_group": False,
                  "chat_type": "private"},
    )
    ingress.process_real_inbound(answer)
    assert answer.metadata.get("callback_query") is None
    assert answer.metadata["_camera_answer"] == "yes"
    saved_final = await runtime_turn(pool, answer, ingress)
    assert saved_final.metadata["nutrition_append_event_id"] == "honcho-2"
    assert len(honcho.messages) == 2
    return ingress, request, pool, honcho, saved_final, answer


@pytest.mark.asyncio
async def test_whole_portion_native_selection_commits_once_with_capture_time(
    tmp_path, monkeypatch,
):
    ingress, root, bus, _ = _ingress(tmp_path, FakeTelegram())
    request = _candidate(root)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    await bus.consume_inbound()
    attempt = ingress._attempts[request["candidate_id"]]
    capture = ingress._attempt_capture_time(attempt)
    honcho = RuntimeHoncho()
    pool = runtime_pool(tmp_path, ingress, honcho, monkeypatch)
    selected = await _native_callback(
        bus, label="Всю порцию", target=attempt["photo_id"],
        options=["Всю порцию", "Нет, не ела"],
        prompt="Сколько риса вы съели?",
    )
    selected.metadata["callback_query_id"] = "synthetic-whole-portion"
    ingress.process_real_inbound(selected)
    assert selected.metadata["_camera_answer"] == "yes"
    saved = await runtime_turn(pool, selected, ingress)
    assert saved.metadata["nutrition_append_event_id"] == "honcho-2"
    assert len(honcho.messages) == 2
    assert datetime.fromisoformat(
        honcho.messages[-1].metadata["decision_trace"]["annotations"]["nutrition"]["meal_at"]
    ) == capture
    assert ingress._attempts[request["candidate_id"]]["camera_commit"]["event_id"] == "honcho-2"
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "reply"),
    [
        ("И еще добавь одно куриное яйцо", False),
    ],
)
async def test_same_meal_correction_gets_only_unique_camera_source_target(
    tmp_path, monkeypatch, text, reply,
):
    ingress, bus, request, _pool, _honcho, _saved, _final = await _save_callback_portion(
        tmp_path, monkeypatch
    )
    attempt = ingress._attempts[request["candidate_id"]]
    metadata = {
        "message_id": "synthetic-same-meal-correction",
        "_telegram_raw_text": text,
        "is_group": False,
        "chat_type": "private",
    }
    if reply:
        metadata["reply_to_message_id"] = str(attempt["photo_id"])
    correction = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=text,
        metadata=metadata,
    )
    ingress.process_real_inbound(correction)
    assert correction.metadata["_camera_context_meal_target"] is _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY, (
        attempt.get("confirmed_camera_context"), attempt.get("camera_commit"), correction.metadata
    )
    assert correction.metadata["_camera_context_meal_candidate_id"] == request["candidate_id"]
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("target_kind", ["saved_ack", "photo", "no_reply"])
async def test_saved_acknowledgement_binds_label_only_package_correction(
    tmp_path, monkeypatch, target_kind,
):
    from tests.test_ohmo.test_nutrition_dialogue_review_regressions import _ScriptedEngine, _trace

    ingress, _bus, request, pool, honcho, _saved, _final = await _save_callback_portion(
        tmp_path, monkeypatch
    )
    attempt = ingress._attempts[request["candidate_id"]]
    # The final saved acknowledgement is delivered through the ordinary outbound
    # receipt path and must be present in the attempt's retained reply IDs.
    saved_ack_id = str(attempt["reply_ids"][-1])
    text = "125г в упаковке, в 100г 85,4 ккал"
    reply_target = (
        saved_ack_id if target_kind == "saved_ack"
        else str(attempt["photo_id"]) if target_kind == "photo"
        else None
    )
    correction_metadata = {
        "message_id": "synthetic-package-label-correction",
        "_telegram_raw_text": text, "is_group": False,
        "chat_type": "private",
    }
    if reply_target is not None:
        correction_metadata["reply_to_message_id"] = reply_target
    correction = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=text,
        metadata=correction_metadata,
    )
    ingress.process_real_inbound(correction)
    assert correction.metadata.get("_camera_context_meal_target") is _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY
    assert correction.metadata.get("_camera_context_meal_candidate_id") == request["candidate_id"]
    assert correction.metadata.get("_camera_context_meal_target_kind") == "package_label"

    original = honcho.messages[-1].metadata["decision_trace"]["annotations"]["nutrition"]
    original_event_id = attempt["camera_commit"]["event_id"]
    original_date = original["meal_at"]
    original_items = list(original["items"])
    corrected_items = [dict(item) for item in original_items]
    corrected_items[0].update({"quantity_text": "125 г", "energy_kcal_best": 106.75})
    trace = _trace({
        "schema_version": 2, "record_type": "meal_correction",
        "changed_fields": ["items", "energy_kcal_min", "energy_kcal_max", "energy_kcal_best"],
        "items": corrected_items, "energy_kcal_min": 106.75,
        "energy_kcal_max": 106.75, "energy_kcal_best": 106.75,
    })
    scripted = _ScriptedEngine(pool, [(trace, "Исправила калорийность по упаковке.")])
    scripted.messages = pool._test_bundle.engine.messages
    pool._test_bundle.engine = scripted
    corrected = await runtime_turn(pool, correction, ingress)
    assert corrected.metadata["nutrition_append_event_id"] != original_event_id
    assert "meal_at" not in corrected.metadata["nutrition_committed_annotation"]
    assert corrected.metadata["nutrition_committed_annotation"]["energy_kcal_best"] == 106.75
    assert len(corrected.metadata["nutrition_committed_annotation"]["items"]) == len(original_items)
    from ohmo.evals.nutrition_persistence import _event_from_raw, _fold_events

    original_message = next(message for message in honcho.messages if message.id == original_event_id)
    assert original_message.metadata["decision_trace"]["annotations"]["nutrition"] == original
    rows = []
    for message in (original_message, honcho.messages[-1]):
        row = _event_from_raw({
            "id": message.id, "peer_id": message.peer_id, "session_id": message.session_id,
            "workspace_id": message.workspace_id, "created_at": message.created_at.isoformat(),
            "metadata": dict(message.metadata),
        })
        row["_created_at"] = message.created_at
        rows.append(row)
    current = _fold_events(rows, "UTC")
    assert datetime.fromisoformat(current["meal_at"]) == datetime.fromisoformat(original_date)
    assert current["energy_kcal_best"] == 106.75
    assert [item["name"] for item in current["items"]] == [
        item["name"] for item in corrected_items
    ]
    selected = honcho.messages[-1].metadata["selected_source"]
    assert set(selected) == {
        "schema_version", "tenant_id", "source_principal", "gateway_session_id",
        "source_message_id", "append_source_message_id", "is_private", "is_forwarded",
        "is_group", "original_receipt_event_id", "original_receipt_client_op_id",
    }
    assert selected["schema_version"] == 2
    assert selected["gateway_session_id"] == original_message.metadata["gateway_session_id"]
    assert selected["source_message_id"] == original_message.metadata["source_message_id"]
    assert selected["original_receipt_event_id"] == original_event_id
    assert selected["original_receipt_client_op_id"] == original_message.metadata["client_op_id"]

    replay_metadata = {
        "message_id": "synthetic-package-label-correction",
        "_telegram_raw_text": text, "is_group": False,
        "chat_type": "private",
    }
    if reply_target is not None:
        replay_metadata["reply_to_message_id"] = reply_target
    replay = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=text,
        metadata=replay_metadata,
    )
    ingress.process_real_inbound(replay)
    pool._active_message = replay
    replay_updates = [
        update async for update in pool.stream_message(replay, ingress.config.session_key)
    ]
    assert not any(update.kind == "final" for update in replay_updates)
    assert len(honcho.messages) == 6
    assert ingress._attempts[request["candidate_id"]]["context_meal_projection"]["event_id"] == corrected.metadata[
        "nutrition_append_event_id"
    ]
    await ingress.close()


@pytest.mark.asyncio
async def test_package_label_correction_rejects_meal_date_mutation_before_append(
    tmp_path, monkeypatch,
):
    from tests.test_ohmo.test_nutrition_dialogue_review_regressions import _ScriptedEngine, _trace

    ingress, _bus, request, pool, honcho, _saved, _final = await _save_callback_portion(
        tmp_path, monkeypatch
    )
    attempt = ingress._attempts[request["candidate_id"]]
    text = "125г в упаковке, в 100г 85,4 ккал"
    label_reply = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=text,
        metadata={"message_id": "synthetic-label-with-date-mutation",
                  "reply_to_message_id": str(attempt["reply_ids"][-1]),
                  "_telegram_raw_text": text, "is_group": False,
                  "chat_type": "private"},
    )
    ingress.process_real_inbound(label_reply)
    assert label_reply.metadata.get("_camera_context_meal_target") is _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY
    items = [dict(item) for item in honcho.messages[-1].metadata[
        "decision_trace"]["annotations"]["nutrition"]["items"]]
    items[0].update({"quantity_text": "125 г", "energy_kcal_best": 106.75})
    trace = _trace({
        "schema_version": 2, "record_type": "meal_correction",
        "changed_fields": ["items", "energy_kcal_best", "meal_at"],
        "items": items, "energy_kcal_best": 106.75,
        "meal_at": "2026-10-08T08:00:00Z",
    })
    scripted = _ScriptedEngine(pool, [(trace, "Исправила калорийность по упаковке.")])
    scripted.messages = pool._test_bundle.engine.messages
    pool._test_bundle.engine = scripted
    before_count = len(honcho.messages)
    with pytest.raises(
        ValueError,
        match="Camera-context correction must patch the one retained consumed meal",
    ):
        await runtime_turn(pool, label_reply, ingress)
    assert len(honcho.messages) == before_count
    assert "context_meal_projection" not in attempt
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("target_kind", ["photo", "no_reply"])
async def test_package_label_correction_keeps_photo_and_unique_context_routes(
    tmp_path, monkeypatch, target_kind,
):
    ingress, _bus, request, _pool, _honcho, _saved, _final = await _save_callback_portion(
        tmp_path, monkeypatch
    )
    attempt = ingress._attempts[request["candidate_id"]]
    text = "125г в упаковке, в 100г 85,4 ккал"
    metadata = {"message_id": f"synthetic-label-{target_kind}",
                "_telegram_raw_text": text, "is_group": False,
                "chat_type": "private"}
    if target_kind == "photo":
        metadata["reply_to_message_id"] = str(attempt["photo_id"])
    message = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=text,
        metadata=metadata,
    )
    ingress.process_real_inbound(message)
    assert message.metadata.get("_camera_context_meal_target") is _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY
    assert message.metadata.get("_camera_context_meal_candidate_id") == request["candidate_id"]
    assert message.metadata.get("_camera_context_meal_target_kind") == "package_label"
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("control", [
    "unknown_ack", "foreign_owner", "group", "forwarded",
    "new_meal", "nonmeal_reminder", "explicit_denial", "intervening_media",
    "ambiguous", "no_reply_ambiguous",
    "no_reply_interrupted", "no_reply_active_attention",
])
async def test_package_label_correction_rejects_untrusted_targets_and_sources(
    tmp_path, monkeypatch, control,
):
    ingress, _bus, request, _pool, _honcho, _saved, _final = await _save_callback_portion(
        tmp_path, monkeypatch
    )
    attempt = ingress._attempts[request["candidate_id"]]
    text = "125г в упаковке, в 100г 85,4 ккал"
    target = str(attempt["reply_ids"][-1])
    sender = "123"
    metadata = {"message_id": f"synthetic-label-control-{control}",
                "reply_to_message_id": target, "_telegram_raw_text": text,
                "is_group": False, "chat_type": "private"}
    media = []
    if control == "unknown_ack":
        metadata["reply_to_message_id"] = "unknown-acknowledgement"
    elif control == "foreign_owner":
        sender = "foreign-owner"
    elif control == "group":
        metadata["is_group"] = True
    elif control == "forwarded":
        metadata["is_forwarded"] = True
    elif control == "new_meal":
        text = "Добавь новый прием пищи: 125г в упаковке, в 100г 85,4 ккал"
        metadata["_telegram_raw_text"] = text
    elif control == "nonmeal_reminder":
        text = "Добавь напоминание купить творог: 125г в упаковке, в 100г 85,4 ккал"
        metadata["_telegram_raw_text"] = text
    elif control == "explicit_denial":
        text = "Я не ела это. 125г в упаковке, в 100г 85,4 ккал"
        metadata["reply_to_message_id"] = str(attempt["photo_id"])
        metadata["_telegram_raw_text"] = text
    elif control == "intervening_media":
        media = [attempt["snapshot"]]
    elif control in {"ambiguous", "no_reply_ambiguous"}:
        if control.startswith("no_reply"):
            metadata.pop("reply_to_message_id")
        ingress._attempts["synthetic-duplicate-candidate"] = dict(attempt)
    elif control == "no_reply_interrupted":
        metadata.pop("reply_to_message_id")
        attempt["context_interrupted"] = True
    elif control == "no_reply_active_attention":
        metadata.pop("reply_to_message_id")
        active = dict(attempt)
        active.update({"state": "photo_sent", "attention_active": True})
        ingress._attempts["synthetic-active-context"] = active
    message = InboundMessage(
        channel="telegram", sender_id=sender, chat_id="123", content=text,
        media=media, metadata=metadata,
    )
    ingress.process_real_inbound(message)
    assert message.metadata.get("_camera_context_meal_target") is not _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY
    assert message.metadata.get("_selected_source_binding") is None
    if control == "explicit_denial":
        assert message.metadata.get("_camera_correction") is CAMERA_AUTHORITY
        assert message.metadata.get("_camera_answer") == "no"
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Съели утром", "Съели сегодня"])
async def test_sparse_morning_or_today_phrase_selects_one_committed_camera_meal(
    tmp_path, monkeypatch, text,
):
    ingress, _bus, request, _pool, _honcho, _saved, _final = await _save_callback_portion(
        tmp_path, monkeypatch
    )
    message = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=text,
        metadata={"message_id": "synthetic-sparse-morning", "_telegram_raw_text": text,
                  "is_group": False, "chat_type": "private"},
    )
    ingress.process_real_inbound(message)
    assert message.metadata.get("_camera_context_meal_target") is _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY
    assert message.metadata.get("_camera_context_meal_candidate_id") == request["candidate_id"]
    assert message.metadata.get("_camera_context_meal_target_kind") == "dated_time"
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "Добавь встречу в календарь", "Добавь задачу купить яйцо",
    "Добавь напоминание купить яйцо", "Добавь новый прием пищи: рис сегодня",
    "Добавь яйцо в список покупок",
])
async def test_unrelated_add_task_or_calendar_does_not_select_camera_meal(
    tmp_path, monkeypatch, text,
):
    ingress, _bus, _request, _pool, _honcho, _saved, _final = await _save_callback_portion(
        tmp_path, monkeypatch
    )
    message = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=text,
        metadata={"message_id": "synthetic-unrelated-add", "_telegram_raw_text": text,
                  "is_group": False, "chat_type": "private"},
    )
    ingress.process_real_inbound(message)
    assert message.metadata.get("_camera_context_meal_target") is None
    assert message.metadata.get("_camera_context_meal_candidate_id") is None
    await ingress.close()


@pytest.mark.asyncio
async def test_unlisted_food_addition_offers_context_but_ordinary_reply_stays_ordinary(
    tmp_path, monkeypatch,
):
    from tests.test_ohmo.test_nutrition_dialogue_review_regressions import _ScriptedEngine

    ingress, _bus, request, pool, honcho, _saved, _final = await _save_callback_portion(
        tmp_path, monkeypatch
    )
    text = "И еще добавь хумус"
    addition = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=text,
        metadata={"message_id": "synthetic-unlisted-food-addition",
                  "_telegram_raw_text": text, "is_group": False, "chat_type": "private"},
    )
    ingress.process_real_inbound(addition)
    assert addition.metadata.get("_camera_context_meal_target") is _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY
    scripted = _ScriptedEngine(pool, [(None, "Поняла, добавлю хумус в список покупок.")])
    scripted.messages = pool._test_bundle.engine.messages
    pool._test_bundle.engine = scripted
    result = await runtime_turn(pool, addition, ingress)
    assert result.text == "Поняла, добавлю хумус в список покупок."
    assert "nutrition_append_event_id" not in result.metadata
    assert "context_meal_projection" not in ingress._attempts[request["candidate_id"]]
    await ingress.close()


@pytest.mark.asyncio
async def test_addition_with_package_details_keeps_existing_photo_reply_route(
    tmp_path, monkeypatch,
):
    ingress, _bus, request, _pool, _honcho, _saved, _final = await _save_callback_portion(
        tmp_path, monkeypatch
    )
    attempt = ingress._attempts[request["candidate_id"]]
    text = "Добавь яйцо; творог 125г в упаковке, в 100г 85,4 ккал"
    addition = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=text,
        metadata={"message_id": "synthetic-addition-with-label-details",
                  "reply_to_message_id": str(attempt["photo_id"]),
                  "_telegram_raw_text": text, "is_group": False,
                  "chat_type": "private"},
    )
    ingress.process_real_inbound(addition)
    assert addition.metadata.get("_camera_context_meal_target") is _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY
    assert addition.metadata.get("_camera_context_meal_target_kind") == "item_addition"
    await ingress.close()


@pytest.mark.asyncio
async def test_contextual_addition_rejects_new_meal_observation(tmp_path, monkeypatch):
    from tests.test_ohmo.test_nutrition_dialogue_review_regressions import _ScriptedEngine, _trace

    ingress, _bus, request, pool, honcho, _saved, _final = await _save_callback_portion(
        tmp_path, monkeypatch
    )
    text = "И еще добавь хумус"
    addition = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=text,
        metadata={"message_id": "synthetic-unlisted-food-observation",
                  "_telegram_raw_text": text, "is_group": False, "chat_type": "private"},
    )
    ingress.process_real_inbound(addition)
    observation = _trace({
        "schema_version": 2, "record_type": "meal_observation",
        "consumption_status": "consumed", "basis": ["owner_statement"],
        "energy_kcal_best": 200,
    })
    scripted = _ScriptedEngine(pool, [(observation, "Записала хумус отдельным приёмом пищи.")])
    scripted.messages = pool._test_bundle.engine.messages
    pool._test_bundle.engine = scripted
    with pytest.raises(ValueError, match="correct the retained meal"):
        await runtime_turn(pool, addition, ingress)
    assert len(honcho.messages) == 4
    assert "context_meal_projection" not in ingress._attempts[request["candidate_id"]]
    await ingress.close()


@pytest.mark.asyncio
async def test_contextual_egg_and_dated_meal_corrections_are_durable_and_replay_stable(
    tmp_path, monkeypatch,
):
    from tests.test_ohmo.test_nutrition_dialogue_review_regressions import _ScriptedEngine, _trace
    from ohmo.evals.nutrition_persistence import _event_from_raw, _fold_events

    ingress, _bus, request, pool, honcho, _, _ = await _save_callback_portion(
        tmp_path, monkeypatch
    )
    original = honcho.messages[-1].metadata["decision_trace"]["annotations"]["nutrition"]
    original_snapshot = json.loads(json.dumps(original))
    original_event_id = ingress._attempts[request["candidate_id"]]["camera_commit"]["event_id"]
    original_items = list(original["items"])
    with_egg = original_items + [{"name": "куриное яйцо", "quantity_text": "1 штука"}]
    addition_trace = _trace({
        "schema_version": 2, "record_type": "meal_correction", "changed_fields": [
            "items", "energy_kcal_min", "energy_kcal_max", "energy_kcal_best",
        ],
        "items": with_egg, "energy_kcal_min": 270, "energy_kcal_max": 300,
        "energy_kcal_best": 285,
    })
    scripted = _ScriptedEngine(pool, [(addition_trace, "Добавила яйцо к той же порции.")])
    scripted.messages = pool._test_bundle.engine.messages
    pool._test_bundle.engine = scripted
    addition = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="И еще добавь одно куриное яйцо",
        metadata={"message_id": "synthetic-egg-addition", "_telegram_raw_text":
                  "И еще добавь одно куриное яйцо", "is_group": False, "chat_type": "private"},
    )
    ingress.process_real_inbound(addition)
    first = await runtime_turn(pool, addition, ingress)
    assert "context_meal_projection" in ingress._attempts[request["candidate_id"]]
    assert first.metadata["nutrition_append_event_id"] != original_event_id
    assert honcho.messages[-1].metadata["reply_to_source_message_id"] is None
    original_message = next(message for message in honcho.messages if message.id == original_event_id)
    assert honcho.messages[-1].metadata["target_meal_id"] == derive_meal_id(
        tenant_id=ingress.config.tenant_id,
        source_principal=f"telegram:{ingress.config.principal}",
        gateway_session_id=pool._test_bundle.session_id,
        source_message_id=original_message.metadata["source_message_id"],
    )
    assert original_message.metadata["decision_trace"]["annotations"]["nutrition"] == original_snapshot
    correction_event = _event_from_raw({
        "id": honcho.messages[-1].id, "peer_id": honcho.messages[-1].peer_id,
        "session_id": honcho.messages[-1].session_id,
        "workspace_id": honcho.messages[-1].workspace_id,
        "created_at": honcho.messages[-1].created_at.isoformat(),
        "metadata": dict(honcho.messages[-1].metadata),
    })
    original_event = _event_from_raw({
        "id": original_message.id, "peer_id": original_message.peer_id,
        "session_id": original_message.session_id,
        "workspace_id": original_message.workspace_id,
        "created_at": original_message.created_at.isoformat(),
        "metadata": dict(original_message.metadata),
    })
    correction_event["_created_at"] = honcho.messages[-1].created_at
    original_event["_created_at"] = original_message.created_at
    folded = _fold_events([original_event, correction_event], "UTC")
    assert [item["name"] for item in correction_event["annotation"]["items"]] == [
        item["name"] for item in with_egg
    ]
    assert folded["energy_kcal_best"] == 285

    original_session = pool._test_bundle.session_id
    rotated_session = "synthetic-camera-session-after-restart"
    pool._test_bundle.session_id = rotated_session
    date_trace = _trace({
        "schema_version": 2, "record_type": "meal_correction",
        "changed_fields": ["meal_date"], "meal_date": "2026-10-08",
    })
    sparse_trace = _trace({
        "schema_version": 2, "record_type": "meal_correction",
        "changed_fields": ["meal_at"], "meal_at": "2026-10-08T08:00:00Z",
    })
    scripted = _ScriptedEngine(pool, [
        (date_trace, "Обновила состав и дату этой порции."),
        (sparse_trace, "Обновила время этой порции."),
        (sparse_trace, "Повторное исправление уже сохранено."),
    ])
    scripted.messages = pool._test_bundle.engine.messages
    pool._test_bundle.engine = scripted
    dated = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Рис и яйцо съели сегодня утром",
        metadata={"message_id": "synthetic-egg-date", "_telegram_raw_text":
                  "Рис и яйцо съели сегодня утром", "is_group": False, "chat_type": "private"},
    )
    ingress.process_real_inbound(dated)
    second = await runtime_turn(pool, dated, ingress)
    assert "nutrition_append_event_id" in second.metadata, (second.text, second.metadata)
    assert second.metadata["nutrition_append_event_id"] not in {
        original_event_id, first.metadata["nutrition_append_event_id"]
    }
    date_correction = honcho.messages[-1]
    assert pool._test_bundle.session_id == rotated_session
    assert date_correction.metadata["gateway_session_id"] == rotated_session
    assert date_correction.metadata["selected_source"]["gateway_session_id"] == original_session
    assert date_correction.metadata["selected_source"]["schema_version"] == 2
    assert date_correction.metadata["selected_source"]["original_receipt_event_id"] == original_event_id
    assert date_correction.metadata["selected_source"]["original_receipt_client_op_id"] == original_message.metadata["client_op_id"]
    assert date_correction.metadata["target_meal_id"] == derive_meal_id(
        tenant_id=ingress.config.tenant_id,
        source_principal=f"telegram:{ingress.config.principal}",
        gateway_session_id=original_session,
        source_message_id=original_message.metadata["source_message_id"],
    )
    rotated_prompt = pool._test_bundle.engine.system_prompt
    assert "куриное яйцо — 1 штука" in rotated_prompt
    assert "latest retained correction projection" in rotated_prompt
    assert original_event_id not in rotated_prompt
    assert "synthetic-egg-date" not in rotated_prompt
    assert "370" not in rotated_prompt

    sparse = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content="Съели утром",
        metadata={"message_id": "synthetic-sparse-morning-finalizer",
                  "_telegram_raw_text": "Съели утром", "is_group": False,
                  "chat_type": "private"},
    )
    ingress.process_real_inbound(sparse)
    assert sparse.metadata.get("_camera_context_meal_target") is _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY
    third = await runtime_turn(pool, sparse, ingress)
    assert third.metadata["nutrition_append_event_id"] not in {
        original_event_id, first.metadata["nutrition_append_event_id"],
        second.metadata["nutrition_append_event_id"],
    }
    assert honcho.messages[-1].metadata["reply_to_source_message_id"] is None
    latest = honcho.messages[-1].metadata["decision_trace"]["annotations"]["nutrition"]
    sparse_prompt = pool._test_bundle.engine.system_prompt
    assert "куриное яйцо — 1 штука" in sparse_prompt
    assert "Trusted owner-message time:" in sparse_prompt
    assert original_event_id not in sparse_prompt
    assert "synthetic-sparse-morning-finalizer" not in sparse_prompt
    assert re.search(r"\b370\b", sparse_prompt) is None
    assert latest["record_type"] == "meal_correction"
    assert latest["meal_at"] == "2026-10-08T08:00:00Z"
    assert "items" not in latest
    assert latest["changed_fields"] == ["meal_at"]
    stored_date = next(message for message in honcho.messages if message.id == second.metadata["nutrition_append_event_id"])
    assert stored_date.metadata["decision_trace"]["annotations"]["nutrition"]["meal_date"] == "2026-10-08"

    restarted = CameraIngress(
        ingress.config, workspace=tmp_path, bus=type(_bus)(), telegram=FakeTelegram()
    )
    replay = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Съели утром", timestamp=sparse.timestamp,
        metadata={"message_id": "synthetic-sparse-morning-finalizer", "_telegram_raw_text":
                  "Съели утром", "is_group": False, "chat_type": "private"},
    )
    restarted.process_real_inbound(replay)
    assert replay.metadata["_camera_context_meal_target"] is _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY
    replayed = await runtime_turn(pool, replay, restarted)
    assert replayed.metadata["nutrition_append_event_id"] == third.metadata[
        "nutrition_append_event_id"
    ]
    assert len(honcho.messages) == 10
    assert honcho.messages[-1].id == third.metadata["nutrition_append_event_id"]

    named_date_trace = _trace({
        "schema_version": 2, "record_type": "meal_correction",
        "changed_fields": ["meal_date"], "meal_date": "2026-10-08",
    })
    scripted = _ScriptedEngine(pool, [(named_date_trace, "Обновила дату порции с яйцом.")])
    scripted.messages = pool._test_bundle.engine.messages
    pool._test_bundle.engine = scripted
    named_followup = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Яйцо съела сегодня",
        timestamp=datetime.fromisoformat("2026-10-08T12:00:00+00:00"),
        metadata={"message_id": "synthetic-named-egg-after-rotation",
                  "_telegram_raw_text": "Яйцо съела сегодня", "is_group": False,
                  "chat_type": "private"},
    )
    restarted.process_real_inbound(named_followup)
    assert named_followup.metadata["_camera_context_meal_target"] is _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY
    named_result = await runtime_turn(pool, named_followup, restarted)
    third_receipt = next(message for message in honcho.messages
                         if message.id == third.metadata["nutrition_append_event_id"])
    assert named_result.metadata["nutrition_append_event_id"] not in {
        original_event_id, first.metadata["nutrition_append_event_id"],
        second.metadata["nutrition_append_event_id"], third.metadata["nutrition_append_event_id"],
    }
    assert honcho.messages[-1].metadata["target_meal_id"] == third_receipt.metadata["target_meal_id"]
    named_prompt = pool._test_bundle.engine.system_prompt
    assert "куриное яйцо — 1 штука" in named_prompt
    assert named_result.metadata["nutrition_committed_annotation"]["record_type"] == "meal_correction"
    assert len(honcho.messages) == 12
    assert original_event["annotation"]["items"] == original_snapshot["items"]
    await restarted.close()
    await ingress.close()


@pytest.mark.asyncio
async def test_contextual_followup_upgrades_exact_legacy_camera_journal_from_receipt(
    tmp_path, monkeypatch,
):
    from tests.test_ohmo.test_nutrition_dialogue_review_regressions import _ScriptedEngine, _trace

    ingress, _bus, request, pool, honcho, _, _ = await _save_callback_portion(
        tmp_path, monkeypatch
    )
    attempt = ingress._attempts[request["candidate_id"]]
    complete_commit = dict(attempt["camera_commit"])
    legacy_fields = (
        "event_id", "source_message_id", "client_op_id", "candidate_id",
        "tenant_id", "principal", "meal_at", "record_type", "consumption_status",
    )
    attempt["camera_commit"] = {key: complete_commit[key] for key in legacy_fields}
    ingress._save_attempts()
    original = honcho.messages[-1]
    original_items = original.metadata["decision_trace"]["annotations"]["nutrition"]["items"]
    correction = _trace({
        "schema_version": 2, "record_type": "meal_correction",
        "changed_fields": ["items", "energy_kcal_best"],
        "items": [*original_items, {"name": "hummus", "quantity_text": "1 spoon"}],
        "energy_kcal_best": 290,
    })
    scripted = _ScriptedEngine(pool, [(correction, "Добавила хумус к этой порции.")])
    scripted.messages = pool._test_bundle.engine.messages
    pool._test_bundle.engine = scripted
    message = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="И еще добавь хумус",
        metadata={"message_id": "synthetic-legacy-context-followup",
                  "_telegram_raw_text": "И еще добавь хумус", "is_group": False,
                  "chat_type": "private"},
    )
    ingress.process_real_inbound(message)
    final = await runtime_turn(pool, message, ingress)
    upgraded = attempt["camera_commit"]
    assert upgraded["event_id"] == original.id
    assert upgraded["client_op_id"] == complete_commit["client_op_id"]
    assert upgraded["gateway_session_id"] == complete_commit["gateway_session_id"]
    assert upgraded["annotation"] == complete_commit["annotation"]
    assert final.metadata["nutrition_append_event_id"] == honcho.messages[-1].id
    assert honcho.messages[-1].metadata["selected_source"]["schema_version"] == 2
    assert honcho.messages[-1].metadata["selected_source"]["original_receipt_event_id"] == original.id
    await ingress.close()


@pytest.mark.asyncio
async def test_explicit_photo_reply_patches_the_same_camera_meal(
    tmp_path, monkeypatch,
):
    from tests.test_ohmo.test_nutrition_dialogue_review_regressions import _ScriptedEngine, _trace

    ingress, _bus, request, pool, honcho, _, _ = await _save_callback_portion(
        tmp_path, monkeypatch
    )
    attempt = ingress._attempts[request["candidate_id"]]
    original = next(
        message for message in honcho.messages
        if message.id == attempt["camera_commit"]["event_id"]
    )
    nutrition = original.metadata["decision_trace"]["annotations"]["nutrition"]
    items = list(nutrition["items"]) + [
        {"name": "куриное яйцо", "quantity_text": "1 штука"}
    ]
    trace = _trace({
        "schema_version": 2, "record_type": "meal_correction",
        "changed_fields": ["items", "energy_kcal_best"],
        "items": items, "energy_kcal_best": 285,
    })
    scripted = _ScriptedEngine(pool, [(trace, "Добавила яйцо к этой порции.")])
    scripted.messages = pool._test_bundle.engine.messages
    pool._test_bundle.engine = scripted
    reply = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="И еще добавь одно куриное яйцо",
        metadata={"message_id": "synthetic-photo-reply-correction",
                  "reply_to_message_id": str(attempt["photo_id"]),
                  "_telegram_raw_text": "И еще добавь одно куриное яйцо",
                  "is_group": False, "chat_type": "private"},
    )
    ingress.process_real_inbound(reply)
    assert reply.metadata["_camera_context_meal_candidate_id"] == request["candidate_id"]
    result = await runtime_turn(pool, reply, ingress)
    assert "nutrition_append_event_id" in result.metadata, (result.text, result.metadata)
    assert result.metadata["nutrition_append_event_id"] != original.id
    correction = honcho.messages[-1]
    assert correction.metadata["reply_to_source_message_id"] == original.metadata["source_message_id"]
    assert correction.metadata["target_meal_id"] == derive_meal_id(
        tenant_id=ingress.config.tenant_id,
        source_principal=f"telegram:{ingress.config.principal}",
        gateway_session_id=pool._test_bundle.session_id,
        source_message_id=original.metadata["source_message_id"],
    )
    assert correction.metadata["decision_trace"]["annotations"]["nutrition"]["items"] == items
    assert original.metadata["decision_trace"]["annotations"]["nutrition"]["items"] == nutrition["items"]
    await ingress.close()


@pytest.mark.asyncio
async def test_runtime_reconciles_identical_typed_composition_repeat(
    tmp_path, monkeypatch
):
    label = "Всё: яйцо и рис."
    ingress, request, pool, honcho, saved_final, _ = await _save_typed_confirmation(
        tmp_path, monkeypatch, label
    )
    attempt = ingress._attempts[request["candidate_id"]]
    replay = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=label,
        metadata={"message_id": "typed-camera-composition-repeat",
                  "reply_to_message_id": str(attempt["photo_id"]),
                  "_telegram_raw_text": label, "is_group": False,
                  "chat_type": "private"},
    )
    ingress.process_real_inbound(replay)
    assert replay.metadata.get("_camera_typed_replay_candidate") == request["candidate_id"]
    before_turns = len(pool._test_bundle.engine.turns)
    final = await runtime_turn(pool, replay, ingress)
    assert final.text == "Эта порция уже записана."
    assert final.metadata["nutrition_append_event_id"] == saved_final.metadata[
        "nutrition_append_event_id"
    ]
    assert replay.metadata.get("_camera_typed_replay") is True
    assert len(pool._test_bundle.engine.turns) == before_turns
    assert len(honcho.messages) == 2
    await ingress.close()


@pytest.mark.asyncio
async def test_runtime_reconciles_new_tap_without_a_second_nutrition_append(
    tmp_path, monkeypatch
):
    ingress, bus, request, pool, honcho, saved_portion, saved_final = (
        await _save_callback_portion(tmp_path, monkeypatch)
    )
    turns_before_replay = len(pool._test_bundle.engine.turns)
    commit_before_replay = dict(ingress._attempts[request["candidate_id"]]["camera_commit"])

    replay = await _native_callback(
        bus, label="2 кусочка",
        target=ingress._attempts[request["candidate_id"]]["reply_ids"][0],
        options=["1 кусочек", "2 кусочка", "Половину порции"],
        prompt="Вы съели это? Сколько примерно вы съели?",
    )
    replay.metadata["callback_query_id"] = "callback-portion-2"
    ingress.process_real_inbound(replay)
    assert replay.metadata["_camera_turn_id"] == saved_portion.metadata["_camera_turn_id"]
    assert replay.metadata["_camera_existing_meal_replay"] is True

    public_final = await runtime_turn(pool, replay, ingress)
    assert public_final.text == "Эта порция уже записана."
    assert public_final.metadata["nutrition_append_event_id"] == saved_final.metadata[
        "nutrition_append_event_id"
    ]
    assert len(honcho.messages) == 4
    assert len(pool._test_bundle.engine.turns) == turns_before_replay
    assert ingress._attempts[request["candidate_id"]]["camera_commit"] == commit_before_replay
    await ingress.close()


@pytest.mark.asyncio
async def test_runtime_reconciles_identical_typed_reply_without_native_click_marker(
    tmp_path, monkeypatch
):
    ingress, request, pool, honcho, saved_final, first_answer = (
        await _save_typed_confirmation(tmp_path, monkeypatch)
    )
    attempt = ingress._attempts[request["candidate_id"]]
    label = first_answer.content
    replay = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=label,
        metadata={
            "message_id": "typed-camera-replay-2",
            "reply_to_message_id": str(attempt["reply_ids"][0]),
            "_telegram_raw_text": label,
            "is_group": False,
            "chat_type": "private",
        },
    )
    ingress.process_real_inbound(replay)
    assert replay.metadata.get("callback_query") is None
    assert replay.metadata.get("_camera_typed_replay_candidate") == request["candidate_id"]
    original_pair = [
        (message.id, message.content, dict(message.metadata)) for message in honcho.messages
    ]
    assert [item[0] for item in original_pair] == ["honcho-1", "honcho-2"]
    public_final = await runtime_turn(pool, replay, ingress)
    assert replay.metadata.get("_camera_typed_replay") is True
    assert public_final.text == "Эта порция уже записана."
    assert public_final.metadata["nutrition_append_event_id"] == saved_final.metadata[
        "nutrition_append_event_id"
    ]
    assert [(message.id, message.content, dict(message.metadata))
            for message in honcho.messages] == original_pair
    assert sum(
        message.metadata.get("decision_trace", {}).get("annotations", {})
        .get("nutrition", {}).get("record_type") == "meal_observation"
        for message in honcho.messages
    ) == 1
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_text", ["Я съела другую порцию", "Я съела это вчера"])
@pytest.mark.parametrize("reply_to_photo", [True, False])
async def test_changed_typed_reply_reaches_existing_runtime_flow(
    tmp_path, monkeypatch, changed_text, reply_to_photo,
):
    ingress, request, pool, honcho, _, _ = await _save_typed_confirmation(
        tmp_path, monkeypatch, reply_to_photo=reply_to_photo,
    )
    attempt = ingress._attempts[request["candidate_id"]]
    turns_before = len(pool._test_bundle.engine.turns)
    changed = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=changed_text,
        metadata={"message_id": "typed-camera-changed-reply",
                  **({"reply_to_message_id": str(attempt["photo_id"])}
                     if reply_to_photo else {}),
                  "_telegram_raw_text": changed_text, "is_group": False,
                  "chat_type": "private"},
    )
    ingress.process_real_inbound(changed)
    if changed_text == "Я съела это вчера":
        assert changed.metadata.get("_camera_typed_replay_candidate") is None
        if reply_to_photo:
            from openharness.evals import TRACE_FINALIZATION
            from openharness.engine.stream_events import AssistantTextDelta

            assert changed.metadata.get("_camera_ordinary_date_correction") is CAMERA_AUTHORITY
            assert changed.metadata.get("_camera_unbound") is None
            engine = pool._test_bundle.engine

            async def date_correction(user_message, *, wellness_actor=None):
                del wellness_actor
                engine.messages.append(user_message)
                engine.turns.append((changed.content, [], changed.timestamp))
                recorder = engine.decision_trace_recorder
                recorder.trace_requirement_signals(changed.content)
                recorder.record(TRACE_FINALIZATION, {
                    "schema_version": 1,
                    "trace_event_id": "changed-date-typed-repeat",
                    "annotations": {"nutrition": {
                        "schema_version": 2,
                        "record_type": "meal_correction",
                        "changed_fields": ["meal_date"],
                        "meal_date": "2026-10-05",
                    }},
                })
                yield AssistantTextDelta(text="Изменение сохранено; баланс обновляется.")

            engine.submit_message = date_correction
            result = await runtime_turn(pool, changed, ingress)
            assert result.metadata["nutrition_append_event_id"] == honcho.messages[-1].id
            assert honcho.messages[-1].metadata["decision_trace"]["annotations"]["nutrition"][
                "changed_fields"
            ] == ["meal_date"]
        else:
            assert changed.metadata.get("_camera_ordinary_date_correction") is None
            turns_before = len(pool._test_bundle.engine.turns)
            result = await runtime_turn(pool, changed, ingress)
            assert result.text != "Эта порция уже записана."
            assert result.metadata.get("nutrition_append_event_id") is None
            assert len(pool._test_bundle.engine.turns) == turns_before + 1
        await ingress.close()
        return
    assert changed.metadata.get("_camera_typed_replay_candidate") == request["candidate_id"]
    result = await runtime_turn(pool, changed, ingress)
    assert changed.metadata.get("_camera_typed_replay_candidate") is None
    assert result.text != "Эта порция уже записана."
    if reply_to_photo:
        assert changed.metadata["reply_to_message_id"] == str(attempt["photo_id"])
    else:
        assert "reply_to_message_id" not in changed.metadata
    assert len(pool._test_bundle.engine.turns) == turns_before + 1
    assert len(honcho.messages) == 2
    assert attempt["camera_commit"]["event_id"] == "honcho-2"
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "receipt_failure", ["missing", "foreign", "stale", "bad_native_route", "wrong_native_target"],
)
async def test_context_repeat_rejects_missing_or_misbound_original_receipt(
    tmp_path, monkeypatch, receipt_failure,
):
    ingress, request, pool, honcho, _, first_answer = await _save_typed_confirmation(
        tmp_path, monkeypatch, reply_to_photo=False,
    )
    backend = pool._shadow_backend_for_scope(None)
    reconcile = backend.reconcile_durable_exchange

    async def altered_receipt(user_op, assistant_op):
        receipt = await reconcile(user_op, assistant_op)
        if receipt_failure == "missing":
            return None
        from dataclasses import replace

        metadata = dict(receipt.assistant_metadata)
        if receipt_failure == "foreign":
            metadata["tenant_id"] = "another-owner"
        elif receipt_failure == "stale":
            metadata["source_message_id"] = "old-source"
        elif receipt_failure == "bad_native_route":
            metadata["camera_route"] = "reply"
            metadata["camera_reply_to_native_message_id"] = "999"
        elif receipt_failure == "wrong_native_target":
            metadata["camera_reply_to_native_message_id"] = "999"
        return replace(receipt, assistant_metadata=metadata)

    pool._shadow_backend_for_scope = lambda _scope: SimpleNamespace(
        reconcile_durable_exchange=altered_receipt
    )
    replay = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=first_answer.content,
        metadata={"message_id": "context-repeat-bad-receipt",
                  "_telegram_raw_text": first_answer.content,
                  "is_group": False, "chat_type": "private"},
    )
    ingress.process_real_inbound(replay)
    pool._active_message = replay
    updates = [
        update async for update in pool.stream_message(replay, ingress.config.session_key)
    ]
    assert not any(update.kind == "final" and update.text == "Эта порция уже записана." for update in updates)
    assert len(honcho.messages) == 2
    assert ingress._attempts[request["candidate_id"]]["camera_commit"]["event_id"] == "honcho-2"
    await ingress.close()


@pytest.mark.asyncio
async def test_new_owner_photo_interrupts_completed_context_repeat(tmp_path, monkeypatch):
    ingress, request, pool, honcho, saved_final, first_answer = await _save_typed_confirmation(
        tmp_path, monkeypatch, reply_to_photo=False,
    )
    attempt = ingress._attempts[request["candidate_id"]]
    unrelated_photo = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content="new photo",
        metadata={"message_id": "unrelated-photo-after-save", "is_group": False,
                  "chat_type": "private"},
    )
    unrelated_photo.media.append("owner-new-photo.jpg")
    ingress.process_real_inbound(unrelated_photo)
    assert unrelated_photo.media == ["owner-new-photo.jpg"]
    assert attempt["context_interrupted"] is True

    replay = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=first_answer.content,
        metadata={"message_id": "context-repeat-after-new-photo",
                  "_telegram_raw_text": first_answer.content,
                  "is_group": False, "chat_type": "private"},
    )
    ingress.process_real_inbound(replay)
    assert replay.metadata.get("_camera_typed_replay_candidate") is None
    before_turns = len(pool._test_bundle.engine.turns)
    result = await runtime_turn(pool, replay, ingress)
    assert result.text != "Эта порция уже записана."
    assert len(pool._test_bundle.engine.turns) == before_turns + 1
    assert attempt["camera_commit"]["event_id"] == saved_final.metadata[
        "nutrition_append_event_id"
    ]
    assert len(honcho.messages) == 2
    await ingress.close()


@pytest.mark.asyncio
async def test_context_repeat_reconciles_after_journal_restart(tmp_path, monkeypatch):
    ingress, request, pool, honcho, saved_final, first_answer = await _save_typed_confirmation(
        tmp_path, monkeypatch, reply_to_photo=False,
    )
    restarted, _, _, _ = _ingress(tmp_path, FakeTelegram())
    pool._camera_ingress = restarted
    replay = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=first_answer.content,
        metadata={"message_id": "context-repeat-after-restart",
                  "_telegram_raw_text": first_answer.content,
                  "is_group": False, "chat_type": "private"},
    )
    restarted.process_real_inbound(replay)
    assert replay.metadata.get("_camera_typed_replay_candidate") == request["candidate_id"]
    result = await runtime_turn(pool, replay, restarted)
    assert result.text == "Эта порция уже записана."
    assert result.metadata["nutrition_append_event_id"] == saved_final.metadata[
        "nutrition_append_event_id"
    ]
    assert len(honcho.messages) == 2
    await ingress.close()
    await restarted.close()


@pytest.mark.asyncio
async def test_context_repeat_rejects_foreign_owner_and_new_active_source(tmp_path, monkeypatch):
    ingress, request, pool, honcho, _, first_answer = await _save_typed_confirmation(
        tmp_path, monkeypatch, reply_to_photo=False,
    )
    foreign = InboundMessage(
        channel="telegram", sender_id="other-owner", chat_id="123",
        content=first_answer.content,
        metadata={"message_id": "foreign-context-repeat",
                  "_telegram_raw_text": first_answer.content},
    )
    ingress.process_real_inbound(foreign)
    assert foreign.metadata.get("_camera_typed_replay_candidate") is None

    root = Path(tmp_path)
    newer = _candidate(root, index=1)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, newer))[0] == 202
    ambiguous = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=first_answer.content,
        metadata={"message_id": "context-repeat-with-new-source",
                  "_telegram_raw_text": first_answer.content},
    )
    ingress.process_real_inbound(ambiguous)
    assert ambiguous.metadata.get("_camera_typed_replay_candidate") is None
    assert ambiguous.metadata.get("_camera_existing_meal_replay") is None
    assert len(honcho.messages) == 2
    await ingress.close()


@pytest.mark.asyncio
async def test_completed_camera_meal_accepts_bound_owner_date_correction(
    tmp_path, monkeypatch,
):
    from openharness.evals import TRACE_FINALIZATION
    from openharness.engine.stream_events import AssistantTextDelta

    ingress, root, bus, _ = _ingress(tmp_path, FakeTelegram())
    request = _candidate(root)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    await bus.consume_inbound()
    honcho = RuntimeHoncho()
    pool = runtime_pool(tmp_path, ingress, honcho, monkeypatch)
    engine = pool._test_bundle.engine

    async def scripted_submit(user_message, *, wellness_actor=None):
        del wellness_actor
        engine.messages.append(user_message)
        message = pool._active_message
        engine.turns.append((message.content, list(message.media), message.timestamp))
        recorder = engine.decision_trace_recorder
        recorder.trace_requirement_signals(message.content)
        if message.content == "Да, я съела только часть":
            yield AssistantTextDelta(text="Сколько примерно вы съели?")
            return
        if message.content == "2 кусочка":
            nutrition = {
                "schema_version": 2,
                "record_type": "meal_observation",
                "basis": ["image", "owner_statement"],
                "consumption_status": "consumed",
                "is_estimate": True,
                "energy_kcal_best": 125,
                "items": [{
                    "name": "synthetic meal",
                    "quantity_text": "original 125 kcal portion",
                    "energy_kcal_best": 125,
                }],
            }
            text = "Записано. Баланс обновляется."
        elif message.content == "1 кусочек":
            nutrition = {
                "schema_version": 2,
                "record_type": "meal_correction",
                "changed_fields": ["items", "energy_kcal_best"],
                "items": [{
                    "name": "synthetic meal",
                    "quantity_text": "corrected 53 kcal portion",
                    "energy_kcal_best": 53,
                }],
                "energy_kcal_best": 53,
            }
            text = "Изменение сохранено; баланс обновляется."
        else:
            nutrition = {
                "schema_version": 2,
                "record_type": "meal_correction",
                "changed_fields": ["meal_at"],
                "meal_at": "2026-10-05T18:45:00+00:00",
            }
            text = "Исправила дату приёма пищи."
        recorder.record(TRACE_FINALIZATION, {
            "schema_version": 1,
            "trace_event_id": f"g9-{message.metadata['message_id']}",
            "annotations": {"nutrition": nutrition},
        })
        yield AssistantTextDelta(text=text)

    async def invalid_meal_submit(user_message, *, wellness_actor=None):
        del wellness_actor
        engine.messages.append(user_message)
        message = pool._active_message
        engine.turns.append((message.content, list(message.media), message.timestamp))
        recorder = engine.decision_trace_recorder
        recorder.trace_requirement_signals(message.content)
        recorder.record(TRACE_FINALIZATION, {
            "schema_version": 1,
            "trace_event_id": "g9-date-correction-cannot-duplicate-meal",
            "annotations": {"nutrition": {
                "schema_version": 2,
                "record_type": "meal_observation",
                "basis": ["owner_statement"],
                "consumption_status": "consumed",
                "is_estimate": True,
                "energy_kcal_best": 999,
                "items": [{
                    "name": "must not be appended",
                    "quantity_text": "extra meal",
                    "energy_kcal_best": 999,
                }],
            }},
        })
        yield AssistantTextDelta(text="Записала ещё один приём пищи.")

    engine.submit_message = scripted_submit
    attempt = ingress._attempts[request["candidate_id"]]
    photo_id = attempt["photo_id"]
    first = await _native_callback(
        bus, label="Да, я съела только часть", target=photo_id,
        options=["Да, я съела только часть", "Нет, не ела"],
        prompt="Съели ли вы это?",
    )
    first.metadata["callback_query_id"] = "g9-first-meal"
    ingress.process_real_inbound(first)
    await runtime_turn(pool, first, ingress)

    quantity = await _native_callback(
        bus, label="2 кусочка", target=attempt["reply_ids"][-1],
        options=["1 кусочек", "2 кусочка", "Половину порции"],
        prompt="Сколько примерно вы съели?",
    )
    quantity.metadata["callback_query_id"] = "g9-quantity"
    ingress.process_real_inbound(quantity)
    saved = await runtime_turn(pool, quantity, ingress)
    original_event_id = saved.metadata["nutrition_append_event_id"]
    original_row = next(row for row in honcho.messages if row.id == original_event_id)
    original_nutrition = original_row.metadata["decision_trace"]["annotations"]["nutrition"]
    assert original_nutrition["energy_kcal_best"] == 125
    capture_time = attempt["camera_commit"]["meal_at"]
    assert original_nutrition["meal_at"].replace("Z", "+00:00") == capture_time.replace(
        "Z", "+00:00"
    )

    portion = await _native_callback(
        bus, label="1 кусочек", target=attempt["reply_ids"][0],
        options=["1 кусочек", "2 кусочка", "Половину порции"],
        prompt="Сколько примерно вы съели?",
    )
    portion.metadata["callback_query_id"] = "g9-portion-correction"
    ingress.process_real_inbound(portion)
    portion_final = await runtime_turn(pool, portion, ingress)
    portion_event_id = portion_final.metadata["nutrition_append_event_id"]
    portion_row = next(row for row in honcho.messages if row.id == portion_event_id)
    portion_nutrition = portion_row.metadata["decision_trace"]["annotations"]["nutrition"]
    assert portion_nutrition["record_type"] == "meal_correction"
    assert portion_nutrition["changed_fields"] == ["items", "energy_kcal_best"]
    assert portion_nutrition["energy_kcal_best"] == 53
    assert "meal_at" not in portion_nutrition and "meal_date" not in portion_nutrition

    before_date = [
        (row.id, row.content, dict(row.metadata)) for row in honcho.messages
    ]
    foreign_date = InboundMessage(
        channel="telegram", sender_id="other-owner", chat_id="123",
        content="Это было 2026-10-05 в 18:45 UTC",
        metadata={"message_id": "g9-foreign-date", "reply_to_message_id": str(photo_id)},
    )
    ingress.process_real_inbound(foreign_date)
    assert foreign_date.metadata.get("_camera_ordinary_date_correction") is None
    stale_date = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Это было 2026-10-05 в 18:45 UTC",
        metadata={"message_id": "g9-stale-date", "reply_to_message_id": "999999"},
    )
    ingress.process_real_inbound(stale_date)
    assert stale_date.metadata.get("_camera_ordinary_date_correction") is None
    assert stale_date.metadata.get("_camera_authority") is not CAMERA_AUTHORITY
    date_reply = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Это было 2026-10-05 в 18:45 UTC",
        metadata={
            "message_id": "g9-explicit-date-reply",
            "reply_to_message_id": str(photo_id),
            "_telegram_raw_text": "Это было 2026-10-05 в 18:45 UTC",
            "is_group": False,
            "chat_type": "private",
        },
    )
    ingress.process_real_inbound(date_reply)
    assert date_reply.metadata.get("_camera_ordinary_date_correction") is CAMERA_AUTHORITY
    assert date_reply.metadata.get("_camera_unbound") is None
    assert date_reply.metadata.get("_camera_authority") is not CAMERA_AUTHORITY
    date_final = await runtime_turn(pool, date_reply, ingress)
    assert date_final.metadata["nutrition_append_event_id"]
    assert date_final.metadata["nutrition_append_event_id"] not in {
        original_event_id, portion_event_id,
    }
    assert len(honcho.messages) == len(before_date) + 2
    assert [
        (row.id, row.content, dict(row.metadata)) for row in honcho.messages[:len(before_date)]
    ] == before_date
    date_assistant = honcho.messages[-1]
    date_nutrition = date_assistant.metadata["decision_trace"]["annotations"]["nutrition"]
    assert date_nutrition["record_type"] == "meal_correction"
    assert date_nutrition["changed_fields"] == ["meal_at"]
    assert date_nutrition["meal_at"].replace("Z", "+00:00") == "2026-10-05T18:45:00+00:00"
    assert date_assistant.metadata["reply_to_source_message_id"] == str(photo_id)
    assert sum(
        row.metadata.get("decision_trace", {}).get("annotations", {})
        .get("nutrition", {}).get("record_type") == "meal_observation"
        for row in honcho.messages
    ) == 1
    assert attempt["camera_commit"]["event_id"] == original_event_id

    rows_after_valid_date = [
        (row.id, row.content, dict(row.metadata)) for row in honcho.messages
    ]
    malformed_date = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Это было 2026-10-05 в 18:45 UTC",
        metadata={
            "message_id": "g9-date-must-not-save-another-meal",
            "reply_to_message_id": str(photo_id),
            "_telegram_raw_text": "Это было 2026-10-05 в 18:45 UTC",
            "is_group": False,
            "chat_type": "private",
        },
    )
    ingress.process_real_inbound(malformed_date)
    engine.submit_message = invalid_meal_submit
    with pytest.raises(ValueError, match="explicit date-only correction"):
        await runtime_turn(pool, malformed_date, ingress)
    assert [
        (row.id, row.content, dict(row.metadata)) for row in honcho.messages
    ] == rows_after_valid_date
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("date_text", "field", "value"),
    [
        ("Это было вчера", "meal_date", "2026-10-05"),
        ("Да, это было 2026-10-05 в 18:45 UTC", "meal_at", "2026-10-05T18:45:00+00:00"),
    ],
)
async def test_completed_camera_natural_and_affirmative_date_corrections_are_date_only(
    tmp_path, monkeypatch, date_text, field, value,
):
    from openharness.evals import TRACE_FINALIZATION
    from openharness.engine.stream_events import AssistantTextDelta

    ingress, request, pool, honcho, _, _ = await _save_typed_confirmation(tmp_path, monkeypatch)
    attempt = ingress._attempts[request["candidate_id"]]
    engine = pool._test_bundle.engine
    message = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=date_text,
        metadata={"message_id": f"date-correction-{field}",
                  "reply_to_message_id": str(attempt["photo_id"]),
                  "_telegram_raw_text": date_text, "is_group": False,
                  "chat_type": "private"},
    )

    async def date_only_submit(user_message, *, wellness_actor=None):
        del wellness_actor
        engine.messages.append(user_message)
        engine.turns.append((message.content, [], message.timestamp))
        recorder = engine.decision_trace_recorder
        recorder.trace_requirement_signals(message.content)
        annotation = {
            "schema_version": 2,
            "record_type": "meal_correction",
            "changed_fields": [field],
            field: value,
        }
        recorder.record(TRACE_FINALIZATION, {
            "schema_version": 1,
            "trace_event_id": f"date-only-{field}",
            "annotations": {"nutrition": annotation},
        })
        yield AssistantTextDelta(text="Изменение сохранено; баланс обновляется.")

    engine.submit_message = date_only_submit
    before = [(row.id, row.content, dict(row.metadata)) for row in honcho.messages]
    ingress.process_real_inbound(message)
    assert message.metadata.get("_camera_ordinary_date_correction") is CAMERA_AUTHORITY
    assert message.metadata.get("_camera_unbound") is None
    final = await runtime_turn(pool, message, ingress)

    assert final.text == "Изменение сохранено; баланс обновляется."
    assert final.metadata["nutrition_append_event_id"] == honcho.messages[-1].id
    assert len(honcho.messages) == len(before) + 2
    assert [(row.id, row.content, dict(row.metadata)) for row in honcho.messages[:len(before)]] == before
    corrected = honcho.messages[-1]
    stored = corrected.metadata["decision_trace"]["annotations"]["nutrition"]
    assert stored["record_type"] == "meal_correction"
    assert stored["changed_fields"] == [field]
    observed = stored[field].replace("Z", "+00:00") if field == "meal_at" else stored[field]
    assert observed == value
    assert not {"items", "energy_kcal_best", "consumption_status"} & set(stored)
    assert corrected.metadata["reply_to_source_message_id"] == str(attempt["photo_id"])
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "denial_text",
    ["Нет, я это не ела 2026-10-05", "Нет, я это не ела вчера"],
)
async def test_dated_denial_to_retained_camera_photo_uses_denial_correction(
    tmp_path, monkeypatch, denial_text,
):
    from openharness.evals import TRACE_FINALIZATION
    from openharness.engine.stream_events import AssistantTextDelta

    ingress, request, pool, honcho, _, _ = await _save_typed_confirmation(tmp_path, monkeypatch)
    attempt = ingress._attempts[request["candidate_id"]]
    original_commit = dict(attempt["camera_commit"])
    original_rows = [(row.id, row.content, dict(row.metadata), row.created_at) for row in honcho.messages]
    engine = pool._test_bundle.engine
    denial = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=denial_text,
        metadata={
            "message_id": f"dated-denial-{attempt['photo_id']}",
            "reply_to_message_id": str(attempt["photo_id"]),
            "_telegram_raw_text": denial_text,
            "is_group": False,
            "chat_type": "private",
        },
    )

    async def denial_submit(user_message, *, wellness_actor=None):
        del wellness_actor
        engine.messages.append(user_message)
        engine.turns.append((denial_text, [], denial.timestamp))
        recorder = engine.decision_trace_recorder
        recorder.trace_requirement_signals(denial_text)
        recorder.record(TRACE_FINALIZATION, {
            "schema_version": 1,
            "trace_event_id": f"dated-denial-{attempt['photo_id']}",
            "annotations": {"nutrition": {
                "schema_version": 2,
                "record_type": "meal_correction",
                "changed_fields": ["consumption_status", "energy_kcal_best"],
                "consumption_status": "not_consumed",
                "energy_kcal_best": 0,
            }},
        })
        yield AssistantTextDelta(text="Записано. Баланс обновлён: 125 ккал.")

    engine.submit_message = denial_submit
    ingress.process_real_inbound(denial)
    try:
        final = await runtime_turn(pool, denial, ingress)
    except ValueError as error:
        assert [
            (row.id, row.content, dict(row.metadata), row.created_at)
            for row in honcho.messages
        ] == original_rows
        raise AssertionError("actual runtime rejected a valid dated owner denial") from error
    assert denial.metadata.get("_camera_answer") == "no"
    assert denial.metadata.get("_camera_correction") is CAMERA_AUTHORITY
    assert denial.metadata.get("_camera_candidate_id") == request["candidate_id"]
    assert denial.metadata.get("_camera_ordinary_date_correction") is None
    correction = attempt["camera_correction_commit"]
    assert correction["kind"] == "denial"
    assert correction["target_event_id"] == original_commit["event_id"]
    assert correction["target_source_message_id"] == original_commit["source_message_id"]
    assert correction["source_message_id"] == denial.metadata["message_id"]
    assert final.text == "Изменение сохранено; баланс обновляется."
    assert "125 ккал" not in final.text
    assert final.metadata["nutrition_append_event_id"] == correction["event_id"]
    assert final.metadata["nutrition_sync_status"] == "pending"
    assert attempt["camera_commit"] == original_commit
    assert len(honcho.messages) == len(original_rows) + 2
    assert [
        (row.id, row.content, dict(row.metadata), row.created_at)
        for row in honcho.messages[:len(original_rows)]
    ] == original_rows
    correction_row = honcho.messages[-1]
    assert correction_row.id == correction["event_id"]
    nutrition = correction_row.metadata["decision_trace"]["annotations"]["nutrition"]
    assert nutrition["record_type"] == "meal_correction"
    assert nutrition["consumption_status"] == "not_consumed"
    assert nutrition["energy_kcal_best"] == 0

    replay = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=denial_text,
        metadata={
            "message_id": denial.metadata["message_id"],
            "reply_to_message_id": str(attempt["photo_id"]),
            "_telegram_raw_text": denial_text,
            "is_group": False,
            "chat_type": "private",
        },
    )
    ingress.process_real_inbound(replay)
    assert replay.metadata.get("_camera_correction_replay") is CAMERA_AUTHORITY
    rows_before_replay = len(honcho.messages)
    replay_final = await runtime_turn(pool, replay, ingress)
    assert replay_final.metadata["nutrition_append_event_id"] == correction["event_id"]
    assert len(honcho.messages) == rows_before_replay
    assert attempt["camera_commit"] == original_commit
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("trace_kind", ["new_meal", "quantity_correction"])
async def test_camera_date_correction_rejects_meal_or_quantity_trace(
    tmp_path, monkeypatch, trace_kind,
):
    from openharness.evals import TRACE_FINALIZATION
    from openharness.engine.stream_events import AssistantTextDelta

    ingress, request, pool, honcho, _, _ = await _save_typed_confirmation(tmp_path, monkeypatch)
    attempt = ingress._attempts[request["candidate_id"]]
    engine = pool._test_bundle.engine
    text = "Да, это было 2026-10-05 в 18:45 UTC"
    message = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=text,
        metadata={"message_id": f"bad-date-trace-{trace_kind}",
                  "reply_to_message_id": str(attempt["photo_id"]),
                  "_telegram_raw_text": text, "is_group": False,
                  "chat_type": "private"},
    )

    async def incompatible_submit(user_message, *, wellness_actor=None):
        del wellness_actor
        engine.messages.append(user_message)
        engine.turns.append((message.content, [], message.timestamp))
        recorder = engine.decision_trace_recorder
        recorder.trace_requirement_signals(message.content)
        nutrition = (
            {
                "schema_version": 2, "record_type": "meal_observation",
                "basis": ["owner_statement"], "consumption_status": "consumed",
                "is_estimate": True, "energy_kcal_best": 999,
                "items": [{"name": "must not be appended", "quantity_text": "new meal",
                           "energy_kcal_best": 999}],
            }
            if trace_kind == "new_meal" else
            {
                "schema_version": 2, "record_type": "meal_correction",
                "changed_fields": ["items", "energy_kcal_best"],
                "energy_kcal_best": 53,
                "items": [{"name": "wrong quantity", "quantity_text": "1 piece",
                           "energy_kcal_best": 53}],
            }
        )
        recorder.record(TRACE_FINALIZATION, {
            "schema_version": 1,
            "trace_event_id": f"bad-date-{trace_kind}",
            "annotations": {"nutrition": nutrition},
        })
        yield AssistantTextDelta(text="Записано. Баланс обновлён: 999 ккал.")

    engine.submit_message = incompatible_submit
    before = [(row.id, row.content, dict(row.metadata)) for row in honcho.messages]
    ingress.process_real_inbound(message)
    assert message.metadata.get("_camera_ordinary_date_correction") is CAMERA_AUTHORITY
    with pytest.raises(ValueError, match="explicit date-only correction"):
        await runtime_turn(pool, message, ingress)
    assert [(row.id, row.content, dict(row.metadata)) for row in honcho.messages] == before
    await ingress.close()


@pytest.mark.asyncio
async def test_date_correction_source_must_be_unique_owner_reply_to_retained_meal(
    tmp_path, monkeypatch,
):
    ingress, request, _, _, _, _ = await _save_typed_confirmation(tmp_path, monkeypatch)
    attempt = ingress._attempts[request["candidate_id"]]
    probes = [
        InboundMessage(
            channel="telegram", sender_id="123", chat_id="123", content="Это было вчера",
            metadata={"message_id": "date-source-missing", "_telegram_raw_text": "Это было вчера"},
        ),
        InboundMessage(
            channel="telegram", sender_id="other-owner", chat_id="123", content="Это было вчера",
            metadata={"message_id": "date-source-foreign",
                      "reply_to_message_id": str(attempt["photo_id"]),
                      "_telegram_raw_text": "Это было вчера"},
        ),
        InboundMessage(
            channel="telegram", sender_id="123", chat_id="123", content="Это было вчера",
            metadata={"message_id": "date-source-stale", "reply_to_message_id": "999999",
                      "_telegram_raw_text": "Это было вчера"},
        ),
    ]
    for message in probes:
        ingress.process_real_inbound(message)
        assert message.metadata.get("_camera_ordinary_date_correction") is None
        assert message.metadata.get("_camera_typed_replay_candidate") is None

    conflicting = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Это другой приём пищи, было 2026-10-05 в 18:45 UTC",
        metadata={"message_id": "date-source-conflict",
                  "reply_to_message_id": str(attempt["photo_id"]),
                  "_telegram_raw_text": "Это другой приём пищи, было 2026-10-05 в 18:45 UTC"},
    )
    ingress.process_real_inbound(conflicting)
    assert conflicting.metadata.get("_camera_ordinary_date_correction") is None
    assert conflicting.metadata.get("_camera_unbound") is CAMERA_AUTHORITY
    await ingress.close()


@pytest.mark.asyncio
async def test_durable_camera_denial_replaces_stale_status_and_replay_is_truthful(
    tmp_path, monkeypatch,
):
    from openharness.evals import TRACE_FINALIZATION
    from openharness.engine.stream_events import AssistantTextDelta
    from ohmo.evals.nutrition_persistence import _fold_events

    ingress, bus, request, pool, honcho, _, saved_final = await _save_callback_portion(
        tmp_path, monkeypatch
    )
    attempt = ingress._attempts[request["candidate_id"]]
    original_event = saved_final.metadata["nutrition_append_event_id"]
    full_commit = attempt["camera_commit"]
    attempt["camera_commit"] = {
        key: full_commit[key]
        for key in (
            "event_id", "source_message_id", "client_op_id", "candidate_id",
            "tenant_id", "principal", "meal_at", "record_type", "consumption_status",
        )
    }
    ingress._save_attempts()
    engine = pool._test_bundle.engine

    async def denial_submit(user_message, *, wellness_actor=None):
        del wellness_actor
        engine.messages.append(user_message)
        engine.turns.append((user_message.text, [], pool._active_message.timestamp))
        recorder = engine.decision_trace_recorder
        recorder.trace_requirement_signals(user_message.text)
        recorder.record(TRACE_FINALIZATION, {
            "schema_version": 1,
            "trace_event_id": "durable-denial-status",
            "annotations": {"nutrition": {
                "schema_version": 2, "record_type": "meal_correction",
                "changed_fields": ["consumption_status", "energy_kcal_best"],
                "consumption_status": "not_consumed", "energy_kcal_best": 0,
            }},
        })
        yield AssistantTextDelta(
            text="Не удалось подтвердить сохранение изменения. Записано. Баланс обновлён: 105 ккал."
        )

    engine.submit_message = denial_submit
    denial = await _native_callback(
        bus, label="Нет, не ела", target=attempt["photo_id"],
        options=["Да, я это съела", "Нет, не ела"], prompt="Съели ли вы это?",
    )
    denial.metadata["callback_query_id"] = "g11-denial"
    ingress.process_real_inbound(denial)
    before_rows = len(honcho.messages)
    original_rows = [(row.id, row.content, dict(row.metadata)) for row in honcho.messages]
    denial_final = await runtime_turn(pool, denial, ingress)
    commit = attempt["camera_correction_commit"]
    assert attempt["camera_commit"]["annotation"] == full_commit["annotation"]
    assert attempt["camera_commit"]["gateway_session_id"] == full_commit["gateway_session_id"]
    assert commit["kind"] == "denial"
    assert commit["target_event_id"] == original_event
    assert denial_final.text == "Изменение сохранено; баланс обновляется."
    assert "105 ккал" not in denial_final.text
    assert denial_final.metadata["nutrition_append_event_id"] == commit["event_id"]
    assert denial_final.metadata["nutrition_sync_status"] == "pending"
    assert attempt["camera_commit"]["event_id"] == original_event
    assert len(honcho.messages) == before_rows + 2
    assert [(row.id, row.content, dict(row.metadata)) for row in honcho.messages[:before_rows]] == original_rows

    denial_row = honcho.messages[-1]
    assert denial_row.id == commit["event_id"]
    assert denial_row.metadata["camera_candidate_id"] == request["candidate_id"]
    assert denial_row.metadata["camera_operation_id"] == request["candidate_id"]
    assert denial_row.metadata["camera_answer_bound"] == "no"
    assert denial_row.metadata["camera_correction_bound"] is True
    assert denial_row.metadata["camera_original_event_id"] == original_event
    assert denial_row.metadata["client_op_id"] == commit["client_op_id"]
    denied_nutrition = denial_row.metadata["decision_trace"]["annotations"]["nutrition"]
    assert denied_nutrition["record_type"] == "meal_correction"
    assert denied_nutrition["consumption_status"] == "not_consumed"
    assert denied_nutrition["energy_kcal_best"] == 0

    restarted, _, _, _ = _ingress(tmp_path, FakeTelegram())
    pool._camera_ingress = restarted
    repeat = await _native_callback(
        bus, label="Нет, не ела",
        target=restarted._attempts[request["candidate_id"]]["photo_id"],
        options=["Да, я это съела", "Нет, не ела"], prompt="Съели ли вы это?",
    )
    repeat.metadata["callback_query_id"] = "g11-denial"
    restarted.process_real_inbound(repeat)
    rows_before_repeat = len(honcho.messages)
    repeat_final = await runtime_turn(pool, repeat, restarted)
    assert repeat_final.text == "Исправление уже записано."
    assert repeat_final.metadata["nutrition_append_event_id"] == commit["event_id"]
    assert len(honcho.messages) == rows_before_repeat
    fold_input = []
    for row in honcho.messages:
        nutrition = row.metadata.get("decision_trace", {}).get("annotations", {}).get("nutrition")
        if isinstance(nutrition, dict):
            fold_input.append({"event_id": row.id, "root_source_message_id": attempt["camera_commit"]["source_message_id"],
                               "_created_at": row.created_at, "annotation": nutrition})
    folded = _fold_events(fold_input)
    assert folded["consumed"] is False
    assert folded["energy_kcal_best"] == 0
    assert folded["latest_event_id"] == commit["event_id"]
    await ingress.close()
    await restarted.close()


@pytest.mark.asyncio
async def test_dated_new_meal_without_camera_reply_keeps_ordinary_source_binding(
    tmp_path, monkeypatch,
):
    from openharness.evals import TRACE_FINALIZATION
    from openharness.engine.stream_events import AssistantTextDelta

    ingress, request, pool, honcho, _, _ = await _save_typed_confirmation(tmp_path, monkeypatch)
    original_event = ingress._attempts[request["candidate_id"]]["camera_commit"]["event_id"]
    engine = pool._test_bundle.engine
    text = "Да, я съела другой приём пищи вчера"
    message = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=text,
        metadata={"message_id": "ordinary-new-meal-with-date",
                  "_telegram_raw_text": text, "is_group": False,
                  "chat_type": "private"},
    )

    async def ordinary_meal_submit(user_message, *, wellness_actor=None):
        del wellness_actor
        engine.messages.append(user_message)
        engine.turns.append((message.content, [], message.timestamp))
        recorder = engine.decision_trace_recorder
        recorder.trace_requirement_signals(message.content)
        recorder.record(TRACE_FINALIZATION, {
            "schema_version": 1,
            "trace_event_id": "ordinary-new-dated-meal",
            "annotations": {"nutrition": {
                "schema_version": 2, "record_type": "meal_observation",
                "basis": ["owner_statement"], "consumption_status": "consumed",
                "is_estimate": True, "meal_date": "2026-10-05",
                "energy_kcal_best": 340,
                "items": [{"name": "ordinary new meal", "quantity_text": "one meal",
                           "energy_kcal_best": 340}],
            }},
        })
        yield AssistantTextDelta(text="Приём пищи записан.")

    engine.submit_message = ordinary_meal_submit
    ingress.process_real_inbound(message)
    assert message.metadata.get("_camera_ordinary_date_correction") is None
    assert message.metadata.get("_camera_typed_replay_candidate") is None
    assert message.metadata.get("_camera_unbound") is None
    final = await runtime_turn(pool, message, ingress)
    assert final.metadata["nutrition_append_event_id"] == honcho.messages[-1].id
    ordinary_row = honcho.messages[-1]
    stored = ordinary_row.metadata["decision_trace"]["annotations"]["nutrition"]
    assert stored["record_type"] == "meal_observation"
    assert ordinary_row.metadata["source_message_id"] == "ordinary-new-meal-with-date"
    assert ordinary_row.metadata["ingest_source"] == "telegram"
    assert ordinary_row.metadata["confirmation_required"] is False
    assert ordinary_row.metadata.get("camera_candidate_id") is None
    assert ingress._attempts[request["candidate_id"]]["camera_commit"]["event_id"] == original_event
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt_failure", ["missing", "foreign", "append_failure"])
async def test_camera_denial_never_claims_saved_without_valid_receipt(
    tmp_path, monkeypatch, receipt_failure,
):
    from dataclasses import replace

    from openharness.evals import TRACE_FINALIZATION
    from openharness.engine.stream_events import AssistantTextDelta

    ingress, bus, request, pool, honcho, _, _ = await _save_callback_portion(tmp_path, monkeypatch)
    attempt = ingress._attempts[request["candidate_id"]]
    engine = pool._test_bundle.engine

    async def denial_submit(user_message, *, wellness_actor=None):
        del wellness_actor
        engine.messages.append(user_message)
        engine.turns.append((user_message.text, [], pool._active_message.timestamp))
        recorder = engine.decision_trace_recorder
        recorder.trace_requirement_signals(user_message.text)
        recorder.record(TRACE_FINALIZATION, {
            "schema_version": 1,
            "trace_event_id": f"invalid-denial-receipt-{receipt_failure}",
            "annotations": {"nutrition": {
                "schema_version": 2, "record_type": "meal_correction",
                "changed_fields": ["consumption_status", "energy_kcal_best"],
                "consumption_status": "not_consumed", "energy_kcal_best": 0,
            }},
        })
        yield AssistantTextDelta(text="Изменение сохранено; баланс обновляется.")

    engine.submit_message = denial_submit
    denial = await _native_callback(
        bus, label="Нет, не ела", target=attempt["photo_id"],
        options=["Да, я это съела", "Нет, не ела"], prompt="Съели ли вы это?",
    )
    denial.metadata["callback_query_id"] = f"g11-invalid-{receipt_failure}"
    ingress.process_real_inbound(denial)
    before = len(honcho.messages)
    backend = pool._shadow_backend_for_scope(None)
    append_exchange = backend.append_exchange

    async def failed_or_corrupt_append(*args, **kwargs):
        if receipt_failure == "append_failure":
            raise OSError("synthetic denial append failure")
        if receipt_failure == "missing":
            return None
        receipt = await append_exchange(*args, **kwargs)
        return replace(receipt, assistant_metadata={
            **receipt.assistant_metadata, "tenant_id": "foreign-owner",
        })

    monkeypatch.setattr(backend, "append_exchange", failed_or_corrupt_append)
    expected_error = OSError if receipt_failure == "append_failure" else ValueError
    with pytest.raises(expected_error):
        await runtime_turn(pool, denial, ingress)
    assert attempt.get("camera_correction_commit") is None
    assert denial.metadata.get("nutrition_append_event_id") is None
    if receipt_failure == "foreign":
        assert len(honcho.messages) == before + 2
    else:
        assert len(honcho.messages) == before
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt_failure", ["missing", "foreign"])
async def test_typed_repeat_never_claims_saved_without_its_exact_receipt(
    tmp_path, monkeypatch, receipt_failure
):
    ingress, request, pool, honcho, _, first_answer = await _save_typed_confirmation(
        tmp_path, monkeypatch
    )
    backend = pool._shadow_backend_for_scope(None)
    reconcile = backend.reconcile_durable_exchange

    async def altered_receipt(user_op, assistant_op):
        receipt = await reconcile(user_op, assistant_op)
        if receipt_failure == "missing":
            return None
        from dataclasses import replace

        return replace(receipt, assistant_metadata={
            **receipt.assistant_metadata, "tenant_id": "another-owner",
        })

    pool._shadow_backend_for_scope = lambda _scope: SimpleNamespace(
        reconcile_durable_exchange=altered_receipt
    )
    replay = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=first_answer.content,
        metadata={"message_id": "typed-camera-invalid-receipt",
                  "reply_to_message_id": str(ingress._attempts[request["candidate_id"]]["photo_id"]),
                  "_telegram_raw_text": first_answer.content, "is_group": False,
                  "chat_type": "private"},
    )
    ingress.process_real_inbound(replay)
    pool._active_message = replay
    updates = [
        update async for update in pool.stream_message(replay, ingress.config.session_key)
    ]
    errors = [update for update in updates if update.kind == "error"]
    assert errors
    assert all("уже записана" not in update.text.casefold() for update in updates)
    assert len(honcho.messages) == 2
    assert ingress._attempts[request["candidate_id"]]["camera_commit"]["event_id"] == "honcho-2"
    await ingress.close()


@pytest.mark.asyncio
async def test_runtime_transport_retry_survives_camera_journal_restart(tmp_path, monkeypatch):
    ingress, bus, request, pool, honcho, saved_portion, saved_final = (
        await _save_callback_portion(tmp_path, monkeypatch)
    )
    restarted, _, _, _ = _ingress(tmp_path, FakeTelegram())
    pool._camera_ingress = restarted
    retry = await _native_callback(
        bus, label="2 кусочка",
        target=restarted._attempts[request["candidate_id"]]["reply_ids"][0],
        options=["1 кусочек", "2 кусочка", "Половину порции"],
        prompt="Вы съели это? Сколько примерно вы съели?",
    )
    retry.metadata["callback_query_id"] = "callback-portion-1"
    restarted.process_real_inbound(retry)
    assert retry.metadata["_camera_existing_meal_replay"] is True
    public_final = await runtime_turn(pool, retry, restarted)
    assert public_final.text == "Эта порция уже записана."
    assert public_final.metadata["nutrition_append_event_id"] == saved_final.metadata[
        "nutrition_append_event_id"
    ]
    assert len(honcho.messages) == 4
    assert restarted._attempts[request["candidate_id"]]["camera_commit"]["event_id"] == "honcho-4"
    await ingress.close()
    await restarted.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["missing", "invalid", "foreign", "stale"]
)
async def test_runtime_replay_never_claims_saved_without_exact_receipt(
    tmp_path, monkeypatch, failure
):
    ingress, bus, request, pool, honcho, saved_portion, _ = await _save_callback_portion(
        tmp_path, monkeypatch
    )
    backend = pool._shadow_backend_for_scope(None)
    reconcile = backend.reconcile_durable_exchange

    async def altered_receipt(user_op, assistant_op):
        receipt = await reconcile(user_op, assistant_op)
        if failure == "missing":
            return None
        if failure == "invalid":
            from dataclasses import replace

            return replace(receipt, user_content=None)
        if failure == "foreign":
            from dataclasses import replace

            return replace(receipt, assistant_metadata={
                **receipt.assistant_metadata, "tenant_id": "another-owner",
            })
        if failure == "stale":
            from dataclasses import replace

            metadata = dict(receipt.assistant_metadata)
            metadata["source_message_id"] = "old-source"
            return replace(receipt, assistant_metadata=metadata)
        return receipt

    pool._shadow_backend_for_scope = lambda _scope: SimpleNamespace(
        reconcile_durable_exchange=altered_receipt
    )
    replay = await _native_callback(
        bus, label="2 кусочка",
        target=ingress._attempts[request["candidate_id"]]["reply_ids"][0],
        options=["1 кусочек", "2 кусочка", "Половину порции"],
        prompt="Вы съели это? Сколько примерно вы съели?",
    )
    replay.metadata["callback_query_id"] = "callback-portion-replay"
    ingress.process_real_inbound(replay)
    assert replay.metadata.get("_camera_existing_meal_replay") is True
    pool._active_message = replay
    updates = [
        update async for update in pool.stream_message(replay, ingress.config.session_key)
    ]
    public_error = next(update for update in updates if update.kind == "error")
    assert "записана" not in public_error.text.casefold()
    assert "не добавлена" in public_error.text.casefold()
    assert len(honcho.messages) == 4
    assert ingress._attempts[request["candidate_id"]]["camera_commit"]["event_id"] == "honcho-4"
    assert saved_portion.metadata["_camera_turn_id"] == replay.metadata["_camera_turn_id"]
    await ingress.close()


@pytest.mark.asyncio
async def test_changed_native_portion_appends_a_bound_immutable_correction(
    tmp_path, monkeypatch
):
    from openharness.api.usage import UsageSnapshot
    from openharness.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
    from openharness.engine.stream_events import (
        AssistantTurnComplete,
        ToolExecutionCompleted,
        ToolExecutionStarted,
    )
    from openharness.evals import TRACE_FINALIZATION

    ingress, bus, request, pool, honcho, saved_portion, saved_final = (
        await _save_callback_portion(tmp_path, monkeypatch)
    )
    attempt = ingress._attempts[request["candidate_id"]]
    original_event_id = saved_final.metadata["nutrition_append_event_id"]

    def row_state(item):
        return (
            item.id, item.content, item.peer_id, item.session_id,
            json.loads(json.dumps(item.metadata, ensure_ascii=False, allow_nan=False)),
            item.created_at, item.workspace_id, item.token_count,
        )

    original_rows = [row_state(item) for item in honcho.messages]
    original_meal = next(
        item for item in honcho.messages
        if item.id == original_event_id
    )
    assert (
        original_meal.metadata["decision_trace"]["annotations"]["nutrition"]
        ["consumption_status"] == "consumed"
    )
    engine = pool._test_bundle.engine
    pool._test_bundle.current_settings = lambda: SimpleNamespace(model="offline-camera-fixture")
    turns_before_correction = len(engine.turns)

    async def correction_turn(user_message):
        engine.messages.append(user_message)
        engine.turns.append((user_message.text, [], saved_portion.timestamp))
        tool_id = "toolu-camera-portion-correction"
        tool_input = {
            "kind": "trace_finalization",
            "payload": {
                "schema_version": 1,
                "trace_event_id": "camera-portion-correction",
                "annotations": {"nutrition": {
                    "schema_version": 2,
                    "record_type": "meal_correction",
                    "changed_fields": ["items", "energy_kcal_best"],
                    "items": [{
                        "name": "Мягкий творог Синтетик 5%, упаковка 125 г",
                        "quantity_text": "1 piece",
                        "energy_kcal_best": 53,
                    }],
                    "energy_kcal_best": 53,
                }},
            },
        }
        engine.decision_trace_recorder.record(
            TRACE_FINALIZATION, tool_input["payload"]
        )
        # The query engine completes the tool-use API message before it makes
        # the follow-up call that supplies the user-facing assistant final.
        yield AssistantTurnComplete(
            message=ConversationMessage(role="assistant", content=[
                ToolUseBlock(id=tool_id, name="trace", input=tool_input),
            ]),
            usage=UsageSnapshot(),
        )
        yield ToolExecutionStarted(
            tool_name="trace", tool_input=tool_input, tool_call_id=tool_id,
        )
        yield ToolExecutionCompleted(
            tool_name="trace", output="trace accepted", is_error=False,
            tool_call_id=tool_id,
        )
        yield AssistantTurnComplete(
            message=ConversationMessage(role="assistant", content=[
                TextBlock(text="Уменьшила учтённую порцию."),
            ]),
            usage=UsageSnapshot(),
        )

    engine.submit_message = correction_turn
    changed = await _native_callback(
        bus, label="1 кусочек", target=attempt["reply_ids"][0],
        options=["1 кусочек", "2 кусочка", "Половину порции"],
        prompt="Вы съели это? Сколько примерно вы съели?",
    )
    changed.metadata["callback_query_id"] = "changed-native-portion-correction"
    ingress.process_real_inbound(changed)
    assert changed.metadata.get("_camera_existing_meal_replay") is True
    final = await runtime_turn(pool, changed, ingress)

    assert final.text == "Изменение сохранено; баланс обновляется."
    assert final.metadata["nutrition_append_event_id"] != original_event_id
    assert len(engine.turns) == turns_before_correction + 1
    assert len(honcho.messages) == len(original_rows) + 2
    assert [row_state(item) for item in honcho.messages[:len(original_rows)]] == original_rows
    correction_user, correction_assistant = honcho.messages[-2:]
    assert correction_user.metadata["client_op_id"] == f"{changed.metadata['_camera_turn_id']}:user"
    assert correction_assistant.metadata["camera_correction_bound"] is True
    assert correction_assistant.metadata["camera_original_event_id"] == original_event_id
    stored = correction_assistant.metadata["decision_trace"]["annotations"]["nutrition"]
    assert stored["record_type"] == "meal_correction"
    assert "consumption_status" not in stored
    assert stored["changed_fields"] == ["items", "energy_kcal_best"]
    assert stored["items"][0]["quantity_text"] == "1 piece"
    assert stored["energy_kcal_best"] == 53
    assert "meal_at" not in stored and "meal_date" not in stored
    nutrition_types = [
        item.metadata["decision_trace"]["annotations"]["nutrition"]["record_type"]
        for item in honcho.messages
        if item.metadata.get("role") == "assistant"
        and isinstance(item.metadata.get("decision_trace"), dict)
        and isinstance(item.metadata["decision_trace"].get("annotations"), dict)
        and isinstance(
            item.metadata["decision_trace"]["annotations"].get("nutrition"), dict
        )
    ]
    assert nutrition_types.count("meal_observation") == 1
    assert nutrition_types.count("meal_correction") == 1
    assert attempt["camera_correction_commit"]["target_event_id"] == original_event_id
    assert attempt["camera_commit"]["event_id"] == original_event_id
    assert saved_portion.metadata["_camera_turn_id"] == attempt["answer_turn_id"]
    assert attempt["camera_correction"] == "completed"
    await ingress.close()


@pytest.mark.asyncio
async def test_completed_replay_stays_unbound_during_denial_or_uncertain_delivery(
    tmp_path, monkeypatch
):
    ingress, bus, request, _, _, _, _ = await _save_callback_portion(tmp_path, monkeypatch)
    attempt = ingress._attempts[request["candidate_id"]]
    denial = await _native_callback(
        bus, label="Нет, не ел(а)", target=attempt["photo_id"],
        options=["Да, я это съел(а)", "Нет, не ел(а)"],
        prompt="Съели ли вы это?",
    )
    denial.metadata["callback_query_id"] = "callback-denial"
    ingress.process_real_inbound(denial)
    assert denial.metadata["_camera_correction"] is CAMERA_AUTHORITY
    assert attempt["camera_correction"] == "answering"

    replay_during_correction = await _native_callback(
        bus, label="2 кусочка", target=attempt["reply_ids"][0],
        options=["1 кусочек", "2 кусочка", "Половину порции"],
        prompt="Вы съели это? Сколько примерно вы съели?",
    )
    replay_during_correction.metadata["callback_query_id"] = "callback-during-correction"
    ingress.process_real_inbound(replay_during_correction)
    assert replay_during_correction.metadata.get("_camera_existing_meal_replay") is None

    attempt["camera_correction"] = None
    attempt["state"] = "delivery_unknown"
    ingress._save_attempts()
    replay_after_uncertain_delivery = await _native_callback(
        bus, label="2 кусочка", target=attempt["reply_ids"][0],
        options=["1 кусочек", "2 кусочка", "Половину порции"],
        prompt="Вы съели это? Сколько примерно вы съели?",
    )
    replay_after_uncertain_delivery.metadata["callback_query_id"] = "callback-uncertain"
    ingress.process_real_inbound(replay_after_uncertain_delivery)
    assert replay_after_uncertain_delivery.metadata.get("_camera_existing_meal_replay") is None
    await ingress.close()


async def _complete_callback_meal(ingress, root, bus, *, label: str, event_id: str):
    request = _candidate(root)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    attempt = ingress._attempts[request["candidate_id"]]
    callback = await _native_callback(
        bus,
        label=label,
        target=attempt["photo_id"],
        options=["Да, я это съел(а)", "Нет, не ел(а)"],
        prompt="Съели ли вы это? Фото сделано 2026-10-01.",
    )
    callback.metadata["callback_query_id"] = "callback-1"
    ingress.process_real_inbound(callback)
    assert callback.metadata["_camera_answer"] == "yes"
    receipt, nutrition = _observed_camera_meal_receipt(
        request["candidate_id"], callback.metadata["_camera_turn_id"],
        request["capture_time"], event_id=event_id,
    )
    receipt = type(receipt)(
        user_message_id=receipt.user_message_id,
        assistant_message_id=receipt.assistant_message_id,
        user_client_op_id=receipt.user_client_op_id,
        assistant_client_op_id=receipt.assistant_client_op_id,
        assistant_metadata=receipt.assistant_metadata,
        assistant_content="Записано.",
        user_content=label,
    )
    ingress.record_committed_meal(callback, receipt, nutrition)
    ingress.complete(callback, recorded=True)
    ingress.note_assistant_receipt(
        OutboundMessage(
            channel="telegram", chat_id="123", content="Записано",
            metadata={"_camera_authority": CAMERA_AUTHORITY,
                      "_camera_candidate_id": request["candidate_id"],
                      "_camera_final": CAMERA_AUTHORITY,
                      "_camera_turn_id": callback.metadata["_camera_turn_id"]},
        ),
        OutboundDeliveryReceipt(channel="telegram", chat_id="123", native_message_ids=(91,)),
    )
    return request, callback, receipt


def test_completed_same_option_callback_routes_to_verified_existing_meal(tmp_path):
    async def run():
        ingress, root, bus, _ = _ingress(tmp_path)
        request, original, receipt = await _complete_callback_meal(
            ingress, root, bus, label="Да, я это съел(а)", event_id="event-1"
        )
        replay = await _native_callback(
            bus,
            label="Да, я это съел(а)",
            target=ingress._attempts[request["candidate_id"]]["photo_id"],
            options=["Да, я это съел(а)", "Нет, не ел(а)"],
            prompt="Съели ли вы это? Фото сделано 2026-10-01.",
        )
        replay.metadata["callback_query_id"] = "callback-2"
        ingress.process_real_inbound(replay)
        assert replay.metadata["_camera_existing_meal_replay"] is True
        assert replay.metadata["_camera_ingress_callback_eligible"] is True
        assert replay.metadata["_camera_turn_id"] == original.metadata["_camera_turn_id"]
        assert replay.metadata.get("_camera_unbound") is None
        assert ingress._attempts[request["candidate_id"]]["camera_commit"]["event_id"] == "event-1"

        restarted, _, _, _ = _ingress(tmp_path)
        retry = await _native_callback(
            bus,
            label="Да, я это съел(а)",
            target=restarted._attempts[request["candidate_id"]]["photo_id"],
            options=["Да, я это съел(а)", "Нет, не ел(а)"],
            prompt="Съели ли вы это? Фото сделано 2026-10-01.",
        )
        retry.metadata["callback_query_id"] = "callback-2"
        restarted.process_real_inbound(retry)
        assert retry.metadata["_camera_existing_meal_replay"] is True
        assert restarted._attempts[request["candidate_id"]]["camera_commit"]["event_id"] == "event-1"
        assert receipt.user_content == "Да, я это съел(а)"
        await ingress.close()
        await restarted.close()

    asyncio.run(run())


def test_context_source_identical_typed_repeat_uses_saved_receipt(tmp_path, monkeypatch):
    async def run():
        ingress, request, pool, honcho, saved_final, first_answer = (
            await _save_typed_confirmation(
                tmp_path, monkeypatch, "Да, я это съела", reply_to_photo=False,
            )
        )
        attempt = ingress._attempts[request["candidate_id"]]
        assert attempt.get("context_interrupted", False) is False
        replay = InboundMessage(
            channel="telegram", sender_id="123", chat_id="123",
            content="Да, я это съела",
            metadata={"message_id": "context-camera-repeat-2",
                      "_telegram_raw_text": "Да, я это съела",
                      "is_group": False, "chat_type": "private"},
        )
        ingress.process_real_inbound(replay)
        assert replay.metadata.get("_camera_typed_replay_candidate") == request["candidate_id"]
        assert "reply_to_message_id" not in replay.metadata
        before = [(item.id, item.content) for item in honcho.messages]
        final = await runtime_turn(pool, replay, ingress)
        assert final.text == "Эта порция уже записана."
        assert [(item.id, item.content) for item in honcho.messages] == before
        assert len(honcho.messages) == 2
        assert attempt["camera_commit"]["event_id"] == saved_final.metadata[
            "nutrition_append_event_id"
        ]
        await ingress.close()

    asyncio.run(run())


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_target", ["unrelated", "old_camera_photo"])
async def test_new_owner_photo_reply_interrupts_context_replay_after_journal_reload(
    tmp_path, monkeypatch, reply_target,
):
    ingress, request, pool, honcho, saved_final, first_answer = await _save_typed_confirmation(
        tmp_path, monkeypatch, reply_to_photo=False,
    )
    attempt = ingress._attempts[request["candidate_id"]]
    target = "999" if reply_target == "unrelated" else str(attempt["photo_id"])
    new_photo = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content="new owner photo",
        metadata={"message_id": f"new-owner-photo-{reply_target}",
                  "reply_to_message_id": target, "is_group": False, "chat_type": "private"},
        media=["new-owner-photo.jpg"],
    )
    ingress.process_real_inbound(new_photo)
    assert new_photo.media == ["new-owner-photo.jpg"]
    assert attempt["context_interrupted"] is True

    restarted, _, _, _ = _ingress(tmp_path, FakeTelegram())
    pool._camera_ingress = restarted
    assert restarted._attempts[request["candidate_id"]]["context_interrupted"] is True
    replay = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=first_answer.content,
        metadata={"message_id": f"generic-repeat-after-{reply_target}-photo",
                  "_telegram_raw_text": first_answer.content,
                  "is_group": False, "chat_type": "private"},
    )
    restarted.process_real_inbound(replay)
    assert replay.metadata.get("_camera_typed_replay_candidate") is None
    before_turns = len(pool._test_bundle.engine.turns)
    result = await runtime_turn(pool, replay, restarted)
    assert result.text != "Эта порция уже записана."
    assert len(pool._test_bundle.engine.turns) == before_turns + 1
    assert len(honcho.messages) == 2
    assert restarted._attempts[request["candidate_id"]]["camera_commit"]["event_id"] == saved_final.metadata[
        "nutrition_append_event_id"
    ]
    await ingress.close()
    await restarted.close()


@pytest.mark.asyncio
async def test_foreign_owner_photo_reply_does_not_interrupt_context_replay(tmp_path, monkeypatch):
    ingress, request, pool, honcho, saved_final, first_answer = await _save_typed_confirmation(
        tmp_path, monkeypatch, reply_to_photo=False,
    )
    attempt = ingress._attempts[request["candidate_id"]]
    foreign_photo = InboundMessage(
        channel="telegram", sender_id="other-owner", chat_id="123", content="foreign photo",
        metadata={"message_id": "foreign-photo", "reply_to_message_id": str(attempt["photo_id"])},
        media=["foreign-photo.jpg"],
    )
    ingress.process_real_inbound(foreign_photo)
    assert attempt.get("context_interrupted", False) is False
    replay = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=first_answer.content,
        metadata={"message_id": "same-owner-repeat-after-foreign-photo",
                  "_telegram_raw_text": first_answer.content,
                  "is_group": False, "chat_type": "private"},
    )
    ingress.process_real_inbound(replay)
    assert replay.metadata.get("_camera_typed_replay_candidate") == request["candidate_id"]
    result = await runtime_turn(pool, replay, ingress)
    assert result.text == "Эта порция уже записана."
    assert len(honcho.messages) == 2
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("current_route", ["reply", "callback"])
@pytest.mark.parametrize("after_new_photo", [False, True])
async def test_context_origin_receipt_reconciles_explicit_source_route_after_restart(
    tmp_path, monkeypatch, current_route, after_new_photo,
):
    ingress, request, pool, honcho, saved_final, first_answer = await _save_typed_confirmation(
        tmp_path, monkeypatch, reply_to_photo=False,
    )
    original = honcho.messages[1]
    assert original.metadata["camera_route"] == "context"
    assert "camera_reply_to_native_message_id" not in original.metadata
    attempt = ingress._attempts[request["candidate_id"]]
    if after_new_photo:
        new_photo = InboundMessage(
            channel="telegram", sender_id="123", chat_id="123", content="new owner photo",
            metadata={"message_id": f"new-photo-before-{current_route}",
                      "reply_to_message_id": str(attempt["photo_id"]),
                      "is_group": False, "chat_type": "private"},
            media=["new-owner-photo.jpg"],
        )
        ingress.process_real_inbound(new_photo)
        assert attempt["context_interrupted"] is True
    restarted, _, bus, _ = _ingress(tmp_path, FakeTelegram())
    pool._camera_ingress = restarted
    assert restarted._attempts[request["candidate_id"]].get("context_interrupted", False) is after_new_photo
    if current_route == "reply":
        replay = InboundMessage(
            channel="telegram", sender_id="123", chat_id="123", content=first_answer.content,
            metadata={"message_id": "context-origin-explicit-reply-replay",
                      "reply_to_message_id": str(attempt["photo_id"]),
                      "_telegram_raw_text": first_answer.content,
                      "is_group": False, "chat_type": "private"},
        )
    else:
        replay = await _native_callback(
            bus, label=first_answer.content, target=attempt["photo_id"],
            options=[first_answer.content, "Нет, не ела"], prompt="Вы съели это?",
        )
        replay.metadata["callback_query_id"] = "context-origin-callback-replay"
    restarted.process_real_inbound(replay)
    if current_route == "reply":
        assert replay.metadata.get("_camera_typed_replay_candidate") == request["candidate_id"]
        assert replay.metadata.get("_camera_route") == "reply"
    else:
        assert replay.metadata.get("_camera_existing_meal_replay") is True
        assert replay.metadata.get("_camera_ingress_callback_eligible") is True
    before_turns = len(pool._test_bundle.engine.turns)
    result = await runtime_turn(pool, replay, restarted)
    assert result.text == "Эта порция уже записана."
    assert result.metadata["nutrition_append_event_id"] == saved_final.metadata[
        "nutrition_append_event_id"
    ]
    assert len(honcho.messages) == 2
    assert len(pool._test_bundle.engine.turns) == before_turns
    assert restarted._attempts[request["candidate_id"]]["camera_commit"]["event_id"] == saved_final.metadata[
        "nutrition_append_event_id"
    ]
    await ingress.close()
    await restarted.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("current_route", ["reply", "callback"])
@pytest.mark.parametrize(
    "receipt_failure",
    ["missing", "foreign_owner", "foreign_session", "stale_source", "bad_operation",
     "bad_capture_time", "bad_original_route", "bad_original_native_binding",
     "present_null_native_binding"],
)
async def test_cross_route_context_receipt_stays_fail_closed_after_journal_restart(
    tmp_path, monkeypatch, current_route, receipt_failure,
):
    ingress, request, pool, honcho, _, first_answer = await _save_typed_confirmation(
        tmp_path, monkeypatch, reply_to_photo=False,
    )
    attempt = ingress._attempts[request["candidate_id"]]
    restarted, _, bus, _ = _ingress(tmp_path, FakeTelegram())
    pool._camera_ingress = restarted
    if current_route == "reply":
        replay = InboundMessage(
            channel="telegram", sender_id="123", chat_id="123", content=first_answer.content,
            metadata={"message_id": f"bad-cross-route-{receipt_failure}",
                      "reply_to_message_id": str(attempt["photo_id"]),
                      "_telegram_raw_text": first_answer.content,
                      "is_group": False, "chat_type": "private"},
        )
    else:
        replay = await _native_callback(
            bus, label=first_answer.content, target=attempt["photo_id"],
            options=[first_answer.content, "Нет, не ела"], prompt="Вы съели это?",
        )
        replay.metadata["callback_query_id"] = f"bad-cross-route-{receipt_failure}"
    restarted.process_real_inbound(replay)
    backend = pool._shadow_backend_for_scope(None)
    reconcile = backend.reconcile_durable_exchange

    async def altered_receipt(user_op, assistant_op):
        receipt = await reconcile(user_op, assistant_op)
        if receipt_failure == "missing":
            return None
        from dataclasses import replace

        assistant_metadata = dict(receipt.assistant_metadata)
        trace = dict(assistant_metadata.get("decision_trace", {}))
        annotations = dict(trace.get("annotations", {}))
        nutrition = dict(annotations.get("nutrition", {}))
        if receipt_failure == "foreign_owner":
            assistant_metadata["tenant_id"] = "another-owner"
        elif receipt_failure == "foreign_session":
            assistant_metadata["gateway_session_id"] = "another-session"
        elif receipt_failure == "stale_source":
            assistant_metadata["source_message_id"] = "stale-source"
        elif receipt_failure == "bad_operation":
            assistant_metadata["camera_operation_id"] = "another-operation"
        elif receipt_failure == "bad_capture_time":
            nutrition["meal_at"] = "2025-01-01T00:00:00+00:00"
            annotations["nutrition"] = nutrition
            trace["annotations"] = annotations
            assistant_metadata["decision_trace"] = trace
        elif receipt_failure == "bad_original_route":
            assistant_metadata["camera_route"] = "reply"
        elif receipt_failure == "bad_original_native_binding":
            assistant_metadata["camera_reply_to_native_message_id"] = "999"
        elif receipt_failure == "present_null_native_binding":
            assistant_metadata["camera_reply_to_native_message_id"] = None
        return replace(receipt, assistant_metadata=assistant_metadata)

    pool._shadow_backend_for_scope = lambda _scope: SimpleNamespace(
        reconcile_durable_exchange=altered_receipt
    )
    pool._active_message = replay
    updates = [update async for update in pool.stream_message(replay, restarted.config.session_key)]
    assert not any(
        update.kind == "final" and update.text == "Эта порция уже записана."
        for update in updates
    )
    assert any(update.kind == "error" for update in updates)
    assert len(honcho.messages) == 2
    assert restarted._attempts[request["candidate_id"]]["camera_commit"]["event_id"] == "honcho-2"
    await ingress.close()
    await restarted.close()


def test_completed_meal_does_not_rebind_different_photo_owner_or_callback(tmp_path):
    async def run():
        ingress, root, bus, _ = _ingress(tmp_path)
        request, _, _ = await _complete_callback_meal(
            ingress, root, bus, label="Да, я это съел(а)", event_id="event-1"
        )
        attempt = ingress._attempts[request["candidate_id"]]
        different_photo = await _native_callback(
            bus, label="Да, я это съел(а)", target=999,
            options=["Да, я это съел(а)", "Нет, не ел(а)"],
            prompt="Съели ли вы это?",
        )
        ingress.process_real_inbound(different_photo)
        assert "_camera_existing_meal_replay" not in different_photo.metadata
        assert "_camera_unbound" not in different_photo.metadata

        foreign = InboundMessage(
            channel="telegram", sender_id="other-owner", chat_id="123",
            content="Да, я это съел(а)", metadata={
                "callback_query": True, "native_message_id": attempt["photo_id"],
                "callback_data": "ask:0",
            },
        )
        ingress.process_real_inbound(foreign)
        assert "_camera_existing_meal_replay" not in foreign.metadata

        unrelated = InboundMessage(
            channel="telegram", sender_id="123", chat_id="123", content="Привет",
            metadata={"callback_query": True, "native_message_id": attempt["photo_id"],
                      "callback_data": "menu:help"},
        )
        ingress.process_real_inbound(unrelated)
        assert "_camera_existing_meal_replay" not in unrelated.metadata
        await ingress.close()

    asyncio.run(run())


def test_durable_reconciliation_retains_original_user_option_text():
    async def run():
        honcho = _Honcho()
        backend = ShadowMemoryBackend(_Base(), honcho, conversation_learning=True)
        user_metadata, assistant_metadata = _metadata("candidate-1")
        receipt = await backend.append_exchange(
            "Small portion", "Saved", user_metadata=user_metadata,
            assistant_metadata=assistant_metadata, durable=True,
        )
        reconciled = await backend.reconcile_durable_exchange(
            receipt.user_client_op_id, receipt.assistant_client_op_id
        )
        assert receipt.user_content == "Small portion"
        assert reconciled.user_content == "Small portion"
        assert reconciled == receipt
        assert len(honcho.messages) == 2

    asyncio.run(run())


def test_camera_probe_requires_public_existing_meal_response_and_immutable_event():
    replay = SimpleNamespace(metadata={
        "_camera_existing_meal_replay": True,
        "_camera_candidate_id": "candidate-1",
        "_camera_turn_id": "turn-1",
    })
    commit = {"event_id": "event-1", "record_type": "meal_observation"}
    outbound = SimpleNamespace(
        content="Эта порция уже записана.",
        metadata={"nutrition_append_event_id": "event-1"},
    )
    status = _validate_completed_photo_replay(
        replay=replay,
        candidate_id="candidate-1",
        turn_id="turn-1",
        delivered=[(outbound, object())],
        delivery_receipt=object(),
        existing_event_id="event-1",
        original_commit=commit,
        current_commit=dict(commit),
    )
    assert status == "Эта порция уже записана."


def test_camera_probe_accepts_typed_replay_binding_without_native_replay_marker():
    replay = SimpleNamespace(metadata={
        "_camera_typed_replay": True,
        "_camera_candidate_id": "candidate-1",
        "_camera_turn_id": "turn-1",
    })
    commit = {"event_id": "event-1", "record_type": "meal_observation"}
    outbound = SimpleNamespace(
        content="Эта порция уже записана.",
        metadata={"nutrition_append_event_id": "event-1"},
    )
    assert _validate_completed_photo_replay(
        replay=replay,
        candidate_id="candidate-1",
        turn_id="turn-1",
        delivered=[(outbound, object())],
        delivery_receipt=object(),
        existing_event_id="event-1",
        original_commit=commit,
        current_commit=dict(commit),
        typed_replay=True,
    ) == outbound.content


@pytest.mark.parametrize("bad_evidence", ["unbound", "status", "event", "delivery", "commit"])
def test_camera_probe_rejects_incoherent_replay_evidence(bad_evidence):
    replay_metadata = {
        "_camera_existing_meal_replay": True,
        "_camera_candidate_id": "candidate-1",
        "_camera_turn_id": "turn-1",
    }
    if bad_evidence == "unbound":
        replay_metadata["_camera_unbound"] = object()
    replay = SimpleNamespace(metadata=replay_metadata)
    commit = {"event_id": "event-1"}
    outbound = SimpleNamespace(
        content="Эта порция уже записана.",
        metadata={"nutrition_append_event_id": "wrong-event" if bad_evidence == "event" else "event-1"},
    )
    with pytest.raises(AssertionError):
        _validate_completed_photo_replay(
            replay=replay,
            candidate_id="candidate-1",
            turn_id="turn-1",
            delivered=[(
                SimpleNamespace(
                    content="Не записано" if bad_evidence == "status" else outbound.content,
                    metadata=outbound.metadata,
                ),
                object(),
            )],
            delivery_receipt=None if bad_evidence == "delivery" else object(),
            existing_event_id="event-1",
            original_commit=commit,
            current_commit={"event_id": "event-2"} if bad_evidence == "commit" else dict(commit),
        )

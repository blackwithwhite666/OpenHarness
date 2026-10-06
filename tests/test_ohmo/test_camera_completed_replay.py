"""Focused regressions for a completed native Camera option replay."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openharness.channels.bus.events import InboundMessage, OutboundDeliveryReceipt, OutboundMessage

from ohmo.gateway.camera import CAMERA_AUTHORITY
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

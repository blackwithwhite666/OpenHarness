"""Focused regressions for a completed native Camera option replay."""

from __future__ import annotations

import asyncio
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


async def _save_typed_confirmation(tmp_path, monkeypatch):
    ingress, root, bus, _ = _ingress(tmp_path, FakeTelegram())
    request = _candidate(root)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    await bus.consume_inbound()
    honcho = RuntimeHoncho()
    pool = runtime_pool(tmp_path, ingress, honcho, monkeypatch)
    photo_id = ingress._attempts[request["candidate_id"]]["photo_id"]
    answer = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content="Да, я это съела",
        metadata={"message_id": "typed-camera-first-confirmation",
                  "reply_to_message_id": str(photo_id),
                  "_telegram_raw_text": "Да, я это съела", "is_group": False,
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
async def test_changed_typed_reply_reaches_existing_runtime_flow(
    tmp_path, monkeypatch, changed_text
):
    ingress, request, pool, honcho, _, _ = await _save_typed_confirmation(tmp_path, monkeypatch)
    attempt = ingress._attempts[request["candidate_id"]]
    turns_before = len(pool._test_bundle.engine.turns)
    changed = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=changed_text,
        metadata={"message_id": "typed-camera-changed-reply",
                  "reply_to_message_id": str(attempt["photo_id"]),
                  "_telegram_raw_text": changed_text, "is_group": False,
                  "chat_type": "private"},
    )
    ingress.process_real_inbound(changed)
    assert changed.metadata.get("_camera_typed_replay_candidate") == request["candidate_id"]
    result = await runtime_turn(pool, changed, ingress)
    assert changed.metadata.get("_camera_typed_replay_candidate") is None
    assert result.text != "Эта порция уже записана."
    assert changed.metadata["reply_to_message_id"] == str(attempt["photo_id"])
    assert len(pool._test_bundle.engine.turns) == turns_before + 1
    assert len(honcho.messages) == 2
    assert attempt["camera_commit"]["event_id"] == "honcho-2"
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
    "failure", ["missing", "invalid", "foreign", "stale", "changed_portion"]
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
    label = "1 кусочек" if failure == "changed_portion" else "2 кусочка"
    replay = await _native_callback(
        bus, label=label,
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

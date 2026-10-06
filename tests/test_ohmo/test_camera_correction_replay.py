"""Natural Camera quantity-correction replay and later denial regressions."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from openharness.channels.bus.events import InboundMessage

from tests.test_ohmo.test_camera_completed_replay import _save_callback_portion
from tests.test_ohmo.test_camera_ingress import _ingress, _native_callback
from tests.test_ohmo.test_nutrition_dialogue_stream import FakeTelegram, _turn as runtime_turn


@pytest.mark.asyncio
async def test_quantity_correction_repeat_restart_then_denial(tmp_path, monkeypatch):
    from openharness.engine.stream_events import AssistantTextDelta
    from openharness.evals import TRACE_FINALIZATION
    from ohmo.evals.nutrition_persistence import _fold_events

    tmp_path.mkdir(parents=True, exist_ok=True)
    ingress, bus, request, pool, honcho, saved_portion, saved_final = (
        await _save_callback_portion(tmp_path, monkeypatch)
    )
    attempt = ingress._attempts[request["candidate_id"]]
    original_event = saved_final.metadata["nutrition_append_event_id"]
    original_rows = [(row.id, row.content, dict(row.metadata), row.created_at) for row in honcho.messages]
    engine = pool._test_bundle.engine

    async def camera_correction(user_message):
        engine.messages.append(user_message)
        engine.turns.append((user_message.text, [], saved_portion.timestamp))
        denied = pool._active_message.metadata.get("_camera_portion_correction") is None
        selected = pool._active_message.metadata.get("native_keyboard_selected_label", "1 piece")
        calories = 27 if selected == "Половину порции" else 53
        nutrition = (
            {"schema_version": 2, "record_type": "meal_correction",
             "changed_fields": ["consumption_status", "energy_kcal_best"],
             "consumption_status": "not_consumed", "energy_kcal_best": 0}
            if denied else
            {"schema_version": 2, "record_type": "meal_correction",
             "changed_fields": ["items", "energy_kcal_best"],
             "items": [{"name": "Мягкий творог Синтетик 5%, упаковка 125 г",
                        "quantity_text": selected, "energy_kcal_best": calories}],
             "energy_kcal_best": calories}
        )
        engine.decision_trace_recorder.record(
            TRACE_FINALIZATION,
            {"schema_version": 1, "trace_event_id": f"camera-correction-{len(engine.turns)}",
             "annotations": {"nutrition": nutrition}},
        )
        yield AssistantTextDelta(text="Исправление сохранено.")

    engine.submit_message = camera_correction
    changed = await _native_callback(
        bus, label="1 кусочек", target=attempt["reply_ids"][0],
        options=["1 кусочек", "2 кусочка", "Половину порции"],
        prompt="Вы съели это? Сколько примерно вы съели?",
    )
    changed.metadata["callback_query_id"] = "correction-1"
    ingress.process_real_inbound(changed)
    corrected = await runtime_turn(pool, changed, ingress)
    correction_turn = changed.metadata["_camera_turn_id"]
    correction_receipt = await pool._shadow_backend_for_scope(None).reconcile_durable_exchange(
        f"{correction_turn}:user", f"{correction_turn}:assistant"
    )
    assert correction_receipt.assistant_message_id == corrected.metadata["nutrition_append_event_id"]
    assert attempt["camera_correction_commit"]["kind"] == "portion"
    assert attempt["camera_commit"]["event_id"] == original_event
    turns_after_correction = len(engine.turns)
    first_correction_rows = [
        (row.id, row.content, dict(row.metadata), row.created_at)
        for row in honcho.messages[len(original_rows):]
    ]
    assert [(row.id, row.content, dict(row.metadata), row.created_at)
            for row in honcho.messages[:len(original_rows)]] == original_rows
    restarted, _, _, _ = _ingress(tmp_path, FakeTelegram())
    pool._camera_ingress = restarted
    half = await _native_callback(
        bus, label="Половину порции", target=restarted._attempts[request["candidate_id"]]["reply_ids"][0],
        options=["1 кусочек", "2 кусочка", "Половину порции"],
        prompt="Вы съели это? Сколько примерно вы съели?",
    )
    half.metadata["callback_query_id"] = "correction-half-new-operation"
    restarted.process_real_inbound(half)
    replayed_operation = half.metadata["_camera_turn_id"]
    half_final = await runtime_turn(pool, half, restarted)
    half_turn = half.metadata["_camera_turn_id"]
    assert half_turn != correction_turn
    assert half_turn != replayed_operation
    half_receipt = await pool._shadow_backend_for_scope(None).reconcile_durable_exchange(
        f"{half_turn}:user", f"{half_turn}:assistant"
    )
    assert half_receipt.assistant_message_id == half_final.metadata.get("nutrition_append_event_id"), half_final
    assert half_receipt.assistant_message_id != correction_receipt.assistant_message_id
    assert restarted._attempts[request["candidate_id"]]["camera_correction_commit"]["kind"] == "portion"
    assert restarted._attempts[request["candidate_id"]]["camera_correction_commit"]["target_event_id"] == original_event
    assert len(honcho.messages) == len(original_rows) + 4
    assert len(engine.turns) == turns_after_correction + 1
    assert [(row.id, row.content, dict(row.metadata), row.created_at)
            for row in honcho.messages[:len(original_rows) + 2]] == original_rows + first_correction_rows
    assert honcho.messages[-1].metadata["decision_trace"]["annotations"]["nutrition"]["items"][0]["quantity_text"] == "Половину порции"

    quantity_fold = _fold_events([
        {"event_id": row.id,
         "root_source_message_id": original_rows[3][2]["source_message_id"],
         "_created_at": row.created_at,
         "annotation": row.metadata["decision_trace"]["annotations"]["nutrition"]}
        for row in honcho.messages
        if row.metadata.get("role") == "assistant"
        and isinstance(row.metadata.get("decision_trace"), dict)
        and isinstance(row.metadata["decision_trace"].get("annotations"), dict)
        and isinstance(row.metadata["decision_trace"]["annotations"].get("nutrition"), dict)
    ])
    assert quantity_fold["consumed"] is True
    assert quantity_fold["energy_kcal_best"] == 27
    assert quantity_fold["meal_date"] == original_rows[3][2]["decision_trace"]["annotations"]["nutrition"]["meal_at"][:10]
    assert quantity_fold["latest_event_id"] == half_receipt.assistant_message_id

    current_restarted, _, _, _ = _ingress(tmp_path, FakeTelegram())
    pool._camera_ingress = current_restarted
    retry = await _native_callback(
        bus, label="Половину порции", target=current_restarted._attempts[request["candidate_id"]]["reply_ids"][0],
        options=["1 кусочек", "2 кусочка", "Половину порции"],
        prompt="Вы съели это? Сколько примерно вы съели?",
    )
    retry.metadata["callback_query_id"] = "same-half-fresh-callback"
    current_restarted.process_real_inbound(retry)
    assert retry.metadata["_camera_correction_replay"] is not None
    replay_final = await runtime_turn(pool, retry, current_restarted)
    assert replay_final.text == "Эта порция уже записана."
    assert replay_final.metadata["nutrition_append_event_id"] == half_receipt.assistant_message_id
    assert len(honcho.messages) == len(original_rows) + 4
    assert len(engine.turns) == turns_after_correction + 1

    denial = await _native_callback(
        bus, label="Нет, не ела", target=current_restarted._attempts[request["candidate_id"]]["photo_id"],
        options=["Да, я это съела", "Нет, не ела"], prompt="Съели ли вы это?",
    )
    denial.metadata["callback_query_id"] = "denial-after-portion-correction"
    current_restarted.process_real_inbound(denial)
    assert denial.metadata["_camera_correction"] is not None
    assert denial.metadata["_camera_turn_id"] != correction_turn
    await runtime_turn(pool, denial, current_restarted)
    denial_commit = current_restarted._attempts[request["candidate_id"]]["camera_correction_commit"]
    assert denial_commit["kind"] == "denial"
    assert denial_commit["event_id"] != correction_receipt.assistant_message_id
    assert current_restarted._attempts[request["candidate_id"]]["camera_commit"]["event_id"] == original_event
    assert len(honcho.messages) == len(original_rows) + 6
    denied_restarted, _, _, _ = _ingress(tmp_path, FakeTelegram())
    pool._camera_ingress = denied_restarted
    denial_retry = await _native_callback(
        bus, label="Нет, не ела",
        target=denied_restarted._attempts[request["candidate_id"]]["photo_id"],
        options=["Да, я это съела", "Нет, не ела"], prompt="Съели ли вы это?",
    )
    denial_retry.metadata["callback_query_id"] = "denial-after-portion-correction"
    denied_restarted.process_real_inbound(denial_retry)
    assert denial_retry.metadata["_camera_correction_replay"] is not None
    denial_retry_final = await runtime_turn(pool, denial_retry, denied_restarted)
    assert denial_retry_final.metadata["nutrition_append_event_id"] == denial_commit["event_id"]
    assert len(honcho.messages) == len(original_rows) + 6
    assistant_rows = [row for row in honcho.messages if row.metadata.get("role") == "assistant"
                      and isinstance(row.metadata.get("decision_trace"), dict)
                      and isinstance(row.metadata["decision_trace"].get("annotations"), dict)
                      and isinstance(row.metadata["decision_trace"]["annotations"].get("nutrition"), dict)]
    assert [row.metadata["decision_trace"]["annotations"]["nutrition"]["record_type"]
            for row in assistant_rows].count("meal_observation") == 1
    assert [row.metadata["decision_trace"]["annotations"]["nutrition"]["record_type"]
            for row in assistant_rows].count("meal_correction") == 3
    folded = _fold_events([
        {"event_id": row.id,
         "root_source_message_id": original_rows[3][2]["source_message_id"],
         "_created_at": row.created_at,
         "annotation": row.metadata["decision_trace"]["annotations"]["nutrition"]}
        for row in assistant_rows
    ])
    assert folded["consumed"] is False
    assert folded["energy_kcal_best"] == 0
    original_meal_at = original_rows[3][2]["decision_trace"]["annotations"]["nutrition"]["meal_at"]
    assert folded["meal_date"] == original_meal_at[:10]
    await ingress.close()
    await restarted.close()
    await current_restarted.close()
    await denied_restarted.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "foreign", "stale"])
async def test_quantity_correction_repeat_requires_latest_exact_receipt(
    tmp_path, monkeypatch, failure
):
    from openharness.engine.stream_events import AssistantTextDelta
    from openharness.evals import TRACE_FINALIZATION

    tmp_path.mkdir(parents=True, exist_ok=True)
    ingress, bus, request, pool, honho, saved_portion, _ = await _save_callback_portion(
        tmp_path, monkeypatch
    )
    attempt = ingress._attempts[request["candidate_id"]]
    engine = pool._test_bundle.engine

    async def correction_turn(user_message):
        engine.messages.append(user_message)
        engine.turns.append((user_message.text, [], saved_portion.timestamp))
        engine.decision_trace_recorder.record(TRACE_FINALIZATION, {
            "schema_version": 1, "trace_event_id": "correction-for-replay-control",
            "annotations": {"nutrition": {
                "schema_version": 2, "record_type": "meal_correction",
                "changed_fields": ["items", "energy_kcal_best"],
                "items": [{"name": "Мягкий творог Синтетик 5%, упаковка 125 г",
                           "quantity_text": "1 piece", "energy_kcal_best": 53}],
                "energy_kcal_best": 53,
            }},
        })
        yield AssistantTextDelta(text="Исправление сохранено.")

    engine.submit_message = correction_turn
    changed = await _native_callback(
        bus, label="1 кусочек", target=attempt["reply_ids"][0],
        options=["1 кусочек", "2 кусочка", "Половину порции"], prompt="Сколько съели?",
    )
    changed.metadata["callback_query_id"] = "control-correction"
    ingress.process_real_inbound(changed)
    await runtime_turn(pool, changed, ingress)
    turns_before_replay = len(engine.turns)
    backend = pool._shadow_backend_for_scope(None)
    reconcile = backend.reconcile_durable_exchange
    correction_op = attempt["camera_correction_commit"]["client_op_id"]

    async def altered(user_op, assistant_op):
        receipt = await reconcile(user_op, assistant_op)
        if assistant_op != correction_op:
            return receipt
        if failure == "missing":
            return None
        if failure == "foreign":
            return replace(receipt, assistant_metadata={
                **receipt.assistant_metadata, "tenant_id": "another-owner"
            })
        return replace(receipt, assistant_client_op_id="stale:assistant")

    pool._shadow_backend_for_scope = lambda _scope: SimpleNamespace(reconcile_durable_exchange=altered)
    replay = await _native_callback(
        bus, label="1 кусочек", target=attempt["reply_ids"][0],
        options=["1 кусочек", "2 кусочка", "Половину порции"], prompt="Сколько съели?",
    )
    replay.metadata["callback_query_id"] = "control-replay"
    ingress.process_real_inbound(replay)
    updates = [update async for update in pool.stream_message(replay, ingress.config.session_key)]
    assert any(update.kind == "error" for update in updates)
    assert len(honho.messages) == 6
    assert len(engine.turns) == turns_before_replay
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [
    "fresh_quantity", "crash_first", "crash_subsequent", "delivery_unknown",
    "foreign_session", "foreign_native_target", "wrong_answer_bound",
    "masked_date_clear", "typed_current", "typed_obsolete", "typed_denial_repeat",
    "typed_pending_first", "typed_pending_subsequent", "typed_pending_missing",
    "context_denial_after_native", "context_denial_after_context",
])
async def test_later_correction_review_boundaries(tmp_path, monkeypatch, case):
    from openharness.engine.stream_events import AssistantTextDelta
    from openharness.evals import TRACE_FINALIZATION

    tmp_path.mkdir(parents=True, exist_ok=True)
    ingress, bus, request, pool, honcho, saved_portion, saved_final = (
        await _save_callback_portion(tmp_path, monkeypatch)
    )
    candidate_id = request["candidate_id"]
    engine = pool._test_bundle.engine

    async def correction_model(user_message):
        engine.messages.append(user_message)
        engine.turns.append((user_message.text, [], saved_portion.timestamp))
        active = pool._active_message
        if case == "typed_obsolete" and active.metadata.get("native_keyboard_selected_label") is None:
            yield AssistantTextDelta(text="Review ordinary model path reached.")
            return
        denied = active.metadata.get("_camera_portion_correction") is None
        selected = active.metadata.get("native_keyboard_selected_label", "1 кусочек")
        calories = 27 if selected == "Половину порции" else 53
        annotation = (
            {"schema_version": 2, "record_type": "meal_correction",
             "changed_fields": ["consumption_status", "energy_kcal_best"],
             "consumption_status": "not_consumed", "energy_kcal_best": 0}
            if denied else
            {"schema_version": 2, "record_type": "meal_correction",
             "changed_fields": ["items", "energy_kcal_best"],
             "items": [{"name": "Synthetic food", "quantity_text": selected,
                        "energy_kcal_best": calories}], "energy_kcal_best": calories}
        )
        engine.decision_trace_recorder.record(TRACE_FINALIZATION, {
            "schema_version": 1, "trace_event_id": f"review-correction-{len(engine.turns)}",
            "annotations": {"nutrition": annotation},
        })
        yield AssistantTextDelta(text="Исправление сохранено.")

    engine.submit_message = correction_model

    async def native(active, label, callback_id):
        target = active._attempts[candidate_id]["reply_ids"][0]
        msg = await _native_callback(
            bus, label=label, target=target,
            options=["1 кусочек", "2 кусочка", "Половину порции", "Нет, не ела"],
            prompt="Вы съели это? Сколько примерно вы съели?",
        )
        msg.metadata["callback_query_id"] = callback_id
        active.process_real_inbound(msg)
        return msg

    async def contextual(active, label, source_id, *, reply=False):
        metadata = {"message_id": source_id, "_telegram_raw_text": label,
                    "is_group": False, "chat_type": "private"}
        if reply:
            metadata["reply_to_message_id"] = str(active._attempts[candidate_id]["reply_ids"][0])
        msg = InboundMessage(channel="telegram", sender_id="123", chat_id="123",
                             content=label, metadata=metadata)
        active.process_real_inbound(msg)
        return msg

    async def write_initial_correction(*, crash=False, unknown_delivery=False):
        msg = await native(ingress, "1 кусочек", "review-first-quantity")
        if crash:
            original_record = ingress.record_committed_correction

            def interrupt_after_durable_append(*_args, **_kwargs):
                raise RuntimeError("fixture interruption after durable correction append")

            ingress.record_committed_correction = interrupt_after_durable_append
            try:
                await runtime_turn(pool, msg, ingress)
            except RuntimeError as exc:
                assert "after durable correction append" in str(exc)
            else:
                raise AssertionError("crash boundary did not interrupt journal commit")
            finally:
                ingress.record_committed_correction = original_record
            assert len(honcho.messages) == 6
            assert ingress._attempts[candidate_id]["camera_correction"] == "answering"
            assert ingress._attempts[candidate_id].get("camera_correction_commit") is None
            op_turn = msg.metadata["_camera_turn_id"]
            durable = await pool._shadow_backend_for_scope(None).reconcile_durable_exchange(
                f"{op_turn}:user", f"{op_turn}:assistant"
            )
            assert durable is not None
            return msg, durable

        if unknown_delivery:
            original_note = ingress.note_assistant_receipt
            ingress.note_assistant_receipt = lambda outbound, _receipt: ingress.note_assistant_failure(outbound)
            try:
                await runtime_turn(pool, msg, ingress)
            finally:
                ingress.note_assistant_receipt = original_note
            assert ingress._attempts[candidate_id]["camera_correction"] == "delivery_unknown"
        else:
            await runtime_turn(pool, msg, ingress)
        turn = msg.metadata["_camera_turn_id"]
        durable = await pool._shadow_backend_for_scope(None).reconcile_durable_exchange(
            f"{turn}:user", f"{turn}:assistant"
        )
        return msg, durable

    if case.startswith("context_denial"):
        denial = (await native(ingress, "Нет, не ела", "review-context-denial-native")
                  if case == "context_denial_after_native"
                  else await contextual(ingress, "Нет, не ела", "review-context-denial-first"))
        await runtime_turn(pool, denial, ingress)
        saved_event = ingress._attempts[candidate_id]["camera_correction_commit"]["event_id"]
        assert ingress._attempts[candidate_id]["camera_correction_commit"]["kind"] == "denial"
        restarted, _, _, _ = _ingress(tmp_path, FakeTelegram())
        pool._camera_ingress = restarted
        repeat = await contextual(restarted, "Нет, не ела", "review-context-denial-repeat")
        assert repeat.metadata["_camera_route"] == "context"
        assert repeat.metadata.get("reply_to_message_id") is None
        before_rows, before_turns = len(honcho.messages), len(engine.turns)
        repeated = await runtime_turn(pool, repeat, restarted)
        assert repeated.metadata["nutrition_append_event_id"] == saved_event
        assert len(honcho.messages) == before_rows and len(engine.turns) == before_turns
        await ingress.close()
        await restarted.close()
        return

    first_crash = case in {"crash_first", "typed_pending_first", "typed_pending_missing"}
    first, first_receipt = await write_initial_correction(
        crash=first_crash, unknown_delivery=case == "delivery_unknown"
    )
    original_rows = [(r.id, r.content, dict(r.metadata), r.created_at) for r in honcho.messages[:4]]

    if case in {"crash_subsequent", "typed_pending_subsequent"}:
        assert not first_crash
        first_event = first_receipt.assistant_message_id
        active, _, _, _ = _ingress(tmp_path, FakeTelegram())
        pool._camera_ingress = active
        previous_rows = [(r.id, r.content, dict(r.metadata), r.created_at) for r in honcho.messages]
        half = await native(active, "Половину порции", "review-second-quantity-crash")
        old_record = active.record_committed_correction

        def interrupt_second_append(*_args, **_kwargs):
            raise RuntimeError("fixture interruption after second durable correction append")

        active.record_committed_correction = interrupt_second_append
        try:
            await runtime_turn(pool, half, active)
        except RuntimeError as exc:
            assert "second durable correction append" in str(exc)
        else:
            raise AssertionError("second crash boundary did not interrupt journal commit")
        finally:
            active.record_committed_correction = old_record
        second_turn = half.metadata["_camera_turn_id"]
        second_receipt = await pool._shadow_backend_for_scope(None).reconcile_durable_exchange(
            f"{second_turn}:user", f"{second_turn}:assistant"
        )
        assert len(honcho.messages) == 8
        assert active._attempts[candidate_id]["camera_correction"] == "answering"
        assert active._attempts[candidate_id]["camera_correction_commit"]["event_id"] == first_event
        recovered, _, _, _ = _ingress(tmp_path, FakeTelegram())
        pool._camera_ingress = recovered
        if case == "typed_pending_subsequent":
            retry = await contextual(recovered, "Половину порции", "review-second-typed-crash", reply=True)
            result = await runtime_turn(pool, retry, recovered)
            assert result.metadata["nutrition_append_event_id"] == second_receipt.assistant_message_id
        else:
            retry = await native(recovered, "Половину порции", "review-second-quantity-crash")
            result = await runtime_turn(pool, retry, recovered)
            assert result.metadata["nutrition_append_event_id"] == second_receipt.assistant_message_id
        assert len(honcho.messages) == 8
        assert recovered._attempts[candidate_id]["camera_correction_commit"]["event_id"] == second_receipt.assistant_message_id
        assert [(r.id, r.content, dict(r.metadata), r.created_at) for r in honcho.messages[:len(previous_rows)]] == previous_rows
        await ingress.close()
        await active.close()
        await recovered.close()
        return

    if first_crash or case == "delivery_unknown":
        active, _, _, _ = _ingress(tmp_path, FakeTelegram())
        pool._camera_ingress = active
        before_rows, before_turns = len(honcho.messages), len(engine.turns)
        if case in {"typed_pending_first", "typed_pending_missing"}:
            replay = await contextual(active, "1 кусочек", "review-first-typed-crash", reply=True)
            if case == "typed_pending_missing":
                backend = pool._shadow_backend_for_scope(None)
                reconcile = backend.reconcile_durable_exchange

                async def missing(user_op, assistant_op):
                    turn = replay.metadata["_camera_turn_id"]
                    return None if assistant_op == f"{turn}:assistant" else await reconcile(user_op, assistant_op)

                pool._shadow_backend_for_scope = lambda _scope: SimpleNamespace(reconcile_durable_exchange=missing)
            if case == "typed_pending_missing":
                pool._active_message = replay
                updates = [item async for item in pool.stream_message(replay, active.config.session_key)]
                assert any(item.kind == "error" for item in updates)
                assert all(item.metadata.get("nutrition_append_event_id") is None for item in updates)
                assert active._attempts[candidate_id].get("camera_correction_commit") is None
                assert active._attempts[candidate_id]["camera_correction"] == "answering"
                assert len(honcho.messages) == before_rows and len(engine.turns) == before_turns
            else:
                result = await runtime_turn(pool, replay, active)
                assert result.metadata["nutrition_append_event_id"] == first_receipt.assistant_message_id
                assert active._attempts[candidate_id]["camera_correction_commit"]["event_id"] == first_receipt.assistant_message_id
        else:
            replay = await native(active, "1 кусочек", "review-first-quantity")
            result = await runtime_turn(pool, replay, active)
            assert result.metadata["nutrition_append_event_id"] == first_receipt.assistant_message_id
        assert len(honcho.messages) == before_rows
        assert len(engine.turns) == before_turns
        expected_states = {"answering"} if case == "typed_pending_missing" else {"completed", "final_queued"}
        assert active._attempts[candidate_id]["camera_correction"] in expected_states
        await ingress.close()
        await active.close()
        return

    if case == "fresh_quantity":
        active, _, _, _ = _ingress(tmp_path, FakeTelegram())
        pool._camera_ingress = active
        before_rows, before_turns = len(honcho.messages), len(engine.turns)
        half = await native(active, "Половину порции", "review-new-half-quantity")
        result = await runtime_turn(pool, half, active)
        latest = active._attempts[candidate_id]["camera_correction_commit"]
        assert result.metadata["nutrition_append_event_id"] == latest["event_id"]
        assert latest["event_id"] != first_receipt.assistant_message_id
        assert latest["target_event_id"] == saved_final.metadata["nutrition_append_event_id"]
        assert len(honcho.messages) == before_rows + 2
        assert len(engine.turns) == before_turns + 1
        assert [(r.id, r.content, dict(r.metadata), r.created_at) for r in honcho.messages[:6]][:4] == original_rows
        assert honcho.messages[-1].metadata["decision_trace"]["annotations"]["nutrition"]["items"][0]["quantity_text"] == "Половину порции"
        await ingress.close()
        await active.close()
        return

    if case == "delivery_unknown":
        raise AssertionError("handled above")

    active, _, _, _ = _ingress(tmp_path, FakeTelegram())
    pool._camera_ingress = active
    if case == "typed_denial_repeat":
        denial = await native(active, "Нет, не ела", "review-denial")
        await runtime_turn(pool, denial, active)
        saved_event = active._attempts[candidate_id]["camera_correction_commit"]["event_id"]
        active, _, _, _ = _ingress(tmp_path, FakeTelegram())
        pool._camera_ingress = active
        label = "Нет, не ела"
    elif case.startswith("foreign_") or case in {"wrong_answer_bound", "masked_date_clear"}:
        latest = active._attempts[candidate_id]["camera_correction_commit"]
        backend = pool._shadow_backend_for_scope(None)
        reconcile = backend.reconcile_durable_exchange

        async def altered(user_op, assistant_op):
            receipt = await reconcile(user_op, assistant_op)
            if assistant_op != latest["client_op_id"]:
                return receipt
            md = json.loads(json.dumps(receipt.assistant_metadata, ensure_ascii=False))
            if case == "masked_date_clear":
                nutrition = md["decision_trace"]["annotations"]["nutrition"]
                nutrition["changed_fields"] += ["meal_at", "meal_date"]
                nutrition["meal_at"] = None
                nutrition["meal_date"] = None
            else:
                key = {
                    "foreign_session": "gateway_session_id",
                    "foreign_native_target": "camera_reply_to_native_message_id",
                    "wrong_answer_bound": "camera_answer_bound",
                }[case]
                md[key] = "no" if case == "wrong_answer_bound" else "foreign-value"
            return replace(receipt, assistant_metadata=md)

        pool._shadow_backend_for_scope = lambda _scope: SimpleNamespace(reconcile_durable_exchange=altered)
        label = "1 кусочек"
        active, _, _, _ = _ingress(tmp_path, FakeTelegram())
        pool._camera_ingress = active
    else:
        label = "2 кусочка" if case == "typed_obsolete" else "1 кусочек"

    before_rows, before_turns = len(honcho.messages), len(engine.turns)
    if case.startswith("typed_"):
        replay = InboundMessage(
            channel="telegram", sender_id="123", chat_id="123", content=label,
            metadata={"message_id": f"review-typed-{case}",
                      "reply_to_message_id": str(active._attempts[candidate_id]["reply_ids"][0]),
                      "_telegram_raw_text": label, "is_group": False, "chat_type": "private"},
        )
        active.process_real_inbound(replay)
    else:
        replay = await native(active, label, f"review-replay-{case}")
    pool._active_message = replay
    updates = [item async for item in pool.stream_message(replay, active.config.session_key)]
    if case in {"foreign_session", "foreign_native_target", "wrong_answer_bound", "masked_date_clear"}:
        assert any(item.kind == "error" for item in updates)
        assert len(honcho.messages) == before_rows
        assert len(engine.turns) == before_turns
    elif case == "typed_obsolete":
        assert any(item.kind == "final" and item.metadata.get("nutrition_append_event_id") is None for item in updates)
        assert len(honcho.messages) == before_rows
        assert len(engine.turns) == before_turns + 1
    else:
        expected_event = saved_event if case == "typed_denial_repeat" else first_receipt.assistant_message_id
        assert any(item.kind == "final" and item.metadata.get("nutrition_append_event_id") == expected_event for item in updates)
        assert len(honcho.messages) == before_rows
        assert len(engine.turns) == before_turns
    assert [(r.id, r.content, dict(r.metadata), r.created_at) for r in honcho.messages[:4]] == original_rows
    await ingress.close()
    await active.close()

"""Full runtime eval capture for receipt-verified Camera saved replies."""

from datetime import date, datetime, timedelta
from types import SimpleNamespace
import sqlite3
from dataclasses import replace

import pytest

from openharness.channels.bus.events import (
    InboundMessage, OutboundDeliveryReceipt, OutboundMessage,
)
from openharness.engine.stream_events import AssistantTextDelta
from openharness.evals import TRACE_FINALIZATION
from ohmo.evals.nutrition_persistence import (
    Goal, Manifest, derive_meal_id, export_eval_dialogue, validate_dialogue_binding,
)
from tests.test_ohmo.test_camera_completed_replay import (
    _native_callback, _save_callback_portion, _save_typed_confirmation,
)


@pytest.mark.asyncio
async def test_native_saved_camera_replay_is_exported_as_its_own_actual_turn(tmp_path):
    """A real runtime replay stays visible without appending another meal."""
    import tests.test_ohmo.test_camera_f84_joint_runtime as joint
    import tests.test_ohmo.test_camera_ingress as ingress_tests
    from openharness.channels.bus.events import OutboundMessage

    ingress, root, bus, telegram = ingress_tests._ingress(tmp_path)
    request = ingress_tests._candidate(
        root, capture_time=joint.BASE, classifier_decision="food",
    )
    initial, _, _, options, clicked, channel, _ = await ingress_tests._actual_native_camera_prompt(
        ingress, root, bus, request,
        question="Что из этого вы съели?",
        options=["2 яйца и рис", "Не ела", "Только оценить состав"], selected_index=0,
        source_analysis="На фото рис и яйца.",
    )
    assert clicked is not None

    pool, bundle, server, client = joint.setup(str(tmp_path / "runtime"))
    ingress_tests._configure_joint_camera_runtime(pool, bundle, ingress)
    bundle.engine.annotation = None
    bundle.engine.answer = (
        "На фото рис и яйца. [[ask: Что из этого вы съели? | 2 яйца и рис | Не ела]]"
    )
    initial_updates = [
        update async for update in pool.stream_message(initial, "telegram:123")
    ]
    initial_final = next(update for update in initial_updates if update.kind == "final")
    assert "[[ask:" in initial_final.text

    runtime_message, _, _ = joint.inbound(
        pool, clicked.metadata["message_id"], clicked.content,
        when=joint.BASE + timedelta(minutes=101), media=clicked.media,
        metadata_extra=clicked.metadata,
    )
    bundle.engine.annotation = joint.observation(
        meal_at=joint.BASE, energy_kcal_best=500,
        items=[{"name": "egg and rice", "quantity_text": "2 яйца и рис", "energy_kcal_best": 500}],
    )
    bundle.engine.answer = "Записала порцию."
    original_updates = [
        update async for update in pool.stream_message(runtime_message, "telegram:123")
    ]
    original_final = next(update for update in original_updates if update.kind == "final")
    operation_id = original_final.metadata["nutrition_append_event_id"]
    attempt = ingress._attempts[request["candidate_id"]]
    sent = OutboundMessage(
        channel="telegram", chat_id="123", content=original_final.text,
        metadata=original_final.metadata,
    )
    delivery = await channel.send(sent)
    ingress.note_assistant_receipt(sent, delivery)

    before_rows = len(server.rows)
    before_engine_messages = len(bundle.engine.messages)
    replay = await ingress_tests._native_callback(
        bus, label="2 яйца и рис", target=clicked.metadata["message_id"],
        options=options, prompt=clicked.metadata["native_keyboard_prompt"],
    )
    ingress.process_real_inbound(replay)
    runtime_replay, _, _ = joint.inbound(
        pool, replay.metadata["message_id"], replay.content,
        when=joint.BASE + timedelta(minutes=102), metadata_extra=replay.metadata,
    )
    replay_updates = [
        update async for update in pool.stream_message(runtime_replay, "telegram:123")
    ]
    replay_final = next(update for update in replay_updates if update.kind == "final")
    assert replay_final.text == "Эта порция уже записана."
    assert replay_final.metadata["nutrition_append_event_id"] == operation_id
    assert len(server.rows) == before_rows
    assert len(bundle.engine.messages) == before_engine_messages

    eval_root = pool._workspace / "evals"
    with sqlite3.connect(eval_root / "evals.sqlite") as connection:
        episode_rows = connection.execute(
            "SELECT episode_id, created_at FROM episodes WHERE session_id = ? ORDER BY created_at, episode_id",
            ("gateway-session",),
        ).fetchall()
    episode_ids = [row[0] for row in episode_rows]
    exported = export_eval_dialogue(eval_root, episode_ids=episode_ids)
    matching = [episode for episode in exported["episodes"]
                if any(turn.get("role") == "assistant" and turn.get("text") == replay_final.text
                       for turn in episode.get("dialogue", []))]
    assert len(matching) == 1
    episode = matching[0]
    assert episode["episode"]["user_text"] == replay.content
    context = episode["episode"]["metadata"]["trusted_camera_context"]
    assert context["source_message_id"] == str(replay.metadata["message_id"])
    assert context["operation_id"].endswith(":assistant")
    assert context["candidate_id"] == request["candidate_id"]
    assert episode["episode"]["metadata"]["inbound"]["metadata"]["callback_query_id"] == (
        replay.metadata["callback_query_id"]
    )
    assert attempt["camera_commit"]["event_id"] == operation_id
    original_row = next(row for row in server.rows if row["id"] == operation_id)
    metadata = original_row["metadata"]
    started = datetime.fromisoformat(episode_rows[0][1].replace("Z", "+00:00"))
    as_of = datetime.fromisoformat(episode_rows[-1][1].replace("Z", "+00:00"))
    goal = Goal(
        case_id="saved-camera-replay", episode_ids=episode_ids,
        owner_id=metadata["tenant_id"], principal_id=metadata["source_principal"],
        workspace_id="review-workspace", eval_workspace=str(pool._workspace),
        peer_id="ohmo", canonical_owner_id=metadata["tenant_id"], canonical_login="owner",
        session_id="review-session", gateway_session_id="gateway-session",
        source_message_id=metadata["source_message_id"],
        meal_date=date.fromisoformat(request["capture_time"][:10]),
        trajectory_started_at=started, trajectory_as_of=as_of,
        logical_turn_id=metadata["logical_turn_id"],
        trace_episode_id=metadata["decision_trace_episode_id"],
        operation_id=metadata["client_op_id"],
        canonical_meal_id=derive_meal_id(
            tenant_id=metadata["tenant_id"], source_principal=metadata["source_principal"],
            gateway_session_id="gateway-session", source_message_id=metadata["source_message_id"],
        ), expected_consumed=True, expected_kcal=500,
        expectation_origin="reviewed_user_dialogue", expectation_source="synthetic-runtime-test",
    )
    binding = validate_dialogue_binding(Manifest(schema_version=1, goals=[goal]), exported)
    assert binding[goal.case_id]["complete"] is True, binding
    without_initial = Goal.model_validate({
        **goal.model_dump(), "episode_ids": episode_ids[1:],
    })
    incomplete = validate_dialogue_binding(
        Manifest(schema_version=1, goals=[without_initial]),
        export_eval_dialogue(eval_root, episode_ids=episode_ids[1:]),
    )
    assert incomplete[goal.case_id]["reason"] == (
        "include the initial Camera context episode matching this photo receipt"
    ), incomplete
    await client.aclose()
    await telegram.aclose() if hasattr(telegram, "aclose") else None
    await ingress.close()


def _exported_episodes(pool):
    eval_root = pool._workspace / "evals"
    with sqlite3.connect(eval_root / "evals.sqlite") as connection:
        ids = [row[0] for row in connection.execute(
            "SELECT episode_id FROM episodes WHERE session_id = ? ORDER BY created_at, episode_id",
            (pool._test_bundle.session_id,),
        )]
    return export_eval_dialogue(eval_root, episode_ids=ids)["episodes"]


def _find_episode(episodes, *, user_text, assistant_text):
    matches = [item for item in episodes
               if item["episode"]["user_text"] == user_text
               and any(turn.get("role") == "assistant" and turn.get("text") == assistant_text
                       for turn in item["dialogue"])]
    assert len(matches) == 1, matches
    return matches[0]


async def _stream(pool, message, ingress):
    pool._active_message = message
    return [
        update async for update in pool.stream_message(message, ingress.config.session_key)
    ]


def _note_delivery(ingress, update, native_id):
    message = OutboundMessage(
        channel="telegram", chat_id="123", content=update.text, metadata=update.metadata,
    )
    ingress.note_assistant_receipt(
        message,
        OutboundDeliveryReceipt(
            channel="telegram", chat_id="123", native_message_ids=(native_id,)
        ),
    )


def _install_correction_model(pool, saved_portion):
    engine = pool._test_bundle.engine

    async def correction_model(user_message):
        engine.messages.append(user_message)
        engine.turns.append((user_message.text, [], saved_portion.timestamp))
        selected = pool._active_message.metadata.get("native_keyboard_selected_label") or user_message.text
        engine.decision_trace_recorder.record(TRACE_FINALIZATION, {
            "schema_version": 1,
            "trace_event_id": f"eval-correction-{len(engine.turns)}",
            "annotations": {"nutrition": {
                "schema_version": 2,
                "record_type": "meal_correction",
                "changed_fields": ["items", "energy_kcal_best"],
                "items": [{"name": "synthetic food", "quantity_text": selected,
                           "energy_kcal_best": 53}],
                "energy_kcal_best": 53,
            }},
        })
        yield AssistantTextDelta(text="Исправление сохранено.")

    engine.submit_message = correction_model


@pytest.mark.asyncio
async def test_typed_saved_replay_is_captured_without_model_or_meal_append(tmp_path, monkeypatch):
    ingress, request, pool, honho, saved_final, first_answer = await _save_typed_confirmation(
        tmp_path, monkeypatch
    )
    attempt = ingress._attempts[request["candidate_id"]]
    before = _exported_episodes(pool)
    before_turns = len(pool._test_bundle.engine.turns)
    before_messages = len(honho.messages)
    replay = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content=first_answer.content,
        metadata={
            "message_id": "typed-camera-replay-eval-2",
            "reply_to_message_id": str(attempt["reply_ids"][0]),
            "_telegram_raw_text": first_answer.content, "is_group": False,
            "chat_type": "private",
        },
    )
    ingress.process_real_inbound(replay)
    assert replay.metadata.get("_camera_typed_replay_candidate") == request["candidate_id"]
    updates = [update async for update in pool.stream_message(replay, ingress.config.session_key)]
    final = next(update for update in updates if update.kind == "final")
    assert final.text == "Эта порция уже записана."
    assert final.metadata["nutrition_append_event_id"] == saved_final.metadata[
        "nutrition_append_event_id"
    ]
    assert len(pool._test_bundle.engine.turns) == before_turns
    assert len(honho.messages) == before_messages
    after = _exported_episodes(pool)
    assert len(after) == len(before) + 1
    item = _find_episode(after, user_text=replay.content, assistant_text=final.text)
    assert item["dialogue_complete"] is True
    episode_metadata = item["episode"]["metadata"]
    context = episode_metadata["trusted_camera_context"]
    assert context["source_message_id"] == replay.metadata["message_id"]
    assert context["candidate_id"] == request["candidate_id"]
    assert context["operation_id"] == f'{attempt["answer_turn_id"]}:assistant'
    assert (
        episode_metadata["inbound"]["metadata"]["_camera_candidate_id"]
        == request["candidate_id"]
    )
    await ingress.close()


@pytest.mark.asyncio
async def test_changed_typed_choice_does_not_leave_fast_replay_episode(tmp_path, monkeypatch):
    ingress, request, pool, honho, _, _ = await _save_typed_confirmation(tmp_path, monkeypatch)
    attempt = ingress._attempts[request["candidate_id"]]
    before = _exported_episodes(pool)
    turns_before = len(pool._test_bundle.engine.turns)
    changed = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content="Я съела другую порцию",
        metadata={
            "message_id": "typed-camera-changed-eval",
            "reply_to_message_id": str(attempt["photo_id"]),
            "_telegram_raw_text": "Я съела другую порцию", "is_group": False,
            "chat_type": "private",
        },
    )
    ingress.process_real_inbound(changed)
    assert changed.metadata.get("_camera_typed_replay_candidate") == request["candidate_id"]
    updates = await _stream(pool, changed, ingress)
    final = next(update for update in updates if update.kind == "final")
    assert final.text != "Эта порция уже записана."
    assert len(pool._test_bundle.engine.turns) == turns_before + 1
    assert len(honho.messages) == 2
    after = _exported_episodes(pool)
    assert len(after) == len(before) + 1
    episode = _find_episode(after, user_text=changed.content, assistant_text=final.text)
    assert episode["dialogue_complete"] is True
    assert "trusted_camera_context" not in episode["episode"]["metadata"]
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("transport", "failure"), [("native", "missing"), ("typed", "foreign")])
async def test_replay_receipt_error_is_captured_without_saved_claim(
    tmp_path, monkeypatch, transport, failure,
):
    if transport == "native":
        ingress, bus, request, pool, honho, saved_portion, _ = await _save_callback_portion(
            tmp_path, monkeypatch
        )
    else:
        ingress, request, pool, honho, _, _ = await _save_typed_confirmation(
            tmp_path, monkeypatch
        )
        saved_portion = None
    attempt = ingress._attempts[request["candidate_id"]]
    backend = pool._shadow_backend_for_scope(None)
    reconcile = backend.reconcile_durable_exchange

    async def altered_receipt(user_op, assistant_op):
        receipt = await reconcile(user_op, assistant_op)
        target = attempt["camera_commit"]["client_op_id"]
        if assistant_op != target:
            return receipt
        if failure == "missing":
            return None
        return replace(receipt, assistant_metadata={
            **receipt.assistant_metadata, "tenant_id": "foreign-owner",
        })

    pool._shadow_backend_for_scope = lambda _scope: SimpleNamespace(
        reconcile_durable_exchange=altered_receipt
    )
    before = _exported_episodes(pool)
    before_turns, before_messages = len(pool._test_bundle.engine.turns), len(honho.messages)
    if transport == "native":
        replay = await _native_callback(
            bus, label="2 кусочка", target=attempt["reply_ids"][0],
            options=["1 кусочек", "2 кусочка", "Половину порции"], prompt="Сколько съели?",
        )
        replay.metadata["callback_query_id"] = "callback-capture-receipt-error"
        ingress.process_real_inbound(replay)
    else:
        replay = InboundMessage(
            channel="telegram", sender_id="123", chat_id="123", content="Да, я это съела",
            metadata={
                "message_id": "typed-camera-receipt-error",
                "reply_to_message_id": str(attempt["reply_ids"][0]),
                "_telegram_raw_text": "Да, я это съела", "is_group": False,
                "chat_type": "private",
            },
        )
        ingress.process_real_inbound(replay)
    updates = [update async for update in pool.stream_message(replay, ingress.config.session_key)]
    error = next(update for update in updates if update.kind == "error")
    assert "записана" not in error.text.casefold()
    assert len(pool._test_bundle.engine.turns) == before_turns
    assert len(honho.messages) == before_messages == (4 if transport == "native" else 2)
    after = _exported_episodes(pool)
    assert len(after) == len(before) + 1
    item = _find_episode(after, user_text=replay.content, assistant_text=error.text)
    assert item["dialogue_complete"] is True
    episode_metadata = item["episode"]["metadata"]
    assert episode_metadata["trusted_camera_context"]["source_message_id"] == str(
        replay.metadata["message_id"]
    )
    assert (
        episode_metadata["inbound"]["metadata"]["_camera_candidate_id"]
        == request["candidate_id"]
    )
    assert not any(turn.get("role") == "assistant" and "уже записана" in turn.get("text", "")
                   for turn in item["dialogue"])
    if saved_portion is not None:
        assert saved_portion.metadata["_camera_turn_id"] == replay.metadata["_camera_turn_id"]
    await ingress.close()


@pytest.mark.asyncio
async def test_capture_disabled_replay_writes_no_episode(tmp_path, monkeypatch):
    ingress, bus, request, pool, honho, _, _ = await _save_callback_portion(tmp_path, monkeypatch)
    attempt = ingress._attempts[request["candidate_id"]]
    before = _exported_episodes(pool)
    monkeypatch.setenv("OHMO_EVALS_CAPTURE", "off")
    replay = await _native_callback(
        bus, label="2 кусочка", target=attempt["reply_ids"][0],
        options=["1 кусочек", "2 кусочка", "Половину порции"], prompt="Сколько съели?",
    )
    replay.metadata["callback_query_id"] = "callback-capture-disabled"
    ingress.process_real_inbound(replay)
    with pytest.raises(ValueError, match="Camera turn requires nutrition finalization validation"):
        async for _ in pool.stream_message(replay, ingress.config.session_key):
            pass
    assert len(_exported_episodes(pool)) == len(before)
    assert len(honho.messages) == 4
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["native", "typed"])
async def test_latest_quantity_correction_replay_is_captured_per_transport(
    tmp_path, monkeypatch, transport,
):
    ingress, bus, request, pool, honho, saved_portion, _ = await _save_callback_portion(
        tmp_path, monkeypatch
    )
    attempt = ingress._attempts[request["candidate_id"]]
    _install_correction_model(pool, saved_portion)
    changed = await _native_callback(
        bus, label="1 кусочек", target=attempt["reply_ids"][0],
        options=["1 кусочек", "2 кусочка", "Половину порции"], prompt="Сколько съели?",
    )
    changed.metadata["callback_query_id"] = "quantity-correction-current"
    ingress.process_real_inbound(changed)
    changed_updates = await _stream(pool, changed, ingress)
    changed_final = next(update for update in changed_updates if update.kind == "final")
    correction_turn = changed.metadata["_camera_turn_id"]
    correction_event = attempt["camera_correction_commit"]["event_id"]
    assert changed_final.metadata["nutrition_append_event_id"] == correction_event
    _note_delivery(ingress, changed_final, 880)

    before = _exported_episodes(pool)
    turns_before, rows_before = len(pool._test_bundle.engine.turns), len(honho.messages)
    if transport == "native":
        replay = await _native_callback(
            bus, label="1 кусочек", target=attempt["reply_ids"][0],
            options=["1 кусочек", "2 кусочка", "Половину порции"], prompt="Сколько съели?",
        )
        replay.metadata["callback_query_id"] = "quantity-correction-replay-native"
        ingress.process_real_inbound(replay)
    else:
        replay = InboundMessage(
            channel="telegram", sender_id="123", chat_id="123", content="1 кусочек",
            metadata={
                "message_id": "quantity-correction-replay-typed",
                "reply_to_message_id": str(attempt["reply_ids"][0]),
                "_telegram_raw_text": "1 кусочек", "is_group": False,
                "chat_type": "private",
            },
        )
        ingress.process_real_inbound(replay)
    assert replay.metadata.get("_camera_correction_replay") is not None
    updates = await _stream(pool, replay, ingress)
    final = next(update for update in updates if update.kind == "final")
    assert final.text == "Эта порция уже записана."
    assert final.metadata["nutrition_append_event_id"] == correction_event
    assert len(honho.messages) == rows_before
    assert len(pool._test_bundle.engine.turns) == turns_before
    after = _exported_episodes(pool)
    assert len(after) == len(before) + 1
    item = _find_episode(after, user_text=replay.content, assistant_text=final.text)
    assert item["dialogue_complete"] is True
    context = item["episode"]["metadata"]["trusted_camera_context"]
    assert context["operation_id"] == f"{correction_turn}:assistant"
    assert context["candidate_id"] == request["candidate_id"]
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["native", "typed"])
async def test_changed_saved_selection_creates_no_orphan_fast_episode(
    tmp_path, monkeypatch, transport,
):
    if transport == "typed":
        ingress, request, pool, honho, _, _ = await _save_typed_confirmation(
            tmp_path, monkeypatch
        )
        attempt = ingress._attempts[request["candidate_id"]]
        changed = InboundMessage(
            channel="telegram", sender_id="123", chat_id="123",
            content="Я съела другую порцию",
            metadata={
                "message_id": "typed-camera-changed-eval-2",
                "reply_to_message_id": str(attempt["photo_id"]),
                "_telegram_raw_text": "Я съела другую порцию", "is_group": False,
                "chat_type": "private",
            },
        )
        ingress.process_real_inbound(changed)
    else:
        ingress, bus, request, pool, honho, saved_portion, _ = await _save_callback_portion(
            tmp_path, monkeypatch
        )
        attempt = ingress._attempts[request["candidate_id"]]
        changed = await _native_callback(
            bus, label="1 кусочек", target=attempt["reply_ids"][0],
            options=["1 кусочек", "2 кусочка", "Половину порции"], prompt="Сколько съели?",
        )
        changed.metadata["callback_query_id"] = "changed-native-choice"
        ingress.process_real_inbound(changed)
        _install_correction_model(pool, saved_portion)

    before = _exported_episodes(pool)
    turns_before = len(pool._test_bundle.engine.turns)
    updates = await _stream(pool, changed, ingress)
    final = next(update for update in updates if update.kind == "final")
    assert final.text != "Эта порция уже записана."
    assert len(pool._test_bundle.engine.turns) == turns_before + 1
    after = _exported_episodes(pool)
    assert len(after) == len(before) + 1
    episode = _find_episode(after, user_text=changed.content, assistant_text=final.text)
    assert episode["dialogue_complete"] is True
    assert len(honho.messages) == (6 if transport == "native" else 2)
    await ingress.close()

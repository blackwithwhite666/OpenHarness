"""Full runtime eval capture for receipt-verified Camera saved replies."""

from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
import sqlite3
from dataclasses import replace

import pytest

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _install_correction_model(pool, saved_portion, *, generated_quantity_text=None):
    engine = pool._test_bundle.engine

    async def correction_model(user_message):
        engine.messages.append(user_message)
        engine.turns.append((user_message.text, [], saved_portion.timestamp))
        selected = (pool._active_message.metadata.get("native_keyboard_selected_label")
                    or user_message.text)
        quantity_text = selected if generated_quantity_text is None else generated_quantity_text
        engine.decision_trace_recorder.record(TRACE_FINALIZATION, {
            "schema_version": 1,
            "trace_event_id": f"eval-correction-{len(engine.turns)}",
            "annotations": {"nutrition": {
                "schema_version": 2,
                "record_type": "meal_correction",
                "changed_fields": ["items", "energy_kcal_best"],
                "items": [{"name": "synthetic food", "quantity_text": quantity_text,
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


def _stamp(value):
    return value if isinstance(value, datetime) else datetime.fromisoformat(value.replace('Z', '+00:00'))


def _safe(value):
    if hasattr(value, 'model_dump'):
        return _plain_json_value(value.model_dump(mode='json'))
    if is_dataclass(value):
        return _plain_json_value(asdict(value))
    if isinstance(value, (datetime, Path)):
        return value.isoformat() if isinstance(value, datetime) else str(value)
    raise TypeError(f'Unserialized evidence type: {type(value).__name__}')


def _plain_json_value(value):
    """Materialize immutable runtime annotation mappings for the test snapshot."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(item) for item in value]
    if hasattr(value, 'model_dump'):
        return _plain_json_value(value.model_dump(mode='json'))
    if is_dataclass(value):
        return _plain_json_value(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f'Unsupported annotation value in test snapshot: {type(value).__name__}')


def _honcho_query_rows(rows, goal):
    """Apply the existing bounded peer/workspace/session read to actual messages."""
    return [row for row in rows if row['peer_id'] == goal.peer_id
            and row['workspace_id'] == goal.workspace_id
            and row['session_id'] == goal.session_id
            and goal.trajectory_started_at <= _stamp(row['created_at']) <= goal.trajectory_as_of]


def _diagnostic_update(update, camera_authority):
    """Copy a stream update while omitting only process-local Camera capabilities."""
    capability_fields = {
        '_camera_authority', '_camera_final', '_camera_correction',
        '_camera_clarification_final',
    }
    metadata = {}
    for key, value in update.metadata.items():
        if key in capability_fields and value is camera_authority:
            continue
        metadata[key] = value
    return {
        'kind': update.kind,
        'text': update.text,
        'metadata': metadata,
        'media': list(update.media) if update.media is not None else None,
    }


def _projection_from_actual_events(
    rows, goal, fold_events, *, expected_generated_quantity_text='1 кусочек',
):
    """Existing fold/test-snapshot seam; never synthesize a persisted event."""
    events = []
    projected = None
    for row in sorted(rows, key=lambda row: (_stamp(row['created_at']), row['id'])):
        metadata = row['metadata']
        raw_nutrition = metadata.get('decision_trace', {}).get('annotations', {}).get('nutrition')
        if raw_nutrition is None:
            continue
        nutrition = _plain_json_value(raw_nutrition)
        assert metadata['tenant_id'] == goal.owner_id
        assert metadata['source_principal'] == goal.principal_id
        assert metadata['source_message_id'] == goal.source_message_id
        assert metadata['gateway_session_id'] == goal.gateway_session_id
        if nutrition['record_type'] == 'meal_observation':
            assert projected is None
            projected = deepcopy(nutrition)
        else:
            assert nutrition['record_type'] == 'meal_correction' and projected is not None
            assert metadata['reply_to_source_message_id'] == goal.source_message_id
            for field in nutrition['changed_fields']:
                projected[field] = deepcopy(nutrition.get(field))
        events.append({'event_id': row['id'], 'root_source_message_id': goal.source_message_id,
                       'annotation': deepcopy(nutrition), '_created_at': _stamp(row['created_at'])})
    assert len(events) == 2 and len({event['event_id'] for event in events}) == 2
    folded = fold_events(events, goal.meal_timezone)
    assert folded['revision'] == 2 and folded['consumed'] is True
    assert folded['energy_kcal_best'] == 53
    assert projected['items'][0]['quantity_text'] == expected_generated_quantity_text
    latest = next(row for row in rows if row['id'] == folded['latest_event_id'])
    record = {
        'meal_id': goal.canonical_meal_id, 'revision': folded['revision'], 'status': 'active',
        'latest_event_id': folded['latest_event_id'], 'day': folded['meal_date'],
        'provisional': folded['meal_at'] is None, 'capture_time': _stamp(latest['created_at']).isoformat(),
        'meal_at': folded['meal_at'], 'meal_date': folded['meal_date_field'],
        'source_message_id': goal.source_message_id, 'ingest_source': 'dropbox_camera',
        'confirmation_required': True, 'reply_to_source_message_id': None,
        'received_at': None, 'is_forwarded': False, 'source_message_at': None,
        'is_estimate': projected.get('is_estimate', True), 'basis': projected.get('basis', []),
        'consumption_status': 'consumed', 'items': projected.get('items', []),
        'confidence': projected.get('confidence', 'medium'),
        'assumptions': projected.get('assumptions', []), 'warnings': projected.get('warnings', []),
    }
    for field in ('energy_kcal_min', 'energy_kcal_max', 'energy_kcal_best',
                  'protein_g', 'fat_g', 'carbohydrate_g'):
        record[field] = projected.get(field)
    start = datetime.combine(goal.meal_date, datetime.min.time(), timezone.utc)
    end = goal.trajectory_as_of
    queried = end + timedelta(seconds=1)
    return {
        'complete': True, 'user_id': goal.canonical_owner_id, 'login': goal.canonical_login,
        'start': start.isoformat(), 'end': end.isoformat(), 'queried_at': queried.isoformat(),
        'meals': [record], 'unassigned': [],
    }, folded, queried


@pytest.mark.asyncio
@pytest.mark.parametrize(('repeat_transport', 'generated_quantity_text', 'typed_wire_shape'), [
    ('native', None, False),
    ('typed', None, False),
    ('typed', '1 кусочек ', True),
])
async def test_maintained_camera_correction_history_reaches_normal_a1(
    tmp_path, monkeypatch, repeat_transport, generated_quantity_text, typed_wire_shape,
):
    evidence_path = tmp_path / 'camera-correction-runtime-evidence-private.json'
    state = {'phase': 'module_attestation', 'normal_a1': None,
             'projection_kind': 'synthetic projection from actual in-memory Honcho events through existing fold/test seam'}
    holders = {}

    def checkpoint(phase):
        state['phase'] = phase
        if 'honcho' in holders:
            state['actual_honcho_messages'] = [_safe(row) for row in holders['honcho'].messages]
        evidence_path.write_text(json.dumps(state, default=_safe, ensure_ascii=False, indent=2)+'\n')

    checkpoint('module_attestation')
    try:
        from ohmo.evals import nutrition_persistence as grader
        from ohmo.gateway.camera import CAMERA_AUTHORITY
        from tests.test_ohmo import test_camera_completed_replay as completed
        from tests.test_ohmo import test_camera_replay_eval_capture as capture

        module_path = Path(grader.__file__).resolve()
        assert module_path == REPO_ROOT / 'ohmo/evals/nutrition_persistence.py'
        assert Path(grader.validate_dialogue_binding.__code__.co_filename).resolve() == module_path
        assert Path(completed.__file__).resolve() == REPO_ROOT / 'tests/test_ohmo/test_camera_completed_replay.py'
        assert Path(capture.__file__).resolve() == REPO_ROOT / 'tests/test_ohmo/test_camera_replay_eval_capture.py'
        assert Path(completed._save_callback_portion.__code__.co_filename).resolve() == Path(completed.__file__).resolve()
        assert Path(capture._stream.__code__.co_filename).resolve() == Path(capture.__file__).resolve()
        assert Path(capture._install_correction_model.__code__.co_filename).resolve() == Path(capture.__file__).resolve()
        source_sha = hashlib.sha256(module_path.read_bytes()).hexdigest()
        state['module_attestation'] = {'file': str(module_path), 'before_sha256': source_sha,
                                     'binder_code_filename': grader.validate_dialogue_binding.__code__.co_filename}
        helper_paths = [Path(completed.__file__).resolve(), Path(capture.__file__).resolve(),
                        REPO_ROOT / 'tests/test_ohmo/test_nutrition_dialogue_stream.py',
                        REPO_ROOT / 'tests/test_ohmo/test_camera_ingress.py',
                        REPO_ROOT / 'ohmo/gateway/runtime.py', REPO_ROOT / 'ohmo/gateway/camera.py',
                        REPO_ROOT / 'ohmo/evals/recorder.py']
        state['source_before_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in helper_paths}

        # Observe the maintained helper's actual initial envelope and pool. Run that
        # envelope before its FIRST callback; never insert or relabel an episode later.
        original_ingress, original_pool, original_callback = completed._ingress, completed.runtime_pool, completed._native_callback

        def observe_ingress(*args, **kwargs):
            result = original_ingress(*args, **kwargs)
            ingress, _, bus, _ = result
            holders['ingress'] = ingress
            consume = bus.consume_inbound
            async def observe_consumed():
                message = await consume()
                if message.sender_id == '__camera__':
                    assert 'initial' not in holders
                    holders['initial'] = message
                return message
            monkeypatch.setattr(bus, 'consume_inbound', observe_consumed)
            return result

        def observe_pool(*args, **kwargs):
            pool = original_pool(*args, **kwargs)
            assert Path(pool.stream_message.__code__.co_filename).resolve() == REPO_ROOT / 'ohmo/gateway/runtime.py'
            holders['pool'], holders['honcho'] = pool, args[2]
            return pool

        async def callback_after_initial(*args, **kwargs):
            if not holders.get('initial_streamed'):
                checkpoint('capture_actual_initial_before_first_callback')
                initial_updates = await capture._stream(holders['pool'], holders['initial'], holders['ingress'])
                state['initial_updates'] = [asdict(update) for update in initial_updates]
                assert any(update.kind == 'final' for update in initial_updates)
                assert holders['honcho'].messages == [], 'Initial analysis unexpectedly appended an owner exchange'
                holders['initial_streamed'] = True
                checkpoint('maintained_helper_original_clarification_and_save')
            return await original_callback(*args, **kwargs)

        monkeypatch.setattr(completed, '_ingress', observe_ingress)
        monkeypatch.setattr(completed, 'runtime_pool', observe_pool)
        monkeypatch.setattr(completed, '_native_callback', callback_after_initial)
        checkpoint('maintained_helper_setup')
        ingress, bus, request, pool, honcho, saved_portion, saved_final = await completed._save_callback_portion(tmp_path, monkeypatch)
        attempt = ingress._attempts[request['candidate_id']]
        original_id = saved_final.metadata['nutrition_append_event_id']
        original_row = next(row for row in honcho.messages if row.id == original_id)
        original_rows = [_safe(row) for row in honcho.messages]
        original_commit = deepcopy(attempt['camera_commit'])
        original_operation = original_row.metadata['client_op_id']
        capture._install_correction_model(
            pool, saved_portion, generated_quantity_text=generated_quantity_text
        )
        changed = await capture._native_callback(bus, label='1 кусочек', target=attempt['reply_ids'][0],
            options=['1 кусочек', '2 кусочка', 'Половину порции'], prompt='Сколько съели?')
        changed.metadata['callback_query_id'] = 'g8-current-quantity-correction'
        ingress.process_real_inbound(changed)
        checkpoint('actual_quantity_correction')
        changed_updates = await capture._stream(pool, changed, ingress)
        corrected_final = next(update for update in changed_updates if update.kind == 'final')
        correction_id = corrected_final.metadata['nutrition_append_event_id']
        correction_turn = changed.metadata['_camera_turn_id']
        assert correction_id != original_id and f'{correction_turn}:assistant' != original_operation
        assert [_safe(row) for row in honcho.messages[:len(original_rows)]] == original_rows
        assert attempt['camera_commit'] == original_commit
        assert attempt['camera_correction_commit']['event_id'] == correction_id
        capture._note_delivery(ingress, corrected_final, 880)
        original_receipt = await pool._shadow_backend_for_scope(None).reconcile_durable_exchange(
            original_row.metadata['client_op_id'].removesuffix(':assistant')+':user', original_operation)
        correction_receipt = await pool._shadow_backend_for_scope(None).reconcile_durable_exchange(
            f'{correction_turn}:user', f'{correction_turn}:assistant')
        assert original_receipt.assistant_message_id == original_id
        assert correction_receipt.assistant_message_id == correction_id
        state['actual_receipts'] = {'original': asdict(original_receipt), 'correction': asdict(correction_receipt)}
        state['actual_operations'] = {'original': original_operation, 'correction': f'{correction_turn}:assistant'}
        before_repeat = [_safe(row) for row in honcho.messages]
        turns_before = len(pool._test_bundle.engine.turns)
        if repeat_transport == 'native':
            replay = await capture._native_callback(bus, label='1 кусочек', target=attempt['reply_ids'][0],
                options=['1 кусочек', '2 кусочка', 'Половину порции'], prompt='Сколько съели?')
            replay.metadata['callback_query_id'] = 'g11-current-correction-repeat'
        else:
            typed_metadata = {
                "message_id": 777 if typed_wire_shape else 'g11-current-correction-repeat-typed',
                "reply_to_message_id": str(attempt['reply_ids'][0]),
                "_telegram_raw_text": '1 кусочек', 'is_group': False,
            }
            if not typed_wire_shape:
                typed_metadata['chat_type'] = 'private'
            replay = InboundMessage(
                channel='telegram', sender_id='123', chat_id='123', content='1 кусочек',
                metadata=typed_metadata,
            )
        ingress.process_real_inbound(replay)
        checkpoint('actual_current_correction_repeat')
        replay_updates = await capture._stream(pool, replay, ingress)
        replay_final = next(update for update in replay_updates if update.kind == 'final')
        assert replay_final.text == 'Эта порция уже записана.'
        assert replay_final.metadata['nutrition_append_event_id'] == correction_id
        assert len(pool._test_bundle.engine.turns) == turns_before
        assert [_safe(row) for row in honcho.messages] == before_repeat
        assert attempt['camera_commit'] == original_commit
        state['actual_updates'] = {'original': _diagnostic_update(saved_final, CAMERA_AUTHORITY),
            'correction': [_diagnostic_update(update, CAMERA_AUTHORITY) for update in changed_updates],
            'repeat': [_diagnostic_update(update, CAMERA_AUTHORITY) for update in replay_updates]}
        checkpoint('ordinary_export_all_actual_episodes')
        observed_episodes = capture._exported_episodes(pool)
        state['export'] = grader.export_eval_dialogue(pool._workspace / 'evals',
            episode_ids=[item['episode']['episode_id'] for item in observed_episodes])
        episodes = state['export']['episodes']
        assert episodes == observed_episodes, 'Normal exporter differs from the maintained export helper'
        assert len(episodes) == 5, 'Expect actual initial + clarification + original save + correction + repeat'
        assert all(item['dialogue_complete'] is True for item in episodes)
        assert sum(item.get('trusted_camera_context', {}).get('kind') == 'initial_context' for item in episodes) == 1
        for item in episodes[1:4]:
            context = item['trusted_camera_context']
            click = item['episode']['metadata']['inbound']['metadata']
            assert click['callback_query'] is True
            assert click['_camera_route'] == 'callback'
            assert click['_camera_ingress_callback_eligible'] is True
            assert '_camera_existing_meal_replay' not in click
            assert str(click['message_id']) == context['source_message_id']
            assert click['native_message_id'] == click['message_id']
            assert click['_camera_native_binding'] == str(click['message_id'])
        repeat_metadata = episodes[4]['episode']['metadata']['inbound']['metadata']
        if repeat_transport == 'typed':
            assert repeat_metadata['message_id'] == (
                777 if typed_wire_shape else 'g11-current-correction-repeat-typed'
            )
            assert repeat_metadata['reply_to_message_id'] == str(attempt['reply_ids'][0])
            assert repeat_metadata['_camera_correction_replay_typed'] is True
            assert not set(grader._CAMERA_CALLBACK_SHAPE_FIELDS) & repeat_metadata.keys()
            assert episodes[4]['dialogue'][0]['text'] == repeat_metadata['_telegram_raw_text'] == '1 кусочек'
            assert ('chat_type' not in repeat_metadata) is typed_wire_shape
        else:
            assert repeat_metadata['callback_query'] is True
            assert repeat_metadata['_camera_route'] == 'callback'
            assert repeat_metadata['callback_query_id'] == 'g11-current-correction-repeat'
        correction_episode = next(item for item in episodes if item['episode']['episode_id'] ==
            next(row for row in honcho.messages if row.id == correction_id).metadata['decision_trace_episode_id'])
        assert correction_episode['trusted_camera_context']['operation_id'] == f'{correction_turn}:assistant'
        assert correction_episode['trusted_camera_context']['candidate_id'] == request['candidate_id']
        correction_row = next(row for row in honcho.messages if row.id == correction_id)
        raw_persisted_correction = correction_row.metadata['decision_trace']['annotations']['nutrition']
        raw_executed_correction = correction_episode['turn_provenance'][0][
            'gateway_final_metadata']['nutrition_finalization']['annotation']
        assert raw_executed_correction == raw_persisted_correction
        parsed_correction = grader._event_from_raw(_safe(correction_row))
        assert parsed_correction['annotation'] == grader._validated_nutrition_annotation(
            raw_persisted_correction)
        assert parsed_correction['annotation'] == grader._validated_nutrition_annotation(
            raw_executed_correction)
        assert grader._validated_nutrition_annotation(parsed_correction['annotation']) is None
        metadata = original_row.metadata
        capture_date = _stamp(request['capture_time']).date()
        started = min(_stamp(item['episode']['created_at']) for item in episodes)
        as_of = max(_stamp(item['episode']['created_at']) for item in episodes)
        goal = grader.Goal(case_id='g8-actual-camera-correction-history',
            episode_ids=[item['episode']['episode_id'] for item in episodes], owner_id=metadata['tenant_id'],
            principal_id=metadata['source_principal'], workspace_id=original_row.workspace_id,
            eval_workspace=str(pool._workspace), peer_id=original_row.peer_id,
            canonical_owner_id=metadata['tenant_id'], canonical_login='synthetic-owner', session_id=original_row.session_id,
            gateway_session_id=metadata['gateway_session_id'], source_message_id=metadata['source_message_id'],
            meal_date=capture_date, meal_timezone='UTC', trajectory_started_at=started, trajectory_as_of=as_of,
            logical_turn_id=metadata['logical_turn_id'], trace_episode_id=metadata['decision_trace_episode_id'],
            operation_id=original_operation, canonical_meal_id=grader.derive_meal_id(tenant_id=metadata['tenant_id'],
                source_principal=metadata['source_principal'], gateway_session_id=metadata['gateway_session_id'],
                source_message_id=metadata['source_message_id']), expected_consumed=True, expected_kcal=53.0,
            tolerance_fraction=0.30, expectation_origin='explicit_fixture',
            expectation_source='maintained-offline-runtime-helper-current-quantity-53kcal')
        manifest = grader.Manifest(schema_version=1, goals=[goal])
        rows = [_safe(row) for row in honcho.messages]
        scoped_rows = _honcho_query_rows(rows, goal)
        assert len(rows) == 6 and len(scoped_rows) == 3
        assert any('nutrition' not in row['metadata'].get('decision_trace', {}).get('annotations', {})
                   for row in scoped_rows), 'Bounded query dropped the non-nutrition clarification'
        state['full_actual_honcho_history'] = rows
        state['goal'] = goal.model_dump(mode='json')
        binding = grader.validate_dialogue_binding(manifest, state['export'])[goal.case_id]
        state['binding'] = binding
        checkpoint('normal_binding_before_projection')
        assert binding['complete'] is True, binding
        checkpoint('synthetic_projection_from_actual_events')
        snapshot, folded, now = _projection_from_actual_events(
            scoped_rows, goal, grader._fold_events,
            expected_generated_quantity_text=generated_quantity_text or '1 кусочек',
        )
        honcho_snapshot = {'complete': True, 'workspace_id': goal.workspace_id, 'session_id': goal.session_id,
            'owner_id': goal.owner_id, 'since': started.isoformat(), 'until': as_of.isoformat(),
            'queried_at': now.isoformat(), 'messages': scoped_rows}
        canonical = grader.bind_wellness_snapshot(snapshot, goal=goal)
        state.update({'goal': goal.model_dump(mode='json'), 'honcho_snapshot': honcho_snapshot,
            'synthetic_projection': snapshot, 'folded_actual_events': folded, 'bound_canonical': canonical})
        state['normal_a1'] = grader.grade_manifest(manifest, honcho_snapshot, canonical, now=now,
            reviewed_turn_sources=binding['reviewed_turn_sources'],
            reviewed_turn_provenance=binding['reviewed_turn_provenance'])[0]
        checkpoint('normal_result')
        state['module_attestation']['after_sha256'] = hashlib.sha256(module_path.read_bytes()).hexdigest()
        state['source_after_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in helper_paths}
        assert state['module_attestation']['after_sha256'] == source_sha, 'Grader changed during peer execution'
        assert state['source_after_sha256'] == state['source_before_sha256'], 'Runtime/helpers changed during peer execution'
        assert binding['complete'] is True, binding
        assert state['normal_a1']['a1'] == 'PASS', state['normal_a1']

        def assert_actual_correction_mutation_rejected(name, mutate, *, binding_must_fail):
            bad_export = deepcopy(state['export'])
            bad_honcho = deepcopy(honcho_snapshot)
            mutate(bad_export, bad_honcho)
            bad_binding = grader.validate_dialogue_binding(manifest, bad_export)[goal.case_id]
            bad_canonical = grader.bind_wellness_snapshot(snapshot, goal=goal)
            bad_grade = grader.grade_manifest(
                manifest, bad_honcho, bad_canonical, now=now,
                reviewed_turn_sources=bad_binding.get('reviewed_turn_sources', {}),
                reviewed_turn_provenance=bad_binding.get('reviewed_turn_provenance', {}),
            )[0]
            if binding_must_fail:
                assert bad_binding['complete'] is False, (name, bad_binding)
                assert bad_binding['reviewed_turn_sources'] == {}, (name, bad_binding)
                assert bad_binding['reviewed_turn_provenance'] == {}, (name, bad_binding)
            assert bad_grade['a1'] != 'PASS', (name, bad_grade)

        def wrong_correction_operation(export, _honcho):
            item = export['episodes'][3]
            item['trusted_camera_context']['operation_id'] = 'forged-correction:assistant'
            item['episode']['metadata']['trusted_camera_context']['operation_id'] = 'forged-correction:assistant'

        def wrong_finalizer(export, _honcho):
            annotation = export['episodes'][3]['turn_provenance'][0]['gateway_final_metadata']
            annotation['nutrition_finalization']['annotation']['energy_kcal_best'] = 54.0

        def wrong_append_receipt(export, _honcho):
            export['episodes'][3]['turn_provenance'][0]['gateway_final_metadata'][
                'nutrition_append_event_id'] = 'unmatched-correction-event'

        def wrong_native_source(export, _honcho):
            export['episodes'][3]['episode']['metadata']['inbound']['metadata'][
                '_camera_native_binding'] = '501'

        def wrong_correction_owner(export, _honcho):
            export['episodes'][3]['trusted_camera_context']['tenant_id'] = 'other-owner'
            export['episodes'][3]['episode']['metadata']['trusted_camera_context'][
                'tenant_id'] = 'other-owner'

        def changed_repeat_selection(export, _honcho):
            repeated = export['episodes'][4]
            metadata = repeated['episode']['metadata']['inbound']['metadata']
            if repeat_transport == 'typed':
                metadata['_telegram_raw_text'] = '2 кусочка'
            else:
                metadata.update({
                    'native_keyboard_selected_index': 1,
                    'native_keyboard_selected_label': '2 кусочка',
                    'callback_data': 'ask:1',
                    'native_keyboard_reflection': '✅ 2 кусочка',
                })
            repeated['dialogue'][0]['text'] = '2 кусочка'

        def wrong_typed_source(export, _honcho):
            export['episodes'][4]['episode']['metadata']['inbound']['metadata'][
                'message_id'] = 'different-typed-source'

        def wrong_typed_operation(export, _honcho):
            repeated = export['episodes'][4]
            logical = 'forged-typed-repeat-turn'
            operation = f'{logical}:assistant'
            repeated['trusted_camera_context'].update(
                logical_turn_id=logical, operation_id=operation)
            episode_metadata = repeated['episode']['metadata']
            episode_metadata['trusted_camera_context'].update(
                logical_turn_id=logical, operation_id=operation)
            episode_metadata['trusted_camera_turn_provenance'].update(
                logical_turn_id=logical, operation_id=operation)
            episode_metadata['inbound']['metadata']['_camera_turn_id'] = logical
            repeated['turn_provenance'][0].update(
                logical_turn_id=logical, operation_id=operation)

        def wrong_typed_receipt(export, _honcho):
            export['episodes'][4]['turn_provenance'][0][
                'gateway_final_metadata']['nutrition_append_event_id'] = 'forged-current-receipt'

        def wrong_typed_owner(export, _honcho):
            repeated = export['episodes'][4]
            repeated['trusted_camera_context']['tenant_id'] = 'different-owner'
            repeated['episode']['metadata']['trusted_camera_context']['tenant_id'] = 'different-owner'

        def repeat_before_correction_receipt(export, _honcho):
            correction_time = _stamp(export['episodes'][3]['episode']['created_at'])
            export['episodes'][4]['episode']['created_at'] = (correction_time - timedelta(milliseconds=1)).isoformat()

        for name, mutation, binding_must_fail in (
            ('correction operation', wrong_correction_operation, True),
            ('executed sparse finalizer', wrong_finalizer, False),
            ('correction append receipt', wrong_append_receipt, False),
            ('native source target', wrong_native_source, True),
            ('correction owner', wrong_correction_owner, True),
            ('repeat selection', changed_repeat_selection, repeat_transport != 'typed'),
            ('repeat chronology', repeat_before_correction_receipt, False),
        ):
            assert_actual_correction_mutation_rejected(
                name, mutation, binding_must_fail=binding_must_fail,
            )
        if repeat_transport == 'typed':
            for name, mutation, binding_must_fail in (
                ('typed current quantity mismatch', changed_repeat_selection, False),
                ('typed source mismatch', wrong_typed_source, True),
                ('typed correction operation mismatch', wrong_typed_operation, True),
                ('typed receipt mismatch', wrong_typed_receipt, False),
                ('typed owner mismatch', wrong_typed_owner, True),
                ('typed replay chronology', repeat_before_correction_receipt, False),
            ):
                assert_actual_correction_mutation_rejected(
                    name, mutation, binding_must_fail=binding_must_fail,
                )
    except BaseException as exc:
        state['failure'] = {'type': type(exc).__name__, 'message': str(exc), 'phase': state['phase']}
        checkpoint(state['phase'])
        raise
    finally:
        if 'ingress' in holders:
            await holders['ingress'].close()

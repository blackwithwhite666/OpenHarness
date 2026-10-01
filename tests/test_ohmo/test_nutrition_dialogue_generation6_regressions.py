"""Six bounded frozen-candidate concerns; all data and model output synthetic."""

import copy
from datetime import datetime, timedelta, timezone

import pytest

from tests.test_ohmo.test_nutrition_dialogue_review_regressions import (
    _open_camera,
    _pool,
    _Honcho,
    _ScriptedEngine,
    _trace,
    _consumed_trace,
    _owner_message,
    _turn,
)
from tests.test_ohmo.test_camera_ingress import _ingress, _candidate, _admit
from ohmo.gateway.camera import (
    CAMERA_AUTHORITY,
    CAMERA_CONTEXT_QUESTION_AUTHORITY,
    CameraIngress,
)


def engine(pool, trace, text):
    pool._test_bundle.engine = _ScriptedEngine(pool, [(trace, text)])


async def pending(tmp_path, monkeypatch):
    ingress, root, bus, telegram, request = await _open_camera(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    engine(pool, None, "Сколько примерно вы съели?")
    first = _owner_message("Я съела только часть", 9101)
    ingress.process_real_inbound(first)
    await _turn(pool, first, ingress)
    attempt = ingress._attempts[request["candidate_id"]]
    attempt["admitted_at"] = (datetime.now(timezone.utc) - timedelta(minutes=115)).isoformat()
    ingress._sweep_expired_attempts()
    return ingress, root, bus, pool, honcho, request, attempt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer,required",
    [
        ("Записала две груши: примерно 120 ккал, диапазон 100–140.", ["груши", "120", "100–140"]),
        (
            "Две груши: 120 ккал (100–140), но запись не удалось сохранить; 2+2=4.",
            ["груши", "120", "100–140", "2+2=4"],
        ),
        (
            "Две груши: 120 ккал. Сохраните рисунок в PNG, чтобы сохранить прозрачность.",
            ["груши", "120", "PNG"],
        ),
    ],
)
async def test_food_and_independent_storage_answer_survive(tmp_path, monkeypatch, answer, required):
    ingress, root, bus, telegram = _ingress(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    engine(pool, _consumed_trace(), answer)
    msg = _owner_message(
        "Я съела две груши, запиши и дай диапазон калорий; как сохранить рисунок?", 9201
    )
    try:
        result = await _turn(pool, msg, ingress)
        assert result.metadata["nutrition_append_event_id"] == "honcho-2"
        for value in [result.text, honcho.messages[1].content]:
            assert all(term in value for term in required)
            assert "не удалось сохранить" not in value
    finally:
        await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("text,reply", [("Я съела кусочек", False), ("Да", True)])
async def test_generic_bound_owner_can_ask_quantity_without_annotation(
    tmp_path, monkeypatch, text, reply
):
    ingress, root, bus, telegram, request = await _open_camera(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    attempt = ingress._attempts[request["candidate_id"]]
    before = (attempt["snapshot"], attempt["photo_id"], attempt["capture_time"])
    msg = _owner_message(text, 9301)
    if reply:
        msg.metadata["reply_to_message_id"] = attempt["photo_id"]
    ingress.process_real_inbound(msg)
    engine(pool, None, "Какой примерно размер кусочка или сколько граммов вы съели?")
    try:
        result = await _turn(pool, msg, ingress)
        assert "nutrition_append_event_id" not in result.metadata and "camera_commit" not in attempt
        assert "сколько" in result.text.casefold()
        assert (attempt["snapshot"], attempt["photo_id"], attempt["capture_time"]) == before
        trace = _consumed_trace()
        trace["annotations"]["nutrition"]["basis"] = ["image", "owner_statement"]
        engine(pool, trace, "Два кусочка: около 120 ккал.")
        quantity = _owner_message("2 кусочка", 9302)
        ingress.process_real_inbound(quantity)
        final = await _turn(pool, quantity, ingress)
        assert final.metadata["nutrition_append_event_id"] == "honcho-4"
    finally:
        await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["только наличными", "только карты", "2 часа", "полтора года"])
async def test_unrelated_shapes_do_not_force_portion_question_or_missing_trace_error(
    tmp_path, monkeypatch, text
):
    ingress, root, bus, pool, honcho, request, attempt = await pending(tmp_path, monkeypatch)
    msg = _owner_message(text, 9401)
    ingress.process_real_inbound(msg)
    answer = "Это ответ о времени или способе оплаты."
    engine(pool, None, answer)
    try:
        result = await _turn(pool, msg, ingress)
        assert result.text == answer
        assert "nutrition_append_event_id" not in result.metadata and "camera_commit" not in attempt
    finally:
        await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [False, True])
async def test_mixed_consumed_and_weather_keeps_real_meal_and_capture(tmp_path, monkeypatch, reply):
    ingress, root, bus, pool, honcho, request, attempt = await pending(tmp_path, monkeypatch)
    msg = _owner_message("Я съела две груши, запиши и скажи погоду", 9501)
    if reply:
        msg.metadata["reply_to_message_id"] = attempt["reply_ids"][-1]
    ingress.process_real_inbound(msg)
    trace = _consumed_trace()
    trace["annotations"]["nutrition"]["basis"] = ["image", "owner_statement"]
    engine(pool, trace, "Две груши: примерно 120 ккал. Прогноз погоды недоступен.")
    try:
        result = await _turn(pool, msg, ingress)
        actual = honcho.messages[-1].metadata
        assert result.metadata["nutrition_append_event_id"] == attempt["camera_commit"]["event_id"]
        assert actual["source_message_id"] == "9501"
        assert (
            actual["decision_trace"]["annotations"]["nutrition"]["meal_at"]
            == attempt["capture_time"]
        )
        assert "120" in result.text and "Прогноз" in result.text
    finally:
        await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("second_state", ["photo_sent", "clarifying"])
async def test_new_camera_denial_cannot_correct_unrelated_older_meal(
    tmp_path, monkeypatch, second_state
):
    ingress, root, bus, telegram, first = await _open_camera(tmp_path, index=981)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    trace = _consumed_trace()
    trace["annotations"]["nutrition"]["basis"] = ["image", "owner_statement"]
    engine(pool, trace, "Две груши: около 120 ккал.")
    ate = _owner_message("Я съела две груши", 9601)
    ingress.process_real_inbound(ate)
    await _turn(pool, ate, ingress)
    old = ingress._attempts[first["candidate_id"]]
    second = _candidate(root, index=982)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, second))[0] == 202
    await bus.consume_inbound()
    if second_state == "clarifying":
        engine(pool, None, "Сколько примерно вы съели?")
        part = _owner_message("Я съела только часть", 9602)
        ingress.process_real_inbound(part)
        await _turn(pool, part, ingress)
    deny = _owner_message("Я не ела", 9603)
    ingress.process_real_inbound(deny)
    try:
        if deny.metadata.get("_camera_correction") is CAMERA_AUTHORITY:
            correction = _trace(
                {
                    "schema_version": 2,
                    "record_type": "meal_correction",
                    "changed_fields": ["consumption_status"],
                    "consumption_status": "not_consumed",
                }
            )
            engine(pool, correction, "Отказ от употребления записан.")
            await _turn(pool, deny, ingress)
        assert deny.metadata.get("_camera_candidate_id") != first["candidate_id"]
        assert "camera_correction_commit" not in old
    finally:
        await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("original_status", ["unknown", "consumed"])
async def test_stored_retry_content_and_committed_metadata_authoritative(
    tmp_path, monkeypatch, original_status
):
    ingress, root, bus, telegram = _ingress(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    first_trace = _consumed_trace()
    first_trace["annotations"]["nutrition"]["consumption_status"] = original_status
    proposed = copy.deepcopy(_consumed_trace())
    proposed["annotations"]["nutrition"].update(
        meal_date="2026-09-28",
        energy_kcal_best=180,
        items=[{"name": "grapes", "quantity_text": "three handfuls"}],
    )
    pool._test_bundle.engine = _ScriptedEngine(
        pool,
        [
            (first_trace, "Две груши: 120 ккал."),
            (proposed, "Готово, еда учтена! Виноград: 180 ккал."),
        ],
    )
    original = _owner_message("Запиши две съеденные груши", 9701)
    try:
        await _turn(pool, original, ingress)
        await pool._shadow_backend_for_scope(None).await_pending()
        retry = _owner_message(original.content, 9701)
        retry.timestamp = original.timestamp
        result = await _turn(pool, retry, ingress)
        assert len(honcho.messages) == 2
        if original_status == "consumed":
            assert result.text == honcho.messages[1].content
            assert result.metadata["nutrition_committed_annotation"]["energy_kcal_best"] == 120
            assert result.metadata["nutrition_committed_annotation"]["meal_date"] is None
            assert result.metadata["nutrition_model_proposal_annotation"]["energy_kcal_best"] == 180
            assert result.metadata["nutrition_proposal_matches_committed"] is False
        else:
            assert "nutrition_append_event_id" not in result.metadata
            assert "учтена" not in result.text and "Готово" not in result.text
    finally:
        await ingress.close()


@pytest.mark.asyncio
async def test_context_question_receipt_survives_restart_and_bare_quantity_keeps_capture(
    tmp_path, monkeypatch
):
    ingress, root, bus, _telegram = _ingress(tmp_path)
    capture = datetime(2026, 9, 27, 7, tzinfo=timezone.utc)
    request = _candidate(root, index=772, capture_time=capture)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    await bus.consume_inbound()
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    attempt = ingress._attempts[request["candidate_id"]]
    attempt["admitted_at"] = (datetime.now(timezone.utc) - timedelta(minutes=115)).isoformat()
    ingress._save_attempts()
    ingress._sweep_expired_attempts()
    original = (attempt["snapshot"], attempt["photo_id"], attempt["capture_time"])

    hint = _owner_message("только виноград", 7721)
    ingress.process_real_inbound(hint)
    assert hint.metadata.get("_camera_answer") is None
    assert hint.metadata.get("_camera_context_question") is CAMERA_CONTEXT_QUESTION_AUTHORITY
    assert hint.metadata["_camera_candidate_id"] == request["candidate_id"]
    assert attempt["snapshot"] in hint.media
    pool._test_bundle.engine = _ScriptedEngine(
        pool, [(None, "Сколько горстей винограда вы съели?")]
    )
    question = await _turn(pool, hint, ingress)
    assert "Сколько" in question.text
    assert "nutrition_append_event_id" not in question.metadata
    assert attempt.get("camera_commit") is None
    await pool._shadow_backend_for_scope(None).await_pending()
    assert len(honcho.messages) == 2
    assert honcho.messages[1].metadata["camera_finalizer_outcome"] == "clarification"
    assert honcho.messages[1].metadata["camera_context_only"] is True
    assert "decision_trace" not in honcho.messages[1].metadata

    await ingress.close()
    reopened = CameraIngress(ingress.config, workspace=tmp_path, bus=bus, telegram=_telegram)
    reopened.mark_restart_unknown()
    pool._camera_ingress = reopened

    repeated_hint = _owner_message("только виноград", 7721)
    repeated_hint.timestamp = hint.timestamp
    reopened.process_real_inbound(repeated_hint)
    replay = await _turn(pool, repeated_hint, reopened)
    assert replay.text == question.text
    assert len(honcho.messages) == 2

    trace = _consumed_trace()
    trace["annotations"]["nutrition"].update(
        basis=["image", "owner_statement"],
        items=[{"name": "grapes", "quantity_text": "3 handfuls"}],
    )
    pool._test_bundle.engine = _ScriptedEngine(
        pool, [(trace, "Три горсти винограда: примерно 120 ккал.")]
    )
    quantity = _owner_message("3 горсти винограда", 7722)
    reopened.process_real_inbound(quantity)
    assert quantity.metadata.get("_camera_candidate_id") == request["candidate_id"]
    assert quantity.metadata.get("_camera_answer") == "yes"
    assert original[0] in quantity.media
    final = await _turn(pool, quantity, reopened)
    saved = honcho.messages[-1]
    nutrition = saved.metadata["decision_trace"]["annotations"]["nutrition"]
    assert final.metadata["nutrition_append_event_id"] == saved.id
    assert final.metadata["nutrition_sync_status"] == "pending"
    assert saved.metadata["ingest_source"] == "dropbox_camera"
    assert saved.metadata["source_message_id"] == "7722"
    assert nutrition["items"][0]["name"] == "grapes"
    assert nutrition["items"][0]["quantity_text"] == "3 handfuls"
    assert nutrition["meal_at"] == original[2]
    assert len(honcho.messages) == 4
    assert (
        reopened._attempts[request["candidate_id"]]["snapshot"],
        reopened._attempts[request["candidate_id"]]["photo_id"],
        reopened._attempts[request["candidate_id"]]["capture_time"],
    ) == original
    await reopened.close()

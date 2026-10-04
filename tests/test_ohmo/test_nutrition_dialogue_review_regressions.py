"""Promoted synthetic acceptance probes for nutrition dialogue repairs."""

from __future__ import annotations

import base64
from io import BytesIO
from datetime import datetime, timedelta, timezone
from PIL import Image

import pytest

from openharness.channels.bus.events import InboundMessage
from openharness.engine.messages import AttachmentRefBlock
from openharness.evals import TRACE_FINALIZATION
from openharness.engine.stream_events import AssistantTextDelta
from openharness.mcp.types import McpToolInfo
from openharness.tools.base import ToolExecutionContext, ToolRegistry
from openharness.tools.mcp_tool import McpToolAdapter
from ohmo.attachment_store import AttachmentStore
from ohmo.conversation_image_tool import LoadConversationImageInput, LoadConversationImageTool
from ohmo.gateway.camera import CAMERA_AUTHORITY, CameraIngress
from ohmo.gateway.runtime import OhmoSessionRuntimePool, _build_inbound_user_message
from ohmo.workspace import initialize_workspace
from tests.test_ohmo.test_camera_ingress import FakeTelegram, _admit, _candidate, _ingress
from tests.test_ohmo.test_nutrition_dialogue_stream import (
    PNG_BYTES,
    _Honcho,
    _pool,
    _turn,
)


class _ScriptedEngine:
    def __init__(self, pool, steps):
        self.pool = pool
        self.steps = list(steps)
        self.decision_trace_recorder = None
        self.tool_metadata = {}
        self.messages = []

    def set_decision_trace_recorder(self, recorder):
        self.decision_trace_recorder = recorder

    def set_system_prompt(self, _prompt):
        pass

    async def submit_message(self, user_message):
        self.messages.append(user_message)
        payload, answer = self.steps.pop(0)
        self.decision_trace_recorder.trace_requirement_signals(
            self.pool._active_message.content
        )
        if getattr(self, "read_wellness", False):
            tool = self.pool._test_bundle.tool_registry.get(
                "mcp__worfalomey__get_wellness_data"
            )
            result = await tool.execute(
                tool.input_model(params={"interval": "1d"}),
                ToolExecutionContext(cwd=self.pool._workspace),
            )
            assert result.is_error is False
        if payload is not None:
            self.decision_trace_recorder.record(TRACE_FINALIZATION, payload)
        yield AssistantTextDelta(text=answer)


def _trace(annotation):
    return {
        "schema_version": 1,
        "trace_event_id": "synthetic-review-finalization",
        "annotations": {"nutrition": annotation},
    }


def _consumed_trace():
    return _trace({
        "schema_version": 2,
        "record_type": "meal_observation",
        "consumption_status": "consumed",
        "basis": ["owner_statement"],
        "energy_kcal_best": 120,
        "items": [{"name": "pears", "quantity_text": "2 pears"}],
    })


def _owner_message(text: str, message_id: int) -> InboundMessage:
    return InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content=text,
        metadata={"message_id": message_id, "_telegram_raw_text": text, "is_group": False},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("consumption", ["unknown", "consumed"])
async def test_v1_estimates_keep_ordinary_runtime_compatibility(
    tmp_path, monkeypatch, consumption
):
    ingress, _root, _bus, _telegram = _ingress(tmp_path)
    pool = _pool(tmp_path, ingress, _Honcho(), monkeypatch)
    v1 = _trace({
        "schema_version": 1,
        "record_type": "meal_estimate",
        "consumption_status": consumption,
        "basis": ["owner_statement"],
        "energy_kcal_best": 120,
    })
    answer = "Груши: примерно 120 ккал."
    pool._test_bundle.engine = _ScriptedEngine(pool, [(v1, answer)])
    result = await _turn(pool, _owner_message("Оцени калории груш", 8501), ingress)
    assert result.text == answer
    assert "nutrition_append_event_id" not in result.metadata
    await ingress.close()


@pytest.mark.asyncio
async def test_ordinary_success_preserves_answer_and_is_receipt_bound(tmp_path, monkeypatch):
    ingress, _root, _bus, _telegram = _ingress(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    answer = "Две груши, 120 ккал (примерно 100–140). Оценка неточная без массы."
    pool._test_bundle.engine = _ScriptedEngine(pool, [(_consumed_trace(), answer)])
    result = await _turn(
        pool,
        _owner_message("Съела две груши; запиши и скажи калории с неопределённостью", 8502),
        ingress,
    )
    assert result.metadata["nutrition_append_event_id"] == "honcho-2"
    assert result.metadata["nutrition_sync_status"] == "pending"
    assert answer in result.text
    assert "не привязан к дате" in result.text
    assert answer in honcho.messages[1].content
    assert "не привязан к дате" in honcho.messages[1].content
    await ingress.close()


@pytest.mark.asyncio
async def test_storage_status_drops_conflicting_projection_clause_and_preserves_other_answer(
    tmp_path, monkeypatch
):
    ingress, _root, _bus, _telegram = _ingress(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    answer = (
        "Две порции риса, около 120 ккал (100–140). В проекции пока пусто, "
        "запись не удалось сохранить. 2+2=4."
    )
    pool._test_bundle.engine = _ScriptedEngine(pool, [(_consumed_trace(), answer)])
    result = await _turn(pool, _owner_message("Запиши съеденный рис и посчитай 2+2", 8505), ingress)
    assert result.metadata["nutrition_append_event_id"] == honcho.messages[1].id
    assert "120" in result.text and "100–140" in result.text and "2+2=4" in result.text
    assert "проекции" not in result.text.casefold()
    assert result.text == honcho.messages[1].content
    await ingress.close()


@pytest.mark.asyncio
async def test_retry_renders_authoritative_stored_annotation_and_content(tmp_path, monkeypatch):
    ingress, _root, _bus, _telegram = _ingress(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    original = _consumed_trace()
    original["annotations"]["nutrition"].update(
        items=[{"name": "pears", "quantity_text": "2 pears"}], energy_kcal_best=120
    )
    changed = _consumed_trace()
    changed["annotations"]["nutrition"].update(
        items=[{"name": "grapes", "quantity_text": "3 handfuls"}],
        energy_kcal_best=180,
        meal_date="2026-10-01",
    )
    pool._test_bundle.engine = _ScriptedEngine(
        pool, [(original, "Две груши: 120 ккал."), (changed, "Три горсти винограда: 180 ккал.")]
    )
    source = _owner_message("Запиши две груши", 8506)
    first = await _turn(pool, source, ingress)
    retry = _owner_message(source.content, 8506)
    retry.timestamp = source.timestamp
    second = await _turn(pool, retry, ingress)
    assert len(honcho.messages) == 2
    assert second.metadata["nutrition_append_event_id"] == first.metadata["nutrition_append_event_id"]
    assert "120" in second.text and "180" not in second.text
    assert "не привязан к дате" in second.text
    assert second.metadata["nutrition_proposal_matches_committed"] is False
    assert second.metadata["nutrition_model_proposal_annotation"]["items"][0]["name"] == "grapes"
    assert second.metadata["nutrition_committed_annotation"]["items"][0]["name"] == "pears"
    assert second.text == honcho.messages[1].content
    await ingress.close()


@pytest.mark.asyncio
async def test_dated_new_meals_stay_ordinary_and_replay_deduplicates(tmp_path, monkeypatch):
    ingress, _root, _bus, _telegram, request = await _open_camera(tmp_path, index=819)
    attempt = ingress._attempts[request["candidate_id"]]
    attempt["admitted_at"] = (datetime.now(timezone.utc) - timedelta(minutes=115)).isoformat()
    ingress._sweep_expired_attempts()
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)

    steps = []
    for day in ("2026-09-28", "2026-09-29", "2026-09-28"):
        trace = _consumed_trace()
        trace["annotations"]["nutrition"].update(
            meal_date=day,
            explicit_new_consumption=True,
            basis=["owner_statement"],
            items=[{"name": "oatmeal", "quantity_text": "one bowl"}],
        )
        steps.append((trace, f"Овсянка за {day}: примерно 240 ккал."))
    pool._test_bundle.engine = _ScriptedEngine(pool, steps)

    first_message = _owner_message("Я съела новый завтрак 2026-09-28, запиши", 8191)
    ingress.process_real_inbound(first_message)
    assert first_message.metadata.get("_camera_authority") is not CAMERA_AUTHORITY
    assert first_message.metadata.get("_camera_unbound") is not CAMERA_AUTHORITY
    first = await _turn(pool, first_message, ingress)
    assert first.metadata["nutrition_append_event_id"] == "honcho-2"

    second_message = _owner_message("Я съела новый завтрак 2026-09-29, запиши", 8192)
    second = await _turn(pool, second_message, ingress)
    assert second.metadata["nutrition_append_event_id"] == "honcho-4"
    assert second.metadata["nutrition_append_event_id"] != first.metadata["nutrition_append_event_id"]

    replay = _owner_message(first_message.content, 8191)
    replay.timestamp = first_message.timestamp
    repeated = await _turn(pool, replay, ingress)
    assert repeated.metadata["nutrition_append_event_id"] == first.metadata["nutrition_append_event_id"]
    assert repeated.metadata["nutrition_proposal_matches_committed"] is True
    assert len(honcho.messages) == 4
    assert attempt["state"] == "photo_sent"
    assert "camera_commit" not in attempt
    await ingress.close()


@pytest.mark.asyncio
async def test_wellness_projection_read_precedes_append_and_cannot_override_receipt(
    tmp_path, monkeypatch
):
    ingress, _root, _bus, _telegram = _ingress(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)

    class Manager:
        calls = 0

        async def call_tool(self, server_name, tool_name, arguments):
            assert (server_name, tool_name) == ("worfalomey", "get_wellness_data")
            assert arguments["params"]["interval"] == "1d"
            assert honcho.messages == []
            self.calls += 1
            return "synthetic empty pre-append nutrition projection"

    manager = Manager()
    pool._test_bundle.tool_registry = ToolRegistry()
    pool._test_bundle.tool_registry.register(McpToolAdapter(
        manager,
        McpToolInfo(
            server_name="worfalomey",
            name="get_wellness_data",
            description="synthetic fixture wellness read",
            input_schema={
                "type": "object",
                "properties": {"params": {"type": "object"}},
            },
        ),
    ))
    answer = (
        "Рис, два кусочка, примерно 120 ккал (100–140). "
        "В базе этой еды ещё нет, сохранение не получилось. 2+2=4."
    )
    engine = _ScriptedEngine(pool, [(_consumed_trace(), answer)])
    engine.read_wellness = True
    pool._test_bundle.engine = engine
    result = await _turn(pool, _owner_message("Запиши съеденные груши", 8504), ingress)
    assert manager.calls == 1
    assert result.metadata["nutrition_append_event_id"] == "honcho-2"
    for value in (result.text, honcho.messages[1].content):
        assert "Рис" in value and "два кусочка" in value
        assert "120" in value and "100–140" in value and "2+2=4" in value
        assert "В базе" not in value and "не получилось" not in value
        assert "не привязан к дате" in value
    assert result.text == honcho.messages[1].content
    assert result.metadata["nutrition_sync_status"] == "pending"
    await ingress.close()


@pytest.mark.asyncio
async def test_retry_cannot_claim_consumption_from_new_trace_when_receipt_is_unknown(
    tmp_path, monkeypatch
):
    ingress, _root, _bus, _telegram = _ingress(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    unknown = _trace({
        "schema_version": 2,
        "record_type": "meal_observation",
        "consumption_status": "unknown",
        "basis": ["owner_statement"],
        "energy_kcal_best": 120,
    })
    answer = "Две груши: 120 ккал."
    pool._test_bundle.engine = _ScriptedEngine(
        pool, [(unknown, "Пока только оценка."), (_consumed_trace(), answer)]
    )
    original = _owner_message("Запиши две груши", 8503)
    await _turn(pool, original, ingress)
    await pool._shadow_backend_for_scope(None).await_pending()
    retry = _owner_message("Запиши две груши", 8503)
    result = await _turn(pool, retry, ingress)
    actual = honcho.messages[1].metadata["decision_trace"]["annotations"]["nutrition"]
    assert actual["consumption_status"] == "unknown"
    assert len(honcho.messages) == 2
    assert "nutrition_append_event_id" not in result.metadata
    assert "Записано" not in result.text
    await ingress.close()


async def _open_camera(tmp_path, *, index: int = 811):
    ingress, root, bus, telegram = _ingress(tmp_path, FakeTelegram())
    request = _candidate(root, index=index)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    await bus.consume_inbound()
    return ingress, root, bus, telegram, request


@pytest.mark.asyncio
async def test_clarification_restart_replays_question_receipt_without_new_exchange(
    tmp_path, monkeypatch
):
    ingress, _root, bus, telegram, request = await _open_camera(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    original = _owner_message("Я съела только часть", 8101)
    ingress.process_real_inbound(original)
    first = await _turn(pool, original, ingress)
    assert "Сколько" in first.text
    assert len(honcho.messages) == 2
    await ingress.close()

    reopened = CameraIngress(ingress.config, workspace=tmp_path, bus=bus, telegram=telegram)
    replay = _owner_message("Я съела только часть", 8101)
    reopened.process_real_inbound(replay)
    assert replay.metadata["_camera_duplicate_clarification_replay"] is True
    pool._camera_ingress = reopened
    updates = [u async for u in pool.stream_message(replay, reopened.config.session_key)]
    attempt = reopened._attempts[request["candidate_id"]]
    assert updates == []
    assert attempt["state"] == "clarifying"
    assert "camera_commit" not in attempt
    assert len(honcho.messages) == 2
    await reopened.close()


@pytest.mark.asyncio
async def test_crash_after_question_append_reconciles_as_clarification(tmp_path, monkeypatch):
    ingress, _root, bus, telegram, request = await _open_camera(tmp_path, index=812)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    complete = ingress.complete

    def crash_before_state_transition(message, **kwargs):
        if kwargs.get("clarification"):
            raise RuntimeError("synthetic crash after durable clarification append")
        complete(message, **kwargs)

    ingress.complete = crash_before_state_transition
    original = _owner_message("Я съела только часть", 8201)
    ingress.process_real_inbound(original)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        await _turn(pool, original, ingress)
    assert len(honcho.messages) == 2
    assert "decision_trace" not in honcho.messages[1].metadata
    await ingress.close()

    reopened = CameraIngress(ingress.config, workspace=tmp_path, bus=bus, telegram=telegram)
    reopened.mark_restart_unknown()
    pool._camera_ingress = reopened
    replay = _owner_message("Я съела только часть", 8201)
    reopened.process_real_inbound(replay)
    assert replay.metadata["_camera_reconcile_only"] is True
    updates = [u async for u in pool.stream_message(replay, reopened.config.session_key)]
    attempt = reopened._attempts[request["candidate_id"]]
    assert len(updates) == 1 and "Сколько" in updates[0].text
    assert attempt["state"] == "clarifying"
    assert "camera_commit" not in attempt
    assert len(honcho.messages) == 2
    await reopened.close()


@pytest.mark.asyncio
async def test_clarification_denial_and_unrelated_text_never_become_consumed(
    tmp_path, monkeypatch
):
    ingress, _root, bus, _telegram, request = await _open_camera(tmp_path, index=813)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    partial = _owner_message("Я съела только часть", 8301)
    ingress.process_real_inbound(partial)
    await _turn(pool, partial, ingress)

    for index, text in enumerate(
        ("Спасибо", "Как завтра будет погода?", "Я съела новый завтрак, не тот на фото"),
        start=8302,
    ):
        unrelated = _owner_message(text, index)
        ingress.process_real_inbound(unrelated)
        assert unrelated.metadata.get("_camera_answer") is None

    denial = _owner_message("Я не ела", 8305)
    ingress.process_real_inbound(denial)
    assert denial.metadata.get("_camera_answer") == "no"
    assert denial.metadata.get("_camera_authority") is CAMERA_AUTHORITY
    final = await _turn(pool, denial, ingress)
    assert "nutrition_append_event_id" not in final.metadata
    assert "camera_commit" not in ingress._attempts[request["candidate_id"]]
    await ingress.close()


@pytest.mark.asyncio
async def test_clarification_ttl_releases_attention_but_keeps_quantity_binding(
    tmp_path, monkeypatch
):
    ingress, root, bus, _telegram, request = await _open_camera(tmp_path, index=814)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    partial = _owner_message("Я съела только часть", 8401)
    ingress.process_real_inbound(partial)
    await _turn(pool, partial, ingress)
    attempt = ingress._attempts[request["candidate_id"]]
    attempt["admitted_at"] = (datetime.now(timezone.utc) - timedelta(minutes=115)).isoformat()
    ingress._sweep_expired_attempts()
    assert attempt["attention_active"] is False

    followup = _owner_message("2 кусочка", 8402)
    ingress.process_real_inbound(followup)
    assert followup.metadata.get("_camera_answer") == "yes"
    result = await _turn(pool, followup, ingress)
    assert result.metadata.get("nutrition_append_event_id")
    assert attempt["camera_commit"]["event_id"] == result.metadata["nutrition_append_event_id"]
    assert len(honcho.messages) == 4
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["photo_sent", "clarifying"])
async def test_generic_food_identification_is_context_not_consumption(
    tmp_path, monkeypatch, state
):
    ingress, _root, bus, telegram, request = await _open_camera(
        tmp_path, index=816 if state == "photo_sent" else 817
    )
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    if state == "clarifying":
        partial = _owner_message("Я съела только часть", 8161)
        ingress.process_real_inbound(partial)
        await _turn(pool, partial, ingress)
    attempt = ingress._attempts[request["candidate_id"]]
    attempt["admitted_at"] = (datetime.now(timezone.utc) - timedelta(minutes=115)).isoformat()
    ingress._sweep_expired_attempts()

    hint = _owner_message("только фасоль", 8162)
    ingress.process_real_inbound(hint)
    assert hint.metadata.get("_camera_context_hint") is CAMERA_AUTHORITY
    # Keep the original available to semantic model inference without treating
    # a noun-only food identification as a consumption confirmation.
    assert hint.metadata.get("_camera_answer") is None
    assert attempt["snapshot"] in hint.media
    camera_prompt = OhmoSessionRuntimePool._with_camera_turn_context(
        "base", hint, ingress.trusted_capture_time_for_answer(hint)
    )
    assert "does not confirm eating" in camera_prompt
    assert "owner's confirmed-consumption turn" not in camera_prompt
    pool._test_bundle.engine = _ScriptedEngine(pool, [(None, "Если речь о фасоли, уточните количество.")])
    result = await _turn(pool, hint, ingress)
    assert "количество" in result.text.casefold()
    assert "nutrition_append_event_id" not in result.metadata
    assert "camera_commit" not in attempt
    await pool._shadow_backend_for_scope(None).await_pending()
    assert attempt["state"] == "clarifying"
    assert len(honcho.messages) == (4 if state == "clarifying" else 2)
    assert honcho.messages[-2].content == hint.content
    assert honcho.messages[-1].metadata["camera_finalizer_outcome"] == "clarification"
    assert "decision_trace" not in honcho.messages[-1].metadata

    unrelated = _owner_message("Как завтра будет погода?", 8163)
    ingress.process_real_inbound(unrelated)
    assert unrelated.metadata.get("_camera_context_hint") is not CAMERA_AUTHORITY
    assert unrelated.metadata.get("_camera_answer") is None
    pool._test_bundle.engine = _ScriptedEngine(pool, [(None, "Прогноз погоды на завтра недоступен.")])
    unrelated_result = await _turn(pool, unrelated, ingress)
    assert "Сколько" not in unrelated_result.text
    assert "nutrition_append_event_id" not in unrelated_result.metadata
    assert "camera_commit" not in attempt
    assert attempt["state"] == "clarifying"
    for message_id, text in ((8164, "Спасибо"), (8165, "Оплати счёт")):
        off_topic = _owner_message(text, message_id)
        ingress.process_real_inbound(off_topic)
        assert off_topic.metadata.get("_camera_context_hint") is not CAMERA_AUTHORITY
        assert off_topic.metadata.get("_camera_answer") is None
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("quantity", ["полтора яблока", "3 кусочка хлеба", "немного каши"])
async def test_natural_quantity_after_ttl_finalizes_one_trusted_meal(
    tmp_path, monkeypatch, quantity
):
    ingress, _root, bus, _telegram, request = await _open_camera(
        tmp_path, index=818
    )
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    partial = _owner_message("Я съела только часть", 8181)
    ingress.process_real_inbound(partial)
    await _turn(pool, partial, ingress)
    attempt = ingress._attempts[request["candidate_id"]]
    attempt["admitted_at"] = (datetime.now(timezone.utc) - timedelta(minutes=115)).isoformat()
    ingress._sweep_expired_attempts()

    trace = _consumed_trace()
    trace["annotations"]["nutrition"]["basis"] = ["image", "owner_statement"]
    trace["annotations"]["nutrition"]["items"] = [
        {"name": "oats", "quantity_text": quantity}
    ]
    pool._test_bundle.engine = _ScriptedEngine(pool, [(trace, "Приём учтён." )])
    answer = _owner_message(quantity, 8182)
    ingress.process_real_inbound(answer)
    assert answer.metadata.get("_camera_answer") == "yes"
    result = await _turn(pool, answer, ingress)
    assert result.metadata["nutrition_append_event_id"] == attempt["camera_commit"]["event_id"]
    assert len(honcho.messages) == 4
    saved = honcho.messages[-1].metadata["decision_trace"]["annotations"]["nutrition"]
    assert saved["items"][0]["quantity_text"] == quantity
    await ingress.close()


@pytest.mark.asyncio
async def test_followup_can_retrieve_original_camera_image_and_source_denial_fails_closed(
    tmp_path, monkeypatch
):
    output = BytesIO()
    Image.new("RGB", (12, 9), color=(30, 120, 45)).save(output, format="JPEG")
    original_bytes = output.getvalue()
    ingress, root, bus, _telegram = _ingress(tmp_path, FakeTelegram())
    request = _candidate(root, index=815, image_bytes=original_bytes)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    await bus.consume_inbound()
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    pool._attachment_store = AttachmentStore(pool._workspace)
    monkeypatch.setattr("ohmo.gateway.runtime._build_inbound_user_message", _build_inbound_user_message)
    pool._register_conversation_image_tool = OhmoSessionRuntimePool._register_conversation_image_tool.__get__(pool)
    pool._test_bundle.tool_registry = ToolRegistry()

    partial = _owner_message("Я съела только часть", 8151)
    ingress.process_real_inbound(partial)
    await _turn(pool, partial, ingress)
    attempt = ingress._attempts[request["candidate_id"]]
    followup = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content="whole plate",
        metadata={
            "message_id": 8152,
            "reply_to_message_id": attempt["reply_ids"][-1],
            "_telegram_raw_text": "whole plate",
            "is_group": False,
        },
    )
    ingress.process_real_inbound(followup)
    assert followup.metadata.get("_camera_answer") == "yes"
    assert attempt["snapshot"] in followup.media
    current = _build_inbound_user_message(
        followup, pool._attachment_store, session_key=ingress.config.session_key
    )
    ref = next(block for block in current.content if isinstance(block, AttachmentRefBlock))
    pool._register_conversation_image_tool(pool._test_bundle, current_message=current)
    tool = pool._test_bundle.tool_registry.get("load_conversation_image")
    assert isinstance(tool, LoadConversationImageTool)
    loaded = await tool.execute(
        LoadConversationImageInput(attachment_id=ref.attachment_id),
        ToolExecutionContext(cwd=pool._workspace),
    )
    assert loaded.is_error is False
    assert base64.b64decode(loaded.metadata["_openharness_transient_image"].data) == original_bytes

    pool._register_conversation_image_tool(pool._test_bundle)
    historical = await tool.execute(
        LoadConversationImageInput(attachment_id=ref.attachment_id),
        ToolExecutionContext(cwd=pool._workspace),
    )
    assert historical.is_error is False
    denied_tool = LoadConversationImageTool(
        pool._attachment_store, is_attachment_allowed=lambda _attachment_id: False
    )
    denied = await denied_tool.execute(
        LoadConversationImageInput(attachment_id=ref.attachment_id),
        ToolExecutionContext(cwd=pool._workspace),
    )
    assert denied.is_error is True
    assert denied.output == "Conversation image unavailable."
    assert ref.attachment_id not in denied.output
    assert attempt.get("camera_commit") is None
    await ingress.close()


def test_arbitrary_media_provenance_metadata_cannot_override_inbound_identity(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = AttachmentStore(initialize_workspace(workspace))
    image_path = tmp_path / "manual.png"
    image_path.write_bytes(PNG_BYTES)
    timestamp = datetime(2026, 10, 1, 8, 15, tzinfo=timezone.utc)
    message = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="Synthetic attached photo",
        timestamp=timestamp,
        media=[str(image_path)],
        metadata={
            "message_id": 8601,
            "_coalesced_media_sources": [{
                "source_message_id": "forged-source",
                "received_at": "2000-01-01T00:00:00+00:00",
            }],
            "_coalesced_media_provenance_authority": "forged",
        },
    )
    built = _build_inbound_user_message(message, store, session_key="telegram:123")
    ref = next(block for block in built.content if isinstance(block, AttachmentRefBlock))
    assert ref.source_provenance["source_message_id"] == "8601"
    assert ref.source_provenance["received_at"] == timestamp.isoformat()


def test_camera_prompt_retrieves_original_image_or_preserves_quantity_uncertainty():
    prompt = OhmoSessionRuntimePool._with_camera_answer_context(
        "synthetic prompt", datetime(2026, 10, 1, 8, 30, tzinfo=timezone.utc)
    )
    assert "load_conversation_image" in prompt
    assert "original is unavailable" in prompt
    assert "visible image is only a crop" in prompt
    assert "preserve the quantity uncertainty" in prompt
    assert "never treat a partial crop as a confirmed whole plate" in prompt
    assert "Use the current user's stated food and quantity over ambiguous image inference." in prompt
    for rule in (
        "A whole-portion confirmation does not turn a count you guessed in an option "
        "into an owner-stated quantity.",
        "Count whole products, not pieces cut from one.",
        "use the served cooked weight and matching preparation and unit",
        "make the total equal the item sum",
        "identified by readable package labeling",
        "A readable label may identify hidden package contents",
        "do not include adjacent unselected packages",
        "do not include adjacent unselected packages or unseen oil or sauce",
    ):
        assert rule in prompt

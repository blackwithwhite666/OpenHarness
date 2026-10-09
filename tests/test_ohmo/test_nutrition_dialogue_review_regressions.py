"""Promoted synthetic acceptance probes for nutrition dialogue repairs."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone

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
from ohmo.gateway.camera import CAMERA_AUTHORITY
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
        self.turns = []
        self.load_attachment_ids = []
        self.select_attachment_ids = set()
        self.loaded_results = []
        self.system_prompt = ""

    def set_decision_trace_recorder(self, recorder):
        self.decision_trace_recorder = recorder

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    async def submit_message(self, user_message):
        self.messages.append(user_message)
        self.turns.append(self.pool._active_message.content)
        payload, answer = self.steps.pop(0)
        self.decision_trace_recorder.trace_requirement_signals(
            self.pool._active_message.content
        )
        for attachment_id in self.load_attachment_ids:
            tool = self.pool._test_bundle.tool_registry.get("load_conversation_image")
            assert tool is not None
            result = await tool.execute(
                LoadConversationImageInput(
                    attachment_id=attachment_id,
                    select_as_nutrition_source=attachment_id in self.select_attachment_ids,
                ),
                ToolExecutionContext(cwd=self.pool._workspace),
            )
            self.loaded_results.append(result)
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
    assert result.text.endswith("Записано; баланс обновляется.")
    assert answer in honcho.messages[1].content
    assert result.metadata["nutrition_actual_append_receipt"]["event_id"] == honcho.messages[1].id
    await ingress.close()


@pytest.mark.asyncio
async def test_storage_status_preserves_model_answer_and_adds_receipt_status(
    tmp_path, monkeypatch
):
    ingress, _root, _bus, _telegram = _ingress(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    answer = (
        "Две порции риса, около 120 ккал (100–140). 2+2=4."
    )
    rice_trace = _consumed_trace()
    rice_trace["annotations"]["nutrition"]["items"] = [
        {"name": "rice", "quantity_text": "2 servings"}
    ]
    pool._test_bundle.engine = _ScriptedEngine(pool, [(rice_trace, answer)])
    result = await _turn(pool, _owner_message("Запиши съеденный рис и посчитай 2+2", 8505), ingress)
    assert result.metadata["nutrition_append_event_id"] == honcho.messages[1].id
    assert "120" in result.text and "100–140" in result.text and "2+2=4" in result.text
    assert result.text.endswith("Записано; баланс обновляется.")
    assert result.text.startswith(honcho.messages[1].content)
    await ingress.close()


@pytest.mark.asyncio
async def test_retry_keeps_authoritative_annotation_without_new_exchange(tmp_path, monkeypatch):
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
    replay_updates = [u async for u in pool.stream_message(retry, ingress.config.session_key)]
    assert len(honcho.messages) == 2
    assert not any(update.kind == "final" for update in replay_updates)
    assert first.metadata["nutrition_append_event_id"] == honcho.messages[1].id
    assert len(pool._test_bundle.engine.turns) == 1
    assert honcho.messages[1].metadata["decision_trace"]["annotations"]["nutrition"]["items"][0]["name"] == "pears"
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
    replay_updates = [u async for u in pool.stream_message(replay, ingress.config.session_key)]
    assert not any(update.kind == "final" for update in replay_updates)
    assert len(honcho.messages) == 4
    assert len(pool._test_bundle.engine.turns) == 2
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
        "2+2=4."
    )
    rice_trace = _consumed_trace()
    rice_trace["annotations"]["nutrition"]["items"] = [
        {"name": "rice", "quantity_text": "2 pieces"}
    ]
    engine = _ScriptedEngine(pool, [(rice_trace, answer)])
    engine.read_wellness = True
    pool._test_bundle.engine = engine
    result = await _turn(pool, _owner_message("Запиши съеденный рис и посчитай 2+2", 8504), ingress)
    assert manager.calls == 1
    assert result.metadata["nutrition_append_event_id"] == "honcho-2"
    for value in (result.text, honcho.messages[1].content):
        assert "Рис" in value and "два кусочка" in value
        assert "120" in value and "100–140" in value and "2+2=4" in value
    assert result.text == honcho.messages[1].content + "\nЗаписано; баланс обновляется."
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
    replay_updates = [u async for u in pool.stream_message(retry, ingress.config.session_key)]
    actual = honcho.messages[1].metadata["decision_trace"]["annotations"]["nutrition"]
    assert actual["consumption_status"] == "unknown"
    assert len(honcho.messages) == 2
    assert not any(update.kind == "final" for update in replay_updates)
    await ingress.close()


async def _open_camera(tmp_path, *, index: int = 811, capture_time=None):
    ingress, root, bus, telegram = _ingress(tmp_path, FakeTelegram())
    request = _candidate(root, index=index, capture_time=capture_time)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    ingress._test_photo_event = await bus.consume_inbound()
    return ingress, root, bus, telegram, request


def _install_delivered_photo(pool, ingress):
    """Keep the delivered assistant-origin photo in ordinary conversation history."""
    pool._attachment_store = AttachmentStore(pool._workspace)
    pool._test_bundle.tool_registry = ToolRegistry()
    del pool._register_conversation_image_tool
    photo = _build_inbound_user_message(
        ingress._test_photo_event,
        pool._attachment_store,
        session_key=ingress.config.session_key,
    )
    for block in photo.content:
        if isinstance(block, AttachmentRefBlock):
            block.source_provenance["gateway_session_id"] = pool._test_bundle.session_id
    pool._test_bundle.engine.messages.append(photo)
    return next(block for block in photo.content if isinstance(block, AttachmentRefBlock))


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "Я съела только часть", "Я не ела", "только фасоль", "2 кусочка",
    "Спасибо", "Как завтра будет погода?",
])
async def test_owner_text_is_not_camera_interpreted(tmp_path, monkeypatch, text):
    ingress, _root, _bus, _telegram, request = await _open_camera(tmp_path)
    try:
        honcho = _Honcho()
        pool = _pool(tmp_path, ingress, honcho, monkeypatch)
        answer = _owner_message(text, 8101)
        ingress.process_real_inbound(answer)
        assert answer.media == []
        assert not any(key.startswith("_camera_") for key in answer.metadata)
        pool._test_bundle.engine = _ScriptedEngine(pool, [(None, "Обычный ответ модели.")])
        final = await _turn(pool, answer, ingress)
        await pool._shadow_backend_for_scope(None).await_pending()
        assert final.text == "Обычный ответ модели."
        assert "nutrition_append_event_id" not in final.metadata
        assert len(honcho.messages) == 2
        assert ingress._attempts[request["candidate_id"]]["state"] == "photo_sent"
        assert "camera_commit" not in ingress._attempts[request["candidate_id"]]
    finally:
        await ingress.close()


@pytest.mark.asyncio
async def test_selected_delivered_photo_uses_capture_time_and_receipt(tmp_path, monkeypatch):
    capture_time = datetime.fromisoformat("2026-10-04T08:49:10+03:00")
    ingress, _root, _bus, _telegram, request = await _open_camera(
        tmp_path, index=810, capture_time=capture_time
    )
    try:
        honcho = _Honcho()
        pool = _pool(tmp_path, ingress, honcho, monkeypatch)
        nutrition = _consumed_trace()["annotations"]["nutrition"]
        nutrition.update(
            basis=["image", "owner_statement"], energy_kcal_best=215,
            items=[{"name": "oatmeal", "quantity_text": "1 bowl"}],
        )
        engine = _ScriptedEngine(pool, [(_trace(nutrition), "Записала порцию: 215 ккал.")])
        pool._test_bundle.engine = engine
        ref = _install_delivered_photo(pool, ingress)
        engine.load_attachment_ids = [ref.attachment_id]
        engine.select_attachment_ids = {ref.attachment_id}
        answer = _owner_message("На фото моя овсянка; я съела одну миску", 8102)
        answer.timestamp = datetime.fromisoformat("2026-10-05T07:49:54+00:00")
        ingress.process_real_inbound(answer)
        final = await _turn(pool, answer, ingress)
        assert len(engine.loaded_results) == 1
        assert engine.loaded_results[0].is_error is False
        assert "explicitly selected nutrition source" in engine.loaded_results[0].output
        assert final.text.endswith("Записано; баланс обновляется.")
        event_id = final.metadata["nutrition_append_event_id"]
        assert event_id == honcho.messages[-1].id
        assert len(honcho.messages) == 2
        saved = honcho.messages[-1].metadata
        assert saved["photo_occurrence_source"]["camera_candidate_id"] == request["candidate_id"]
        assert saved["photo_occurrence_source"]["source_message_id"] == str(
            ingress._attempts[request["candidate_id"]]["photo_id"]
        )
        assert datetime.fromisoformat(saved["decision_trace"]["annotations"]["nutrition"]["meal_at"]) == capture_time
        assert saved["decision_trace"]["annotations"]["nutrition"]["items"][0]["quantity_text"] == "1 bowl"
        assert ingress._attempts[request["candidate_id"]]["attention_active"] is False
        assert "camera_commit" not in ingress._attempts[request["candidate_id"]]
    finally:
        await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("quantity", ["полтора яблока", "3 кусочка хлеба", "немного каши"])
async def test_late_plain_portion_can_be_selected_without_camera_dialogue(
    tmp_path, monkeypatch, quantity
):
    ingress, _root, _bus, _telegram, request = await _open_camera(tmp_path, index=818)
    try:
        honcho = _Honcho()
        pool = _pool(tmp_path, ingress, honcho, monkeypatch)
        trace = _consumed_trace()
        trace["annotations"]["nutrition"].update(
            basis=["image", "owner_statement"],
            items=[{"name": "oats", "quantity_text": quantity}],
        )
        engine = _ScriptedEngine(pool, [(None, "Сколько вы съели?"), (trace, "Приём учтён.")])
        pool._test_bundle.engine = engine
        ref = _install_delivered_photo(pool, ingress)
        partial = _owner_message("Я съела только часть", 8181)
        await _turn(pool, partial, ingress)
        await pool._shadow_backend_for_scope(None).await_pending()
        attempt = ingress._attempts[request["candidate_id"]]
        attempt["admitted_at"] = (datetime.now(timezone.utc) - timedelta(minutes=115)).isoformat()
        ingress._sweep_expired_attempts()
        assert attempt["attention_active"] is False
        engine.load_attachment_ids = [ref.attachment_id]
        engine.select_attachment_ids = {ref.attachment_id}
        answer = _owner_message(quantity, 8182)
        ingress.process_real_inbound(answer)
        assert not any(key.startswith("_camera_") for key in answer.metadata)
        result = await _turn(pool, answer, ingress)
        assert result.metadata["nutrition_append_event_id"] == honcho.messages[-1].id
        assert len(honcho.messages) == 4
        saved = honcho.messages[-1].metadata
        assert saved["photo_occurrence_source"]["source_message_id"] == str(attempt["photo_id"])
        assert saved["decision_trace"]["annotations"]["nutrition"]["items"][0]["quantity_text"] == quantity
    finally:
        await ingress.close()


@pytest.mark.asyncio
async def test_historical_image_is_readable_but_denied_source_stays_unavailable(
    tmp_path, monkeypatch
):
    ingress, _root, _bus, _telegram, request = await _open_camera(tmp_path, index=815)
    try:
        pool = _pool(tmp_path, ingress, _Honcho(), monkeypatch)
        ref = _install_delivered_photo(pool, ingress)
        pool._register_conversation_image_tool(pool._test_bundle)
        tool = pool._test_bundle.tool_registry.get("load_conversation_image")
        assert isinstance(tool, LoadConversationImageTool)
        loaded = await tool.execute(
            LoadConversationImageInput(attachment_id=ref.attachment_id),
            ToolExecutionContext(cwd=pool._workspace),
        )
        assert loaded.is_error is False
        assert base64.b64decode(loaded.metadata["_openharness_transient_image"].data)
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
        assert "camera_commit" not in ingress._attempts[request["candidate_id"]]
    finally:
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


def test_photo_context_is_factual_timestamp_without_dialogue_instructions():
    prompt = OhmoSessionRuntimePool._with_user_photo_context(
        "synthetic prompt", datetime(2026, 10, 1, 8, 30, tzinfo=timezone.utc)
    )
    assert "2026-10-01T08:30:00+00:00" in prompt
    assert "source provenance" in prompt
    assert "the conversation determines its meaning" in prompt
    assert "confirm eating" not in prompt

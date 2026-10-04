"""Real Camera ingress, runtime stream, and durable append regressions."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from openharness.channels.bus.events import InboundMessage, OutboundMessage
from openharness.evals import TRACE_FINALIZATION
from openharness.engine.stream_events import AssistantTextDelta

from ohmo.gateway.camera import CAMERA_AUTHORITY
from ohmo.gateway.bridge import OhmoGatewayBridge
from ohmo.gateway.memory_gate import MemoryScope
from ohmo.gateway.models import GatewayConfig
from ohmo.gateway.runtime import OhmoSessionRuntimePool, _build_inbound_user_message
from ohmo.memory_backend import ShadowMemoryBackend
from ohmo.memory_service.honcho_client import Message
from ohmo.workspace import initialize_workspace
from ohmo.attachment_store import AttachmentStore
from tests.test_ohmo.test_conversation_attachments import PNG_BYTES

from tests.test_ohmo.test_camera_ingress import FakeTelegram, _admit, _candidate, _ingress


class _BaseMemory:
    def __init__(self, root: Path) -> None:
        self._memory_dir = root

    async def append_turn(self, role: str, text: str) -> None:
        del role, text


class _Honcho:
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.fail_before_append = False
        self.timeout_after_append = False
        self.wrong_assistant_operation = False

    async def find_messages_by_client_op_id(self, session: str, operation: str):
        del session
        return [m for m in self.messages if m.metadata.get("client_op_id") == operation]

    async def create_messages(self, session: str, values: list[dict]):
        if self.fail_before_append:
            raise OSError("synthetic append unavailable")
        result = []
        for value in values:
            metadata = dict(value["metadata"])
            if self.wrong_assistant_operation and metadata.get("role") == "assistant":
                metadata["client_op_id"] = "synthetic-wrong-operation"
            message = Message(
                id=f"honcho-{len(self.messages) + 1}",
                content=value["content"],
                peer_id=value["peer_id"],
                session_id=session,
                metadata=metadata,
                created_at=datetime.now(timezone.utc),
                workspace_id="fixture",
                token_count=1,
            )
            self.messages.append(message)
            result.append(message)
        if self.timeout_after_append:
            raise TimeoutError("synthetic response lost after authoritative append")
        return result


class _Engine:
    def __init__(self) -> None:
        self.decision_trace_recorder = None
        self.tool_metadata: dict = {}
        self.messages: list = []
        self.turns: list[tuple[str, list[str], datetime]] = []
        self.pool = None

    def set_decision_trace_recorder(self, recorder) -> None:
        self.decision_trace_recorder = recorder

    def set_system_prompt(self, prompt: str) -> None:
        del prompt

    async def submit_message(self, _user_message):
        self.messages.append(_user_message)
        message = self.pool._active_message
        self.turns.append((message.content, list(message.media), message.timestamp))
        recorder = self.decision_trace_recorder
        recorder.trace_requirement_signals(message.content)
        lowered = message.content.strip().casefold()
        camera_quantity_known = not (
            message.metadata.get("_camera_answer") == "yes"
            and ("только часть" in lowered or "only part" in lowered or "только груши" in lowered)
        )
        if (
            lowered == "2 кусочка"
            or "запиши" in lowered
            or (message.metadata.get("_camera_answer") == "yes" and camera_quantity_known)
            or ("молоко" in lowered and "половина" in lowered and "съела" in lowered)
        ):
            payload = _consumed_payload()
            payload["trace_event_id"] = f"synthetic-meal-{message.metadata.get('message_id')}"
            if message.metadata.get("_camera_answer") == "yes":
                nutrition = payload["annotations"]["nutrition"]
                nutrition["energy_kcal_best"] = 105
                nutrition["items"] = [{
                    "name": "Мягкий творог Синтетик 5%, упаковка 125 г",
                    "quantity_text": "1 pack (125 g)",
                    "energy_kcal_best": 105,
                }]
            if "3 груши" in lowered:
                payload["annotations"]["nutrition"]["items"] = [
                    {"name": "pears", "quantity_text": "3 pears"}
                ]
            if lowered != "2 кусочка" and message.metadata.get("_camera_answer") != "yes":
                payload["annotations"]["nutrition"]["basis"] = ["owner_statement"]
            if "молоко" in lowered:
                payload["annotations"]["nutrition"]["basis"] = ["image", "owner_statement"]
                payload["annotations"]["nutrition"]["items"] = [
                    {"name": "milk", "quantity_text": "half portion"}
                ]
            recorder.record(TRACE_FINALIZATION, payload)
            text = (
                "Пачка: примерно 105 ккал (состав точно неясен). "
                "В журнале творог пока не появился — **сохранение не подтверждено**. "
                "Отдельно: в журнале ужин пока не появился. "
                "В журнале вчерашний творог пока не появился. "
                "В журнале мягкий сыр пока не появился. "
                "В журнале творог пока не появился, а ужин тоже пока отсутствует. "
                "Сохранение витаминов при готовке не подтверждено. "
                "Запись вчерашнего ужина не подтверждена. "
                "Про вчерашний ужин: сохранение не подтверждено. "
                "Речь о витаминах после нагрева: сохранение не подтверждено. "
                "**сохранение не подтверждено**."
                if message.metadata.get("_camera_answer") == "yes"
                else "Запись не удалось сохранить." if lowered != "2 кусочка" else "Спасибо."
            )
        else:
            text = "Сколько примерно вы съели?"
        yield AssistantTextDelta(text=text)


def _consumed_payload() -> dict:
    return {
        "schema_version": 1,
        "trace_event_id": "synthetic-meal-finalization",
        "annotations": {
            "nutrition": {
                "schema_version": 2,
                "record_type": "meal_observation",
                "consumption_status": "consumed",
                "basis": ["image", "owner_statement"],
                "energy_kcal_best": 120,
                "items": [{"name": "pears", "quantity_text": "2 pieces"}],
            }
        },
    }


def _pool(tmp_path: Path, ingress, honcho: _Honcho, monkeypatch) -> OhmoSessionRuntimePool:
    workspace = tmp_path / "workspace"
    initialize_workspace(workspace)
    cfg = GatewayConfig(
        enabled_channels=["telegram"],
        conversation_learning=True,
        evals_capture=True,
        memory_backend="shadow",
        honcho_base_url="https://honcho.fixture.invalid",
        family_principals={"123": "marina"},
        enabled_memory_tenants=("marina",),
        tenant_honcho={"marina": {"workspace": "fixture", "api_key": "fixture", "observed_peer": "owner"}},
        camera_ingress=ingress.config,
    )
    pool = object.__new__(OhmoSessionRuntimePool)
    pool._gateway_config = cfg
    pool._workspace = workspace
    pool._cwd = workspace
    pool._attachment_store = None
    pool._session_backend = SimpleNamespace()
    pool._bundles = {}
    pool._camera_ingress = ingress
    pool._session_owner_principals = {"camera-session": "123"}
    pool._cwd_for_message = lambda *_: workspace
    pool._bind_session_owner = lambda *_: None
    pool._resolve_turn_memory_scope = lambda *_: MemoryScope("marina", ())
    pool._configure_turn_memory_surfaces = lambda *_args, **_kwargs: None
    pool._maybe_schedule_memory_judge = lambda *_args, **_kwargs: None
    pool._todo_cleanup_update = lambda **_kwargs: None

    async def pass_todo_guard(**kwargs):
        kwargs["state"]["reply"] = "".join(kwargs["reply_parts"])
        if False:
            yield None

    pool._guard_todo_final = pass_todo_guard
    backend = ShadowMemoryBackend(
        _BaseMemory(workspace), honcho, conversation_learning=True
    )
    pool._shadow_backend_for_scope = lambda _scope: backend
    engine = _Engine()
    engine.pool = pool
    bundle = SimpleNamespace(
        session_id="camera-session",
        engine=engine,
        commands=SimpleNamespace(lookup=lambda _text: None),
        tool_registry=None,
        cwd=str(workspace),
    )
    pool._test_bundle = bundle

    async def get_bundle(*_args, **_kwargs):
        return bundle

    async def save_snapshot(*_args, **_kwargs):
        return None

    async def runtime_prompt(*_args, **_kwargs):
        return "synthetic system prompt"

    pool.get_bundle = get_bundle
    pool._save_snapshot = save_snapshot
    pool._runtime_system_prompt = runtime_prompt
    pool._register_conversation_image_tool = lambda *_args, **_kwargs: None
    pool._set_group_request_context = lambda *_args: None
    pool._restore_group_request_context = lambda *_args: None
    pool._clear_reminder_context = lambda *_args: None
    monkeypatch.setattr(
        "ohmo.gateway.runtime._build_inbound_user_message",
        lambda message, *_args, **_kwargs: SimpleNamespace(text=message.content),
    )
    return pool


async def _turn(pool, message, ingress):
    pool._active_message = message
    updates = [
        update
        async for update in pool.stream_message(message, ingress.config.session_key)
    ]
    final = next(update for update in updates if update.kind == "final")
    outbound = OutboundMessage(
        channel="telegram",
        chat_id="123",
        content=final.text,
        metadata=final.metadata,
    )
    from openharness.channels.bus.events import OutboundDeliveryReceipt

    candidate_id = message.metadata.get("_camera_candidate_id")
    if isinstance(candidate_id, str):
        native_id = 500 + len(ingress._attempts[candidate_id]["reply_ids"])
        ingress.note_assistant_receipt(
            outbound,
            OutboundDeliveryReceipt(
                channel="telegram", chat_id="123", native_message_ids=(native_id,)
            ),
        )
    return final


@pytest.mark.asyncio
async def test_real_stream_clarification_then_quantity_commits_once(tmp_path, monkeypatch):
    ingress, root, bus, _ = _ingress(tmp_path, FakeTelegram())
    request = _candidate(root, index=70)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    await bus.consume_inbound()
    candidate_id = request["candidate_id"]
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)

    partial = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content="Я съела только часть",
        metadata={"message_id": 701, "_telegram_raw_text": "Я съела только часть", "_synthetic": True, "is_group": False},
    )
    ingress.process_real_inbound(partial)
    assert partial.metadata["_camera_answer"] == "yes"
    first = await _turn(pool, partial, ingress)
    assert "Сколько" in first.text
    assert "camera_commit" not in ingress._attempts[candidate_id]
    assert len(honcho.messages) == 2

    followup = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content="2 кусочка",
        metadata={"message_id": 702, "_telegram_raw_text": "2 кусочка", "_synthetic": True, "is_group": False},
    )
    ingress.process_real_inbound(followup)
    assert followup.metadata.get("_camera_answer") == "yes"
    assert followup.metadata.get("_camera_candidate_id") == candidate_id
    assert followup.metadata.get("_camera_turn_id") != partial.metadata.get("_camera_turn_id")
    second = await _turn(pool, followup, ingress)
    assert second.metadata["nutrition_sync_status"] == "pending"
    assert second.metadata["nutrition_append_event_id"] == "honcho-4"
    assert len(honcho.messages) == 4
    assert ingress._attempts[candidate_id]["camera_commit"]["event_id"] == "honcho-4"

    replay = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content="2 кусочка",
        metadata={"message_id": 702, "_telegram_raw_text": "2 кусочка", "_synthetic": True, "is_group": False},
    )
    ingress.process_real_inbound(replay)
    assert replay.metadata.get("_camera_answer") is None
    assert len(honcho.messages) == 4
    await ingress.close()


@pytest.mark.asyncio
async def test_camera_yes_for_known_single_pack_commits_once_without_quantity_turn(
    tmp_path, monkeypatch
):
    ingress, root, bus, _ = _ingress(tmp_path, FakeTelegram())
    request = _candidate(root, index=79)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    await bus.consume_inbound()
    photo_id = ingress._attempts[request["candidate_id"]]["photo_id"]
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)

    answer = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Да, я это съела",
        metadata={"message_id": 791, "reply_to_message_id": photo_id,
                  "_telegram_raw_text": "Да, я это съела", "is_group": False,
                  "_synthetic": True},
    )
    ingress.process_real_inbound(answer)
    assert answer.metadata["_camera_answer"] == "yes"
    result = await _turn(pool, answer, ingress)

    assert result.metadata["nutrition_append_event_id"] == "honcho-2"
    assert result.text.startswith("Пачка: примерно 105 ккал")
    assert "состав точно неясен" in result.text
    assert "**сохранение не подтверждено**" not in result.text
    assert "В журнале творог пока не появился." not in result.text
    assert "ужин пока не появился" in result.text
    assert "В журнале вчерашний творог пока не появился." in result.text
    assert "В журнале мягкий сыр пока не появился." in result.text
    assert "В журнале творог пока не появился, а ужин тоже пока отсутствует." in result.text
    assert "Сохранение витаминов при готовке не подтверждено." in result.text
    assert "Запись вчерашнего ужина не подтверждена." in result.text
    assert "Про вчерашний ужин: сохранение не подтверждено." in result.text
    assert "Речь о витаминах после нагрева: сохранение не подтверждено." in result.text
    assert "**сохранение не подтверждено**." not in result.text
    assert "Записано. Баланс обновляется." in result.text
    saved = honcho.messages[1].metadata["decision_trace"]["annotations"]["nutrition"]
    assert saved["consumption_status"] == "consumed"
    assert len(saved["items"]) == 1
    assert saved["items"][0]["name"] == "Мягкий творог Синтетик 5%, упаковка 125 г"
    assert saved["items"][0]["quantity_text"] == "1 pack (125 g)"
    assert saved["items"][0]["energy_kcal_best"] == 105
    assert len(honcho.messages) == 2

    replay = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content=answer.content,
        metadata={"message_id": 791, "reply_to_message_id": photo_id,
                  "_telegram_raw_text": answer.content, "is_group": False,
                  "_synthetic": True},
    )
    ingress.process_real_inbound(replay)
    assert replay.metadata.get("_camera_answer") is None
    assert len(honcho.messages) == 2
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["append", "mismatch"])
async def test_camera_receipt_failure_never_returns_saved_status(
    tmp_path, monkeypatch, failure_mode
):
    ingress, root, bus, _ = _ingress(tmp_path, FakeTelegram())
    request = _candidate(root, index=80)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    await bus.consume_inbound()
    photo_id = ingress._attempts[request["candidate_id"]]["photo_id"]
    honcho = _Honcho()
    if failure_mode == "append":
        honcho.fail_before_append = True
    else:
        honcho.wrong_assistant_operation = True
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    answer = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Да, я это съела",
        metadata={"message_id": 801, "reply_to_message_id": photo_id,
                  "_telegram_raw_text": "Да, я это съела", "is_group": False,
                  "_synthetic": True},
    )
    ingress.process_real_inbound(answer)
    pool._active_message = answer
    updates = []
    expected = OSError if failure_mode == "append" else Exception
    with pytest.raises(expected):
        async for update in pool.stream_message(answer, ingress.config.session_key):
            updates.append(update)

    assert not any(update.kind == "final" for update in updates)
    assert all("Записано" not in update.text for update in updates)
    if failure_mode == "append":
        assert honcho.messages == []
    else:
        assert len(honcho.messages) == 2
        assert honcho.messages[1].metadata["client_op_id"] == "synthetic-wrong-operation"
    await ingress.close()


@pytest.mark.asyncio
async def test_context_answer_after_attention_timeout_keeps_capture_and_replay_identity(
    tmp_path, monkeypatch
):
    ingress, root, bus, _ = _ingress(tmp_path, FakeTelegram())
    captured = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(microsecond=0)
    request = _candidate(root, index=73, capture_time=captured)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    await bus.consume_inbound()
    candidate_id = request["candidate_id"]
    attempt = ingress._attempts[candidate_id]
    attempt["admitted_at"] = (datetime.now(timezone.utc) - timedelta(minutes=115)).isoformat()
    ingress._save_attempts()
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)

    answer = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Я съела 3 груши",
        metadata={"message_id": 731, "is_group": False, "_synthetic": True,
                  "_telegram_raw_text": "Я съела 3 груши"},
    )
    ingress.process_real_inbound(answer)
    assert answer.metadata["_camera_candidate_id"] == candidate_id
    assert answer.metadata["_camera_answer"] == "yes"
    assert ingress.trusted_capture_time_for_answer(answer) == captured
    update = await _turn(pool, answer, ingress)
    assert update.metadata["nutrition_append_event_id"] == "honcho-2"
    nutrition = honcho.messages[1].metadata["decision_trace"]["annotations"]["nutrition"]
    assert nutrition["meal_at"] == captured.isoformat()
    assert nutrition.get("meal_date") is None
    assert len(nutrition["items"]) == 1
    assert nutrition["items"][0]["name"] == "pears"
    assert ingress._attempts[candidate_id]["camera_commit"]["event_id"] == "honcho-2"

    replay = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content=answer.content,
        metadata={"message_id": 731, "is_group": False, "_synthetic": True,
                  "_telegram_raw_text": answer.content},
    )
    ingress.process_real_inbound(replay)
    assert replay.metadata.get("_camera_answer") is None
    assert len(honcho.messages) == 2

    foreign = InboundMessage(
        channel="telegram", sender_id="456", chat_id="123",
        content="Я съела 3 груши", metadata={"_telegram_raw_text": "Я съела 3 груши"},
    )
    ingress.process_real_inbound(foreign)
    assert foreign.metadata.get("_camera_authority") is None

    await ingress.close()


@pytest.mark.asyncio
async def test_late_food_identification_then_consumed_quantity_stays_on_original_photo(
    tmp_path, monkeypatch
):
    ingress, root, bus, _ = _ingress(tmp_path, FakeTelegram())
    capture = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(microsecond=0)
    request = _candidate(root, index=74, capture_time=capture)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    await bus.consume_inbound()
    attempt = ingress._attempts[request["candidate_id"]]
    attempt["admitted_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=115)
    ).isoformat()
    ingress._save_attempts()
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)

    identified = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content="только груши",
        metadata={"message_id": 7401, "_telegram_raw_text": "только груши", "is_group": False},
    )
    ingress.process_real_inbound(identified)
    assert identified.metadata.get("_camera_candidate_id") == request["candidate_id"]
    clarification = await _turn(pool, identified, ingress)
    assert "Сколько" in clarification.text
    assert attempt.get("camera_commit") is None
    await pool._shadow_backend_for_scope(None).await_pending()
    assert len(honcho.messages) == 2
    assert honcho.messages[0].content == identified.content
    assert honcho.messages[1].content == clarification.text

    quantity = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content="Я съела 3 груши",
        metadata={"message_id": 7402, "_telegram_raw_text": "Я съела 3 груши", "is_group": False},
    )
    ingress.process_real_inbound(quantity)
    assert quantity.metadata.get("_camera_candidate_id") == request["candidate_id"]
    saved = await _turn(pool, quantity, ingress)
    assert saved.metadata.get("nutrition_append_event_id") == attempt["camera_commit"]["event_id"]
    nutrition = honcho.messages[-1].metadata["decision_trace"]["annotations"]["nutrition"]
    assert nutrition["items"] == [{
        "name": "pears", "quantity_text": "3 pears",
        "energy_kcal_min": None, "energy_kcal_max": None, "energy_kcal_best": None,
    }]
    assert nutrition["meal_at"] == request["capture_time"]
    await pool._shadow_backend_for_scope(None).await_pending()
    assert len(honcho.messages) == 4
    await ingress.close()


@pytest.mark.asyncio
async def test_camera_context_ambiguity_and_foreign_owner_are_unbound(tmp_path):
    ingress, root, bus, _ = _ingress(tmp_path, FakeTelegram())
    request = _candidate(root, index=75)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    await bus.consume_inbound()
    retained = ingress._attempts[request["candidate_id"]]
    ingress._attempts["synthetic-overlap"] = {
        **retained,
        "state": "photo_sent",
        "camera_commit": None,
        "answer_turn_id": None,
        "final_turn_id": None,
    }
    ambiguous = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Я съела груши", metadata={"_telegram_raw_text": "Я съела груши"},
    )
    ingress.process_real_inbound(ambiguous)
    assert ambiguous.metadata.get("_camera_answer") is None
    assert ambiguous.metadata.get("_camera_unbound") is CAMERA_AUTHORITY
    foreign = InboundMessage(
        channel="telegram", sender_id="456", chat_id="123",
        content="Я съела груши", metadata={"_telegram_raw_text": "Я съела груши"},
    )
    ingress.process_real_inbound(foreign)
    assert foreign.metadata.get("_camera_authority") is None
    await ingress.close()


@pytest.mark.asyncio
async def test_ordinary_meal_append_is_receipt_bound_and_replay_safe(tmp_path, monkeypatch):
    ingress, _root, _bus, _ = _ingress(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    first = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Я съела два кусочка, запиши завтрак",
        metadata={"message_id": 801, "is_group": False, "_synthetic": True},
        timestamp=datetime(2026, 9, 28, 8, 30, tzinfo=timezone.utc),
    )
    result = await _turn(pool, first, ingress)
    assert result.text == honcho.messages[1].content
    assert "Записано; приём пищи пока не привязан к дате." in result.text
    assert result.metadata["nutrition_sync_status"] == "pending"
    event_id = result.metadata["nutrition_append_event_id"]
    assert event_id == "honcho-2"
    assert len(honcho.messages) == 2
    assert honcho.messages[1].metadata["decision_trace"]["annotations"]["nutrition"][
        "record_type"
    ] == "meal_observation"

    replay = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content=first.content,
        metadata={"message_id": 801, "is_group": False, "_synthetic": True},
        timestamp=first.timestamp,
    )
    replay_result = await _turn(pool, replay, ingress)
    assert replay_result.metadata["nutrition_append_event_id"] == event_id
    assert len(honcho.messages) == 2
    assert honcho.messages[1].metadata["received_at"] == first.timestamp.isoformat()

    next_day = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Я съела два кусочка, запиши завтрак",
        metadata={"message_id": 802, "is_group": False, "_synthetic": True},
        timestamp=datetime(2026, 9, 29, 8, 30, tzinfo=timezone.utc),
    )
    next_result = await _turn(pool, next_day, ingress)
    assert next_result.metadata["nutrition_append_event_id"] == "honcho-4"
    assert next_result.metadata["nutrition_append_event_id"] != event_id
    assert len(honcho.messages) == 4
    assert honcho.messages[3].metadata["received_at"] == next_day.timestamp.isoformat()

    await ingress.close()


@pytest.mark.asyncio
async def test_ordinary_meal_append_reconciles_timeout_and_never_claims_unknown(
    tmp_path, monkeypatch
):
    ingress, _root, _bus, _ = _ingress(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    honcho.timeout_after_append = True
    message = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Я съела два кусочка, запиши завтрак",
        metadata={"message_id": 811, "is_group": False, "_synthetic": True},
    )
    result = await _turn(pool, message, ingress)
    assert result.text == honcho.messages[1].content
    assert "Записано; приём пищи пока не привязан к дате." in result.text
    assert result.metadata["nutrition_append_event_id"] == "honcho-2"
    assert len(honcho.messages) == 2

    honcho2 = _Honcho()
    honcho2.fail_before_append = True
    pool2 = _pool(tmp_path / "failed", ingress, honcho2, monkeypatch)
    failed_message = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Я съела два кусочка, запиши завтрак",
        metadata={"message_id": 812, "is_group": False, "_synthetic": True},
    )
    pool2._active_message = failed_message
    failed_updates = []
    with pytest.raises(OSError, match="synthetic append unavailable"):
        async for update in pool2.stream_message(failed_message, ingress.config.session_key):
            failed_updates.append(update)
    assert not any(update.kind == "final" for update in failed_updates)
    assert all("Записано" not in update.text for update in failed_updates)
    assert honcho2.messages == []

    honcho3 = _Honcho()
    honcho3.wrong_assistant_operation = True
    pool3 = _pool(tmp_path / "mismatched", ingress, honcho3, monkeypatch)
    mismatched_message = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Я съела два кусочка, запиши завтрак",
        metadata={"message_id": 813, "is_group": False, "_synthetic": True},
    )
    pool3._active_message = mismatched_message
    mismatched_updates = []
    with pytest.raises(Exception, match="(?i)(receipt|reconcil|operation)"):
        async for update in pool3.stream_message(mismatched_message, ingress.config.session_key):
            mismatched_updates.append(update)
    assert not any(update.kind == "final" for update in mismatched_updates)
    assert all("Записано" not in update.text for update in mismatched_updates)
    assert len(honcho3.messages) == 2
    assert honcho3.messages[1].metadata["client_op_id"] == "synthetic-wrong-operation"
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("annotation", "status"),
    [
        ({
            "schema_version": 2,
            "record_type": "meal_observation",
            "consumption_status": "unknown",
            "basis": ["image"],
            "energy_kcal_best": 100,
        }, "recorded"),
        ({"schema_version": 2, "record_type": "day_summary"}, "recorded"),
        (None, "invalid"),
    ],
)
async def test_camera_clarification_rejects_nonconsumed_summary_and_invalid_trace(
    tmp_path, monkeypatch, annotation, status
):
    ingress, root, bus, _ = _ingress(tmp_path)
    request = _candidate(root, index=72)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    await bus.consume_inbound()
    candidate_id = request["candidate_id"]
    message = InboundMessage(
        channel="telegram", sender_id="123", chat_id="123",
        content="Я съела только часть",
        metadata={"message_id": 721, "_telegram_raw_text": "Я съела только часть", "is_group": False, "_synthetic": True},
    )
    ingress.process_real_inbound(message)
    pool = _pool(tmp_path, ingress, _Honcho(), monkeypatch)
    recorder = SimpleNamespace(
        validated_nutrition_envelope=annotation,
        nutrition_annotation_status=status,
    )
    from ohmo.gateway.turn_context import TurnContext

    ctx = TurnContext(
        principal="123", is_owner=True, is_private=True, channel="telegram",
        chat_id="123", session_id="camera-session", camera_authorized=True,
    )
    with pytest.raises(ValueError):
        await pool._append_conversation_turn(
            turn_ctx=ctx,
            memory_scope=MemoryScope("marina", ()),
            message=message,
            recorder=recorder,
            user_text=message.content,
            assistant_text="must not append",
        )
    assert ingress._attempts[candidate_id].get("camera_commit") is None
    await ingress.close()


@pytest.mark.asyncio
async def test_bridge_coalesces_photo_details_portion_and_consumption_into_one_trace(
    tmp_path, monkeypatch
):
    ingress, _root, bus, _ = _ingress(tmp_path, FakeTelegram())
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    pool._attachment_store = AttachmentStore(pool._workspace)
    built_messages = []
    def capture_builder(*args, **kwargs):
        built = _build_inbound_user_message(*args, **kwargs)
        built_messages.append(built)
        return built
    monkeypatch.setattr("ohmo.gateway.runtime._build_inbound_user_message", capture_builder)
    original_stream = pool.stream_message
    first_running = asyncio.Event()

    async def stream_with_active_message(message, session_key):
        if message.content == "slow synthetic turn":
            yield SimpleNamespace(
                kind="progress", text="synthetic progress",
                metadata={"_session_key": session_key, "_progress": True},
            )
            first_running.set()
            await asyncio.Future()
        pool._active_message = message
        async for update in original_stream(message, session_key):
            yield update

    pool.stream_message = stream_with_active_message
    bridge = OhmoGatewayBridge(
        bus=bus,
        runtime_pool=pool,
        workspace=pool._workspace,
        message_coalesce_window=0.08,
        message_coalesce_media_window=0.3,
        message_coalesce_max=10,
        camera_ingress=ingress,
    )
    bridge_task = asyncio.create_task(bridge.run())
    first_stamp = datetime(2026, 9, 30, 23, 59, 58, tzinfo=timezone.utc)
    second_stamp = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)
    final_stamp = datetime(2026, 10, 1, 0, 0, 2, tzinfo=timezone.utc)
    first_photo = tmp_path / "photo-before-midnight.png"
    second_photo = tmp_path / "photo-after-midnight.png"
    first_photo.write_bytes(PNG_BYTES)
    second_photo.write_bytes(PNG_BYTES)
    messages = [
        InboundMessage(
            channel="telegram", sender_id="123", chat_id="123", content="Фото молока",
            timestamp=first_stamp, media=[str(first_photo)],
            metadata={"message_id": 901, "is_group": False},
        ),
        InboundMessage(
            channel="telegram", sender_id="123", chat_id="123", content="Добавила молоко",
            timestamp=first_stamp, metadata={"message_id": 902, "is_group": False},
        ),
        InboundMessage(
            channel="telegram", sender_id="123", chat_id="123", content="Ещё фото порции",
            timestamp=second_stamp, media=[str(second_photo)],
            metadata={"message_id": 903, "is_group": False},
        ),
        InboundMessage(
            channel="telegram", sender_id="123", chat_id="123", content="половина порции",
            timestamp=second_stamp, metadata={"message_id": 904, "is_group": False},
        ),
        InboundMessage(
            channel="telegram", sender_id="123", chat_id="123", content="я это съела",
            timestamp=final_stamp, metadata={"message_id": 905, "is_group": False},
        ),
    ]
    try:
        await bus.publish_inbound(InboundMessage(
            channel="telegram", sender_id="123", chat_id="123",
            content="slow synthetic turn", timestamp=first_stamp,
            metadata={"message_id": 900, "is_group": False},
        ))
        await asyncio.wait_for(bus.consume_outbound(), timeout=3)
        await asyncio.wait_for(first_running.wait(), timeout=3)
        for message in messages:
            ingress.process_real_inbound(message)
            await bus.publish_inbound(message)
        final = None
        while final is None:
            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=3)
            if outbound.metadata.get("nutrition_append_event_id"):
                final = outbound
    finally:
        bridge.stop()
        bridge_task.cancel()
        await asyncio.gather(bridge_task, return_exceptions=True)
    assert final is not None
    assert len(honcho.messages) == 2
    observed_text, observed_media, observed_timestamp = pool._active_message.content, pool._active_message.media, pool._active_message.timestamp
    assert observed_text == "Фото молока\n\nДобавила молоко\n\nЕщё фото порции\n\nполовина порции\n\nя это съела"
    assert observed_media == [str(first_photo), str(second_photo)]
    assert observed_timestamp == final_stamp
    nutrition = honcho.messages[1].metadata["decision_trace"]["annotations"]["nutrition"]
    assert nutrition["items"] == [{
        "name": "milk",
        "quantity_text": "half portion",
        "energy_kcal_min": None,
        "energy_kcal_max": None,
        "energy_kcal_best": None,
    }]
    assert final.metadata["nutrition_sync_status"] == "pending"
    assert final.metadata["nutrition_append_event_id"] == "honcho-2"
    refs = [
        block for block in built_messages[-1].content
        if getattr(block, "type", None) == "attachment_ref"
    ]
    assert [ref.source_provenance["source_message_id"] for ref in refs] == ["901", "903"]
    assert [ref.source_provenance["received_at"] for ref in refs] == [
        first_stamp.isoformat(), second_stamp.isoformat()
    ]
    assert pool._test_bundle.engine.turns == [
        (observed_text, observed_media, final_stamp)
    ]
    await ingress.close()

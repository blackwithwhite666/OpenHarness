"""Synthetic end-to-end regressions for photo receipt and correction context."""
from __future__ import annotations

import copy
import json
import os
from io import BytesIO
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image
from openharness.api.usage import UsageSnapshot
from openharness.channels.bus.events import InboundMessage
from openharness.engine.messages import AttachmentRefBlock, ConversationMessage
from openharness.engine.stream_events import AssistantTextDelta
from openharness.evals import TRACE_FINALIZATION
from openharness.tools.base import ToolExecutionContext, ToolRegistry
from ohmo.conversation_image_tool import LoadConversationImageInput
from ohmo.evals import GatewayEvalRecorder
from ohmo.gateway.memory_gate import MemoryScope
from ohmo.gateway.models import GatewayConfig
import ohmo.gateway.runtime as runtime_module
from ohmo.gateway.runtime import (
    OhmoSessionRuntimePool, _build_inbound_user_message,
)
from ohmo.gateway.turn_context import build_turn_context
from ohmo.memory_backend import ConversationReconciliationError, ShadowMemoryBackend
from ohmo.memory_service.honcho_client import HonchoClient
from ohmo.session_storage import save_session_snapshot, load_latest
from ohmo.attachment_store import AttachmentStore
from ohmo.workspace import initialize_workspace
from tests.test_ohmo.test_conversation_attachments import PNG_BYTES
from tests.test_ohmo.test_nutrition_dialogue_stream import _BaseMemory

ROOT = Path(__file__).resolve().parent
BASE = datetime(2026, 10, 1, 20, 59, 58, tzinfo=timezone.utc)
SCOPE = MemoryScope("review-tenant", ())


def observation(**extra):
    return {"schema_version": 2, "record_type": "meal_observation",
            "consumption_status": "consumed", "basis": ["image"],
            "energy_kcal_best": 750, **extra}


def correction(day="2026-10-02"):
    return {"schema_version": 2, "record_type": "meal_correction",
            "changed_fields": ["meal_date"], "meal_date": day}


def original_photo_time_correction(meal_at):
    return {"schema_version": 2, "record_type": "meal_correction",
            "changed_fields": ["meal_at"], "meal_at": meal_at.isoformat()}


class MockHoncho:
    def __init__(self, namespace="joint", wrong_operation=False,
                 assistant_metadata_patch=None, omit_assistant=False):
        self.rows = []
        self.calls = []
        self.namespace = namespace
        self.wrong_operation = wrong_operation
        self.assistant_metadata_patch = assistant_metadata_patch or {}
        self.omit_assistant = omit_assistant

    def transport(self, request):
        assert request.url.host == "honcho.review.invalid"
        self.calls.append(str(request.url.path))
        body = json.loads(request.content)
        if request.url.path.endswith("/messages/list"):
            op = body["filters"]["metadata"]["client_op_id"]
            rows = [row for row in self.rows if row["metadata"].get("client_op_id") == op]
            return httpx.Response(200, json={"items": rows, "page": 1, "pages": 1, "total": len(rows)})
        assert request.url.path.endswith("/messages")
        created = []
        for value in body["messages"]:
            if self.omit_assistant and value.get("metadata", {}).get("role") == "assistant":
                continue
            row = {**value, "id": f"event-{self.namespace}-{len(self.rows)+1}", "session_id": "review-session",
                   "workspace_id": "review-workspace", "token_count": 1,
                   "created_at": (BASE+timedelta(seconds=len(self.rows))).isoformat()}
            if self.wrong_operation and row["metadata"]["role"] == "assistant":
                row["metadata"]["client_op_id"] = "wrong-operation:assistant"
            if row["metadata"].get("role") == "assistant":
                row["metadata"].update(self.assistant_metadata_patch)
            self.rows.append(row)
            created.append(row)
        return httpx.Response(200, json=created)


class ScriptEngine:
    def __init__(self):
        self.messages = []
        self.tool_metadata = {}
        self.total_usage = UsageSnapshot()
        self.decision_trace_recorder = None
        self.system_prompt = ""
        self.script = []
        self.annotation = None
        self.answer = "Бургер примерно 700–800 ккал."
        self.captured_tools = []
        self.observed_prompts = []
        self.last_load_output = None

    def set_system_prompt(self, value):
        self.system_prompt = value

    def set_decision_trace_recorder(self, value):
        self.decision_trace_recorder = value

    async def submit_message(self, user_message):
        self.observed_prompts.append(self.system_prompt)
        if isinstance(user_message, ConversationMessage):
            self.messages.append(user_message)
        for attachment_id in self.script:
            tool = self.bundle.tool_registry.get("load_conversation_image")
            self.captured_tools.append(tool)
            result = await tool.execute(LoadConversationImageInput(attachment_id=attachment_id),
                                        ToolExecutionContext(cwd=self.pool._workspace))
            self.last_load_failed = result.is_error
            self.last_load_output = result.output
        payload = {"schema_version": 1, "trace_event_id": f"review-trace-{len(self.messages)}",
                   "annotations": {"nutrition": self.annotation} if self.annotation else {}}
        self.decision_trace_recorder.record(TRACE_FINALIZATION, payload)
        yield AssistantTextDelta(text=self.answer)


def setup(name, *, wrong_operation=False, assistant_metadata_patch=None, omit_assistant=False):
    workspace = ROOT / name
    initialize_workspace(workspace)
    server = MockHoncho(namespace=name, wrong_operation=wrong_operation,
                        assistant_metadata_patch=assistant_metadata_patch,
                        omit_assistant=omit_assistant)
    client = HonchoClient("https://honcho.review.invalid", "synthetic-token", "review-workspace",
                          transport=httpx.MockTransport(server.transport))
    backend = ShadowMemoryBackend(_BaseMemory(workspace), client, conversation_learning=True,
                                  session="review-session")
    pool = object.__new__(OhmoSessionRuntimePool)
    pool._gateway_config = GatewayConfig(conversation_learning=True,
        family_principals={"123": "review-tenant"}, enabled_memory_tenants=("review-tenant",))
    pool._workspace = workspace
    pool._attachment_store = AttachmentStore(workspace)
    pool._session_owner_principals = {"gateway-session": "123"}
    pool._shadow_backend_for_scope = lambda scope: backend
    pool._maybe_schedule_memory_judge = lambda *args, **kwargs: None
    pool._set_group_request_context = lambda *args: None
    pool._restore_group_request_context = lambda *args: None
    pool._clear_reminder_context = lambda *args: None
    async def base_prompt(*args, **kwargs):
        prompt_path = Path(__file__).resolve().parents[2] / "src/openharness/skills/bundled/content/calory.md"
        return prompt_path.read_text(encoding="utf-8")
    async def save_snapshot(*args, **kwargs):
        return None
    pool._runtime_system_prompt = base_prompt
    pool._save_snapshot = save_snapshot
    engine = ScriptEngine()
    engine.pool = pool
    bundle = SimpleNamespace(engine=engine, tool_registry=ToolRegistry(), session_id="gateway-session", review_backend=backend)
    engine.bundle = bundle
    return pool, bundle, server, client


def inbound(pool, native_id, content="", *, media=None, when=BASE, metadata_extra=None):
    metadata = {"message_id": native_id, "is_group": False}
    metadata.update(metadata_extra or {})
    message = InboundMessage(channel="telegram", sender_id="123", chat_id="123", content=content,
        timestamp=when, media=media or [], metadata=metadata)
    ctx = build_turn_context(message, session_id="gateway-session")
    user = _build_inbound_user_message(message, pool._attachment_store, session_key="telegram:123")
    for block in user.content:
        if isinstance(block, AttachmentRefBlock):
            block.source_provenance["gateway_session_id"] = "gateway-session"
    return message, ctx, user


async def turn(pool, bundle, message, ctx, user, annotation, *, loads=(), answer=None):
    recorder = GatewayEvalRecorder.start(workspace=pool._workspace, bundle=bundle, message=message,
        session_key="telegram:123", user_text=message.content, user_goal=message.content)
    repeat = pool._known_user_photo_repeat(message, bundle.engine.messages)
    default = pool._trusted_user_photo_time(message, turn_ctx=ctx, history=bundle.engine.messages)
    if default:
        recorder.set_authoritative_nutrition_meal_at(default, preserve_explicit=True)
    bundle.engine.script = list(loads)
    bundle.engine.annotation = copy.deepcopy(annotation)
    if answer is not None:
        bundle.engine.answer = answer
    updates = [update async for update in pool._stream_engine_message(bundle=bundle, message=message,
        session_key="telegram:123", user_prompt=user.text, user_message=user, turn_ctx=ctx,
        memory_scope=SCOPE, recorder=recorder, todo_lifecycle=False,
        user_photo_meal_at=default, user_photo_repeat=repeat)]
    await bundle.review_backend.await_pending()
    return next(update for update in updates if update.kind == "final"), recorder


async def runtime():
    out = {}
    handoff = {}
    photo = ROOT / "synthetic.png"
    photo.write_bytes(PNG_BYTES)

    pool, bundle, server, client = setup("direct-correction")
    msg, ctx, user = inbound(pool, "photo-1", media=[str(photo)])
    original, _ = await turn(
        pool, bundle, msg, ctx, user, observation(),
        answer="Бургер примерно 700–800 ккал. Я не знаю, когда ты его съел; "
        "подтверждения сохранения записи тоже нет.",
    )
    ref = next(block for block in user.content if isinstance(block, AttachmentRefBlock))
    # Restore through the existing real snapshot serializer/loader.
    save_session_snapshot(cwd=pool._workspace, workspace=pool._workspace, model="offline",
        system_prompt="BASE", messages=bundle.engine.messages, usage=UsageSnapshot(),
        session_id=bundle.session_id, session_key="telegram:123")
    bundle.engine.messages = [ConversationMessage.model_validate(row)
                              for row in load_latest(pool._workspace)["messages"]]
    msg2, ctx2, user2 = inbound(pool, "date-correction", "Это было 2 октября", when=BASE+timedelta(days=1))
    final, _ = await turn(
        pool, bundle, msg2, ctx2, user2, correction(), loads=[ref.attachment_id],
        answer="Бургер примерно 700–800 ккал. Точное время в запись не подставляю; "
        "проверка журнала пока не подтвердила, что дата обновилась.",
    )
    out["direct_correction"] = {"observation_status": original.text, "correction_status": final.text,
                              "correction_delivery": final.metadata}
    assert "700–800 ккал" in original.text
    assert original.metadata["nutrition_committed_annotation"]["energy_kcal_best"] == 750
    assert "не знаю" not in original.text
    assert original.text == server.rows[1]["content"]
    assert final.text == "Изменение сохранено; баланс обновляется."
    assert "подтверждения сохранения записи тоже нет" not in original.text
    assert "пока не подтвердила" not in final.text
    assert "Изменение сохранено; баланс обновляется." in final.text
    assert "2026-10-02" not in final.text and "2 октября" not in final.text
    assert final.metadata["nutrition_sync_status"] == "pending"
    assert "2026-10-01T20:59:58" in bundle.engine.observed_prompts[0]
    prompt = bundle.engine.observed_prompts[1]
    assert "When correcting or deleting a prior meal from a historical photo" in prompt
    assert "there is no native reply" in prompt
    assert "always use `load_conversation_image`" in prompt
    assert "including for a date-only correction" in prompt
    assert "«съел когда написал»" in bundle.engine.observed_prompts[1]
    committed_event = final.metadata["nutrition_append_event_id"]
    before_retry = len(server.rows)
    retry, _ = await turn(
        pool, bundle, msg2, ctx2, user2, correction("2026-10-03"),
        loads=[ref.attachment_id], answer="Бургер примерно 900 ккал; дату меняю на 3 октября.",
    )
    assert len(server.rows) == before_retry
    assert retry.metadata["nutrition_append_event_id"] == committed_event
    assert "900 ккал" not in retry.text and "3 октября" not in retry.text
    assert retry.text == "Изменение сохранено; баланс обновляется."
    # Retained tool from a unique successful load must be inert after teardown.
    late = bundle.engine.captured_tools[-1]
    msg2.metadata.pop("_selected_source_binding", None)
    await late.execute(LoadConversationImageInput(attachment_id=ref.attachment_id), ToolExecutionContext(cwd=pool._workspace))
    out["late_unique_callback"] = {"binding_after_teardown": "_selected_source_binding" in msg2.metadata}
    assert not out["late_unique_callback"]["binding_after_teardown"]
    # Already-targeted native reply without an image load.
    native, native_ctx, native_user = inbound(pool, "native-correction", "Это было 3 октября")
    native.metadata["reply_to_message_id"] = "photo-1"
    before = len(server.rows)
    native_final, _ = await turn(pool, bundle, native, native_ctx, native_user, correction("2026-10-03"))
    out["native_reply_without_load"] = {"new_rows": len(server.rows)-before, "response": native_final.text}
    assert out["native_reply_without_load"]["new_rows"] == 2
    assert server.rows[-1]["metadata"]["reply_to_source_message_id"] == "photo-1"
    assert native_final.metadata["nutrition_sync_status"] == "pending"
    handoff["direct"] = copy.deepcopy(server.rows)

    # One callback image came from the first message in a burst while the
    # committed nutrition append belongs to the later coalesced turn.
    burst_photo_at = datetime(2026, 10, 1, 23, 59, 58, tzinfo=timezone.utc)
    burst_append_at = datetime(2026, 10, 2, 0, 0, 3, tzinfo=timezone.utc)
    coalesced_photo = ROOT / "coalesced.png"
    image_bytes = BytesIO()
    Image.new("RGB", (2, 2), color=(13, 27, 41)).save(image_bytes, format="PNG")
    coalesced_photo.write_bytes(image_bytes.getvalue())
    burst, burst_ctx, burst_user = inbound(
        pool, "burst-append", "Съел бургер", media=[str(coalesced_photo)], when=burst_append_at,
        metadata_extra={
            "_coalesced_media_provenance_authority": runtime_module.COALESCED_ATTACHMENT_PROVENANCE_AUTHORITY,
            "_coalesced_media_sources": [{
                "source_message_id": "burst-original-photo",
                "received_at": burst_photo_at.isoformat(),
            }],
        },
    )
    burst_final, _ = await turn(
        pool, bundle, burst, burst_ctx, burst_user,
        observation(meal_at=burst_photo_at.isoformat()),
    )
    burst_ref = next(block for block in burst_user.content if isinstance(block, AttachmentRefBlock))
    assert burst_ref.attachment_id != ref.attachment_id
    burst_correction, burst_correction_ctx, burst_correction_user = inbound(
        pool, "burst-date-correction", "Это было 1 октября",
        when=burst_append_at + timedelta(minutes=1),
    )
    burst_fixed, _ = await turn(
        pool, bundle, burst_correction, burst_correction_ctx, burst_correction_user,
        correction("2026-10-01"), loads=[burst_ref.attachment_id],
    )
    assert burst_final.metadata["nutrition_sync_status"] == "pending"
    assert burst_fixed.metadata["nutrition_sync_status"] == "pending"
    assert server.rows[-1]["metadata"]["selected_source"]["source_message_id"] == "burst-original-photo"
    assert server.rows[-1]["metadata"]["selected_source"]["append_source_message_id"] == "burst-append"
    assert burst_fixed.text == "Изменение сохранено; баланс обновляется."
    handoff["coalesced"] = copy.deepcopy(server.rows)
    await client.aclose()

    # Reproduce the reported legacy shape through real current gateway append:
    # the original photo ref has a trusted native timestamp, while its existing
    # 750-kcal observation is undated. The contextual phrase is not a native
    # reply; loading the exact image must bind that original event.
    pool, bundle, server, client = setup("original-photo-time-correction")
    original_at = datetime(2026, 10, 1, 23, 57, 12, tzinfo=timezone.utc)
    legacy_photo = ROOT / "legacy-undated-photo.png"
    legacy_image = BytesIO()
    Image.new("RGB", (3, 2), color=(91, 37, 11)).save(legacy_image, format="PNG")
    legacy_photo.write_bytes(legacy_image.getvalue())
    legacy, legacy_ctx, legacy_user = inbound(
        pool, "legacy-original-photo", "Что на фото?", media=[str(legacy_photo)], when=original_at,
    )
    original_final, _ = await turn(
        pool, bundle, legacy, legacy_ctx, legacy_user, observation(),
        answer="Бургер примерно 700–800 ккал.",
    )
    original_event = server.rows[-1]
    original_nutrition = original_event["metadata"]["decision_trace"]["annotations"]["nutrition"]
    assert original_nutrition.get("meal_at") is None
    assert original_event["metadata"]["received_at"] == original_at.isoformat()
    legacy_ref = next(block for block in legacy_user.content if isinstance(block, AttachmentRefBlock))
    date_only, date_ctx, date_user = inbound(
        pool, "original-photo-date-correction", "Это было 2 октября",
        when=original_at + timedelta(days=1),
    )
    date_final, _ = await turn(
        pool, bundle, date_only, date_ctx, date_user, correction("2026-10-02"),
        loads=[legacy_ref.attachment_id],
        answer="Дата сохранена; точное время не указываю.",
    )
    assert date_final.text == "Изменение сохранено; баланс обновляется."
    date_event = server.rows[-1]
    assert date_event["metadata"]["selected_source"]["source_message_id"] == "legacy-original-photo"
    assert isinstance(date_event["metadata"].get("target_meal_id"), str)
    phrase, phrase_ctx, phrase_user = inbound(
        pool, "original-photo-time-correction", "Съел когда написал",
        when=original_at + timedelta(days=2),
    )
    phrase_final, _ = await turn(
        pool, bundle, phrase, phrase_ctx, phrase_user,
        original_photo_time_correction(original_at), loads=[legacy_ref.attachment_id],
        answer="Бургер примерно 700–800 ккал. Точное время в запись не подставляю; "
        "проверка журнала пока не подтвердила, что дата обновилась.",
    )
    assert phrase_final.metadata["nutrition_sync_status"] == "pending"
    assert phrase_final.text == "Изменение сохранено; баланс обновляется."
    assert "Verified original Telegram photo send time (UTC): 2026-10-01T23:57:12+00:00" in bundle.engine.last_load_output
    phrase_event = server.rows[-1]
    assert phrase_event["metadata"]["target_meal_id"] == date_event["metadata"]["target_meal_id"]
    assert phrase_event["metadata"]["selected_source"]["source_message_id"] == "legacy-original-photo"
    assert phrase_event["metadata"]["selected_source"]["append_source_message_id"] == "legacy-original-photo"
    assert isinstance(phrase_event["metadata"].get("target_meal_id"), str)
    phrase_event_id = phrase_final.metadata["nutrition_append_event_id"]
    before_phrase_retry = len(server.rows)
    phrase_retry, _ = await turn(
        pool, bundle, phrase, phrase_ctx, phrase_user,
        original_photo_time_correction(original_at + timedelta(hours=4)),
        loads=[legacy_ref.attachment_id],
        answer="Бургер примерно 900 ккал; время исправлено на полночь.",
    )
    assert len(server.rows) == before_phrase_retry
    assert phrase_retry.metadata["nutrition_append_event_id"] == phrase_event_id
    assert phrase_retry.metadata["nutrition_sync_status"] == "pending"
    assert phrase_retry.text == "Изменение сохранено; баланс обновляется."
    assert "900 ккал" not in phrase_retry.text
    handoff["legacy_photo_time"] = copy.deepcopy(server.rows)
    await client.aclose()

    pool, bundle, server, client = setup("portion-completion")
    msg, ctx, user = inbound(pool, "uncertain-photo", media=[str(photo)])
    await turn(pool, bundle, msg, ctx, user, None, answer="Сколько примерно вы съели?")
    ref = next(block for block in user.content if isinstance(block, AttachmentRefBlock))
    msg2, ctx2, user2 = inbound(
        pool, "portion-answer", "Я съел половину 2 октября около 22:15",
        when=BASE+timedelta(days=1),
    )
    await turn(
        pool, bundle, msg2, ctx2, user2,
        observation(meal_at="2026-10-02T22:15:00+00:00"),
        loads=[ref.attachment_id],
    )
    portion_receipt = server.rows[-1]
    provenance = next(
        block.source_provenance
        for history_message in bundle.engine.messages
        for block in history_message.content
        if isinstance(block, AttachmentRefBlock) and block.attachment_id == ref.attachment_id
    )
    assert provenance["consumed_occurrences"][0]["append_source_message_id"] == "portion-answer"
    save_session_snapshot(cwd=pool._workspace, workspace=pool._workspace, model="offline",
        system_prompt="BASE", messages=bundle.engine.messages, usage=UsageSnapshot(),
        session_id=bundle.session_id, session_key="telegram:123")
    bundle.engine.messages = [ConversationMessage.model_validate(row)
                              for row in load_latest(pool._workspace)["messages"]]
    msg3, ctx3, user3 = inbound(pool, "portion-date-correction", "Это было 3 октября", when=BASE+timedelta(days=2))
    final, _ = await turn(
        pool, bundle, msg3, ctx3, user3, correction("2026-10-03"), loads=[ref.attachment_id]
    )
    out["portion_then_correction"] = {"response": final.text, "delivery": final.metadata}
    handoff["portion"] = copy.deepcopy(server.rows)
    assert final.metadata["nutrition_sync_status"] == "pending"
    assert server.rows[-1]["metadata"]["selected_source"]["append_source_message_id"] == "portion-answer"
    assert server.rows[-1]["metadata"]["target_meal_id"]
    assert portion_receipt["metadata"]["source_message_id"] == "portion-answer"
    await client.aclose()

    pool, bundle, server, client = setup("failed-load")
    msg, ctx, user = inbound(pool, "prior-photo", media=[str(photo)])
    await turn(pool, bundle, msg, ctx, user, None)
    ref = next(block for block in user.content if isinstance(block, AttachmentRefBlock))
    follow, follow_ctx, follow_user = inbound(pool, "ambiguous-portion", "Я съел это")
    final, recorder = await turn(pool, bundle, follow, follow_ctx, follow_user, observation(),
                                 loads=[ref.attachment_id, "0"*64])
    out["failed_second_load"] = {"binding_remains": "_selected_source_binding" in follow.metadata,
        "stored_meal_at": recorder.validated_nutrition_envelope.get("meal_at"), "response": final.text}
    assert out["failed_second_load"]["binding_remains"] is False
    assert out["failed_second_load"]["stored_meal_at"] is None
    # Invoke the retained first tool after inference teardown, as a stale callback.
    late = bundle.engine.captured_tools[0]
    await late.execute(LoadConversationImageInput(attachment_id=ref.attachment_id), ToolExecutionContext(cwd=pool._workspace))
    out["late_callback"] = {"binding_after_teardown": "_selected_source_binding" in follow.metadata}
    assert not out["late_callback"]["binding_after_teardown"]
    await client.aclose()

    pool, bundle, server, client = setup("wrong-receipt", wrong_operation=True)
    msg, ctx, user = inbound(pool, "prior-photo", media=[str(photo)])
    await turn(pool, bundle, msg, ctx, user, None)
    ref = next(block for block in user.content if isinstance(block, AttachmentRefBlock))
    msg2, ctx2, user2 = inbound(pool, "wrong-operation-correction", "Это было 2 октября")
    try:
        final, _ = await turn(pool, bundle, msg2, ctx2, user2, correction(), loads=[ref.attachment_id])
    except ConversationReconciliationError:
        out["wrong_operation_receipt"] = {"rejected": True,
            "stored_operation": server.rows[-1]["metadata"]["client_op_id"]}
    else:
        out["wrong_operation_receipt"] = {"rejected": False, "response": final.text}
    assert out["wrong_operation_receipt"]["rejected"] is True
    await client.aclose()

    # A consumed occurrence is retained only after a receipt proves this
    # exact private turn in this gateway session. The same guard rejects
    # wrong source/operation metadata and an unavailable assistant receipt.
    invalid_receipts = {
        "wrong-session": {"gateway_session_id": "another-session"},
        "group-receipt": {"is_group": True},
        "forwarded-receipt": {"is_forwarded": True},
        "wrong-source": {"source_message_id": "another-current-source"},
        "wrong-operation": {"client_op_id": "another-current-operation"},
    }
    for name, metadata_patch in invalid_receipts.items():
        pool, bundle, server, client = setup(
            name, assistant_metadata_patch=metadata_patch
        )
        first, first_ctx, first_user = inbound(
            pool, f"{name}-photo", media=[str(photo)]
        )
        await turn(pool, bundle, first, first_ctx, first_user, None,
                   answer="Сколько примерно вы съели?")
        original_ref = next(
            block for block in first_user.content
            if isinstance(block, AttachmentRefBlock)
        )
        consumed, consumed_ctx, consumed_user = inbound(
            pool, f"{name}-portion", "Я съел половину",
            when=BASE + timedelta(minutes=5),
        )
        try:
            final, _ = await turn(
                pool, bundle, consumed, consumed_ctx, consumed_user,
                observation(meal_at=BASE.isoformat()), loads=[original_ref.attachment_id],
            )
        except ConversationReconciliationError:
            assert name == "wrong-operation"
        else:
            assert name != "wrong-operation"
            assert final.text == "Не удалось подтвердить сохранение записи."
            assert "nutrition_sync_status" not in final.metadata
            assert "nutrition_append_event_id" not in final.metadata
        assert not any(
            isinstance(block, AttachmentRefBlock)
            and block.source_provenance.get("consumed_occurrences")
            for history in bundle.engine.messages for block in history.content
        )
        await client.aclose()

    pool, bundle, server, client = setup("unavailable-receipt", omit_assistant=True)
    first, first_ctx, first_user = inbound(
        pool, "unavailable-photo", media=[str(photo)]
    )
    await turn(pool, bundle, first, first_ctx, first_user, None,
               answer="Сколько примерно вы съели?")
    original_ref = next(
        block for block in first_user.content if isinstance(block, AttachmentRefBlock)
    )
    consumed, consumed_ctx, consumed_user = inbound(
        pool, "unavailable-portion", "Я съел половину",
        when=BASE + timedelta(minutes=5),
    )
    with pytest.raises(ConversationReconciliationError):
        await turn(
            pool, bundle, consumed, consumed_ctx, consumed_user,
            observation(meal_at=BASE.isoformat()), loads=[original_ref.attachment_id],
        )
    assert not any(
        isinstance(block, AttachmentRefBlock)
        and block.source_provenance.get("consumed_occurrences")
        for history in bundle.engine.messages for block in history.content
    )
    await client.aclose()

    pool, bundle, server, client = setup("repeat-photo")
    first, first_ctx, first_user = inbound(pool, "repeat-photo-first", media=[str(photo)])
    await turn(pool, bundle, first, first_ctx, first_user, None, answer="Сколько примерно вы съели?")
    repeat, repeat_ctx, repeat_user = inbound(
        pool, "repeat-photo-status", "Что было на фото?", media=[str(photo)],
        when=BASE + timedelta(hours=1),
    )
    await turn(pool, bundle, repeat, repeat_ctx, repeat_user, None, answer="На фото бургер.")
    assert "Previously seen user photo" in bundle.engine.observed_prompts[-1]
    assert all(
        not (
            isinstance(row.get("metadata", {}).get("decision_trace"), dict)
            and row["metadata"]["decision_trace"].get("annotations", {}).get("nutrition")
        )
        for row in server.rows
    )
    new_meal, new_ctx, new_user = inbound(
        pool, "explicit-new-photo-meal", "Это отдельная новая порция, запиши её",
        media=[str(photo)], when=BASE + timedelta(hours=2),
    )
    new_annotation = observation(explicit_new_consumption=True, meal_date="2026-10-02")
    saved, _ = await turn(pool, bundle, new_meal, new_ctx, new_user, new_annotation)
    assert saved.metadata["nutrition_sync_status"] == "pending"
    assert "2026-10-01T20:59:58" not in bundle.engine.observed_prompts[-1]
    new_trace = server.rows[-1]["metadata"]["decision_trace"]["annotations"]["nutrition"]
    assert new_trace["explicit_new_consumption"] is True
    assert new_trace.get("meal_at") is None
    assert new_trace["meal_date"] == "2026-10-02"
    await client.aclose()

    handoff_path = Path(os.environ.get("CAMERA_F84_JOINT_HANDOFF", ROOT / "generated-events.json"))
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.write_text(json.dumps(handoff, ensure_ascii=False, indent=2)+"\n")
    return out, handoff


async def status_stream():
    photo = ROOT / "synthetic.png"
    photo.write_bytes(PNG_BYTES)
    pool, bundle, server, client = setup("status-stream")
    msg, ctx, user = inbound(pool, "dated-status-photo", media=[str(photo)])
    first, _ = await turn(
        pool,
        bundle,
        msg,
        ctx,
        user,
        observation(energy_kcal_min=650, energy_kcal_max=900),
        answer=(
            "На фото бургер; оценка — ≈750 ккал (примерно 650–900 ккал). "
            "Вес и состав точно не известны. Я не знаю, когда ты его съел, "
            "поэтому пока не отношу к сегодняшнему итогу; подтверждения "
            "сохранения записи тоже нет. Записано; приём пищи пока не "
            "привязан к дате."
        ),
    )
    committed_time = first.metadata["nutrition_committed_annotation"]["meal_at"]
    assert committed_time == "2026-10-01T20:59:58Z"
    assert first.text == server.rows[1]["content"]
    for committed_text in (first.text, server.rows[1]["content"]):
        assert "бургер" in committed_text.lower()
        assert "750 ккал" in committed_text
        assert "650–900 ккал" in committed_text
        assert "не знаю, когда" not in committed_text
        assert "не отношу к сегодняшнему итогу" not in committed_text
        assert "подтверждения сохранения записи" not in committed_text
        assert "приём пищи пока не привязан к дате" not in committed_text
        assert ".." not in committed_text
    assert first.text.endswith("Записано. Баланс обновляется.")
    ref = next(block for block in user.content if isinstance(block, AttachmentRefBlock))
    msg2, ctx2, user2 = inbound(pool, "dated-status-correction", "Это было 2 октября")
    second, _ = await turn(pool, bundle, msg2, ctx2, user2, correction(), loads=[ref.attachment_id], answer=
        "Бургер — примерно 750 ккал, но проверка журнала пока не подтвердила, что дата обновилась.")
    msg3, ctx3, user3 = inbound(pool, "time-caveat-correction", "Уточняю запись")
    third, _ = await turn(
        pool, bundle, msg3, ctx3, user3,
        {"schema_version": 2, "record_type": "meal_correction",
         "changed_fields": ["confidence"], "confidence": "high"},
        loads=[ref.attachment_id],
        answer="Бургер — 750 ккал; точное время в запись не подставляю.",
    )
    assert len(server.rows) == 6
    assert first.text == server.rows[1]["content"]
    assert "Записано. Баланс обновляется." in first.text
    assert second.text == "Изменение сохранено; баланс обновляется."
    assert third.text == "Изменение сохранено; баланс обновляется."
    assert second.metadata["nutrition_sync_status"] == "pending"
    assert third.metadata["nutrition_sync_status"] == "pending"
    await client.aclose()


@pytest.mark.asyncio
async def test_f84_receipt_stream_binding_prompt_and_joint_event_handoff(tmp_path):
    global ROOT
    previous_root, ROOT = ROOT, tmp_path
    try:
        out, handoff = await runtime()
        assert out["direct_correction"]["correction_delivery"]["nutrition_append_event_id"]
        assert out["portion_then_correction"]["delivery"]["nutrition_append_event_id"]
        assert handoff["direct"] and handoff["portion"]
        assert handoff["coalesced"] and handoff["legacy_photo_time"]
        await status_stream()
    finally:
        ROOT = previous_root

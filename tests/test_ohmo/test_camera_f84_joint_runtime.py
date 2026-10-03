"""Synthetic end-to-end regressions for photo receipt and correction context."""
from __future__ import annotations

import copy
import json
import os
from io import BytesIO
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

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


def assert_annotation_round_trip(value, *, sparse):
    from ohmo.evals.nutrition_trace import NutritionAnnotationV2

    parsed = NutritionAnnotationV2.model_validate(value)
    assert parsed.model_dump(mode="json", exclude_unset=sparse) == value
    return parsed


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
    final_update = next(update for update in updates if update.kind == "final")
    recorder.record_gateway_final(text=final_update.text, metadata=final_update.metadata)
    recorder.finish(status="completed")
    return final_update, recorder


def grade_contextual_receipts(pool, server, original_final, correction_final,
                              original_recorder, correction_recorder, *, source_id, meal_day, case_id,
                              additional_turns=(), canonical_revision=2,
                              canonical_meal_at=None, omit_explicit_meal_date=False):
    """Grade real recorder exports and SDK receipts with synthetic wellness boundary data."""
    from ohmo.evals.nutrition_persistence import (
        Goal, Manifest, bind_wellness_snapshot, derive_meal_id, export_eval_dialogue,
        grade_manifest, validate_dialogue_binding,
    )
    recorders = [original_recorder, *(recorder for _, recorder in additional_turns), correction_recorder]
    episode_ids = list(dict.fromkeys(recorder.episode_id for recorder in recorders))
    original_id = original_final.metadata["nutrition_append_event_id"]
    correction_id = correction_final.metadata["nutrition_append_event_id"]
    original_row = next(row for row in server.rows if row["id"] == original_id)
    correction_row = next(row for row in server.rows if row["id"] == correction_id)
    goal = Goal(
        case_id=case_id, episode_ids=episode_ids,
        owner_id="review-tenant", principal_id="telegram:123", workspace_id="review-workspace",
        eval_workspace=str(pool._workspace), peer_id="ohmo", canonical_owner_id="review-tenant",
        canonical_login="owner", session_id="review-session", gateway_session_id="gateway-session",
        source_message_id=source_id, meal_date=meal_day, meal_timezone="UTC",
        trajectory_started_at=datetime.fromisoformat(original_row["created_at"].replace("Z", "+00:00")),
        trajectory_as_of=datetime.fromisoformat(correction_row["created_at"].replace("Z", "+00:00")),
        logical_turn_id=original_row["metadata"]["logical_turn_id"],
        trace_episode_id=original_row["metadata"]["decision_trace_episode_id"],
        operation_id=original_row["metadata"]["client_op_id"],
        canonical_meal_id=derive_meal_id(tenant_id="review-tenant", source_principal="telegram:123",
            gateway_session_id="gateway-session", source_message_id=source_id),
        expected_consumed=True, expected_kcal=750, expectation_origin="reviewed_user_dialogue",
        expectation_source="synthetic-joint-runtime-review")
    manifest = Manifest(schema_version=1, goals=[goal])
    exported = export_eval_dialogue(recorders[0].store.root,
        episode_ids=episode_ids)
    binding = validate_dialogue_binding(manifest, exported)[case_id]
    assert binding["complete"] is True, binding
    rows = [row for row in server.rows
            if row["peer_id"] == goal.peer_id
            and row["session_id"] == goal.session_id
            and row["workspace_id"] == goal.workspace_id
            and goal.trajectory_started_at <= datetime.fromisoformat(
                row["created_at"].replace("Z", "+00:00")) <= goal.trajectory_as_of]
    original_annotation = original_row["metadata"]["decision_trace"]["annotations"]["nutrition"]
    original_date_value = original_annotation.get("meal_date")
    if original_date_value is None and original_annotation.get("meal_at"):
        original_date_value = datetime.fromisoformat(original_annotation["meal_at"].replace("Z", "+00:00")).date().isoformat()
    lower_day = min(date.fromisoformat(original_date_value or meal_day.isoformat()), meal_day)
    meal_tz = ZoneInfo(goal.meal_timezone)
    start = datetime.combine(lower_day, datetime.min.time(), meal_tz).isoformat()
    corrected_day_end = (datetime.combine(
        goal.meal_date + timedelta(days=1), datetime.min.time(), meal_tz
    ) - timedelta(microseconds=1))
    wellness_end = max(goal.trajectory_as_of.astimezone(meal_tz), corrected_day_end)
    queried = (wellness_end + timedelta(seconds=1)).isoformat()
    honcho = {"complete": True, "workspace_id": goal.workspace_id, "session_id": goal.session_id,
        "owner_id": goal.owner_id, "since": goal.trajectory_started_at.isoformat(),
        "until": goal.trajectory_as_of.isoformat(), "queried_at": queried, "messages": rows}
    canonical = {"meal_id": goal.canonical_meal_id, "revision": canonical_revision, "status": "active",
        "latest_event_id": correction_id, "day": meal_day.isoformat(), "provisional": True,
        "capture_time": correction_row["created_at"], "meal_at": canonical_meal_at,
        "meal_date": None if omit_explicit_meal_date else meal_day.isoformat(),
        "source_message_id": source_id, "ingest_source": "telegram", "confirmation_required": False,
        "reply_to_source_message_id": None, "received_at": None, "is_forwarded": False,
        "source_message_at": None, "is_estimate": True, "basis": ["image"],
        "consumption_status": "consumed", "energy_kcal_min": 750, "energy_kcal_max": 750,
        "energy_kcal_best": 750, "protein_g": None, "fat_g": None, "carbohydrate_g": None,
        "items": [], "confidence": "medium", "assumptions": [], "warnings": []}
    wellness = bind_wellness_snapshot({"complete": True, "user_id": goal.canonical_owner_id,
        "login": goal.canonical_login, "start": start, "end": wellness_end.isoformat(),
        "queried_at": queried, "meals": [canonical], "unassigned": []}, goal=goal)
    result = grade_manifest(manifest, honcho, wellness, now=datetime.fromisoformat(queried),
        reviewed_turn_sources=binding["reviewed_turn_sources"],
        reviewed_turn_provenance=binding["reviewed_turn_provenance"])[0]
    if correction_recorder.episode_id != correction_row["metadata"].get("decision_trace_episode_id"):
        retry_episode_id = correction_recorder.episode_id
        accepted_episode_id = correction_row["metadata"]["decision_trace_episode_id"]
        base_provenance = binding["reviewed_turn_provenance"]

        def assert_retry_proof_rejected(mutated_provenance):
            rejected = grade_manifest(manifest, honcho, wellness,
                now=goal.trajectory_as_of + timedelta(seconds=1),
                reviewed_turn_sources=binding["reviewed_turn_sources"],
                reviewed_turn_provenance=mutated_provenance)[0]
            assert rejected["a1"] == "INCONCLUSIVE" and rejected["a2"] == "NOT_RUN", rejected

        for mutation in (
            lambda turn: turn["gateway_final_metadata"].update(
                nutrition_append_event_id="different-accepted-event"),
            lambda turn: turn.update(principal_id="telegram:foreign"),
            lambda turn: turn.update(source_message_id="different-source"),
            lambda turn: turn.update(operation_id="different-operation:assistant"),
            lambda turn: turn["gateway_final_metadata"].update(
                nutrition_committed_annotation={"record_type": "meal_correction", "meal_at": "2030-01-01T00:00:00Z"}),
        ):
            mutated_provenance = copy.deepcopy(base_provenance)
            mutation(mutated_provenance[retry_episode_id][0])
            assert_retry_proof_rejected(mutated_provenance)

        for field in ("nutrition_finalization", "nutrition_committed_annotation"):
            mutated_provenance = copy.deepcopy(base_provenance)
            mutated_provenance[accepted_episode_id][0]["gateway_final_metadata"].pop(field)
            assert_retry_proof_rejected(mutated_provenance)
    return result


async def runtime():
    out = {}
    handoff = {}
    photo = ROOT / "synthetic.png"
    photo.write_bytes(PNG_BYTES)

    pool, bundle, server, client = setup("direct-correction")
    msg, ctx, user = inbound(pool, "photo-1", media=[str(photo)])
    original, original_recorder = await turn(
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
    final, correction_recorder = await turn(
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
    photo_stamp = original.metadata["nutrition_committed_annotation"]["meal_at"]
    assert photo_stamp and datetime.fromisoformat(photo_stamp.replace("Z", "+00:00")) == BASE
    assert original.metadata["nutrition_consumed_occurrence"]["receipt_event_id"]
    assert original.metadata["nutrition_consumed_occurrence"]["photo_source_message_id"] == "photo-1"
    assert original.metadata["nutrition_consumed_occurrence"]["append_source_message_id"] == "photo-1"
    assert final.metadata["nutrition_committed_annotation"]["record_type"] == "meal_correction"
    context_evidence = final.metadata["nutrition_context_evidence"]
    assert context_evidence["consumed_source_message_id"] == "photo-1"
    assert context_evidence["photo_source_message_id"] == "photo-1"
    assert context_evidence["photo_received_at"]
    assert context_evidence["original_receipt_event_id"] == original.metadata["nutrition_append_event_id"]
    assert context_evidence["current_receipt_event_id"] == final.metadata["nutrition_append_event_id"]
    from ohmo.evals.nutrition_persistence import export_eval_dialogue
    direct_export = export_eval_dialogue(original_recorder.store.root,
        episode_ids=[original_recorder.episode_id, correction_recorder.episode_id])
    exported_correction = next(item for item in direct_export["episodes"]
        if item["episode"]["episode_id"] == correction_recorder.episode_id)
    exported_turn = exported_correction["turn_provenance"][0]
    exported_original = next(item for item in direct_export["episodes"]
        if item["episode"]["episode_id"] == original_recorder.episode_id)["turn_provenance"][0]
    assert exported_turn["gateway_final_metadata"]["nutrition_append_event_id"] == final.metadata["nutrition_append_event_id"]
    assert exported_turn["gateway_final_metadata"]["nutrition_committed_annotation"] == final.metadata["nutrition_committed_annotation"]
    assert exported_turn["gateway_final_metadata"]["nutrition_context_evidence"] == context_evidence
    assert exported_turn["gateway_final_metadata"]["nutrition_finalization"]["annotation"]["record_type"] == "meal_correction"
    exported_final = exported_turn["gateway_final_metadata"]
    assert exported_final["nutrition_model_proposal_annotation"] == (
        exported_final["nutrition_finalization"]["annotation"]
    )
    assert exported_final["nutrition_proposal_matches_committed"] is (
        exported_final["nutrition_model_proposal_annotation"]
        == exported_final["nutrition_committed_annotation"]
    )
    for serialized in (
        exported_final["nutrition_model_proposal_annotation"],
        exported_final["nutrition_committed_annotation"],
        exported_final["nutrition_finalization"]["annotation"],
    ):
        parsed = assert_annotation_round_trip(serialized, sparse=True)
        assert parsed.record_type == "meal_correction"
    for serialized in (
        exported_original["gateway_final_metadata"]["nutrition_committed_annotation"],
        exported_original["gateway_final_metadata"]["nutrition_finalization"]["annotation"],
    ):
        parsed = assert_annotation_round_trip(serialized, sparse=False)
        assert parsed.record_type == "meal_observation"
    from ohmo.evals.nutrition_persistence import (
        Goal, Manifest, bind_wellness_snapshot, derive_meal_id, grade_manifest,
        validate_dialogue_binding,
    )
    original_row = next(row for row in server.rows
        if row["metadata"].get("client_op_id") == context_evidence["original_operation_id"]
        and row["metadata"].get("role") == "assistant")
    correction_row = next(row for row in server.rows
        if row["id"] == context_evidence["current_receipt_event_id"])
    goal_value = Goal(
        case_id="direct-photo-date-correction",
        episode_ids=[original_recorder.episode_id, correction_recorder.episode_id],
        owner_id="review-tenant", principal_id="telegram:123", workspace_id="review-workspace",
        eval_workspace=str(pool._workspace), peer_id="ohmo", canonical_owner_id="review-tenant",
        canonical_login="owner", session_id="review-session", gateway_session_id="gateway-session",
        source_message_id="photo-1", meal_date=date(2026, 10, 2),
        meal_timezone="UTC", trajectory_started_at=datetime(2026, 10, 1, 20, 59, 58, tzinfo=timezone.utc),
        trajectory_as_of=datetime(2026, 10, 2, 20, 59, 58, tzinfo=timezone.utc),
        logical_turn_id=original_row["metadata"]["logical_turn_id"],
        trace_episode_id=original_row["metadata"]["decision_trace_episode_id"],
        operation_id=context_evidence["original_operation_id"],
        canonical_meal_id=derive_meal_id(tenant_id="review-tenant", source_principal="telegram:123",
            gateway_session_id="gateway-session", source_message_id="photo-1"),
        expected_consumed=True, expected_kcal=750, expectation_origin="reviewed_user_dialogue",
        expectation_source="synthetic-joint-runtime-review",
    )
    manifest = Manifest(schema_version=1, goals=[goal_value])
    dialogue_binding = validate_dialogue_binding(manifest, direct_export)["direct-photo-date-correction"]
    assert dialogue_binding["complete"] is True
    scoped_rows = [row for row in server.rows
                   if row["peer_id"] == goal_value.peer_id
                   and row["session_id"] == goal_value.session_id
                   and row["workspace_id"] == goal_value.workspace_id
                   and datetime.fromisoformat("2026-10-01T00:00:00+00:00") <= datetime.fromisoformat(
                       row["created_at"].replace("Z", "+00:00"))
                   <= datetime.fromisoformat("2026-10-02T23:59:59.999999+00:00")]
    honcho_snapshot = {"complete": True, "workspace_id": "review-workspace", "session_id": "review-session",
        "owner_id": "review-tenant", "since": "2026-10-01T00:00:00+00:00",
        "until": "2026-10-02T23:59:59.999999+00:00", "queried_at": "2026-10-03T00:00:00+00:00",
        "messages": scoped_rows}
    canonical = {
        "meal_id": goal_value.canonical_meal_id, "revision": 2, "status": "active",
        "latest_event_id": correction_row["id"], "day": "2026-10-02", "provisional": True,
        "capture_time": correction_row["created_at"], "meal_at": None, "meal_date": "2026-10-02",
        "source_message_id": "photo-1", "ingest_source": "telegram", "confirmation_required": False,
        "reply_to_source_message_id": None, "received_at": None, "is_forwarded": False,
        "source_message_at": None, "is_estimate": True, "basis": ["image"],
        "consumption_status": "consumed", "energy_kcal_min": 750, "energy_kcal_max": 750,
        "energy_kcal_best": 750, "protein_g": None, "fat_g": None, "carbohydrate_g": None,
        "items": [], "confidence": "medium", "assumptions": [], "warnings": [],
    }
    wellness_snapshot_bound = bind_wellness_snapshot({"complete": True, "user_id": "review-tenant", "login": "owner",
        "start": "2026-10-01T00:00:00+00:00", "end": "2026-10-02T23:59:59.999999+00:00",
        "queried_at": "2026-10-03T00:00:00+00:00", "meals": [canonical], "unassigned": []}, goal=goal_value)
    grade = grade_manifest(manifest, honcho_snapshot, wellness_snapshot_bound,
        now=datetime(2026, 10, 3, tzinfo=timezone.utc),
        reviewed_turn_sources=dialogue_binding["reviewed_turn_sources"],
        reviewed_turn_provenance=dialogue_binding["reviewed_turn_provenance"])[0]
    assert (grade["a1"], grade["stage"], grade.get("actual_latest_event_id")) == (
        "PASS", "SAME_EVENT_PROJECTED", correction_row["id"]), grade
    assert canonical["meal_at"] is None and canonical["revision"] == 2 and canonical["day"] == "2026-10-02"

    def regrade(rows=None, provenance=None, wellness_snapshot=None, now=None):
        return grade_manifest(manifest, {**honcho_snapshot,
            "messages": copy.deepcopy(scoped_rows if rows is None else rows)},
            copy.deepcopy(wellness_snapshot_bound if wellness_snapshot is None else wellness_snapshot),
            now=now or datetime(2026, 10, 3, tzinfo=timezone.utc),
            reviewed_turn_sources=dialogue_binding["reviewed_turn_sources"],
            reviewed_turn_provenance=copy.deepcopy(
                dialogue_binding["reviewed_turn_provenance"] if provenance is None else provenance))[0]

    current_provenance = dialogue_binding["reviewed_turn_provenance"][correction_recorder.episode_id][0]
    for bad_provenance in (
        {**dialogue_binding["reviewed_turn_provenance"], correction_recorder.episode_id: [{
            **current_provenance, "gateway_final_metadata": {
                **current_provenance["gateway_final_metadata"], "nutrition_context_evidence": {
                    **context_evidence, "schema_version": 99}}}]},
        {**dialogue_binding["reviewed_turn_provenance"], correction_recorder.episode_id: [{
            **current_provenance, "gateway_final_metadata": {
                **current_provenance["gateway_final_metadata"], "nutrition_context_evidence": {
                    **context_evidence, "tenant_id": "foreign-owner"}}}]},
        {**dialogue_binding["reviewed_turn_provenance"], correction_recorder.episode_id: [{
            **current_provenance, "gateway_final_metadata": {
                key: value for key, value in current_provenance["gateway_final_metadata"].items()
                if key != "nutrition_context_evidence"}}]},
        {**dialogue_binding["reviewed_turn_provenance"], correction_recorder.episode_id: [{
            **current_provenance, "gateway_final_metadata": {
                key: value for key, value in current_provenance["gateway_final_metadata"].items()
                if key != "nutrition_finalization"}}]},
    ):
        inconclusive = regrade(provenance=bad_provenance)
        assert inconclusive["a1"] == "INCONCLUSIVE" and inconclusive["a2"] == "NOT_RUN"
    missing_original_execution = copy.deepcopy(dialogue_binding["reviewed_turn_provenance"])
    original_turn = missing_original_execution[original_recorder.episode_id][0]
    original_turn["gateway_final_metadata"].pop("nutrition_consumed_occurrence")
    missing_execution = regrade(provenance=missing_original_execution)
    assert missing_execution["a1"] == "INCONCLUSIVE" and missing_execution["a2"] == "NOT_RUN"
    missing_finalization = copy.deepcopy(dialogue_binding["reviewed_turn_provenance"])
    missing_finalization[original_recorder.episode_id][0]["gateway_final_metadata"].pop("nutrition_finalization")
    no_execution = regrade(provenance=missing_finalization)
    assert no_execution["a1"] == "INCONCLUSIVE" and no_execution["a2"] == "NOT_RUN"
    current_mutations = [
        lambda final: final["nutrition_finalization"]["annotation"].update(meal_date="2030-01-01"),
        lambda final: final["nutrition_finalization"].update(
            annotation={"record_type": "meal_deletion"}),
        lambda final: final.pop("nutrition_model_proposal_annotation"),
        lambda final: final.update(
            nutrition_proposal_matches_committed=not final.get("nutrition_proposal_matches_committed")),
        lambda final: final.update(
            nutrition_model_proposal_annotation={"schema_version": 99, "record_type": "meal_deletion"}),
    ]
    if correction_recorder.episode_id == correction_row["metadata"].get("decision_trace_episode_id"):
        current_mutations.append(
            lambda final: final["nutrition_model_proposal_annotation"].update(meal_date="2030-01-01")
        )
    for mutate_current in current_mutations:
        mutated = copy.deepcopy(dialogue_binding["reviewed_turn_provenance"])
        mutate_current(mutated[correction_recorder.episode_id][0]["gateway_final_metadata"])
        invalid_execution = regrade(provenance=mutated)
        assert invalid_execution["a1"] == "INCONCLUSIVE" and invalid_execution["a2"] == "NOT_RUN", invalid_execution
    retry_turns = [
        (episode_id, turn)
        for episode_id, turns in dialogue_binding["reviewed_turn_provenance"].items()
        for turn in turns
        if episode_id != correction_recorder.episode_id
        and turn.get("gateway_final_metadata", {}).get("nutrition_append_event_id") == correction_row["id"]
    ]
    if retry_turns:
        retry_episode, _ = retry_turns[0]
        for mutate_retry in (
            lambda final: final.pop("nutrition_finalization"),
            lambda final: final["nutrition_finalization"]["annotation"].update(meal_date="2030-01-01"),
            lambda final: final.pop("nutrition_model_proposal_annotation"),
        ):
            mutated = copy.deepcopy(dialogue_binding["reviewed_turn_provenance"])
            mutate_retry(mutated[retry_episode][0]["gateway_final_metadata"])
            invalid_retry = regrade(provenance=mutated)
            assert invalid_retry["a1"] == "INCONCLUSIVE" and invalid_retry["a2"] == "NOT_RUN", invalid_retry
    short_day = copy.deepcopy(wellness_snapshot_bound)
    short_day["end"] = "2026-10-02T00:00:00+00:00"
    incomplete_days = regrade(wellness_snapshot=short_day)
    assert incomplete_days["a1"] == "INCONCLUSIVE" and incomplete_days["stage"] == "TELEGENT_BOUNDS_MISMATCH"
    capped_query = regrade(now=datetime(2026, 10, 2, 12, tzinfo=timezone.utc))
    assert capped_query["a1"] == "INCONCLUSIVE" and capped_query["stage"] == "TELEGENT_BOUNDS_MISMATCH"
    missing_original = regrade(rows=[row for row in scoped_rows if row["id"] != original_row["id"]])
    assert missing_original["a1"] == "INCONCLUSIVE" and missing_original["a2"] == "NOT_RUN"
    stale_receipt_rows = copy.deepcopy(scoped_rows)
    next(row for row in stale_receipt_rows if row["id"] == correction_row["id"])["metadata"]["client_op_id"] = "stale:assistant"
    stale_receipt = regrade(rows=stale_receipt_rows)
    assert stale_receipt["a1"] == "INCONCLUSIVE" and stale_receipt["a2"] == "NOT_RUN"
    wrong_target_rows = copy.deepcopy(scoped_rows)
    next(row for row in wrong_target_rows if row["id"] == correction_row["id"])["metadata"]["target_meal_id"] = "wrong-meal-id"
    wrong_target_grade = regrade(rows=wrong_target_rows)
    assert wrong_target_grade["a1"] == "FAIL" and wrong_target_grade["stage"] == "HONCHO_TARGET_MISMATCH"
    assert wrong_target_grade["a2"] == "NOT_RUN"
    malformed_selection_rows = copy.deepcopy(scoped_rows)
    next(row for row in malformed_selection_rows if row["id"] == correction_row["id"])["metadata"].pop("selected_source")
    malformed_selection = regrade(rows=malformed_selection_rows)
    assert malformed_selection["a1"] == "INCONCLUSIVE" and malformed_selection["a2"] == "NOT_RUN"
    wrong_selection_rows = copy.deepcopy(scoped_rows)
    next(row for row in wrong_selection_rows if row["id"] == correction_row["id"])["metadata"]["selected_source"]["append_source_message_id"] = "other-occurrence"
    wrong_selection = regrade(rows=wrong_selection_rows)
    assert wrong_selection["a1"] == "FAIL" and wrong_selection["stage"] == "HONCHO_TARGET_MISMATCH"
    assert wrong_selection["a2"] == "NOT_RUN"
    contradiction_rows = copy.deepcopy(scoped_rows)
    next(row for row in contradiction_rows if row["id"] == correction_row["id"])["metadata"]["reply_to_source_message_id"] = "other-photo"
    contradiction_grade = regrade(rows=contradiction_rows)
    assert contradiction_grade["a1"] == "FAIL" and contradiction_grade["stage"] == "HONCHO_TARGET_MISMATCH"
    assert contradiction_grade["a2"] == "NOT_RUN"
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
    assert retry.metadata["nutrition_committed_annotation"] == final.metadata["nutrition_committed_annotation"]
    assert retry.metadata["nutrition_context_evidence"] == final.metadata["nutrition_context_evidence"]
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
    burst_final, burst_recorder = await turn(
        pool, bundle, burst, burst_ctx, burst_user,
        observation(meal_at=burst_photo_at.isoformat()),
    )
    burst_ref = next(block for block in burst_user.content if isinstance(block, AttachmentRefBlock))
    assert burst_ref.attachment_id != ref.attachment_id
    burst_correction, burst_correction_ctx, burst_correction_user = inbound(
        pool, "burst-date-correction", "Это было 1 октября",
        when=burst_append_at + timedelta(minutes=1),
    )
    burst_fixed, burst_correction_recorder = await turn(
        pool, bundle, burst_correction, burst_correction_ctx, burst_correction_user,
        correction("2026-10-01"), loads=[burst_ref.attachment_id],
    )
    assert burst_final.metadata["nutrition_sync_status"] == "pending"
    assert burst_fixed.metadata["nutrition_sync_status"] == "pending"
    assert burst_final.metadata["nutrition_consumed_occurrence"]["photo_source_message_id"] == "burst-original-photo"
    assert burst_final.metadata["nutrition_consumed_occurrence"]["append_source_message_id"] == "burst-append"
    assert assert_annotation_round_trip(
        burst_final.metadata["nutrition_committed_annotation"], sparse=False
    ).record_type == "meal_observation"
    assert burst_fixed.metadata["nutrition_context_evidence"]["photo_source_message_id"] == "burst-original-photo"
    assert burst_fixed.metadata["nutrition_context_evidence"]["consumed_source_message_id"] == "burst-append"
    assert server.rows[-1]["metadata"]["selected_source"]["source_message_id"] == "burst-original-photo"
    assert server.rows[-1]["metadata"]["selected_source"]["append_source_message_id"] == "burst-append"
    assert burst_fixed.text == "Изменение сохранено; баланс обновляется."
    coalesced_grade = grade_contextual_receipts(pool, server, burst_final, burst_fixed,
        burst_recorder, burst_correction_recorder, source_id="burst-append",
        meal_day=date(2026, 10, 1), case_id="coalesced-physical-vs-append")
    assert (coalesced_grade["a1"], coalesced_grade["stage"], coalesced_grade["actual_latest_event_id"]) == (
        "PASS", "SAME_EVENT_PROJECTED", burst_fixed.metadata["nutrition_append_event_id"])
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
    original_final, legacy_recorder = await turn(
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
    date_final, date_recorder = await turn(
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
    phrase_final, phrase_recorder = await turn(
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
    phrase_retry, phrase_retry_recorder = await turn(
        pool, bundle, phrase, phrase_ctx, phrase_user,
        original_photo_time_correction(original_at + timedelta(hours=4)),
        loads=[legacy_ref.attachment_id],
        answer="Бургер примерно 900 ккал; время исправлено на полночь.",
    )
    assert len(server.rows) == before_phrase_retry
    assert phrase_retry.metadata["nutrition_append_event_id"] == phrase_event_id
    assert phrase_retry.metadata["nutrition_sync_status"] == "pending"
    assert phrase_retry.metadata["nutrition_committed_annotation"] == phrase_final.metadata["nutrition_committed_annotation"]
    assert phrase_retry.metadata["nutrition_proposal_matches_committed"] is False
    for field in ("nutrition_committed_annotation", "nutrition_model_proposal_annotation"):
        assert assert_annotation_round_trip(phrase_retry.metadata[field], sparse=True).record_type == "meal_correction"
    assert phrase_retry.text == "Изменение сохранено; баланс обновляется."
    assert "900 ккал" not in phrase_retry.text
    phrase_retry_grade = grade_contextual_receipts(
        pool, server, original_final, phrase_retry, legacy_recorder, phrase_retry_recorder,
        source_id="legacy-original-photo", meal_day=date(2026, 10, 1),
        case_id="reconciled-contextual-time-retry",
        additional_turns=((date_final, date_recorder), (phrase_final, phrase_recorder),
                          (phrase_retry, phrase_retry_recorder)),
        canonical_revision=3, canonical_meal_at=original_at.isoformat(),
        omit_explicit_meal_date=True)
    assert (phrase_retry_grade["a1"], phrase_retry_grade["stage"],
            phrase_retry_grade["actual_latest_event_id"]) == (
        "PASS", "SAME_EVENT_PROJECTED", phrase_event_id)
    phrase_operations = {phrase_event["metadata"]["client_op_id"]}
    assert sum(row["metadata"].get("client_op_id") in phrase_operations for row in server.rows) == 1
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
    portion_final, portion_recorder = await turn(
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
    occurrence = portion_final.metadata["nutrition_consumed_occurrence"]
    photo_binding = portion_receipt["metadata"]["photo_occurrence_source"]
    assert photo_binding["source_message_id"] == "uncertain-photo"
    assert photo_binding["append_source_message_id"] == "uncertain-photo"
    assert photo_binding["is_private"] is True and photo_binding["is_forwarded"] is False
    assert photo_binding["is_group"] is False and photo_binding["attachment_id"] == ref.attachment_id
    assert occurrence["photo_source_message_id"] == photo_binding["source_message_id"]
    assert occurrence["append_source_message_id"] == portion_receipt["metadata"]["source_message_id"]
    assert occurrence["receipt_event_id"] == portion_receipt["id"]
    assert occurrence["client_op_id"] == portion_receipt["metadata"]["client_op_id"]
    save_session_snapshot(cwd=pool._workspace, workspace=pool._workspace, model="offline",
        system_prompt="BASE", messages=bundle.engine.messages, usage=UsageSnapshot(),
        session_id=bundle.session_id, session_key="telegram:123")
    bundle.engine.messages = [ConversationMessage.model_validate(row)
                              for row in load_latest(pool._workspace)["messages"]]
    msg3, ctx3, user3 = inbound(pool, "portion-date-correction", "Это было 3 октября", when=BASE+timedelta(days=2))
    final, portion_correction_recorder = await turn(
        pool, bundle, msg3, ctx3, user3, correction("2026-10-03"), loads=[ref.attachment_id]
    )
    out["portion_then_correction"] = {"response": final.text, "delivery": final.metadata}
    handoff["portion"] = copy.deepcopy(server.rows)
    assert final.metadata["nutrition_sync_status"] == "pending"
    assert portion_final.metadata["nutrition_consumed_occurrence"]["photo_source_message_id"] == "uncertain-photo"
    assert portion_final.metadata["nutrition_consumed_occurrence"]["append_source_message_id"] == "portion-answer"
    assert final.metadata["nutrition_context_evidence"]["photo_source_message_id"] == "uncertain-photo"
    assert final.metadata["nutrition_context_evidence"]["consumed_source_message_id"] == "portion-answer"
    assert server.rows[-1]["metadata"]["selected_source"]["append_source_message_id"] == "portion-answer"
    assert server.rows[-1]["metadata"]["target_meal_id"]
    assert portion_receipt["metadata"]["source_message_id"] == "portion-answer"
    portion_grade = grade_contextual_receipts(pool, server, portion_final, final,
        portion_recorder, portion_correction_recorder, source_id="portion-answer",
        meal_day=date(2026, 10, 3), case_id="portion-context-restored")
    assert (portion_grade["a1"], portion_grade["stage"],
            portion_grade.get("actual_latest_event_id")) == (
        "PASS", "SAME_EVENT_PROJECTED", final.metadata["nutrition_append_event_id"]), {
            key: portion_grade.get(key) for key in ("a1", "stage", "reason", "a2")
        }
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


@pytest.mark.asyncio
async def test_receipt_photo_occurrence_patch_cannot_authorize_unrelated_text_consumption(tmp_path):
    global ROOT
    previous_root, ROOT = ROOT, tmp_path
    try:
        photo = ROOT / "synthetic.png"
        photo.write_bytes(PNG_BYTES)
        pool, bundle, server, client = setup("forged-occurrence")
        old_photo, old_ctx, old_user = inbound(pool, "never-consumed-photo", media=[str(photo)])
        await turn(pool, bundle, old_photo, old_ctx, old_user, None, answer="What food is shown?")
        ref = next(block for block in old_user.content if isinstance(block, AttachmentRefBlock))
        provenance = ref.source_provenance
        server.assistant_metadata_patch = {
            "photo_occurrence_source": {
                "schema_version": 1, "tenant_id": "review-tenant",
                "source_principal": provenance["principal"],
                "gateway_session_id": provenance["gateway_session_id"],
                "source_message_id": provenance["source_message_id"],
                "append_source_message_id": provenance["append_source_message_id"],
                "attachment_id": ref.attachment_id, "received_at": provenance["received_at"],
                "chat_id": provenance["chat_id"], "session_key": provenance["session_key"],
                "is_private": True, "is_forwarded": False, "is_group": False,
            }
        }
        unrelated, unrelated_ctx, unrelated_user = inbound(
            pool, "unrelated-text-meal", "I ate a different meal", when=BASE + timedelta(minutes=1)
        )
        accepted, accepted_recorder = await turn(
            pool, bundle, unrelated, unrelated_ctx, unrelated_user,
            observation(meal_at=BASE.isoformat()), loads=(),
        )
        assert bundle.engine.captured_tools == []
        assert not unrelated.media
        assert "nutrition_consumed_occurrence" not in accepted.metadata
        assert not ref.source_provenance.get("consumed_occurrences")
        server.assistant_metadata_patch = {}
        edit, edit_ctx, edit_user = inbound(
            pool, "forged-photo-correction", "Correct that photo date", when=BASE + timedelta(days=1)
        )
        corrected, correction_recorder = await turn(
            pool, bundle, edit, edit_ctx, edit_user, correction(), loads=[ref.attachment_id]
        )
        assert "nutrition_context_evidence" not in corrected.metadata
        result = grade_contextual_receipts(
            pool, server, accepted, corrected, accepted_recorder, correction_recorder,
            source_id="unrelated-text-meal", meal_day=date(2026, 10, 2),
            case_id="forged-original-occurrence",
        )
        assert result["a1"] == "INCONCLUSIVE" and result["a2"] == "NOT_RUN", result
        await client.aclose()
    finally:
        ROOT = previous_root


@pytest.mark.asyncio
async def test_native_reply_with_selected_photo_still_requires_context_receipt_proof(tmp_path, monkeypatch):
    global ROOT
    previous_root, ROOT = ROOT, tmp_path
    photo = ROOT / "synthetic.png"
    photo.write_bytes(PNG_BYTES)
    pool, bundle, server, client = setup("native-with-selected-context-proof")
    try:
        original_message, original_ctx, original_user = inbound(
            pool, "native-context-photo", media=[str(photo)]
        )
        original, original_recorder = await turn(
            pool, bundle, original_message, original_ctx, original_user, observation()
        )
        ref = next(block for block in original_user.content if isinstance(block, AttachmentRefBlock))
        edit_message, edit_ctx, edit_user = inbound(
            pool, "native-context-correction", "Correct the date",
            when=BASE + timedelta(days=1),
            metadata_extra={"reply_to_message_id": "native-context-photo"},
        )
        edit, edit_recorder = await turn(
            pool, bundle, edit_message, edit_ctx, edit_user, correction(),
            loads=[ref.attachment_id],
        )
        correction_receipt = next(
            row for row in server.rows
            if row["id"] == edit.metadata["nutrition_append_event_id"]
        )
        assert correction_receipt["metadata"]["reply_to_source_message_id"] == "native-context-photo"
        assert correction_receipt["metadata"]["selected_source"]["append_source_message_id"] == "native-context-photo"
        assert edit.metadata["nutrition_context_evidence"]

        import ohmo.evals.nutrition_persistence as persistence
        original_grade = persistence.grade_manifest
        captured = []

        def capture_grade(manifest, honcho, wellness, **kwargs):
            captured.append(copy.deepcopy((manifest, honcho, wellness, kwargs)))
            return original_grade(manifest, honcho, wellness, **kwargs)

        monkeypatch.setattr(persistence, "grade_manifest", capture_grade)
        baseline = grade_contextual_receipts(
            pool, server, original, edit, original_recorder, edit_recorder,
            source_id="native-context-photo", meal_day=date(2026, 10, 2),
            case_id="native-reply-with-selected-context",
        )
        assert baseline["a1"] == "PASS", baseline
        assert len(captured) == 1
        manifest, honcho, wellness, kwargs = captured[0]

        for mutation in (
            lambda final: final.pop("nutrition_finalization"),
            lambda final: final.pop("nutrition_context_evidence"),
            lambda final: final["nutrition_context_evidence"].update(tenant_id="foreign-owner"),
            lambda final: final["nutrition_context_evidence"].update(schema_version=99),
        ):
            mutated_provenance = copy.deepcopy(kwargs["reviewed_turn_provenance"])
            mutation(mutated_provenance[edit_recorder.episode_id][0]["gateway_final_metadata"])
            result = original_grade(
                manifest, honcho, wellness,
                **{**kwargs, "reviewed_turn_provenance": mutated_provenance},
            )[0]
            assert result["a1"] == "INCONCLUSIVE" and result["a2"] == "NOT_RUN", result

        contradicted = copy.deepcopy(honcho)
        contradicted_row = next(
            row for row in contradicted["messages"]
            if row["id"] == correction_receipt["id"]
        )
        contradicted_row["metadata"]["reply_to_source_message_id"] = "different-photo"
        mismatch = original_grade(manifest, contradicted, wellness, **kwargs)[0]
        assert mismatch["a1"] == "FAIL" and mismatch["stage"] == "HONCHO_TARGET_MISMATCH", mismatch
    finally:
        await client.aclose()
        ROOT = previous_root


@pytest.mark.asyncio
async def test_native_only_reply_correction_passes_without_contextual_selection(tmp_path):
    global ROOT
    previous_root, ROOT = ROOT, tmp_path
    photo = ROOT / "synthetic.png"
    photo.write_bytes(PNG_BYTES)
    native_pool, native_bundle, native_server, native_client = setup("native-only-correction")
    try:
        native_photo, native_photo_ctx, native_photo_user = inbound(
            native_pool, "native-only-photo", media=[str(photo)]
        )
        native_original, native_original_recorder = await turn(
            native_pool, native_bundle, native_photo, native_photo_ctx, native_photo_user,
            observation(),
        )
        native_edit_message, native_edit_ctx, native_edit_user = inbound(
            native_pool, "native-only-correction", "Correct the date",
            when=BASE + timedelta(days=1),
            metadata_extra={"reply_to_message_id": "native-only-photo"},
        )
        native_edit, native_edit_recorder = await turn(
            native_pool, native_bundle, native_edit_message, native_edit_ctx, native_edit_user,
            correction(),
        )
        native_receipt = next(
            row for row in native_server.rows
            if row["id"] == native_edit.metadata["nutrition_append_event_id"]
        )
        assert "selected_source" not in native_receipt["metadata"]
        assert "nutrition_context_evidence" not in native_edit.metadata
        native_grade = grade_contextual_receipts(
            native_pool, native_server, native_original, native_edit,
            native_original_recorder, native_edit_recorder,
            source_id="native-only-photo", meal_day=date(2026, 10, 2),
            case_id="native-only-correction",
        )
        assert native_grade["a1"] == "PASS", native_grade
    finally:
        await native_client.aclose()
        ROOT = previous_root

"""Session-aware runtime pool for ohmo gateway."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import re
import string
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path

from ohmo.attachment_store import AttachmentStore
from ohmo.contact_registry import ContactStore
from ohmo.conversation_image_tool import LoadConversationImageTool
from ohmo.evals import GatewayEvalRecorder
from ohmo.evals.recorder import _contains_nutrition_record_intent
from ohmo.evals.nutrition_trace import (
    NutritionAnnotationV2,
)
from ohmo.evals.nutrition_persistence import derive_meal_id
from ohmo.gateway.attachment_fingerprints import compute_attachment_fingerprints
from ohmo.gateway.camera import (
    CAMERA_AUTHORITY,
    CAMERA_CONTEXT_QUESTION_AUTHORITY,
    _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY,
    COALESCED_ATTACHMENT_PROVENANCE_AUTHORITY,
    RetainedAttachmentEvidence,
    _validate_camera_portion_correction_annotation,
)
from ohmo.gateway.config import load_gateway_config
from ohmo.gateway.group_tool import (
    CreateFeishuGroup,
    OhmoCreateFeishuGroupTool,
    PublishGroupWelcome,
)
from ohmo.gateway.memory_gate import (
    GateDecision,
    MemoryScope,
    evaluate_memory_gate,
    principal_isolated_session,
    resolve_memory_scope,
)
from ohmo.gateway.profile_context import render_profile_context
from ohmo.gateway.provider_commands import (
    handle_gateway_model_command,
    handle_gateway_provider_command,
)
from ohmo.gateway.router import session_key_for_message
from ohmo.gateway.selected_source import resolve_selected_photo_source
from ohmo.gateway.turn_context import (
    TurnContext,
    build_turn_context,
    canonical_principal,
    is_private_message,
)
from ohmo.group_registry import load_managed_group_record, normalize_cwd
from ohmo.memory import create_memory_command_backend, ensure_catalog_migrated
from ohmo.memory_backend import (
    CatalogMemoryBackend,
    ConversationAppendReceipt,
    ConversationReconciliationError,
    FileMemoryBackend,
    MemoryBackend,
    ShadowMemoryBackend,
    make_memory_backend,
    make_tenant_shadow_backend,
)
from ohmo.memory_judge import judge_enabled, judge_interval, run_memory_judge
from ohmo.memory_store import MemoryStore
from ohmo.memory_tool import OhmoMemoryTool
from ohmo.prompt_seam import compose_runtime_prompt, prepare_turn
from ohmo.prompts import build_ohmo_system_prompt
from ohmo.reminders.store import ReminderStore
from ohmo.reminders.tool import (
    RemindCancelTool,
    RemindCreateTool,
    RemindListTool,
    WellnessTenantResolver,
)
from ohmo.session_storage import (
    OhmoSessionBackend,
    _snapshot_message_dict,
    clear_session_work_dir,
    get_session_work_dir,
    reap_stale_work_dirs,
)
from ohmo.todo_store import TodoStore, canonicalize_todos
from ohmo.todo_write_tool import OhmoTodoWriteTool
from ohmo.workspace import (
    get_memory_dir,
    get_plugins_dir,
    get_sessions_dir,
    get_skills_dir,
    initialize_workspace,
)
from openharness.channels.bus.events import InboundMessage
from openharness.commands import CommandContext, CommandResult, lookup_skill_slash_command
from openharness.engine.messages import (
    AttachmentRefBlock,
    ConversationMessage,
    ImageBlock,
    TextBlock,
    sanitize_conversation_messages,
)
from openharness.engine.query import MaxTurnsExceeded
from openharness.engine.stream_events import (
    AssistantTextDelta,
    AssistantTurnComplete,
    CompactProgressEvent,
    ErrorEvent,
    StatusEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from openharness.prompts import build_runtime_system_prompt
from openharness.tools.mcp_tool import McpToolAdapter, WellnessLoginInjectingAdapter
from openharness.mcp.wellness_delegation import TrustedWellnessActor
from openharness.ui.runtime import (
    RuntimeBundle,
    _last_user_text,
    build_runtime,
    close_runtime,
    start_runtime,
)

logger = logging.getLogger(__name__)


_CHANNEL_THINKING_PHRASES = (
    "🤔 想一想…",
    "🧠 琢磨中…",
    "✨ 整理一下思路…",
    "🔎 看看这个…",
    "🪄 捋一捋线索…",
)

_CHANNEL_THINKING_PHRASES_EN = (
    "🤔 Thinking it through…",
    "🧠 Working on it…",
    "🔎 Looking into it…",
    "🧩 Following the thread…",
    "📝 Pulling it together…",
)

_TEXT_PREVIEW_BYTES = 4096
_TEXT_PREVIEW_CHARS = 900
_BINARY_HEAD_BYTES = 32
_FINAL_REPLY_IMAGE_PATH_RE = re.compile(
    r"(?P<path>(?:[A-Za-z]:[\\/]|/)[^\r\n`\"'<>|?*\x00]+?\.(?:png|jpe?g|webp|gif|bmp))",
    re.IGNORECASE,
)
_IMAGE_FALLBACK_NOTE = (
    "[Image attachment omitted because the active model does not support image input. "
    "Use the attachment paths and summaries above if needed.]"
)
_NO_GROUP_REQUEST = object()
_UNRESOLVED_MEMORY_SCOPE = object()
_SELECTED_SOURCE_AUTHORITY = object()
_GROUP_TOOL_NAME = "ohmo_create_feishu_group"
_WELLNESS_TOOL_NAME = "mcp__worfalomey__get_wellness_data"
_GROUP_AGENT_PROMPT_PREFIX = "The user invoked `/group` from a Feishu private chat."
_GROUP_AGENT_PROMPT_REQUEST_MARKER = "User /group request:"
_GROUP_METADATA_KEYS = (
    "task_focus_state",
    "recent_work_log",
    "recent_verified_work",
    "compact_checkpoints",
    "compact_last",
)
DEFAULT_REMINDER_TZ = "Europe/Moscow"
DEFAULT_REMINDER_MAX_PER_CHAT = 50
_CONVERSATION_TRACE_DISABLED_STATUS = "disabled"
_TODO_RECONCILIATION_MAX_ATTEMPTS = 2
_TODO_TOOL_NAME = "todo_write"
_TODO_STATE_READ_ERROR_MARKER = "TODO_STATE_READ_ERROR"
_USER_PHOTO_ADVISORY_RE = re.compile(
    r"\b(?:what\s+is\s+this|identify|advice|recipe|how\s+to\s+cook|"
    r"how\s+many\s+calories|calories?|"
    r"сколько\s+калори\w*|посоветуй|рецепт|что\s+это|что\s+на\s+фото|"
    r"как\s+приготовить|оцени\s+калори\w*)\b",
    re.IGNORECASE,
)


class TodoRuntimeStateError(RuntimeError):
    """The trusted todo snapshot could not be read safely."""

    def __init__(self, session_id: str, cause: BaseException) -> None:
        self.session_id = session_id
        self.cause = cause
        super().__init__(
            "Todo runtime state is unavailable for session "
            f"{session_id!r}: {type(cause).__name__}: {cause}"
        )


def _trusted_utc_iso(value: object) -> str | None:
    """Normalize a trusted aware datetime/ISO value to UTC, rejecting ambiguity."""
    timestamp: datetime
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        try:
            timestamp = datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    else:
        return None
    try:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            return None
        return timestamp.astimezone(timezone.utc).isoformat()
    except (OverflowError, ValueError):
        return None


def _trusted_inbound_event_time(message: InboundMessage) -> datetime | None:
    """Use Telegram's native received_at marker when the adapter supplies it."""
    metadata = message.metadata or {}
    value: object = message.timestamp
    if message.channel == "telegram" and "received_at" in metadata:
        value = metadata.get("received_at")
    normalized = _trusted_utc_iso(value)
    if normalized is None:
        return None
    return datetime.fromisoformat(normalized)


def _wellness_actor_for_turn(turn_ctx: TurnContext | None) -> TrustedWellnessActor | None:
    """Capture a numeric principal only from a private Telegram admission."""
    if turn_ctx is None or turn_ctx.channel.strip().lower() != "telegram":
        return None
    principal = canonical_principal(turn_ctx.channel, turn_ctx.principal)
    if (
        turn_ctx.is_private is not True
        or not principal.isascii()
        or not principal.isdigit()
        or principal.startswith("0")
    ):
        return None
    return TrustedWellnessActor(principal)


@dataclass(frozen=True)
class GatewayStreamUpdate:
    """One outbound update produced while processing a channel message."""

    kind: str
    text: str
    metadata: dict[str, object]
    media: list[str] | None = None


@dataclass(frozen=True)
class _DecisionTraceRecorderRestore:
    engine: object
    previous: object | None


def _evals_capture_enabled(config) -> bool:
    """Whether to capture an eval episode for this turn.

    The ``OHMO_EVALS_CAPTURE`` env var, when set to a non-empty value,
    overrides the persistent ``gateway.json`` ``evals_capture`` flag
    (default on): ``0``/``false``/``no``/``off`` disable capture, anything
    else enables it. Lets ops flip capture without editing config.
    """
    raw = os.getenv("OHMO_EVALS_CAPTURE")
    if raw is not None and raw.strip():
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return bool(getattr(config, "evals_capture", True))


def _install_gateway_decision_trace_recorder(
    engine: object,
    recorder: GatewayEvalRecorder | None,
) -> _DecisionTraceRecorderRestore | None:
    if recorder is None:
        return None
    set_recorder = getattr(engine, "set_decision_trace_recorder", None)
    if not callable(set_recorder):
        return None
    previous = getattr(engine, "decision_trace_recorder", None)
    set_recorder(recorder.decision_trace_recorder)
    return _DecisionTraceRecorderRestore(engine=engine, previous=previous)


def _restore_gateway_decision_trace_recorder(
    restore: _DecisionTraceRecorderRestore | None,
) -> None:
    if restore is None:
        return
    set_recorder = getattr(restore.engine, "set_decision_trace_recorder", None)
    if callable(set_recorder):
        set_recorder(restore.previous)


def _message_identity_for_turn(message: InboundMessage) -> str:
    metadata = message.metadata or {}

    for key in ("message_id", "messageId", "message-id"):
        if key not in metadata:
            continue
        message_id = metadata[key]
        if message_id is not None:
            rendered_id = str(message_id).strip()
            if rendered_id:
                return rendered_id

    timestamp = getattr(message, "timestamp", None)
    if isinstance(timestamp, datetime):
        return timestamp.isoformat()
    if isinstance(timestamp, date):
        return timestamp.isoformat()
    if timestamp is not None:
        rendered_timestamp = str(timestamp).strip()
        if rendered_timestamp:
            return rendered_timestamp
    return "unknown"


def _logical_turn_id_for_conversation(
    *,
    turn_ctx: TurnContext,
    message: InboundMessage,
) -> str:
    seed = "\x00".join(
        (
            str(turn_ctx.channel),
            str(turn_ctx.chat_id),
            canonical_principal(turn_ctx.channel, turn_ctx.principal),
            str(turn_ctx.session_id),
            _message_identity_for_turn(message),
        )
    ).encode("utf-8")
    return f"ohmo-turn-{hashlib.sha256(seed).hexdigest()}"


def _normalize_source_message_ref(value: object) -> str | None:
    """Normalize a channel-owned message reference into a plain string id.

    The raw channel name (Telegram ``message_id`` / ``reply_to_message_id``)
    never crosses the gateway boundary: it is normalized here so another
    channel can implement the same contract. The model never authors these.
    """
    if value is None or isinstance(value, bool):
        return None
    rendered = str(value).strip()
    return rendered or None


def _exact_context_receipt_replay(
    projection: object, receipt: object, expected_metadata: Mapping[str, object],
    *, expected_user_text: str | None = None,
    expected_native_target: str | None = None,
) -> bool:
    """Match a retained correction to its exact immutable current-operation envelope."""
    if not isinstance(projection, Mapping):
        return False
    metadata = getattr(receipt, "assistant_metadata", None)
    if not isinstance(metadata, Mapping):
        return False
    trace = metadata.get("decision_trace")
    annotations = trace.get("annotations") if isinstance(trace, Mapping) else None
    nutrition = annotations.get("nutrition") if isinstance(annotations, Mapping) else None
    selected = expected_metadata.get("selected_source")
    if not isinstance(selected, Mapping):
        # A runtime reset can rebuild the turn before the Camera context binding
        # is attached to user metadata. The retained projection is server-owned
        # and must still match the immutable receipt below.
        selected = projection.get("selected_source")
    expected_target = expected_metadata.get("target_meal_id")
    if not isinstance(expected_target, str):
        expected_target = projection.get("target_meal_id_receipt")
    checks = [
        ("selected", isinstance(selected, Mapping)),
        ("event_id", projection.get("event_id") == getattr(receipt, "assistant_message_id", None)),
        ("assistant_op", projection.get("client_op_id") == getattr(receipt, "assistant_client_op_id", None)),
        ("user_text", expected_user_text is None or projection.get("user_text") == expected_user_text),
        ("logical_turn", projection.get("logical_turn_id") == metadata.get("logical_turn_id")),
        ("assistant_suffix", projection.get("client_op_id") == f"{projection.get('logical_turn_id')}:assistant"),
        ("user_suffix", projection.get("user_client_op_id") == f"{projection.get('logical_turn_id')}:user"),
        ("user_op", getattr(receipt, "user_client_op_id", None) == projection.get("user_client_op_id")),
        ("session", projection.get("gateway_session_id") == metadata.get("gateway_session_id")),
        ("current_source", projection.get("source_message_id_current") == metadata.get("source_message_id")),
        ("expected_source", projection.get("source_message_id_current") == expected_metadata.get("source_message_id")),
        ("stored_time", projection.get("received_at") == metadata.get("received_at")),
        ("current_time", projection.get("received_at") == expected_metadata.get("received_at")),
        ("stored_tenant", projection.get("tenant_id_receipt") == metadata.get("tenant_id")),
        ("current_tenant", projection.get("tenant_id_receipt") == expected_metadata.get("tenant_id")),
        ("stored_principal", projection.get("source_principal_receipt") == metadata.get("source_principal")),
        ("current_principal", projection.get("source_principal_receipt") == expected_metadata.get("source_principal")),
        ("stored_reply", projection.get("reply_to_source_message_id") == metadata.get("reply_to_source_message_id")),
        ("current_reply", projection.get("reply_to_source_message_id") == expected_metadata.get("reply_to_source_message_id")),
        ("native_target", projection.get("reply_to_native_message_id") == expected_native_target),
        ("selected_source", projection.get("selected_source") == metadata.get("selected_source") == selected),
        ("stored_target", projection.get("target_meal_id_receipt") == metadata.get("target_meal_id")),
        ("current_target", metadata.get("target_meal_id") == expected_target),
        ("episode", projection.get("decision_trace_episode_id") == metadata.get("decision_trace_episode_id")),
        ("nutrition", projection.get("nutrition_annotation") == nutrition),
    ]
    return all(passed for _name, passed in checks)


def _build_conversation_turn_metadata(
    *,
    turn_ctx: TurnContext,
    message: InboundMessage,
    scope: MemoryScope,
    recorder: GatewayEvalRecorder | None = None,
) -> tuple[str, dict[str, object], dict[str, object]]:
    retained_camera_turn = message.metadata.get("_camera_turn_id")
    logical_turn_id = (
        retained_camera_turn
        if message.metadata.get("_camera_authority") is CAMERA_AUTHORITY
        and isinstance(retained_camera_turn, str)
        and retained_camera_turn
        else _logical_turn_id_for_conversation(turn_ctx=turn_ctx, message=message)
    )
    message_metadata = message.metadata or {}
    source_principal = (
        f"{turn_ctx.channel}:{canonical_principal(turn_ctx.channel, turn_ctx.principal)}"
    )
    decision_trace_status = (
        recorder.decision_trace_status
        if recorder is not None
        else _CONVERSATION_TRACE_DISABLED_STATUS
    )
    nutrition_annotation_status = (
        recorder.nutrition_annotation_status
        if recorder is not None
        else _CONVERSATION_TRACE_DISABLED_STATUS
    )
    attachment_fingerprints = compute_attachment_fingerprints(message.media)
    source_image_attachment_count = sum(
        1 for media_path in message.media or [] if _is_image_attachment(media_path)
    )
    base_metadata: dict[str, object] = {
        "tenant_id": scope.private_tenant,
        "source_principal": source_principal,
        "gateway_session_id": turn_ctx.session_id,
        "logical_turn_id": logical_turn_id,
        "client_op_id": f"{logical_turn_id}:user",
        "decision_trace_status": decision_trace_status,
        "nutrition_annotation_status": nutrition_annotation_status,
        "decision_trace_episode_id": recorder.episode_id if recorder is not None else None,
        "received_at": _trusted_utc_iso(_trusted_inbound_event_time(message)),
        "is_forwarded": turn_ctx.is_forwarded,
        "is_group": not turn_ctx.is_private,
        "source_message_at": _trusted_utc_iso(message_metadata.get("source_message_at")),
        "source_message_id": _normalize_source_message_ref(message_metadata.get("message_id")),
        "reply_to_source_message_id": _normalize_source_message_ref(
            message_metadata.get("reply_to_message_id")
        ),
        "attachment_fingerprints": attachment_fingerprints,
        "source_image_attachment_count": source_image_attachment_count,
    }
    selected_binding = message.metadata.get("_selected_source_binding")
    if (
        isinstance(selected_binding, tuple)
        and len(selected_binding) == 2
        and selected_binding[0] is _SELECTED_SOURCE_AUTHORITY
        and isinstance(selected_binding[1], Mapping)
    ):
        binding = selected_binding[1]
        if (
            message.metadata.get("_camera_context_meal_target")
            is _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY
            and _normalize_source_message_ref(message.metadata.get("reply_to_message_id"))
            is not None
        ):
            base_metadata["reply_to_source_message_id"] = binding.get("append_source_message_id")
        raw_nutrition = recorder.validated_nutrition_envelope if recorder is not None else None
        try:
            correction = NutritionAnnotationV2.model_validate(raw_nutrition)
        except (TypeError, ValueError):
            correction = None
        if (
            correction is not None
            and correction.record_type in {"meal_correction", "meal_deletion"}
            and scope.private_tenant == binding.get("tenant_id")
            and turn_ctx.is_private
            and not turn_ctx.is_forwarded
            and binding.get("source_principal") == source_principal
            and (
                binding.get("gateway_session_id") == turn_ctx.session_id
                or message.metadata.get("_camera_context_meal_target")
                is _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY
            )
            and isinstance(binding.get("gateway_session_id"), str)
            and binding.get("gateway_session_id")
            and isinstance(binding.get("append_source_message_id"), str)
        ):
            contextual_target = (
                message.metadata.get("_camera_context_meal_target")
                is _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY
            )
            evidence = {
                "schema_version": 2 if contextual_target else 1,
                **{
                    key: binding[key]
                    for key in (
                        "tenant_id", "source_principal", "gateway_session_id",
                        "source_message_id", "append_source_message_id", "is_private",
                        "is_forwarded", "is_group",
                    )
                },
            }
            if contextual_target:
                original_event_id = binding.get("original_receipt_event_id")
                original_operation_id = binding.get("original_receipt_client_op_id")
                if (
                    not isinstance(original_event_id, str) or not original_event_id
                    or not isinstance(original_operation_id, str) or not original_operation_id
                    or not original_operation_id.endswith(":assistant")
                    or original_operation_id != binding.get("client_op_id")
                    or original_event_id != binding.get("event_id")
                ):
                    return logical_turn_id, {}, {}
                evidence["original_receipt_event_id"] = original_event_id
                evidence["original_receipt_client_op_id"] = original_operation_id
            base_metadata["selected_source"] = evidence
            base_metadata["target_meal_id"] = derive_meal_id(
                tenant_id=scope.private_tenant,
                source_principal=source_principal,
                gateway_session_id=binding["gateway_session_id"],
                source_message_id=binding["append_source_message_id"],
            )
        elif (
            correction is not None
            and correction.record_type == "meal_observation"
            and correction.consumption_status == "consumed"
            and scope.private_tenant == binding.get("tenant_id")
            and turn_ctx.is_private
            and not turn_ctx.is_forwarded
            and binding.get("source_principal") == source_principal
            and binding.get("gateway_session_id") == turn_ctx.session_id
            and isinstance(binding.get("append_source_message_id"), str)
            and isinstance(binding.get("attachment_id"), str)
        ):
            base_metadata["photo_occurrence_source"] = {
                key: binding[key]
                for key in (
                    "schema_version", "tenant_id", "source_principal",
                    "gateway_session_id", "source_message_id",
                    "append_source_message_id", "attachment_id", "received_at",
                    "chat_id", "session_key",
                    "is_private", "is_forwarded", "is_group",
                )
            }
    if (
        turn_ctx.camera_authorized
        and message.metadata.get("_camera_authority") is CAMERA_AUTHORITY
        and message.sender_id != "__camera__"
    ):
        base_metadata.update(ingest_source="dropbox_camera", confirmation_required=True)
    user_metadata = dict(base_metadata)
    assistant_metadata = dict(base_metadata)
    # Never accept operation/provenance fields from channel metadata. The
    # values above are gateway-owned and regenerated for each ordinary turn.
    assistant_metadata["client_op_id"] = f"{logical_turn_id}:assistant"
    decision_trace = recorder.decision_trace_envelope if recorder is not None else None
    if decision_trace is not None:
        assistant_metadata["decision_trace"] = dict(decision_trace)
    raw_nutrition = recorder.validated_nutrition_envelope if recorder is not None else None
    nutrition_v2 = bool(
        isinstance(raw_nutrition, Mapping)
        and raw_nutrition.get("schema_version", 1) == 2
    )
    camera_yes = (
        turn_ctx.camera_authorized
        and message.metadata.get("_camera_authority") is CAMERA_AUTHORITY
        and message.metadata.get("_camera_answer") == "yes"
    )
    if camera_yes and not nutrition_v2:
        assistant_metadata["camera_finalizer_outcome"] = "clarification"
    if (
        turn_ctx.camera_authorized
        and message.metadata.get("_camera_context_question")
        is CAMERA_CONTEXT_QUESTION_AUTHORITY
    ):
        for metadata in (user_metadata, assistant_metadata):
            metadata["camera_candidate_id"] = message.metadata["_camera_candidate_id"]
            metadata["camera_operation_id"] = message.metadata["_camera_candidate_id"]
            metadata["camera_context_only"] = True
            metadata["camera_route"] = "context"
        assistant_metadata["camera_finalizer_outcome"] = "clarification"
    return logical_turn_id, user_metadata, assistant_metadata


def _camera_eval_capture_provenance(
    *, message: InboundMessage, turn_ctx: TurnContext, scope: MemoryScope,
    camera_config: object, camera_ingress: object | None, logical_turn_id: str,
    assistant_metadata: Mapping[str, object],
) -> tuple[dict[str, str] | None, dict[str, object] | None]:
    """Capture trusted Camera receipt context after the runtime authorization gate."""
    metadata = message.metadata or {}
    config = camera_config
    attempts = getattr(camera_ingress, "_attempts", None)
    if (not turn_ctx.camera_authorized or metadata.get("_camera_authority") is not CAMERA_AUTHORITY
            or getattr(camera_ingress, "config", None) != config or not getattr(config, "enabled", False)
            or message.channel != "telegram" or str(message.chat_id) != getattr(config, "chat_id", None)
            or scope.private_tenant != getattr(config, "tenant_id", None)
            or not isinstance(attempts, dict)):
        return None, None
    candidate_id = metadata.get("_camera_candidate_id")
    attempt = attempts.get(candidate_id) if isinstance(candidate_id, str) else None
    photo_id = attempt.get("photo_id") if isinstance(attempt, dict) else None
    if (not isinstance(candidate_id, str) or not candidate_id or not isinstance(attempt, dict)
            or attempt.get("candidate_id", candidate_id) != candidate_id
            or attempt.get("photo_delivery_confirmed") is not True
            or type(photo_id) is not int or photo_id <= 0):
        return None, None
    principal = f"telegram:{canonical_principal('telegram', str(getattr(config, 'principal', '')))}"
    common: dict[str, object] = {
        "candidate_id": candidate_id, "native_photo_id": photo_id,
        "tenant_id": scope.private_tenant, "gateway_session_id": turn_ctx.session_id,
        "recipient_principal": principal,
    }
    if message.sender_id == "__camera__":
        source_principal = assistant_metadata.get("source_principal")
        operation_id = assistant_metadata.get("client_op_id")
        if (metadata.get("_synthetic") is not True or metadata.get("_camera_photo_id") != photo_id
                or assistant_metadata.get("source_message_id") is not None or not message.media
                or not isinstance(logical_turn_id, str) or not logical_turn_id
                or operation_id != f"{logical_turn_id}:assistant"
                or not isinstance(source_principal, str) or not source_principal):
            return None, None
        return None, {**common, "kind": "initial_context", "source_principal": source_principal,
                      "logical_turn_id": logical_turn_id, "operation_id": operation_id}
    source = assistant_metadata.get("source_message_id")
    op = assistant_metadata.get("client_op_id")
    principal_id = assistant_metadata.get("source_principal")
    retained_turn = metadata.get("_camera_turn_id")
    if (turn_ctx.principal != canonical_principal("telegram", str(getattr(config, "principal", "")))
            or not isinstance(source, str) or not source or retained_turn != logical_turn_id
            or not isinstance(op, str) or op != f"{logical_turn_id}:assistant" or principal_id != principal):
        return None, None
    native_binding = metadata.get("_camera_native_binding")
    allowed_bindings = {str(photo_id), *(str(value) for value in attempt.get("reply_ids", []))}
    if native_binding is not None and (not isinstance(native_binding, str) or native_binding not in allowed_bindings):
        return None, None
    turn = {"source_message_id": source, "principal_id": principal_id,
            "logical_turn_id": logical_turn_id, "operation_id": op}
    return turn, {**common, "kind": "owner_turn", **turn}


def _nutrition_annotation_metadata(annotation: NutritionAnnotationV2) -> dict[str, object]:
    """Keep correction metadata sparse so changed_fields remains authoritative."""
    return annotation.model_dump(
        mode="json", exclude_unset=annotation.record_type == "meal_correction"
    )


def _append_nutrition_saved_status(answer: str, annotation: NutritionAnnotationV2) -> str:
    # Resolve only a compound storage-failure assertion. Keep neighboring
    # nutrition facts and independent advice, including positive save advice.
    storage = re.compile(
        r"\b(?:баз\w*|проекц\w*|сохран\w*|запис\w*|database|projection|save\w*|record\w*)\b",
        re.IGNORECASE,
    )
    failure = re.compile(
        r"\b(?:не\s+(?:удалось|смог\w*|получил\w*|получится|могу)|failed\s+to|could\s+not|couldn't)\b",
        re.IGNORECASE,
    )
    failed_storage_action = re.compile(
        r"\b(?:не\s+(?:удалось|смог\w*|получил\w*|получится|могу)\s+"
        r"(?:сохран\w*|запис\w*)|"
        r"(?:failed\s+to|could\s+not|couldn't)\s+(?:save\w*|record\w*))\b",
        re.IGNORECASE,
    )
    empty_store = re.compile(
        r"\b(?:баз\w*|проекц\w*|database|projection)\b.{0,80}"
        r"\b(?:нет|пуст\w*|отсутств\w*|empty|no\s+record)\b",
        re.IGNORECASE,
    )
    known_meal_time = annotation.meal_at is not None or annotation.meal_date is not None
    uncertain_date = re.compile(
        r"(?:[,;]\s*)?(?:я\s+)?не\s+(?:знаю|уверен\w*)\b"
        r"[^.!?;]{0,100}\b(?:когда|дат\w*|врем\w*)\b[^.!?;]*",
        re.IGNORECASE,
    )
    stale_projection = re.compile(
        r"(?:[,;]\s*)?\bпровер\w*\s+(?:журнал\w*|баз\w*|проекц\w*)\b"
        r"[^.!?;]*\bне\s+подтверд\w*\b[^.!?;]*\b(?:дат\w*|врем\w*|обнов\w*|примен\w*)\b[^.!?;]*|"
        r"(?:[,;]\s*)?\bне\s+подтверд\w*\b[^.!?;]*\b(?:дат\w*|врем\w*|обнов\w*|примен\w*)\b[^.!?;]*",
        re.IGNORECASE,
    )
    precise_time_disclaimer = re.compile(
        r"(?:[,;]\s*)?\bточн\w*\s+врем\w*\b[^.!?;]{0,50}"
        r"\bне\s+(?:подстав\w*|внес\w*|запис\w*|указ\w*)\b[^.!?;]*",
        re.IGNORECASE,
    )
    unconfirmed_save = re.compile(
        r"(?:[,;]\s*)?\bподтвержден\w*[^.!?;]{0,60}\bнет\b|"
        r"(?:[,;]\s*)?\bне\s+подтвержден\w*[^.!?;]*\b(?:сохран|запис|примен|обнов)\w*",
        re.IGNORECASE,
    )
    standalone_unconfirmed_save = re.compile(
        r"\s*(?:\*\*|__|\*|_)?сохранение\s+не\s+подтверждено"
        r"(?:\*\*|__|\*|_)?\s*[.!?]?\s*",
        re.IGNORECASE,
    )
    food_terms = {
        term.casefold()
        for item in annotation.items
        for term in re.findall(r"[\w-]+", item.name)
        if len(term) >= 4
        and term.casefold() not in {"pack", "упаковка", "упаковки"}
        and not term.casefold().endswith(("ый", "ий", "ая", "ое", "ее", "ые", "ие"))
    }
    applied_date_claim = re.compile(
        r"(?:[,;]\s*)?\b(?:дат\w*|врем\w*)\b[^.!?;]{0,50}"
        r"\b(?:обновил\w*|применил\w*|отразил\w*|перенес\w*)\b[^.!?;]*",
        re.IGNORECASE,
    )
    cleaned_answer = answer.strip()
    for pattern in (
        stale_projection,
        precise_time_disclaimer,
        unconfirmed_save,
        applied_date_claim,
    ):
        cleaned_answer = pattern.sub("", cleaned_answer)
    sentence_parts = re.split(r"(?<=[.!?])(?=\s|$)", cleaned_answer)
    for index, sentence in enumerate(sentence_parts):
        current_food_missing = food_terms and any(
            re.fullmatch(
                rf"\s*в\s+журнал\w*\s+{re.escape(term)}\s+пока\s+не\s+"
                r"(?:появил\w*|добавил\w*|отразил\w*)\s*[.!?]?\s*",
                sentence,
                re.IGNORECASE,
            )
            or re.fullmatch(
                rf"\s*в\s+журнал\w*\s+{re.escape(term)}\s+пока\s+не\s+"
                r"(?:появил\w*|добавил\w*|отразил\w*)\s*[—–-]\s*"
                r"(?:\*\*|__|\*|_)?сохранение\s+не\s+подтверждено"
                r"(?:\*\*|__|\*|_)?\s*[.!?]?\s*",
                sentence,
                re.IGNORECASE,
            )
            for term in food_terms
        )
        if standalone_unconfirmed_save.fullmatch(sentence) or current_food_missing:
            sentence_parts[index] = ""
    cleaned_answer = "".join(sentence_parts)
    if known_meal_time:
        cleaned_answer = uncertain_date.sub("", cleaned_answer)
    cleaned_answer = re.sub(r"\s+([,;.!?])", r"\1", cleaned_answer)
    cleaned_answer = re.sub(r"([.!?])\s*[,;]", r"\1", cleaned_answer)
    # A dated committed receipt supersedes this standalone status left by an
    # earlier model draft. Keep the same wording for genuinely undated meals.
    stale_undated_status = re.compile(
        r"Записано;\s*приём пищи пока не привязан к дате\.", re.IGNORECASE
    )
    if known_meal_time:
        cleaned_answer = re.sub(
            r"(?<=[.!?])\s+Записано;\s*приём пищи пока не привязан к дате\.\s*$",
            "",
            cleaned_answer,
            flags=re.IGNORECASE,
        )
        if stale_undated_status.fullmatch(cleaned_answer.strip()):
            cleaned_answer = ""
        cleaned_answer = re.sub(r"(?<!\.)\.\.(?!\.)", ".", cleaned_answer)
    cleaned = []
    for sentence in re.split(r"(?<=[.!?])\s+", cleaned_answer):
        if known_meal_time and stale_undated_status.fullmatch(sentence.strip()):
            continue
        clauses = re.split(r"[,;]|\s+но\s+", sentence, flags=re.IGNORECASE)
        compound_failure = bool(storage.search(sentence) and failure.search(sentence))
        if not compound_failure:
            if sentence.strip():
                cleaned.append(sentence.strip())
            continue
        for part in clauses:
            if empty_store.search(part):
                continue
            if storage.search(part) and failure.search(part):
                # Remove the assertion itself, not its object/complement:
                # e.g. retain "two pears: about 120 kcal" in a failed-save
                # pre-append sentence once the authoritative append wins.
                part, substitutions = failed_storage_action.subn("", part)
                if substitutions:
                    part = part.strip(" ,;:—-.")
                    if storage.fullmatch(part):
                        part = ""
                else:
                    # Some failure phrasing puts the storage noun after the
                    # failure verb. Keep the existing conservative clause
                    # removal when no specific failed action is identifiable.
                    part = ""
            if part.strip():
                cleaned.append(part.strip())
    content = " ".join(part for part in cleaned if part).strip()
    status = (
        "Записано. Баланс обновляется."
        if annotation.meal_at is not None or annotation.meal_date is not None
        else "Записано; приём пищи пока не привязан к дате."
    )
    return f"{content}\n{status}" if content else status


def _committed_nutrition_reply(
    annotation: NutritionAnnotationV2,
    *,
    status: str,
    assistant_content: str | None = None,
) -> str:
    """Keep receipt content while applying the committed observation status."""
    if annotation.record_type != "meal_observation":
        return status
    if assistant_content is not None:
        content = re.sub(
            r"(?:\n)?(?:Записано\. Баланс обновляется\.|"
            r"Записано; приём пищи пока не привязан к дате\.)\s*$",
            "",
            assistant_content.strip(),
            flags=re.IGNORECASE,
        ).strip()
        return _append_nutrition_saved_status(content, annotation) if content else status
    minimum = annotation.energy_kcal_min
    maximum = annotation.energy_kcal_max
    best = annotation.energy_kcal_best
    if minimum is not None and maximum is not None:
        amount = (
            f"Примерно {minimum:g}–{maximum:g} ккал."
            if minimum < maximum else f"Примерно {minimum:g} ккал."
        )
    elif best is not None:
        amount = f"Примерно {best:g} ккал."
    elif minimum is not None:
        amount = f"Не менее {minimum:g} ккал."
    elif maximum is not None:
        amount = f"Не более {maximum:g} ккал."
    else:
        return status
    return f"{amount}\n{status}"


def _receipt_has_consumed_nutrition(metadata: Mapping[str, object]) -> bool:
    trace = metadata.get("decision_trace")
    annotations = trace.get("annotations") if isinstance(trace, Mapping) else None
    nutrition = annotations.get("nutrition") if isinstance(annotations, Mapping) else None
    if not isinstance(nutrition, Mapping) or nutrition.get("schema_version") != 2:
        return False
    try:
        validated = NutritionAnnotationV2.model_validate(nutrition)
    except (TypeError, ValueError):
        return False
    return (
        validated.record_type == "meal_observation"
        and validated.consumption_status == "consumed"
    )


def _record_consumed_photo_occurrence(
    history: list[ConversationMessage],
    binding: Mapping[str, object],
    *,
    receipt_event_id: str,
    client_op_id: str,
    append_source_message_id: str,
) -> bool:
    """Persist one uniquely matched, receipt-proven consumption on its photo ref."""
    attachment_id = binding.get("attachment_id")
    if (
        type(binding.get("schema_version")) is not int
        or binding.get("schema_version") != 1
        or binding.get("is_private") is not True
        or binding.get("is_forwarded") is not False
        or binding.get("is_group") is not False
        or not isinstance(receipt_event_id, str) or not receipt_event_id
        or not isinstance(client_op_id, str) or not client_op_id
        or not isinstance(append_source_message_id, str) or not append_source_message_id
    ):
        return False
    expected = {
        "principal": binding.get("source_principal"),
        "chat_id": binding.get("chat_id"),
        "session_key": binding.get("session_key"),
        "gateway_session_id": binding.get("gateway_session_id"),
        "source_message_id": binding.get("source_message_id"),
    }
    if not isinstance(attachment_id, str) or not all(
        isinstance(value, str) and value for value in expected.values()
    ):
        return False
    matches = [
        block
        for message in history
        for block in message.content
        if isinstance(block, AttachmentRefBlock)
        and block.attachment_id == attachment_id
        and isinstance(block.source_provenance, dict)
        and all(block.source_provenance.get(key) == value for key, value in expected.items())
    ]
    if len(matches) != 1:
        return False
    for message in history:
        for block in message.content:
            provenance = block.source_provenance if isinstance(block, AttachmentRefBlock) else None
            occurrences = provenance.get("consumed_occurrences") if isinstance(provenance, dict) else None
            if not isinstance(occurrences, list):
                continue
            for prior in occurrences:
                if not isinstance(prior, Mapping):
                    continue
                if (
                    prior.get("receipt_event_id") == receipt_event_id
                    and prior.get("client_op_id") == client_op_id
                    and block.attachment_id != attachment_id
                ):
                    return False
    provenance = matches[0].source_provenance
    assert isinstance(provenance, dict)
    occurrences = provenance.get("consumed_occurrences")
    if not isinstance(occurrences, list):
        occurrences = []
    occurrence = {
        "append_source_message_id": append_source_message_id,
        "receipt_event_id": receipt_event_id,
        "client_op_id": client_op_id,
    }
    if occurrence not in occurrences:
        provenance["consumed_occurrences"] = [*occurrences, occurrence]
    return True


def _current_verified_photo_occurrence_source(
    history: list[ConversationMessage], *, message: InboundMessage,
    receipt_metadata: Mapping[str, object], session_key: str,
) -> dict[str, object] | None:
    """Bind a just-accepted append receipt to its unique authenticated inbound photo ref."""
    source_principal = receipt_metadata.get("source_principal")
    gateway_session_id = receipt_metadata.get("gateway_session_id")
    append_source_id = receipt_metadata.get("source_message_id")
    if (message.channel != "telegram" or not is_private_message(message)
            or not (message.media and any(_is_image_attachment(item) for item in message.media))
            or not all(isinstance(item, str) and item for item in
                       (source_principal, gateway_session_id, append_source_id))
            or source_principal != f"telegram:{canonical_principal('telegram', str(message.sender_id))}"
            or _normalize_source_message_ref(message.metadata.get("message_id")) != append_source_id
            or receipt_metadata.get("is_group") is not False
            or receipt_metadata.get("is_forwarded") is not False):
        return None

    coalesced_sources = message.metadata.get("_coalesced_media_sources")
    coalesced_authorized = (
        message.metadata.get("_coalesced_media_provenance_authority")
        is COALESCED_ATTACHMENT_PROVENANCE_AUTHORITY
    )
    matches: list[dict[str, object]] = []
    for historical in history:
        for block in historical.content:
            if not isinstance(block, AttachmentRefBlock) or not isinstance(block.source_provenance, Mapping):
                continue
            provenance = block.source_provenance
            photo_source_id = provenance.get("source_message_id")
            received_at = provenance.get("received_at")
            if (provenance.get("schema_version") != 1
                    or provenance.get("channel") != "telegram"
                    or provenance.get("principal") != source_principal
                    or provenance.get("chat_id") != str(message.chat_id)
                    or provenance.get("session_key") != session_key
                    or provenance.get("gateway_session_id") != gateway_session_id
                    or provenance.get("is_group") is not False
                    or provenance.get("is_forwarded") is not False
                    or provenance.get("timestamp_authority") != "inbound_event_timestamp"
                    or not isinstance(photo_source_id, str) or not photo_source_id
                    or not isinstance(received_at, str) or not received_at
                    or provenance.get("append_source_message_id") != append_source_id):
                continue
            try:
                received = datetime.fromisoformat(received_at.replace("Z", "+00:00"))
            except ValueError:
                continue
            if received.tzinfo is None or received.utcoffset() is None:
                continue
            if (photo_source_id == append_source_id
                    and _trusted_utc_iso(received) != receipt_metadata.get("received_at")):
                continue
            if photo_source_id != append_source_id:
                if not coalesced_authorized or not isinstance(coalesced_sources, list):
                    continue
                matching_sources = []
                for item in coalesced_sources:
                    if not isinstance(item, Mapping):
                        continue
                    listed_id = _normalize_source_message_ref(item.get("source_message_id"))
                    listed_time = item.get("received_at")
                    if not isinstance(listed_time, str):
                        continue
                    try:
                        listed_dt = datetime.fromisoformat(listed_time.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                    if (listed_id == photo_source_id and listed_dt.tzinfo is not None
                            and listed_dt.utcoffset() is not None
                            and _trusted_utc_iso(listed_dt) == _trusted_utc_iso(received)):
                        matching_sources.append(item)
                if len(matching_sources) != 1:
                    continue
            matches.append({
                "schema_version": 1, "tenant_id": receipt_metadata.get("tenant_id"),
                "source_principal": source_principal, "gateway_session_id": gateway_session_id,
                "source_message_id": photo_source_id, "append_source_message_id": append_source_id,
                "attachment_id": block.attachment_id, "received_at": _trusted_utc_iso(received),
                "chat_id": str(message.chat_id), "session_key": session_key,
                "is_private": True, "is_forwarded": False, "is_group": False,
            })
    return matches[0] if len(matches) == 1 else None


def _reminder_wellness_tenants(config) -> WellnessTenantResolver:
    """Build the reminders' server-side principal -> wellness tenant mapping.

    Owner principals may carry ``id|username``; only the canonical numeric
    prefix is trusted (same canonicalization GatewayConfig validates against).
    """
    owners = tuple(
        canonical
        for owner in config.owner_principals
        if (canonical := str(owner).strip().split("|", 1)[0].strip())
    )
    return WellnessTenantResolver(
        owner_principals=owners,
        family_principals=dict(config.family_principals),
        enabled_tenants=tuple(config.enabled_memory_tenants),
    )


_SCHEDULER_SENDER = "__scheduler__"


@dataclass(frozen=True)
class _TrustedReminderWellnessAdmission:
    """Immutable result of validating one internal scheduler delivery."""

    reminder_id: str
    wellness_tenant: str
    wellness_principal: str
    private_chat_id: str


def _is_real_user_turn(message: InboundMessage) -> bool:
    """Return whether todo lifecycle rules may act on this inbound turn."""
    metadata = message.metadata or {}
    return not (bool(metadata.get("_synthetic")) or message.sender_id == _SCHEDULER_SENDER)


def _todo_unresolved_items(snapshot: list[dict[str, str]]) -> list[dict[str, str]]:
    return [item for item in snapshot if item.get("status") in {"pending", "in_progress"}]


def _todo_unresolved_text(snapshot: list[dict[str, str]]) -> str:
    items = _todo_unresolved_items(snapshot)
    if not items:
        return "none"
    return "; ".join(
        f"{item.get('content', '<unnamed>')} ({item.get('status', 'unknown')})" for item in items
    )


def _trusted_reminder_wellness(
    message: InboundMessage,
) -> _TrustedReminderWellnessAdmission | None:
    """Return a scheduler-stamped wellness-only reminder scope.

    The scheduler supplies the authenticated creator principal and the runtime
    revalidates its tenant mapping before exposing wellness data.
    """
    if str(message.channel).strip().lower() != "telegram":
        return None
    metadata = message.metadata or {}
    if message.sender_id != _SCHEDULER_SENDER or metadata.get("_synthetic") is not True:
        return None
    reminder_id = metadata.get("_reminder_id")
    created_by = metadata.get("_reminder_created_by")
    principal = metadata.get("_reminder_wellness_principal")
    tenant = metadata.get("_reminder_wellness_tenant")
    if not all(isinstance(item, str) for item in (reminder_id, created_by, principal, tenant)):
        return None
    reminder_id = reminder_id.strip()
    created_by = created_by.strip()
    principal = principal.strip()
    tenant = tenant.strip()
    if not reminder_id or not created_by or not principal or not tenant:
        return None
    chat_id = str(message.chat_id).strip()
    canonical_created_by = canonical_principal("telegram", created_by)
    canonical_principal_id = canonical_principal("telegram", principal)
    if (
        not canonical_created_by.isascii()
        or not canonical_created_by.isdigit()
        or canonical_created_by.startswith("0")
        or canonical_created_by != chat_id
        or not chat_id.isascii()
        or not chat_id.isdigit()
        or chat_id.startswith("0")
        or not canonical_principal_id.isdigit()
        or canonical_principal_id != canonical_created_by
        or principal != created_by
        or message.session_key_override != f"telegram:reminder:{reminder_id}"
        or _is_group_message(message)
    ):
        return None
    return _TrustedReminderWellnessAdmission(
        reminder_id=reminder_id,
        wellness_tenant=tenant,
        wellness_principal=principal,
        private_chat_id=chat_id,
    )


class OhmoSessionRuntimePool:
    """Maintain one runtime bundle per chat/thread session."""

    def __init__(
        self,
        *,
        cwd: str | Path,
        workspace: str | Path | None = None,
        provider_profile: str,
        model: str | None = None,
        max_turns: int | None = None,
        effort: str | None = None,
        create_feishu_group: CreateFeishuGroup | None = None,
        publish_group_welcome: PublishGroupWelcome | None = None,
        contact_store: ContactStore | None = None,
        default_tz: str = DEFAULT_REMINDER_TZ,
        reminder_max_per_chat: int = DEFAULT_REMINDER_MAX_PER_CHAT,
    ) -> None:
        self._cwd = str(Path(cwd).resolve())
        self._workspace = workspace
        self._provider_profile = provider_profile
        self._model = model
        self._max_turns = max_turns
        self._effort = effort
        self._create_feishu_group = create_feishu_group
        self._publish_group_welcome = publish_group_welcome
        self._contact_store = contact_store
        self._default_tz = default_tz
        self._reminder_max_per_chat = reminder_max_per_chat
        self._workspace = initialize_workspace(workspace)
        self._gateway_config = load_gateway_config(self._workspace)
        self._session_backend = OhmoSessionBackend(self._workspace)
        self._attachment_store = self._session_backend.attachment_store
        self._todo_store = TodoStore(self._workspace)
        self._memory_store = MemoryStore(self._workspace)
        self._prompt_memory_backend = make_memory_backend(self._gateway_config, self._workspace)
        self._catalog_memory_backends: dict[tuple[str, str | None], CatalogMemoryBackend] = {}
        self._tenant_shadow_backends: dict[str, ShadowMemoryBackend] = {}
        self._judge_turn_counts: dict[str, int] = {}
        self._judge_tasks: dict[str, asyncio.Task] = {}
        self._reminder_store = ReminderStore(workspace=self._workspace)
        self._reminder_lock = asyncio.Lock()
        self._bundles: dict[str, RuntimeBundle] = {}
        self._gateway_config_generation = 0
        self._session_owner_principals: dict[str, str | None] = {}
        reaped = reap_stale_work_dirs(self._workspace)
        if reaped:
            logger.info("ohmo runtime reaped %d stale per-chat work dir(s) at startup", reaped)

    @property
    def active_sessions(self) -> int:
        return len(self._bundles)

    async def aclose(self) -> None:
        """Close resources owned by the shared prompt-memory backend."""
        close = getattr(self._prompt_memory_backend, "aclose", None)
        if not callable(close):
            return
        try:
            await close()
        except Exception:
            logger.warning("ohmo memory backend close failed", exc_info=True)

    def _remote_admin_allowed(self, command) -> bool:
        if not getattr(command, "remote_admin_opt_in", False):
            return False
        if not self._gateway_config.allow_remote_admin_commands:
            return False
        allowed = {
            str(name).strip().lower()
            for name in self._gateway_config.allowed_remote_admin_commands
            if str(name).strip()
        }
        return command.name.lower() in allowed

    def _handle_gateway_scoped_command(
        self, command_name: str, args: str
    ) -> tuple[str, bool] | None:
        lowered = command_name.lower()
        if lowered == "provider":
            result = handle_gateway_provider_command(args, workspace=self._workspace)
        elif lowered == "model":
            result = handle_gateway_model_command(args, workspace=self._workspace)
        else:
            return None
        if result[1]:
            self._gateway_config = load_gateway_config(self._workspace)
            self._provider_profile = self._gateway_config.provider_profile
            self._gateway_config_generation += 1
        return result

    async def get_bundle(
        self,
        session_key: str,
        latest_user_prompt: str | None = None,
        cwd: str | Path | None = None,
        include_todo: bool = True,
    ) -> RuntimeBundle:
        """Return an existing bundle or create a new one."""
        initial_memory_scope = self._resolve_turn_memory_scope(None)
        session_cwd = str(Path(cwd or self._cwd).expanduser().resolve())
        bundle = self._bundles.get(session_key)
        if bundle is not None:
            bundle_cwd = str(Path(getattr(bundle, "cwd", self._cwd)).resolve())
            if bundle_cwd != session_cwd:
                logger.info(
                    "ohmo runtime recreating session for cwd change session_key=%s old_cwd=%s new_cwd=%s",
                    session_key,
                    bundle_cwd,
                    session_cwd,
                )
                await close_runtime(bundle)
                self._bundles.pop(session_key, None)
            elif (
                getattr(bundle, "_gateway_config_generation", self._gateway_config_generation)
                != self._gateway_config_generation
            ):
                logger.info(
                    "ohmo runtime lazily refreshing stale gateway configuration session_key=%s session_id=%s",
                    session_key,
                    bundle.session_id,
                )
                return await self._refresh_bundle(
                    session_key,
                    bundle,
                    latest_user_prompt,
                    include_todo=include_todo,
                )
            else:
                logger.info(
                    "ohmo runtime reusing session session_key=%s session_id=%s prompt=%r",
                    session_key,
                    bundle.session_id,
                    _content_snippet(latest_user_prompt or ""),
                )
                return bundle

        snapshot = self._session_backend.load_latest_for_session_key(session_key)
        logger.info(
            "ohmo runtime creating session session_key=%s restored=%s prompt=%r",
            session_key,
            bool(snapshot),
            _content_snippet(latest_user_prompt or ""),
        )
        bundle = await build_runtime(
            cwd=session_cwd,
            model=self._model,
            max_turns=self._max_turns,
            effort=self._effort,
            system_prompt=build_ohmo_system_prompt(
                session_cwd,
                workspace=self._workspace,
                extra_prompt=None,
                include_ohmo_memory=False,
            ),
            active_profile=self._provider_profile,
            session_backend=self._session_backend,
            enforce_max_turns=True,  # cap each prompt at settings.max_turns by default (was unlimited)
            restore_messages=_sanitize_snapshot_messages(
                snapshot.get("messages") if snapshot else None
            ),
            restore_tool_metadata=_sanitize_group_command_metadata(
                snapshot.get("tool_metadata") if snapshot else None
            ),
            extra_skill_dirs=(str(get_skills_dir(self._workspace)),),
            extra_plugin_roots=(str(get_plugins_dir(self._workspace)),),
            memory_backend=create_memory_command_backend(
                self._workspace,
                backend_kind=self._gateway_config.memory_backend,
            ),
            include_project_memory=False,
            autodream_context=(
                self._autodream_context() if initial_memory_scope is not None else None
            ),
        )
        if snapshot and snapshot.get("session_id"):
            bundle.session_id = str(snapshot["session_id"])
        self._configure_attachment_boundary(bundle)
        self._register_gateway_tools(
            bundle,
            memory_engaged=initial_memory_scope is not None,
        )
        self._configure_turn_memory_surfaces(
            bundle,
            None,
            memory_scope=initial_memory_scope,
        )
        await start_runtime(bundle)
        bundle.engine.set_system_prompt(
            await self._runtime_system_prompt(
                bundle,
                latest_user_prompt,
                memory_scope=initial_memory_scope,
                include_todo=include_todo,
            )
        )
        if hasattr(bundle.engine, "set_cache_key"):
            bundle.engine.set_cache_key(bundle.session_id)
        setattr(bundle, "_gateway_config_generation", self._gateway_config_generation)
        logger.info(
            "ohmo runtime started session_key=%s session_id=%s restored_messages=%s",
            session_key,
            bundle.session_id,
            len(snapshot.get("messages") or []) if snapshot else 0,
        )
        self._bundles[session_key] = bundle
        return bundle

    async def camera_retained_attachment_history(
        self, *, since: datetime, until: datetime
    ) -> list[RetainedAttachmentEvidence]:
        """Read attachment refs from the one configured private session snapshot."""
        config = self._gateway_config.camera_ingress
        if (
            not config.enabled
            or not config.session_key
            or not config.principal
            or not config.chat_id
            or since.tzinfo is None
            or until.tzinfo is None
            or since > until
        ):
            raise ValueError("Camera retained-history scope is invalid")
        attachment_snapshot = getattr(
            self._session_backend, "load_camera_attachment_snapshot", None
        )
        if attachment_snapshot is None:
            raise ValueError("complete retained attachment snapshot traversal is unavailable")
        snapshot = attachment_snapshot(config.session_key)
        if snapshot is None:
            raise ValueError("retained private attachment history is unavailable")
        snapshot_session_id = snapshot["session_id"]
        evidence: list[RetainedAttachmentEvidence] = []
        verified_objects: dict[str, tuple[int, str, dict[str, object]]] = {}
        for message_index, raw in enumerate(snapshot["messages"]):
            message = ConversationMessage.model_validate(raw)
            if message.role != "user":
                continue
            refs = [block for block in message.content if isinstance(block, AttachmentRefBlock)]
            inline_images = [block for block in message.content if isinstance(block, ImageBlock)]
            if inline_images:
                raise ValueError("retained user image lacks a durable attachment reference")
            if not refs:
                continue
            message_fingerprints: list[dict[str, object]] = []
            message_time: datetime | None = None
            source: dict[str, object] | None = None
            timestamp_authority: str | None = None
            for ref in refs:
                provenance = ref.source_provenance
                legacy_timestamp = provenance is None
                if provenance is None:
                    # Legacy snapshots lack per-message source provenance. Scope
                    # this narrow fallback to the exact configured private owner
                    # session; object mtime is observation evidence only, never
                    # a receipt or consumption timestamp.
                    if (
                        config.session_key != f"telegram:{config.principal}"
                        or config.chat_id != config.principal
                    ):
                        raise ValueError("legacy attachment has no verifiable owner-local session")
                    received = None
                    authority = "owner_local_object_mtime_observed_retention"
                else:
                    if not isinstance(provenance, dict) or provenance.get("schema_version") != 1:
                        raise ValueError("retained attachment timestamp provenance is invalid")
                    if provenance.get("principal") != f"telegram:{config.principal}":
                        continue
                    if (
                        provenance.get("channel") != "telegram"
                        or provenance.get("chat_id") != config.chat_id
                    ):
                        continue
                    if provenance.get("session_key") != config.session_key:
                        raise ValueError("retained attachment belongs to a different session key")
                    if provenance.get("gateway_session_id") != snapshot_session_id:
                        raise ValueError("retained attachment belongs to a different snapshot session")
                    if provenance.get("is_group") is True or provenance.get("is_forwarded") is True:
                        continue
                    if provenance.get("is_group") is not False or provenance.get("is_forwarded") is not False:
                        raise ValueError("retained attachment private-source provenance is incomplete")
                    if provenance.get("timestamp_authority") != "inbound_event_timestamp":
                        raise ValueError("retained attachment timestamp authority is invalid")
                    received_at = provenance.get("received_at")
                    if not isinstance(received_at, str) or len(received_at) > 64:
                        raise ValueError("retained attachment timestamp is missing")
                    try:
                        received = datetime.fromisoformat(received_at)
                    except ValueError as error:
                        raise ValueError("retained attachment timestamp is malformed") from error
                    if received.tzinfo is None or received.utcoffset() is None:
                        raise ValueError("retained attachment timestamp is not timezone-aware")
                    received = received.astimezone(timezone.utc)
                    authority = "inbound_event_timestamp"
                    if not since <= received <= until:
                        continue
                if legacy_timestamp:
                    received = self._attachment_store.observed_object_mtime(
                        ref.attachment_id, max_bytes=10 * 1024 * 1024
                    )
                    if not since <= received <= until:
                        continue
                if not 0 < ref.byte_size <= 10 * 1024 * 1024:
                    raise ValueError("retained attachment exceeds its byte bound")
                cached = verified_objects.get(ref.attachment_id)
                if cached is None:
                    stored = self._attachment_store.load_image(
                        ref.attachment_id, max_bytes=10 * 1024 * 1024
                    )
                    if (
                        stored.ref.byte_size != ref.byte_size
                        or stored.ref.media_type != ref.media_type
                        or hashlib.sha256(stored.data).hexdigest() != ref.attachment_id
                    ):
                        raise ValueError("retained attachment ref does not match stored bytes")
                    if legacy_timestamp and stored.observed_retention_at != received:
                        raise ValueError("legacy attachment changed during observed-mtime validation")
                    from ohmo.gateway.attachment_fingerprints import fingerprint_image_bytes

                    descriptor = fingerprint_image_bytes(stored.data)
                    fingerprint: dict[str, object] = {"sha256": ref.attachment_id}
                    if descriptor is not None and isinstance(descriptor.get("phash"), str):
                        fingerprint["phash"] = descriptor["phash"]
                        fingerprint["phash_algorithm"] = descriptor["phash_algorithm"]
                    verified_objects[ref.attachment_id] = (
                        stored.ref.byte_size, stored.ref.media_type, fingerprint
                    )
                else:
                    cached_size, cached_media, fingerprint = cached
                    if cached_size != ref.byte_size or cached_media != ref.media_type:
                        raise ValueError("repeated retained attachment ref conflicts with stored object")
                message_fingerprints.append(fingerprint)
                message_time = received
                source = provenance if isinstance(provenance, dict) else {}
                timestamp_authority = authority
            if not message_fingerprints:
                continue
            assert message_time is not None and source is not None
            target = message.event_id or (
                source.get("source_message_id") if isinstance(source, dict) else None
            )
            target_authority = "native_event_id"
            if not isinstance(target, str) or not target or len(target) > 256:
                # Old private snapshots can retain a user photo with no native
                # Telegram event id and no gateway provenance. Bind only a
                # local duplicate target to this exact configured owner/session,
                # snapshot, serialized message position, and verified ref group.
                # This is observed-retention identity, never source authentication
                # or evidence of the original message/album identity.
                if any(ref.source_provenance is not None for ref in refs):
                    raise ValueError("retained attachment has no stable source identity")
                identity = {
                    "scope": "legacy_retained_ref_group_v1",
                    "principal": f"telegram:{config.principal}",
                    "session_key": config.session_key,
                    "snapshot_session_id": snapshot_session_id,
                    "snapshot_identity": snapshot["snapshot_identity"],
                    "message_index": message_index,
                    "attachment_ids": [ref.attachment_id for ref in refs],
                }
                target = "retained:" + hashlib.sha256(
                    json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                target_authority = "owner_local_observed_retention_ref_group"
            if not isinstance(target, str) or not target or len(target) > 256:
                raise ValueError("retained attachment has no stable source identity")
            metadata: dict[str, object] = {
                "role": "user",
                "source_principal": f"telegram:{config.principal}",
                "is_group": False,
                "is_forwarded": False,
                "attachment_fingerprints": message_fingerprints,
                "source_snapshot_session_id": snapshot_session_id,
                "source_session_key": config.session_key,
                "source_chat_id": config.chat_id,
                "source_channel": "telegram",
                "timestamp_authority": timestamp_authority,
                "target_authority": target_authority,
                "received_at": message_time.isoformat(),
            }
            evidence.append(
                RetainedAttachmentEvidence(
                    id=target,
                    peer_id=f"telegram:{config.principal}",
                    session_id=snapshot_session_id,
                    metadata=metadata,
                    created_at=message_time,
                )
            )
        return evidence

    async def reset_session(self, session_key: str) -> bool:
        """Hard-reset a session for /new: drop the in-memory bundle and the
        persisted per-session-key 'latest' pointer, so the next message starts a
        brand-new conversation (new session_id, empty history). Returns True when
        there was a live bundle to drop."""
        bundle = self._bundles.pop(session_key, None)
        had_bundle = bundle is not None
        session_id = getattr(bundle, "session_id", None)
        if isinstance(session_id, str):
            self._session_owner_principals.pop(session_id, None)
        # Reset the judge cadence + cancel any in-flight judge for this session.
        self._judge_turn_counts.pop(session_key, None)
        judge_task = self._judge_tasks.pop(session_key, None)
        if judge_task is not None:
            judge_task.cancel()
        if bundle is not None:
            try:
                await close_runtime(bundle)
            except Exception:
                logger.warning(
                    "ohmo runtime reset close failed session_key=%s", session_key, exc_info=True
                )
        clear = getattr(self._session_backend, "clear_session_key", None)
        if clear is not None:
            try:
                clear(session_key)
            except Exception:
                logger.warning(
                    "ohmo runtime reset clear-snapshot failed session_key=%s",
                    session_key,
                    exc_info=True,
                )
        # No TODO file to wipe: to-do lists are per-session_id (see TodoStore),
        # and clearing the snapshot above means the next message mints a FRESH
        # session_id → a brand-new empty list. The previous conversation's list
        # file is kept on disk (the agent can still be pointed back at it).
        # Wipe this chat's scratch/work dir so /new starts on a clean cwd (it is
        # recreated lazily on the next write).
        try:
            clear_session_work_dir(session_key, self._workspace)
        except Exception:
            logger.warning(
                "ohmo runtime reset work-dir clear failed session_key=%s",
                session_key,
                exc_info=True,
            )
        logger.info(
            "ohmo runtime session reset session_key=%s had_bundle=%s", session_key, had_bundle
        )
        return had_bundle

    def _bind_session_owner(
        self,
        message: InboundMessage,
        session_key: str,
        turn_ctx: TurnContext,
    ) -> str | None:
        """Record a principal only when normal gateway routing proves the binding."""
        if turn_ctx.camera_authorized and message.sender_id == "__camera__":
            return self._session_owner_principals.get(turn_ctx.session_id)
        candidate = None
        if (
            message.session_key_override is None
            and session_key_for_message(message) == session_key
            and turn_ctx.principal
        ):
            candidate = turn_ctx.principal

        session_id = turn_ctx.session_id
        if session_id not in self._session_owner_principals:
            self._session_owner_principals[session_id] = candidate
        elif candidate is not None:
            current = self._session_owner_principals[session_id]
            if current is None:
                self._session_owner_principals[session_id] = candidate
            elif current != candidate:
                self._session_owner_principals[session_id] = None
        return self._session_owner_principals[session_id]

    def _trusted_camera_answer_time(
        self, message: InboundMessage, *, session_key: str, turn_ctx: TurnContext
    ) -> datetime | None:
        camera = self._gateway_config.camera_ingress
        if (
            not camera.enabled
            or not turn_ctx.camera_authorized
            or turn_ctx.principal != camera.principal
            or canonical_principal(message.channel, message.sender_id) != camera.principal
            or message.metadata.get("_camera_answer") != "yes"
            or message.metadata.get("_camera_authority") is not CAMERA_AUTHORITY
            or message.channel != "telegram"
            or str(message.chat_id) != camera.chat_id
            or session_key != camera.session_key
            or not message.media
        ):
            return None
        ingress = getattr(self, "_camera_ingress", None)
        return ingress.trusted_capture_time_for_answer(message) if ingress is not None else None

    def _trusted_user_photo_time(
        self, message: InboundMessage, *, turn_ctx: TurnContext,
        history: list[ConversationMessage] | None = None,
    ) -> datetime | None:
        """Return the authenticated participant's original image message time."""
        metadata = message.metadata or {}
        principal = canonical_principal(message.channel, message.sender_id)
        config = self._gateway_config
        owners = {
            canonical_principal("telegram", owner)
            for owner in getattr(config, "owner_principals", ())
            if str(owner).strip()
        }
        family_principals = getattr(config, "family_principals", {})
        family_tenant = family_principals.get(principal)
        owner_principal = turn_ctx.is_owner is True and principal in owners
        family_principal = False
        if family_tenant is not None:
            scope = self._resolve_turn_memory_scope(turn_ctx)
            family_principal = scope is not None and scope.private_tenant == family_tenant
        configured_private_principal = owner_principal or family_principal
        content = message.content or ""
        if (
            message.sender_id == "__camera__"
            or message.channel != "telegram"
            or not configured_private_principal
            or not turn_ctx.is_private
            or not is_private_message(message)
            or turn_ctx.channel.strip().lower() != "telegram"
            or canonical_principal(turn_ctx.channel, turn_ctx.principal) != principal
            or turn_ctx.is_forwarded
            or metadata.get("_camera_authority") is CAMERA_AUTHORITY
            or not _normalize_source_message_ref(metadata.get("message_id"))
            or not any(_is_image_attachment(path) for path in message.media or [])
            or OhmoSessionRuntimePool._known_user_photo_repeat(message, history or [])
            or (
                _USER_PHOTO_ADVISORY_RE.search(content)
                and not _contains_nutrition_record_intent(content)
            )
        ):
            return None
        coalesced_sources = (
            metadata.get("_coalesced_media_sources")
            if metadata.get("_coalesced_media_provenance_authority")
            is COALESCED_ATTACHMENT_PROVENANCE_AUTHORITY
            else None
        )
        if isinstance(coalesced_sources, list):
            for media_index, path in enumerate(message.media or []):
                if not _is_image_attachment(path) or media_index >= len(coalesced_sources):
                    continue
                source = coalesced_sources[media_index]
                if not isinstance(source, Mapping):
                    continue
                source_id = _normalize_source_message_ref(source.get("source_message_id"))
                source_time = source.get("received_at")
                if source_id is not None and isinstance(source_time, str):
                    normalized = _trusted_utc_iso(source_time)
                    if normalized is not None:
                        return datetime.fromisoformat(normalized)
            return None
        return _trusted_inbound_event_time(message)

    @staticmethod
    def _known_user_photo_repeat(
        message: InboundMessage, history: list[ConversationMessage]
    ) -> bool:
        """Recognize exact bytes already seen in this trusted private owner session."""
        current = compute_attachment_fingerprints(message.media or [])
        current_hashes = {
            item.get("sha256") for item in current if isinstance(item, Mapping)
        }
        if not current_hashes:
            return False
        principal = f"{message.channel}:{canonical_principal(message.channel, message.sender_id)}"
        for prior in history:
            if prior.role != "user":
                continue
            for block in prior.content:
                if not isinstance(block, AttachmentRefBlock):
                    continue
                provenance = block.source_provenance
                if (
                    not isinstance(provenance, Mapping)
                    or provenance.get("principal") != principal
                    or provenance.get("chat_id") != str(message.chat_id)
                    or provenance.get("is_group") is not False
                    or provenance.get("is_forwarded") is not False
                    or provenance.get("timestamp_authority") != "inbound_event_timestamp"
                ):
                    continue
                if block.attachment_id in current_hashes:
                    return True
        return False

    @staticmethod
    def _with_user_photo_context(
        prompt: str, sent_at: datetime | None, *, known_repeat: bool = False
    ) -> str:
        if known_repeat:
            return (
                prompt + "\n\n# Previously seen user photo\n"
                "This exact image was already sent by the owner in this private session. "
                "Treat it as the existing photo/source; do not record another meal unless "
                "the user explicitly says this is a new, separate consumption. Do not use "
                "the resent image's send time as a meal-time default."
            )
        if sent_at is None:
            return prompt
        return (
            prompt + "\n\n# Trusted user photo intake\n"
                "The authenticated participant sent the attached image directly at "
            f"{sent_at.isoformat()}. Treat a clear food photo as a direct request to log "
            "the pictured food as consumed; do not ask whether it was eaten. If the image "
            "does not clearly establish food or its portion, ask one useful clarification. "
            "For nonfood images, or if the accompanying text asks only for identification, "
            "advice, a recipe, or a calorie estimate, do not create a meal. An "
            "explicit meal date or time in the user's text overrides the photo send time. "
                "Use the gateway's trusted photo send time as the default only when no explicit "
                "alternative was given. This is a gateway send-date convention, not evidence "
                "that the food was physically eaten at that exact instant. Claim persistence "
                "only after a durable append receipt."
        )

    @staticmethod
    def _with_camera_answer_context(prompt: str, capture_time: datetime | None) -> str:
        if capture_time is None:
            return prompt
        return (
            prompt + "\n\n# Verified Camera answer for this turn\n"
            "The gateway bound the current owner's affirmative consumption answer to the "
            "attached Camera photo and verified its capture time. The earlier Camera "
            "analysis-only instruction applied to the earlier photo turn; this is the "
            "owner's confirmed-consumption turn. Use the current user's stated food and "
            "quantity over ambiguous image inference. An independently stated owner "
            "quantity or actually selected portion is authoritative. A whole-portion "
            "confirmation does not turn a count you guessed in an option into an "
            "owner-stated quantity. Earlier assistant analysis and any count it proposed "
            "are provisional, not independent image or owner evidence. Before finalizing, "
            "verify each product count against the current original pixels, using the "
            "attached original or `load_conversation_image` when needed; distinguish cut "
            "sections of one unit from multiple complete units. Count whole products, not "
            "pieces cut from one. For "
            "continuous foods without an owner-stated weight or measure, infer a plausible "
            "served cooked edible mass or household measure from the visible portion and "
            "useful co-visible scale, then estimate energy from a matching typical kcal per "
            "unit. Record the estimated quantity and its assumptions and uncertainty in the "
            "nutrition trace. Treat the point estimate as the most likely central portion, "
            "not a precautionary upper bound; use a range when uncertainty supports one. Do "
            "not assume a maximal portion or added fats. For cooked food, use the served "
            "cooked weight and matching preparation and unit; "
            "keep each item's kcal consistent with its food and quantity, and make the "
            "total equal the item sum. Include only food supported by the confirmed portion: "
            "visible in the photo, identified by readable package labeling, or explicitly "
            "stated by the owner. A readable label may identify hidden package contents; "
            "do not include adjacent unselected packages or unseen oil or sauce. Before "
            "answering, call `trace` "
            "with a valid `trace_finalization` payload and `annotations.nutrition` using "
            "the calory skill's schema v2: `schema_version` 2, `record_type` `meal_observation`, "
            "`consumption_status` `consumed`, and `basis` including `image`. Estimate "
            "at least one total energy kcal field from the food and quantity; do not "
            "invent a fixed calorie value. "
            "When a follow-up quantity depends on details of the earlier original photo, use "
            "`load_conversation_image` to retrieve that retained source before estimating. "
            "If the original is unavailable or the visible image is only a crop, preserve the "
            "quantity uncertainty and ask one useful question; never treat a partial crop as "
            "a confirmed whole plate or fabricate calories. "
            "Set `meal_at` and `meal_date` to null in the model payload: the gateway "
            "stamps the authoritative capture time into the validated annotation. "
            "Use a valid, unique `trace_event_id` for the finalization. Only claim the "
            "meal was recorded after a trusted durable append receipt."
        )

    @staticmethod
    def _with_camera_correction_context(prompt: str) -> str:
        return (
            prompt + "\n\n# Verified Camera meal denial\n"
            "The authenticated owner explicitly denied consumption of the exact previously "
            "committed Camera meal. Finalize only a schema-v2 `meal_correction` whose "
            "`consumption_status` is `not_consumed` and whose `changed_fields` includes "
            "`consumption_status`. Do not create another meal or infer a different target; "
            "the gateway supplies the previously committed source and event target."
        )

    @classmethod
    def _with_camera_turn_context(
        cls, prompt: str, message: InboundMessage, capture_time: datetime | None
    ) -> str:
        if message.metadata.get("_camera_context_meal_target") is _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY:
            selected = message.metadata.get("_selected_source_binding")
            binding = (
                selected[1]
                if isinstance(selected, tuple) and len(selected) == 2
                and selected[0] is _SELECTED_SOURCE_AUTHORITY
                and isinstance(selected[1], Mapping)
                else None
            )
            items = binding.get("context_items") if isinstance(binding, Mapping) else None
            if binding is None or not isinstance(items, list):
                return prompt
            rendered_items = []
            for item in items:
                if not isinstance(item, Mapping):
                    return prompt
                name, quantity = item.get("name"), item.get("quantity_text")
                if not isinstance(name, str) or not isinstance(quantity, str):
                    return prompt
                rendered_items.append(f"{name} — {quantity}")
            if not rendered_items:
                return prompt
            received_at = message.timestamp
            trusted_timestamp = (
                received_at.astimezone(timezone.utc).isoformat()
                if isinstance(received_at, datetime)
                and received_at.tzinfo is not None
                and received_at.utcoffset() is not None
                else None
            )
            target_kind = message.metadata.get("_camera_context_meal_target_kind")
            if target_kind == "item_addition":
                instruction = (
                    "The owner is adding an item to this same meal. Preserve the listed current "
                    "composition, add only the item they requested, and finalize a sparse "
                    "schema-v2 `meal_correction` with `items` in `changed_fields`. Do not "
                    "append another meal observation or alter consumption status."
                )
            elif target_kind == "package_label":
                instruction = (
                    "The owner supplied the package mass and label kcal per 100 g for this "
                    "already-saved meal. Correct the matching packaged item from that label "
                    "and package mass, preserve every other food component and the meal date, "
                    "and recalculate the meal totals. Finalize a schema-v2 `meal_correction` "
                    "with `items` and each recalculated energy field in `changed_fields`. "
                    "Do not ask for another consumption confirmation, create a new meal "
                    "observation, or alter consumed status."
                )
            else:
                instruction = (
                    "The owner explicitly reports when this same meal was eaten. Finalize a "
                    "sparse schema-v2 `meal_correction` that changes only the date/time the "
                    "owner supplied. Preserve current item composition, quantities, calories, "
                    "and consumed status; do not repeat an item confirmation or append another "
                    "meal observation. For a sparse ‘Съели утром’, use this message's trusted "
                    "timestamp to resolve the morning date."
                )
            return (
                prompt + "\n\n# Verified retained Camera meal context\n"
                "Camera ingress authenticated this private owner's message and bound it to "
                "the one retained consumed meal. This is trusted same-meal context even if "
                "the runtime conversation has restarted. The following items come from the "
                "latest retained correction projection, when one exists; they are the current "
                "composition, not a request to repeat the original photo analysis:\n"
                + "\n".join(f"- {item}" for item in rendered_items)
                + "\n"
                + (
                    f"Trusted owner-message time: {trusted_timestamp} (UTC).\n"
                    if trusted_timestamp is not None else ""
                )
                + instruction
                + " Do not expose internal source or receipt identifiers in the reply. "
                "Do not invent nutrition values; leave unsupported values unset."
            )
        if message.metadata.get("_camera_portion_correction") is CAMERA_AUTHORITY:
            previous = message.metadata.get("_camera_prior_portion_label")
            selected = message.metadata.get("native_keyboard_selected_label")
            if not isinstance(previous, str) or not isinstance(selected, str):
                return prompt
            return (
                prompt + "\n\n# Verified Camera portion correction\n"
                "The authenticated owner changed the selected portion for the same "
                "already-recorded Camera meal. The original saved selection was "
                f"{previous!r}; the owner's current native selection is {selected!r}. "
                "Use the retained original photo and correct only the existing meal. "
                "Finalize a sparse schema-v2 `meal_correction`; omit "
                "`consumption_status` so the saved consumed status stays unchanged. "
                "Include `items` and every recalculated nutrition field in "
                "`changed_fields`, and provide replacements only for masked fields. "
                "Keep the original meal date and time unchanged. Do not append a second "
                "meal observation or describe the old portion as saved again."
            )
        if message.metadata.get("_camera_context_hint") is CAMERA_AUTHORITY:
            prompt += (
                "\n\n# Camera food context only\n"
                "The attached original photo is retained context for the owner's short "
                "food-identification message. That message alone does not confirm eating "
                "or authorize a meal record. Use context only when the current text is "
                "clearly about the pictured food; answer unrelated requests normally. "
                "If food is identified but consumed quantity remains unknown, ask one "
                "useful quantity question. Do not infer calories or consumption."
            )
            return prompt
        prompt = cls._with_camera_answer_context(prompt, capture_time)
        if (
            message.metadata.get("_camera_authority") is CAMERA_AUTHORITY
            and message.metadata.get("_camera_answer") == "no"
            and message.metadata.get("_camera_correction") is CAMERA_AUTHORITY
        ):
            prompt = cls._with_camera_correction_context(prompt)
        return prompt

    @staticmethod
    def _camera_final_delivery_metadata(message: InboundMessage) -> dict[str, object]:
        metadata = message.metadata
        if (
            metadata.get("_camera_authority") is not CAMERA_AUTHORITY
            or (
                metadata.get("_camera_answer") not in {"yes", "no"}
                and metadata.get("_camera_context_question")
                is not CAMERA_CONTEXT_QUESTION_AUTHORITY
            )
            or not isinstance(metadata.get("_camera_candidate_id"), str)
        ):
            return {}
        result: dict[str, object] = {
            "_camera_authority": CAMERA_AUTHORITY,
            "_camera_candidate_id": metadata["_camera_candidate_id"],
            "_camera_turn_id": metadata.get("_camera_turn_id"),
            "_camera_final": CAMERA_AUTHORITY,
        }
        if metadata.get("_camera_correction") is CAMERA_AUTHORITY:
            result["_camera_correction"] = CAMERA_AUTHORITY
        if metadata.get("_camera_clarification_final") is CAMERA_AUTHORITY:
            result["_camera_clarification_final"] = CAMERA_AUTHORITY
        return result

    async def stream_message(self, message: InboundMessage, session_key: str):
        """Submit an inbound channel message and yield progress + final reply updates."""
        if (
            message.metadata.get("_camera_authority") is CAMERA_AUTHORITY
            and message.metadata.get("_camera_duplicate_clarification_replay") is True
        ):
            return
        todo_lifecycle = _is_real_user_turn(message)
        wellness_reminder = _trusted_reminder_wellness(message)
        user_message = _build_inbound_user_message(
            message, self._attachment_store, session_key=session_key
        )
        user_prompt = user_message.text
        command_prompt = (message.content or "").strip()
        session_cwd = self._cwd_for_message(message, session_key)
        if todo_lifecycle:
            existing_bundle = self._bundles.get(session_key)
            if existing_bundle is not None:
                existing_bundle._todo_prompt_read_failure_logged = False
        bundle = await self.get_bundle(
            session_key,
            latest_user_prompt=user_prompt,
            cwd=session_cwd,
            include_todo=todo_lifecycle,
        )
        # Bind durable media refs to this exact configured gateway session.
        if isinstance(user_message, ConversationMessage):
            user_message.content = [
                block.model_copy(
                    update={
                        "source_provenance": {
                            **block.source_provenance,
                            "gateway_session_id": bundle.session_id,
                        }
                    }
                )
                if isinstance(block, AttachmentRefBlock) and block.source_provenance is not None
                else block
                for block in user_message.content
            ]
        turn_ctx = build_turn_context(
            message,
            session_id=bundle.session_id,
            owner_principals=self._gateway_config.owner_principals,
        )
        camera_authorized = (
            message.metadata.get("_camera_authority") is CAMERA_AUTHORITY
            and message.channel == "telegram"
            and str(message.chat_id) == self._gateway_config.camera_ingress.chat_id
            and session_key == self._gateway_config.camera_ingress.session_key
            and self._gateway_config.camera_ingress.enabled
            and message.sender_id.split("|", 1)[0]
            in {"__camera__", self._gateway_config.camera_ingress.principal}
        )
        if camera_authorized:
            turn_ctx = replace(turn_ctx, camera_authorized=True)
        if (
            camera_authorized or message.metadata.get("_camera_unbound") is CAMERA_AUTHORITY
        ) and not _evals_capture_enabled(self._gateway_config):
            raise ValueError("Camera turn requires nutrition finalization validation")
        self._bind_session_owner(message, session_key, turn_ctx)
        memory_scope = (
            MemoryScope(
                private_tenant=self._gateway_config.camera_ingress.tenant_id,
                shared_tenants=(),
            )
            if camera_authorized and message.sender_id == "__camera__"
            else self._resolve_turn_memory_scope(turn_ctx)
        )

        def start_camera_fast_replay_eval(
            *, typed_candidate: bool = False, candidate_id: object = None,
            operation_turn_id: object = None,
        ) -> GatewayEvalRecorder | None:
            """Start ordinary capture for a receipt-verified Camera fast reply."""
            if not (camera_authorized or typed_candidate) or not _evals_capture_enabled(
                self._gateway_config
            ):
                return None
            capture_message = message
            capture_turn_ctx = turn_ctx
            if typed_candidate and not camera_authorized:
                attempt = self._camera_ingress._attempts.get(candidate_id)
                if (
                    not isinstance(candidate_id, str)
                    or not isinstance(operation_turn_id, str)
                    or not isinstance(attempt, dict)
                    or attempt.get("candidate_id", candidate_id) != candidate_id
                    or attempt.get("state") != "completed"
                    or not isinstance(attempt.get("camera_commit"), Mapping)
                ):
                    return None
                capture_message = replace(
                    message,
                    metadata={
                        **message.metadata,
                        "_camera_authority": CAMERA_AUTHORITY,
                        "_camera_candidate_id": candidate_id,
                        "_camera_turn_id": operation_turn_id,
                    },
                )
                capture_turn_ctx = replace(turn_ctx, camera_authorized=True)
            logical_turn_id, _, assistant_metadata = _build_conversation_turn_metadata(
                turn_ctx=capture_turn_ctx, message=capture_message, scope=memory_scope
            )
            provenance, context = _camera_eval_capture_provenance(
                message=capture_message,
                turn_ctx=capture_turn_ctx,
                scope=memory_scope,
                camera_config=getattr(self._gateway_config, "camera_ingress", None),
                camera_ingress=getattr(self, "_camera_ingress", None),
                logical_turn_id=logical_turn_id or "",
                assistant_metadata=assistant_metadata,
            )
            if provenance is None or context is None:
                return None
            return GatewayEvalRecorder.start(
                workspace=self._workspace,
                bundle=bundle,
                message=capture_message,
                session_key=session_key,
                user_text=command_prompt,
                user_goal=user_prompt,
                trusted_turn_provenance=provenance,
                trusted_camera_context=context,
            )

        def finish_camera_fast_replay_eval(
            recorder: GatewayEvalRecorder | None, update: GatewayStreamUpdate
        ) -> None:
            if recorder is None:
                return
            if update.kind == "error":
                recorder.record_gateway_error(text=update.text, metadata=update.metadata)
                status = "error"
            else:
                recorder.record_gateway_final(text=update.text, metadata=update.metadata)
                status = "completed"
            try:
                recorder.record_resource_snapshot(
                    workspace=self._workspace, bundle=bundle, phase="world_after"
                )
            except Exception:
                logger.exception("ohmo eval world_after snapshot failed")
            recorder.finish(status=status)

        self._configure_turn_memory_surfaces(
            bundle,
            turn_ctx,
            memory_scope=memory_scope,
        )
        typed_replay_candidate_id = message.metadata.get("_camera_typed_replay_candidate")
        typed_replay_candidate = False
        if (
            isinstance(typed_replay_candidate_id, str)
            and message.channel == "telegram"
            and str(message.chat_id) == self._gateway_config.camera_ingress.chat_id
            and session_key == self._gateway_config.camera_ingress.session_key
            and self._gateway_config.camera_ingress.enabled
            and message.sender_id.split("|", 1)[0] == self._gateway_config.camera_ingress.principal
        ):
            typed_attempt = self._camera_ingress._attempts.get(typed_replay_candidate_id)
            typed_target = message.metadata.get("reply_to_message_id")
            typed_replay_candidate = bool(
                isinstance(typed_attempt, dict)
                and typed_attempt.get("state") == "completed"
                and isinstance(typed_attempt.get("camera_commit"), Mapping)
                and isinstance(typed_attempt.get("answer_turn_id"), str)
                and typed_attempt.get("finalizer_status") == "committed"
                and (
                    typed_attempt.get("camera_correction") is None
                    or (
                        typed_attempt.get("camera_correction") == "completed"
                        and isinstance(typed_attempt.get("camera_correction_commit"), dict)
                        and typed_attempt["camera_correction_commit"].get("kind") == "portion"
                    )
                )
                and (
                    (
                        typed_target is not None
                        and str(typed_target) in {
                            str(typed_attempt.get("photo_id")),
                            *map(str, typed_attempt.get("reply_ids", [])),
                        }
                    )
                    or (
                        typed_target is None
                        and self._camera_ingress.is_completed_context_typed_repeat(
                            message, typed_replay_candidate_id,
                        )
                    )
                )
            )
        completed_replay = (
            camera_authorized and message.metadata.get("_camera_existing_meal_replay") is True
        ) or typed_replay_candidate
        if (
            camera_authorized
            and message.metadata.get("_camera_correction_replay") is CAMERA_AUTHORITY
        ):
            candidate_id = message.metadata.get("_camera_candidate_id")
            turn_id = message.metadata.get("_camera_turn_id")
            attempt = self._camera_ingress._attempts.get(candidate_id)
            latest = attempt.get("camera_correction_commit") if isinstance(attempt, dict) else None
            pending_operation = bool(
                isinstance(attempt, dict)
                and attempt.get("camera_correction_turn_id") == turn_id
                and attempt.get("camera_correction") in {
                    "answering", "final_queued", "delivery_unknown"
                }
                and attempt.get("camera_correction_kind") in {"portion", "denial"}
            )
            latest_is_operation = (
                isinstance(latest, Mapping)
                and latest.get("client_op_id") == f"{turn_id}:assistant"
            )
            correction_kind = (
                latest.get("kind") if latest_is_operation
                else attempt.get("camera_correction_kind") if isinstance(attempt, dict)
                else None
            )
            backend = self._shadow_backend_for_scope(memory_scope)
            try:
                if (
                    backend is None or not isinstance(candidate_id, str)
                    or not isinstance(turn_id, str)
                    or correction_kind not in {"portion", "denial"}
                    or (not latest_is_operation and not pending_operation)
                ):
                    raise ConversationReconciliationError(
                        "latest Camera correction receipt is unavailable"
                    )
                receipt = await backend.reconcile_durable_exchange(
                    f"{turn_id}:user", f"{turn_id}:assistant"
                )
                typed_correction_replay = (
                    message.metadata.get("_camera_correction_replay_typed") is True
                )
                selected_label = (
                    message.content if typed_correction_replay
                    else message.metadata.get("native_keyboard_selected_label")
                )
                metadata = receipt.assistant_metadata if isinstance(receipt, ConversationAppendReceipt) else None
                trace = metadata.get("decision_trace") if isinstance(metadata, Mapping) else None
                annotations = trace.get("annotations") if isinstance(trace, Mapping) else None
                nutrition = annotations.get("nutrition") if isinstance(annotations, Mapping) else None
                correction = NutritionAnnotationV2.model_validate(nutrition)
                original = attempt.get("camera_commit")
                original_turn = attempt.get("answer_turn_id") if isinstance(attempt, dict) else None
                original_receipt = (
                    await backend.reconcile_durable_exchange(
                        f"{original_turn}:user", f"{original_turn}:assistant"
                    ) if isinstance(original_turn, str) else None
                )
                if not isinstance(original_receipt, ConversationAppendReceipt):
                    raise ConversationReconciliationError(
                        "original Camera meal receipt is unavailable"
                    )
                original_metadata = (
                    original_receipt.assistant_metadata
                    if isinstance(original_receipt, ConversationAppendReceipt) else None
                )
                original_trace = (
                    original_metadata.get("decision_trace")
                    if isinstance(original_metadata, Mapping) else None
                )
                original_annotations = (
                    original_trace.get("annotations") if isinstance(original_trace, Mapping) else None
                )
                original_nutrition = (
                    original_annotations.get("nutrition")
                    if isinstance(original_annotations, Mapping) else None
                )
                original_annotation = NutritionAnnotationV2.model_validate(original_nutrition)
                retained_targets = {
                    str(attempt.get("photo_id")), *map(str, attempt.get("reply_ids", []))
                }
                committed_binding = metadata.get("camera_reply_to_native_message_id") \
                    if isinstance(metadata, Mapping) else None
                requested_binding = (
                    message.metadata.get("reply_to_message_id") if typed_correction_replay
                    else message.metadata.get("_camera_native_binding")
                )
                original_native_binding = original_metadata.get("camera_reply_to_native_message_id") \
                    if isinstance(original_metadata, Mapping) else None
                trusted_context_replay = (
                    requested_binding is None
                    and message.metadata.get("_camera_route") == "context"
                    and message.metadata.get("_camera_authority") is CAMERA_AUTHORITY
                    and message.metadata.get("_camera_correction") is CAMERA_AUTHORITY
                    and message.metadata.get("_camera_correction_replay") is CAMERA_AUTHORITY
                    and typed_correction_replay
                )
                if trusted_context_replay:
                    # Contextual denials have no reply target. Ingress bound
                    # this owner message to one retained Camera context; use
                    # the native target from the verified original meal.
                    requested_binding = original_native_binding
                if correction_kind == "portion":
                    _validate_camera_portion_correction_annotation(nutrition)
                if (
                    not isinstance(metadata, Mapping) or not isinstance(original, Mapping)
                    or not isinstance(original_metadata, Mapping)
                    or original_receipt.assistant_message_id != original.get("event_id")
                    or original_receipt.assistant_client_op_id != original.get("client_op_id")
                    or original_receipt.assistant_client_op_id != f"{original_turn}:assistant"
                    or original_receipt.user_client_op_id != f"{original_turn}:user"
                    or original_metadata.get("role") != "assistant"
                    or original_metadata.get("client_op_id") != f"{original_turn}:assistant"
                    or original_metadata.get("logical_turn_id") != original_turn
                    or original_metadata.get("gateway_session_id") != turn_ctx.session_id
                    or original_metadata.get("tenant_id") != memory_scope.private_tenant
                    or original_metadata.get("source_principal")
                    != f"telegram:{canonical_principal(message.channel, turn_ctx.principal)}"
                    or original_metadata.get("camera_candidate_id") != candidate_id
                    or original_metadata.get("camera_operation_id") != candidate_id
                    or original_metadata.get("camera_answer_bound") != "yes"
                    or str(original_metadata.get("camera_reply_to_native_message_id")) not in retained_targets
                    or original_metadata.get("source_message_id") != original.get("source_message_id")
                    or original_metadata.get("ingest_source") != "dropbox_camera"
                    or original_metadata.get("confirmation_required") is not True
                    or original_metadata.get("is_group") is not False
                    or original_metadata.get("is_forwarded") is not False
                    or original_annotation.record_type != "meal_observation"
                    or original_annotation.consumption_status != "consumed"
                    or "image" not in original_annotation.basis
                    or original_annotation.meal_at != self._camera_ingress._attempt_capture_time(attempt)
                    or original_annotation.meal_date is not None
                    or original_annotation.explicit_new_consumption
                    or (latest_is_operation and receipt.assistant_message_id != latest.get("event_id"))
                    or receipt.assistant_client_op_id != f"{turn_id}:assistant"
                    or receipt.user_client_op_id != f"{turn_id}:user"
                    or not isinstance(receipt.user_content, str)
                    or metadata.get("role") != "assistant"
                    or metadata.get("client_op_id") != f"{turn_id}:assistant"
                    or metadata.get("logical_turn_id") != turn_id
                    or metadata.get("gateway_session_id") != turn_ctx.session_id
                    or metadata.get("tenant_id") != memory_scope.private_tenant
                    or metadata.get("source_principal")
                    != f"telegram:{canonical_principal(message.channel, turn_ctx.principal)}"
                    or (
                        latest_is_operation
                        and metadata.get("source_message_id") != latest.get("source_message_id")
                    )
                    or not isinstance(metadata.get("source_message_id"), str)
                    or not metadata.get("source_message_id")
                    or metadata.get("camera_candidate_id") != candidate_id
                    or metadata.get("camera_operation_id") != candidate_id
                    or metadata.get("camera_answer_bound")
                    != ("yes" if correction_kind == "portion" else "no")
                    or (
                        committed_binding is None and not trusted_context_replay
                    )
                    or (
                        committed_binding is not None
                        and str(committed_binding) not in retained_targets
                    )
                    or requested_binding is None
                    or str(requested_binding) not in retained_targets
                    or (
                        not typed_correction_replay
                        and committed_binding != str(requested_binding)
                    )
                    or metadata.get("camera_original_event_id") != original.get("event_id")
                    or metadata.get("reply_to_source_message_id") != original.get("source_message_id")
                    or (
                        latest_is_operation
                        and latest.get("target_event_id") != original.get("event_id")
                    )
                    or (
                        latest_is_operation
                        and latest.get("target_source_message_id") != original.get("source_message_id")
                    )
                    or metadata.get("camera_correction_bound") is not True
                    or metadata.get("ingest_source") != "dropbox_camera"
                    or metadata.get("confirmation_required") is not True
                    or metadata.get("is_group") is not False
                    or metadata.get("is_forwarded") is not False
                    or correction.record_type != "meal_correction"
                    or (
                        correction_kind == "portion"
                        and (
                            correction.consumption_status != "unknown"
                            or "items" not in correction.changed_fields
                        )
                    )
                    or (
                        correction_kind == "denial"
                        and (
                            correction.consumption_status != "not_consumed"
                            or "consumption_status" not in correction.changed_fields
                        )
                    )
                    or correction.meal_at is not None
                    or correction.meal_date is not None
                ):
                    raise ConversationReconciliationError(
                        "latest Camera correction did not prove the selected saved portion"
                    )
                if receipt.user_content == selected_label:
                    fast_eval_recorder = start_camera_fast_replay_eval()
                    if not latest_is_operation:
                        message.metadata.update(
                            _camera_authority=CAMERA_AUTHORITY,
                            _camera_correction=CAMERA_AUTHORITY,
                            _camera_answer="yes" if correction_kind == "portion" else "no",
                        )
                        if correction_kind == "portion":
                            message.metadata["_camera_portion_correction"] = CAMERA_AUTHORITY
                        else:
                            message.metadata.pop("_camera_portion_correction", None)
                        self._camera_ingress.record_committed_correction(
                            message, receipt, nutrition
                        )
                    fast_update = GatewayStreamUpdate(
                        kind="final",
                        text=(
                            "Эта порция уже записана."
                            if correction_kind == "portion"
                            else "Исправление уже записано."
                        ),
                        metadata={
                            "_session_key": session_key,
                            "camera_reconciled": candidate_id,
                            "nutrition_append_event_id": receipt.assistant_message_id,
                            **self._camera_final_delivery_metadata(message),
                        },
                    )
                    finish_camera_fast_replay_eval(fast_eval_recorder, fast_update)
                    yield fast_update
                    return
                if typed_correction_replay:
                    for key in (
                        "_camera_correction_replay", "_camera_correction_replay_typed",
                        "_camera_authority", "_camera_candidate_id", "_camera_answer",
                        "_camera_correction", "_camera_portion_correction", "_camera_turn_id",
                    ):
                        message.metadata.pop(key, None)
                    camera_authorized = False
                    typed_replay_candidate = False
                    turn_ctx = replace(turn_ctx, camera_authorized=False)
                elif (
                    correction_kind == "portion"
                    and isinstance(selected_label, str)
                    and selected_label.strip()
                    and message.metadata.get("native_keyboard_reflection_confirmed") is True
                    and isinstance(message.metadata.get("callback_query_id"), str)
                    and message.metadata.get("callback_query_id")
                ):
                    self._camera_ingress.authorize_recovered_camera_portion_correction(
                        message, candidate_id, original_turn, receipt, selected_label
                    )
                else:
                    raise ConversationReconciliationError(
                        "changed Camera selection is not a verified offered quantity"
                    )
            except (ConversationReconciliationError, ValueError, TypeError):
                fast_eval_recorder = start_camera_fast_replay_eval()
                logger.warning("completed Camera correction replay unresolved candidate=%s", candidate_id)
                fast_update = GatewayStreamUpdate(
                    kind="error",
                    text="Не получилось подтвердить запись этой порции. Новая запись не добавлена.",
                    metadata={"_session_key": session_key},
                )
                finish_camera_fast_replay_eval(fast_eval_recorder, fast_update)
                yield fast_update
                return
        if completed_replay:
            candidate_id = (
                typed_replay_candidate_id if typed_replay_candidate
                else message.metadata.get("_camera_candidate_id")
            )
            attempt = self._camera_ingress._attempts.get(candidate_id)
            turn_id = (
                attempt.get("answer_turn_id") if typed_replay_candidate and isinstance(attempt, dict)
                else message.metadata.get("_camera_turn_id")
            )
            commit = attempt.get("camera_commit") if isinstance(attempt, dict) else None
            backend = self._shadow_backend_for_scope(memory_scope)
            try:
                if (
                    backend is None
                    or not isinstance(candidate_id, str)
                    or not isinstance(turn_id, str)
                    or not isinstance(commit, Mapping)
                ):
                    raise ConversationReconciliationError(
                        "completed Camera meal receipt is unavailable"
                    )
                receipt = await backend.reconcile_durable_exchange(
                    f"{turn_id}:user", f"{turn_id}:assistant"
                )
                typed_replay = typed_replay_candidate
                context_typed_replay = bool(
                    typed_replay
                    and message.metadata.get("_camera_route") == "context"
                    and self._camera_ingress.is_completed_context_typed_repeat(
                        message, candidate_id,
                    )
                )
                selected_label = (
                    message.content if typed_replay
                    else message.metadata.get("native_keyboard_selected_label")
                )
                replay_binding = (
                    message.metadata.get("reply_to_message_id") if typed_replay
                    else message.metadata.get("_camera_native_binding")
                )
                if not isinstance(receipt, ConversationAppendReceipt):
                    raise ConversationReconciliationError(
                        "completed Camera exchange receipt is unavailable"
                    )
                assistant_metadata = receipt.assistant_metadata
                committed_binding = (
                    assistant_metadata.get("camera_reply_to_native_message_id")
                    if isinstance(assistant_metadata, Mapping) else None
                )
                typed_source_ids = (
                    {
                        str(attempt.get("photo_id")),
                        *map(str, attempt.get("reply_ids", [])),
                    }
                    if typed_replay and isinstance(attempt, dict) else set()
                )
                retained_source_ids = (
                    {
                        str(attempt.get("photo_id")),
                        *map(str, attempt.get("reply_ids", [])),
                    }
                    if isinstance(attempt, dict) else set()
                )
                original_route = (
                    assistant_metadata.get("camera_route")
                    if isinstance(assistant_metadata, Mapping) else None
                )
                original_context_receipt = bool(
                    original_route == "context"
                    and isinstance(assistant_metadata, Mapping)
                    and "camera_reply_to_native_message_id" not in assistant_metadata
                )
                original_source_bound_receipt = bool(
                    original_route in {"reply", "callback"}
                    and committed_binding is not None
                    and str(committed_binding) in retained_source_ids
                )
                current_route = message.metadata.get("_camera_route")
                current_binding_is_retained = (
                    replay_binding is not None
                    and str(replay_binding) in retained_source_ids
                )
                trusted_explicit_source_replay = bool(
                    current_binding_is_retained
                    and (
                        (typed_replay and current_route == "reply")
                        or (
                            camera_authorized
                            and message.metadata.get("_camera_existing_meal_replay") is True
                            and current_route in {"reply", "callback"}
                        )
                    )
                )
                replay_receipt_route_valid = bool(
                    (
                        original_context_receipt
                        and (context_typed_replay or trusted_explicit_source_replay)
                    )
                    or (
                        original_source_bound_receipt
                        and (
                            str(committed_binding) in typed_source_ids
                            if typed_replay
                            else committed_binding == str(replay_binding)
                        )
                    )
                )
                durable_replay = bool(
                    typed_replay
                    or message.metadata.get("_camera_existing_meal_replay") is True
                )
                trace = (
                    assistant_metadata.get("decision_trace")
                    if isinstance(assistant_metadata, Mapping) else None
                )
                annotations = trace.get("annotations") if isinstance(trace, Mapping) else None
                nutrition = annotations.get("nutrition") if isinstance(annotations, Mapping) else None
                changed_native_portion = (
                    not typed_replay
                    and isinstance(receipt.user_content, str)
                    and isinstance(selected_label, str)
                    and receipt.user_content != selected_label
                )
                if (
                    not isinstance(assistant_metadata, Mapping)
                    or not isinstance(trace, Mapping)
                    or not isinstance(annotations, Mapping)
                    or not isinstance(nutrition, Mapping)
                ):
                    raise ConversationReconciliationError(
                        "completed Camera receipt lacks validated nutrition evidence"
                    )
                validated = NutritionAnnotationV2.model_validate(nutrition)
                source_message_id = _normalize_source_message_ref(
                    message.metadata.get("message_id")
                )
                if (
                    receipt.assistant_message_id != commit.get("event_id")
                    or receipt.assistant_client_op_id != commit.get("client_op_id")
                    or receipt.user_client_op_id != f"{turn_id}:user"
                    or receipt.assistant_client_op_id != f"{turn_id}:assistant"
                    or not isinstance(receipt.user_content, str)
                    or not isinstance(selected_label, str)
                    or assistant_metadata.get("role") != "assistant"
                    or assistant_metadata.get("client_op_id") != f"{turn_id}:assistant"
                    or assistant_metadata.get("logical_turn_id") != turn_id
                    or assistant_metadata.get("camera_candidate_id") != candidate_id
                    or assistant_metadata.get("camera_operation_id") != candidate_id
                    or assistant_metadata.get("camera_answer_bound") != "yes"
                    or not replay_receipt_route_valid
                    or assistant_metadata.get("gateway_session_id") != turn_ctx.session_id
                    or assistant_metadata.get("tenant_id") != memory_scope.private_tenant
                    or assistant_metadata.get("source_principal")
                    != f"telegram:{canonical_principal(message.channel, turn_ctx.principal)}"
                    or assistant_metadata.get("source_message_id")
                    != commit.get("source_message_id")
                    or (
                        not durable_replay
                        and assistant_metadata.get("source_message_id") != source_message_id
                    )
                    or assistant_metadata.get("ingest_source") != "dropbox_camera"
                    or assistant_metadata.get("confirmation_required") is not True
                    or assistant_metadata.get("is_group") is not False
                    or assistant_metadata.get("is_forwarded") is not False
                    or validated.record_type != "meal_observation"
                    or validated.consumption_status != "consumed"
                    or "image" not in validated.basis
                    or validated.meal_at != self._camera_ingress._attempt_capture_time(attempt)
                    or validated.meal_date is not None
                    or validated.explicit_new_consumption
                ):
                    raise ConversationReconciliationError(
                        "completed Camera receipt did not prove the same saved portion"
                    )
                if typed_replay and receipt.user_content != selected_label:
                    # This is a changed reply, not an idempotent repeat. Let the
                    # established conversation/source-selection flow handle it.
                    message.metadata.pop("_camera_typed_replay_candidate", None)
                elif changed_native_portion:
                    self._camera_ingress.authorize_recovered_camera_portion_correction(
                        message, candidate_id, turn_id, receipt, selected_label
                    )
                else:
                    fast_eval_recorder = start_camera_fast_replay_eval(
                        typed_candidate=typed_replay,
                        candidate_id=candidate_id,
                        operation_turn_id=turn_id,
                    )
                    if typed_replay:
                        message.metadata.update(
                            _camera_typed_replay=True,
                            _camera_candidate_id=candidate_id,
                            _camera_turn_id=turn_id,
                        )
                    self._camera_ingress.recover_legacy_committed_meal(
                        candidate_id, turn_id, receipt
                    )
                    fast_update = GatewayStreamUpdate(
                        kind="final",
                        text="Эта порция уже записана.",
                        metadata={
                            "_session_key": session_key,
                            "camera_reconciled": candidate_id,
                            "nutrition_append_event_id": receipt.assistant_message_id,
                        },
                    )
                    finish_camera_fast_replay_eval(fast_eval_recorder, fast_update)
                    yield fast_update
                    return
            except (ConversationReconciliationError, ValueError):
                logger.warning("completed Camera replay remains unresolved candidate=%s", candidate_id)
                fast_eval_recorder = start_camera_fast_replay_eval(
                    typed_candidate=typed_replay_candidate,
                    candidate_id=candidate_id,
                    operation_turn_id=turn_id,
                )
                fast_update = GatewayStreamUpdate(
                    kind="error",
                    text="Не получилось подтвердить запись этой порции. Новая запись не добавлена.",
                    metadata={"_session_key": session_key},
                )
                finish_camera_fast_replay_eval(fast_eval_recorder, fast_update)
                yield fast_update
                return
        if camera_authorized and (
            message.metadata.get("_camera_legacy_reconcile") is True
            or message.metadata.get("_camera_reconcile_then_correction") is CAMERA_AUTHORITY
        ):
            candidate_id = message.metadata.get("_camera_candidate_id")
            original_turn = message.metadata.get("_camera_original_turn_id") or message.metadata.get(
                "_camera_turn_id"
            )
            backend = self._shadow_backend_for_scope(memory_scope)
            try:
                if backend is None or not isinstance(original_turn, str):
                    raise ConversationReconciliationError("legacy Camera receipt is unavailable")
                receipt = await backend.reconcile_durable_exchange(
                    f"{original_turn}:user", f"{original_turn}:assistant"
                )
                if receipt is None:
                    raise ConversationReconciliationError(
                        "legacy Camera operation has no observed committed exchange"
                    )
                if message.metadata.get("_camera_legacy_reconcile") is True:
                    self._camera_ingress.recover_legacy_committed_meal(
                        candidate_id, original_turn, receipt
                    )
                else:
                    original_message = replace(
                        message,
                        metadata={
                            **message.metadata,
                            "_camera_answer": "yes",
                            "_camera_turn_id": original_turn,
                        },
                    )
                    self._camera_ingress.record_committed_meal(
                        original_message, receipt, None
                    )
                self._camera_ingress.authorize_recovered_camera_denial(
                    message, candidate_id, original_turn
                )
            except (ConversationReconciliationError, ValueError, TypeError):
                if isinstance(candidate_id, str) and isinstance(original_turn, str):
                    self._camera_ingress.mark_finalizer_unknown(candidate_id, original_turn)
                logger.warning("legacy Camera commit remains unresolved candidate=%s", candidate_id)
                yield GatewayStreamUpdate(
                    kind="error",
                    text="I couldn't verify the original Camera meal, so no correction was recorded.",
                    metadata={"_session_key": session_key},
                )
                return
        if camera_authorized and message.metadata.get("_camera_reconcile_only") is True:
            candidate_id = message.metadata.get("_camera_candidate_id")
            turn_id = message.metadata.get("_camera_turn_id")
            backend = self._shadow_backend_for_scope(memory_scope)
            try:
                if backend is None or not isinstance(turn_id, str):
                    raise ConversationReconciliationError("Camera durable receipt is unavailable")
                receipt = await backend.reconcile_durable_exchange(
                    f"{turn_id}:user", f"{turn_id}:assistant"
                )
                if receipt is None:
                    raise ConversationReconciliationError(
                        "Camera operation has no observed committed exchange"
                    )
                if message.metadata.get("_camera_correction") is CAMERA_AUTHORITY:
                    trace = receipt.assistant_metadata.get("decision_trace") if receipt.assistant_metadata else None
                    annotations = trace.get("annotations") if isinstance(trace, Mapping) else None
                    nutrition = annotations.get("nutrition") if isinstance(annotations, Mapping) else None
                    validated = NutritionAnnotationV2.model_validate(nutrition)
                    if (
                        validated.record_type != "meal_correction"
                        or validated.consumption_status != "not_consumed"
                        or "consumption_status" not in validated.changed_fields
                    ):
                        raise ConversationReconciliationError(
                            "Camera correction metadata does not match its retained operation"
                        )
                    self._camera_ingress.record_committed_correction(
                        message, receipt, nutrition
                    )
                    text = "Исправление записано, баланс обновляется."
                elif message.metadata.get("_camera_answer") == "yes":
                    trace = receipt.assistant_metadata.get("decision_trace") if receipt.assistant_metadata else None
                    if (
                        receipt.assistant_metadata.get("camera_finalizer_outcome") == "clarification"
                    ):
                        self._camera_ingress.complete(
                            message, recorded=False, clarification=True
                        )
                        text = "Сколько примерно вы съели? Можно указать количество или долю порции."
                        message.metadata["_camera_clarification_final"] = CAMERA_AUTHORITY
                        yield GatewayStreamUpdate(
                            kind="final", text=text,
                            metadata={
                                "_session_key": session_key,
                                **self._camera_final_delivery_metadata(message),
                            },
                        )
                        return
                    annotations = trace.get("annotations") if isinstance(trace, Mapping) else None
                    nutrition = annotations.get("nutrition") if isinstance(annotations, Mapping) else None
                    validated = NutritionAnnotationV2.model_validate(nutrition)
                    trusted_time = self._camera_ingress.trusted_capture_time_for_answer(message)
                    if (
                        validated.record_type != "meal_observation"
                        or validated.consumption_status != "consumed"
                        or "image" not in validated.basis
                        or validated.explicit_new_consumption
                        or trusted_time is None
                        or validated.meal_at != trusted_time
                        or validated.meal_date is not None
                    ):
                        raise ConversationReconciliationError(
                            "Camera committed metadata does not match its retained operation"
                        )
                    self._camera_ingress.record_committed_meal(
                        message, receipt, validated.model_dump(mode="json")
                    )
                    self._camera_ingress.mark_reconciled_meal_ready(candidate_id, turn_id)
                    self._camera_ingress.complete(message, recorded=True)
                    text = "Записано, баланс обновляется."
                else:
                    raise ConversationReconciliationError(
                        "Camera denial receipt is known; no meal was appended"
                    )
                yield GatewayStreamUpdate(
                    kind="final", text=text,
                    metadata={
                        "_session_key": session_key,
                        "camera_reconciled": candidate_id,
                        **(
                            {
                                "nutrition_append_event_id": receipt.assistant_message_id,
                                "nutrition_sync_status": "pending",
                            }
                            if message.metadata.get("_camera_answer") == "yes"
                            else {}
                        ),
                        **self._camera_final_delivery_metadata(message),
                    },
                )
            except (ConversationReconciliationError, ValueError, TypeError):
                if isinstance(candidate_id, str) and isinstance(turn_id, str):
                    self._camera_ingress.mark_finalizer_unknown(candidate_id, turn_id)
                logger.warning("Camera durable operation remains unresolved candidate=%s", candidate_id)
                yield GatewayStreamUpdate(
                    kind="error",
                    text="I couldn't verify whether this Camera meal was recorded. It was not retried.",
                    metadata={"_session_key": session_key},
                )
            return
        if (
            camera_authorized
            and message.metadata.get("_camera_context_question_reconcile") is True
        ):
            candidate_id = message.metadata.get("_camera_candidate_id")
            turn_id = message.metadata.get("_camera_turn_id")
            backend = self._shadow_backend_for_scope(memory_scope)
            try:
                if backend is None or not isinstance(turn_id, str):
                    raise ConversationReconciliationError("Camera context receipt is unavailable")
                receipt = await backend.reconcile_durable_exchange(
                    f"{turn_id}:user", f"{turn_id}:assistant"
                )
                if (
                    receipt is None
                    or not isinstance(receipt.assistant_metadata, Mapping)
                    or receipt.assistant_metadata.get("camera_candidate_id") != candidate_id
                    or receipt.assistant_metadata.get("camera_context_only") is not True
                    or receipt.assistant_metadata.get("camera_finalizer_outcome") != "clarification"
                    or receipt.assistant_metadata.get("source_message_id")
                    != _normalize_source_message_ref(message.metadata.get("message_id"))
                    or not isinstance(receipt.assistant_content, str)
                ):
                    raise ConversationReconciliationError("Camera context receipt did not match source")
                self._camera_ingress.complete(
                    message, recorded=False, clarification=True
                )
                yield GatewayStreamUpdate(
                    kind="final",
                    text=receipt.assistant_content,
                    metadata={
                        "_session_key": session_key,
                        "camera_reconciled": candidate_id,
                        **self._camera_final_delivery_metadata(message),
                    },
                )
            except (ConversationReconciliationError, ValueError, TypeError):
                logger.warning("Camera context receipt remains unresolved candidate=%s", candidate_id)
                yield GatewayStreamUpdate(
                    kind="error",
                    text="I couldn't verify the previous Camera clarification. It was not repeated.",
                    metadata={"_session_key": session_key},
                )
            return
        camera_meal_at = self._trusted_camera_answer_time(
            message, session_key=session_key, turn_ctx=turn_ctx
        )
        prior_messages = getattr(bundle.engine, "messages", [])
        prior_messages = prior_messages if isinstance(prior_messages, list) else []
        user_photo_repeat = self._known_user_photo_repeat(message, prior_messages)
        user_photo_meal_at = (
            None
            if camera_meal_at is not None
            else self._trusted_user_photo_time(
                message, turn_ctx=turn_ctx, history=prior_messages
            )
        )
        system_prompt = await self._runtime_system_prompt(
                    bundle,
                    user_prompt,
                    turn_ctx=turn_ctx,
                    memory_scope=memory_scope,
                    include_todo=todo_lifecycle,
                )
        system_prompt = self._with_camera_turn_context(
            system_prompt, message, camera_meal_at
        )
        system_prompt = self._with_user_photo_context(
            system_prompt, user_photo_meal_at, known_repeat=user_photo_repeat
        )
        bundle.engine.set_system_prompt(system_prompt)
        logger.debug(
            "ohmo turn identity principal=%s owner=%s private=%s channel=%s chat_id=%s session_id=%s",
            turn_ctx.principal,
            turn_ctx.is_owner,
            turn_ctx.is_private,
            turn_ctx.channel,
            turn_ctx.chat_id,
            turn_ctx.session_id,
        )
        engine_metadata = getattr(bundle.engine, "tool_metadata", None)
        if isinstance(engine_metadata, dict):
            engine_metadata["ohmo_reminder_ctx"] = {
                "channel": message.channel,
                "chat_id": str(message.chat_id),
                "session_key": session_key,
                "sender_id": str(message.sender_id),
                "username": str(message.metadata.get("username") or "").strip(),
                "first_name": str(message.metadata.get("first_name") or "").strip(),
                "display_name": str(message.metadata.get("sender_display_name") or "").strip(),
                "chat_type": str(message.metadata.get("chat_type") or "").strip().lower(),
                # Group signal for the creator-only cancel ACL. Telegram emits
                # only ``is_group`` (bool), never ``chat_type``; Feishu sets
                # ``chat_type``. Accept either so the ACL works on both.
                "is_group": _is_group_message(message),
                "tz": message.metadata.get("tz") or "",
            }
        logger.info(
            "ohmo runtime processing start channel=%s chat_id=%s session_key=%s session_id=%s content=%r",
            message.channel,
            message.chat_id,
            session_key,
            bundle.session_id,
            _content_snippet(user_prompt),
        )

        camera_logical_turn_id, _, camera_turn_metadata = (
            _build_conversation_turn_metadata(turn_ctx=turn_ctx, message=message, scope=memory_scope)
            if camera_authorized
            else (None, None, {})
        )
        camera_turn_provenance, camera_capture_context = _camera_eval_capture_provenance(
            message=message, turn_ctx=turn_ctx, scope=memory_scope,
            camera_config=getattr(self._gateway_config, "camera_ingress", None),
            camera_ingress=getattr(self, "_camera_ingress", None),
            logical_turn_id=camera_logical_turn_id or "", assistant_metadata=camera_turn_metadata,
        )
        recorder = (
            GatewayEvalRecorder.start(
                workspace=self._workspace,
                bundle=bundle,
                message=message,
                session_key=session_key,
                user_text=command_prompt,
                user_goal=user_prompt,
                trusted_turn_provenance=camera_turn_provenance,
                trusted_camera_context=camera_capture_context,
            )
            if _evals_capture_enabled(self._gateway_config)
            else None
        )
        if recorder is not None and camera_meal_at is not None:
            recorder.set_authoritative_nutrition_meal_at(camera_meal_at)
        elif recorder is not None and user_photo_meal_at is not None:
            recorder.set_authoritative_nutrition_meal_at(
                user_photo_meal_at, preserve_explicit=True
            )
            recorder.mark_trusted_direct_photo_intent()
        camera_context_question = (
            message.metadata.get("_camera_context_question")
            is CAMERA_CONTEXT_QUESTION_AUTHORITY
        )
        if recorder is not None and (
            (
                camera_authorized
                and message.metadata.get("_camera_answer") != "yes"
                and not camera_context_question
            )
            or (
                camera_authorized
                and message.metadata.get("_camera_answer") == "yes"
                and camera_meal_at is None
            )
            or message.metadata.get("_camera_unbound") is CAMERA_AUTHORITY
        ) and message.metadata.get("_camera_correction") is not CAMERA_AUTHORITY:
            recorder.forbid_nutrition_record()
        if (
            recorder is not None
            and message.metadata.get("_camera_context_hint") is CAMERA_AUTHORITY
            and message.metadata.get("_camera_answer") != "yes"
            and not camera_context_question
        ):
            recorder.forbid_nutrition_record()
        if recorder is not None and message.metadata.get("_camera_context_unrelated") is CAMERA_AUTHORITY:
            recorder.forbid_nutrition_record()
        if recorder is not None and camera_authorized and (
            message.sender_id == "__camera__"
            or message.metadata.get("_camera_answer") == "yes"
        ):
            recorder.forbid_explicit_new_consumption()
        episode_status = "completed"
        decision_trace_restore = _install_gateway_decision_trace_recorder(
            bundle.engine,
            recorder,
        )
        suspended_todo_tool = None
        if not todo_lifecycle:
            tools = getattr(getattr(bundle, "tool_registry", None), "_tools", None)
            if isinstance(tools, dict):
                suspended_todo_tool = tools.pop(_TODO_TOOL_NAME, None)

        async def record_updates(updates):
            nonlocal episode_status
            async for update in updates:
                if update.kind == "final":
                    if recorder is not None:
                        recorder.record_gateway_final(text=update.text, metadata=update.metadata)
                elif update.kind == "assistant_update":
                    if recorder is not None:
                        recorder.record_gateway_update(text=update.text, metadata=update.metadata)
                elif update.kind == "error":
                    episode_status = "error"
                    if recorder is not None:
                        recorder.record_gateway_error(text=update.text, metadata=update.metadata)
                yield update

        try:
            command_context: CommandContext | None = None

            def get_command_context() -> CommandContext:
                nonlocal command_context
                if command_context is None:
                    command_context = CommandContext(
                        engine=bundle.engine,
                        hooks_summary=getattr(bundle, "hook_summary", lambda: "")(),
                        mcp_summary=getattr(bundle, "mcp_summary", lambda: "")(),
                        plugin_summary=getattr(bundle, "plugin_summary", lambda: "")(),
                        cwd=getattr(bundle, "cwd", str(self._cwd)),
                        tool_registry=getattr(bundle, "tool_registry", None),
                        app_state=getattr(bundle, "app_state", None),
                        session_backend=getattr(bundle, "session_backend", self._session_backend),
                        session_id=getattr(bundle, "session_id", None),
                        extra_skill_dirs=getattr(bundle, "extra_skill_dirs", ()),
                        extra_plugin_roots=getattr(bundle, "extra_plugin_roots", ()),
                        memory_backend=create_memory_command_backend(
                            self._workspace,
                            backend_kind=self._gateway_config.memory_backend,
                        ),
                        include_project_memory=False,
                    )
                return command_context

            parsed = bundle.commands.lookup(command_prompt)
            if parsed is None and not message.media:
                parsed = lookup_skill_slash_command(command_prompt, get_command_context())
            if parsed is not None and not message.media:
                command, args = parsed
                command_name = str(getattr(command, "name", "") or "")
                gateway_result = self._handle_gateway_scoped_command(command_name, args)
                if gateway_result is not None:
                    message_text, refresh_runtime = gateway_result
                    result = CommandResult(message=message_text, refresh_runtime=refresh_runtime)
                    async for update in record_updates(
                        self._stream_command_result(
                            bundle=bundle,
                            message=message,
                            session_key=session_key,
                            user_prompt=user_prompt,
                            result=result,
                            turn_ctx=turn_ctx,
                            memory_scope=memory_scope,
                            recorder=recorder,
                            todo_lifecycle=todo_lifecycle,
                            camera_meal_at=camera_meal_at,
                            user_photo_meal_at=user_photo_meal_at,
                            user_photo_repeat=user_photo_repeat,
                            wellness_reminder=wellness_reminder,
                        )
                    ):
                        yield update
                    return
                remote_allowed = getattr(command, "remote_invocable", True)
                if not remote_allowed and self._remote_admin_allowed(command):
                    remote_allowed = True
                    logger.warning(
                        "ohmo gateway remote administrative command accepted channel=%s chat_id=%s sender_id=%s command=%s",
                        message.channel,
                        message.chat_id,
                        message.sender_id,
                        command_name,
                    )
                if not remote_allowed:
                    result = CommandResult(
                        message=f"/{command_name} is only available in the local OpenHarness UI."
                    )
                    async for update in record_updates(
                        self._stream_command_result(
                            bundle=bundle,
                            message=message,
                            session_key=session_key,
                            user_prompt=user_prompt,
                            result=result,
                            turn_ctx=turn_ctx,
                            memory_scope=memory_scope,
                            recorder=recorder,
                            todo_lifecycle=todo_lifecycle,
                            camera_meal_at=camera_meal_at,
                            user_photo_meal_at=user_photo_meal_at,
                            user_photo_repeat=user_photo_repeat,
                            wellness_reminder=wellness_reminder,
                        )
                    ):
                        yield update
                    return
                result = await command.handler(
                    args,
                    get_command_context(),
                )
                async for update in record_updates(
                    self._stream_command_result(
                        bundle=bundle,
                        message=message,
                        session_key=session_key,
                        user_prompt=user_prompt,
                        result=result,
                        turn_ctx=turn_ctx,
                        memory_scope=memory_scope,
                        recorder=recorder,
                        todo_lifecycle=todo_lifecycle,
                        camera_meal_at=camera_meal_at,
                        user_photo_meal_at=user_photo_meal_at,
                        user_photo_repeat=user_photo_repeat,
                        wellness_reminder=wellness_reminder,
                    )
                ):
                    yield update
                return

            async for update in record_updates(
                self._stream_engine_message(
                    bundle=bundle,
                    message=message,
                    session_key=session_key,
                    user_prompt=user_prompt,
                    user_message=user_message,
                    turn_ctx=turn_ctx,
                    memory_scope=memory_scope,
                    recorder=recorder,
                    todo_lifecycle=todo_lifecycle,
                    camera_meal_at=camera_meal_at,
                    user_photo_meal_at=user_photo_meal_at,
                    user_photo_repeat=user_photo_repeat,
                    wellness_reminder=wellness_reminder,
                )
            ):
                yield update
        except Exception as exc:
            episode_status = "exception"
            if recorder is not None:
                recorder.record_exception(exc)
            raise
        finally:
            if suspended_todo_tool is not None:
                registry = getattr(bundle, "tool_registry", None)
                if registry is not None:
                    registry.register(suspended_todo_tool)
            _restore_gateway_decision_trace_recorder(decision_trace_restore)
            if recorder is not None:
                try:
                    recorder.record_resource_snapshot(
                        workspace=self._workspace, bundle=bundle, phase="world_after"
                    )
                except Exception:
                    logger.exception("ohmo eval world_after snapshot failed")
                recorder.finish(status=episode_status)

    async def _stream_command_result(
        self,
        *,
        bundle: RuntimeBundle,
        message: InboundMessage,
        session_key: str,
        user_prompt: str,
        result,
        turn_ctx: TurnContext,
        memory_scope: MemoryScope | None,
        recorder: GatewayEvalRecorder | None = None,
        todo_lifecycle: bool = True,
        camera_meal_at: datetime | None = None,
        user_photo_meal_at: datetime | None = None,
        user_photo_repeat: bool = False,
        wellness_reminder: _TrustedReminderWellnessAdmission | None = None,
    ):
        if result.refresh_runtime:
            bundle = await self._refresh_bundle(
                session_key,
                bundle,
                user_prompt,
                turn_ctx=turn_ctx,
                memory_scope=memory_scope,
                include_todo=todo_lifecycle,
            )
        self._register_conversation_image_tool(bundle)

        todo_error = getattr(bundle, "_todo_runtime_error", None)
        if todo_lifecycle and isinstance(todo_error, TodoRuntimeStateError):
            yield GatewayStreamUpdate(
                kind="error",
                text=self._todo_runtime_error_text(todo_error),
                metadata={"_session_key": session_key},
            )
            return

        if result.message:
            yield GatewayStreamUpdate(
                kind="final",
                text=result.message,
                metadata={"_session_key": session_key, "_command": True},
            )

        if result.submit_prompt is not None:
            original_model = bundle.engine.model
            if result.submit_model:
                bundle.engine.set_model(result.submit_model)
            try:
                async for update in self._stream_engine_message(
                    bundle=bundle,
                    message=message,
                    session_key=session_key,
                    user_prompt=result.submit_prompt,
                    user_message=result.submit_prompt,
                    turn_ctx=turn_ctx,
                    memory_scope=memory_scope,
                    recorder=recorder,
                    todo_lifecycle=todo_lifecycle,
                    camera_meal_at=camera_meal_at,
                    user_photo_meal_at=user_photo_meal_at,
                    user_photo_repeat=user_photo_repeat,
                    wellness_reminder=wellness_reminder,
                ):
                    yield update
            finally:
                if result.submit_model:
                    bundle.engine.set_model(original_model)
            return

        if result.continue_pending:
            settings = bundle.current_settings()
            if bundle.enforce_max_turns:
                bundle.engine.set_max_turns(settings.max_turns)
            continue_prompt = await self._runtime_system_prompt(
                bundle,
                _last_user_text(bundle.engine.messages),
                turn_ctx=turn_ctx,
                memory_scope=memory_scope,
                include_todo=todo_lifecycle,
            )
            bundle.engine.set_system_prompt(
                self._with_camera_turn_context(continue_prompt, message, camera_meal_at)
            )
            todo_error = getattr(bundle, "_todo_runtime_error", None)
            if todo_lifecycle and isinstance(todo_error, TodoRuntimeStateError):
                yield GatewayStreamUpdate(
                    kind="error",
                    text=self._todo_runtime_error_text(todo_error),
                    metadata={"_session_key": session_key},
                )
                return
            turns = (
                result.continue_turns
                if result.continue_turns is not None
                else bundle.engine.max_turns
            )
            reply_parts: list[str] = []
            stream_error = False
            max_turns_exceeded = False
            decision_trace_restore = _install_gateway_decision_trace_recorder(
                bundle.engine,
                recorder,
            )
            self._register_conversation_image_tool(bundle)
            try:
                try:
                    async for event in bundle.engine.continue_pending(max_turns=turns):
                        async for update in self._convert_stream_event(
                            event=event,
                            bundle=bundle,
                            message=message,
                            session_key=session_key,
                            content=user_prompt,
                            reply_parts=reply_parts,
                            recorder=recorder,
                        ):
                            if update.kind == "error":
                                stream_error = True
                            yield update
                except MaxTurnsExceeded as exc:
                    max_turns_exceeded = True
                    stream_error = True
                    yield GatewayStreamUpdate(
                        kind="error",
                        text=f"Stopped after {exc.max_turns} turns (max_turns).",
                        metadata={"_session_key": session_key},
                    )
            finally:
                _restore_gateway_decision_trace_recorder(decision_trace_restore)
                self._register_conversation_image_tool(bundle)
            reply = "".join(reply_parts).strip()
            if stream_error or max_turns_exceeded:
                await self._save_snapshot(bundle, session_key, user_prompt)
                return
            guard_state = {"reply": reply, "error": None}
            if todo_lifecycle and reply and not stream_error and not max_turns_exceeded:
                async for update in self._guard_todo_final(
                    bundle=bundle,
                    message=message,
                    session_key=session_key,
                    user_prompt=user_prompt,
                    turn_ctx=turn_ctx,
                    memory_scope=memory_scope,
                    reply_parts=reply_parts,
                    emitted_media=set(),
                    recorder=recorder,
                    state=guard_state,
                    camera_meal_at=camera_meal_at,
                ):
                    yield update
            await self._save_snapshot(bundle, session_key, user_prompt)
            if guard_state["error"] is not None:
                yield GatewayStreamUpdate(
                    kind="error",
                    text=str(guard_state["error"]),
                    metadata={"_session_key": session_key},
                )
                return
            reply = str(guard_state["reply"] or "")
            if reply:
                yield GatewayStreamUpdate(
                    kind="final",
                    text=reply,
                    metadata={"_session_key": session_key},
                )
                if todo_lifecycle:
                    try:
                        cleanup = self._todo_cleanup_update(bundle=bundle, session_key=session_key)
                        if cleanup is not None:
                            yield cleanup
                    except Exception:
                        logger.warning(
                            "ohmo.todo.cleanup_failure session_id=%s",
                            bundle.session_id,
                            exc_info=True,
                        )
            return

        await self._save_snapshot(bundle, session_key, user_prompt)

    async def _stream_engine_message(
        self,
        *,
        bundle: RuntimeBundle,
        message: InboundMessage,
        session_key: str,
        user_prompt: str,
        user_message: ConversationMessage | str,
        turn_ctx: TurnContext,
        memory_scope: MemoryScope | None,
        recorder: GatewayEvalRecorder | None = None,
        todo_lifecycle: bool = True,
        camera_meal_at: datetime | None = None,
        user_photo_meal_at: datetime | None = None,
        user_photo_repeat: bool = False,
        wellness_reminder: _TrustedReminderWellnessAdmission | None = None,
    ):
        message.metadata.pop("_selected_source_binding", None)
        if message.metadata.get("_camera_context_meal_target") is _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY:
            ingress = getattr(self, "_camera_ingress", None)
            candidate_id = message.metadata.get("_camera_context_meal_candidate_id")
            attempt = (
                ingress._attempts.get(candidate_id)
                if ingress is not None and isinstance(candidate_id, str) else None
            )
            legacy_commit = attempt.get("camera_commit") if isinstance(attempt, dict) else None
            legacy_fields = {
                "event_id", "source_message_id", "client_op_id", "candidate_id",
                "tenant_id", "principal", "meal_at", "record_type", "consumption_status",
            }
            if isinstance(legacy_commit, Mapping) and set(legacy_commit) == legacy_fields:
                original_op = legacy_commit.get("client_op_id")
                original_turn = (
                    original_op.removesuffix(":assistant")
                    if isinstance(original_op, str) and original_op.endswith(":assistant")
                    else None
                )
                if (
                    isinstance(original_turn, str)
                    and original_turn in {attempt.get("answer_turn_id"), attempt.get("final_turn_id")}
                    and isinstance(memory_scope, MemoryScope)
                ):
                    backend = self._shadow_backend_for_scope(memory_scope)
                    if backend is not None:
                        try:
                            original_receipt = await backend.reconcile_durable_exchange(
                                f"{original_turn}:user", f"{original_turn}:assistant"
                            )
                            if original_receipt is not None:
                                ingress.recover_legacy_committed_meal(
                                    candidate_id, original_turn, original_receipt
                                )
                        except (ConversationReconciliationError, TypeError, ValueError):
                            # Keep the legacy journal unresolved. Contextual routing
                            # below remains fail-closed unless its exact durable
                            # original exchange upgrades successfully.
                            pass
            binding = (
                ingress.contextual_meal_source_binding(
                    message, candidate_id=candidate_id, gateway_session_id=bundle.session_id
                )
                if ingress is not None and isinstance(candidate_id, str)
                else None
            )
            if (
                binding is not None
                and isinstance(memory_scope, MemoryScope)
                and turn_ctx.is_private
                and not turn_ctx.is_forwarded
                and message.sender_id.split("|", 1)[0]
                == self._gateway_config.camera_ingress.principal
                and str(message.chat_id) == self._gateway_config.camera_ingress.chat_id
                and binding.get("tenant_id") == memory_scope.private_tenant
                and binding.get("source_principal")
                == f"telegram:{canonical_principal('telegram', turn_ctx.principal)}"
                and isinstance(binding.get("gateway_session_id"), str)
                and bool(binding.get("gateway_session_id"))
            ):
                message.metadata["_selected_source_binding"] = (
                    _SELECTED_SOURCE_AUTHORITY, binding
                )
                attempt = ingress._attempts.get(candidate_id)
                projection = (
                    attempt.get("context_meal_projection")
                    if isinstance(attempt, dict) else None
                )
                source_message_id = _normalize_source_message_ref(
                    message.metadata.get("message_id")
                )
                projections = [
                    saved for saved in (
                        *(attempt.get("context_meal_projection_history", [])
                          if isinstance(attempt, dict)
                          and isinstance(attempt.get("context_meal_projection_history"), list)
                          else []),
                        projection,
                    )
                    if isinstance(saved, Mapping)
                    and saved.get("source_message_id_current") == source_message_id
                ]
                if projections:
                    backend = self._shadow_backend_for_scope(memory_scope)
                    expected_metadata = _build_conversation_turn_metadata(
                        turn_ctx=turn_ctx, message=message, scope=memory_scope,
                        recorder=recorder,
                    )[1]
                    for saved_projection in projections:
                        receipt = None
                        try:
                            if backend is not None:
                                receipt = await backend.reconcile_durable_exchange(
                                    str(saved_projection.get("user_client_op_id") or ""),
                                    str(saved_projection.get("client_op_id") or ""),
                                )
                        except (ConversationReconciliationError, TypeError, ValueError):
                            receipt = None
                        if (
                            receipt is not None
                            and _exact_context_receipt_replay(
                                saved_projection, receipt, expected_metadata,
                                expected_user_text=message.content or user_prompt,
                                expected_native_target=_normalize_source_message_ref(
                                    message.metadata.get("reply_to_message_id")
                                ),
                            )
                        ):
                            yield GatewayStreamUpdate(
                                kind="final",
                                text=receipt.assistant_content,
                                metadata={
                                    "_session_key": session_key,
                                    "nutrition_append_event_id": receipt.assistant_message_id,
                                    "nutrition_sync_status": "pending",
                                },
                            )
                            await self._save_snapshot(bundle, session_key, user_prompt)
                            return
                    yield GatewayStreamUpdate(
                        kind="error",
                        text="Could not verify the saved Camera correction for this message.",
                        metadata={"_session_key": session_key},
                    )
                    return
        todo_error = getattr(bundle, "_todo_runtime_error", None)
        if todo_lifecycle and isinstance(todo_error, TodoRuntimeStateError):
            yield GatewayStreamUpdate(
                kind="error",
                text=self._todo_runtime_error_text(todo_error),
                metadata={"_session_key": session_key},
            )
            return
        system_prompt = await self._runtime_system_prompt(
            bundle,
            user_prompt,
            turn_ctx=turn_ctx,
            memory_scope=memory_scope,
            include_todo=todo_lifecycle,
        )
        system_prompt = self._with_camera_turn_context(system_prompt, message, camera_meal_at)
        system_prompt = self._with_user_photo_context(
            system_prompt, user_photo_meal_at, known_repeat=user_photo_repeat
        )
        bundle.engine.set_system_prompt(system_prompt)
        todo_error = getattr(bundle, "_todo_runtime_error", None)
        if todo_lifecycle and isinstance(todo_error, TodoRuntimeStateError):
            yield GatewayStreamUpdate(
                kind="error",
                text=self._todo_runtime_error_text(todo_error),
                metadata={"_session_key": session_key},
            )
            return
        reply_parts: list[str] = []
        emitted_media: set[str] = set()
        stream_error = False
        yield GatewayStreamUpdate(
            kind="progress",
            text=_format_channel_progress(
                channel=message.channel,
                kind="thinking",
                text="Thinking...",
                session_key=session_key,
                content=user_prompt,
            ),
            metadata={
                "_progress": True,
                "_session_key": session_key,
                # Provider-neutral inference activity: the turn's model call
                # starts here. Quiet channels render this as an explicit
                # "inference" state instead of a canned thinking line.
                "progress_event": {"kind": "inference", "state": "active"},
            },
        )
        previous_group_request = self._set_group_request_context(bundle, message, session_key)
        active_turn = {"active": True}
        attempted_attachment_ids: set[str] = set()
        successful_attachment_ids: set[str] = set()
        successful_source_bindings: dict[str, Mapping[str, object]] = {}
        camera_original_attachment_ids: set[str] = set()
        if camera_meal_at is not None:
            ingress = getattr(self, "_camera_ingress", None)
            candidate_id = message.metadata.get("_camera_candidate_id")
            attempt = ingress._attempts.get(candidate_id) if ingress is not None else None
            admitted_image_sha256 = (
                attempt.get("image_sha256") if isinstance(attempt, dict) else None
            )
            if (
                isinstance(candidate_id, str)
                and isinstance(admitted_image_sha256, str)
                and re.fullmatch(r"[0-9a-f]{64}", admitted_image_sha256)
            ):
                camera_original_attachment_ids = {admitted_image_sha256}

        def restore_independent_time_default() -> None:
            if recorder is None:
                return
            recorder.set_authoritative_nutrition_meal_at(
                camera_meal_at or user_photo_meal_at,
                preserve_explicit=True,
            )

        def on_attachment_load_started(attachment_id: str) -> None:
            if not active_turn["active"]:
                return
            attempted_attachment_ids.add(attachment_id)
            message.metadata.pop("_selected_source_binding", None)
            restore_independent_time_default()

        def bind_loaded_attachment(attachment_id: str) -> str | None:
            if not active_turn["active"]:
                return None
            if camera_meal_at is not None:
                successful_attachment_ids.add(attachment_id)
            if (
                attachment_id not in attempted_attachment_ids
                or (camera_meal_at is None and len(attempted_attachment_ids) != 1)
            ):
                message.metadata.pop("_selected_source_binding", None)
                restore_independent_time_default()
                return None
            if memory_scope is None or not isinstance(memory_scope, MemoryScope):
                return None
            principal = canonical_principal("telegram", turn_ctx.principal)
            family_tenant = self._gateway_config.family_principals.get(principal)
            binding = resolve_selected_photo_source(
                attachment_id=attachment_id,
                history=getattr(bundle.engine, "messages", []),
                message=message,
                turn_ctx=turn_ctx,
                session_key=session_key,
                gateway_session_id=bundle.session_id,
                tenant_id=memory_scope.private_tenant,
                authorized_participant=(
                    turn_ctx.is_owner is True
                    or family_tenant == memory_scope.private_tenant
                ),
            )
            if binding is None:
                message.metadata.pop("_selected_source_binding", None)
                restore_independent_time_default()
                return None
            stamped_time = _trusted_utc_iso(binding.get("received_at"))
            if stamped_time is None:
                message.metadata.pop("_selected_source_binding", None)
                restore_independent_time_default()
                return None
            if camera_meal_at is not None:
                successful_source_bindings[attachment_id] = dict(binding)
                if len(successful_attachment_ids) != 1:
                    message.metadata.pop("_selected_source_binding", None)
                    restore_independent_time_default()
                    return None
            message.metadata["_selected_source_binding"] = (
                _SELECTED_SOURCE_AUTHORITY, binding
            )
            if recorder is not None and camera_meal_at is None:
                recorder.set_authoritative_nutrition_meal_at(
                    datetime.fromisoformat(stamped_time), preserve_explicit=True,
                    historical_photo=True,
                )
            return stamped_time
        decision_trace_restore = _install_gateway_decision_trace_recorder(
            bundle.engine,
            recorder,
        )
        self._register_conversation_image_tool(
            bundle,
            current_message=user_message,
            on_attachment_load_started=on_attachment_load_started,
            on_attachment_loaded=bind_loaded_attachment,
        )
        try:
            admitted_actor = self._wellness_actor_for_submission(
                bundle, turn_ctx, wellness_reminder
            )
            if admitted_actor is None:
                turn_events = bundle.engine.submit_message(user_message)
            else:
                turn_events = bundle.engine.submit_message(
                    user_message, wellness_actor=admitted_actor
                )
            async for event in turn_events:
                if isinstance(event, ErrorEvent) and _should_retry_without_image_input(
                    event.message,
                    [*bundle.engine.messages, user_message]
                    if isinstance(user_message, ConversationMessage)
                    else bundle.engine.messages,
                ):
                    if recorder is not None:
                        recorder.record_engine_error(event)
                    logger.warning(
                        "ohmo runtime image input rejected; retrying without image blocks session_key=%s session_id=%s message=%r",
                        session_key,
                        bundle.session_id,
                        _content_snippet(event.message),
                    )
                    _strip_image_blocks_from_engine_history(bundle.engine)
                    yield GatewayStreamUpdate(
                        kind="progress",
                        text=_format_channel_progress(
                            channel=message.channel,
                            kind="image_fallback",
                            text=event.message,
                            session_key=session_key,
                            content=user_prompt,
                        ),
                        metadata={
                            "_progress": True,
                            "_session_key": session_key,
                            "_image_fallback": True,
                        },
                    )
                    async for retry_event in bundle.engine.continue_pending(
                        max_turns=bundle.engine.max_turns
                    ):
                        async for update in self._convert_stream_event(
                            event=retry_event,
                            bundle=bundle,
                            message=message,
                            session_key=session_key,
                            content=user_prompt,
                            reply_parts=reply_parts,
                            recorder=recorder,
                        ):
                            if update.kind == "error":
                                stream_error = True
                            _remember_update_media(emitted_media, update)
                            yield update
                    break
                async for update in self._convert_stream_event(
                    event=event,
                    bundle=bundle,
                    message=message,
                    session_key=session_key,
                    content=user_prompt,
                    reply_parts=reply_parts,
                    recorder=recorder,
                ):
                    if update.kind == "error":
                        stream_error = True
                    _remember_update_media(emitted_media, update)
                    yield update
        except MaxTurnsExceeded as exc:
            yield GatewayStreamUpdate(
                kind="error",
                text=f"Stopped after {exc.max_turns} turns (max_turns).",
                metadata={"_session_key": session_key},
            )
            self._restore_group_request_context(bundle, previous_group_request)
            self._clear_reminder_context(bundle)
            await self._save_snapshot(bundle, session_key, user_prompt)
            return
        except Exception:
            self._restore_group_request_context(bundle, previous_group_request)
            self._clear_reminder_context(bundle)
            raise
        finally:
            active_turn["active"] = False
            _restore_gateway_decision_trace_recorder(decision_trace_restore)
            self._register_conversation_image_tool(bundle)
        self._restore_group_request_context(bundle, previous_group_request)
        self._clear_reminder_context(bundle)
        if stream_error:
            await self._save_snapshot(bundle, session_key, user_prompt)
            return
        reply = "".join(reply_parts).strip()
        guard_state = {"reply": reply, "error": None}
        if todo_lifecycle and reply and not stream_error:
            async for update in self._guard_todo_final(
                bundle=bundle,
                message=message,
                session_key=session_key,
                user_prompt=user_prompt,
                turn_ctx=turn_ctx,
                memory_scope=memory_scope,
                reply_parts=reply_parts,
                emitted_media=emitted_media,
                recorder=recorder,
                state=guard_state,
                camera_meal_at=camera_meal_at,
            ):
                yield update

        await self._save_snapshot(bundle, session_key, user_prompt)
        if guard_state["error"] is not None:
            yield GatewayStreamUpdate(
                kind="error",
                text=str(guard_state["error"]),
                metadata={"_session_key": session_key},
            )
            return
        reply = str(guard_state["reply"] or "")

        camera_portion_correction = (
            message.metadata.get("_camera_portion_correction") is CAMERA_AUTHORITY
        )
        camera_yes = (
            turn_ctx.camera_authorized
            and message.metadata.get("_camera_answer") == "yes"
            and message.metadata.get("_camera_authority") is CAMERA_AUTHORITY
            and not camera_portion_correction
        )
        camera_clarification = bool(
            camera_yes
            and message.metadata.get("_camera_clarification_allowed") is CAMERA_AUTHORITY
            and recorder is not None
            and recorder.validated_nutrition_envelope is None
            and recorder.nutrition_annotation_status in {"missing", "not_applicable"}
        )
        finalizer_nutrition = (
            NutritionAnnotationV2.model_validate(recorder.validated_nutrition_envelope)
            if recorder is not None
            and recorder.validated_nutrition_envelope is not None
            and recorder.validated_nutrition_envelope.get("schema_version", 1) == 2
            else None
        )
        ordinary_meal = bool(
            not camera_yes
            and finalizer_nutrition is not None
            and finalizer_nutrition.record_type == "meal_observation"
            and finalizer_nutrition.consumption_status == "consumed"
        )
        requested_correction = bool(
            finalizer_nutrition is not None
            and finalizer_nutrition.record_type in {"meal_correction", "meal_deletion"}
            and message.metadata.get("_camera_correction") is not CAMERA_AUTHORITY
        )
        selected_binding = message.metadata.get("_selected_source_binding")
        native_reply_target = _normalize_source_message_ref(
            message.metadata.get("reply_to_message_id")
        )
        principal_id = canonical_principal("telegram", turn_ctx.principal)
        native_reply_authorized = bool(
            native_reply_target is not None
            and turn_ctx.channel == "telegram"
            and message.channel == "telegram"
            and turn_ctx.is_private
            and not turn_ctx.is_forwarded
            and is_private_message(message)
            and not turn_ctx.camera_authorized
            and isinstance(memory_scope, MemoryScope)
            and memory_scope.private_tenant
            and (
                turn_ctx.is_owner is True
                or self._gateway_config.family_principals.get(principal_id)
                == memory_scope.private_tenant
            )
        )
        ordinary_correction = bool(
            requested_correction
            and isinstance(selected_binding, tuple)
            and len(selected_binding) == 2
            and selected_binding[0] is _SELECTED_SOURCE_AUTHORITY
            or requested_correction and native_reply_authorized
        )
        if message.metadata.get("_camera_ordinary_date_correction") is CAMERA_AUTHORITY:
            ingress = getattr(self, "_camera_ingress", None)
            camera_config = self._gateway_config.camera_ingress
            retained = [
                attempt for attempt in getattr(ingress, "_attempts", {}).values()
                if native_reply_target is not None
                and native_reply_target in {
                    str(attempt.get("photo_id")),
                    *map(str, attempt.get("reply_ids", [])),
                }
            ]
            attempt = retained[0] if len(retained) == 1 else None
            latest = attempt.get("camera_correction_commit") if isinstance(attempt, dict) else None
            retained_meal = bool(
                ingress is not None
                and camera_config.enabled
                and str(message.chat_id) == camera_config.chat_id
                and message.sender_id.split("|", 1)[0] == camera_config.principal
                and session_key == camera_config.session_key
                and len(retained) == 1
                and isinstance(attempt, dict)
                and attempt.get("state") == "completed"
                and attempt.get("finalizer_status") == "committed"
                and isinstance(attempt.get("camera_commit"), dict)
                and (
                    attempt.get("camera_correction") is None
                    or (
                        attempt.get("camera_correction") == "completed"
                        and isinstance(latest, dict)
                        and latest.get("kind") == "portion"
                    )
                )
            )
            changed_fields = (
                set(finalizer_nutrition.changed_fields)
                if finalizer_nutrition is not None else set()
            )
            if (
                not native_reply_authorized
                or not ordinary_correction
                or not retained_meal
                or finalizer_nutrition is None
                or finalizer_nutrition.record_type != "meal_correction"
                or not changed_fields
                or not changed_fields <= {"meal_at", "meal_date"}
                or not changed_fields & {"meal_at", "meal_date"}
                or ("meal_at" in changed_fields and finalizer_nutrition.meal_at is None)
                or ("meal_date" in changed_fields and finalizer_nutrition.meal_date is None)
            ):
                raise ValueError(
                    "Camera source-bound date correction requires an explicit date-only correction"
                )
        if message.metadata.get("_camera_context_meal_target") is _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY:
            candidate_id = message.metadata.get("_camera_context_meal_candidate_id")
            ingress = getattr(self, "_camera_ingress", None)
            binding = (
                ingress.contextual_meal_source_binding(
                    message, candidate_id=candidate_id, gateway_session_id=turn_ctx.session_id
                )
                if ingress is not None and isinstance(candidate_id, str)
                else None
            )
            changed_fields = (
                set(finalizer_nutrition.changed_fields)
                if finalizer_nutrition is not None else set()
            )
            if (
                finalizer_nutrition is not None
                and finalizer_nutrition.record_type == "meal_observation"
            ):
                raise ValueError(
                    "Camera-context nutrition must correct the retained meal, not append another meal"
                )
            target_kind = message.metadata.get("_camera_context_meal_target_kind")
            required_change = (
                "items" in changed_fields
                if target_kind == "item_addition"
                else (
                    "items" in changed_fields
                    and "energy_kcal_best" in changed_fields
                    and finalizer_nutrition.energy_kcal_best is not None
                    and not changed_fields & {"meal_at", "meal_date"}
                )
                if target_kind == "package_label"
                else bool(changed_fields & {"meal_at", "meal_date"})
            )
            if finalizer_nutrition is not None and finalizer_nutrition.record_type in {
                "meal_correction", "meal_deletion",
            }:
                if (
                    binding is None
                    or not ordinary_correction
                    or not isinstance(selected_binding, tuple)
                    or selected_binding[0] is not _SELECTED_SOURCE_AUTHORITY
                    or selected_binding[1] != binding
                    or finalizer_nutrition.record_type != "meal_correction"
                    or not changed_fields
                    or "consumption_status" in changed_fields
                    or finalizer_nutrition.explicit_new_consumption
                    or not required_change
                    or not changed_fields <= {
                        "basis", "meal_at", "meal_date", "is_estimate", "energy_kcal_min",
                        "energy_kcal_max", "energy_kcal_best", "protein_g", "fat_g",
                        "carbohydrate_g", "items", "confidence", "assumptions", "warnings",
                    }
                ):
                    raise ValueError(
                        "Camera-context correction must patch the one retained consumed meal"
                    )
        if camera_clarification:
            # A context-bound but unconfirmed answer may be an unrelated owner
            # turn. Preserve the model's ordinary response while keeping the
            # operation open; only a direct consumed-answer path gets the fixed ask.
            if message.metadata.get("_camera_context_hint") is not CAMERA_AUTHORITY:
                reply = "Сколько примерно вы съели? Можно указать количество или долю порции."
            message.metadata["_camera_clarification_final"] = CAMERA_AUTHORITY
        elif camera_yes and finalizer_nutrition is not None:
            reply = _append_nutrition_saved_status(reply, finalizer_nutrition)
        elif ordinary_meal:
            reply = _append_nutrition_saved_status(reply, finalizer_nutrition)
        elif ordinary_correction:
            pass

        self._maybe_schedule_memory_judge(
            bundle,
            session_key,
            turn_ctx=turn_ctx,
            memory_scope=memory_scope,
        )
        if reply:
            append_receipt = await self._append_conversation_turn(
                turn_ctx=turn_ctx,
                memory_scope=memory_scope,
                message=message,
                recorder=recorder,
                user_text=message.content or user_prompt,
                assistant_text=reply,
                camera_loaded_attachment_ids=frozenset(successful_attachment_ids),
                camera_original_attachment_ids=frozenset(camera_original_attachment_ids),
                camera_loaded_source_bindings=dict(successful_source_bindings),
            )
            camera_ingress = getattr(self, "_camera_ingress", None)
            camera_commit = None
            camera_correction_commit = None
            if camera_ingress is not None and turn_ctx.camera_authorized:
                candidate_id = message.metadata.get("_camera_candidate_id")
                attempt = camera_ingress._attempts.get(candidate_id)
                if isinstance(attempt, dict):
                    camera_commit = attempt.get("camera_commit")
                    camera_correction_commit = attempt.get("camera_correction_commit")
            camera_meal_saved = bool(
                append_receipt is not None
                and isinstance(camera_commit, dict)
                and isinstance(camera_commit.get("annotation"), Mapping)
                and camera_commit.get("event_id") == append_receipt.assistant_message_id
                and camera_commit.get("client_op_id") == append_receipt.assistant_client_op_id
            )
            correction_kind = (
                "portion" if camera_portion_correction else
                "denial" if (
                    message.metadata.get("_camera_correction") is CAMERA_AUTHORITY
                    and message.metadata.get("_camera_answer") == "no"
                ) else None
            )
            correction_metadata = (
                append_receipt.assistant_metadata
                if append_receipt is not None
                and isinstance(append_receipt.assistant_metadata, Mapping) else {}
            )
            candidate_id = message.metadata.get("_camera_candidate_id")
            original_event_id = camera_commit.get("event_id") if isinstance(camera_commit, dict) else None
            original_source_id = (
                camera_commit.get("source_message_id") if isinstance(camera_commit, dict) else None
            )
            camera_correction_saved = bool(
                correction_kind is not None
                and append_receipt is not None
                and isinstance(camera_correction_commit, dict)
                and isinstance(camera_commit, dict)
                and camera_commit.get("candidate_id") == candidate_id
                and camera_correction_commit.get("kind") == correction_kind
                and camera_correction_commit.get("event_id") == append_receipt.assistant_message_id
                and camera_correction_commit.get("client_op_id") == append_receipt.assistant_client_op_id
                and camera_correction_commit.get("source_message_id")
                == correction_metadata.get("source_message_id")
                and correction_metadata.get("source_message_id")
                == _normalize_source_message_ref(message.metadata.get("message_id"))
                and append_receipt.user_client_op_id
                == f"{camera_correction_commit.get('client_op_id', '').removesuffix(':assistant')}:user"
                and camera_correction_commit.get("target_event_id") == original_event_id
                and camera_correction_commit.get("target_source_message_id") == original_source_id
                and correction_metadata.get("role") == "assistant"
                and correction_metadata.get("client_op_id") == append_receipt.assistant_client_op_id
                and correction_metadata.get("camera_candidate_id") == candidate_id
                and correction_metadata.get("camera_operation_id") == candidate_id
                and correction_metadata.get("camera_original_event_id") == original_event_id
                and correction_metadata.get("reply_to_source_message_id") == original_source_id
                and correction_metadata.get("camera_answer_bound")
                == ("yes" if correction_kind == "portion" else "no")
                and correction_metadata.get("camera_correction_bound") is True
                and correction_metadata.get("gateway_session_id") == turn_ctx.session_id
                and isinstance(memory_scope, MemoryScope)
                and correction_metadata.get("tenant_id") == memory_scope.private_tenant
                and correction_metadata.get("source_principal")
                == f"telegram:{canonical_principal(message.channel, turn_ctx.principal)}"
                and correction_metadata.get("ingest_source") == "dropbox_camera"
                and correction_metadata.get("confirmation_required") is True
            )
            if camera_correction_saved:
                stored_trace = correction_metadata.get("decision_trace")
                stored_annotations = (
                    stored_trace.get("annotations") if isinstance(stored_trace, Mapping) else None
                )
                stored_nutrition = (
                    stored_annotations.get("nutrition")
                    if isinstance(stored_annotations, Mapping) else None
                )
                try:
                    stored_correction = NutritionAnnotationV2.model_validate(stored_nutrition)
                except (TypeError, ValueError):
                    camera_correction_saved = False
                else:
                    camera_correction_saved = bool(
                        stored_correction.record_type == "meal_correction"
                        and (
                            (
                                correction_kind == "denial"
                                and stored_correction.consumption_status == "not_consumed"
                                and "consumption_status" in stored_correction.changed_fields
                            )
                            or (
                                correction_kind == "portion"
                                and stored_correction.consumption_status == "unknown"
                                and "items" in stored_correction.changed_fields
                                and not {"consumption_status", "meal_at", "meal_date"}
                                & set(stored_correction.changed_fields)
                            )
                        )
                    )
            ordinary_meal_saved = bool(
                ordinary_meal and append_receipt is not None
                and isinstance(append_receipt.assistant_metadata, Mapping)
                and append_receipt.user_client_op_id == f"{append_receipt.assistant_metadata.get('logical_turn_id')}:user"
                and append_receipt.assistant_client_op_id == f"{append_receipt.assistant_metadata.get('logical_turn_id')}:assistant"
                and append_receipt.assistant_metadata.get("role") == "assistant"
                and append_receipt.assistant_metadata.get("client_op_id") == append_receipt.assistant_client_op_id
                and append_receipt.assistant_metadata.get("tenant_id") == memory_scope.private_tenant
                and append_receipt.assistant_metadata.get("source_principal") == f"{message.channel}:{canonical_principal(message.channel, turn_ctx.principal)}"
                and append_receipt.assistant_metadata.get("gateway_session_id") == turn_ctx.session_id
                and append_receipt.assistant_metadata.get("source_message_id") == _normalize_source_message_ref(message.metadata.get("message_id"))
                and append_receipt.assistant_metadata.get("received_at")
                == _trusted_utc_iso(_trusted_inbound_event_time(message))
                and append_receipt.assistant_metadata.get("is_group") is False
                and append_receipt.assistant_metadata.get("is_forwarded") is False
                and append_receipt.assistant_metadata.get("ingest_source") == "telegram"
                and append_receipt.assistant_metadata.get("confirmation_required") is False
                and _receipt_has_consumed_nutrition(append_receipt.assistant_metadata)
                and isinstance(append_receipt.assistant_content, str)
            )
            consumed_occurrence_source: Mapping[str, object] | None = None
            if ordinary_meal_saved and append_receipt is not None:
                stored_metadata = append_receipt.assistant_metadata
                if isinstance(stored_metadata, Mapping):
                    append_source = stored_metadata.get("source_message_id")
                    receipt_photo_source = stored_metadata.get("photo_occurrence_source")
                    locally_bound_photo_source = None
                    if isinstance(memory_scope, MemoryScope):
                        local_metadata = _build_conversation_turn_metadata(
                            turn_ctx=turn_ctx,
                            message=message,
                            scope=memory_scope,
                            recorder=recorder,
                        )[2]
                        candidate = local_metadata.get("photo_occurrence_source")
                        if isinstance(candidate, Mapping):
                            locally_bound_photo_source = dict(candidate)
                    current_photo_source = _current_verified_photo_occurrence_source(
                        getattr(bundle.engine, "messages", []), message=message,
                        receipt_metadata=stored_metadata, session_key=session_key,
                    )
                    occurrence_source = None
                    if (isinstance(receipt_photo_source, Mapping)
                            and locally_bound_photo_source is not None
                            and dict(receipt_photo_source) == locally_bound_photo_source):
                        occurrence_source = locally_bound_photo_source
                    elif (isinstance(receipt_photo_source, Mapping)
                          and current_photo_source is not None
                          and dict(receipt_photo_source) == current_photo_source):
                        occurrence_source = current_photo_source
                    elif receipt_photo_source is None:
                        occurrence_source = current_photo_source
                    if (
                        isinstance(occurrence_source, Mapping)
                        and isinstance(append_source, str)
                        and isinstance(occurrence_source.get("source_message_id"), str)
                        and occurrence_source.get("source_message_id")
                        and isinstance(occurrence_source.get("append_source_message_id"), str)
                        and occurrence_source.get("append_source_message_id")
                        and occurrence_source.get("tenant_id") == stored_metadata.get("tenant_id")
                        and occurrence_source.get("source_principal") == stored_metadata.get("source_principal")
                        and occurrence_source.get("gateway_session_id") == stored_metadata.get("gateway_session_id")
                        and _record_consumed_photo_occurrence(
                        getattr(bundle.engine, "messages", []),
                        occurrence_source,
                        receipt_event_id=append_receipt.assistant_message_id,
                        client_op_id=append_receipt.assistant_client_op_id,
                        append_source_message_id=append_source,
                        )
                    ):
                        consumed_occurrence_source = occurrence_source
                        await self._save_snapshot(bundle, session_key, user_prompt)
            expected_selected = (
                _build_conversation_turn_metadata(
                    turn_ctx=turn_ctx,
                    message=message,
                    scope=memory_scope,
                    recorder=recorder,
                )[1]
                if ordinary_correction
                and append_receipt is not None
                and isinstance(memory_scope, MemoryScope)
                else {}
            )
            context_attempt = None
            context_projection = None
            if (
                message.metadata.get("_camera_context_meal_target")
                is _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY
                and camera_ingress is not None
            ):
                context_candidate = message.metadata.get("_camera_context_meal_candidate_id")
                context_attempt = (
                    camera_ingress._attempts.get(context_candidate)
                    if isinstance(context_candidate, str) else None
                )
                context_projection = (
                    context_attempt.get("context_meal_projection")
                    if isinstance(context_attempt, dict) else None
                )
            exact_context_receipt_replay = bool(
                append_receipt is not None
                and _exact_context_receipt_replay(
                    context_projection, append_receipt, expected_selected,
                    expected_user_text=message.content or user_prompt,
                    expected_native_target=_normalize_source_message_ref(
                        message.metadata.get("reply_to_message_id")
                    ),
                )
                and isinstance(expected_selected.get("selected_source"), Mapping)
                and context_projection.get("source_message_id")
                == expected_selected["selected_source"].get("source_message_id")
            )
            context_receipt_evidence: dict[str, object] | None = None
            ordinary_correction_saved = bool(
                ordinary_correction
                and append_receipt is not None
                and isinstance(append_receipt.assistant_metadata, Mapping)
                and append_receipt.user_client_op_id == f"{append_receipt.assistant_metadata.get('logical_turn_id')}:user"
                and append_receipt.assistant_client_op_id == f"{append_receipt.assistant_metadata.get('logical_turn_id')}:assistant"
                and (
                    append_receipt.assistant_client_op_id
                    == f"{expected_selected.get('logical_turn_id')}:assistant"
                    or exact_context_receipt_replay
                )
                and append_receipt.assistant_metadata.get("role") == "assistant"
                and append_receipt.assistant_metadata.get("client_op_id") == append_receipt.assistant_client_op_id
                and isinstance(memory_scope, MemoryScope)
                and append_receipt.assistant_metadata.get("tenant_id") == memory_scope.private_tenant
                and append_receipt.assistant_metadata.get("source_principal") == f"{message.channel}:{canonical_principal(message.channel, turn_ctx.principal)}"
                and (
                    append_receipt.assistant_metadata.get("gateway_session_id") == turn_ctx.session_id
                    or exact_context_receipt_replay
                )
                and append_receipt.assistant_metadata.get("source_message_id") == _normalize_source_message_ref(message.metadata.get("message_id"))
                and append_receipt.assistant_metadata.get("reply_to_source_message_id") == expected_selected.get(
                    "reply_to_source_message_id"
                )
                and (
                    append_receipt.assistant_metadata.get("received_at")
                    == _trusted_utc_iso(_trusted_inbound_event_time(message))
                    or exact_context_receipt_replay
                )
                and append_receipt.assistant_metadata.get("is_forwarded") is False
                and append_receipt.assistant_metadata.get("is_group") is False
                and append_receipt.assistant_metadata.get("ingest_source") == "telegram"
                and append_receipt.assistant_metadata.get("confirmation_required") is False
                and append_receipt.assistant_metadata.get("target_meal_id") == expected_selected.get("target_meal_id")
                and append_receipt.assistant_metadata.get("selected_source") == expected_selected.get("selected_source")
                and isinstance(append_receipt.assistant_content, str)
            )
            committed_annotation = None
            if ordinary_meal_saved and append_receipt is not None:
                stored_trace = append_receipt.assistant_metadata.get("decision_trace")
                stored_annotations = (
                    stored_trace.get("annotations")
                    if isinstance(stored_trace, Mapping) else None
                )
                stored_nutrition = (
                    stored_annotations.get("nutrition")
                    if isinstance(stored_annotations, Mapping) else None
                )
                try:
                    committed_annotation = NutritionAnnotationV2.model_validate(stored_nutrition)
                except (TypeError, ValueError):
                    ordinary_meal_saved = False
                else:
                    # A durable retry may carry a different model proposal.
                    # The final answer must describe the exact event already saved.
                    meal_status = (
                        "Записано. Баланс обновляется."
                        if committed_annotation.meal_at is not None
                        or committed_annotation.meal_date is not None
                        else "Записано; приём пищи пока не привязан к дате."
                    )
                    reply = _committed_nutrition_reply(
                        committed_annotation,
                        status=meal_status,
                        assistant_content=append_receipt.assistant_content,
                    )
            if ordinary_correction_saved and append_receipt is not None:
                stored_trace = append_receipt.assistant_metadata.get("decision_trace")
                stored_annotations = stored_trace.get("annotations") if isinstance(stored_trace, Mapping) else None
                stored_nutrition = stored_annotations.get("nutrition") if isinstance(stored_annotations, Mapping) else None
                try:
                    stored_correction = NutritionAnnotationV2.model_validate(stored_nutrition)
                except (TypeError, ValueError):
                    ordinary_correction_saved = False
                else:
                    if stored_correction.record_type not in {"meal_correction", "meal_deletion"}:
                        ordinary_correction_saved = False
                    else:
                        if (
                            message.metadata.get("_camera_context_meal_target")
                            is _CAMERA_CONTEXT_MEAL_TARGET_AUTHORITY
                        ):
                            candidate_id = message.metadata.get(
                                "_camera_context_meal_candidate_id"
                            )
                            if (
                                camera_ingress is None
                                or not isinstance(candidate_id, str)
                                or not camera_ingress.record_contextual_meal_projection(
                                    message, candidate_id, append_receipt,
                                    stored_correction,
                                )
                            ):
                                ordinary_correction_saved = False
                                reply = "Не удалось подтвердить сохранение изменения."
                                continue_context_projection = False
                            else:
                                continue_context_projection = True
                                context_receipt_evidence = camera_ingress.contextual_meal_receipt_evidence(
                                    message, candidate_id, append_receipt,
                                    stored_correction,
                                )
                                if context_receipt_evidence is None:
                                    ordinary_correction_saved = False
                                    reply = "Не удалось подтвердить сохранение изменения."
                                    continue_context_projection = False
                        else:
                            continue_context_projection = True
                        if not continue_context_projection:
                            pass
                        else:
                            reply = (
                                "Удаление сохранено; баланс обновляется."
                                if stored_correction.record_type == "meal_deletion"
                                else "Изменение сохранено; баланс обновляется."
                            )
            if camera_ingress is not None and turn_ctx.camera_authorized:
                camera_ingress.complete(
                    message,
                    recorded=camera_meal_saved or camera_correction_saved,
                    clarification=(
                        camera_clarification
                        or message.metadata.get("_camera_context_question")
                        is CAMERA_CONTEXT_QUESTION_AUTHORITY
                    ),
                )
            logger.info(
                "ohmo runtime processing complete session_key=%s session_id=%s reply=%r",
                session_key,
                bundle.session_id,
                _content_snippet(reply),
            )
            final_media = _extract_final_reply_media(reply, emitted_media)
            metadata: dict[str, object] = {
                "_session_key": session_key,
                **self._camera_final_delivery_metadata(message),
            }
            if camera_meal_saved and isinstance(camera_commit, dict):
                committed_camera_annotation = NutritionAnnotationV2.model_validate(
                    camera_commit["annotation"]
                )
                metadata.update(
                    nutrition_append_event_id=camera_commit["event_id"],
                    nutrition_sync_status="pending",
                    nutrition_committed_annotation=_nutrition_annotation_metadata(
                        committed_camera_annotation
                    ),
                    nutrition_model_proposal_annotation=_nutrition_annotation_metadata(
                        committed_camera_annotation
                    ),
                    nutrition_proposal_matches_committed=True,
                    nutrition_consumed_occurrence={
                        "schema_version": 1,
                        "tenant_id": camera_commit["tenant_id"],
                        "source_principal": f"telegram:{camera_commit['principal']}",
                        "gateway_session_id": camera_commit["gateway_session_id"],
                        "photo_source_message_id": camera_commit["source_message_id"],
                        "append_source_message_id": camera_commit["source_message_id"],
                        "photo_received_at": camera_commit["photo_received_at"],
                        "receipt_event_id": camera_commit["event_id"],
                        "client_op_id": camera_commit["client_op_id"],
                    },
                )
            elif message.metadata.get("_camera_correction") is CAMERA_AUTHORITY:
                if camera_correction_saved and isinstance(camera_correction_commit, dict):
                    reply = "Изменение сохранено; баланс обновляется."
                    metadata.update(
                        nutrition_append_event_id=camera_correction_commit["event_id"],
                        nutrition_sync_status="pending",
                    )
                else:
                    reply = "Не удалось подтвердить сохранение изменения."
            elif camera_yes and finalizer_nutrition is not None:
                reply = "Не удалось подтвердить сохранение записи."
            elif ordinary_meal:
                if ordinary_meal_saved and append_receipt is not None:
                    metadata.update(
                        nutrition_append_event_id=append_receipt.assistant_message_id,
                        nutrition_sync_status="pending",
                        nutrition_committed_annotation=(
                            _nutrition_annotation_metadata(committed_annotation)
                            if committed_annotation is not None else None
                        ),
                        nutrition_model_proposal_annotation=(
                            _nutrition_annotation_metadata(finalizer_nutrition)
                            if finalizer_nutrition is not None else None
                        ),
                        nutrition_proposal_matches_committed=(
                            finalizer_nutrition.model_dump(mode="json")
                            == committed_annotation.model_dump(mode="json")
                            if finalizer_nutrition is not None
                            and committed_annotation is not None else False
                        ),
                    )
                    occurrence = consumed_occurrence_source
                    if isinstance(occurrence, Mapping):
                        # This is emitted only after the persisted receipt passed
                        # the full observation checks above and the occurrence
                        # was attached to exactly one historical photo reference.
                        if any(
                            isinstance(block, AttachmentRefBlock)
                            and block.attachment_id == occurrence.get("attachment_id")
                            and isinstance(block.source_provenance, dict)
                            and block.source_provenance.get("source_message_id") == occurrence.get("source_message_id")
                            and block.source_provenance.get("received_at") == occurrence.get("received_at")
                            and any(
                                isinstance(item, Mapping)
                                and item.get("receipt_event_id") == append_receipt.assistant_message_id
                                and item.get("client_op_id") == append_receipt.assistant_client_op_id
                                and item.get("append_source_message_id")
                                == append_receipt.assistant_metadata.get("source_message_id")
                                for item in block.source_provenance.get("consumed_occurrences", [])
                            )
                            for historical in getattr(bundle.engine, "messages", [])
                            for block in historical.content
                        ):
                            metadata["nutrition_consumed_occurrence"] = {
                                "schema_version": 1,
                                "tenant_id": occurrence.get("tenant_id"),
                                "source_principal": occurrence.get("source_principal"),
                                "gateway_session_id": occurrence.get("gateway_session_id"),
                                "photo_source_message_id": occurrence.get("source_message_id"),
                                "append_source_message_id": append_receipt.assistant_metadata.get("source_message_id"),
                                "photo_received_at": occurrence.get("received_at"),
                                "receipt_event_id": append_receipt.assistant_message_id,
                                "client_op_id": append_receipt.assistant_client_op_id,
                            }
                else:
                    reply = "Не удалось подтвердить сохранение записи."
            elif requested_correction:
                if ordinary_correction_saved and append_receipt is not None:
                    committed_annotation = None
                    stored_trace = append_receipt.assistant_metadata.get("decision_trace")
                    stored_annotations = stored_trace.get("annotations") if isinstance(stored_trace, Mapping) else None
                    stored_nutrition = stored_annotations.get("nutrition") if isinstance(stored_annotations, Mapping) else None
                    try:
                        committed_annotation = NutritionAnnotationV2.model_validate(stored_nutrition)
                    except (TypeError, ValueError):
                        committed_annotation = None
                    metadata.update(
                        nutrition_append_event_id=append_receipt.assistant_message_id,
                        nutrition_sync_status="pending",
                    )
                    if committed_annotation is not None:
                        metadata["nutrition_committed_annotation"] = _nutrition_annotation_metadata(
                            committed_annotation
                        )
                    metadata["nutrition_model_proposal_annotation"] = (
                        _nutrition_annotation_metadata(finalizer_nutrition)
                        if finalizer_nutrition is not None else None
                    )
                    metadata["nutrition_proposal_matches_committed"] = (
                        finalizer_nutrition.model_dump(mode="json")
                        == committed_annotation.model_dump(mode="json")
                        if finalizer_nutrition is not None and committed_annotation is not None else False
                    )
                    if context_receipt_evidence is not None:
                        metadata["nutrition_context_evidence"] = context_receipt_evidence
                    if (committed_annotation is not None
                            and isinstance(selected_binding, tuple) and len(selected_binding) == 2
                            and selected_binding[0] is _SELECTED_SOURCE_AUTHORITY
                            and isinstance(selected_binding[1], Mapping)):
                        binding = selected_binding[1]
                        occurrence_matches = []
                        attachment_id = binding.get("attachment_id")
                        if isinstance(attachment_id, str):
                            for historical in getattr(bundle.engine, "messages", []):
                                for block in historical.content:
                                    if not isinstance(block, AttachmentRefBlock):
                                        continue
                                    provenance = block.source_provenance
                                    occurrences = provenance.get("consumed_occurrences") if isinstance(provenance, Mapping) else None
                                    if (block.attachment_id != attachment_id or not isinstance(occurrences, list)
                                            or provenance.get("source_message_id") != binding.get("source_message_id")):
                                        continue
                                    occurrence_matches.extend(
                                        item for item in occurrences if isinstance(item, Mapping)
                                        and item.get("append_source_message_id") == binding.get("append_source_message_id")
                                    )
                        if len(occurrence_matches) == 1:
                            occurrence = occurrence_matches[0]
                            metadata["nutrition_context_evidence"] = {
                                "schema_version": 1,
                                "tenant_id": binding.get("tenant_id"),
                                "source_principal": binding.get("source_principal"),
                                "gateway_session_id": binding.get("gateway_session_id"),
                                "photo_source_message_id": binding.get("source_message_id"),
                                "photo_received_at": binding.get("received_at"),
                                "consumed_source_message_id": binding.get("append_source_message_id"),
                                "target_meal_id": append_receipt.assistant_metadata.get("target_meal_id"),
                                "original_receipt_event_id": occurrence.get("receipt_event_id"),
                                "original_operation_id": occurrence.get("client_op_id"),
                                "current_receipt_event_id": append_receipt.assistant_message_id,
                                "current_operation_id": append_receipt.assistant_client_op_id,
                                "current_logical_turn_id": append_receipt.assistant_metadata.get("logical_turn_id"),
                                "current_trace_episode_id": append_receipt.assistant_metadata.get("decision_trace_episode_id"),
                            }
                else:
                    reply = "Не удалось подтвердить сохранение изменения."
            if final_media:
                metadata.update({"_media": final_media, "_final_media_fallback": True})
            message.metadata.pop("_selected_source_binding", None)
            yield GatewayStreamUpdate(
                kind="final",
                text=reply,
                metadata=metadata,
                media=final_media or None,
            )
            if todo_lifecycle:
                try:
                    cleanup = self._todo_cleanup_update(bundle=bundle, session_key=session_key)
                    if cleanup is not None:
                        yield cleanup
                except Exception:
                    logger.warning(
                        "ohmo.todo.cleanup_failure session_id=%s",
                        bundle.session_id,
                        exc_info=True,
                    )

    async def _append_conversation_turn(
        self,
        *,
        turn_ctx: TurnContext,
        memory_scope: MemoryScope | None | object = _UNRESOLVED_MEMORY_SCOPE,
        message: InboundMessage,
        recorder: GatewayEvalRecorder | None = None,
        user_text: str,
        assistant_text: str,
        camera_loaded_attachment_ids: frozenset[str] = frozenset(),
        camera_original_attachment_ids: frozenset[str] = frozenset(),
        camera_loaded_source_bindings: Mapping[str, Mapping[str, object]] | None = None,
    ) -> ConversationAppendReceipt | None:
        # A Camera analysis turn is model-visible, but the producer is not the
        # configured account. Do not append it as if the user wrote it.
        if turn_ctx.camera_authorized and turn_ctx.principal == "__camera__":
            return None
        camera_bound_answer = (
            turn_ctx.camera_authorized
            and turn_ctx.principal == self._gateway_config.camera_ingress.principal
            and message.metadata.get("_camera_answer") in {"yes", "no"}
            and message.metadata.get("_camera_authority") is CAMERA_AUTHORITY
        )
        camera_portion_correction = (
            message.metadata.get("_camera_portion_correction") is CAMERA_AUTHORITY
        )
        camera_yes = (
            camera_bound_answer
            and message.metadata.get("_camera_answer") == "yes"
            and not camera_portion_correction
        )
        camera_correction = (
            message.metadata.get("_camera_authority") is CAMERA_AUTHORITY
            and message.metadata.get("_camera_correction") is CAMERA_AUTHORITY
            and (
                message.metadata.get("_camera_answer") == "no"
                or camera_portion_correction
            )
        )
        selected_binding = message.metadata.get("_selected_source_binding")
        camera_context_question = (
            turn_ctx.camera_authorized
            and turn_ctx.principal == self._gateway_config.camera_ingress.principal
            and message.metadata.get("_camera_context_question")
            is CAMERA_CONTEXT_QUESTION_AUTHORITY
        )
        annotation = recorder.validated_nutrition_envelope if recorder is not None else None
        validated = (
            NutritionAnnotationV2.model_validate(annotation)
            if isinstance(annotation, Mapping) and annotation.get("schema_version", 1) == 2
            else None
        )
        ordinary_correction = bool(
            validated is not None
            and validated.record_type in {"meal_correction", "meal_deletion"}
            and isinstance(selected_binding, tuple)
            and len(selected_binding) == 2
            and selected_binding[0] is _SELECTED_SOURCE_AUTHORITY
        )
        native_reply_candidate = bool(
            validated is not None
            and validated.record_type in {"meal_correction", "meal_deletion"}
            and not camera_correction
            and turn_ctx.channel == "telegram"
            and message.channel == "telegram"
            and turn_ctx.is_private
            and not turn_ctx.is_forwarded
            and is_private_message(message)
            and _normalize_source_message_ref(message.metadata.get("reply_to_message_id"))
            is not None
        )
        if (
            validated is not None
            and validated.record_type in {"meal_correction", "meal_deletion"}
            and not camera_correction
            and not ordinary_correction
            and not native_reply_candidate
        ):
            return None
        if camera_context_question:
            ingress = getattr(self, "_camera_ingress", None)
            camera_config = self._gateway_config.camera_ingress
            context_scope = self._coerce_memory_scope(turn_ctx, memory_scope)
            if (
                ingress is None
                or not turn_ctx.is_private
                or turn_ctx.is_forwarded
                or turn_ctx.channel != "telegram"
                or str(turn_ctx.principal).split("|", 1)[0] != camera_config.principal
                or str(turn_ctx.chat_id) != camera_config.chat_id
                or context_scope is None
                or context_scope.private_tenant != camera_config.tenant_id
            ):
                raise ValueError("Camera context question is not authorized in the current owner scope")
            ingress.validate_context_question_binding(
                message,
                principal=camera_config.principal,
                chat_id=camera_config.chat_id,
                tenant_id=context_scope.private_tenant,
            )
            if (
                recorder is None
                or annotation is not None
                or recorder.nutrition_annotation_status not in {"missing", "not_applicable"}
            ):
                raise ValueError("Camera context question requires a valid non-consumed finalizer")
        if camera_yes:
            camera_ingress = getattr(self, "_camera_ingress", None)
            trusted_meal_at = (
                camera_ingress.trusted_capture_time_for_answer(message)
                if camera_ingress is not None
                else None
            )
            if trusted_meal_at is None:
                raise ValueError("Camera consumed meal requires trusted capture time")
            if annotation is not None and camera_loaded_attachment_ids:
                current_message_id = _normalize_source_message_ref(
                    message.metadata.get("message_id")
                )
                if (
                    len(camera_loaded_attachment_ids) != 1
                    or camera_loaded_attachment_ids != camera_original_attachment_ids
                ):
                    raise ValueError(
                        "Camera consumed meal selected an image outside the admitted original"
                    )
                loaded_id = next(iter(camera_loaded_attachment_ids))
                loaded_binding = (camera_loaded_source_bindings or {}).get(loaded_id)
                if loaded_binding is not None:
                    if (
                        loaded_binding.get("attachment_id") != loaded_id
                        or loaded_binding.get("source_message_id") != current_message_id
                        or loaded_binding.get("append_source_message_id") != current_message_id
                        or not isinstance(memory_scope, MemoryScope)
                        or loaded_binding.get("tenant_id") != memory_scope.private_tenant
                        or loaded_binding.get("gateway_session_id") != turn_ctx.session_id
                        or loaded_binding.get("source_principal")
                        != f"telegram:{canonical_principal('telegram', turn_ctx.principal)}"
                    ):
                        raise ValueError(
                            "Camera consumed meal selected an image with a conflicting source"
                        )
                    message.metadata["_selected_source_binding"] = (
                        _SELECTED_SOURCE_AUTHORITY, loaded_binding
                    )
            if annotation is None:
                if (
                    recorder is None
                    or message.metadata.get("_camera_clarification_allowed") is not CAMERA_AUTHORITY
                    or recorder.nutrition_annotation_status not in {"missing", "not_applicable"}
                ):
                    raise ValueError(
                        "Camera clarification requires a valid missing nutrition trace"
                    )
            else:
                assert validated is not None
                if (
                    validated.record_type != "meal_observation"
                    or validated.consumption_status != "consumed"
                    or "image" not in validated.basis
                ):
                    raise ValueError(
                        "Camera meal finalization requires a consumed image observation"
                    )
                if validated.meal_at != trusted_meal_at or validated.meal_date is not None:
                    raise ValueError(
                        "Camera consumed meal requires authoritative meal_at without meal_date"
                    )
                if validated.explicit_new_consumption:
                    raise ValueError(
                        "Camera consumption cannot override exact-image replay identity"
                    )
        if camera_correction:
            ingress = getattr(self, "_camera_ingress", None)
            candidate_id = message.metadata.get("_camera_candidate_id")
            attempt = ingress._attempts.get(candidate_id) if ingress is not None else None
            target = attempt.get("camera_commit") if attempt is not None else None
            if not isinstance(target, dict) or annotation is None:
                raise ValueError("Camera denial requires a retained committed meal target")
            assert validated is not None
            if camera_portion_correction:
                _validate_camera_portion_correction_annotation(annotation)
            elif (
                validated.record_type != "meal_correction"
                or validated.consumption_status != "not_consumed"
                or "consumption_status" not in validated.changed_fields
            ):
                raise ValueError("Camera denial requires a validated not_consumed correction")
        if self._gateway_config.conversation_learning is not True:
            if camera_context_question:
                raise ValueError("Camera context clarification requires durable conversation learning")
            return
        scope = self._coerce_memory_scope(turn_ctx, memory_scope)
        if scope is None:
            if camera_context_question:
                raise ValueError("Camera context clarification has no authorized memory scope")
            return
        if not self._honcho_turn_allowed(turn_ctx, scope):
            if camera_context_question:
                raise ValueError("Camera context clarification is not authorized for append")
            return
        if native_reply_candidate and not ordinary_correction:
            ordinary_correction = self._honcho_turn_allowed(turn_ctx, scope)
        if (
            validated is not None
            and validated.record_type in {"meal_correction", "meal_deletion"}
            and not camera_correction
            and not ordinary_correction
        ):
            return
        if camera_correction:
            ingress = getattr(self, "_camera_ingress", None)
            camera_config = self._gateway_config.camera_ingress
            if (
                ingress is None
                or not turn_ctx.camera_authorized
                or not turn_ctx.is_private
                or turn_ctx.is_forwarded
                or turn_ctx.channel != "telegram"
                or str(turn_ctx.principal).split("|", 1)[0] != camera_config.principal
                or str(turn_ctx.chat_id) != camera_config.chat_id
                or scope.private_tenant != camera_config.tenant_id
            ):
                raise ValueError("Camera correction is not authorized in the current memory scope")
            ingress.validate_correction_binding(
                message,
                principal=camera_config.principal,
                chat_id=camera_config.chat_id,
                tenant_id=scope.private_tenant,
            )
        shadow_backend = self._shadow_backend_for_scope(scope)
        if shadow_backend is None:
            if camera_context_question:
                raise ConversationReconciliationError("Camera context receipt is unavailable")
            return
        _, user_metadata, assistant_metadata = _build_conversation_turn_metadata(
            turn_ctx=turn_ctx,
            message=message,
            scope=scope,
            recorder=recorder,
        )
        if (
            ordinary_correction
            and not assistant_metadata.get("target_meal_id")
            and not _normalize_source_message_ref(message.metadata.get("reply_to_message_id"))
        ):
            return None
        if (
            message.channel == "telegram"
            and turn_ctx.is_private
            and not turn_ctx.is_forwarded
            and not turn_ctx.camera_authorized
            and message.sender_id != "__camera__"
        ):
            # Provenance is generated only after the actual memory resolver
            # authorized this private owner/family append.
            for metadata in (user_metadata, assistant_metadata):
                metadata.update(ingest_source="telegram", confirmation_required=False)
        if camera_bound_answer:
            for metadata in (user_metadata, assistant_metadata):
                metadata["camera_candidate_id"] = message.metadata["_camera_candidate_id"]
                metadata["camera_answer_bound"] = message.metadata["_camera_answer"]
                metadata["camera_route"] = message.metadata.get("_camera_route", "context")
                metadata["camera_operation_id"] = message.metadata["_camera_candidate_id"]
                native_binding = message.metadata.get("_camera_native_binding")
                if isinstance(native_binding, str):
                    metadata["camera_reply_to_native_message_id"] = native_binding
        if camera_correction:
            attempt = self._camera_ingress._attempts[message.metadata["_camera_candidate_id"]]
            original = attempt["camera_commit"]
            for metadata in (user_metadata, assistant_metadata):
                metadata["camera_original_event_id"] = original["event_id"]
                metadata["reply_to_source_message_id"] = original["source_message_id"]
                metadata["camera_correction_bound"] = True
        receipt = await shadow_backend.append_exchange(
            user_text,
            assistant_text,
            user_metadata=user_metadata,
            assistant_metadata=assistant_metadata,
            durable=(
                camera_bound_answer
                or camera_context_question
                or ordinary_correction
                or (
                    validated is not None
                    and validated.record_type == "meal_observation"
                    and validated.consumption_status == "consumed"
                )
            ),
        )
        if camera_yes and validated is not None:
            self._camera_ingress.record_committed_meal(
                message, receipt, validated.model_dump(mode="json")
            )
        elif camera_correction:
            self._camera_ingress.record_committed_correction(
                message, receipt, annotation
            )
        return receipt

    async def _convert_stream_event(
        self,
        *,
        event,
        bundle: RuntimeBundle,
        message: InboundMessage,
        session_key: str,
        content: str,
        reply_parts: list[str],
        recorder: GatewayEvalRecorder | None = None,
    ):
        if isinstance(event, AssistantTextDelta):
            reply_parts.append(event.text)
            return
        if isinstance(event, CompactProgressEvent):
            logger.info(
                "ohmo runtime compact progress session_key=%s session_id=%s phase=%s trigger=%s attempt=%s",
                session_key,
                bundle.session_id,
                event.phase,
                event.trigger,
                event.attempt,
            )
            rendered = _format_channel_progress(
                channel=message.channel,
                kind="compact_progress",
                text=event.message or "",
                session_key=session_key,
                content=content,
                compact_phase=event.phase,
                compact_trigger=event.trigger,
                attempt=event.attempt,
            )
            if rendered:
                yield GatewayStreamUpdate(
                    kind="progress",
                    text=rendered,
                    metadata={"_progress": True, "_session_key": session_key, "_compact": True},
                )
            return
        if isinstance(event, StatusEvent):
            logger.info(
                "ohmo runtime status session_key=%s session_id=%s message=%r",
                session_key,
                bundle.session_id,
                _content_snippet(event.message),
            )
            yield GatewayStreamUpdate(
                kind="progress",
                text=_format_channel_progress(
                    channel=message.channel,
                    kind="status",
                    text=event.message,
                    session_key=session_key,
                    content=content,
                ),
                metadata={"_progress": True, "_session_key": session_key},
            )
            return
        if isinstance(event, ToolExecutionStarted):
            if recorder is not None:
                recorder.record_tool_started(event)
            # AssistantTextDelta is provider-public output, even if a tool call
            # follows it.  Keep this turn's text out of the next turn's final
            # accumulation, but deliver it as a durable assistant update rather
            # than relabelling it as hidden reasoning/progress.
            pending_assistant_text = "".join(reply_parts).strip()
            # The public update also supplies a bounded action purpose for the
            # compact tool row; it is not chain-of-thought.
            purpose = _normalize_tool_purpose(pending_assistant_text)
            if pending_assistant_text:
                reply_parts.clear()
                yield GatewayStreamUpdate(
                    kind="assistant_update",
                    text=pending_assistant_text,
                    metadata={"_assistant_update": True, "_session_key": session_key},
                )
            summary = _summarize_tool_input(event.tool_name, event.tool_input)
            logger.info(
                "ohmo runtime tool start session_key=%s session_id=%s tool=%s summary=%r",
                session_key,
                bundle.session_id,
                event.tool_name,
                summary,
            )
            if event.tool_name == "todo_write":
                # Don't show the raw per-item JSON — the full checklist is
                # rendered (post-write) on completion instead, like a todo panel.
                return
            hint = _pretty_tool_name(event.tool_name)
            cid = _short_call_id(event.tool_call_id)
            if cid:
                hint = f"{hint} — {cid}"  # short tool_use id to match the result hint
            args_block = _format_tool_args_block(event.tool_input)
            if args_block:
                hint = f"{hint}\n{args_block}"
            yield GatewayStreamUpdate(
                kind="tool_hint",
                text=_format_channel_progress(
                    channel=message.channel,
                    kind="tool_hint",
                    text=hint,
                    session_key=session_key,
                    content=content,
                ),
                metadata={
                    "_progress": True,
                    "_tool_hint": True,
                    "_session_key": session_key,
                    "progress_event": _tool_progress_event(
                        tool_name=event.tool_name,
                        tool_call_id=event.tool_call_id,
                        display_label=_pretty_tool_name(event.tool_name),
                        phase="started",
                        status="running",
                        purpose=purpose,
                    ),
                },
            )
            return
        if isinstance(event, ToolExecutionCompleted):
            if recorder is not None:
                recorder.record_tool_completed(event)
            logger.info(
                "ohmo runtime tool complete session_key=%s session_id=%s tool=%s is_error=%s",
                session_key,
                bundle.session_id,
                event.tool_name,
                event.is_error,
            )
            if event.tool_name == "todo_write":
                metadata = event.metadata if isinstance(event.metadata, dict) else {}
                changed = metadata.get("changed")
                if not event.is_error and changed is False:
                    logger.info(
                        "ohmo.todo.write.noop session_id=%s",
                        bundle.session_id,
                    )
                todos = metadata.get("todos")
                canonical_todos = None
                if not event.is_error and changed is True and isinstance(todos, list):
                    try:
                        canonical_todos = canonicalize_todos(todos)
                    except (TypeError, ValueError):
                        canonical_todos = None
                if canonical_todos is not None:
                    blocked_count = sum(item.get("status") == "blocked" for item in canonical_todos)
                    if blocked_count:
                        logger.info(
                            "ohmo.todo.blocked.persisted session_id=%s count=%s",
                            bundle.session_id,
                            blocked_count,
                        )
                    yield GatewayStreamUpdate(
                        kind="progress",
                        text="",
                        metadata={
                            "_progress": True,
                            "_session_key": session_key,
                            "progress_event": {
                                "kind": "todo",
                                "todos": canonical_todos,
                                "changed": True,
                                "session_id": str(bundle.session_id),
                            },
                        },
                    )
                return
            # Edit the in-flight progress message to show the OUTCOME: ✅/❌ plus a
            # truncated output — the completion-side mirror of the params hint.
            yield GatewayStreamUpdate(
                kind="tool_hint",
                text=_format_channel_progress(
                    channel=message.channel,
                    kind="tool_hint",
                    text=_format_tool_done(
                        event.tool_name, event.output, event.is_error, event.tool_call_id
                    ),
                    session_key=session_key,
                    content=content,
                ),
                metadata={
                    "_progress": True,
                    "_tool_hint": True,
                    "_session_key": session_key,
                    "progress_event": _tool_progress_event(
                        tool_name=event.tool_name,
                        tool_call_id=event.tool_call_id,
                        display_label=_pretty_tool_name(event.tool_name),
                        phase="completed",
                        status=_tool_completion_status(event),
                    ),
                },
            )
            media = _extract_tool_media(event)
            if media:
                yield GatewayStreamUpdate(
                    kind="media",
                    text=_format_tool_media_caption(event, media),
                    metadata={"_session_key": session_key, "_media": media, "_tool_media": True},
                    media=media,
                )
            return
        if isinstance(event, ErrorEvent):
            if recorder is not None:
                recorder.record_engine_error(event)
            logger.error(
                "ohmo runtime error session_key=%s session_id=%s message=%r",
                session_key,
                bundle.session_id,
                _content_snippet(event.message),
            )
            yield GatewayStreamUpdate(
                kind="error",
                text=event.message,
                metadata={"_session_key": session_key},
            )
            return
        if isinstance(event, AssistantTurnComplete):
            if recorder is not None and event.usage is not None:
                recorder.record_model_call(
                    event,
                    model=str(bundle.current_settings().model or ""),
                )
            if not any(part.strip() for part in reply_parts):
                completed_text = event.message.text.strip()
                if completed_text:
                    reply_parts.append(completed_text)

    async def _save_snapshot(
        self, bundle: RuntimeBundle, session_key: str, user_prompt: str
    ) -> None:
        tool_metadata = _sanitize_group_command_metadata(
            getattr(bundle.engine, "tool_metadata", {}) or {}
        )
        if isinstance(getattr(bundle.engine, "tool_metadata", None), dict) and isinstance(
            tool_metadata, dict
        ):
            bundle.engine.tool_metadata.update(tool_metadata)
        messages = self._attachment_store.externalize_messages(
            _sanitize_group_command_prompts(list(bundle.engine.messages))
        )
        self._attachment_store.assert_externalized(messages)
        if messages != list(bundle.engine.messages):
            if hasattr(bundle.engine, "load_messages"):
                bundle.engine.load_messages(messages)
            else:
                bundle.engine.messages = messages
        active_system_prompt = getattr(bundle.engine, "system_prompt", None)
        if not isinstance(active_system_prompt, str):
            active_system_prompt = await self._runtime_system_prompt(bundle, user_prompt)
        self._session_backend.save_snapshot(
            cwd=getattr(bundle, "cwd", self._cwd),
            model=bundle.current_settings().model,
            system_prompt=active_system_prompt,
            messages=messages,
            usage=bundle.engine.total_usage,
            session_id=bundle.session_id,
            session_key=session_key,
            tool_metadata=tool_metadata,
        )
        logger.info(
            "ohmo runtime saved snapshot session_key=%s session_id=%s message_count=%s",
            session_key,
            bundle.session_id,
            len(bundle.engine.messages),
        )

    async def _refresh_bundle(
        self,
        session_key: str,
        bundle: RuntimeBundle,
        latest_user_prompt: str | None,
        *,
        turn_ctx: TurnContext | None = None,
        memory_scope: MemoryScope | None | object = _UNRESOLVED_MEMORY_SCOPE,
        include_todo: bool = True,
    ) -> RuntimeBundle:
        snapshot = sanitize_conversation_messages(list(bundle.engine.messages))
        prior_session_id = bundle.session_id
        todo_prompt_read_failure_logged = getattr(bundle, "_todo_prompt_read_failure_logged", False)
        bundle_cwd = str(Path(getattr(bundle, "cwd", self._cwd)).resolve())
        scope = self._coerce_memory_scope(turn_ctx, memory_scope)
        engaged = scope is not None
        await close_runtime(bundle)
        refreshed = await build_runtime(
            cwd=bundle_cwd,
            model=self._model,
            max_turns=self._max_turns,
            effort=self._effort,
            system_prompt=build_ohmo_system_prompt(
                bundle_cwd,
                workspace=self._workspace,
                extra_prompt=None,
                include_ohmo_memory=False,
            ),
            active_profile=self._provider_profile,
            session_backend=self._session_backend,
            enforce_max_turns=True,  # cap each prompt at settings.max_turns by default (was unlimited)
            restore_messages=[
                _snapshot_message_dict(message)
                for message in _sanitize_group_command_prompts(snapshot)
            ],
            restore_tool_metadata=_sanitize_group_command_metadata(
                getattr(bundle.engine, "tool_metadata", {}) or {}
            ),
            extra_skill_dirs=(str(get_skills_dir(self._workspace)),),
            extra_plugin_roots=(str(get_plugins_dir(self._workspace)),),
            memory_backend=create_memory_command_backend(
                self._workspace,
                backend_kind=self._gateway_config.memory_backend,
            ),
            include_project_memory=False,
            autodream_context=self._autodream_context() if engaged else None,
        )
        refreshed.session_id = prior_session_id
        self._configure_attachment_boundary(refreshed)
        self._register_gateway_tools(refreshed, memory_engaged=engaged)
        self._configure_turn_memory_surfaces(
            refreshed,
            turn_ctx,
            memory_scope=scope,
        )
        await start_runtime(refreshed)
        refreshed._todo_prompt_read_failure_logged = todo_prompt_read_failure_logged
        refreshed.engine.set_system_prompt(
            await self._runtime_system_prompt(
                refreshed,
                latest_user_prompt,
                turn_ctx=turn_ctx,
                memory_scope=scope,
                include_todo=include_todo,
            )
        )
        if hasattr(refreshed.engine, "set_cache_key"):
            refreshed.engine.set_cache_key(refreshed.session_id)
        setattr(refreshed, "_gateway_config_generation", self._gateway_config_generation)
        self._bundles[session_key] = refreshed
        logger.info(
            "ohmo runtime refreshed session_key=%s session_id=%s message_count=%s",
            session_key,
            refreshed.session_id,
            len(refreshed.engine.messages),
        )
        return refreshed

    def _append_todo_runtime_section(self, bundle: RuntimeBundle, prompt: str) -> str:
        """Append the trusted, deterministic todo snapshot for the live session."""
        session_id = str(getattr(bundle, "session_id", "") or "")
        try:
            snapshot, _ = self._todo_store.read_snapshot(session_id)
        except Exception as exc:
            error = TodoRuntimeStateError(session_id, exc)
            bundle._todo_runtime_error = error
            if not getattr(bundle, "_todo_prompt_read_failure_logged", False):
                bundle._todo_prompt_read_failure_logged = True
                logger.warning(
                    "ohmo.todo.prompt.read_failure session_id=%s",
                    session_id,
                    exc_info=True,
                )
            section = (
                "# Trusted OHMO Todo Runtime State\n"
                f"todo_state_status: {_TODO_STATE_READ_ERROR_MARKER}\n"
                f"session_id: {session_id}\n"
                "active_todos_json: <UNAVAILABLE>\n"
                "todo_state_action: Do not produce a candidate answer; repair or atomically replace "
                "the todo snapshot, then retry."
            )
            return f"{prompt.rstrip()}\n\n{section}"
        bundle._todo_runtime_error = None
        encoded = json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        section = (
            "# Trusted OHMO Todo Runtime State\n"
            "The following state is injected by the runtime and is authoritative.\n"
            f"session_id: {session_id}\n"
            f"active_todos_json: {encoded}\n"
            "Lifecycle rules:\n"
            "- Keep the full snapshot synchronized through todo_write; never infer or fabricate completion.\n"
            "- Pending and in_progress items must remain real work until completed, removed as unnecessary, or blocked with a non-empty reason requiring user or external input.\n"
            "- Completed and blocked items are durable state and must be preserved unless the runtime archives a completed-only plan after a successful final answer."
        )
        return f"{prompt.rstrip()}\n\n{section}"

    @staticmethod
    def _todo_runtime_error_text(error: TodoRuntimeStateError) -> str:
        return (
            f"{error}. No model response was accepted or published. "
            "Repair or atomically replace the todo snapshot, then retry the turn."
        )

    def _read_todo_snapshot(self, session_id: str) -> list[dict[str, str]]:
        try:
            snapshot, _ = self._todo_store.read_snapshot(session_id)
        except Exception as exc:
            raise TodoRuntimeStateError(session_id, exc) from exc
        return snapshot

    @staticmethod
    def _discard_latest_assistant(bundle: RuntimeBundle) -> None:
        history = list(bundle.engine.messages)
        if (
            history
            and getattr(history[-1], "role", None) == "assistant"
            and not getattr(history[-1], "tool_uses", [])
        ):
            history.pop()
            if hasattr(bundle.engine, "load_messages"):
                bundle.engine.load_messages(history)
            else:
                bundle.engine.messages = history

    async def _guard_todo_final(
        self,
        *,
        bundle: RuntimeBundle,
        message: InboundMessage,
        session_key: str,
        user_prompt: str,
        turn_ctx: TurnContext,
        memory_scope: MemoryScope | None,
        reply_parts: list[str],
        emitted_media: set[str],
        recorder: GatewayEvalRecorder | None,
        state: dict[str, str | None],
        camera_meal_at: datetime | None = None,
    ):
        """Reconcile unresolved interactive work before accepting a model final."""
        try:
            snapshot = self._read_todo_snapshot(bundle.session_id)
        except TodoRuntimeStateError as exc:
            self._discard_latest_assistant(bundle)
            state["error"] = self._todo_runtime_error_text(exc)
            return

        unresolved = _todo_unresolved_items(snapshot)
        if not unresolved:
            return

        logger.info(
            "ohmo.todo.guard.trigger session_id=%s unresolved=%s",
            bundle.session_id,
            _todo_unresolved_text(snapshot),
        )
        self._discard_latest_assistant(bundle)
        attempts_used = 0
        reconciliation_error = False
        for attempt in range(1, _TODO_RECONCILIATION_MAX_ATTEMPTS + 1):
            attempts_used = attempt
            try:
                snapshot = self._read_todo_snapshot(bundle.session_id)
            except TodoRuntimeStateError as exc:
                state["error"] = self._todo_runtime_error_text(exc)
                return
            logger.info(
                "ohmo.todo.reconciliation.attempt session_id=%s attempt=%s max_attempts=%s unresolved=%s",
                bundle.session_id,
                attempt,
                _TODO_RECONCILIATION_MAX_ATTEMPTS,
                _todo_unresolved_text(snapshot),
            )
            instruction = (
                "Internal OHMO todo reconciliation. This is not a user message and must not be "
                "repeated as an external exchange. Continue the real work now. Current full "
                "todo snapshot (authoritative JSON): "
                f"{json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}\n"
                "Use tools as needed. Before stopping, atomically resubmit the full plan with "
                "todo_write: every unresolved item must be completed, removed because it is no "
                "longer required, or deliberately blocked with a non-empty reason only when "
                "user or external input is actually required. Never mark work completed merely "
                "to satisfy this check. Then provide the concise final answer."
            )
            system_prompt = await self._runtime_system_prompt(
                bundle,
                user_prompt,
                turn_ctx=turn_ctx,
                memory_scope=memory_scope,
                include_todo=True,
            )
            bundle.engine.set_system_prompt(
                self._with_camera_turn_context(system_prompt, message, camera_meal_at)
            )
            todo_error = getattr(bundle, "_todo_runtime_error", None)
            if isinstance(todo_error, TodoRuntimeStateError):
                state["error"] = self._todo_runtime_error_text(todo_error)
                return

            attempt_base = list(bundle.engine.messages)
            reply_parts.clear()
            attempt_error = False
            try:
                async for event in bundle.engine.submit_internal_message(instruction):
                    async for update in self._convert_stream_event(
                        event=event,
                        bundle=bundle,
                        message=message,
                        session_key=session_key,
                        content=user_prompt,
                        reply_parts=reply_parts,
                        recorder=recorder,
                    ):
                        if update.kind == "error":
                            attempt_error = True
                            continue
                        _remember_update_media(emitted_media, update)
                        yield update
            except MaxTurnsExceeded:
                attempt_error = True
            except Exception:
                attempt_error = True
                logger.warning(
                    "ohmo.todo.reconciliation.error session_id=%s attempt=%s",
                    bundle.session_id,
                    attempt,
                    exc_info=True,
                )

            try:
                snapshot = self._read_todo_snapshot(bundle.session_id)
            except TodoRuntimeStateError as exc:
                if hasattr(bundle.engine, "load_messages"):
                    bundle.engine.load_messages(attempt_base)
                else:
                    bundle.engine.messages = attempt_base
                state["error"] = self._todo_runtime_error_text(exc)
                return
            if attempt_error:
                reconciliation_error = True
                if hasattr(bundle.engine, "load_messages"):
                    bundle.engine.load_messages(attempt_base)
                else:
                    bundle.engine.messages = attempt_base
                break
            if not _todo_unresolved_items(snapshot):
                reconciled_reply = "".join(reply_parts).strip()
                if reconciled_reply:
                    state["reply"] = reconciled_reply
                    logger.info(
                        "ohmo.todo.reconciliation.success session_id=%s attempt=%s",
                        bundle.session_id,
                        attempt,
                    )
                    return
                reconciliation_error = True
                if hasattr(bundle.engine, "load_messages"):
                    bundle.engine.load_messages(attempt_base)
                else:
                    bundle.engine.messages = attempt_base
                break
            reconciliation_error = reconciliation_error or attempt_error
            # submit_internal_message accepted this turn, so preserve the
            # completed tool-use/result trace for the next attempt. Restore
            # attempt_base only on actual failure above; otherwise discard
            # just the unaccepted candidate final.
            self._discard_latest_assistant(bundle)

        try:
            snapshot = self._read_todo_snapshot(bundle.session_id)
        except TodoRuntimeStateError as exc:
            state["error"] = self._todo_runtime_error_text(exc)
            return
        logger.warning(
            "ohmo.todo.reconciliation.exhausted session_id=%s attempts=%s error=%s unresolved=%s",
            bundle.session_id,
            attempts_used,
            reconciliation_error,
            _todo_unresolved_text(snapshot),
        )
        if not _todo_unresolved_items(snapshot):
            state["error"] = (
                "Todo reconciliation resolved the plan, but the accepted final response is missing. "
                "Continue the task in a new turn."
            )
        else:
            state["error"] = (
                "Todo reconciliation could not finish safely; unresolved work: "
                f"{_todo_unresolved_text(snapshot)}. Continue the task in a new turn."
            )

    def _finalize_todo_after_successful_answer(self, *, session_id: str) -> bool:
        """Archive a completed-only plan after its final answer was accepted."""
        snapshot, _ = self._todo_store.read_snapshot(session_id)
        if not snapshot or any(item.get("status") != "completed" for item in snapshot):
            return False
        archived = self._todo_store.active_path(session_id)
        fresh = self._todo_store.new_list(session_id)
        logger.info(
            "ohmo.todo.cleanup session_id=%s archived=%s active=%s",
            session_id,
            archived,
            fresh,
        )
        return True

    def _todo_cleanup_update(
        self, *, bundle: RuntimeBundle, session_key: str
    ) -> GatewayStreamUpdate | None:
        """Return the post-final empty todo snapshot when cleanup succeeded."""
        if not self._finalize_todo_after_successful_answer(session_id=bundle.session_id):
            return None
        return GatewayStreamUpdate(
            kind="progress",
            text="",
            metadata={
                "_progress": True,
                "_session_key": session_key,
                "progress_event": {
                    "kind": "todo",
                    "todos": [],
                    "changed": True,
                    "session_id": str(bundle.session_id),
                },
            },
        )

    async def _runtime_system_prompt(
        self,
        bundle: RuntimeBundle,
        latest_user_prompt: str | None,
        *,
        turn_ctx: TurnContext | None = None,
        memory_scope: MemoryScope | None | object = _UNRESOLVED_MEMORY_SCOPE,
        include_todo: bool = True,
    ) -> str:
        bundle_cwd = str(Path(getattr(bundle, "cwd", self._cwd)).resolve())
        scope = self._coerce_memory_scope(turn_ctx, memory_scope)
        engaged = scope is not None
        memory_free_base = build_ohmo_system_prompt(
            bundle_cwd,
            workspace=self._workspace,
            extra_prompt=None,
            include_ohmo_memory=False,
            include_ohmo_workspace=engaged,
        )
        profile_context = render_profile_context(self._gateway_config, turn_ctx)
        if profile_context:
            memory_free_base = f"{memory_free_base}\n\n{profile_context}"
        session_owner_principal = (
            self._session_owner_principals.get(turn_ctx.session_id)
            if turn_ctx is not None
            else None
        )
        backend = (
            self._memory_backend_for_scope(scope)
            if scope is not None
            else self._prompt_memory_backend
        )
        derived_backend = self._shadow_backend_for_scope(scope) if scope is not None else None
        snapshot = await prepare_turn(
            backend,
            turn_ctx=turn_ctx,
            principal_isolated=principal_isolated_session(
                turn_ctx,
                session_owner_principal,
            ),
            visible_recall=self._gateway_config.visible_recall,
            latest_user_prompt=latest_user_prompt,
            owner_principals=self._gateway_config.owner_principals,
            memory_engaged_override=engaged,
            derived_backend=derived_backend,
            derived_recall_allowed_override=(
                self._honcho_turn_allowed(turn_ctx, scope) if scope is not None else False
            ),
        )
        gate_decision = getattr(snapshot, "gate_decision", None)
        if gate_decision is not None:
            logger.debug(
                "ohmo memory gate allowed=%s reasons=%s session_id=%s",
                gate_decision.allowed,
                gate_decision.reasons,
                turn_ctx.session_id if turn_ctx is not None else "",
            )
        if not hasattr(bundle, "current_settings"):
            prompt = compose_runtime_prompt(
                memory_free_base,
                snapshot,
                memory_engaged=engaged,
            )
            return self._append_todo_runtime_section(bundle, prompt) if include_todo else prompt
        settings = bundle.current_settings()
        if not hasattr(settings, "system_prompt"):
            prompt = compose_runtime_prompt(
                memory_free_base,
                snapshot,
                memory_engaged=engaged,
            )
            return self._append_todo_runtime_section(bundle, prompt) if include_todo else prompt
        base = settings.system_prompt or memory_free_base
        if profile_context and profile_context not in base:
            base = f"{base}\n\n{profile_context}"
        if include_todo:
            base = self._append_todo_runtime_section(bundle, base)
        composed_settings = settings.model_copy(
            update={
                "system_prompt": compose_runtime_prompt(
                    base,
                    snapshot,
                    memory_engaged=engaged,
                )
            }
        )
        return build_runtime_system_prompt(
            composed_settings,
            cwd=bundle_cwd,
            latest_user_prompt=latest_user_prompt,
            extra_skill_dirs=getattr(bundle, "extra_skill_dirs", ()),
            extra_plugin_roots=getattr(bundle, "extra_plugin_roots", ()),
            include_project_memory=False,
        )

    def _cwd_for_message(self, message: InboundMessage, session_key: str) -> str:
        # A /group-bound chat runs in its deliberately-bound project/repo cwd.
        # Every other (unbound) chat gets its OWN per-chat scratch dir as cwd, so
        # transient output (diagrams, downloads, scratch) is isolated per chat and
        # reaped on /new — instead of all chats sharing the workspace root.
        record = load_managed_group_record(
            workspace=self._workspace,
            channel=message.channel,
            chat_id=message.chat_id,
        )
        cwd = record.get("cwd") if record else None
        if not cwd:
            return str(get_session_work_dir(session_key, self._workspace))
        normalized = normalize_cwd(str(cwd))
        if not Path(normalized).is_dir():
            logger.warning(
                "ohmo managed group cwd does not exist channel=%s chat_id=%s cwd=%s",
                message.channel,
                message.chat_id,
                normalized,
            )
            return str(get_session_work_dir(session_key, self._workspace))
        return normalized

    def session_cwd(self, message: InboundMessage, session_key: str) -> str:
        """The cwd a session runs in — its per-chat work dir, or a /group-bound
        chat's bound project cwd. Public wrapper so the bridge can resolve a
        relative ``[[attach: …]]`` path against the same dir the agent wrote into."""
        return self._cwd_for_message(message, session_key)

    def _memory_gate_decision(self, turn_ctx: TurnContext | None) -> GateDecision:
        session_owner_principal = (
            self._session_owner_principals.get(turn_ctx.session_id)
            if turn_ctx is not None
            else None
        )
        return evaluate_memory_gate(
            turn_ctx,
            principal_isolated=principal_isolated_session(
                turn_ctx,
                session_owner_principal,
            ),
        )

    def _honcho_turn_allowed(
        self,
        turn_ctx: TurnContext | None,
        scope: MemoryScope,
    ) -> bool:
        """Apply the legacy owner gate or require an exact resolved family scope."""
        if (
            turn_ctx is not None
            and turn_ctx.camera_authorized
            and turn_ctx.principal == "__camera__"
            and scope
            == MemoryScope(
                private_tenant=self._gateway_config.camera_ingress.tenant_id,
                shared_tenants=(),
            )
        ):
            return True
        if scope.private_tenant == "owner":
            return self._memory_gate_decision(turn_ctx).allowed
        return self._resolve_turn_memory_scope(turn_ctx) == scope

    def _resolve_turn_memory_scope(
        self,
        turn_ctx: TurnContext | None,
    ) -> MemoryScope | None:
        """Resolve one turn's catalog audience, preserving single-user legacy."""
        if not self._gateway_config.owner_principals and not self._gateway_config.family_principals:
            return MemoryScope(private_tenant="owner", shared_tenants=())
        if turn_ctx is None:
            return None
        session_owner_principal = self._session_owner_principals.get(turn_ctx.session_id)
        scope = resolve_memory_scope(
            self._gateway_config,
            turn_ctx,
            principal_isolated=principal_isolated_session(
                turn_ctx,
                session_owner_principal,
            ),
        )
        if scope is not None and scope.private_tenant == "owner" and turn_ctx.is_owner is not True:
            return None
        return scope

    def _coerce_memory_scope(
        self,
        turn_ctx: TurnContext | None,
        memory_scope: MemoryScope | None | object,
    ) -> MemoryScope | None:
        if memory_scope is _UNRESOLVED_MEMORY_SCOPE:
            return self._resolve_turn_memory_scope(turn_ctx)
        return memory_scope if isinstance(memory_scope, MemoryScope) else None

    def _catalog_backend_for_scope(self, scope: MemoryScope) -> CatalogMemoryBackend:
        """Bind the shared catalog/embedder to one private+shared audience."""
        shared_tenant_id = scope.shared_tenants[0] if scope.shared_tenants else None
        key = (scope.private_tenant, shared_tenant_id)
        cache = getattr(self, "_catalog_memory_backends", None)
        if cache is None:
            cache = {}
            self._catalog_memory_backends = cache
        cached = cache.get(key)
        if cached is not None:
            return cached

        source: MemoryBackend = self._prompt_memory_backend
        if isinstance(source, ShadowMemoryBackend):
            source = source._base
        if isinstance(source, CatalogMemoryBackend):
            catalog = source._catalog
            embedder = source._embedder
            model = source._embedding_model
            embedding_timeout = source._embedding_timeout
        else:
            catalog = ensure_catalog_migrated(self._workspace)
            embedder = None
            model = "BAAI/bge-m3"
            embedding_timeout = 2.0

        backend = CatalogMemoryBackend(
            catalog,
            self._workspace,
            tenant_id=scope.private_tenant,
            shared_tenant_id=shared_tenant_id,
            embedder=embedder,
            owns_embedder=False,
            model=model,
            embedding_timeout=embedding_timeout,
        )
        cache[key] = backend
        return backend

    def _memory_backend_for_scope(self, scope: MemoryScope) -> MemoryBackend:
        # Empty identity registries are the pre-authz single-user deployment:
        # preserve its exact backend and file/catalog behavior.
        if not self._gateway_config.owner_principals and not self._gateway_config.family_principals:
            return self._prompt_memory_backend
        return self._catalog_backend_for_scope(scope)

    def _shadow_backend_for_scope(
        self,
        scope: MemoryScope,
    ) -> ShadowMemoryBackend | None:
        """Return one cached Honcho shadow bound only to the private tenant."""
        source = self._prompt_memory_backend
        if scope.private_tenant == "owner" and isinstance(source, ShadowMemoryBackend):
            return source
        if self._gateway_config.memory_backend != "shadow":
            return None

        cache = getattr(self, "_tenant_shadow_backends", None)
        if cache is None:
            cache = {}
            self._tenant_shadow_backends = cache
        cached = cache.get(scope.private_tenant)
        if cached is not None:
            return cached

        backend = make_tenant_shadow_backend(
            self._gateway_config,
            self._catalog_backend_for_scope(scope),
            self._workspace,
            tenant_id=scope.private_tenant,
        )
        if backend._honcho_client is None:
            return None
        cache[scope.private_tenant] = backend
        return backend

    def _memory_engaged(self, turn_ctx: TurnContext | None) -> bool:
        return self._resolve_turn_memory_scope(turn_ctx) is not None

    def _autodream_context(self) -> dict[str, object]:
        return {
            "memory_dir": str(get_memory_dir(self._workspace)),
            "session_dir": str(get_sessions_dir(self._workspace)),
            "app_label": "ohmo personal memory",
            "runner_module": "ohmo",
        }

    def _configure_turn_memory_surfaces(
        self,
        bundle: RuntimeBundle,
        turn_ctx: TurnContext | None,
        *,
        memory_scope: MemoryScope | None | object = _UNRESOLVED_MEMORY_SCOPE,
    ) -> bool:
        scope = self._coerce_memory_scope(turn_ctx, memory_scope)
        engaged = scope is not None
        backend = self._memory_backend_for_scope(scope) if scope is not None else None
        self._register_memory_tool(
            bundle,
            memory_engaged=engaged,
            backend=backend,
        )
        autodream_context = self._autodream_context() if engaged else None
        bundle.autodream_context = autodream_context
        metadata = getattr(getattr(bundle, "engine", None), "tool_metadata", None)
        if isinstance(metadata, dict):
            if autodream_context is None:
                metadata.pop("autodream_context", None)
            else:
                metadata["autodream_context"] = autodream_context
        self._bind_wellness_turn(bundle, turn_ctx)
        return engaged

    def _apply_reminder_wellness_turn(
        self,
        bundle: RuntimeBundle,
        reminder: _TrustedReminderWellnessAdmission,
    ) -> TrustedWellnessActor | None:
        """Mint invocation authority after current scheduler and tenant checks."""
        if type(reminder) is not _TrustedReminderWellnessAdmission:
            return None
        tenant = self._validated_reminder_wellness_tenant(reminder)
        if tenant is None:
            return None
        principal = canonical_principal("telegram", reminder.wellness_principal)
        if principal != reminder.private_chat_id:
            return None
        registry = getattr(bundle, "tool_registry", None)
        tool = registry.get(_WELLNESS_TOOL_NAME) if registry is not None else None
        if not isinstance(tool, WellnessLoginInjectingAdapter):
            return None
        return TrustedWellnessActor(principal)

    def _wellness_actor_for_submission(
        self,
        bundle: RuntimeBundle,
        turn_ctx: TurnContext | None,
        reminder: _TrustedReminderWellnessAdmission | None,
    ) -> TrustedWellnessActor | None:
        """Choose one invocation actor from live or freshly checked admission."""
        if reminder is not None:
            return self._apply_reminder_wellness_turn(bundle, reminder)
        registry = getattr(bundle, "tool_registry", None)
        tool = registry.get(_WELLNESS_TOOL_NAME) if registry is not None else None
        if not isinstance(tool, WellnessLoginInjectingAdapter):
            return None
        return _wellness_actor_for_turn(turn_ctx)

    def _validated_reminder_wellness_tenant(
        self, reminder: _TrustedReminderWellnessAdmission
    ) -> str | None:
        if type(reminder) is not _TrustedReminderWellnessAdmission:
            return None
        canonical = canonical_principal("telegram", reminder.wellness_principal)
        if canonical != reminder.private_chat_id:
            return None
        resolved = _reminder_wellness_tenants(self._gateway_config).resolve(canonical)
        if resolved is None or resolved != reminder.wellness_tenant:
            logger.warning(
                "ohmo reminder wellness rejected tenant=%r principal=%s resolved=%r reminder_id=%s",
                reminder.wellness_tenant,
                canonical,
                resolved,
                reminder.reminder_id,
            )
            return None
        return reminder.wellness_tenant

    def _bind_wellness_turn(
        self,
        bundle: RuntimeBundle,
        turn_ctx: TurnContext | None,
    ) -> None:
        """Bind wellness to the authenticated Telegram principal for a turn."""
        if turn_ctx is None:
            self._bind_wellness_principal(bundle, None)
            return
        principal = canonical_principal(turn_ctx.channel, turn_ctx.principal)
        owners = {
            canonical_principal("telegram", owner)
            for owner in self._gateway_config.owner_principals
            if str(owner).strip()
        }
        family_tenant = self._gateway_config.family_principals.get(principal)
        legacy_owner_mode = (
            not self._gateway_config.family_principals
            and not self._gateway_config.enabled_memory_tenants
        )
        family_enabled = family_tenant is not None and (
            legacy_owner_mode or family_tenant in self._gateway_config.enabled_memory_tenants
        )
        owner_turn = (
            turn_ctx.is_owner is True
            and turn_ctx.channel.strip().lower() == "telegram"
            and principal in owners
        )
        family_turn = (
            turn_ctx.channel.strip().lower() == "telegram"
            and principal.isdigit()
            and not owner_turn
            and family_enabled
        )
        if not owner_turn and not family_turn:
            self._bind_wellness_principal(bundle, None)
            return
        self._bind_wellness_principal(
            bundle,
            principal,
            owner_turn=owner_turn,
            family_turn=family_turn,
        )

    def _bind_wellness_principal(
        self,
        bundle: RuntimeBundle,
        principal: str | None,
        *,
        owner_turn: bool = False,
        family_turn: bool = False,
    ) -> None:
        """Bind only a trusted principal; no tenant is sent to Telegent."""
        if principal is not None and not owner_turn and not family_turn:
            owners = {
                canonical_principal("telegram", owner)
                for owner in self._gateway_config.owner_principals
                if str(owner).strip()
            }
            owner_turn = principal in owners
            family_tenant = self._gateway_config.family_principals.get(principal)
            family_turn = (
                principal.isdigit()
                and not owner_turn
                and family_tenant is not None
                and (
                    not self._gateway_config.enabled_memory_tenants
                    or family_tenant in self._gateway_config.enabled_memory_tenants
                )
            )
        registry = getattr(bundle, "tool_registry", None)
        if registry is None:
            return
        tool = registry.get(_WELLNESS_TOOL_NAME)
        if isinstance(tool, McpToolAdapter):
            tool = WellnessLoginInjectingAdapter(tool)
            registry.register(tool)
        if not isinstance(tool, WellnessLoginInjectingAdapter):
            return
        trusted_login = self._trusted_contact_login(principal)
        tool.set_trusted_principal(
            principal,
            trusted_login=trusted_login,
            channel="telegram" if principal is not None else "",
            owner_turn=owner_turn,
            family_turn=family_turn,
        )

    def _trusted_contact_login(self, principal: str | None) -> str | None:
        """Resolve a login only from the exact authenticated Telegram contact."""
        if self._contact_store is None or principal is None:
            return None
        canonical = canonical_principal("telegram", principal)
        if not canonical.isdigit():
            return None
        contact = self._contact_store.get("telegram", canonical)
        if contact is None or not contact.username:
            return None
        return contact.username

    def _register_gateway_tools(
        self,
        bundle: RuntimeBundle,
        *,
        memory_engaged: bool = True,
    ) -> None:
        self._unregister_group_tool(bundle)
        self._register_todo_tool(bundle)
        self._register_memory_tool(bundle, memory_engaged=memory_engaged)
        self._register_reminder_tools(bundle)
        self._register_conversation_image_tool(bundle)

    def _configure_attachment_boundary(self, bundle: RuntimeBundle) -> None:
        setter = getattr(getattr(bundle, "engine", None), "set_durable_message_transform", None)
        if callable(setter):
            setter(self._attachment_store.externalize_messages)

    def _register_conversation_image_tool(
        self,
        bundle: RuntimeBundle,
        *,
        current_message: ConversationMessage | str | None = None,
        on_attachment_load_started=None,
        on_attachment_loaded=None,
    ) -> None:
        registry = getattr(bundle, "tool_registry", None)
        tools = getattr(registry, "_tools", None)
        if registry is None or not isinstance(tools, dict):
            return
        active_ids = frozenset(
            block.attachment_id
            for block in getattr(current_message, "content", ())
            if isinstance(block, AttachmentRefBlock)
        )
        setattr(bundle, "_ohmo_active_attachment_ids", active_ids)
        if not active_ids and not self._conversation_has_attachment_refs(bundle):
            tools.pop(LoadConversationImageTool.name, None)
            return
        registry.register(
            LoadConversationImageTool(
                self._attachment_store,
                is_attachment_allowed=lambda attachment_id: self._conversation_attachment_allowed(
                    bundle, attachment_id
                ),
                on_load_started=on_attachment_load_started,
                on_loaded=on_attachment_loaded,
            )
        )

    @staticmethod
    def _conversation_has_attachment_refs(bundle: RuntimeBundle) -> bool:
        return any(
            isinstance(block, AttachmentRefBlock)
            for message in getattr(getattr(bundle, "engine", None), "messages", [])
            for block in getattr(message, "content", ())
        )

    @staticmethod
    def _conversation_attachment_allowed(
        bundle: RuntimeBundle,
        attachment_id: str,
    ) -> bool:
        if attachment_id in getattr(bundle, "_ohmo_active_attachment_ids", ()):
            return True
        return any(
            isinstance(block, AttachmentRefBlock) and block.attachment_id == attachment_id
            for message in getattr(getattr(bundle, "engine", None), "messages", [])
            for block in getattr(message, "content", ())
        )

    def _register_memory_tool(
        self,
        bundle: RuntimeBundle,
        *,
        memory_engaged: bool = True,
        backend: MemoryBackend | None = None,
    ) -> None:
        """Register the model-callable ``memory`` tool — disciplined curation
        (unicode-safe slugs, dedup, per-entry + store char bounds with
        consolidate-on-overflow) through the same configured backend that
        supplies prompt recall, so writes are visible on the next turn."""
        registry = getattr(bundle, "tool_registry", None)
        if registry is None:
            return
        if not memory_engaged:
            tools = getattr(registry, "_tools", None)
            if isinstance(tools, dict):
                tools.pop(OhmoMemoryTool.name, None)
            return
        registry.register(OhmoMemoryTool(backend or self._prompt_memory_backend))

    def _maybe_schedule_memory_judge(
        self,
        bundle: RuntimeBundle,
        session_key: str,
        *,
        turn_ctx: TurnContext | None = None,
        memory_scope: MemoryScope | None | object = _UNRESOLVED_MEMORY_SCOPE,
    ) -> None:
        """Schedule the background memory judge off the hot path, on a per-session
        turn cadence. Opt-in via OHMO_MEMORY_JUDGE; never blocks the reply (the
        snapshot of inputs is taken now, the LLM call runs in a tracked task)."""
        scope = self._coerce_memory_scope(turn_ctx, memory_scope)
        if scope is None:
            return
        if not judge_enabled():
            return
        backend = self._memory_backend_for_scope(scope)
        if (
            not self._gateway_config.owner_principals
            and not self._gateway_config.family_principals
            and not isinstance(backend, FileMemoryBackend)
        ):
            return
        count = self._judge_turn_counts.get(session_key, 0) + 1
        self._judge_turn_counts[session_key] = count
        if count % judge_interval() != 0:
            return
        # Skip if a judge for this session is still running — never overlap two
        # judges on the shared store, and never orphan a tracked task.
        inflight = self._judge_tasks.get(session_key)
        if inflight is not None and not inflight.done():
            return
        try:
            # Snapshot inputs NOW — the live bundle is reused by the next turn.
            messages = list(bundle.engine.messages)
            api_client = bundle.engine.api_client
            settings = bundle.current_settings()
            model = settings.model
            timeout = float(getattr(settings, "timeout", None) or 30.0)
        except Exception:
            logger.warning(
                "ohmo memory judge schedule failed session_key=%s", session_key, exc_info=True
            )
            return
        task = asyncio.create_task(
            self._run_memory_judge_task(
                session_key,
                api_client,
                model,
                messages,
                timeout,
                backend=backend,
            ),
            name=f"ohmo-memory-judge:{session_key}",
        )
        self._judge_tasks[session_key] = task

        def _pop(finished: asyncio.Task, key: str = session_key, this: asyncio.Task = task) -> None:
            # Identity-checked: a finishing task must not evict a newer one.
            if self._judge_tasks.get(key) is this:
                self._judge_tasks.pop(key, None)

        task.add_done_callback(_pop)

    async def _run_memory_judge_task(
        self,
        session_key,
        api_client,
        model,
        messages,
        timeout,
        *,
        backend: MemoryBackend | None = None,
    ) -> None:
        try:
            if isinstance(backend, CatalogMemoryBackend):
                store = backend.judge_store()
            else:
                store = self._memory_store
            outcome = await run_memory_judge(
                api_client=api_client,
                model=model,
                messages=messages,
                store=store,
                timeout=timeout,
            )
            # Always log a fired run — even a no-op ("nothing to save") — so the
            # judge's liveness is observable in journald (otherwise a quiet judge
            # is indistinguishable from one that never fired).
            logger.info(
                "ohmo memory judge ran session_key=%s applied=%s skipped=%s reason=%r",
                session_key,
                outcome.applied,
                outcome.skipped,
                outcome.reason,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("ohmo memory judge crashed session_key=%s", session_key)

    def _register_todo_tool(self, bundle: RuntimeBundle) -> None:
        """Override the default ``todo_write`` with a per-session one — the list
        lives in ``TodoStore`` keyed by this bundle's live ``session_id`` (read
        lazily so it tracks ``/new``), so chats never share a TODO file."""
        registry = getattr(bundle, "tool_registry", None)
        if registry is None:
            return
        registry.register(OhmoTodoWriteTool(self._todo_store, lambda: bundle.session_id))

    def _register_reminder_tools(self, bundle: RuntimeBundle) -> None:
        """Register the per-session reminder tools (create/list/cancel). They
        share one ReminderStore + asyncio.Lock with the scheduler; delivery
        context is read from engine.tool_metadata['ohmo_reminder_ctx']."""
        registry = getattr(bundle, "tool_registry", None)
        if registry is None:
            return
        registry.register(
            RemindCreateTool(
                self._reminder_store,
                self._reminder_lock,
                default_tz=self._default_tz,
                max_per_chat=self._reminder_max_per_chat,
                wellness_tenants=_reminder_wellness_tenants(self._gateway_config),
            )
        )
        registry.register(
            RemindListTool(self._reminder_store, self._reminder_lock, default_tz=self._default_tz)
        )
        registry.register(RemindCancelTool(self._reminder_store, self._reminder_lock))

    def _register_group_tool(self, bundle: RuntimeBundle) -> None:
        if self._create_feishu_group is None or not hasattr(bundle, "tool_registry"):
            return
        if bundle.tool_registry is None or bundle.tool_registry.get(_GROUP_TOOL_NAME) is not None:
            return
        bundle.tool_registry.register(
            OhmoCreateFeishuGroupTool(
                workspace=self._workspace,
                create_group=self._create_feishu_group,
                publish_group_welcome=self._publish_group_welcome,
            )
        )

    @staticmethod
    def _unregister_group_tool(bundle: RuntimeBundle) -> None:
        registry = getattr(bundle, "tool_registry", None)
        tools = getattr(registry, "_tools", None)
        if isinstance(tools, dict):
            tools.pop(_GROUP_TOOL_NAME, None)

    def _set_group_request_context(
        self,
        bundle: RuntimeBundle,
        message: InboundMessage,
        session_key: str,
    ) -> object:
        metadata = getattr(bundle.engine, "tool_metadata", {})
        previous = metadata.get("ohmo_group_request", _NO_GROUP_REQUEST)
        if not message.metadata.get("_ohmo_group_command"):
            metadata.pop("ohmo_group_request", None)
            metadata.pop("_suppress_next_user_goal", None)
            self._unregister_group_tool(bundle)
            return _NO_GROUP_REQUEST
        self._register_group_tool(bundle)
        metadata["_suppress_next_user_goal"] = True
        metadata["ohmo_group_request"] = {
            "channel": message.channel,
            "chat_type": str(message.metadata.get("chat_type") or "").strip().lower(),
            "sender_id": str(message.sender_id),
            "source_chat_id": str(message.chat_id),
            "source_session_key": session_key,
            "sender_display_name": message.metadata.get("sender_display_name"),
            "raw_request": message.metadata.get("_ohmo_group_raw_request") or "",
            "used": False,
        }
        return previous

    @staticmethod
    def _restore_group_request_context(bundle: RuntimeBundle, previous: object) -> None:
        metadata = getattr(bundle.engine, "tool_metadata", {})
        del previous
        metadata.pop("ohmo_group_request", None)
        metadata.pop("_suppress_next_user_goal", None)
        OhmoSessionRuntimePool._unregister_group_tool(bundle)

    @staticmethod
    def _clear_reminder_context(bundle: RuntimeBundle) -> None:
        """Drop the per-message reminder context after a turn."""
        metadata = getattr(bundle.engine, "tool_metadata", None)
        if isinstance(metadata, dict):
            metadata.pop("ohmo_reminder_ctx", None)


_GROUP_CHAT_TYPES = frozenset({"group", "supergroup", "chat", "channel", "room"})


def _is_group_message(message: InboundMessage) -> bool:
    """True when an inbound message originates from a shared/group chat.

    Telegram emits only ``is_group`` (bool); Feishu/others set ``chat_type``.
    Accept either so a group-only ACL (e.g. creator-only reminder cancel) is
    enforced on every channel, not just the ones that happen to set chat_type.
    """
    metadata = message.metadata or {}
    if bool(metadata.get("is_group")):
        return True
    return str(metadata.get("chat_type") or "").strip().lower() in _GROUP_CHAT_TYPES


def _content_snippet(text: str, *, limit: int = 160) -> str:
    """Return a compact single-line preview for logs."""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _sanitize_snapshot_messages(raw_messages: object) -> list[dict[str, object]] | None:
    """Validate and sanitize restored messages from persisted ohmo snapshots."""
    if not raw_messages or not isinstance(raw_messages, list):
        return None
    messages: list[ConversationMessage] = []
    for raw in raw_messages:
        try:
            messages.append(ConversationMessage.model_validate(raw))
        except ValueError:
            logger.warning(
                "ohmo runtime skipped invalid restored message while sanitizing snapshot"
            )
    return [
        _snapshot_message_dict(message) for message in _sanitize_group_command_prompts(messages)
    ]


def _extract_tool_media(event: ToolExecutionCompleted) -> list[str]:
    """Return local media paths produced by a tool completion event."""
    if event.is_error or not isinstance(event.metadata, dict):
        return []
    raw_paths = event.metadata.get("paths") or event.metadata.get("media")
    if isinstance(raw_paths, str):
        candidates = [raw_paths]
    elif isinstance(raw_paths, list):
        candidates = [str(item) for item in raw_paths if isinstance(item, str) and item.strip()]
    else:
        candidates = []
    media: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        if not path.is_file():
            continue
        resolved = str(path)
        if resolved not in seen:
            seen.add(resolved)
            media.append(resolved)
    return media


def _remember_update_media(seen: set[str], update: GatewayStreamUpdate) -> None:
    """Track media already emitted during this gateway turn."""
    raw_media = update.media or (update.metadata or {}).get("_media") or []
    if isinstance(raw_media, str):
        candidates = [raw_media]
    elif isinstance(raw_media, list):
        candidates = [str(item) for item in raw_media if isinstance(item, str) and item.strip()]
    else:
        candidates = []
    for raw in candidates:
        try:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = path.resolve()
            seen.add(str(path))
        except (OSError, RuntimeError, ValueError):
            logger.debug("ohmo runtime skipped invalid emitted media path", exc_info=True)
            continue


def _extract_final_reply_media(reply: str, emitted_media: set[str]) -> list[str]:
    """Return local image paths mentioned in final text that were not already emitted."""
    media: list[str] = []
    seen = set(emitted_media)
    for match in _FINAL_REPLY_IMAGE_PATH_RE.finditer(reply or ""):
        raw = match.group("path").strip(" \t\r\n\"'.,;:，。；：、)]}")
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            continue
        if not path.is_file():
            continue
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        media.append(resolved)
    return media


def _format_tool_media_caption(event: ToolExecutionCompleted, media: list[str]) -> str:
    """Return a short caption for media generated by tools."""
    if event.tool_name == "image_generation":
        provider = ""
        if isinstance(event.metadata, dict):
            provider = str(event.metadata.get("provider") or "").strip()
        suffix = f" via {provider}" if provider else ""
        names = ", ".join(Path(path).name for path in media)
        return f"已生成图片{suffix}：{names}"
    names = ", ".join(Path(path).name for path in media)
    return f"已生成文件：{names}"


def _sanitize_group_command_prompts(
    messages: list[ConversationMessage],
) -> list[ConversationMessage]:
    """Replace internal /group tool-driving prompts with durable user-facing history."""
    return [_sanitize_group_command_prompt(message) for message in messages]


def _sanitize_group_command_prompt(message: ConversationMessage) -> ConversationMessage:
    changed = False
    content: list[TextBlock | ImageBlock | AttachmentRefBlock] = []
    for block in message.content:
        if isinstance(block, TextBlock) and _GROUP_AGENT_PROMPT_PREFIX in block.text:
            content.append(TextBlock(text=_format_group_command_history_note(block.text)))
            changed = True
        else:
            content.append(block)
    if not changed:
        return message
    return message.model_copy(update={"content": content})


def _format_group_command_history_note(prompt: str) -> str:
    raw_request = prompt
    if _GROUP_AGENT_PROMPT_REQUEST_MARKER in prompt:
        raw_request = prompt.split(_GROUP_AGENT_PROMPT_REQUEST_MARKER, 1)[1].strip()
    raw_request = raw_request.strip() or "(empty request)"
    return f"[Handled /group request]\nThe user asked ohmo to create a Feishu group:\n{raw_request}"


def _sanitize_group_command_metadata(raw_metadata: object) -> object:
    """Remove internal /group tool-driving text from compact carry-over metadata."""
    if not isinstance(raw_metadata, dict):
        return raw_metadata
    sanitized = dict(raw_metadata)
    for key in _GROUP_METADATA_KEYS:
        if key in sanitized:
            sanitized[key] = _sanitize_group_command_metadata_value(sanitized[key])
    return sanitized


def _sanitize_group_command_metadata_value(value: object) -> object:
    if isinstance(value, str):
        if _GROUP_AGENT_PROMPT_PREFIX in value:
            return _format_group_command_history_note(value)
        return value
    if isinstance(value, dict):
        return {key: _sanitize_group_command_metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_group_command_metadata_value(item) for item in value]
    return value


def _summarize_tool_input(tool_name: str, tool_input: dict[str, object]) -> str:
    if not tool_input:
        return ""
    for key in ("url", "query", "pattern", "path", "file_path", "command"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            return text if len(text) <= 120 else text[:120] + "..."
    try:
        raw = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)
    except TypeError:
        raw = str(tool_input)
    return raw if len(raw) <= 120 else raw[:120] + "..."


def _render_todo_checklist(path: str | Path | None) -> str | None:
    """Render a to-do list file as a compact chat checklist (a Claude-Code style
    todo panel) — ``📋 To-do`` then one ``⬜``/``✅`` line per item.

    Takes the path to the session's active list (resolved by ``TodoStore``).
    Returns ``None`` when there is no list (so nothing is shown). Used after a
    ``todo_write`` call instead of echoing the raw per-item JSON.
    """
    if not path:
        return None
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    rows: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- [x]"):
            rows.append("✅ " + stripped[5:].strip())
        elif stripped.startswith("- [ ]"):
            rows.append("⬜ " + stripped[5:].strip())
    if not rows:
        return None
    return "📋 To-do\n" + "\n".join(rows)


def _pretty_tool_name(tool_name: str) -> str:
    """Human-facing tool label: drop the noisy ``mcp__<server>__`` prefix and turn
    ``read_calendar_event`` into ``Read calendar event``."""
    name = tool_name
    if name.startswith("mcp__"):
        parts = name.split("__")
        if len(parts) >= 3:
            name = parts[-1]
    label = name.replace("_", " ").strip()
    if not label:
        return tool_name
    return label[0].upper() + label[1:]


_TOOL_PURPOSE_MAX_WORDS = 20


def _normalize_tool_purpose(text: str) -> str:
    """Model-authored pre-tool narration -> a bounded action purpose.

    The first meaningful line, whitespace-collapsed and truncated to <=20
    words. This is an action label ("Проверяю расписание поездов"), not
    chain-of-thought. Empty when there is no usable narration.
    """
    for raw in (text or "").splitlines():
        line = " ".join(raw.split())
        if not line:
            continue
        words = line.split()
        if len(words) > _TOOL_PURPOSE_MAX_WORDS:
            line = " ".join(words[:_TOOL_PURPOSE_MAX_WORDS]).rstrip("….,;:") + "…"
        return line
    return ""


def _tool_progress_event(
    *,
    tool_name: str,
    tool_call_id: str,
    display_label: str,
    phase: str,
    status: str,
    purpose: str = "",
) -> dict[str, object]:
    """Build the provider-neutral tool lifecycle payload for channels.

    Tool arguments and results intentionally stay in the existing detailed
    text.  This metadata is only the stable correlation and display contract
    needed by quiet channel renderers: the call id for correlation, a safe
    display label, and the validated model-authored action ``purpose``.
    """
    event: dict[str, object] = {
        "kind": "tool",
        "tool": tool_name,
        "tool_call_id": tool_call_id,
        "display_label": display_label,
        "phase": phase,
        "status": status,
    }
    if purpose:
        event["purpose"] = purpose
    return event


def _tool_completion_status(event: ToolExecutionCompleted) -> str:
    """Normalize optional runtime cancellation metadata before channel delivery."""
    metadata = event.metadata if isinstance(event.metadata, dict) else {}
    raw_status = metadata.get("status")
    normalized = raw_status.strip().lower() if isinstance(raw_status, str) else ""
    if normalized in {"cancelled", "canceled", "stopped", "aborted"}:
        return "stopped"
    if normalized in {"failed", "failure", "error", "errored"}:
        return "failed"
    return "failed" if event.is_error else "succeeded"


def _format_tool_args_block(tool_input: dict[str, object]) -> str:
    """Render tool args as a fenced code block (so the chat renders monospace and
    Telegram's markdown can't mangle ``__name__`` etc.). Empty string for no args.

    A lone string arg (a command, a url, a path) is shown as-is; anything richer
    is pretty-printed JSON.
    """
    if not tool_input:
        return ""
    if len(tool_input) == 1:
        ((key, value),) = tuple(tool_input.items())
        if isinstance(value, str) and value.strip():
            body = value.strip()
            if len(body) > 600:
                body = body[:600] + "\n…"
            lang = "bash" if key in ("command", "code") else ""
            return f"```{lang}\n{body}\n```"
    try:
        pretty = json.dumps(tool_input, ensure_ascii=False, indent=2, sort_keys=True)
    except TypeError:
        pretty = str(tool_input)
    if len(pretty) > 1200:
        pretty = pretty[:1200] + "\n…"
    return f"```json\n{pretty}\n```"


def _short_call_id(call_id: str) -> str:
    """A short, stable tag to pair a tool call's start hint (`🛠️ Bash — a1b2`)
    with its result hint (`Bash — a1b2 ✅`). Both events carry the same Anthropic
    tool_use id; we show its tail so the two messages can be matched even when
    they arrive as separate messages. Empty string for no id."""
    cid = (call_id or "").strip()
    if not cid:
        return ""
    tail = cid.rsplit("_", 1)[-1]  # drop the 'toolu_' prefix, keep the random part
    return (tail or cid)[-4:]


def _format_tool_result_block(output: str) -> str:
    """Render a tool's output as a truncated fenced block — the completion-side
    mirror of ``_format_tool_args_block`` (params). Empty string for no output."""
    body = (output or "").strip()
    if not body:
        return ""
    if len(body) > 600:
        body = body[:600] + "\n…"
    return f"```\n{body}\n```"


def _format_tool_done(tool_name: str, output: str, is_error: bool, call_id: str = "") -> str:
    """A tool-completion hint: the pretty name + a short call-id (to match the
    start hint) + ✅/❌ + a truncated output block. Used once a tool returns so the
    user sees the OUTCOME (which call it was + success + a clipped result)."""
    mark = "❌" if is_error else "✅"
    name = _pretty_tool_name(tool_name)
    cid = _short_call_id(call_id)
    head = f"{name} — {cid} {mark}" if cid else f"{name} {mark}"
    block = _format_tool_result_block(output)
    return f"{head}\n{block}" if block else head


def _format_channel_progress(
    *,
    channel: str,
    kind: str,
    text: str,
    session_key: str,
    content: str,
    compact_phase: str | None = None,
    compact_trigger: str | None = None,
    attempt: int | None = None,
) -> str:
    if channel not in {
        "feishu",
        "telegram",
        "slack",
        "discord",
        "matrix",
        "whatsapp",
        "email",
        "dingtalk",
        "qq",
        "wechat",
    }:
        return text
    prefers_chinese = _prefers_chinese_progress(content)
    if kind == "thinking":
        if channel == "telegram":
            # Quiet Telegram renders a stable per-turn Russian header plus an
            # explicit inference state instead of a canned thinking line
            # (bead agents-playgroud-fe2); verbose Telegram sends no separate
            # canned line either — the narration itself is shown as 🧠 text.
            return ""
        seed = f"{session_key}|{content}".encode()
        phrases = _CHANNEL_THINKING_PHRASES if prefers_chinese else _CHANNEL_THINKING_PHRASES_EN
        idx = int(hashlib.sha256(seed).hexdigest(), 16) % len(phrases)
        return phrases[idx]
    if kind == "reasoning":
        # The model's own interstitial narration before a tool call, shown live
        # with a thinking marker. Unlike "thinking" (a canned placeholder), the
        # text is the model's real words — pass it through verbatim (it is
        # already in the user's language) under a single 🧠 prefix.
        normalized = text.strip()
        return normalized if normalized.startswith("🧠") else f"🧠 {normalized}"
    if kind == "tool_hint":
        if prefers_chinese:
            if text.startswith("Using "):
                return "🛠️ " + text.replace("Using ", "正在使用 ", 1)
            return f"🛠️ {text}"
        return text if text.startswith("🛠️ ") else f"🛠️ {text}"
    if kind == "image_fallback":
        if prefers_chinese:
            return "🖼️ 当前模型不支持图片输入，我先改用附件路径和摘要继续。"
        return "🖼️ The active model does not support image input. I’ll retry with attachment paths and summaries."
    if kind == "status":
        normalized = text.strip()
        if normalized == "Auto-compacting conversation memory to keep things fast and focused.":
            if prefers_chinese:
                return "🧠 聊天有点长啦，我先帮你蹦蹦跳跳压缩一下记忆，马上带着重点回来～"
            return "🧠 This chat is getting long — I’m doing a quick memory squeeze and hopping right back with the good bits."
        if text.startswith(("🤔", "🧠", "✨", "🔎", "🪄", "🛠️", "🫧")):
            return text
        return f"🫧 {text}"
    if kind == "compact_progress":
        if compact_phase == "hooks_start":
            if prefers_chinese:
                if compact_trigger == "reactive":
                    return "🫧 上下文有点超长，我先准备压缩一下记忆，然后立刻继续重试～"
                return "🫧 我先把上下文和记忆准备一下，马上开始压缩重点～"
            if compact_trigger == "reactive":
                return "🫧 The context got too large. I’m preparing a quick memory compaction before retrying."
            return "🫧 Let me get the context ready before I compact the conversation."
        if compact_phase == "context_collapse_start":
            if prefers_chinese:
                return "🫧 我先把太长的上下文折叠一下，让后面的压缩更快一点～"
            return "🫧 I’m collapsing the oversized context first so compaction can move faster."
        if compact_phase == "context_collapse_end":
            if prefers_chinese:
                return "🫧 上下文已经先收紧了一层，继续压缩重点～"
            return "🫧 The context is trimmed down now. Continuing with the main compaction."
        if compact_phase in {"session_memory_start", "compact_start"}:
            if prefers_chinese:
                if compact_phase == "session_memory_start":
                    return "🧠 我先把前面的聊天重点悄悄捋顺一下，马上继续～"
                if compact_trigger == "reactive":
                    return "🧠 这轮上下文太长了，我先压缩一下记忆，然后马上继续重试～"
                return "🧠 聊天有点长啦，我先帮你悄悄压缩一下记忆，马上继续～"
            if compact_phase == "session_memory_start":
                return "🧠 Let me quickly condense the earlier parts of this chat, then I’ll keep going."
            if compact_trigger == "reactive":
                return (
                    "🧠 The context is too large for this turn. I’ll compact the memory and retry."
                )
            return "🧠 This chat is getting long. I’ll compact the memory and keep going."
        if compact_phase == "compact_retry":
            suffix = f" (attempt {attempt})" if attempt is not None else ""
            if prefers_chinese:
                return f"🔁 压缩记忆这一步有点卡，我换个方式再试一次{suffix}。"
            return f"🔁 Compaction got stuck, trying a lighter retry{suffix}."
        if compact_phase == "compact_failed":
            if prefers_chinese:
                return "⚠️ 这次记忆压缩没成功，我先跳过它继续处理你的消息。"
            return "⚠️ Memory compaction did not complete. I’m skipping it and continuing."
        return ""
    return text


def _build_inbound_user_message(
    message: InboundMessage,
    attachment_store: AttachmentStore | None = None,
    *,
    session_key: str | None = None,
) -> ConversationMessage:
    """Convert an inbound channel message into user content blocks."""
    content: list[TextBlock | ImageBlock | AttachmentRefBlock] = []
    speaker_context = _build_speaker_context(message)
    base = (message.content or "").strip()
    if speaker_context:
        content.append(TextBlock(text=speaker_context))
    if base:
        content.append(TextBlock(text=base))

    attachment_notes = _build_attachment_notes(message.media)
    if attachment_notes:
        prefix = "\n\n" if base else ""
        content.append(TextBlock(text=prefix + attachment_notes))

    source_timestamp = _trusted_utc_iso(_trusted_inbound_event_time(message))
    source_provenance: dict[str, object] = {
        "schema_version": 1,
        "channel": str(message.channel),
        "principal": (
            f"{str(message.channel).strip().lower()}:"
            f"{canonical_principal(message.channel, str(message.sender_id))}"
        ),
        "chat_id": str(message.chat_id),
        "session_key": session_key or message.session_key,
        "received_at": source_timestamp,
        "timestamp_authority": "inbound_event_timestamp" if source_timestamp else None,
        "is_group": not is_private_message(message),
        "is_forwarded": build_turn_context(message, session_id="").is_forwarded,
        "source_message_id": _normalize_source_message_ref(
            (message.metadata or {}).get("message_id")
        ),
        # The current user event can coalesce multiple inbound messages. Keep
        # its append identity next to the original attachment's own source ID.
        "append_source_message_id": _normalize_source_message_ref(
            (message.metadata or {}).get("message_id")
        ),
    }
    coalesced_sources = (
        message.metadata.get("_coalesced_media_sources")
        if message.metadata.get("_coalesced_media_provenance_authority")
        is COALESCED_ATTACHMENT_PROVENANCE_AUTHORITY
        else None
    )
    for media_index, media_path in enumerate(message.media):
        if not _is_image_attachment(media_path):
            continue
        try:
            image = ImageBlock.from_path(media_path)
            if attachment_store is not None:
                ref = attachment_store.ingest_image_block(image)
                attachment_provenance = dict(source_provenance)
                if isinstance(coalesced_sources, list) and media_index < len(coalesced_sources):
                    source = coalesced_sources[media_index]
                    if isinstance(source, Mapping):
                        source_id = _normalize_source_message_ref(source.get("source_message_id"))
                        received_at = source.get("received_at")
                        if source_id is not None and isinstance(received_at, str):
                            try:
                                attachment_time = datetime.fromisoformat(received_at)
                            except ValueError:
                                attachment_time = None
                            trusted_time = _trusted_utc_iso(attachment_time)
                            if trusted_time is not None:
                                attachment_provenance["source_message_id"] = source_id
                                attachment_provenance["received_at"] = trusted_time
                content.append(ref.model_copy(update={"source_provenance": attachment_provenance}))
            content.append(image)
        except Exception:
            logger.exception("ohmo runtime failed to encode image attachment path=%s", media_path)

    return ConversationMessage(
        role="user",
        content=content,
        event_id=_event_id_for_inbound_message(message),
    )


def _event_id_for_inbound_message(message: InboundMessage) -> str | None:
    """Derive idempotency only from gateway-validated channel identifiers."""
    metadata = message.metadata or {}
    if metadata.get("callback_query") is True:
        callback_query_id = metadata.get("callback_query_id")
        if (
            not isinstance(callback_query_id, str)
            or not callback_query_id.strip()
            or callback_query_id != callback_query_id.strip()
            or len(callback_query_id) > 512
            or any(ord(character) < 32 or ord(character) == 127 for character in callback_query_id)
        ):
            raise ValueError("callback_query_id is missing or malformed")
        seed = "\x00".join(
            (
                str(message.channel).strip().lower(),
                str(message.chat_id),
                canonical_principal(message.channel, str(message.sender_id)),
                "callback-query",
                callback_query_id,
            )
        ).encode("utf-8")
        return f"ohmo-event-{hashlib.sha256(seed).hexdigest()}"
    source_message_id = next(
        (
            normalized
            for key in ("message_id", "messageId", "message-id")
            if (normalized := _normalize_source_message_ref(metadata.get(key))) is not None
        ),
        None,
    )
    if source_message_id is None:
        return None
    seed = "\x00".join(
        (
            str(message.channel).strip().lower(),
            str(message.chat_id),
            canonical_principal(message.channel, str(message.sender_id)),
            source_message_id,
        )
    ).encode("utf-8")
    return f"ohmo-event-{hashlib.sha256(seed).hexdigest()}"


def _should_retry_without_image_input(
    error_message: str, messages: list[ConversationMessage]
) -> bool:
    """Return True when a provider rejects image input and history contains images."""
    if not _history_has_image_blocks(messages):
        return False
    normalized = error_message.lower()
    image_signal = any(
        phrase in normalized
        for phrase in (
            "image input",
            "image_url",
            "multimodal",
            "vision",
            "image content",
        )
    )
    rejection_signal = any(
        phrase in normalized
        for phrase in (
            "no endpoints found",
            "not support",
            "does not support",
            "unsupported",
            "cannot support",
            "can't support",
        )
    )
    return image_signal and rejection_signal


def _history_has_image_blocks(messages: list[ConversationMessage]) -> bool:
    return any(
        any(isinstance(block, ImageBlock) for block in message.content) for message in messages
    )


def _strip_image_blocks_from_engine_history(engine) -> None:
    messages = _strip_image_blocks_from_messages(list(engine.messages))
    if hasattr(engine, "load_messages"):
        engine.load_messages(messages)
    else:
        engine.messages = messages


def _strip_image_blocks_from_messages(
    messages: list[ConversationMessage],
) -> list[ConversationMessage]:
    return [_strip_image_blocks_from_message(message) for message in messages]


def _strip_image_blocks_from_message(message: ConversationMessage) -> ConversationMessage:
    if not any(isinstance(block, ImageBlock) for block in message.content):
        return message
    content = [block for block in message.content if not isinstance(block, ImageBlock)]
    if not any(isinstance(block, TextBlock) for block in content):
        content.append(TextBlock(text=_IMAGE_FALLBACK_NOTE))
    return message.model_copy(update={"content": content})


def _build_speaker_context(message: InboundMessage) -> str:
    """Tell the agent who sent the message — in BOTH group and direct chats — so
    it can recognise the owner vs. e.g. a family member on the allowlist.

    Previously only group messages carried a speaker header, so in a 1:1 chat the
    agent never saw the sender's Telegram handle and couldn't tell who it was
    talking to.
    """
    metadata = message.metadata or {}
    chat_type = str(metadata.get("chat_type") or "").strip().lower()
    username = str(metadata.get("username") or "").strip()
    first_name = str(metadata.get("first_name") or "").strip()
    label = (
        str(metadata.get("sender_display_name") or "").strip()
        or str(metadata.get("sender_label") or "").strip()
        or first_name
        or username
        or str(message.sender_id).strip()
        or "unknown"
    )
    handle = f" (@{username})" if username else ""
    if chat_type == "group":
        return (
            "[Channel speaker]\n"
            f"This message was sent in a group chat by: {label}{handle}\n"
            f"Sender id: {message.sender_id}"
        )
    return f"[Speaker]\nDirect message from: {label}{handle}\nSender id: {message.sender_id}"


def _build_attachment_notes(media_paths: list[str]) -> str:
    """Build textual attachment notes for non-image context and persistence."""
    if not media_paths:
        return ""
    lines = [
        "[Channel attachments]",
        "The following attachments were downloaded locally for this message.",
        "Inspect them by path if needed.",
    ]
    for media_path in media_paths:
        lines.append(f"- {_describe_media_path(media_path)}")
        summary = _summarize_attachment(media_path)
        if summary:
            for part in summary.splitlines():
                lines.append(f"  {part}")
    return "\n".join(lines).strip()


def _describe_media_path(media_path: str) -> str:
    """Return a short type + path description for an inbound attachment."""
    suffix = Path(media_path).suffix.lower()
    if _is_image_attachment(media_path):
        kind = "image"
    elif suffix in {".mp3", ".wav", ".m4a", ".opus", ".aac"}:
        kind = "audio"
    elif suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        kind = "video"
    else:
        kind = "file"
    filename = os.path.basename(media_path)
    return f"{kind}: {filename} (path: {media_path})"


def _is_image_attachment(media_path: str) -> bool:
    mime, _ = mimetypes.guess_type(media_path)
    return bool(mime and mime.startswith("image/"))


def _summarize_attachment(media_path: str) -> str:
    """Return a compact summary/header for a downloaded attachment."""
    path = Path(media_path)
    if not path.exists() or not path.is_file():
        return "summary: attachment is unavailable on disk"
    try:
        stat = path.stat()
    except OSError:
        return "summary: attachment metadata is unavailable"

    mime, _ = mimetypes.guess_type(str(path))
    summary_lines = [f"summary: size={stat.st_size} bytes mime={mime or 'unknown'}"]
    try:
        head = path.read_bytes()[:_TEXT_PREVIEW_BYTES]
    except OSError:
        return "\n".join(summary_lines)

    if _is_image_attachment(str(path)):
        return "\n".join(summary_lines)

    text_preview = _decode_text_preview(head)
    if text_preview is not None:
        summary_lines.append(f"text preview: {text_preview}")
        return "\n".join(summary_lines)

    head_hex = head[:_BINARY_HEAD_BYTES].hex(" ")
    if head_hex:
        summary_lines.append(f"binary header: {head_hex}")
    return "\n".join(summary_lines)


def _decode_text_preview(data: bytes) -> str | None:
    """Return a compact text preview when a file looks text-like."""
    if not data:
        return ""
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    printable = sum(
        1 for char in decoded if char in string.printable or char.isprintable() or char in "\n\r\t"
    )
    if printable / max(len(decoded), 1) < 0.9:
        return None
    normalized = " ".join(decoded.split())
    if not normalized:
        return ""
    if len(normalized) > _TEXT_PREVIEW_CHARS:
        return normalized[: _TEXT_PREVIEW_CHARS - 3] + "..."
    return normalized


def _prefers_chinese_progress(content: str) -> bool:
    cjk_count = 0
    latin_count = 0
    for char in content:
        codepoint = ord(char)
        if (
            0x4E00 <= codepoint <= 0x9FFF
            or 0x3400 <= codepoint <= 0x4DBF
            or 0x20000 <= codepoint <= 0x2A6DF
            or 0x2A700 <= codepoint <= 0x2B73F
            or 0x2B740 <= codepoint <= 0x2B81F
            or 0x2B820 <= codepoint <= 0x2CEAF
            or 0xF900 <= codepoint <= 0xFAFF
        ):
            cjk_count += 1
        elif ("A" <= char <= "Z") or ("a" <= char <= "z"):
            latin_count += 1
    if cjk_count == 0:
        return False
    if latin_count == 0:
        return True
    return cjk_count >= latin_count

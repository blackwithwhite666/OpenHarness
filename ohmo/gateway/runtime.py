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
from ohmo.evals.nutrition_trace import (
    NutritionAnnotationV2,
)
from ohmo.evals.nutrition_persistence import derive_meal_id
from ohmo.gateway.attachment_fingerprints import compute_attachment_fingerprints
from ohmo.gateway.camera import (
    CAMERA_AUTHORITY,
    COALESCED_ATTACHMENT_PROVENANCE_AUTHORITY,
    RetainedAttachmentEvidence,
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
from ohmo.memory_service.honcho_client import HonchoError
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
_NATIVE_REPLY_SOURCE_AUTHORITY = object()
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
    logical_turn_id = _logical_turn_id_for_conversation(turn_ctx=turn_ctx, message=message)
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
    selected = message.metadata.get("_selected_source_binding")
    native_selected = message.metadata.get("_native_reply_source_binding")
    explicit_selection = (
        isinstance(selected, tuple)
        and len(selected) == 2
        and selected[0] is _SELECTED_SOURCE_AUTHORITY
    )
    if (
        not explicit_selection
        and isinstance(native_selected, tuple)
        and len(native_selected) == 2
        and native_selected[0] is _NATIVE_REPLY_SOURCE_AUTHORITY
        and isinstance(native_selected[1], Mapping)
    ):
        selected = (_SELECTED_SOURCE_AUTHORITY, native_selected[1])
    binding = (
        selected[1]
        if isinstance(selected, tuple)
        and len(selected) == 2
        and selected[0] is _SELECTED_SOURCE_AUTHORITY
        and isinstance(selected[1], Mapping)
        else None
    )
    raw_nutrition = recorder.validated_nutrition_envelope if recorder is not None else None
    try:
        nutrition = NutritionAnnotationV2.model_validate(raw_nutrition)
    except (TypeError, ValueError):
        nutrition = None
    if (
        binding is not None
        and nutrition is not None
        and nutrition.record_type in {"meal_correction", "meal_deletion"}
    ):
        receipt_event = binding.get("original_receipt_event_id")
        receipt_operation = binding.get("original_receipt_client_op_id")
        original_source = binding.get("original_source_message_id")
        original_append_source = binding.get("original_append_source_message_id")
        original_session = binding.get("original_gateway_session_id")
        if (
            scope.private_tenant != binding.get("tenant_id")
            or not turn_ctx.is_private
            or turn_ctx.is_forwarded
            or binding.get("source_principal") != source_principal
            or not all(
                isinstance(value, str) and value.strip()
                for value in (
                    receipt_event, receipt_operation, original_source,
                    original_append_source, original_session,
                )
            )
            or not str(receipt_operation).endswith(":assistant")
            or original_source != original_append_source
        ):
            return logical_turn_id, {}, {}
        reply_target = base_metadata.get("reply_to_source_message_id")
        if reply_target is not None:
            base_metadata["reply_to_native_message_id"] = reply_target
        # Explicit selection uses the reader's contextual receipt path. Its
        # native reply is context, not an alternative target or a direct-reply
        # shortcut around the selected original event and operation receipt.
        if explicit_selection:
            base_metadata["reply_to_source_message_id"] = None
        evidence = {
            "schema_version": 2,
            "tenant_id": scope.private_tenant,
            "source_principal": source_principal,
            "gateway_session_id": original_session,
            "source_message_id": original_source,
            "append_source_message_id": original_append_source,
            "is_private": True,
            "is_forwarded": False,
            "is_group": False,
            "original_receipt_event_id": receipt_event,
            "original_receipt_client_op_id": receipt_operation,
        }
        base_metadata["selected_source"] = evidence
        base_metadata["target_meal_id"] = derive_meal_id(
            tenant_id=scope.private_tenant,
            source_principal=source_principal,
            gateway_session_id=original_session,
            source_message_id=original_append_source,
        )
    elif (
        binding is not None
        and nutrition is not None
        and nutrition.record_type == "meal_observation"
        and nutrition.consumption_status == "consumed"
        and scope.private_tenant == binding.get("tenant_id")
        and turn_ctx.is_private
        and not turn_ctx.is_forwarded
        and binding.get("source_principal") == source_principal
        and isinstance(binding.get("photo_source_message_id"), str)
        and isinstance(binding.get("attachment_id"), str)
        and isinstance(base_metadata.get("source_message_id"), str)
    ):
        base_metadata["photo_occurrence_source"] = {
            "schema_version": 1,
            "tenant_id": scope.private_tenant,
            "source_principal": source_principal,
            "gateway_session_id": turn_ctx.session_id,
            "photo_gateway_session_id": binding.get("photo_gateway_session_id"),
            "source_message_id": binding["photo_source_message_id"],
            "append_source_message_id": base_metadata["source_message_id"],
            "attachment_id": binding["attachment_id"],
            "received_at": binding.get("received_at"),
            "chat_id": str(message.chat_id),
            "session_key": binding.get("session_key") or message.session_key or message.session_key_override,
            "is_private": True,
            "is_forwarded": False,
            "is_group": False,
            "source_origin": binding.get("source_origin", "telegram"),
            "origin_principal": binding.get("origin_principal"),
            "camera_candidate_id": binding.get("camera_candidate_id"),
            "native_photo_message_id": binding.get("native_photo_message_id"),
        }
    user_metadata = dict(base_metadata)
    assistant_metadata = dict(base_metadata)
    assistant_metadata["client_op_id"] = f"{logical_turn_id}:assistant"
    decision_trace = recorder.decision_trace_envelope if recorder is not None else None
    if decision_trace is not None:
        assistant_metadata["decision_trace"] = dict(decision_trace)
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
        "capture_time": _trusted_utc_iso(metadata.get("_camera_capture_time")),
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


def _confirmed_nutrition_append_receipt(
    append_receipt: ConversationAppendReceipt,
    receipt_metadata: Mapping[str, object],
    annotation: NutritionAnnotationV2,
) -> dict[str, object] | None:
    """Export a compact projection of the exact persisted append receipt."""
    event_id = append_receipt.assistant_message_id
    op_id = append_receipt.assistant_client_op_id
    logical = receipt_metadata.get("logical_turn_id")
    if (
        not isinstance(event_id, str) or not event_id
        or not isinstance(op_id, str) or not isinstance(logical, str) or not logical
        or op_id != f"{logical}:assistant"
        or append_receipt.user_client_op_id != f"{logical}:user"
        or receipt_metadata.get("client_op_id") != op_id
        or receipt_metadata.get("role") != "assistant"
        or receipt_metadata.get("ingest_source") not in {"telegram", "dropbox_camera"}
        or not all(isinstance(receipt_metadata.get(key), str) and receipt_metadata.get(key)
                   for key in ("tenant_id", "source_principal", "gateway_session_id",
                               "source_message_id", "decision_trace_episode_id"))
    ):
        return None
    receipt: dict[str, object] = {
        "schema_version": 1,
        "event_id": event_id,
        "client_op_id": op_id,
        "user_client_op_id": append_receipt.user_client_op_id,
        "logical_turn_id": logical,
        "trace_episode_id": receipt_metadata["decision_trace_episode_id"],
        "tenant_id": receipt_metadata["tenant_id"],
        "source_principal": receipt_metadata["source_principal"],
        "gateway_session_id": receipt_metadata["gateway_session_id"],
        "source_message_id": receipt_metadata["source_message_id"],
        "ingest_source": receipt_metadata["ingest_source"],
        "annotation": _nutrition_annotation_metadata(annotation),
    }
    for key in ("selected_source", "photo_occurrence_source", "target_meal_id"):
        if key in receipt_metadata:
            receipt[key] = receipt_metadata[key]
    return receipt


def _committed_nutrition_reply(
    annotation: NutritionAnnotationV2,
    *,
    status: str,
    assistant_content: str | None = None,
) -> str:
    """Report only the status proven by the append receipt."""
    if assistant_content is not None:
        content = assistant_content.strip()
        return f"{content}\n{status}" if content else status
    if annotation.record_type != "meal_observation":
        return status
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
        "principal": binding.get("origin_principal") or binding.get("source_principal"),
        "chat_id": binding.get("chat_id"),
        "session_key": binding.get("session_key"),
        "gateway_session_id": binding.get("photo_gateway_session_id"),
        "source_message_id": binding.get("source_message_id"),
    }
    if binding.get("source_origin") == "dropbox_camera":
        expected.update(
            source_origin="dropbox_camera",
            owner_principal=binding.get("source_principal"),
            camera_candidate_id=binding.get("camera_candidate_id"),
            native_photo_message_id=binding.get("native_photo_message_id"),
        )
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
        "gateway_session_id": binding.get("gateway_session_id"),
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
                "photo_gateway_session_id": gateway_session_id,
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
    def _with_user_photo_context(prompt: str, sent_at: datetime | None) -> str:
        """Add the authenticated source time as factual context only."""
        if sent_at is None:
            return prompt
        return (
            prompt + "\n\n# Trusted image source time\n"
            "The authenticated participant sent the attached image at "
            f"{sent_at.isoformat()}. This timestamp is source provenance; "
            "the conversation determines its meaning."
        )

    async def stream_message(self, message: InboundMessage, session_key: str):
        """Submit an inbound channel message and yield progress + final reply updates."""
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
        self._bind_session_owner(message, session_key, turn_ctx)
        memory_scope = (
            MemoryScope(
                private_tenant=self._gateway_config.camera_ingress.tenant_id,
                shared_tenants=(),
            )
            if camera_authorized and message.sender_id == "__camera__"
            else self._resolve_turn_memory_scope(turn_ctx)
        )

        prior_messages = getattr(bundle.engine, "messages", [])
        prior_messages = prior_messages if isinstance(prior_messages, list) else []
        user_photo_meal_at = self._trusted_user_photo_time(
            message, turn_ctx=turn_ctx, history=prior_messages
        )
        system_prompt = await self._runtime_system_prompt(
                    bundle,
                    user_prompt,
                    turn_ctx=turn_ctx,
                    memory_scope=memory_scope,
                    include_todo=todo_lifecycle,
                )
        system_prompt = self._with_user_photo_context(
            system_prompt, user_photo_meal_at
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
        if recorder is not None and user_photo_meal_at is not None:
            recorder.set_authoritative_nutrition_meal_at(
                user_photo_meal_at, preserve_explicit=True
            )
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
                            user_photo_meal_at=user_photo_meal_at,
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
                            user_photo_meal_at=user_photo_meal_at,
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
                        user_photo_meal_at=user_photo_meal_at,
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
                    user_photo_meal_at=user_photo_meal_at,
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
        user_photo_meal_at: datetime | None = None,
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
                    user_photo_meal_at=user_photo_meal_at,
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
            bundle.engine.set_system_prompt(continue_prompt)
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
        user_photo_meal_at: datetime | None = None,
        wellness_reminder: _TrustedReminderWellnessAdmission | None = None,
    ):
        message.metadata.pop("_selected_source_binding", None)
        todo_error = getattr(bundle, "_todo_runtime_error", None)
        if todo_lifecycle and isinstance(todo_error, TodoRuntimeStateError):
            yield GatewayStreamUpdate(
                kind="error",
                text=self._todo_runtime_error_text(todo_error),
                metadata={"_session_key": session_key},
            )
            return
        # A finalized gateway turn can be replayed by the transport after the
        # engine has already advanced. Reconcile its exact durable operation
        # before QueryEngine applies its intentionally strict stale-turn rule.
        if await self._confirmed_exact_owner_replay(
            bundle=bundle,
            message=message,
            user_message=user_message,
            user_text=message.content or user_prompt,
            turn_ctx=turn_ctx,
            memory_scope=memory_scope,
        ):
            return
        system_prompt = await self._runtime_system_prompt(
            bundle,
            user_prompt,
            turn_ctx=turn_ctx,
            memory_scope=memory_scope,
            include_todo=todo_lifecycle,
        )
        system_prompt = self._with_user_photo_context(
            system_prompt, user_photo_meal_at
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
        loaded_attachment_ids: set[str] = set()

        def on_attachment_load_started(attachment_id: str) -> None:
            if active_turn["active"]:
                # Loading is for visual context only. It does not select a meal.
                return

        def note_attachment_loaded(attachment_id: str) -> str | None:
            if active_turn["active"]:
                loaded_attachment_ids.add(attachment_id)
            return None

        def select_loaded_source(attachment_id: str) -> str | None:
            if not active_turn["active"] or attachment_id not in loaded_attachment_ids:
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
                return None
            message.metadata["_selected_source_binding"] = (
                _SELECTED_SOURCE_AUTHORITY, binding
            )
            stamped_time = _trusted_utc_iso(binding.get("received_at"))
            if recorder is not None and stamped_time is not None:
                recorder.set_authoritative_nutrition_meal_at(
                    datetime.fromisoformat(stamped_time),
                    preserve_explicit=True,
                    historical_photo=True,
                )
            return stamped_time or "source-verified"

        def begin_source_selection(_attachment_id: str) -> None:
            # A failed explicit choice cannot leave an earlier target or its
            # trusted default time eligible for finalization.
            message.metadata.pop("_selected_source_binding", None)
            if recorder is not None:
                recorder.set_authoritative_nutrition_meal_at(None)
        decision_trace_restore = _install_gateway_decision_trace_recorder(
            bundle.engine,
            recorder,
        )
        self._register_conversation_image_tool(
            bundle,
            current_message=user_message,
            on_attachment_load_started=on_attachment_load_started,
            on_attachment_loaded=note_attachment_loaded,
            on_source_selected=select_loaded_source,
            on_source_selection_started=begin_source_selection,
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

        raw_annotation = recorder.validated_nutrition_envelope if recorder is not None else None
        try:
            finalizer_nutrition = NutritionAnnotationV2.model_validate(raw_annotation)
        except (TypeError, ValueError):
            finalizer_nutrition = None
        ordinary_meal = bool(
            finalizer_nutrition is not None
            and finalizer_nutrition.record_type == "meal_observation"
            and finalizer_nutrition.consumption_status == "consumed"
        )
        requested_correction = bool(
            finalizer_nutrition is not None
            and finalizer_nutrition.record_type in {"meal_correction", "meal_deletion"}
        )
        selected_binding = message.metadata.get("_selected_source_binding")
        ordinary_correction = bool(
            requested_correction
            and isinstance(selected_binding, tuple)
            and len(selected_binding) == 2
            and selected_binding[0] is _SELECTED_SOURCE_AUTHORITY
            and isinstance(selected_binding[1], Mapping)
        )
        append_receipt = None
        if reply:
            append_receipt = await self._append_conversation_turn(
                turn_ctx=turn_ctx,
                memory_scope=memory_scope,
                message=message,
                recorder=recorder,
                user_text=message.content or user_prompt,
                assistant_text=reply,
            )
        native_source_binding = message.metadata.get("_native_reply_source_binding")
        ordinary_correction = ordinary_correction or bool(
            isinstance(native_source_binding, tuple)
            and len(native_source_binding) == 2
            and native_source_binding[0] is _NATIVE_REPLY_SOURCE_AUTHORITY
            and isinstance(native_source_binding[1], Mapping)
        )

        metadata: dict[str, object] = {"_session_key": session_key}
        receipt_metadata = (
            append_receipt.assistant_metadata
            if append_receipt is not None and isinstance(append_receipt.assistant_metadata, Mapping)
            else {}
        )
        receipt_matches_turn = bool(
            append_receipt is not None
            and isinstance(memory_scope, MemoryScope)
            and append_receipt.user_client_op_id == f"{receipt_metadata.get('logical_turn_id')}:user"
            and append_receipt.assistant_client_op_id == f"{receipt_metadata.get('logical_turn_id')}:assistant"
            and receipt_metadata.get("role") == "assistant"
            and receipt_metadata.get("client_op_id") == append_receipt.assistant_client_op_id
            and receipt_metadata.get("tenant_id") == memory_scope.private_tenant
            and receipt_metadata.get("source_principal")
            == f"{message.channel}:{canonical_principal(message.channel, turn_ctx.principal)}"
            and receipt_metadata.get("gateway_session_id") == turn_ctx.session_id
            and receipt_metadata.get("source_message_id")
            == _normalize_source_message_ref(message.metadata.get("message_id"))
            and receipt_metadata.get("is_group") is False
            and receipt_metadata.get("is_forwarded") is False
            and isinstance(append_receipt.assistant_content, str)
        )
        selected_binding = message.metadata.get("_selected_source_binding")
        selected_source = (
            selected_binding[1]
            if isinstance(selected_binding, tuple)
            and len(selected_binding) == 2
            and selected_binding[0] is _SELECTED_SOURCE_AUTHORITY
            and isinstance(selected_binding[1], Mapping)
            else None
        )
        if (
            selected_source is not None
            and selected_source.get("source_origin") == "dropbox_camera"
            and selected_source.get("tenant_id") == getattr(memory_scope, "private_tenant", None)
            and selected_source.get("source_principal")
            == f"{message.channel}:{canonical_principal(message.channel, turn_ctx.principal)}"
        ):
            metadata["nutrition_selected_source_review"] = dict(selected_source)
        if (
            receipt_matches_turn
            and selected_source is not None
            and selected_source.get("source_origin") == "dropbox_camera"
            and isinstance(selected_source.get("camera_candidate_id"), str)
            and isinstance(selected_source.get("native_photo_message_id"), str)
        ):
            ingress = getattr(self, "_camera_ingress", None)
            release = getattr(ingress, "note_owner_source_receipt", None)
            if callable(release):
                release(
                    candidate_id=selected_source["camera_candidate_id"],
                    native_photo_message_id=selected_source["native_photo_message_id"],
                )
        if ordinary_meal:
            stored_meal = None
            if receipt_matches_turn and _receipt_has_consumed_nutrition(receipt_metadata):
                trace = receipt_metadata.get("decision_trace")
                annotations = trace.get("annotations") if isinstance(trace, Mapping) else None
                nutrition = annotations.get("nutrition") if isinstance(annotations, Mapping) else None
                try:
                    stored_meal = NutritionAnnotationV2.model_validate(nutrition)
                except (TypeError, ValueError):
                    stored_meal = None
            if stored_meal is not None and append_receipt is not None:
                metadata.update(
                    nutrition_append_event_id=append_receipt.assistant_message_id,
                    nutrition_actual_append_receipt=_confirmed_nutrition_append_receipt(
                        append_receipt, receipt_metadata, stored_meal
                    ),
                    nutrition_sync_status="pending",
                    nutrition_committed_annotation=_nutrition_annotation_metadata(stored_meal),
                    nutrition_model_proposal_annotation=_nutrition_annotation_metadata(finalizer_nutrition),
                    nutrition_proposal_matches_committed=(
                        stored_meal.model_dump(mode="json")
                        == finalizer_nutrition.model_dump(mode="json")
                    ),
                )
                reply = _committed_nutrition_reply(
                    stored_meal,
                    status="Записано; баланс обновляется.",
                    assistant_content=append_receipt.assistant_content,
                )
                source = receipt_metadata.get("photo_occurrence_source")
                if isinstance(source, Mapping) and _record_consumed_photo_occurrence(
                    getattr(bundle.engine, "messages", []),
                    source,
                    receipt_event_id=append_receipt.assistant_message_id,
                    client_op_id=append_receipt.assistant_client_op_id,
                    append_source_message_id=str(receipt_metadata.get("source_message_id") or ""),
                ):
                    await self._save_snapshot(bundle, session_key, user_prompt)
            else:
                reply = _committed_nutrition_reply(
                    finalizer_nutrition,
                    status="Не удалось подтвердить сохранение записи.",
                    assistant_content=reply,
                )
        elif requested_correction:
            expected = (
                _build_conversation_turn_metadata(
                    turn_ctx=turn_ctx, message=message, scope=memory_scope, recorder=recorder
                )[2]
                if isinstance(memory_scope, MemoryScope) and ordinary_correction
                else {}
            )
            stored_correction = None
            if (
                receipt_matches_turn
                and ordinary_correction
                and receipt_metadata.get("selected_source") == expected.get("selected_source")
                and receipt_metadata.get("target_meal_id") == expected.get("target_meal_id")
            ):
                trace = receipt_metadata.get("decision_trace")
                annotations = trace.get("annotations") if isinstance(trace, Mapping) else None
                nutrition = annotations.get("nutrition") if isinstance(annotations, Mapping) else None
                try:
                    stored_correction = NutritionAnnotationV2.model_validate(nutrition)
                except (TypeError, ValueError):
                    stored_correction = None
            if stored_correction is not None and stored_correction.record_type in {
                "meal_correction", "meal_deletion"
            } and append_receipt is not None:
                metadata.update(
                    nutrition_append_event_id=append_receipt.assistant_message_id,
                    nutrition_actual_append_receipt=_confirmed_nutrition_append_receipt(
                        append_receipt, receipt_metadata, stored_correction
                    ),
                    nutrition_sync_status="pending",
                    nutrition_committed_annotation=_nutrition_annotation_metadata(stored_correction),
                    nutrition_model_proposal_annotation=_nutrition_annotation_metadata(finalizer_nutrition),
                    nutrition_proposal_matches_committed=(
                        stored_correction.model_dump(mode="json")
                        == finalizer_nutrition.model_dump(mode="json")
                    ),
                )
                reply = _committed_nutrition_reply(
                    stored_correction,
                    status="Изменение сохранено; баланс обновляется.",
                    assistant_content=append_receipt.assistant_content,
                )
            else:
                reply = _committed_nutrition_reply(
                    finalizer_nutrition,
                    status="Не удалось подтвердить сохранение изменения.",
                    assistant_content=reply,
                )
        final_media = _extract_final_reply_media(reply, emitted_media)
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

    async def _confirmed_exact_owner_replay(
        self,
        *,
        bundle: RuntimeBundle,
        message: InboundMessage,
        user_message: ConversationMessage | str,
        user_text: str,
        turn_ctx: TurnContext,
        memory_scope: MemoryScope | None,
    ) -> bool:
        """Recognize only a fully matched prior private owner exchange."""
        source_id = _normalize_source_message_ref((message.metadata or {}).get("message_id"))
        owner_principal = canonical_principal("telegram", turn_ctx.principal)
        configured_tenant = self._gateway_config.family_principals.get(owner_principal)
        if (
            source_id is None
            or not isinstance(user_message, ConversationMessage)
            or message.channel != "telegram"
            or turn_ctx.channel != "telegram"
            or (turn_ctx.is_owner is not True
                and configured_tenant != getattr(memory_scope, "private_tenant", None))
            or not turn_ctx.is_private
            or turn_ctx.is_forwarded
            or not is_private_message(message)
            or message.sender_id == "__camera__"
            or (message.metadata or {}).get("is_group") is not False
            or memory_scope is None
            or not self._honcho_turn_allowed(turn_ctx, memory_scope)
        ):
            return False
        event_id = user_message.event_id
        history = getattr(bundle.engine, "messages", [])
        if not isinstance(event_id, str) or not isinstance(history, list):
            return False
        matching = [
            index for index, prior in enumerate(history)
            if isinstance(prior, ConversationMessage)
            and prior.role == "user" and prior.event_id == event_id
        ]
        if not matching:
            return False
        logical_turn_id = _logical_turn_id_for_conversation(turn_ctx=turn_ctx, message=message)
        backend = self._shadow_backend_for_scope(memory_scope)
        if backend is None:
            return False
        try:
            receipt = await backend.reconcile_durable_exchange(
                f"{logical_turn_id}:user", f"{logical_turn_id}:assistant"
            )
        except (ConversationReconciliationError, HonchoError):
            return False
        if receipt is None:
            return False
        metadata = receipt.assistant_metadata
        expected_principal = f"telegram:{canonical_principal('telegram', turn_ctx.principal)}"
        return bool(
            isinstance(receipt.user_message_id, str)
            and receipt.user_message_id
            and isinstance(receipt.assistant_message_id, str)
            and receipt.assistant_message_id
            and receipt.user_client_op_id == f"{logical_turn_id}:user"
            and receipt.assistant_client_op_id == f"{logical_turn_id}:assistant"
            and isinstance(receipt.user_content, str)
            and receipt.user_content == user_text
            and isinstance(receipt.assistant_content, str)
            and bool(receipt.assistant_content.strip())
            and metadata.get("role") == "assistant"
            and metadata.get("logical_turn_id") == logical_turn_id
            and metadata.get("client_op_id") == receipt.assistant_client_op_id
            and metadata.get("tenant_id") == memory_scope.private_tenant
            and metadata.get("source_principal") == expected_principal
            and metadata.get("gateway_session_id") == turn_ctx.session_id == bundle.session_id
            and metadata.get("source_message_id") == source_id
            and metadata.get("is_group") is False
            and metadata.get("is_forwarded") is False
            and metadata.get("ingest_source") in {"telegram", "dropbox_camera"}
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
    ) -> ConversationAppendReceipt | None:
        # A Camera producer may provide image context, but it is never an
        # owner-authored exchange. Owner annotations use this ordinary append.
        if turn_ctx.camera_authorized and turn_ctx.principal == "__camera__":
            return None
        annotation = recorder.validated_nutrition_envelope if recorder is not None else None
        validated = (
            NutritionAnnotationV2.model_validate(annotation)
            if isinstance(annotation, Mapping) and annotation.get("schema_version", 1) == 2
            else None
        )
        selected_binding = message.metadata.get("_selected_source_binding")
        if (
            isinstance(selected_binding, tuple)
            and len(selected_binding) == 2
            and selected_binding[0] is _SELECTED_SOURCE_AUTHORITY
            and not isinstance(selected_binding[1], Mapping)
        ):
            return None
        native_reply_candidate = bool(
            validated is not None
            and validated.record_type in {"meal_correction", "meal_deletion"}
            and turn_ctx.channel == "telegram"
            and message.channel == "telegram"
            and turn_ctx.is_private
            and not turn_ctx.is_forwarded
            and is_private_message(message)
            and _normalize_source_message_ref(message.metadata.get("reply_to_message_id"))
        )
        ordinary_correction = bool(
            validated is not None
            and validated.record_type in {"meal_correction", "meal_deletion"}
            and isinstance(selected_binding, tuple)
            and len(selected_binding) == 2
            and selected_binding[0] is _SELECTED_SOURCE_AUTHORITY
            and isinstance(selected_binding[1], Mapping)
        )
        selected_binding_value = (
            selected_binding[1]
            if isinstance(selected_binding, tuple)
            and len(selected_binding) == 2
            and selected_binding[0] is _SELECTED_SOURCE_AUTHORITY
            and isinstance(selected_binding[1], Mapping)
            else None
        )
        if (
            validated is not None
            and validated.record_type == "meal_observation"
            and validated.consumption_status == "consumed"
            and selected_binding_value is not None
            and isinstance(selected_binding_value.get("original_receipt_event_id"), str)
            and selected_binding_value.get("original_receipt_event_id")
            and validated.explicit_new_consumption is not True
        ):
            # A source selection that already resolves to a consumed append is
            # a duplicate observation unless the structured annotation marks a
            # separate consumption. Corrections remain eligible below.
            return None
        if (
            validated is not None
            and validated.record_type in {"meal_correction", "meal_deletion"}
            and not ordinary_correction
            and not native_reply_candidate
        ):
            return None
        if self._gateway_config.conversation_learning is not True:
            return None
        scope = self._coerce_memory_scope(turn_ctx, memory_scope)
        if scope is None or not self._honcho_turn_allowed(turn_ctx, scope):
            return None
        shadow_backend = self._shadow_backend_for_scope(scope)
        if shadow_backend is None:
            return None
        if native_reply_candidate and not ordinary_correction:
            native_binding = await self._resolve_native_reply_meal_source(
                turn_ctx=turn_ctx, message=message, scope=scope, backend=shadow_backend
            )
            if native_binding is None:
                return None
            message.metadata["_native_reply_source_binding"] = (
                _NATIVE_REPLY_SOURCE_AUTHORITY, native_binding
            )
            ordinary_correction = True
        _, user_metadata, assistant_metadata = _build_conversation_turn_metadata(
            turn_ctx=turn_ctx,
            message=message,
            scope=scope,
            recorder=recorder,
        )
        if ordinary_correction and not assistant_metadata.get("target_meal_id"):
            return None
        if (
            message.channel == "telegram"
            and turn_ctx.is_private
            and not turn_ctx.is_forwarded
            and message.sender_id != "__camera__"
        ):
            binding = (
                selected_binding[1]
                if isinstance(selected_binding, tuple)
                and len(selected_binding) == 2
                and selected_binding[0] is _SELECTED_SOURCE_AUTHORITY
                and isinstance(selected_binding[1], Mapping)
                else None
            )
            origin = binding.get("source_origin") if isinstance(binding, Mapping) else None
            ingest_source = "dropbox_camera" if origin == "dropbox_camera" else "telegram"
            for metadata in (user_metadata, assistant_metadata):
                metadata.update(
                    ingest_source=ingest_source,
                    confirmation_required=ingest_source == "dropbox_camera",
                )
        receipt = await shadow_backend.append_exchange(
            user_text,
            assistant_text,
            user_metadata=user_metadata,
            assistant_metadata=assistant_metadata,
            durable=(
                ordinary_correction
                or (
                    validated is not None
                    and validated.record_type == "meal_observation"
                    and validated.consumption_status == "consumed"
                )
            ),
        )
        return receipt

    async def _resolve_native_reply_meal_source(
        self,
        *,
        turn_ctx: TurnContext,
        message: InboundMessage,
        scope: MemoryScope,
        backend: ShadowMemoryBackend,
    ) -> dict[str, object] | None:
        """Resolve a native reply to one exact durable, owned meal receipt."""
        source_id = _normalize_source_message_ref(
            (message.metadata or {}).get("reply_to_message_id")
        )
        if (
            source_id is None
            or turn_ctx.channel != "telegram"
            or message.channel != "telegram"
            or not turn_ctx.is_private
            or turn_ctx.is_forwarded
            or not is_private_message(message)
            or message.sender_id == "__camera__"
        ):
            return None
        source_message = InboundMessage(
            channel="telegram",
            sender_id=str(turn_ctx.principal),
            chat_id=str(turn_ctx.chat_id),
            content="",
            metadata={"message_id": source_id},
        )
        logical_turn_id = _logical_turn_id_for_conversation(
            turn_ctx=turn_ctx, message=source_message
        )
        try:
            receipt = await backend.reconcile_durable_exchange(
                f"{logical_turn_id}:user", f"{logical_turn_id}:assistant"
            )
        except (ConversationReconciliationError, HonchoError):
            return None
        if receipt is None:
            return None
        metadata = receipt.assistant_metadata
        trace = metadata.get("decision_trace")
        annotations = trace.get("annotations") if isinstance(trace, Mapping) else None
        raw_nutrition = annotations.get("nutrition") if isinstance(annotations, Mapping) else None
        try:
            original_nutrition = NutritionAnnotationV2.model_validate(raw_nutrition)
        except (TypeError, ValueError):
            return None
        source_principal = (
            f"telegram:{canonical_principal('telegram', turn_ctx.principal)}"
        )
        if (
            receipt.user_client_op_id != f"{logical_turn_id}:user"
            or receipt.assistant_client_op_id != f"{logical_turn_id}:assistant"
            or metadata.get("role") != "assistant"
            or metadata.get("client_op_id") != receipt.assistant_client_op_id
            or metadata.get("tenant_id") != scope.private_tenant
            or metadata.get("source_principal") != source_principal
            or metadata.get("gateway_session_id") != turn_ctx.session_id
            or metadata.get("source_message_id") != source_id
            or metadata.get("is_group") is not False
            or metadata.get("is_forwarded") is not False
            or metadata.get("ingest_source") != "telegram"
            or not isinstance(receipt.assistant_message_id, str)
            or not receipt.assistant_message_id
            or original_nutrition.record_type != "meal_observation"
            or original_nutrition.consumption_status != "consumed"
        ):
            return None
        return {
            "schema_version": 2,
            "tenant_id": scope.private_tenant,
            "source_principal": source_principal,
            "gateway_session_id": turn_ctx.session_id,
            "source_message_id": source_id,
            "append_source_message_id": source_id,
            "original_source_message_id": source_id,
            "original_append_source_message_id": source_id,
            "original_gateway_session_id": turn_ctx.session_id,
            "original_receipt_event_id": receipt.assistant_message_id,
            "original_receipt_client_op_id": receipt.assistant_client_op_id,
            "is_private": True,
            "is_forwarded": False,
            "is_group": False,
        }

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
            bundle.engine.set_system_prompt(system_prompt)
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
        on_source_selected=None,
        on_source_selection_started=None,
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
                on_source_selected=on_source_selected,
                on_source_selection_started=on_source_selection_started,
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
    camera = message.metadata or {}
    if (
        message.sender_id == "__camera__"
        and camera.get("_camera_authority") is CAMERA_AUTHORITY
        and camera.get("_synthetic") is True
        and camera.get("_camera_source_origin") == "dropbox_camera"
        and camera.get("_camera_photo_delivery_confirmed") is True
        and type(camera.get("_camera_photo_id")) is int
        and camera.get("_camera_photo_id", 0) > 0
        and isinstance(camera.get("_camera_candidate_id"), str)
        and camera.get("_camera_candidate_id")
        and isinstance(camera.get("_camera_owner_principal"), str)
        and camera.get("_camera_owner_principal")
    ):
        capture_time = _trusted_utc_iso(camera.get("_camera_capture_time"))
        source_provenance.update(
            source_origin="dropbox_camera",
            origin_principal="telegram:__camera__",
            owner_principal=(
                "telegram:" + canonical_principal(
                    "telegram", str(camera["_camera_owner_principal"])
                )
            ),
            camera_candidate_id=camera["_camera_candidate_id"],
            native_photo_message_id=str(camera["_camera_photo_id"]),
            source_message_id=str(camera["_camera_photo_id"]),
            photo_source_message_id=str(camera["_camera_photo_id"]),
            # Delivery time is not a substitute for missing Camera capture
            # time. Keep it separately as delivery evidence for audit/debug.
            received_at=capture_time,
            delivered_at=source_timestamp,
            timestamp_authority="camera_capture_time" if capture_time else "camera_delivery_receipt",
        )
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

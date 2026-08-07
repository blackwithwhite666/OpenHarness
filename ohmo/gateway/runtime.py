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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from ohmo.contact_registry import ContactStore
from ohmo.evals import GatewayEvalRecorder
from ohmo.evals.nutrition_trace import (
    NutritionAnnotationV2,
    build_nutrition_display_summary,
)
from ohmo.gateway.attachment_fingerprints import compute_attachment_fingerprints
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
from ohmo.gateway.provider_commands import (
    handle_gateway_model_command,
    handle_gateway_provider_command,
)
from ohmo.gateway.router import session_key_for_message
from ohmo.gateway.send_message_tool import SendTelegramMessageTool
from ohmo.gateway.turn_context import TurnContext, build_turn_context, canonical_principal
from ohmo.group_registry import load_managed_group_record, normalize_cwd
from ohmo.memory import create_memory_command_backend, ensure_catalog_migrated
from ohmo.memory_backend import (
    CatalogMemoryBackend,
    ConversationAppendReceipt,
    FileMemoryBackend,
    MemoryBackend,
    ShadowMemoryBackend,
    make_memory_backend,
    make_tenant_shadow_backend,
)
from ohmo.memory_judge import judge_enabled, judge_interval, run_memory_judge
from ohmo.memory_store import MemoryStore
from ohmo.memory_tool import OhmoMemoryTool
from ohmo.nutrition_ingest.freshness import normalized_exif_capture_time
from ohmo.nutrition_ingest.models import ExifMetadata
from ohmo.nutrition_ingest.trust import COORDINATOR_TRUST_TOKEN
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
    clear_session_work_dir,
    get_session_work_dir,
    reap_stale_work_dirs,
)
from ohmo.todo_store import TodoStore
from ohmo.todo_write_tool import OhmoTodoWriteTool
from ohmo.workspace import (
    get_memory_dir,
    get_plugins_dir,
    get_sessions_dir,
    get_skills_dir,
    initialize_workspace,
)
from openharness.channels.bus.events import InboundMessage, OutboundMessage
from openharness.commands import CommandContext, CommandResult, lookup_skill_slash_command
from openharness.engine.messages import (
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
from openharness.tools.mcp_tool import McpToolAdapter, WellnessUserIdInjectingAdapter
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
_NUTRITION_SENDER = "__nutrition_ingest__"
_NUTRITION_CANDIDATE_RE = re.compile(r"^dropbox-camera-v1-[0-9a-f]{64}$")


def _trusted_nutrition_request(message: InboundMessage) -> dict[str, str] | None:
    """Validate the coordinator-owned synthetic estimation marker."""
    metadata = message.metadata or {}
    if (
        message.sender_id != _NUTRITION_SENDER
        or metadata.get("_nutrition_trusted") is not True
        or metadata.get("_nutrition_trust_token") is not COORDINATOR_TRUST_TOKEN
    ):
        return None
    if message.channel != "telegram":
        return None
    candidate = metadata.get("_nutrition_candidate_id")
    operation = metadata.get("_nutrition_client_op_id")
    phase = metadata.get("_nutrition_phase")
    principal = metadata.get("_nutrition_principal")
    tenant = metadata.get("_nutrition_tenant_id")
    chat_id = metadata.get("_nutrition_chat_id")
    session_key = metadata.get("_nutrition_session_key")
    if not all(isinstance(value, str) and value.strip() for value in (candidate, operation, phase, principal, tenant, chat_id, session_key)):
        return None
    if not _NUTRITION_CANDIDATE_RE.fullmatch(candidate) or phase != "estimation":
        return None
    if (
        re.fullmatch(r"[1-9][0-9]*", principal) is None
        or tenant != "marina"
        or chat_id != str(message.chat_id)
        or session_key != message.session_key
    ):
        return None
    if operation != f"{candidate}:meal-observation:v1":
        return None
    return {
        "candidate_id": candidate,
        "client_op_id": operation,
        "phase": phase,
        "principal": principal,
        "chat_id": chat_id,
        "session_key": session_key,
    }


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
        return timestamp.astimezone(UTC).isoformat()
    except (OverflowError, ValueError):
        return None


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


def _build_conversation_turn_metadata(
    *,
    turn_ctx: TurnContext,
    message: InboundMessage,
    scope: MemoryScope,
    recorder: GatewayEvalRecorder | None = None,
) -> tuple[str, dict[str, object], dict[str, object]]:
    logical_turn_id = _logical_turn_id_for_conversation(
        turn_ctx=turn_ctx,
        message=message,
    )
    message_metadata = message.metadata or {}
    trusted_nutrition = _trusted_nutrition_request(message)
    source_principal = (
        f"telegram:{trusted_nutrition['principal']}"
        if trusted_nutrition is not None
        else f"{turn_ctx.channel}:{canonical_principal(turn_ctx.channel, turn_ctx.principal)}"
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
    base_metadata: dict[str, object] = {
        "tenant_id": scope.private_tenant,
        "source_principal": source_principal,
        "gateway_session_id": turn_ctx.session_id,
        "logical_turn_id": logical_turn_id,
        "client_op_id": f"{logical_turn_id}:user",
        "decision_trace_status": decision_trace_status,
        "nutrition_annotation_status": nutrition_annotation_status,
        "decision_trace_episode_id": recorder.episode_id if recorder is not None else None,
        "received_at": _trusted_utc_iso(message.timestamp),
        "is_forwarded": turn_ctx.is_forwarded,
        "source_message_at": _trusted_utc_iso(message_metadata.get("source_message_at")),
        "source_message_id": _normalize_source_message_ref(message_metadata.get("message_id")),
        "reply_to_source_message_id": _normalize_source_message_ref(
            message_metadata.get("reply_to_message_id")
        ),
        "attachment_fingerprints": compute_attachment_fingerprints(message.media),
    }
    if trusted_nutrition is not None:
        base_metadata.update(
            {
                "_nutrition_trusted": True,
                "ingest_source": "dropbox_camera",
                "confirmation_required": True,
                "candidate_id": trusted_nutrition["candidate_id"],
                "nutrition_phase": trusted_nutrition["phase"],
            }
        )
    user_metadata = dict(base_metadata)
    assistant_metadata = dict(base_metadata)
    if trusted_nutrition is not None:
        user_metadata["client_op_id"] = f"{trusted_nutrition['candidate_id']}:meal-user:v1"
        assistant_metadata["client_op_id"] = trusted_nutrition["client_op_id"]
    else:
        # Never accept operation/provenance fields from channel metadata.  The
        # values above are gateway-owned and regenerated for ordinary turns.
        assistant_metadata["client_op_id"] = f"{logical_turn_id}:assistant"
    decision_trace = recorder.decision_trace_envelope if recorder is not None else None
    if decision_trace is not None:
        assistant_metadata["decision_trace"] = dict(decision_trace)
    return logical_turn_id, user_metadata, assistant_metadata


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


def _trusted_bound_reminder(message: InboundMessage) -> dict[str, str | None] | None:
    """Return the trusted recipient binding of a scheduler-originated synthetic
    reminder turn, or ``None`` for any other message.

    Trusted means: the scheduler sender sentinel + the synthetic flag + the
    binding the scheduler stamped at fire time. A live user's metadata can
    never mint this binding — channel adapters don't accept these keys and the
    sentinel is only set by the in-process scheduler.
    """
    metadata = message.metadata or {}
    if message.sender_id != _SCHEDULER_SENDER or not metadata.get("_synthetic"):
        return None
    recipient_chat_id = str(metadata.get("_reminder_recipient_chat_id") or "").strip()
    reminder_id = str(metadata.get("_reminder_id") or "").strip()
    if not recipient_chat_id or not reminder_id:
        return None
    return {
        "reminder_id": reminder_id,
        "recipient_chat_id": recipient_chat_id,
        "recipient_principal": str(metadata.get("_reminder_recipient_principal") or "").strip(),
        "recipient_label": str(metadata.get("_reminder_recipient_label") or "").strip(),
        "wellness_tenant": str(metadata.get("_reminder_wellness_tenant") or "").strip() or None,
    }


def _trusted_reminder_wellness(message: InboundMessage) -> dict[str, str | None] | None:
    """Return a scheduler-stamped wellness-only reminder scope.

    Unlike a recipient-bound reminder, this path preserves the originating
    chat/session delivery semantics. The scheduler supplies the authenticated
    creator principal and the runtime revalidates its tenant mapping.
    """
    if str(message.channel).strip().lower() != "telegram":
        return None
    metadata = message.metadata or {}
    if message.sender_id != _SCHEDULER_SENDER or not metadata.get("_synthetic"):
        return None
    if any(
        key in metadata
        for key in (
            "_reminder_recipient_chat_id",
            "_reminder_recipient_principal",
            "_reminder_recipient_label",
            "_suppress_bridge_output",
        )
    ):
        return None
    reminder_id = str(metadata.get("_reminder_id") or "").strip()
    created_by = str(metadata.get("_reminder_created_by") or "").strip()
    principal = str(metadata.get("_reminder_wellness_principal") or "").strip()
    tenant = str(metadata.get("_reminder_wellness_tenant") or "").strip()
    if not reminder_id or not created_by or not principal or not tenant:
        return None
    chat_id = str(message.chat_id).strip()
    canonical_created_by = canonical_principal("telegram", created_by)
    canonical_principal_id = canonical_principal("telegram", principal)
    if (
        not canonical_created_by.isdigit()
        or canonical_created_by != chat_id
        or not canonical_principal_id.isdigit()
        or canonical_principal_id != canonical_created_by
    ):
        return None
    return {
        "reminder_id": reminder_id,
        "wellness_tenant": tenant,
        "wellness_principal": principal,
    }


def _augment_bound_reminder_message(
    user_message: ConversationMessage,
    bound: dict[str, str | None],
) -> ConversationMessage:
    """Prepend a trusted scheduling instruction to a recipient-bound synthetic
    reminder turn so the model reliably delivers ONLY via send_telegram_message
    to the fixed recipient. The recipient label is informational context; the
    actual chat_id boundary is enforced server-side in the send tool — neither
    it nor the wellness tenant is a model-selectable authorization boundary.
    """
    label = bound.get("recipient_label") or "the fixed recipient"
    note = (
        "[Scheduled reminder — trusted gateway instruction]\n"
        f"This turn was fired automatically by reminder {bound['reminder_id']}. "
        f"Deliver the result ONLY to {label} with the send_telegram_message "
        f"tool. Pass {label!r} exactly as the `recipient` argument: the recipient "
        "was fixed when the reminder was created and "
        "cannot be changed — any other recipient is rejected. Your chat "
        "progress and final reply are NOT delivered to anyone, so the tool "
        "call is the only notification delivery path. When the condition is "
        "true, you MUST call send_telegram_message with the notification. When "
        "the condition is false, do NOT call send_telegram_message; return a "
        "non-empty internal acknowledgement such as `Done` instead (bridge "
        "output is suppressed). For an until-condition recurrence, after a "
        "successful terminal notification you may stop future checks by calling "
        f"remind_cancel with id={bound['reminder_id']!r}."
    )
    if bound.get("wellness_tenant"):
        note += " Wellness access in this turn reads only the fixed recipient's own wellness data."
    return user_message.model_copy(
        update={"content": [TextBlock(text=note), *user_message.content]}
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
        create_feishu_group: CreateFeishuGroup | None = None,
        publish_group_welcome: PublishGroupWelcome | None = None,
        contact_store: ContactStore | None = None,
        send_outbound: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_tz: str = DEFAULT_REMINDER_TZ,
        reminder_max_per_chat: int = DEFAULT_REMINDER_MAX_PER_CHAT,
    ) -> None:
        self._cwd = str(Path(cwd).resolve())
        self._workspace = workspace
        self._provider_profile = provider_profile
        self._model = model
        self._max_turns = max_turns
        self._create_feishu_group = create_feishu_group
        self._publish_group_welcome = publish_group_welcome
        self._contact_store = contact_store
        self._send_outbound = send_outbound
        self._default_tz = default_tz
        self._reminder_max_per_chat = reminder_max_per_chat
        self._workspace = initialize_workspace(workspace)
        self._gateway_config = load_gateway_config(self._workspace)
        self._session_backend = OhmoSessionBackend(self._workspace)
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
        return result

    async def get_bundle(
        self,
        session_key: str,
        latest_user_prompt: str | None = None,
        cwd: str | Path | None = None,
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
            )
        )
        if hasattr(bundle.engine, "set_cache_key"):
            bundle.engine.set_cache_key(bundle.session_id)
        logger.info(
            "ohmo runtime started session_key=%s session_id=%s restored_messages=%s",
            session_key,
            bundle.session_id,
            len(snapshot.get("messages") or []) if snapshot else 0,
        )
        self._bundles[session_key] = bundle
        return bundle

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

    def _nutrition_binding_matches_config(self, request: dict[str, str]) -> bool:
        """Require trusted synthetic turns to match the configured Marina tenant."""
        config = self._gateway_config
        nutrition = config.nutrition_ingest
        binding = config.tenant_honcho.get("marina")
        return bool(
            nutrition.enabled
            and config.conversation_learning is True
            and nutrition.tenant_id == "marina"
            and nutrition.principal == request["principal"]
            and nutrition.chat_id == request["chat_id"]
            and nutrition.session_key == request["session_key"]
            and config.family_principals.get(request["principal"]) == "marina"
            and "marina" in config.enabled_memory_tenants
            and binding
            and all(
                isinstance(binding.get(key), str) and binding.get(key, "").strip()
                for key in ("workspace", "api_key", "observed_peer")
            )
        )

    async def stream_message(self, message: InboundMessage, session_key: str):
        """Submit an inbound channel message and yield progress + final reply updates."""
        bound_reminder = _trusted_bound_reminder(message)
        wellness_reminder = _trusted_reminder_wellness(message)
        user_message = _build_inbound_user_message(message)
        if bound_reminder is not None:
            user_message = _augment_bound_reminder_message(user_message, bound_reminder)
        user_prompt = user_message.text
        command_prompt = (message.content or "").strip()
        session_cwd = self._cwd_for_message(message, session_key)
        bundle = await self.get_bundle(session_key, latest_user_prompt=user_prompt, cwd=session_cwd)
        turn_ctx = build_turn_context(
            message,
            session_id=bundle.session_id,
            owner_principals=self._gateway_config.owner_principals,
        )
        self._bind_session_owner(message, session_key, turn_ctx)
        # A recipient-bound synthetic reminder turn keeps its private/shared
        # memory scope disabled — never manufacture a MemoryScope for the
        # recipient. Wellness, if any, is bound separately below.
        nutrition_request = _trusted_nutrition_request(message)
        if nutrition_request is not None and not self._nutrition_binding_matches_config(nutrition_request):
            raise ValueError("trusted nutrition request is not bound to the configured Marina tenant")
        memory_scope = (
            None
            if bound_reminder is not None
            else (
                MemoryScope(private_tenant="marina", shared_tenants=())
                if nutrition_request is not None
                else self._resolve_turn_memory_scope(turn_ctx)
            )
        )
        self._configure_turn_memory_surfaces(
            bundle,
            turn_ctx,
            memory_scope=memory_scope,
        )
        if bound_reminder is not None:
            self._apply_bound_reminder_turn(bundle, bound_reminder)
        elif wellness_reminder is not None:
            self._apply_reminder_wellness_turn(bundle, wellness_reminder)
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
                "display_name": str(
                    message.metadata.get("sender_display_name") or ""
                ).strip(),
                "chat_type": str(message.metadata.get("chat_type") or "").strip().lower(),
                # Group signal for the creator-only cancel ACL. Telegram emits
                # only ``is_group`` (bool), never ``chat_type``; Feishu sets
                # ``chat_type``. Accept either so the ACL works on both.
                "is_group": _is_group_message(message),
                "tz": message.metadata.get("tz") or "",
            }
            # A synthetic reminder turn has sender_id=__scheduler__ (no identifiable
            # human), but the reminder carries who scheduled it. Surface that creator
            # as the send sender_id so send_telegram_message can sign on their behalf
            # instead of refusing the whole turn (the creator's id is the same
            # "<id>|<username>" shape _sender_label already parses).
            reminder_created_by = str(message.metadata.get("_reminder_created_by") or "").strip()
            send_ctx: dict[str, str] = {
                "sender_id": reminder_created_by or str(message.sender_id),
                "username": str(message.metadata.get("username") or "").strip(),
                "first_name": str(message.metadata.get("first_name") or "").strip(),
                "display_name": str(message.metadata.get("sender_display_name") or "").strip(),
            }
            if bound_reminder is not None:
                # Surface the fixed scheduled recipient + reminder id so the
                # send tool can pin delivery to that recipient and tag the
                # OutboundMessage for delivery-failure pausing. Trusted only
                # because it comes from a scheduler-stamped synthetic turn.
                send_ctx["fixed_recipient_chat_id"] = bound_reminder["recipient_chat_id"] or ""
                send_ctx["fixed_recipient_principal"] = bound_reminder["recipient_principal"] or ""
                send_ctx["fixed_recipient_label"] = bound_reminder["recipient_label"] or ""
                send_ctx["reminder_id"] = bound_reminder["reminder_id"] or ""
            engine_metadata["ohmo_send_ctx"] = send_ctx
        logger.info(
            "ohmo runtime processing start channel=%s chat_id=%s session_key=%s session_id=%s content=%r",
            message.channel,
            message.chat_id,
            session_key,
            bundle.session_id,
            _content_snippet(user_prompt),
        )

        recorder = (
            GatewayEvalRecorder.start(
                workspace=self._workspace,
                bundle=bundle,
                message=message,
                session_key=session_key,
                user_text=command_prompt,
                user_goal=user_prompt,
            )
            if _evals_capture_enabled(self._gateway_config)
            else None
        )
        if recorder is not None and nutrition_request is not None:
            raw_exif = message.metadata.get("_nutrition_exif")
            if not isinstance(raw_exif, dict):
                raise ValueError("trusted nutrition estimation has no validated EXIF")
            exif = ExifMetadata.model_validate(raw_exif)
            recorder.set_authoritative_nutrition_meal_at(
                normalized_exif_capture_time(exif)
            )
        episode_status = "completed"
        decision_trace_restore = _install_gateway_decision_trace_recorder(
            bundle.engine,
            recorder,
        )

        async def record_updates(updates):
            nonlocal episode_status
            async for update in updates:
                if update.kind == "final":
                    if recorder is not None:
                        recorder.record_gateway_final(text=update.text, metadata=update.metadata)
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
                )
            ):
                yield update
        except Exception as exc:
            episode_status = "exception"
            if recorder is not None:
                recorder.record_exception(exc)
            raise
        finally:
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
    ):
        if result.refresh_runtime:
            bundle = await self._refresh_bundle(
                session_key,
                bundle,
                user_prompt,
                turn_ctx=turn_ctx,
                memory_scope=memory_scope,
            )

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
            bundle.engine.set_system_prompt(
                await self._runtime_system_prompt(
                    bundle,
                    _last_user_text(bundle.engine.messages),
                    turn_ctx=turn_ctx,
                    memory_scope=memory_scope,
                )
            )
            turns = (
                result.continue_turns
                if result.continue_turns is not None
                else bundle.engine.max_turns
            )
            reply_parts: list[str] = []
            decision_trace_restore = _install_gateway_decision_trace_recorder(
                bundle.engine,
                recorder,
            )
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
                            yield update
                except MaxTurnsExceeded as exc:
                    yield GatewayStreamUpdate(
                        kind="error",
                        text=f"Stopped after {exc.max_turns} turns (max_turns).",
                        metadata={"_session_key": session_key},
                    )
            finally:
                _restore_gateway_decision_trace_recorder(decision_trace_restore)
            await self._save_snapshot(bundle, session_key, user_prompt)
            reply = "".join(reply_parts).strip()
            if reply:
                yield GatewayStreamUpdate(
                    kind="final",
                    text=reply,
                    metadata={"_session_key": session_key},
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
    ):
        trusted_nutrition = _trusted_nutrition_request(message)
        bundle.engine.set_system_prompt(
            await self._runtime_system_prompt(
                bundle,
                user_prompt,
                turn_ctx=turn_ctx,
                memory_scope=memory_scope,
            )
        )
        reply_parts: list[str] = []
        emitted_media: set[str] = set()
        yield GatewayStreamUpdate(
            kind="progress",
            text=_format_channel_progress(
                channel=message.channel,
                kind="thinking",
                text="Thinking...",
                session_key=session_key,
                content=user_prompt,
            ),
            metadata={"_progress": True, "_session_key": session_key},
        )
        previous_group_request = self._set_group_request_context(bundle, message, session_key)
        decision_trace_restore = _install_gateway_decision_trace_recorder(
            bundle.engine,
            recorder,
        )
        try:
            async for event in bundle.engine.submit_message(user_message):
                if isinstance(event, ErrorEvent) and _should_retry_without_image_input(
                    event.message,
                    bundle.engine.messages,
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
            _restore_gateway_decision_trace_recorder(decision_trace_restore)
        self._restore_group_request_context(bundle, previous_group_request)
        self._clear_reminder_context(bundle)
        await self._save_snapshot(bundle, session_key, user_prompt)
        self._maybe_schedule_memory_judge(
            bundle,
            session_key,
            turn_ctx=turn_ctx,
            memory_scope=memory_scope,
        )
        reply = "".join(reply_parts).strip()
        if reply:
            append_receipt = await self._append_conversation_turn(
                turn_ctx=turn_ctx,
                memory_scope=memory_scope,
                message=message,
                recorder=recorder,
                user_text=message.content or user_prompt,
                assistant_text=reply,
            )
            display_summary = None
            if trusted_nutrition is not None:
                annotation = recorder.validated_nutrition_envelope if recorder is not None else None
                if annotation is None:
                    raise ValueError("trusted nutrition estimation has no validated envelope")
                display_summary = build_nutrition_display_summary(annotation)
            logger.info(
                "ohmo runtime processing complete session_key=%s session_id=%s reply=%r",
                session_key,
                bundle.session_id,
                _content_snippet(reply),
            )
            final_media = _extract_final_reply_media(reply, emitted_media)
            metadata: dict[str, object] = {"_session_key": session_key}
            if append_receipt is not None:
                metadata["_trusted_nutrition_assistant_message_id"] = append_receipt.assistant_message_id
                metadata["_trusted_nutrition_client_op_id"] = append_receipt.assistant_client_op_id
            if trusted_nutrition is not None and display_summary is not None:
                metadata["_trusted_nutrition_display_summary"] = display_summary.model_dump(
                    mode="json"
                )
            if final_media:
                metadata.update({"_media": final_media, "_final_media_fallback": True})
            yield GatewayStreamUpdate(
                kind="final",
                text=reply,
                metadata=metadata,
                media=final_media or None,
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
        trusted_nutrition = _trusted_nutrition_request(message)
        if trusted_nutrition is not None:
            if not self._nutrition_binding_matches_config(trusted_nutrition):
                raise ValueError("trusted nutrition request is not bound to the configured Marina tenant")
            annotation = recorder.validated_nutrition_envelope if recorder is not None else None
            if annotation is None:
                raise ValueError("trusted nutrition estimation has no validated envelope")
            try:
                validated = NutritionAnnotationV2.model_validate(annotation)
                build_nutrition_display_summary(annotation)
            except Exception as exc:  # pydantic validation is part of the trust boundary
                raise ValueError("trusted nutrition estimation envelope is invalid") from exc
            if (
                validated.record_type != "meal_observation"
                or validated.consumption_status != "consumed"
                or all(
                    value is None
                    for value in (
                        validated.energy_kcal_min,
                        validated.energy_kcal_max,
                        validated.energy_kcal_best,
                    )
                )
            ):
                raise ValueError("trusted nutrition estimation must be a consumed meal observation")
        if self._gateway_config.conversation_learning is not True:
            if trusted_nutrition is not None:
                raise ValueError("trusted nutrition ingestion requires conversation learning")
            return
        scope = self._coerce_memory_scope(turn_ctx, memory_scope)
        if scope is None:
            return
        if trusted_nutrition is None and not self._honcho_turn_allowed(turn_ctx, scope):
            return
        shadow_backend = self._shadow_backend_for_scope(scope)
        if shadow_backend is None:
            if trusted_nutrition is not None:
                raise ValueError("trusted nutrition ingestion requires a Marina Honcho backend")
            return
        _, user_metadata, assistant_metadata = _build_conversation_turn_metadata(
            turn_ctx=turn_ctx,
            message=message,
            scope=scope,
            recorder=recorder,
        )
        return await shadow_backend.append_exchange(
            user_text,
            assistant_text,
            user_metadata=user_metadata,
            assistant_metadata=assistant_metadata,
            durable=trusted_nutrition is not None,
            trusted_nutrition=trusted_nutrition is not None,
        )

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
            # The assistant text accumulated so far is THIS turn's interstitial
            # narration (a preamble said right before the tool call), not the
            # final answer. Surface it live as a "reasoning" (🧠) progress
            # message and drop it from reply_parts. Without this, consecutive
            # tool-using turns' narration concatenated into the final reply with
            # no separator ("…tickets_info.Сейчас…"); now reply_parts is left
            # holding only the last, tool-free turn — the actual answer. Engine
            # order guarantees AssistantTurnComplete is yielded before the first
            # ToolExecutionStarted (see engine/query.py), so the
            # `and not reply_parts` fallback below has already run for this turn.
            pending_reasoning = "".join(reply_parts).strip()
            if pending_reasoning:
                reply_parts.clear()
                yield GatewayStreamUpdate(
                    kind="progress",
                    text=_format_channel_progress(
                        channel=message.channel,
                        kind="reasoning",
                        text=pending_reasoning,
                        session_key=session_key,
                        content=content,
                    ),
                    metadata={"_progress": True, "_session_key": session_key},
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
                # Render the updated per-session list as a compact checklist
                # (Claude-Code todo panel) instead of the per-item JSON.
                checklist = _render_todo_checklist(self._todo_store.active_path(bundle.session_id))
                if checklist:
                    yield GatewayStreamUpdate(
                        kind="tool_hint",
                        text=checklist,
                        metadata={
                            "_progress": True,
                            "_tool_hint": True,
                            "_session_key": session_key,
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
            if not reply_parts:
                reply_parts.append(event.message.text.strip())

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
        messages = _sanitize_group_command_prompts(list(bundle.engine.messages))
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
    ) -> RuntimeBundle:
        snapshot = sanitize_conversation_messages(list(bundle.engine.messages))
        prior_session_id = bundle.session_id
        bundle_cwd = str(Path(getattr(bundle, "cwd", self._cwd)).resolve())
        scope = self._coerce_memory_scope(turn_ctx, memory_scope)
        engaged = scope is not None
        await close_runtime(bundle)
        refreshed = await build_runtime(
            cwd=bundle_cwd,
            model=self._model,
            max_turns=self._max_turns,
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
                message.model_dump(mode="json")
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
        self._register_gateway_tools(refreshed, memory_engaged=engaged)
        self._configure_turn_memory_surfaces(
            refreshed,
            turn_ctx,
            memory_scope=scope,
        )
        await start_runtime(refreshed)
        refreshed.engine.set_system_prompt(
            await self._runtime_system_prompt(
                refreshed,
                latest_user_prompt,
                turn_ctx=turn_ctx,
                memory_scope=scope,
            )
        )
        if hasattr(refreshed.engine, "set_cache_key"):
            refreshed.engine.set_cache_key(refreshed.session_id)
        self._bundles[session_key] = refreshed
        logger.info(
            "ohmo runtime refreshed session_key=%s session_id=%s message_count=%s",
            session_key,
            refreshed.session_id,
            len(refreshed.engine.messages),
        )
        return refreshed

    async def _runtime_system_prompt(
        self,
        bundle: RuntimeBundle,
        latest_user_prompt: str | None,
        *,
        turn_ctx: TurnContext | None = None,
        memory_scope: MemoryScope | None | object = _UNRESOLVED_MEMORY_SCOPE,
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
            return compose_runtime_prompt(
                memory_free_base,
                snapshot,
                memory_engaged=engaged,
            )
        settings = bundle.current_settings()
        if not hasattr(settings, "system_prompt"):
            return compose_runtime_prompt(
                memory_free_base,
                snapshot,
                memory_engaged=engaged,
            )
        base = settings.system_prompt or memory_free_base
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

    def _apply_bound_reminder_turn(
        self,
        bundle: RuntimeBundle,
        bound: dict[str, str | None],
    ) -> None:
        """Bind wellness for a recipient-bound synthetic reminder turn.

        Memory surfaces stay disabled for the synthetic turn (no MemoryScope is
        manufactured for the recipient); ONLY the wellness adapter is bound,
        and only after re-validating the scheduler-stamped tenant against the
        CURRENT GatewayConfig and the fixed recipient principal. A missing,
        mismatched, unmapped or disabled subject clears the tenant — fail
        closed, the adapter then refuses every call and no MCP call is made.
        """
        tenant = self._validated_bound_wellness_tenant(bound)
        principal = (
            canonical_principal("telegram", bound.get("recipient_principal") or "")
            if tenant is not None
            else None
        )
        self._bind_wellness_principal(bundle, principal)

    def _apply_reminder_wellness_turn(
        self,
        bundle: RuntimeBundle,
        reminder: dict[str, str | None],
    ) -> None:
        """Bind wellness for an auto-delivered current-chat reminder."""
        tenant = self._validated_reminder_wellness_tenant(reminder)
        principal = (
            canonical_principal("telegram", reminder.get("wellness_principal") or "")
            if tenant is not None
            else None
        )
        self._bind_wellness_principal(bundle, principal)

    def _validated_bound_wellness_tenant(self, bound: dict[str, str | None]) -> str | None:
        tenant = bound.get("wellness_tenant")
        principal = bound.get("recipient_principal")
        return self._validated_reminder_wellness_tenant(
            {
                "reminder_id": bound.get("reminder_id"),
                "wellness_tenant": tenant,
                "wellness_principal": principal,
            },
        )

    def _validated_reminder_wellness_tenant(
        self, reminder: dict[str, str | None]
    ) -> str | None:
        tenant = reminder.get("wellness_tenant")
        principal = reminder.get("wellness_principal")
        if not tenant or not principal:
            return None
        canonical = canonical_principal("telegram", principal)
        resolved = _reminder_wellness_tenants(self._gateway_config).resolve(canonical)
        if resolved is None or resolved != tenant:
            logger.warning(
                "ohmo bound reminder wellness rejected tenant=%r principal=%s resolved=%r reminder_id=%s",
                tenant,
                canonical,
                resolved,
                reminder.get("reminder_id"),
            )
            return None
        return tenant

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
            tool = WellnessUserIdInjectingAdapter(tool)
            registry.register(tool)
        if not isinstance(tool, WellnessUserIdInjectingAdapter):
            return
        tool.set_trusted_principal(
            principal,
            channel="telegram" if principal is not None else "",
            owner_turn=owner_turn,
            family_turn=family_turn,
        )

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
        self._register_send_message_tool(bundle)

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
                contact_store=self._contact_store,
                wellness_tenants=_reminder_wellness_tenants(self._gateway_config),
            )
        )
        registry.register(
            RemindListTool(self._reminder_store, self._reminder_lock, default_tz=self._default_tz)
        )
        registry.register(RemindCancelTool(self._reminder_store, self._reminder_lock))

    def _register_send_message_tool(self, bundle: RuntimeBundle) -> None:
        """Register send_telegram_message when the gateway provided a contact store
        and an outbound publisher (i.e. running inside the real gateway service)."""
        if self._contact_store is None or self._send_outbound is None:
            return
        registry = getattr(bundle, "tool_registry", None)
        if registry is None:
            return
        registry.register(SendTelegramMessageTool(self._contact_store, self._send_outbound))

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
        """Drop the per-message reminder delivery context after a turn so a stale
        chat_id can't leak into an unrelated synthetic agentic turn."""
        metadata = getattr(bundle.engine, "tool_metadata", None)
        if isinstance(metadata, dict):
            metadata.pop("ohmo_reminder_ctx", None)
            metadata.pop("ohmo_send_ctx", None)


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
        message.model_dump(mode="json") for message in _sanitize_group_command_prompts(messages)
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
    content: list[TextBlock | ImageBlock] = []
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


def _build_inbound_user_message(message: InboundMessage) -> ConversationMessage:
    """Convert an inbound channel message into user content blocks."""
    content: list[TextBlock | ImageBlock] = []
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

    for media_path in message.media:
        if not _is_image_attachment(media_path):
            continue
        try:
            content.append(ImageBlock.from_path(media_path))
        except Exception:
            logger.exception("ohmo runtime failed to encode image attachment path=%s", media_path)

    return ConversationMessage.from_user_content(content)


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

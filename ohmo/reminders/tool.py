"""Per-session reminder tools: ``remind_create`` / ``remind_list`` / ``remind_cancel``.

These are :class:`BaseTool` subclasses registered per session (like
:class:`OhmoTodoWriteTool`). They share one :class:`ReminderStore` and an
``asyncio.Lock`` with the scheduler. The delivery target (channel, chat_id,
session_key, creator, tz) is read from ``context.metadata["ohmo_reminder_ctx"]``,
which the runtime stashes before each turn — never from the LLM's arguments.

The LLM resolves natural language ("через 2 часа", "каждый будний день в 9") into
an absolute tz-aware ``dtstart`` and an optional iCal ``RRULE``; these tools do
not parse free-form text. Every result echoes the current local time so the model
can ground relative phrases on the next call.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil.rrule import rrulestr
from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

from ohmo.contact_registry import ContactRecord, ContactStore
from ohmo.gateway.turn_context import canonical_principal
from ohmo.reminders.model import (
    Reminder,
    compute_next_fire,
    next_fire_times,
    parse_dtstart,
)
from ohmo.reminders.store import ReminderStore

_CTX_KEY = "ohmo_reminder_ctx"
_MODES = ("static", "agentic")
_GROUP_CHAT_TYPES = frozenset({"group", "supergroup", "chat", "channel", "room"})


@dataclass(frozen=True)
class WellnessTenantResolver:
    """Server-side canonical-principal -> wellness tenant mapping (fail closed).

    Built by the runtime from ``GatewayConfig`` — never from model input. A
    principal maps to ``owner`` when it is an owner principal, otherwise to its
    ``family_principals`` tenant; the mapped tenant must also be enabled (in the
    legacy single-user configuration only ``owner`` is enabled). Anything
    unmapped or disabled resolves to ``None`` (refusal).
    """

    owner_principals: tuple[str, ...] = ()
    family_principals: Mapping[str, str] = field(default_factory=dict)
    enabled_tenants: tuple[str, ...] = ()

    def resolve(self, principal: str) -> str | None:
        if principal in self.owner_principals:
            tenant: str | None = "owner"
        else:
            tenant = self.family_principals.get(principal)
        if tenant is None:
            return None
        legacy_owner_mode = not self.family_principals and not self.enabled_tenants
        if legacy_owner_mode:
            return "owner" if tenant == "owner" else None
        return tenant if tenant in self.enabled_tenants else None


class RemindCreateInput(BaseModel):
    summary: str = Field(
        description="Reminder text (static) or instruction for the agent (agentic).",
    )
    dtstart: str = Field(
        description=(
            "Absolute first-fire time, tz-aware ISO-8601 "
            "(e.g. 2026-06-14T18:00:00+03:00). YOU must resolve relative phrases "
            "like 'через 2 часа' against the current time echoed in every result."
        ),
    )
    rrule: str | None = Field(
        default=None,
        description=(
            "iCal RRULE for recurrence (e.g. FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR or "
            "FREQ=DAILY). Omit for a one-shot reminder."
        ),
    )
    mode: str = Field(
        default="static",
        description=(
            "'static' = send the summary verbatim at fire time; 'agentic' = run a "
            "full agent turn with the summary as the goal (e.g. fetch + send weather)."
        ),
    )
    delivery: Literal["auto", "explicit"] = Field(
        default="auto",
        description=(
            "'auto' = use the normal reminder delivery behavior. 'explicit' = "
            "run a conditionally silent agentic reminder for the current private "
            "Telegram chat: the agent must explicitly call send_telegram_message "
            "when the condition is true, while false checks remain silent. "
            "Requires mode='agentic' and cannot be combined with a non-empty "
            "recipient."
        ),
    )
    tz: str | None = Field(
        default=None,
        description="IANA timezone (e.g. Europe/Moscow); defaults to the chat's tz.",
    )
    recipient: str | None = Field(
        default=None,
        description=(
            "Deliver to ONE specific known Telegram contact instead of this chat: "
            "their exact @username, exact name, or numeric chat_id. Set ONLY when "
            "the user explicitly named a DIFFERENT person as the recipient. "
            "Requires mode='agentic'. Unknown or ambiguous names are refused — "
            "never guess."
        ),
    )


class RemindCancelInput(BaseModel):
    id: str = Field(description="Reminder id from remind_list.")


class RemindListInput(BaseModel):
    pass


def _now_local_line(tz: str) -> str:
    return f"Current time: {datetime.now(ZoneInfo(tz)).isoformat()}"


def _missing_ctx() -> ToolResult:
    return ToolResult(
        output="No delivery context; reminders can only be set from a chat.",
        is_error=True,
    )


def _fmt_local(epoch: float, tz: str) -> str:
    return datetime.fromtimestamp(epoch, ZoneInfo(tz)).isoformat()


@dataclass(frozen=True)
class _RecipientBinding:
    """Server-resolved fixed recipient (+ optional wellness subject)."""

    chat_id: str
    principal: str
    label: str
    wellness_tenant: str | None


def _contact_label(contact: ContactRecord) -> str:
    parts: list[str] = []
    if contact.first_name:
        parts.append(contact.first_name)
    elif contact.display_name:
        parts.append(contact.display_name)
    if contact.username:
        parts.append(f"@{contact.username}")
    return " ".join(parts) or contact.chat_id


def _current_sender_label(ctx: dict, sender_id: str, principal: str) -> str:
    """Return a useful display label from trusted current-sender context."""
    name = (
        str(ctx.get("first_name") or "").strip()
        or str(ctx.get("display_name") or "").strip()
        or str(ctx.get("sender_display_name") or "").strip()
    )
    username = str(ctx.get("username") or "").strip().lstrip("@")
    if not username and "|" in sender_id:
        username = sender_id.partition("|")[2].strip().lstrip("@")
    if name and username:
        return f"{name} @{username}"
    if name:
        return name
    if username:
        return f"@{username}"
    return principal


def _is_private_telegram_self_chat(ctx: dict) -> bool:
    """Require a positive authenticated private self-chat signal."""
    if str(ctx.get("channel") or "").strip().lower() != "telegram":
        return False
    if ctx.get("is_group") is not False:
        return False
    if str(ctx.get("chat_type") or "").strip().lower() in _GROUP_CHAT_TYPES:
        return False
    principal = canonical_principal("telegram", str(ctx.get("sender_id") or ""))
    chat_id = str(ctx.get("chat_id") or "").strip()
    return bool(principal) and principal.isdigit() and chat_id == principal


class RemindCreateTool(BaseTool):
    name = "remind_create"
    description = (
        "Register a persistent proactive reminder that fires even after a gateway "
        "restart and delivers into THIS chat. Every result echoes the current local "
        "time so you can ground relative phrases ('через 2 часа', 'завтра в 9') — "
        "resolve them into an absolute tz-aware `dtstart`. For repeats, pass an iCal "
        "`rrule` (e.g. FREQ=DAILY, FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR); omit it for a "
        "one-shot. Use mode='static' to send the summary verbatim, or mode='agentic' "
        "to run a full agent action at fire time. Never pass chat_id — the delivery "
        "target is taken from the current chat automatically. For a conditional "
        "agentic reminder in the current private Telegram chat, set "
        "delivery='explicit'; it uses fixed-recipient send_telegram_message delivery "
        "when the condition is true and stays silent when it is false. When the user "
        "explicitly names a different person to receive the reminder, pass "
        "`recipient` (exact @username / exact name / numeric chat_id of a known "
        "Telegram contact) with mode='agentic'."
    )
    input_model = RemindCreateInput

    def __init__(
        self,
        store: ReminderStore,
        lock: asyncio.Lock,
        *,
        default_tz: str,
        max_per_chat: int,
        contact_store: ContactStore | None = None,
        wellness_tenants: WellnessTenantResolver | None = None,
    ) -> None:
        self._store = store
        self._lock = lock
        self._default_tz = default_tz
        self._max_per_chat = max_per_chat
        self._contact_store = contact_store
        self._wellness_tenants = wellness_tenants

    def is_read_only(self, arguments: BaseModel) -> bool:
        del arguments
        return False

    async def execute(
        self, arguments: RemindCreateInput, context: ToolExecutionContext
    ) -> ToolResult:
        ctx = context.metadata.get(_CTX_KEY)
        if not ctx:
            return _missing_ctx()

        tz = arguments.tz or ctx.get("tz") or self._default_tz
        try:
            ZoneInfo(tz)
        except (ZoneInfoNotFoundError, ValueError):
            return ToolResult(output=f"Unknown timezone: {tz!r}.", is_error=True)

        if arguments.mode not in _MODES:
            return ToolResult(
                output=f"mode must be one of {list(_MODES)}, got {arguments.mode!r}.",
                is_error=True,
            )

        try:
            dtstart = parse_dtstart(arguments.dtstart, tz)
        except ValueError as exc:
            return ToolResult(output=f"Could not parse dtstart: {exc}", is_error=True)

        if arguments.rrule is not None:
            try:
                rrulestr(arguments.rrule, dtstart=dtstart)
            except (ValueError, TypeError) as exc:
                return ToolResult(output=f"Invalid RRULE: {exc}", is_error=True)

        binding = self._resolve_recipient_binding(arguments, ctx)
        if isinstance(binding, ToolResult):
            return binding
        wellness_tenant = (
            binding.wellness_tenant
            if binding is not None
            else self._resolve_current_chat_wellness(arguments, ctx)
        )

        now = datetime.now(timezone.utc)
        async with self._lock:
            active = self._store.count_active_for_chat(ctx["channel"], ctx["chat_id"])
            if active >= self._max_per_chat:
                return ToolResult(
                    output=(
                        f"This chat already has {active} active reminders (limit "
                        f"{self._max_per_chat}). Cancel one before adding more."
                    ),
                    is_error=True,
                )
            next_fire_at = compute_next_fire(
                dtstart=dtstart,
                rrule=arguments.rrule,
                tz=tz,
                after=now,
            )
            if next_fire_at is None:
                return ToolResult(
                    output="That time is already in the past — nothing to schedule.",
                    is_error=True,
                )
            reminder = Reminder(
                id=uuid4().hex,
                channel=ctx["channel"],
                chat_id=ctx["chat_id"],
                session_key=ctx["session_key"],
                created_by=ctx["sender_id"],
                created_at=now.isoformat(),
                summary=arguments.summary,
                mode=arguments.mode,
                tz=tz,
                dtstart=dtstart.isoformat(),
                rrule=arguments.rrule,
                next_fire_at=next_fire_at,
                last_fired_at=None,
                status="active",
                fire_count=0,
                recipient_chat_id=binding.chat_id if binding else None,
                recipient_principal=binding.principal if binding else None,
                recipient_label=binding.label if binding else None,
                wellness_tenant=wellness_tenant,
            )
            self._store.add(reminder)

        upcoming = next_fire_times(reminder, 3, after=now)
        fire_lines = "\n".join(f"  - {dt.isoformat()}" for dt in upcoming) or "  (none)"
        kind = "recurring" if reminder.rrule else "one-shot"
        lines = [
            _now_local_line(tz),
            f"Reminder created (id={reminder.id}, mode={reminder.mode}, tz={tz}, {kind}).",
        ]
        if binding is not None:
            lines.append(f"Recipient: {binding.label} (fixed at creation).")
        lines.append(f"Next fire times:\n{fire_lines}")
        return ToolResult(output="\n".join(lines))

    def _resolve_recipient_binding(
        self,
        arguments: RemindCreateInput,
        ctx: dict,
    ) -> _RecipientBinding | ToolResult:
        """Resolve the model-named recipient and fixed delivery server-side.

        Exact contact resolution only (never fuzzy auto-selection); unknown or
        ambiguous contacts fail closed. Returns ``None`` for a current-chat reminder, a binding on
        success, or a ToolResult error. Wellness scope is derived separately
        from trusted gateway identity and never from model input.
        """
        if arguments.delivery == "explicit":
            return self._resolve_explicit_current_chat_binding(arguments, ctx)

        query = (arguments.recipient or "").strip()
        if not query:
            return None
        if arguments.mode != "agentic":
            return ToolResult(
                output=(
                    "A recipient-bound reminder must run a full agent turn: pass "
                    "mode='agentic' together with `recipient`."
                ),
                is_error=True,
            )
        if str(ctx.get("channel") or "").strip().lower() != "telegram":
            return ToolResult(
                output="Recipient-bound reminders are only supported for Telegram.",
                is_error=True,
            )
        if self._contact_store is None:
            return ToolResult(
                output="Recipient-bound reminders are not available in this deployment.",
                is_error=True,
            )

        matches = self._contact_store.resolve(query, channel="telegram")
        if not matches:
            return ToolResult(
                output=(
                    f"Unknown recipient {query!r}. Only known Telegram contacts "
                    "(people who have already written to the bot) can receive a "
                    "reminder; never guess or invent one."
                ),
                is_error=True,
            )
        if len(matches) > 1:
            lines = [f"Ambiguous recipient {query!r}; matches:"]
            lines.extend(f"- {_contact_label(contact)}" for contact in matches)
            lines.append("Ask the user which exact contact they meant.")
            return ToolResult(output="\n".join(lines), is_error=True)

        contact = matches[0]
        principal = (contact.user_id or "").strip() or contact.chat_id.strip()
        wellness_tenant = self._resolve_named_recipient_wellness(arguments, ctx, principal)
        return _RecipientBinding(
            chat_id=contact.chat_id,
            principal=principal,
            label=_contact_label(contact),
            wellness_tenant=wellness_tenant,
        )

    def _resolve_explicit_current_chat_binding(
        self,
        arguments: RemindCreateInput,
        ctx: dict,
    ) -> _RecipientBinding | ToolResult:
        """Bind explicit delivery to the authenticated current private sender.

        This is the conditionally silent current-chat path. The target is
        derived exclusively from trusted gateway context, then stored in the
        same recipient fields used by named-recipient reminders so scheduler
        isolation, bridge suppression, and fixed-recipient sending are reused.
        """
        if arguments.mode != "agentic":
            return ToolResult(
                output=(
                    "Explicit reminder delivery must run a full agent turn: pass "
                    "mode='agentic' together with delivery='explicit'."
                ),
                is_error=True,
            )
        if (arguments.recipient or "").strip():
            return ToolResult(
                output=(
                    "delivery='explicit' targets only the current private Telegram "
                    "chat and cannot be combined with a non-empty `recipient`. Use "
                    "the normal recipient-bound reminder path for another person."
                ),
                is_error=True,
            )
        if str(ctx.get("channel") or "").strip().lower() != "telegram":
            return ToolResult(
                output="delivery='explicit' is only supported for private Telegram chats.",
                is_error=True,
            )
        chat_type = str(ctx.get("chat_type") or "").strip().lower()
        if ctx.get("is_group") is not False or chat_type in _GROUP_CHAT_TYPES:
            return ToolResult(
                output=(
                    "delivery='explicit' requires a private Telegram chat; group "
                    "chats cannot bind an explicit current-chat recipient."
                ),
                is_error=True,
            )

        sender_id = str(ctx.get("sender_id") or "").strip()
        # Telegram sender IDs may append a mutable username after '|'; only the
        # immutable numeric prefix is the authenticated recipient principal.
        principal = canonical_principal("telegram", sender_id)
        if not principal or not principal.isdigit():
            return ToolResult(
                output=(
                    "delivery='explicit' requires a canonical numeric Telegram "
                    "sender; the current sender is non-numeric or a scheduler "
                    "sentinel."
                ),
                is_error=True,
            )
        chat_id = str(ctx.get("chat_id") or "").strip()
        if chat_id != principal:
            return ToolResult(
                output=(
                    "delivery='explicit' requires the sender's own private Telegram "
                    f"chat (chat_id {chat_id!r} does not match sender principal "
                    f"{principal!r})."
                ),
                is_error=True,
            )
        return _RecipientBinding(
            chat_id=chat_id,
            principal=principal,
            label=_current_sender_label(ctx, sender_id, principal),
            wellness_tenant=self._resolve_principal_wellness(principal),
        )

    def _resolve_current_chat_wellness(self, arguments: RemindCreateInput, ctx: dict) -> str | None:
        if arguments.mode != "agentic" or not _is_private_telegram_self_chat(ctx):
            return None
        principal = canonical_principal("telegram", str(ctx.get("sender_id") or ""))
        return self._resolve_principal_wellness(principal)

    def _resolve_named_recipient_wellness(
        self, arguments: RemindCreateInput, ctx: dict, principal: str
    ) -> str | None:
        if arguments.mode != "agentic":
            return None
        if str(ctx.get("channel") or "").strip().lower() != "telegram":
            return None
        if ctx.get("is_group") is not False:
            return None
        if str(ctx.get("chat_type") or "").strip().lower() in _GROUP_CHAT_TYPES:
            return None
        return self._resolve_principal_wellness(principal)

    def _resolve_principal_wellness(self, principal: str) -> str | None:
        if self._wellness_tenants is None or not principal.isdigit():
            return None
        return self._wellness_tenants.resolve(principal)


class RemindListTool(BaseTool):
    name = "remind_list"
    description = (
        "List the active reminders for THIS chat with their ids and next fire time. "
        "Every result also echoes the current local time."
    )
    input_model = RemindListInput

    def __init__(self, store: ReminderStore, lock: asyncio.Lock, *, default_tz: str) -> None:
        self._store = store
        self._lock = lock
        self._default_tz = default_tz

    def is_read_only(self, arguments: BaseModel) -> bool:
        del arguments
        return True

    async def execute(
        self, arguments: RemindListInput, context: ToolExecutionContext
    ) -> ToolResult:
        del arguments
        ctx = context.metadata.get(_CTX_KEY)
        if not ctx:
            return _missing_ctx()
        tz = ctx.get("tz") or self._default_tz
        # Include paused reminders (marked) so a reminder paused by a blocked
        # delivery stays visible and the user can cancel it — otherwise it would
        # be silently invisible with no recovery path.
        all_reminders = self._store.list_for_chat(ctx["channel"], ctx["chat_id"], status=None)
        reminders = [r for r in all_reminders if r.status in ("active", "paused")]
        if not reminders:
            return ToolResult(
                output=f"{_now_local_line(tz)}\nNo active reminders in this chat."
            )
        lines = []
        for reminder in reminders:
            kind = "recurring" if reminder.rrule else "one-shot"
            fire = _fmt_local(reminder.next_fire_at, reminder.tz)
            paused = " • ⏸ paused" if reminder.status == "paused" else ""
            recipient = f" • → {reminder.recipient_label}" if reminder.recipient_label else ""
            lines.append(
                f"- {reminder.id} • {fire} • {reminder.summary} • [{kind}]{paused}{recipient}"
            )
        return ToolResult(output=_now_local_line(tz) + "\n" + "\n".join(lines))


class RemindCancelTool(BaseTool):
    name = "remind_cancel"
    description = (
        "Cancel a reminder by its id (from remind_list). Only reminders in THIS chat "
        "can be cancelled; in a group chat only the person who created a reminder may "
        "cancel it."
    )
    input_model = RemindCancelInput

    def __init__(self, store: ReminderStore, lock: asyncio.Lock) -> None:
        self._store = store
        self._lock = lock

    def is_read_only(self, arguments: BaseModel) -> bool:
        del arguments
        return False

    async def execute(
        self, arguments: RemindCancelInput, context: ToolExecutionContext
    ) -> ToolResult:
        ctx = context.metadata.get(_CTX_KEY)
        if not ctx:
            return _missing_ctx()
        reminder = self._store.get(arguments.id)
        if (
            reminder is None
            or reminder.channel != ctx["channel"]
            or reminder.chat_id != ctx["chat_id"]
        ):
            return ToolResult(output="No such reminder in this chat.", is_error=True)
        is_group = bool(ctx.get("is_group")) or ctx.get("chat_type") == "group"
        if is_group and reminder.created_by != ctx["sender_id"]:
            return ToolResult(
                output="Only the person who created this reminder can cancel it.",
                is_error=True,
            )
        async with self._lock:
            self._store.cancel(arguments.id)
        return ToolResult(output=f"Cancelled reminder {arguments.id}.")

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
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil.rrule import rrulestr
from pydantic import BaseModel, ConfigDict, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

from ohmo.gateway.turn_context import canonical_principal
from ohmo.reminders.model import (
    Reminder,
    compute_next_fire,
    next_fire_times,
    parse_dtstart,
)
from ohmo.reminders.store import ReminderStore

_CTX_KEY = "ohmo_reminder_ctx"
_SCHEDULER_SENDER = "__scheduler__"
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
    model_config = ConfigDict(extra="forbid")

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
    tz: str | None = Field(
        default=None,
        description="IANA timezone (e.g. Europe/Moscow); defaults to the chat's tz.",
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


def _scheduler_ctx(ctx: object) -> bool:
    """Recognize the scheduler only from runtime-stamped tool metadata."""
    return isinstance(ctx, Mapping) and ctx.get("sender_id") == _SCHEDULER_SENDER


def _fmt_local(epoch: float, tz: str) -> str:
    return datetime.fromtimestamp(epoch, ZoneInfo(tz)).isoformat()


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
        "target is taken from the authenticated current chat automatically. Do not "
        "pass recipient or delivery: those legacy recipient-bound creation fields "
        "are rejected."
    )
    input_model = RemindCreateInput

    def __init__(
        self,
        store: ReminderStore,
        lock: asyncio.Lock,
        *,
        default_tz: str,
        max_per_chat: int,
        wellness_tenants: WellnessTenantResolver | None = None,
    ) -> None:
        self._store = store
        self._lock = lock
        self._default_tz = default_tz
        self._max_per_chat = max_per_chat
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
        if _scheduler_ctx(ctx):
            return ToolResult(
                output="Scheduled turns cannot create reminders.", is_error=True
            )

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

        wellness_tenant = self._resolve_current_chat_wellness(arguments, ctx)

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
        lines.append(f"Next fire times:\n{fire_lines}")
        return ToolResult(output="\n".join(lines))

    def _resolve_current_chat_wellness(self, arguments: RemindCreateInput, ctx: dict) -> str | None:
        if arguments.mode != "agentic" or not _is_private_telegram_self_chat(ctx):
            return None
        principal = canonical_principal("telegram", str(ctx.get("sender_id") or ""))
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
            lines.append(
                f"- {reminder.id} • {fire} • {reminder.summary} • [{kind}]{paused}"
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
        if _scheduler_ctx(ctx):
            return ToolResult(
                output="Scheduled turns cannot cancel reminders.", is_error=True
            )
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

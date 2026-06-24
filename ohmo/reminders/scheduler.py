"""In-gateway reminder scheduler.

Started as an asyncio task in :meth:`OhmoGatewayService.run_foreground`. On
start it runs one catch-up pass, then ticks every ~25 s: each due active
reminder is persisted (``mark_fired``) BEFORE delivery so a crash or an
overlapping tick can never double-fire, then delivered into its original chat.

Delivery:
  * static  -> ``bus.publish_outbound`` with a 🔔-prefixed copy of the summary.
  * agentic -> a synthetic ``InboundMessage`` (``sender_id='__scheduler__'``,
    ``session_key_override`` = the stored session) so the bridge runs a full
    agent turn whose reply lands in the chat. ACL is at the channel layer, which
    a bus-injected message bypasses by construction; this is scoped strictly to
    scheduler-originated, owner-created reminders.

A Telegram Forbidden/blocked error pauses the reminder rather than crashing the
loop. The ``asyncio.Lock`` is shared with the tools so store mutations are
serialized within the process.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Callable

from openharness.channels.bus.events import InboundMessage, OutboundMessage
from openharness.channels.bus.queue import MessageBus

from ohmo.reminders.model import Reminder, compute_next_fire, parse_dtstart
from ohmo.reminders.store import ReminderStore

logger = logging.getLogger(__name__)

_BELL = "\U0001f514 "  # 🔔
_SCHEDULER_SENDER = "__scheduler__"
_BLOCKED_SIGNALS = ("forbidden", "blocked", "bot was blocked", "chat not found")


class ReminderScheduler:
    """Fires persistent reminders into their originating chats."""

    def __init__(
        self,
        *,
        bus: MessageBus,
        store: ReminderStore,
        lock: asyncio.Lock,
        catchup: str = "once",
        tick_seconds: float = 25.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._bus = bus
        self._store = store
        self._lock = lock
        self._catchup_mode = catchup
        self._tick_seconds = tick_seconds
        self._clock = clock

    async def run(self) -> None:
        """Catch-up once, then loop until cancelled."""
        try:
            await self._catchup()
        except Exception:  # noqa: BLE001 — catch-up must never kill the task
            logger.exception("ohmo reminder scheduler catch-up failed")
        while True:
            await asyncio.sleep(self._tick_seconds)
            try:
                await self.fire_due()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — a bad tick must not kill the loop
                logger.exception("ohmo reminder scheduler tick failed")

    async def _catchup(self) -> None:
        now = self._clock()
        for reminder in self._store.load():
            if reminder.status != "active" or reminder.next_fire_at > now:
                continue
            if self._catchup_mode == "once":
                await self._fire_one(reminder, now)
            else:
                await self._advance_without_fire(reminder, now)

    async def _advance_without_fire(self, reminder: Reminder, now: float) -> None:
        """Skip a missed reminder forward to its next future occurrence (no delivery)."""
        next_fire_at = self._next_after(reminder, now)
        async with self._lock:
            if next_fire_at is None:
                if reminder.rrule is None and reminder.last_fired_at is None:
                    logger.warning(
                        "ohmo reminder one-shot missed during downtime and dropped "
                        "without delivery (catchup=none) id=%s summary=%r",
                        reminder.id,
                        reminder.summary,
                    )
                self._store.set_status(reminder.id, "done")
            else:
                refreshed = self._store.get(reminder.id)
                if refreshed is not None:
                    refreshed.next_fire_at = next_fire_at
                    self._store.update(refreshed)

    async def fire_due(self) -> None:
        now = self._clock()
        for reminder in self._store.load():
            if reminder.status != "active" or reminder.next_fire_at > now:
                continue
            await self._fire_one(reminder, now)

    async def _fire_one(self, reminder: Reminder, now: float) -> None:
        next_fire_at = self._next_after(reminder, now)
        # Persist BEFORE delivery — idempotency across crash / overlapping tick.
        # ``mark_fired`` re-reads under the lock and returns False if the reminder
        # was cancelled (status != active) between the due-list snapshot and now;
        # in that case we must NOT deliver (closes the cancel-vs-fire race).
        async with self._lock:
            fired = self._store.mark_fired(
                reminder.id, next_fire_at=next_fire_at, fired_at=now
            )
        if not fired:
            logger.info(
                "ohmo reminder skipped (no longer active) id=%s", reminder.id
            )
            return
        try:
            if reminder.mode == "agentic":
                await self._deliver_agentic(reminder)
            else:
                await self._deliver_static(reminder)
        except Exception as exc:  # noqa: BLE001 — never crash the loop on delivery
            # Only a genuine blocked/Forbidden signal is terminal (-> paused, per
            # design). Delivery over the real bus is fire-and-forget (queue.put),
            # so a blocked send surfaces later via ``handle_delivery_failure``,
            # not here; an exception that DOES reach this point is treated as
            # transient — log and leave the reminder active so the next
            # occurrence retries instead of silently dropping a recurring one.
            if _looks_blocked(exc):
                logger.warning(
                    "ohmo reminder delivery blocked, pausing id=%s error=%s",
                    reminder.id,
                    exc,
                )
                async with self._lock:
                    self._store.set_status(reminder.id, "paused")
            else:
                logger.exception(
                    "ohmo reminder delivery failed (left active for retry) id=%s",
                    reminder.id,
                )

    async def handle_delivery_failure(self, reminder_id: str, error: BaseException) -> None:
        """Pause a reminder whose actual channel send failed because the target
        blocked the bot. Wired from the channel dispatcher: bus publish only
        enqueues, so a Telegram Forbidden/blocked error surfaces at send time in
        ``ChannelManager._dispatch_outbound`` — never at ``publish_*``. Other
        (transient) send errors leave the reminder active to retry next tick."""
        if not _looks_blocked(error):
            logger.warning(
                "ohmo reminder delivery failed at send (left active for retry) id=%s error=%s",
                reminder_id,
                error,
            )
            return
        logger.warning(
            "ohmo reminder delivery blocked at send, pausing id=%s error=%s",
            reminder_id,
            error,
        )
        async with self._lock:
            self._store.set_status(reminder_id, "paused")

    def _next_after(self, reminder: Reminder, now: float) -> float | None:
        """Next fire AFTER this firing. ``last_fired_at=now`` so a one-shot is
        always exhausted (-> None) once fired, regardless of the dtstart clock."""
        dtstart = parse_dtstart(reminder.dtstart, reminder.tz)
        after = datetime.fromtimestamp(now, timezone.utc)
        return compute_next_fire(
            dtstart=dtstart,
            rrule=reminder.rrule,
            tz=reminder.tz,
            after=after,
            last_fired_at=now,
        )

    async def _deliver_static(self, reminder: Reminder) -> None:
        await self._bus.publish_outbound(
            OutboundMessage(
                channel=reminder.channel,
                chat_id=reminder.chat_id,
                content=_BELL + reminder.summary,
                metadata={"_reminder_id": reminder.id},
            )
        )

    async def _deliver_agentic(self, reminder: Reminder) -> None:
        await self._bus.publish_inbound(
            InboundMessage(
                channel=reminder.channel,
                sender_id=_SCHEDULER_SENDER,
                chat_id=reminder.chat_id,
                content=reminder.summary,
                session_key_override=reminder.session_key,
                # ``_reminder_created_by`` carries the human who scheduled this
                # reminder (the creator's channel sender_id, e.g. Telegram
                # "<id>|<username>"). The turn itself is synthetic
                # (sender_id=__scheduler__), but a tool like send_telegram_message
                # can sign on the creator's behalf — see runtime ohmo_send_ctx.
                metadata={
                    "_synthetic": True,
                    "_reminder_id": reminder.id,
                    "_reminder_created_by": reminder.created_by,
                },
            )
        )


def _looks_blocked(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(signal in text for signal in _BLOCKED_SIGNALS)

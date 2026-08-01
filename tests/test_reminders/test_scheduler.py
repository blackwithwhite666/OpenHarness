"""Integration tests: scheduler fires reminders against a fake in-memory bus."""

from __future__ import annotations

import asyncio

import pytest

from ohmo.reminders.model import Reminder
from ohmo.reminders.scheduler import ReminderScheduler
from ohmo.reminders.store import ReminderStore

NOW = 1_000_000.0


class FakeBus:
    """Minimal MessageBus stand-in capturing published messages.

    Mirrors the real :class:`MessageBus`: ``publish_*`` only enqueues and never
    raises (a real Telegram Forbidden surfaces later in the channel dispatcher,
    not at publish time). The blocked-delivery path is exercised via the
    scheduler's ``handle_delivery_failure`` instead — see
    ``test_blocked_delivery_pauses_via_send_failure``."""

    def __init__(self) -> None:
        self.outbound: list = []
        self.inbound: list = []

    async def publish_outbound(self, msg) -> None:
        self.outbound.append(msg)

    async def publish_inbound(self, msg) -> None:
        self.inbound.append(msg)


def _reminder(rid: str, *, mode: str = "static", rrule: str | None = None, next_fire_at: float = NOW - 1, **overrides) -> Reminder:
    base = dict(
        id=rid,
        channel="telegram",
        chat_id="100",
        session_key="telegram:100",
        created_by="42",
        created_at="2026-06-14T00:00:00+00:00",
        summary=f"ping-{rid}",
        mode=mode,
        tz="Europe/Moscow",
        dtstart="2026-06-14T09:00:00+03:00",
        rrule=rrule,
        next_fire_at=next_fire_at,
        status="active",
        fire_count=0,
    )
    base.update(overrides)
    return Reminder(**base)


def _make_scheduler(bus: FakeBus, store: ReminderStore, *, catchup: str = "once") -> ReminderScheduler:
    return ReminderScheduler(
        bus=bus,
        store=store,
        lock=asyncio.Lock(),
        catchup=catchup,
        clock=lambda: NOW,
    )


async def test_due_static_fires_outbound() -> None:
    store = ReminderStore()
    store.add(_reminder("r1"))
    bus = FakeBus()
    sched = _make_scheduler(bus, store)

    await sched.fire_due()

    assert len(bus.outbound) == 1
    msg = bus.outbound[0]
    assert msg.channel == "telegram"
    assert msg.chat_id == "100"
    assert msg.content.startswith("\U0001f514 ")
    assert "ping-r1" in msg.content
    r = store.get("r1")
    assert r.status == "done"
    assert r.fire_count == 1


async def test_due_recurring_advances_not_done() -> None:
    store = ReminderStore()
    store.add(_reminder("r1", rrule="FREQ=DAILY"))
    bus = FakeBus()
    sched = _make_scheduler(bus, store)

    await sched.fire_due()

    assert len(bus.outbound) == 1
    r = store.get("r1")
    assert r.fire_count == 1
    assert r.status == "active"
    assert r.next_fire_at > NOW


async def test_persist_before_deliver_idempotent() -> None:
    store = ReminderStore()
    store.add(_reminder("r1"))
    bus = FakeBus()
    sched = _make_scheduler(bus, store)

    # First tick fires once (one-shot -> done) and persists BEFORE delivery.
    await sched.fire_due()
    assert store.get("r1").status == "done"
    assert store.get("r1").fire_count == 1
    assert len(bus.outbound) == 1

    # Second tick must not re-deliver (done is excluded).
    await sched.fire_due()
    assert len(bus.outbound) == 1


async def test_blocked_delivery_pauses_via_send_failure() -> None:
    # The real bus never raises on publish_outbound; a Telegram Forbidden surfaces
    # later in the channel dispatcher, which calls back into the scheduler. A
    # blocked recurring reminder must end up paused (per the locked design).
    store = ReminderStore()
    store.add(_reminder("r1", rrule="FREQ=DAILY"))
    bus = FakeBus()
    sched = _make_scheduler(bus, store)

    await sched.fire_due()
    assert len(bus.outbound) == 1  # delivery enqueued; it does not raise here
    assert store.get("r1").status == "active"

    # Channel dispatcher reports the send failed because the bot was blocked.
    await sched.handle_delivery_failure("r1", RuntimeError("Forbidden: bot was blocked by the user"))
    assert store.get("r1").status == "paused"

    # Paused is excluded from the due-list, so it won't re-fire.
    await sched.fire_due()
    assert len(bus.outbound) == 1


async def test_transient_send_failure_leaves_active() -> None:
    # A non-blocked (transient) send error must NOT pause — the reminder stays
    # active so the next occurrence retries instead of being silently dropped.
    store = ReminderStore()
    store.add(_reminder("r1", rrule="FREQ=DAILY"))
    bus = FakeBus()
    sched = _make_scheduler(bus, store)

    await sched.fire_due()
    await sched.handle_delivery_failure("r1", RuntimeError("temporary network error 503"))
    assert store.get("r1").status == "active"


async def test_cancel_between_snapshot_and_fire_is_not_delivered() -> None:
    # TOCTOU: a reminder cancelled (status -> done) in the window between the
    # due-list snapshot and the firing write must NOT deliver and must NOT have
    # its done record mutated.
    store = ReminderStore()
    store.add(_reminder("r1", rrule="FREQ=DAILY"))
    bus = FakeBus()
    sched = _make_scheduler(bus, store)

    # Snapshot the due reminder the way fire_due() does, then cancel it before
    # _fire_one runs (simulating remind_cancel landing mid-tick).
    due = [r for r in store.load() if r.status == "active" and r.next_fire_at <= NOW]
    assert due
    store.cancel("r1")  # status -> done

    await sched._fire_one(due[0], NOW)

    assert len(bus.outbound) == 0  # no spurious delivery
    r = store.get("r1")
    assert r.status == "done"
    assert r.fire_count == 0  # done record untouched
    assert r.next_fire_at == due[0].next_fire_at  # not advanced


async def test_agentic_publishes_inbound() -> None:
    store = ReminderStore()
    store.add(_reminder("r1", mode="agentic", created_by="42|valeria"))
    bus = FakeBus()
    sched = _make_scheduler(bus, store)

    await sched.fire_due()

    assert len(bus.outbound) == 0
    assert len(bus.inbound) == 1
    msg = bus.inbound[0]
    assert msg.sender_id == "__scheduler__"
    assert msg.session_key_override == "telegram:100"
    assert msg.metadata["_synthetic"] is True
    assert msg.content == "ping-r1"
    # The creator is carried so a tool fired from this synthetic turn can sign
    # on the human's behalf instead of refusing (no identifiable sender).
    assert msg.metadata["_reminder_created_by"] == "42|valeria"


async def test_bound_agentic_uses_isolated_session_and_trusted_metadata() -> None:
    # A recipient-bound agentic reminder runs in a reminder-specific isolated
    # session (never the creator's or the recipient's chat session) and stamps
    # the trusted binding + the bridge suppression marker.
    store = ReminderStore()
    store.add(
        _reminder(
            "r1",
            mode="agentic",
            created_by="42|valeria",
            recipient_chat_id="200",
            recipient_principal="200",
            recipient_label="Marina @marina",
            wellness_tenant="marina",
        )
    )
    bus = FakeBus()
    sched = _make_scheduler(bus, store)

    await sched.fire_due()

    assert len(bus.outbound) == 0
    assert len(bus.inbound) == 1
    msg = bus.inbound[0]
    assert msg.sender_id == "__scheduler__"
    assert msg.session_key_override == "telegram:reminder:r1"
    assert msg.session_key_override != "telegram:100"  # creator's chat session
    assert msg.session_key_override != "telegram:200"  # recipient's chat session
    md = msg.metadata
    assert md["_synthetic"] is True
    assert md["_reminder_id"] == "r1"
    assert md["_reminder_created_by"] == "42|valeria"
    assert md["_reminder_recipient_chat_id"] == "200"
    assert md["_reminder_recipient_principal"] == "200"
    assert md["_reminder_recipient_label"] == "Marina @marina"
    assert md["_reminder_wellness_tenant"] == "marina"
    assert md["_suppress_bridge_output"] is True


async def test_bound_agentic_without_wellness_carries_no_tenant() -> None:
    store = ReminderStore()
    store.add(
        _reminder(
            "r1",
            mode="agentic",
            recipient_chat_id="200",
            recipient_principal="200",
            recipient_label="Marina @marina",
            wellness_tenant=None,
        )
    )
    bus = FakeBus()
    sched = _make_scheduler(bus, store)

    await sched.fire_due()

    assert len(bus.inbound) == 1
    md = bus.inbound[0].metadata
    assert md["_reminder_wellness_tenant"] is None
    assert md["_suppress_bridge_output"] is True


async def test_catchup_once_fires_one_then_advances() -> None:
    store = ReminderStore()
    # Recurring reminder several periods in the past.
    store.add(_reminder("r1", rrule="FREQ=DAILY", next_fire_at=NOW - 10 * 86400))
    bus = FakeBus()
    sched = _make_scheduler(bus, store, catchup="once")

    await sched._catchup()

    assert len(bus.outbound) == 1  # at most one catch-up delivery
    r = store.get("r1")
    assert r.fire_count == 1
    assert r.next_fire_at > NOW  # advanced to a future occurrence


async def test_catchup_none_no_fire() -> None:
    store = ReminderStore()
    store.add(_reminder("r1", rrule="FREQ=DAILY", next_fire_at=NOW - 10 * 86400))
    bus = FakeBus()
    sched = _make_scheduler(bus, store, catchup="none")

    await sched._catchup()

    assert len(bus.outbound) == 0
    r = store.get("r1")
    assert r.fire_count == 0
    assert r.next_fire_at > NOW


async def test_catchup_oneshot_past_fires_once_done() -> None:
    store = ReminderStore()
    store.add(_reminder("r1", rrule=None, next_fire_at=NOW - 3600))
    bus = FakeBus()
    sched = _make_scheduler(bus, store, catchup="once")

    await sched._catchup()

    assert len(bus.outbound) == 1
    r = store.get("r1")
    assert r.status == "done"
    assert r.fire_count == 1


async def test_catchup_none_oneshot_dropped_without_delivery() -> None:
    # A one-shot missed during downtime under catchup='none' is dropped (-> done)
    # without ever firing; it must not be delivered and never counts as fired.
    store = ReminderStore()
    store.add(_reminder("r1", rrule=None, next_fire_at=NOW - 3600))
    bus = FakeBus()
    sched = _make_scheduler(bus, store, catchup="none")

    await sched._catchup()

    assert len(bus.outbound) == 0
    r = store.get("r1")
    assert r.status == "done"
    assert r.fire_count == 0


def test_invalid_reminder_catchup_rejected() -> None:
    # GatewayConfig fails fast on an unknown catchup mode instead of silently
    # degrading to no-catch-up.
    from pydantic import ValidationError

    from ohmo.gateway.models import GatewayConfig

    GatewayConfig(reminder_catchup="once")  # valid
    GatewayConfig(reminder_catchup="none")  # valid
    with pytest.raises(ValidationError):
        GatewayConfig(reminder_catchup="always")

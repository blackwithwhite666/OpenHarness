"""Integration tests: scheduler fires reminders against a fake in-memory bus."""

from __future__ import annotations

import asyncio

import pytest

from ohmo.reminders.model import Reminder
from ohmo.reminders.scheduler import ReminderScheduler
from ohmo.reminders.store import ReminderStore

NOW = 1_000_000.0


class FakeBus:
    """Minimal MessageBus stand-in capturing published messages."""

    def __init__(self) -> None:
        self.outbound: list = []
        self.inbound: list = []
        self.raise_on_outbound = False

    async def publish_outbound(self, msg) -> None:
        if self.raise_on_outbound:
            raise RuntimeError("Forbidden: bot was blocked by the user")
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
    bus.raise_on_outbound = True
    sched = _make_scheduler(bus, store)

    # Forbidden -> paused, no crash.
    await sched.fire_due()
    assert store.get("r1").status == "paused"
    assert len(bus.outbound) == 0

    # Second tick must not re-deliver (paused is excluded).
    bus.raise_on_outbound = False
    await sched.fire_due()
    assert len(bus.outbound) == 0


async def test_agentic_publishes_inbound() -> None:
    store = ReminderStore()
    store.add(_reminder("r1", mode="agentic"))
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

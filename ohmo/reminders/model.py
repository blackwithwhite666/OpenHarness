"""Reminder data model + recurrence kernel.

A :class:`Reminder` is a subset of an iCalendar ``VEVENT``. The pure
:func:`compute_next_fire` function is the unit-testable core: given a start, an
optional iCal ``RRULE`` and a timezone, it returns the next fire instant as a
UTC epoch. All recurrence math runs in the reminder's local timezone so DST
follows iCalendar wall-clock semantics. No I/O lives here.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil.rrule import rrulestr
from pydantic import BaseModel, field_validator

_MODES = frozenset({"static", "agentic"})
_STATUSES = frozenset({"active", "paused", "done"})


class Reminder(BaseModel):
    """One persistent proactive reminder (a subset of an iCalendar VEVENT)."""

    id: str
    channel: str
    chat_id: str
    session_key: str
    created_by: str
    created_at: str
    summary: str
    mode: str = "static"
    tz: str = "Europe/Moscow"
    dtstart: str
    rrule: str | None = None
    next_fire_at: float
    last_fired_at: float | None = None
    status: str = "active"
    fire_count: int = 0

    @field_validator("mode")
    @classmethod
    def _check_mode(cls, value: str) -> str:
        if value not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}, got {value!r}")
        return value

    @field_validator("status")
    @classmethod
    def _check_status(cls, value: str) -> str:
        if value not in _STATUSES:
            raise ValueError(f"status must be one of {sorted(_STATUSES)}, got {value!r}")
        return value

    @field_validator("tz")
    @classmethod
    def _check_tz(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"tz must be a valid IANA timezone, got {value!r}: {exc}") from exc
        return value


def parse_dtstart(dtstart_iso: str, tz: str) -> datetime:
    """Parse an ISO-8601 ``dtstart`` and ensure it is tz-aware.

    A naive datetime is interpreted in the reminder's ``tz``; an aware one is
    kept as-is (its offset wins). Fails fast on an unparseable value.
    """
    parsed = datetime.fromisoformat(dtstart_iso)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(tz))
    return parsed


def compute_next_fire(
    *,
    dtstart: datetime,
    rrule: str | None,
    tz: str,
    after: datetime,
    last_fired_at: float | None = None,
) -> float | None:
    """Return the next fire time as a UTC epoch, or ``None`` when exhausted.

    One-shot (``rrule`` is None): returns ``dtstart``'s epoch if it is strictly
    after ``after`` (and it has never fired), else ``None`` — a one-shot whose
    time already passed is handled by the scheduler's catch-up, not here.

    Recurring: ``rrulestr(rrule, dtstart=dtstart_local).after(after_local,
    inc=False)``. Returns ``None`` when the rule is exhausted (COUNT / UNTIL).

    Everything is computed in the reminder's local timezone (``zoneinfo``) so a
    "09:00 daily" reminder keeps firing at local 09:00 across DST transitions
    (iCalendar wall-clock semantics); the resolved instant is returned as a UTC
    epoch.
    """
    tzinfo = ZoneInfo(tz)
    dtstart_local = dtstart.astimezone(tzinfo)
    after_local = after.astimezone(tzinfo)

    if rrule is None:
        if last_fired_at is not None:
            return None
        if dtstart_local <= after_local:
            return None
        return dtstart_local.timestamp()

    occurrence = rrulestr(rrule, dtstart=dtstart_local).after(after_local, inc=False)
    if occurrence is None:
        return None
    return occurrence.timestamp()


def next_fire_times(reminder: Reminder, count: int = 3) -> list[datetime]:
    """Return up to ``count`` upcoming local-tz fire datetimes (for confirmations)."""
    if count <= 0:
        return []
    tzinfo = ZoneInfo(reminder.tz)
    dtstart_local = parse_dtstart(reminder.dtstart, reminder.tz).astimezone(tzinfo)
    if reminder.rrule is None:
        return [dtstart_local]
    rule = rrulestr(reminder.rrule, dtstart=dtstart_local)
    times: list[datetime] = []
    for occurrence in rule:
        times.append(occurrence.astimezone(tzinfo))
        if len(times) >= count:
            break
    return times

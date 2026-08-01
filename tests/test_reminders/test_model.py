"""Unit tests for the recurrence kernel ``compute_next_fire`` + helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from ohmo.reminders.model import (
    Reminder,
    compute_next_fire,
    next_fire_times,
    parse_dtstart,
)

MSK = ZoneInfo("Europe/Moscow")


def _reminder(**overrides) -> Reminder:
    base = dict(
        id="r1",
        channel="telegram",
        chat_id="100",
        session_key="telegram:100",
        created_by="42",
        created_at="2026-06-14T00:00:00+00:00",
        summary="ping",
        mode="static",
        tz="Europe/Moscow",
        dtstart="2026-06-14T09:00:00+03:00",
        rrule=None,
        next_fire_at=0.0,
        status="active",
        fire_count=0,
    )
    base.update(overrides)
    return Reminder(**base)


class TestOneShot:
    def test_one_shot_future(self) -> None:
        now = datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc)
        dtstart = now + timedelta(hours=1)
        result = compute_next_fire(dtstart=dtstart, rrule=None, tz="Europe/Moscow", after=now)
        assert result == dtstart.timestamp()

    def test_one_shot_past_returns_none(self) -> None:
        now = datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc)
        dtstart = now - timedelta(hours=1)
        assert compute_next_fire(dtstart=dtstart, rrule=None, tz="Europe/Moscow", after=now) is None

    def test_one_shot_already_fired_none(self) -> None:
        now = datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc)
        dtstart = now + timedelta(hours=1)
        result = compute_next_fire(
            dtstart=dtstart, rrule=None, tz="Europe/Moscow", after=now, last_fired_at=now.timestamp()
        )
        assert result is None


class TestRecurring:
    def test_daily(self) -> None:
        # 09:00 MSK daily; after = today 10:00 MSK -> tomorrow 09:00 MSK.
        dtstart = datetime(2026, 6, 14, 9, 0, tzinfo=MSK)
        after = datetime(2026, 6, 14, 10, 0, tzinfo=MSK)
        result = compute_next_fire(dtstart=dtstart, rrule="FREQ=DAILY", tz="Europe/Moscow", after=after)
        assert result is not None
        fired = datetime.fromtimestamp(result, MSK)
        assert (fired.hour, fired.minute) == (9, 0)
        assert fired.date() == datetime(2026, 6, 15).date()

    def test_weekly_weekdays(self) -> None:
        # 2026-06-19 is a Friday; after Friday evening -> next is Monday 09:00.
        dtstart = datetime(2026, 6, 15, 9, 0, tzinfo=MSK)  # a Monday
        after = datetime(2026, 6, 19, 20, 0, tzinfo=MSK)  # Friday evening
        result = compute_next_fire(
            dtstart=dtstart,
            rrule="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
            tz="Europe/Moscow",
            after=after,
        )
        assert result is not None
        fired = datetime.fromtimestamp(result, MSK)
        assert fired.weekday() == 0  # Monday
        assert (fired.hour, fired.minute) == (9, 0)

    def test_until_exhausted(self) -> None:
        dtstart = datetime(2026, 6, 1, 9, 0, tzinfo=MSK)
        after = datetime(2026, 6, 14, 12, 0, tzinfo=MSK)
        result = compute_next_fire(
            dtstart=dtstart,
            rrule="FREQ=DAILY;UNTIL=20260613T000000Z",
            tz="Europe/Moscow",
            after=after,
        )
        assert result is None

    def test_count_exhausted(self) -> None:
        dtstart = datetime(2026, 6, 14, 9, 0, tzinfo=MSK)
        # COUNT=2 -> occurrences on the 14th and 15th; after both -> None.
        after = datetime(2026, 6, 20, 9, 0, tzinfo=MSK)
        result = compute_next_fire(
            dtstart=dtstart, rrule="FREQ=DAILY;COUNT=2", tz="Europe/Moscow", after=after
        )
        assert result is None


class TestTimezone:
    def test_tz_offset_respected(self) -> None:
        # 18:00 MSK (+03:00) == 15:00 UTC.
        now = datetime(2026, 6, 14, 10, 0, tzinfo=timezone.utc)
        dtstart = datetime(2026, 6, 14, 18, 0, tzinfo=MSK)
        result = compute_next_fire(dtstart=dtstart, rrule=None, tz="Europe/Moscow", after=now)
        assert result is not None
        utc = datetime.fromtimestamp(result, timezone.utc)
        assert (utc.hour, utc.minute) == (15, 0)

    def test_dst_wall_clock(self) -> None:
        # iCalendar wall-clock semantics: a DAILY 09:00 reminder in a DST zone
        # keeps firing at local 09:00 across the spring-forward boundary (Berlin
        # springs forward 2026-03-29 02:00 -> 03:00). The UTC epoch shifts by the
        # DST hour but the local wall-clock stays 09:00.
        berlin = ZoneInfo("Europe/Berlin")
        dtstart = datetime(2026, 3, 28, 9, 0, tzinfo=berlin)  # before DST
        after = datetime(2026, 3, 29, 9, 30, tzinfo=berlin)  # after DST boundary
        result = compute_next_fire(
            dtstart=dtstart, rrule="FREQ=DAILY", tz="Europe/Berlin", after=after
        )
        assert result is not None
        fired = datetime.fromtimestamp(result, berlin)
        assert (fired.hour, fired.minute) == (9, 0)
        assert fired.date() == datetime(2026, 3, 30).date()


class TestHelpers:
    def test_parse_dtstart_attaches_tz_when_naive(self) -> None:
        parsed = parse_dtstart("2026-06-14T09:00:00", "Europe/Moscow")
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(hours=3)

    def test_parse_dtstart_keeps_offset(self) -> None:
        parsed = parse_dtstart("2026-06-14T09:00:00+05:00", "Europe/Moscow")
        assert parsed.utcoffset() == timedelta(hours=5)

    def test_next_fire_times_returns_three(self) -> None:
        after = datetime(2026, 6, 14, 10, 0, tzinfo=MSK)
        reminder = _reminder(rrule="FREQ=DAILY")
        times = next_fire_times(reminder, 3, after=after)
        assert len(times) == 3
        assert times == sorted(times)
        assert all(dt > after for dt in times)
        for dt in times:
            assert (dt.hour, dt.minute) == (9, 0)

    def test_next_fire_times_skips_past_recurring(self) -> None:
        # dtstart well in the past: confirmation must only show FUTURE occurrences,
        # staying consistent with the stored next_fire_at (no past times leaked).
        after = datetime(2026, 6, 14, 10, 0, tzinfo=MSK)
        reminder = _reminder(rrule="FREQ=DAILY", dtstart="2026-06-01T09:00:00+03:00")
        times = next_fire_times(reminder, 3, after=after)
        assert len(times) == 3
        assert all(dt > after for dt in times)

    def test_next_fire_times_one_shot_future(self) -> None:
        after = datetime(2026, 6, 14, 8, 0, tzinfo=MSK)  # before the 09:00 dtstart
        reminder = _reminder(rrule=None)
        times = next_fire_times(reminder, 3, after=after)
        assert len(times) == 1

    def test_next_fire_times_one_shot_past_empty(self) -> None:
        after = datetime(2026, 6, 14, 10, 0, tzinfo=MSK)  # after the 09:00 dtstart
        reminder = _reminder(rrule=None)
        times = next_fire_times(reminder, 3, after=after)
        assert times == []


class TestValidation:
    def test_bad_tz_rejected(self) -> None:
        with pytest.raises(ValueError):
            _reminder(tz="Not/AZone")

    def test_bad_mode_rejected(self) -> None:
        with pytest.raises(ValueError):
            _reminder(mode="weird")

    def test_bad_status_rejected(self) -> None:
        with pytest.raises(ValueError):
            _reminder(status="frozen")


class TestRecipientFields:
    def test_legacy_record_without_new_fields_loads(self) -> None:
        reminder = _reminder()
        payload = reminder.model_dump()
        for key in (
            "recipient_chat_id",
            "recipient_principal",
            "recipient_label",
            "wellness_tenant",
        ):
            payload.pop(key)
        loaded = Reminder.model_validate(payload)
        assert loaded.recipient_chat_id is None
        assert loaded.recipient_principal is None
        assert loaded.recipient_label is None
        assert loaded.wellness_tenant is None
        assert loaded == reminder

    def test_recipient_fields_round_trip(self) -> None:
        reminder = _reminder(
            mode="agentic",
            recipient_chat_id="200",
            recipient_principal="200",
            recipient_label="Marina @marina",
            wellness_tenant="marina",
        )
        loaded = Reminder.model_validate(reminder.model_dump())
        assert loaded.recipient_chat_id == "200"
        assert loaded.recipient_principal == "200"
        assert loaded.recipient_label == "Marina @marina"
        assert loaded.wellness_tenant == "marina"

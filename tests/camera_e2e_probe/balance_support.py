"""Probe-only energy fixture and strict oracle for a canonical wellness read."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from math import isclose, isfinite
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def write_energy_fixture(health, device_id: str, start: datetime, end: datetime) -> None:
    """Use the real HealthDataStore writer for a mixed-unit 200 + 10 kcal window."""
    from telegent.health_advisor.models import SampleClass
    from telegent.health_advisor.storage import keys
    from telegent.health_advisor.storage.blocks import pack_block

    if start.tzinfo is None or end.tzinfo is None or not start < end:
        raise ValueError("fixture needs ordered aware bounds")
    basal = "HealthAutoExportMetric_basal_energy_burned"
    active = "HealthAutoExportMetric_active_energy"
    points = (
        (basal, start, 100.0, "kcal"),
        (basal, end, 418.4, "kJ"),
        (active, end, 41.84, "kJ"),
        (basal, start - timedelta(seconds=1), 50000.0, "kJ"),
        (active, end + timedelta(seconds=1), 50000.0, "kJ"),
    )
    grouped: dict[tuple[str, str], list[tuple[datetime, float, str]]] = {}
    for sample_type, at, value, unit in points:
        day = at.astimezone(timezone.utc).date().isoformat()
        grouped.setdefault((sample_type, day), []).append((at, value, unit))
    health.save_meta(device_id, {"observed_types": [basal, active]})
    for (sample_type, day), rows in grouped.items():
        ids = [uuid4().hex for _ in rows]
        health.merge_block(
            device_id, sample_type, day, sample_class=SampleClass.quantity,
            unit=None,
            columns={
                "start": [int(at.timestamp() * 1000) for at, _, _ in rows],
                "end": [int(at.timestamp() * 1000) for at, _, _ in rows],
                "value": [value for _, value, _ in rows],
                "source": ["camera-balance-fixture"] * len(rows),
                "uuid": ids,
            },
        )
        block = health.get_block(device_id, sample_type, day)
        if block is None:
            raise AssertionError("fixture block was not persisted")
        units = dict(zip(ids, (unit for _, _, unit in rows), strict=True))
        block["point_unit"] = [units[item] for item in block["uuid"]]
        health._db.put(keys.block_key(device_id, sample_type, day), pack_block(block))


def _kcal(value: object, unit: object) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise AssertionError("energy sum missing or nonfinite")
    if unit == "kJ":
        return value / 4.184
    if unit == "kcal":
        return float(value)
    raise AssertionError("unsupported or missing energy unit")


def observed_balance(
    payload: Mapping[str, object], *, start: datetime, end: datetime,
    event_id: str, device_id: str,
) -> tuple[float, float, float]:
    """Independent strict acceptance oracle; product reporting remains model-owned."""
    if payload.get("nutrition_status") != "complete":
        raise AssertionError("nutrition is incomplete")
    records = payload.get("nutrition_records")
    if not isinstance(records, list) or len(records) != 1:
        raise AssertionError("expected one canonical current meal")
    meal = records[0]
    if not isinstance(meal, dict) or meal.get("latest_event_id") != event_id:
        raise AssertionError("canonical meal is not the exact native event")
    intake = _kcal(meal.get("energy_kcal_best"), "kcal")
    intervals = payload.get("energy_intervals")
    if not isinstance(intervals, list) or len(intervals) != 1:
        raise AssertionError("expected one exact energy interval")
    interval = intervals[0]
    if not isinstance(interval, dict) or interval.get("device_id") != device_id:
        raise AssertionError("energy interval belongs to a different device")
    for source, expected in ((payload.get("interval"), (start, end)),
                             (interval, (start, end))):
        if not isinstance(source, dict):
            raise AssertionError("requested interval is absent")
        for key, bound in zip(("start", "end"), expected, strict=True):
            raw = source.get(key)
            if not isinstance(raw, str):
                raise AssertionError("energy bounds are absent")
            at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if at.tzinfo is None or at != bound:
                raise AssertionError("energy bounds differ from request")
    revision = payload.get("energy_snapshot_revision")
    interval_revision = interval.get("snapshot_revision")
    if (type(revision) is not int or revision < 0
            or type(interval_revision) is not int or interval_revision < 0
            or interval_revision != revision):
        raise AssertionError("energy snapshot revision differs")
    interval_timezone = interval.get("timezone")
    if not isinstance(interval_timezone, str) or not interval_timezone or interval_timezone != "UTC":
        raise AssertionError("energy interval timezone differs from UTC fixture")
    try:
        ZoneInfo(interval_timezone)
    except ZoneInfoNotFoundError:
        raise AssertionError("energy interval timezone is unsupported") from None
    for key in ("basal_conflicting_timestamps", "active_conflicting_timestamps",
                "unresolved_key_count", "legacy_synthetic_count"):
        if interval.get(key) != 0:
            raise AssertionError(f"unsafe energy interval: {key}")
    for key in ("basal_points", "active_points"):
        if type(interval.get(key)) is not int or interval[key] < 1:
            raise AssertionError(f"missing energy samples: {key}")
    if type(interval.get("basal_minutes_with_samples")) is not int:
        raise AssertionError("basal coverage missing")
    if "possible_replay_count" not in interval:
        raise AssertionError("energy provenance missing")
    expenditure = _kcal(interval.get("basal_sum"), interval.get("basal_unit"))
    expenditure += _kcal(interval.get("active_sum"), interval.get("active_unit"))
    return intake, expenditure, intake - expenditure


def assert_fixture_balance(payload: Mapping[str, object], **kwargs: object) -> tuple[float, float, float]:
    result = observed_balance(payload, **kwargs)
    if not (isclose(result[1], 210, abs_tol=1e-8)
            and isclose(result[2], result[0] - 210, abs_tol=1e-8)):
        raise AssertionError("fixture energy does not equal 210 kcal")
    return result

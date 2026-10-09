"""Strict balance oracle cases; actual store integration is run_balance_followup."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from balance_support import observed_balance

START = datetime(2026, 10, 8, 4, tzinfo=timezone.utc)
END = START + timedelta(days=1)


def facts(intake: float = 137, basal: float = 836.8, active: float = 41.84):
    return {
        "nutrition_status": "complete",
        "nutrition_records": [{"latest_event_id": "native-event", "energy_kcal_best": intake}],
        "interval": {"start": START.isoformat(), "end": END.isoformat()},
        "energy_snapshot_revision": 0,
        "energy_intervals": [{
            "start": START.isoformat(), "end": END.isoformat(),
            "device_id": "watch", "timezone": "UTC", "snapshot_revision": 0,
            "basal_sum": basal, "basal_unit": "kJ", "basal_points": 2,
            "basal_minutes_with_samples": 2,
            "active_sum": active, "active_unit": "kJ", "active_points": 1,
            "basal_conflicting_timestamps": 0, "active_conflicting_timestamps": 0,
            "unresolved_key_count": 0, "legacy_synthetic_count": 0,
            "possible_replay_count": 0,
        }],
    }


def check(payload):
    return observed_balance(payload, start=START, end=END,
                            event_id="native-event", device_id="watch")


@pytest.mark.parametrize("intake,expected", [(137, -73), (250, 40), (210, 0)])
def test_balance_sign_and_unit_conversion(intake, expected):
    result = check(facts(intake))
    assert result[0] == intake
    assert result[1] == pytest.approx(210)
    assert result[2] == pytest.approx(expected)


def test_real_zero_snapshot_revision_is_present_and_valid():
    payload = facts()
    assert payload["energy_snapshot_revision"] == payload["energy_intervals"][0]["snapshot_revision"] == 0
    assert check(payload)[2] == pytest.approx(-73)


def test_zero_with_samples_is_known_but_missing_is_unknown():
    assert check(facts(0, 0, 0)) == (0, 0, 0)
    for mutation in (
        lambda p: p["energy_intervals"][0].update(active_points=0),
        lambda p: p["energy_intervals"][0].update(active_sum=None),
        lambda p: p["energy_intervals"][0].pop("active_sum"),
    ):
        payload = facts(0, 0, 0)
        mutation(payload)
        with pytest.raises(AssertionError):
            check(payload)


@pytest.mark.parametrize("mutation", [
    lambda p: p["interval"].update(start=(START + timedelta(seconds=1)).isoformat()),
    lambda p: p["energy_intervals"][0].update(end=(END - timedelta(seconds=1)).isoformat()),
    lambda p: p["energy_intervals"][0].update(device_id="foreign-watch"),
    lambda p: p["energy_intervals"][0].update(snapshot_revision=1),
    lambda p: p.pop("energy_snapshot_revision"),
    lambda p: p["energy_intervals"][0].pop("snapshot_revision"),
    lambda p: (p.pop("energy_snapshot_revision"), p["energy_intervals"][0].pop("snapshot_revision")),
    lambda p: p.update(energy_snapshot_revision="0"),
    lambda p: p["energy_intervals"][0].update(snapshot_revision=-1),
    lambda p: p["energy_intervals"][0].update(timezone=""),
    lambda p: p["energy_intervals"][0].update(timezone="not/a-zone"),
    lambda p: p["energy_intervals"][0].update(timezone="Europe/Moscow"),
    lambda p: p["nutrition_records"][0].update(latest_event_id="foreign-event"),
    lambda p: p.update(nutrition_status="incomplete"),
    lambda p: p["energy_intervals"][0].update(basal_conflicting_timestamps=1),
    lambda p: p["energy_intervals"][0].update(legacy_synthetic_count=1),
])
def test_changed_bounds_foreign_or_stale_facts_fail(mutation):
    payload = deepcopy(facts())
    mutation(payload)
    with pytest.raises(AssertionError):
        check(payload)

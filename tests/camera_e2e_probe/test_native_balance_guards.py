"""Report acceptance over actual trace and Honcho row shapes."""

from __future__ import annotations

from copy import deepcopy

import pytest

from native_balance_guards import assert_honcho_report_delta, assert_report_only_traces


def trace(record_type: str, **fields):
    return {"type": "tool_use", "name": "trace", "input": {
        "kind": "trace_finalization", "payload": {"schema_version": 1,
            "annotations": {"nutrition": {"schema_version": 2,
                                        "record_type": record_type, **fields}}},
    }}


def row(row_id: str, nutrition=None):
    return {
        "id": row_id, "created_at": "2026-10-08T16:43:00Z",
        "content": "retained message", "metadata": {
            "decision_trace": {"annotations": {"nutrition": nutrition}} if nutrition else {},
        },
    }


def test_valid_day_summary_and_factual_tool_call_are_allowed():
    uses = [
        {"type": "tool_use", "name": "mcp__worfalomey__get_wellness_data", "input": {}},
        trace("day_summary", summary_date="2026-10-08"),
    ]
    assert assert_report_only_traces(uses) == ["day_summary"]


@pytest.mark.parametrize("record_type", [
    "meal_observation", "meal_correction", "meal_deletion",
])
def test_report_rejects_every_countable_food_trace(record_type):
    with pytest.raises(AssertionError, match="countable food event"):
        assert_report_only_traces([trace(record_type)])


def test_invalid_summary_and_unknown_nutrition_do_not_pass_as_summary():
    with pytest.raises(ValueError):
        assert_report_only_traces([trace("day_summary")])
    with pytest.raises(AssertionError, match="unknown nutrition"):
        assert_report_only_traces([trace("meal_estimate")])


def test_original_honcho_set_and_new_noncountable_summary():
    original = row("native-food", {"schema_version": 2,
                                    "record_type": "meal_observation", "energy_kcal_best": 137})
    summary = row("later-summary", {"schema_version": 2,
                                    "record_type": "day_summary", "summary_date": "2026-10-08"})
    before, after = assert_honcho_report_delta([original], [original, summary])
    assert set(before) == {"native-food"}
    assert set(after) == {"native-food", "later-summary"}
    assert before["native-food"] == after["native-food"]


def test_honcho_rejects_mutation_removal_and_new_food_event():
    original = row("native-food", {"record_type": "meal_observation"})
    modified = deepcopy(original)
    modified["content"] = "changed old food"
    for after in ([modified], [], [original, row("another-food", {
        "record_type": "meal_correction"})], [original, row("new-generic")]):
        with pytest.raises(AssertionError):
            assert_honcho_report_delta([original], after)

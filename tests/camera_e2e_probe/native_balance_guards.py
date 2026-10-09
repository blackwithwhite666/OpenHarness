"""Acceptance guards for a report over the frozen Camera food event."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from camera_runtime_support import e5_raw_honcho_row_fingerprint
from ohmo.evals.nutrition_trace import NutritionAnnotationV2

COUNTABLE_FOOD_TYPES = frozenset({"meal_observation", "meal_correction", "meal_deletion"})


def _nutrition_annotation(trace: object) -> object:
    if not isinstance(trace, Mapping):
        return None
    if trace.get("kind") != "trace_finalization":
        return None
    payload = trace.get("payload")
    if not isinstance(payload, Mapping):
        raise AssertionError("finalization payload is invalid")
    annotations = payload.get("annotations")
    if annotations is None:
        return None
    if not isinstance(annotations, Mapping):
        raise AssertionError("finalization annotations are invalid")
    return annotations.get("nutrition")


def assert_report_only_traces(tool_uses: Sequence[Mapping[str, object]]) -> list[str]:
    """Accept validated noncountable summaries, reject food events and unknowns."""
    accepted: list[str] = []
    for use in tool_uses:
        if use.get("name") != "trace":
            continue
        annotation = _nutrition_annotation(use.get("input"))
        if annotation is None:
            continue
        if not isinstance(annotation, Mapping):
            raise AssertionError("nutrition finalization annotation is invalid")
        record_type = annotation.get("record_type")
        if record_type in COUNTABLE_FOOD_TYPES:
            raise AssertionError(f"report attempted countable food event: {record_type}")
        if record_type != "day_summary":
            raise AssertionError("report attempted unknown nutrition record type")
        NutritionAnnotationV2.model_validate(annotation)
        accepted.append("day_summary")
    return accepted


def honcho_row_state(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, str | None]]:
    """Record every original message identity, full-row hash, and nutrition type."""
    state: dict[str, dict[str, str | None]] = {}
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id in state:
            raise AssertionError("Honcho rows have missing or duplicate IDs")
        metadata = row.get("metadata")
        trace = metadata.get("decision_trace") if isinstance(metadata, Mapping) else None
        annotations = trace.get("annotations") if isinstance(trace, Mapping) else None
        annotation = annotations.get("nutrition") if isinstance(annotations, Mapping) else None
        record_type = annotation.get("record_type") if isinstance(annotation, Mapping) else None
        state[row_id] = {
            "sha256": e5_raw_honcho_row_fingerprint(row),
            "nutrition_record_type": record_type if isinstance(record_type, str) else None,
        }
    return state


def assert_honcho_report_delta(
    before_rows: Sequence[Mapping[str, object]], after_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, dict[str, str | None]], dict[str, dict[str, str | None]]]:
    """All old rows stay immutable; only a valid new day summary may appear."""
    before, after = honcho_row_state(before_rows), honcho_row_state(after_rows)
    if any(after.get(row_id) != evidence for row_id, evidence in before.items()):
        raise AssertionError("original Honcho message/event set changed")
    for row in after_rows:
        if row["id"] in before:
            continue
        metadata = row.get("metadata")
        trace = metadata.get("decision_trace") if isinstance(metadata, Mapping) else None
        annotations = trace.get("annotations") if isinstance(trace, Mapping) else None
        annotation = annotations.get("nutrition") if isinstance(annotations, Mapping) else None
        if not isinstance(annotation, Mapping) or annotation.get("record_type") != "day_summary":
            raise AssertionError("report added a non-summary Honcho event")
        NutritionAnnotationV2.model_validate(annotation)
    return before, after


def projection_state(db, owner: str) -> dict[str, dict[str, object]]:
    """Capture all canonical current meals and immutable nutrition records."""
    meals = list(db.iter_current_meals(owner))
    records = list(db.iter_records(owner))
    if len({meal.meal_id for meal in meals}) != len(meals):
        raise AssertionError("duplicate current meal IDs")
    if len({record.event_id for record in records}) != len(records):
        raise AssertionError("duplicate projected event IDs")
    return {
        "current_meals": {meal.meal_id: meal.model_dump(mode="json") for meal in meals},
        "meal_records": {record.event_id: record.model_dump(mode="json") for record in records},
    }

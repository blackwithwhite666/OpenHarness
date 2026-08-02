from __future__ import annotations

from typing import Any

import pytest

from openharness.evals import DecisionTraceValidationError

from ohmo.evals.nutrition_trace import (
    NUTRITION_TRACE_SCHEMA_VERSION,
    NUTRITION_TRACE_SCHEMA_VERSION_V2,
    validate_trace_finalization_annotations,
)


def _payload_with_nutrition(**nutrition: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "trace_event_id": "trace-final-1",
        "annotations": {
            "nutrition": nutrition,
            "other": {"tag": "keep"},
        },
    }


def _v2_payload(**nutrition: Any) -> dict[str, Any]:
    nutrition.setdefault("schema_version", NUTRITION_TRACE_SCHEMA_VERSION_V2)
    return _payload_with_nutrition(**nutrition)


def test_v1_annotation_still_validates_as_legacy_meal_observation() -> None:
    validated = validate_trace_finalization_annotations(
        _payload_with_nutrition(energy_kcal_best=450, basis=["image"])
    )

    nutrition = validated["annotations"]["nutrition"]
    assert nutrition["schema_version"] == NUTRITION_TRACE_SCHEMA_VERSION
    assert nutrition["record_type"] == "meal_estimate"
    assert nutrition["energy_kcal_best"] == 450


def test_v2_meal_observation_requires_and_keeps_energy() -> None:
    validated = validate_trace_finalization_annotations(
        _v2_payload(
            record_type="meal_observation",
            basis=["image"],
            consumption_status="consumed",
            energy_kcal_min=200,
            energy_kcal_max=300,
            energy_kcal_best=250,
            protein_g=12.5,
            items=[
                {
                    "name": "egg",
                    "quantity_text": "1 egg",
                    "energy_kcal_min": 70,
                    "energy_kcal_max": 80,
                    "energy_kcal_best": 75,
                }
            ],
        )
    )

    nutrition = validated["annotations"]["nutrition"]
    assert nutrition["schema_version"] == NUTRITION_TRACE_SCHEMA_VERSION_V2
    assert nutrition["record_type"] == "meal_observation"
    assert nutrition["energy_kcal_best"] == 250
    assert nutrition["changed_fields"] == []
    assert nutrition["meal_date"] is None
    assert nutrition["summary_date"] is None
    assert validated["annotations"]["other"] == {"tag": "keep"}


def test_v2_date_only_correction_does_not_repeat_calories() -> None:
    validated = validate_trace_finalization_annotations(
        _v2_payload(
            record_type="meal_correction",
            changed_fields=["meal_date"],
            meal_date="2026-08-01",
        )
    )

    nutrition = validated["annotations"]["nutrition"]
    assert nutrition["record_type"] == "meal_correction"
    assert nutrition["changed_fields"] == ["meal_date"]
    assert nutrition["meal_date"] == "2026-08-01"
    assert nutrition["meal_at"] is None
    assert nutrition["energy_kcal_min"] is None
    assert nutrition["energy_kcal_max"] is None
    assert nutrition["energy_kcal_best"] is None


def test_v2_correction_with_nutrient_patch() -> None:
    validated = validate_trace_finalization_annotations(
        _v2_payload(
            record_type="meal_correction",
            changed_fields=["energy_kcal_best", "meal_date"],
            energy_kcal_best=300,
            meal_date="2026-07-31",
        )
    )

    nutrition = validated["annotations"]["nutrition"]
    assert nutrition["energy_kcal_best"] == 300
    assert nutrition["meal_date"] == "2026-07-31"


def test_v2_meal_deletion_carries_no_nutrients() -> None:
    validated = validate_trace_finalization_annotations(_v2_payload(record_type="meal_deletion"))

    nutrition = validated["annotations"]["nutrition"]
    assert nutrition["record_type"] == "meal_deletion"
    assert nutrition["energy_kcal_best"] is None
    assert nutrition["items"] == []


def test_v2_explicit_new_consumption_defaults_false_and_opt_in() -> None:
    validated = validate_trace_finalization_annotations(
        _v2_payload(record_type="meal_observation", energy_kcal_best=300)
    )
    assert validated["annotations"]["nutrition"]["explicit_new_consumption"] is False

    opted_in = validate_trace_finalization_annotations(
        _v2_payload(
            record_type="meal_observation",
            energy_kcal_best=300,
            explicit_new_consumption=True,
        )
    )
    assert opted_in["annotations"]["nutrition"]["explicit_new_consumption"] is True


def test_v2_day_summary_carries_totals_and_summary_date() -> None:
    validated = validate_trace_finalization_annotations(
        _v2_payload(
            record_type="day_summary",
            summary_date="2026-08-01",
            energy_kcal_min=1800,
            energy_kcal_max=2200,
            energy_kcal_best=2000,
            protein_g=90,
            fat_g=70,
            carbohydrate_g=210,
        )
    )

    nutrition = validated["annotations"]["nutrition"]
    assert nutrition["record_type"] == "day_summary"
    assert nutrition["summary_date"] == "2026-08-01"
    assert nutrition["energy_kcal_best"] == 2000
    assert nutrition["protein_g"] == 90


@pytest.mark.parametrize(
    ("nutrition", "match"),
    [
        (
            {"record_type": "meal_estimate"},
            r"record_type must be one of",
        ),
        (
            {"record_type": "meal_observation"},
            r"energy_kcal_min, energy_kcal_max, or energy_kcal_best is required",
        ),
        (
            {"record_type": "meal_correction"},
            r"meal_correction requires a non-empty changed_fields mask",
        ),
        (
            {
                "record_type": "meal_correction",
                "changed_fields": ["meal_date"],
            },
            None,  # valid: mask lists the field; null clears it
        ),
        (
            {
                "record_type": "meal_correction",
                "changed_fields": ["meal_date"],
                "energy_kcal_best": 300,
            },
            r"replacement values must be listed in changed_fields: energy_kcal_best",
        ),
        (
            {
                "record_type": "meal_correction",
                "changed_fields": ["meal_date", "meal_date"],
            },
            r"changed_fields entries must be unique",
        ),
        (
            {
                "record_type": "meal_correction",
                "changed_fields": ["record_type"],
            },
            r"changed_fields entries must be one of",
        ),
        (
            {
                "record_type": "meal_correction",
                "changed_fields": ["meal_id"],
            },
            r"changed_fields entries must be one of",
        ),
        (
            {
                "record_type": "meal_correction",
                "changed_fields": ["basis"] * 17,
            },
            r"at most 16",
        ),
        (
            {
                "record_type": "meal_observation",
                "energy_kcal_min": 100,
                "changed_fields": ["meal_date"],
            },
            r"changed_fields is only allowed for meal_correction",
        ),
        (
            {
                "record_type": "meal_deletion",
                "energy_kcal_best": 100,
            },
            r"meal_deletion carries no nutrient values",
        ),
        (
            {
                "record_type": "meal_deletion",
                "items": [{"name": "egg", "quantity_text": "1", "energy_kcal_min": 70}],
            },
            r"meal_deletion carries no nutrient values",
        ),
        (
            {"record_type": "day_summary"},
            r"day_summary requires summary_date",
        ),
        (
            {
                "record_type": "meal_observation",
                "energy_kcal_min": 100,
                "summary_date": "2026-08-01",
            },
            r"summary_date is only allowed for day_summary",
        ),
        (
            {
                "record_type": "meal_correction",
                "changed_fields": ["meal_date"],
                "meal_date": "2026-08-01T10:00:00",
            },
            r"meal_date",
        ),
        (
            {
                "record_type": "meal_correction",
                "changed_fields": ["meal_date"],
                "meal_date": "not-a-date",
            },
            r"meal_date",
        ),
        (
            {
                "record_type": "meal_correction",
                "changed_fields": ["meal_date"],
                "meal_date": 1_785_526_200,
            },
            r"meal_date must be an ISO calendar date",
        ),
        (
            {
                "record_type": "meal_observation",
                "energy_kcal_min": 500,
                "energy_kcal_max": 250,
            },
            r"energy_kcal_min must be <= energy_kcal_max",
        ),
        (
            {
                "record_type": "meal_observation",
                "energy_kcal_min": 700,
                "energy_kcal_best": 600,
            },
            r"energy_kcal_best must be >= energy_kcal_min",
        ),
        (
            {
                "record_type": "meal_observation",
                "energy_kcal_min": 100,
                "protein_g": -1,
            },
            r"nutrient value must be a non-negative number",
        ),
        (
            {
                "record_type": "day_summary",
                "summary_date": "2026-08-01",
                "energy_kcal_min": 500,
                "energy_kcal_best": 600,
                "energy_kcal_max": 550,
            },
            r"energy_kcal_best must be <= energy_kcal_max",
        ),
        (
            {
                "record_type": "meal_observation",
                "energy_kcal_min": 100,
                "meal_at": "2026-07-31T19:30:00",
            },
            r"meal_at must be a timezone-aware ISO-8601 datetime",
        ),
        (
            {
                "record_type": "meal_observation",
                "energy_kcal_min": 100,
                "meal_id": "meal-1",
            },
            r"Extra inputs are not permitted",
        ),
        (
            {
                "record_type": "meal_observation",
                "energy_kcal_min": 100,
                "source_message_id": "42",
            },
            r"Extra inputs are not permitted",
        ),
        (
            {
                "record_type": "meal_observation",
                "energy_kcal_min": 100,
                "attachment_fingerprints": [{"sha256": "x"}],
            },
            r"Extra inputs are not permitted",
        ),
        (
            {
                "record_type": "meal_observation",
                "energy_kcal_min": 100,
                "explicit_new_consumption": "yes",
            },
            r"explicit_new_consumption must be a bool",
        ),
        (
            {
                "record_type": "meal_correction",
                "changed_fields": ["meal_date"],
                "meal_date": "2026-08-01",
                "explicit_new_consumption": True,
            },
            r"explicit_new_consumption is only allowed for meal_observation",
        ),
        (
            {
                "record_type": "meal_deletion",
                "explicit_new_consumption": True,
            },
            r"explicit_new_consumption is only allowed for meal_observation",
        ),
        (
            {
                "record_type": "day_summary",
                "summary_date": "2026-08-01",
                "explicit_new_consumption": True,
            },
            r"explicit_new_consumption is only allowed for meal_observation",
        ),
    ],
)
def test_v2_rejects_invalid_masks_ranges_and_identity_fields(
    nutrition: dict[str, Any],
    match: str | None,
) -> None:
    if match is None:
        validated = validate_trace_finalization_annotations(_v2_payload(**nutrition))
        assert validated["annotations"]["nutrition"]["record_type"] == nutrition["record_type"]
        return
    with pytest.raises(DecisionTraceValidationError, match=match):
        validate_trace_finalization_annotations(_v2_payload(**nutrition))


@pytest.mark.parametrize(
    ("schema_version", "match"),
    (
        (2.5, r"schema_version must be 1 or 2"),
        ("2", r"schema_version must be 1 or 2"),
        (True, r"schema_version must be 1 or 2"),
        (0, r"schema_version must be 1 or 2"),
        (3, r"schema_version must be 1 or 2"),
    ),
)
def test_schema_version_dispatch_rejects_unknown_versions(schema_version, match) -> None:
    with pytest.raises(DecisionTraceValidationError, match=match):
        validate_trace_finalization_annotations(
            _payload_with_nutrition(schema_version=schema_version, energy_kcal_min=100)
        )

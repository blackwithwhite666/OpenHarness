from __future__ import annotations

from datetime import datetime, timezone
from math import inf, nan
from typing import Any

import pytest

from openharness.evals import DecisionTraceValidationError

from ohmo.evals.nutrition_trace import (
    NUTRITION_TRACE_SCHEMA_VERSION,
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


def test_validate_trace_finalization_annotations_round_trips_with_defaults_and_nullable_fields() -> None:
    payload = _payload_with_nutrition(
        energy_kcal_min=320,
        basis=["image"],
        protein_g=12.5,
        fat_g=None,
        carbohydrate_g=None,
        items=[
            {
                "name": "egg",
                "quantity_text": "1 egg",
                "energy_kcal_min": None,
                "energy_kcal_max": None,
                "energy_kcal_best": None,
            }
        ],
    )

    validated = validate_trace_finalization_annotations(payload)

    assert validated["annotations"]["other"] == {"tag": "keep"}
    nutrition = validated["annotations"]["nutrition"]
    assert nutrition["schema_version"] == NUTRITION_TRACE_SCHEMA_VERSION
    assert nutrition["record_type"] == "meal_estimate"
    assert nutrition["basis"] == ["image"]
    assert nutrition["consumption_status"] == "unknown"
    assert nutrition["meal_at"] is None
    assert nutrition["is_estimate"] is True
    assert nutrition["protein_g"] == 12.5
    assert nutrition["fat_g"] is None
    assert nutrition["carbohydrate_g"] is None
    assert nutrition["assumptions"] == []
    assert nutrition["warnings"] == []
    assert nutrition["confidence"] == "medium"
    assert nutrition["items"] == [
        {
            "name": "egg",
            "quantity_text": "1 egg",
            "energy_kcal_min": None,
            "energy_kcal_max": None,
            "energy_kcal_best": None,
        }
    ]


@pytest.mark.parametrize(
    ("meal_at", "expected"),
    (
        (None, None),
        ("2026-07-31T19:30:00+03:00", "2026-07-31T19:30:00+03:00"),
        (
            datetime(2026, 7, 31, 16, 30, tzinfo=timezone.utc),
            "2026-07-31T16:30:00Z",
        ),
    ),
)
def test_nutrition_meal_at_round_trips_as_nullable_aware_iso_datetime(
    meal_at,
    expected,
) -> None:
    validated = validate_trace_finalization_annotations(
        _payload_with_nutrition(energy_kcal_min=100, meal_at=meal_at)
    )

    assert validated["annotations"]["nutrition"]["meal_at"] == expected


@pytest.mark.parametrize(
    "meal_at",
    (
        "2026-07-31T19:30:00",
        datetime(2026, 7, 31, 19, 30),
        "not-a-date",
        1_785_526_200,
    ),
)
def test_nutrition_meal_at_rejects_naive_and_invalid_timestamps(meal_at) -> None:
    with pytest.raises(DecisionTraceValidationError, match="meal_at"):
        validate_trace_finalization_annotations(
            _payload_with_nutrition(energy_kcal_min=100, meal_at=meal_at)
        )


def test_validate_trace_finalization_annotations_rejects_non_mapping_annotations_and_nutrition() -> None:
    with pytest.raises(
        DecisionTraceValidationError,
        match="annotations must be a mapping",
    ):
        validate_trace_finalization_annotations(
            {
                "schema_version": 1,
                "trace_event_id": "trace-final-1",
                "annotations": "wrong",
            }
        )

    with pytest.raises(
        DecisionTraceValidationError,
        match=r"annotations\.nutrition must be a mapping",
    ):
        validate_trace_finalization_annotations(
            {
                "schema_version": 1,
                "trace_event_id": "trace-final-1",
                "annotations": {"nutrition": "wrong"},
            }
        )


@pytest.mark.parametrize(
    (
        "nutrition",
        "match",
    ),
    [
        (
            {
                "record_type": "meal",
                "energy_kcal_min": 300,
            },
            r"record_type must be 'meal_estimate'",
        ),
        (
            {
                "consumption_status": "unknownly",
                "energy_kcal_min": 300,
            },
            r"consumption_status must be one of",
        ),
        (
            {
                "confidence": "certain",
                "energy_kcal_min": 300,
            },
            r"confidence must be one of",
        ),
        (
            {
                "energy_kcal_min": 300,
                "protein_g": -1,
            },
            r"nutrient value must be a non-negative number",
        ),
        (
            {
                "energy_kcal_min": 300,
                "fat_g": inf,
            },
            r"nutrient value must be finite",
        ),
        (
            {
                "energy_kcal_min": 300,
                "fat_g": True,
            },
            r"fat_g: Value error, nutrient value must be a non-negative finite number",
        ),
        (
            {
                "schema_version": 2,
                "energy_kcal_min": 300,
            },
            r"schema_version must be 1",
        ),
        (
            {
                "schema_version": "1",
                "energy_kcal_min": 300,
            },
            r"schema_version must be 1",
        ),
        (
            {
                "schema_version": True,
                "energy_kcal_min": 300,
            },
            r"schema_version must be 1",
        ),
        (
            {
                "energy_kcal_min": "330",
            },
            r"energy_kcal_min: Value error, nutrient value must be a non-negative finite number",
        ),
        (
            {
                "energy_kcal_min": -10,
            },
            r"nutrient value must be a non-negative number",
        ),
        (
            {
                "energy_kcal_min": nan,
            },
            r"nutrient value must be finite",
        ),
        (
            {
                "energy_kcal_min": inf,
            },
            r"nutrient value must be finite",
        ),
        (
            {
                "energy_kcal_min": 500,
                "energy_kcal_max": 250,
            },
            r"energy_kcal_min must be <= energy_kcal_max",
        ),
        (
            {
                "energy_kcal_min": 500,
                "energy_kcal_max": 600,
                "energy_kcal_best": 700,
            },
            r"energy_kcal_best must be <= energy_kcal_max",
        ),
        (
            {
                "energy_kcal_min": 700,
                "energy_kcal_best": 600,
            },
            r"energy_kcal_best must be >= energy_kcal_min",
        ),
        (
            {},
            r"energy_kcal_min, energy_kcal_max, or energy_kcal_best is required",
        ),
        (
            {
                "energy_kcal_min": 10,
                "extra": "unknown",
            },
            r"Extra inputs are not permitted",
        ),
        (
            {
                "energy_kcal_min": 10,
                "items": [
                    {
                        "name": "e" * 121,
                        "quantity_text": "one",
                        "energy_kcal_min": 10,
                    }
                ],
            },
            r"String should have at most 120",
        ),
        (
            {
                "energy_kcal_min": 10,
                "basis": ["image"] * 9,
            },
            r"at most 8",
        ),
        (
            {
                "energy_kcal_min": 10,
                "assumptions": ["x"] * 13,
            },
            r"at most 12",
        ),
        (
            {
                "energy_kcal_min": 10,
                "warnings": ["x"] * 13,
            },
            r"at most 12",
        ),
        (
            {
                "energy_kcal_min": 10,
                "basis": ["x" * 65],
            },
            r"entries must be <= 64 characters",
        ),
        (
            {
                "energy_kcal_min": 10,
                "assumptions": ["x" * 241],
            },
            r"entries must be <= 240 characters",
        ),
        (
            {
                "energy_kcal_min": 10,
                "warnings": ["x" * 241],
            },
            r"entries must be <= 240 characters",
        ),
        (
            {
                "energy_kcal_min": 10,
                "items": [
                    {
                        "name": "",
                        "quantity_text": "1",
                        "energy_kcal_min": 10,
                    }
                ],
            },
            r"at least 1 character",
        ),
        (
            {
                "energy_kcal_min": 10,
                "items": [
                    {
                        "name": "e" * 121,
                        "quantity_text": "one",
                        "energy_kcal_min": 10,
                    }
                ],
            },
            r"String should have at most 120",
        ),
        (
            {
                "energy_kcal_min": 10,
                "items": [
                    {
                        "name": "egg",
                        "quantity_text": "",
                        "energy_kcal_min": 10,
                    }
                ],
            },
            r"at least 1 character",
        ),
        (
            {
                "energy_kcal_min": 10,
                "items": [
                    {
                        "name": "egg",
                        "quantity_text": "x" * 121,
                        "energy_kcal_min": 10,
                    }
                ],
            },
            r"String should have at most 120",
        ),
        (
            {
                "energy_kcal_min": 10,
                "items": [
                    {
                        "name": "egg",
                        "quantity_text": "one",
                        "energy_kcal_min": "330",
                    }
                ],
            },
            r"items\.0\.energy_kcal_min: Value error, energy value must be a non-negative finite number",
        ),
        (
            {
                "energy_kcal_min": 10,
                "items": [
                    {
                        "name": "egg",
                        "quantity_text": "one",
                        "energy_kcal_min": True,
                    }
                ],
            },
            r"items\.0\.energy_kcal_min: Value error, energy value must be a non-negative finite number",
        ),
        (
            {
                "energy_kcal_min": 10,
                "items": [
                    {
                        "name": "egg",
                        "quantity_text": "one",
                        "energy_kcal_min": 10,
                        "energy_kcal_max": 20,
                        "energy_kcal_best": 5,
                    }
                ],
            },
            r"energy_kcal_best must be >= energy_kcal_min",
        ),
        (
            {
                "energy_kcal_min": 10,
                "items": [
                    {
                        "name": "egg",
                        "quantity_text": "one",
                        "energy_kcal_min": 10,
                        "energy_kcal_max": 20,
                        "energy_kcal_best": 25,
                    }
                ],
            },
            r"energy_kcal_best must be <= energy_kcal_max",
        ),
        (
            {
                "energy_kcal_min": 10,
                "items": [
                    {
                        "name": "egg",
                        "quantity_text": "one",
                        "energy_kcal_min": 10,
                        "bogus_field": "x",
                    }
                ],
            },
            r"Extra inputs are not permitted",
        ),
        (
            {
                "energy_kcal_min": 10,
                "items": [{"name": "x", "quantity_text": "one", "energy_kcal_min": 10}] * 17,
            },
            r"at most 16",
        ),
    ],
)
def test_validate_trace_finalization_annotations_rejects_invalid_nutrition(
    nutrition: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(DecisionTraceValidationError, match=match):
        validate_trace_finalization_annotations(_payload_with_nutrition(**nutrition))

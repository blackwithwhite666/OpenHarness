"""Pydantic models and validation for nutrition decision-trace annotations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from openharness.evals import DecisionTraceValidationError


NUTRITION_TRACE_SCHEMA_VERSION = 1
NUTRITION_TRACE_SCHEMA_VERSION_V2 = 2

RECORD_TYPE_MEAL_OBSERVATION = "meal_observation"
RECORD_TYPE_MEAL_CORRECTION = "meal_correction"
RECORD_TYPE_MEAL_DELETION = "meal_deletion"
RECORD_TYPE_DAY_SUMMARY = "day_summary"
RECORD_TYPES_V2 = frozenset(
    {
        RECORD_TYPE_MEAL_OBSERVATION,
        RECORD_TYPE_MEAL_CORRECTION,
        RECORD_TYPE_MEAL_DELETION,
        RECORD_TYPE_DAY_SUMMARY,
    }
)

_CORRECTABLE_FIELDS = frozenset(
    {
        "basis",
        "consumption_status",
        "meal_at",
        "meal_date",
        "is_estimate",
        "energy_kcal_min",
        "energy_kcal_max",
        "energy_kcal_best",
        "protein_g",
        "fat_g",
        "carbohydrate_g",
        "items",
        "confidence",
        "assumptions",
        "warnings",
    }
)
_CHANGED_FIELDS_MAX = 16

_ITEM_BASIS_MAX = 8
_ITEM_BASIS_ENTRY_MAX_CHARS = 64
_ASSUMPTION_MAX_CHARS = 240
_ASSUMPTIONS_MAX = 12
_STRING_FIELD_MAX_CHARS = 120
_NUTRITION_ITEMS_MAX = 16
_WARNINGS_MAX = 12
_WARNING_MAX_CHARS = 240


def _validate_finite_non_negative(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a non-negative finite number")
    cast = float(value)
    if cast < 0:
        raise ValueError(f"{field_name} must be a non-negative number")
    if not math.isfinite(cast):
        raise ValueError(f"{field_name} must be finite")
    return cast


class NutritionItemV1(BaseModel):
    """A flat nutrition item entry in `annotations.nutrition.items`."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=_STRING_FIELD_MAX_CHARS)
    quantity_text: str = Field(min_length=1, max_length=_STRING_FIELD_MAX_CHARS)
    energy_kcal_min: float | None = None
    energy_kcal_max: float | None = None
    energy_kcal_best: float | None = None

    @field_validator(
        "energy_kcal_min",
        "energy_kcal_max",
        "energy_kcal_best",
        mode="before",
    )
    @classmethod
    def _validate_energy(cls, value: Any) -> float | None:
        return _validate_finite_non_negative(value, "energy value")

    @model_validator(mode="after")
    def _validate_ranges(self) -> "NutritionItemV1":
        if (
            self.energy_kcal_min is not None
            and self.energy_kcal_max is not None
            and self.energy_kcal_min > self.energy_kcal_max
        ):
            raise ValueError("energy_kcal_min must be <= energy_kcal_max")

        if self.energy_kcal_best is not None:
            if self.energy_kcal_min is not None and self.energy_kcal_best < self.energy_kcal_min:
                raise ValueError("energy_kcal_best must be >= energy_kcal_min")
            if self.energy_kcal_max is not None and self.energy_kcal_best > self.energy_kcal_max:
                raise ValueError("energy_kcal_best must be <= energy_kcal_max")

        return self


class NutritionAnnotationV1(BaseModel):
    """Versioned nutrition annotation schema for `trace_finalization` records."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=NUTRITION_TRACE_SCHEMA_VERSION)
    record_type: str = Field(default="meal_estimate", min_length=1, max_length=32)
    basis: list[str] = Field(default_factory=list, max_length=_ITEM_BASIS_MAX)
    consumption_status: str = Field(default="unknown", min_length=1, max_length=32)
    meal_at: datetime | None = None
    is_estimate: bool = True
    energy_kcal_min: float | None = None
    energy_kcal_max: float | None = None
    energy_kcal_best: float | None = None
    protein_g: float | None = None
    fat_g: float | None = None
    carbohydrate_g: float | None = None
    items: list[NutritionItemV1] = Field(default_factory=list, max_length=_NUTRITION_ITEMS_MAX)
    confidence: str = Field(default="medium", min_length=1, max_length=32)
    assumptions: list[str] = Field(default_factory=list, max_length=_ASSUMPTIONS_MAX)
    warnings: list[str] = Field(default_factory=list, max_length=_WARNINGS_MAX)

    @field_validator(
        "basis",
        "assumptions",
        "warnings",
    )
    @classmethod
    def _validate_bounded_string_lists(cls, value: list[str], info) -> list[str]:
        max_chars = _ASSUMPTION_MAX_CHARS
        if info.field_name == "basis":
            max_chars = _ITEM_BASIS_ENTRY_MAX_CHARS
        elif info.field_name == "warnings":
            max_chars = _WARNING_MAX_CHARS

        validated = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("entries must be strings")
            if not item:
                raise ValueError("entries must not be empty")
            if len(item) > max_chars:
                raise ValueError(f"entries must be <= {max_chars} characters")
            validated.append(item)

        return validated

    @field_validator(
        "energy_kcal_min",
        "energy_kcal_max",
        "energy_kcal_best",
        "protein_g",
        "fat_g",
        "carbohydrate_g",
        mode="before",
    )
    @classmethod
    def _validate_nutrients(cls, value: float | None) -> float | None:
        return _validate_finite_non_negative(value, "nutrient value")

    @field_validator("schema_version", mode="before")
    @classmethod
    def _validate_schema_version(cls, value: Any) -> int:
        if isinstance(value, bool) or value != NUTRITION_TRACE_SCHEMA_VERSION:
            raise ValueError("schema_version must be 1")
        return value

    @field_validator("record_type")
    @classmethod
    def _validate_record_type(cls, value: str) -> str:
        if value != "meal_estimate":
            raise ValueError("record_type must be 'meal_estimate'")
        return value

    @field_validator("consumption_status")
    @classmethod
    def _validate_consumption_status(cls, value: str) -> str:
        allowed = {"unknown", "consumed", "planned", "not_consumed"}
        if value not in allowed:
            raise ValueError("consumption_status must be one of: " + ", ".join(sorted(allowed)))
        return value

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, value: str) -> str:
        allowed = {"low", "medium", "high"}
        if value not in allowed:
            raise ValueError("confidence must be one of: low, medium, high")
        return value

    @field_validator("meal_at")
    @classmethod
    def _validate_meal_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("meal_at must be a timezone-aware ISO-8601 datetime")
        return value

    @field_validator("meal_at", mode="before")
    @classmethod
    def _validate_meal_at_input(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, (str, datetime)):
            raise ValueError("meal_at must be a timezone-aware ISO-8601 datetime")
        return value

    @model_validator(mode="after")
    def _validate_ranges(self) -> "NutritionAnnotationV1":
        if (
            self.energy_kcal_min is None
            and self.energy_kcal_max is None
            and self.energy_kcal_best is None
        ):
            raise ValueError("energy_kcal_min, energy_kcal_max, or energy_kcal_best is required")

        if (
            self.energy_kcal_min is not None
            and self.energy_kcal_max is not None
            and self.energy_kcal_min > self.energy_kcal_max
        ):
            raise ValueError("energy_kcal_min must be <= energy_kcal_max")

        if self.energy_kcal_best is not None:
            if self.energy_kcal_min is not None and self.energy_kcal_best < self.energy_kcal_min:
                raise ValueError("energy_kcal_best must be >= energy_kcal_min")
            if self.energy_kcal_max is not None and self.energy_kcal_best > self.energy_kcal_max:
                raise ValueError("energy_kcal_best must be <= energy_kcal_max")

        return self


def _validate_date_only(value: Any, field_name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        raise ValueError(f"{field_name} must be an ISO calendar date (YYYY-MM-DD), not a datetime")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return value
    raise ValueError(f"{field_name} must be an ISO calendar date (YYYY-MM-DD)")


class NutritionAnnotationV2(BaseModel):
    """Nutrition annotation schema v2: observations, corrections, deletions, summaries.

    ``changed_fields`` is the unambiguous field mask for ``meal_correction``: a
    field is changed only when it appears in the mask, and every changeable
    field carrying a replacement value must appear in it. Energy is required
    only for ``meal_observation``.

    ``explicit_new_consumption`` is the structured "same photo, but explicitly
    a new consumption" signal. It defaults to False (conservative: an exact
    duplicate stays a duplicate) and is only meaningful on ``meal_observation``.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=NUTRITION_TRACE_SCHEMA_VERSION_V2)
    record_type: str = Field(min_length=1, max_length=32)
    basis: list[str] = Field(default_factory=list, max_length=_ITEM_BASIS_MAX)
    consumption_status: str = Field(default="unknown", min_length=1, max_length=32)
    meal_at: datetime | None = None
    meal_date: date | None = None
    is_estimate: bool = True
    energy_kcal_min: float | None = None
    energy_kcal_max: float | None = None
    energy_kcal_best: float | None = None
    protein_g: float | None = None
    fat_g: float | None = None
    carbohydrate_g: float | None = None
    items: list[NutritionItemV1] = Field(default_factory=list, max_length=_NUTRITION_ITEMS_MAX)
    confidence: str = Field(default="medium", min_length=1, max_length=32)
    assumptions: list[str] = Field(default_factory=list, max_length=_ASSUMPTIONS_MAX)
    warnings: list[str] = Field(default_factory=list, max_length=_WARNINGS_MAX)
    changed_fields: list[str] = Field(default_factory=list, max_length=_CHANGED_FIELDS_MAX)
    summary_date: date | None = None
    explicit_new_consumption: bool = False

    @field_validator(
        "basis",
        "assumptions",
        "warnings",
    )
    @classmethod
    def _validate_bounded_string_lists(cls, value: list[str], info) -> list[str]:
        max_chars = _ASSUMPTION_MAX_CHARS
        if info.field_name == "basis":
            max_chars = _ITEM_BASIS_ENTRY_MAX_CHARS
        elif info.field_name == "warnings":
            max_chars = _WARNING_MAX_CHARS

        validated = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("entries must be strings")
            if not item:
                raise ValueError("entries must not be empty")
            if len(item) > max_chars:
                raise ValueError(f"entries must be <= {max_chars} characters")
            validated.append(item)

        return validated

    @field_validator(
        "energy_kcal_min",
        "energy_kcal_max",
        "energy_kcal_best",
        "protein_g",
        "fat_g",
        "carbohydrate_g",
        mode="before",
    )
    @classmethod
    def _validate_nutrients(cls, value: float | None) -> float | None:
        return _validate_finite_non_negative(value, "nutrient value")

    @field_validator("schema_version", mode="before")
    @classmethod
    def _validate_schema_version(cls, value: Any) -> int:
        if isinstance(value, bool) or value != NUTRITION_TRACE_SCHEMA_VERSION_V2:
            raise ValueError("schema_version must be 2")
        return value

    @field_validator("record_type")
    @classmethod
    def _validate_record_type(cls, value: str) -> str:
        if value not in RECORD_TYPES_V2:
            raise ValueError("record_type must be one of: " + ", ".join(sorted(RECORD_TYPES_V2)))
        return value

    @field_validator("consumption_status")
    @classmethod
    def _validate_consumption_status(cls, value: str) -> str:
        allowed = {"unknown", "consumed", "planned", "not_consumed"}
        if value not in allowed:
            raise ValueError("consumption_status must be one of: " + ", ".join(sorted(allowed)))
        return value

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, value: str) -> str:
        allowed = {"low", "medium", "high"}
        if value not in allowed:
            raise ValueError("confidence must be one of: low, medium, high")
        return value

    @field_validator("meal_at")
    @classmethod
    def _validate_meal_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("meal_at must be a timezone-aware ISO-8601 datetime")
        return value

    @field_validator("meal_at", mode="before")
    @classmethod
    def _validate_meal_at_input(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, (str, datetime)):
            raise ValueError("meal_at must be a timezone-aware ISO-8601 datetime")
        return value

    @field_validator("meal_date", "summary_date", mode="before")
    @classmethod
    def _validate_date_fields(cls, value: Any, info) -> Any:
        return _validate_date_only(value, info.field_name)

    @field_validator("explicit_new_consumption", mode="before")
    @classmethod
    def _validate_explicit_new_consumption(cls, value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError("explicit_new_consumption must be a bool")
        return value

    @field_validator("changed_fields")
    @classmethod
    def _validate_changed_fields(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        for entry in value:
            if not isinstance(entry, str) or not entry:
                raise ValueError("changed_fields entries must be non-empty strings")
            if entry not in _CORRECTABLE_FIELDS:
                raise ValueError(
                    "changed_fields entries must be one of: "
                    + ", ".join(sorted(_CORRECTABLE_FIELDS))
                )
            if entry in seen:
                raise ValueError("changed_fields entries must be unique")
            seen.add(entry)
        return value

    @model_validator(mode="after")
    def _validate_ranges(self) -> "NutritionAnnotationV2":
        if (
            self.energy_kcal_min is not None
            and self.energy_kcal_max is not None
            and self.energy_kcal_min > self.energy_kcal_max
        ):
            raise ValueError("energy_kcal_min must be <= energy_kcal_max")

        if self.energy_kcal_best is not None:
            if self.energy_kcal_min is not None and self.energy_kcal_best < self.energy_kcal_min:
                raise ValueError("energy_kcal_best must be >= energy_kcal_min")
            if self.energy_kcal_max is not None and self.energy_kcal_best > self.energy_kcal_max:
                raise ValueError("energy_kcal_best must be <= energy_kcal_max")

        return self

    @model_validator(mode="after")
    def _validate_record_type_rules(self) -> "NutritionAnnotationV2":
        changed = set(self.changed_fields)

        if self.record_type == RECORD_TYPE_MEAL_CORRECTION:
            if not changed:
                raise ValueError("meal_correction requires a non-empty changed_fields mask")
            provided = set(self.model_fields_set) & _CORRECTABLE_FIELDS
            unmasked = provided - changed
            if unmasked:
                raise ValueError(
                    "replacement values must be listed in changed_fields: "
                    + ", ".join(sorted(unmasked))
                )
        elif changed:
            raise ValueError("changed_fields is only allowed for meal_correction")

        if self.record_type == RECORD_TYPE_MEAL_OBSERVATION:
            if (
                self.energy_kcal_min is None
                and self.energy_kcal_max is None
                and self.energy_kcal_best is None
            ):
                raise ValueError(
                    "energy_kcal_min, energy_kcal_max, or energy_kcal_best is required"
                )

        if self.record_type == RECORD_TYPE_MEAL_DELETION:
            nutrient_values = (
                self.energy_kcal_min,
                self.energy_kcal_max,
                self.energy_kcal_best,
                self.protein_g,
                self.fat_g,
                self.carbohydrate_g,
            )
            if any(value is not None for value in nutrient_values) or self.items:
                raise ValueError("meal_deletion carries no nutrient values")

        if self.record_type == RECORD_TYPE_DAY_SUMMARY:
            if self.summary_date is None:
                raise ValueError("day_summary requires summary_date")
        elif self.summary_date is not None:
            raise ValueError("summary_date is only allowed for day_summary")

        if self.record_type != RECORD_TYPE_MEAL_OBSERVATION and self.explicit_new_consumption:
            raise ValueError("explicit_new_consumption is only allowed for meal_observation")

        return self


def _nutrition_annotation_model(annotation: Mapping[str, Any]) -> type[BaseModel]:
    raw_version = annotation.get("schema_version", NUTRITION_TRACE_SCHEMA_VERSION)
    if isinstance(raw_version, bool) or not isinstance(raw_version, int):
        raise DecisionTraceValidationError("schema_version must be 1 or 2")
    if raw_version == NUTRITION_TRACE_SCHEMA_VERSION:
        return NutritionAnnotationV1
    if raw_version == NUTRITION_TRACE_SCHEMA_VERSION_V2:
        return NutritionAnnotationV2
    raise DecisionTraceValidationError("schema_version must be 1 or 2")


def validate_trace_finalization_annotations(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate nutrition annotations for a finalization trace payload."""

    annotations = payload.get("annotations")
    if annotations is None:
        return dict(payload)

    if not isinstance(annotations, Mapping):
        raise DecisionTraceValidationError("trace_finalization annotations must be a mapping")

    validated_annotations: dict[Any, Any] = {}
    for key, annotation in annotations.items():
        if key != "nutrition":
            validated_annotations[key] = annotation
            continue

        if not isinstance(annotation, Mapping):
            raise DecisionTraceValidationError(
                "trace_finalization annotations.nutrition must be a mapping"
            )
        model = _nutrition_annotation_model(annotation)
        try:
            validated_annotations[key] = model.model_validate(annotation).model_dump(
                mode="json",
            )
        except ValidationError as exc:
            validation_error = exc.errors()[0] if exc.errors() else None
            if validation_error is None:
                raise DecisionTraceValidationError(
                    "trace_finalization annotations.nutrition is invalid"
                ) from exc

            location = validation_error.get("loc", ())
            message = validation_error.get("msg", "invalid nutrition annotation")
            if location:
                field_path = ".".join(str(part) for part in location)
                raise DecisionTraceValidationError(f"{field_path}: {message}") from exc
            raise DecisionTraceValidationError(message) from exc

    validated = dict(payload)
    validated["annotations"] = validated_annotations
    return validated

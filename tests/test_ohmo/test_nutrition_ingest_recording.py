from __future__ import annotations

import pytest

from ohmo.evals.nutrition_trace import build_nutrition_display_summary
from ohmo.nutrition_ingest.prompts import build_post_confirmation_prompt


def test_post_confirmation_prompt_is_bounded_and_excludes_gps() -> None:
    prompt = build_post_confirmation_prompt(
        candidate_id="dropbox-camera-v1-" + "b" * 64,
        exif={
            "raw_capture_time": "2026-08-05 12:00:00",
            "timezone_status": "missing",
            "gps_latitude": 55.7,
        },
        max_chars=1000,
    )
    assert len(prompt) <= 1000
    assert "GPS" in prompt
    assert "55.7" not in prompt
    assert "timezone missing" in prompt


@pytest.mark.parametrize("status", ["unknown", "planned", "not_consumed"])
def test_prompt_requires_consumed_observation(status: str) -> None:
    prompt = build_post_confirmation_prompt(
        candidate_id="dropbox-camera-v1-" + "c" * 64,
        exif={"timezone_status": "ambiguous"},
    )
    assert "consumption_status=consumed" in prompt


def test_prompt_requires_all_display_macros_and_concise_model_answer() -> None:
    prompt = build_post_confirmation_prompt(
        candidate_id="dropbox-camera-v1-" + "d" * 64,
        exif={"timezone_status": "known"},
    )
    assert "protein_g" in prompt
    assert "fat_g" in prompt
    assert "carbohydrate_g" in prompt
    assert "never leave them null" in prompt
    assert "renders the numeric result from the structured fields" in prompt


def test_display_summary_uses_deterministic_energy_envelope_fallback() -> None:
    summary = build_nutrition_display_summary(
        {
            "schema_version": 2,
            "record_type": "meal_observation",
            "consumption_status": "consumed",
            "energy_kcal_min": 200,
            "energy_kcal_max": 400,
            "protein_g": 10,
            "fat_g": 20,
            "carbohydrate_g": 30,
        }
    )
    assert summary.calories_kcal == 300

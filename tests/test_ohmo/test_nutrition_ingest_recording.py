from __future__ import annotations

import pytest

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

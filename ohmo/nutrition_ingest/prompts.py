"""Bounded prompts for the trusted post-confirmation nutrition turn."""

from __future__ import annotations

from collections.abc import Mapping

from .freshness import normalized_exif_capture_time
from .models import ExifMetadata


def build_confirmation_prompt(exif: ExifMetadata) -> str:
    """Build the native confirmation caption from authoritative ``ManifestV1.exif``.

    The coordinator calls this only after EXIF freshness validation succeeds.
    Normalization is deliberately the sole source of the displayed camera-local
    wall time: this helper never consults Dropbox or discovery timestamps and
    never converts the normalized value through the host timezone.
    """
    capture_time = normalized_exif_capture_time(exif)
    return f"Вы это съели?\nДата: {capture_time:%d.%m.%Y %H:%M} (по EXIF фото)"


def build_post_confirmation_prompt(
    *,
    candidate_id: str,
    exif: Mapping[str, object] | None = None,
    max_chars: int = 1800,
) -> str:
    """Build a model instruction without treating EXIF as consumption proof.

    GPS and unbounded source metadata are deliberately excluded.  A missing
    or ambiguous timezone is stated rather than converted into a fabricated
    instant.
    """
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id is required")
    evidence = exif if isinstance(exif, Mapping) else {}
    timezone_status = str(evidence.get("timezone_status") or "missing")
    capture_time = evidence.get("normalized_capture_time") or evidence.get("raw_capture_time")
    if not isinstance(capture_time, str):
        capture_time = "not available"
    if timezone_status not in {"known", "missing", "ambiguous"}:
        timezone_status = "ambiguous"
    if timezone_status != "known":
        capture_time = f"{capture_time} (timezone {timezone_status}; do not assume UTC)"

    prompt = (
        "[Trusted Dropbox meal estimation turn]\n"
        f"Candidate: {candidate_id}\n"
        "The user confirmed that the pictured food was consumed. Estimate the meal "
        "from the attached photo and emit a schema-v2 meal_observation with "
        "consumption_status=consumed, finite non-negative calories, and finite "
        "non-negative protein_g, fat_g, and carbohydrate_g values. Populate all "
        "three macro fields; never leave them null. "
        "Do not emit a correction, deletion, planned meal, or pre-confirmation record.\n"
        "EXIF is evidence only, not proof of eating or authoritative meal time. "
        f"Capture time evidence: {capture_time}. GPS is unavailable by design.\n"
        "Keep your own answer concise and state visual portion uncertainty; the "
        "trusted client renders the numeric result from the structured fields."
    )
    return prompt[:max_chars]


__all__ = ["build_confirmation_prompt", "build_post_confirmation_prompt"]

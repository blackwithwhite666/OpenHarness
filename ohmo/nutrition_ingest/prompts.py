"""Bounded prompts for the trusted post-confirmation nutrition turn."""

from __future__ import annotations

from collections.abc import Mapping


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
        "consumption_status=consumed and at least one finite energy value. "
        "Do not emit a correction, deletion, planned meal, or pre-confirmation record.\n"
        "EXIF is evidence only, not proof of eating or authoritative meal time. "
        f"Capture time evidence: {capture_time}. GPS is unavailable by design.\n"
        "Keep the answer concise and state visual portion uncertainty."
    )
    return prompt[:max_chars]


__all__ = ["build_post_confirmation_prompt"]

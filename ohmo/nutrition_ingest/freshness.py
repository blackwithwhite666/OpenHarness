"""Consumer-side freshness validation for producer EXIF provenance."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .models import ExifMetadata

MOSCOW = ZoneInfo("Europe/Moscow")
FRESHNESS_WINDOW = timedelta(days=7)


def exif_freshness_reason(exif: ExifMetadata, now: datetime) -> str | None:
    """Return a bounded skip reason when EXIF freshness cannot be trusted.

    This deliberately consumes only the EXIF object. Dropbox timestamps and
    discovery metadata are not valid substitutes for camera capture time.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("freshness clock must be timezone-aware")
    if exif.timezone_status == "ambiguous":
        return "exif_ambiguous"
    if not exif.normalized_capture_time:
        return "exif_missing"
    try:
        capture_time = normalized_exif_capture_time(exif)
        cutoff = now - FRESHNESS_WINDOW
        if capture_time < cutoff:
            return "exif_stale"
    except (TypeError, ValueError, OverflowError):
        return "exif_invalid"
    return None


def normalized_exif_capture_time(exif: ExifMetadata) -> datetime:
    """Return the authoritative normalized capture time or raise ``ValueError``."""
    if exif.timezone_status == "ambiguous" or not exif.normalized_capture_time:
        raise ValueError("authoritative EXIF capture time is unavailable")
    try:
        capture_time = datetime.fromisoformat(exif.normalized_capture_time)
    except (TypeError, ValueError) as error:
        raise ValueError("authoritative EXIF capture time is invalid") from error
    if capture_time.tzinfo is None or capture_time.utcoffset() is None:
        if exif.timezone_status != "missing" or exif.capture_timezone_offset is not None:
            raise ValueError("offsetless EXIF provenance is inconsistent")
        capture_time = capture_time.replace(tzinfo=MOSCOW)
    return capture_time


__all__ = [
    "FRESHNESS_WINDOW",
    "MOSCOW",
    "exif_freshness_reason",
    "normalized_exif_capture_time",
]

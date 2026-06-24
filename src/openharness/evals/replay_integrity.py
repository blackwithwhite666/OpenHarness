"""High-precision validation for captured eval replay inputs."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openharness.evals.models import EvalEpisode, EvalEvent

DOWNLOAD_FAILED_RE = re.compile(r"download failed", re.IGNORECASE)
MEDIA_MARKER_RE = re.compile(
    r"\[(?:voice|image|document|audio|video|sticker)\]",
    re.IGNORECASE,
)
TELEGRAM_FILE_ID_RE = re.compile(r"\bAg[A-Za-z0-9_-]{14,}\b")
MISSING_FILE_RE = re.compile(r"No such file|not found|ENOENT", re.IGNORECASE)
ABSOLUTE_PATH_RE = re.compile(r"(?<![\w.-])/[^\s\"'<>|;]+")


@dataclass(frozen=True)
class ReplayIntegrityToolFixture:
    """Minimal fixture shape needed by replay-integrity validation."""

    tool_name: str
    is_error: bool = False
    input_text: str = ""
    output_text: str = ""


def replay_integrity(
    episode: EvalEpisode | None,
    events: Sequence[EvalEvent],
    tool_fixtures: Iterable[object],
) -> tuple[bool, str | None]:
    """Return whether a captured eval case has replayable inputs."""
    if episode is None:
        return True, None
    input_text = _primary_input_text(episode, events)
    if DOWNLOAD_FAILED_RE.search(input_text):
        return False, "unreplayable_input:download_failed"
    if not MEDIA_MARKER_RE.sub("", input_text).strip():
        return False, "unreplayable_input:empty"

    if any(_read_file_fixture_is_missing_input(fixture) for fixture in tool_fixtures):
        return False, "missing_input_file"
    # A telegram file-id may sit in the reply-context / gold answer (not just this
    # turn's text) -- scan the whole episode so attachment-dependent turns are caught.
    if TELEGRAM_FILE_ID_RE.search(_episode_scan_text(episode, events)):
        return False, "unrecoverable_media_ref"
    return True, None


def _episode_scan_text(episode: EvalEpisode | None, events: Sequence[EvalEvent]) -> str:
    """All episode-side text (input + gold reply + event payloads) for media-ref scans."""
    parts: list[str] = []
    if episode is not None:
        parts.append(episode.user_text)
        parts.append(episode.user_goal)
    for event in events:
        payload = event.payload if isinstance(event.payload, dict) else {}
        for value in payload.values():
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(p for p in parts if p)


def replay_integrity_tool_fixtures(
    events: Sequence[EvalEvent],
) -> tuple[ReplayIntegrityToolFixture, ...]:
    """Build the minimal fixture view needed when execution fixtures are unavailable."""
    fixtures: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for index, event in enumerate(events):
        if not event.tool_name:
            continue
        call_key = (event.tool_name, event.tool_call_id or f"event-{index}")
        if call_key not in fixtures:
            fixtures[call_key] = {
                "tool_name": event.tool_name,
                "is_error": False,
                "input_text": "",
                "output_text": "",
            }
            order.append(call_key)
        fixture = fixtures[call_key]
        if event.kind == "tool_started":
            fixture["input_text"] = _payload_text(event.payload, ("input", "input_summary"))
        elif event.kind == "tool_completed":
            fixture["is_error"] = bool(event.is_error)
            fixture["output_text"] = _payload_text(event.payload, ("output", "output_summary"))
        else:
            fixture["is_error"] = bool(fixture["is_error"] or event.is_error)
    return tuple(ReplayIntegrityToolFixture(**fixtures[key]) for key in order)


def _primary_input_text(
    episode: EvalEpisode | None,
    events: Sequence[EvalEvent],
) -> str:
    if episode is not None:
        if episode.user_text.strip():
            return episode.user_text
        if episode.user_goal.strip():
            return episode.user_goal

    for event in events:
        if event.kind != "inbound_message":
            continue
        for field_name in ("user_text", "user_goal", "text"):
            value = event.payload.get(field_name)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _read_file_fixture_is_missing_input(fixture: object) -> bool:
    if getattr(fixture, "tool_name", None) != "read_file":
        return False
    # Only a REAL read error counts as a missing input. Matching "not found" /
    # "ENOENT" in the OUTPUT content false-positives on files that merely contain
    # those words (e.g. a skill's docs) -- that over-blocked a passing case.
    if not bool(getattr(fixture, "is_error", False)):
        return False
    input_text = _fixture_text(fixture, "input_text")
    output_text = _fixture_text(fixture, "output_text")
    return _contains_absolute_path(input_text) or _contains_absolute_path(output_text)


def _fixture_text(fixture: object, field_name: str) -> str:
    value = getattr(fixture, field_name, "")
    return value if isinstance(value, str) else ""


def _payload_text(payload: dict[str, Any], field_names: Sequence[str]) -> str:
    for field_name in field_names:
        if field_name not in payload:
            continue
        return _text_value(payload[field_name])
    return ""


def _text_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError):
        return str(type(value).__name__)


def _contains_absolute_path(text: str) -> bool:
    if not text:
        return False
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = None
    if decoded is not None and any(_value_is_absolute_path(value) for value in _walk(decoded)):
        return True
    return any(Path(match.group(0)).is_absolute() for match in ABSOLUTE_PATH_RE.finditer(text))


def _walk(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
        return
    if isinstance(value, list):
        for item in value:
            yield from _walk(item)
        return
    yield value


def _value_is_absolute_path(value: Any) -> bool:
    return isinstance(value, str) and Path(value).expanduser().is_absolute()

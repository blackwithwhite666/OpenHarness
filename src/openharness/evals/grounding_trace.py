"""Trace lookup helpers for faithful grounding attribution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_session_trace(trace_root: Path | None, session_id: str) -> dict[str, object]:
    if trace_root is None:
        return {}
    path = _session_trace_path(trace_root, session_id)
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"trace is not a JSON object: {path}")
    return value


def trace_text_fields(
    trace: dict[str, object], metadata: dict[str, Any]
) -> dict[str, str]:
    trace_metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    task = (
        trace.get("intent")
        or trace.get("task")
        or trace.get("prompt")
        or trace_metadata.get("intent")
        or metadata.get("grounding_task")
        or metadata.get("intent")
        or ""
    )
    answer = (
        trace.get("final_text")
        or trace.get("answer")
        or trace_metadata.get("final_text")
        or metadata.get("grounding_answer")
        or metadata.get("final_text")
        or ""
    )
    return {"task": str(task), "answer": str(answer)}


def _session_trace_path(trace_root: Path, session_id: str) -> Path | None:
    for name in (f"{session_id}-0.json", f"{session_id}.json"):
        path = trace_root / name
        if path.is_file():
            return path
    if not trace_root.is_dir():
        return None
    prefix = f"{session_id}-"
    matches: list[tuple[int, Path]] = []
    for path in trace_root.rglob("*.json"):
        relative_name = path.relative_to(trace_root).as_posix()
        if not relative_name.startswith(prefix):
            continue
        sample = relative_name[len(prefix) : -len(".json")]
        try:
            matches.append((int(sample), path))
        except ValueError:
            continue
    return sorted(matches, key=lambda item: item[0])[0][1] if matches else None

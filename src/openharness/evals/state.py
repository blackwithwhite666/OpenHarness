"""Metadata-only helpers for captured eval world-state deltas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from openharness.evals.models import EvalResourceSnapshot
from openharness.evals.store import EvalStore

_STATE_RESOURCES = ("reminders", "memory", "todos")
_SNAPSHOT_RESOURCE_MAP = {
    "reminders_json": ("reminders", "record_count"),
    "memory_dir": ("memory", "entry_count"),
    "todos_dir": ("todos", "entry_count"),
}


def extract_state_keys(snapshot: EvalResourceSnapshot) -> dict[str, dict[str, Any]]:
    """Extract metadata-only mutation keys from a captured resource snapshot."""
    keys = {
        resource_name: _empty_state_entry(resource_name)
        for resource_name in _STATE_RESOURCES
    }
    for resource in snapshot.resources:
        resource_spec = _SNAPSHOT_RESOURCE_MAP.get(resource.name)
        if resource_spec is None:
            continue
        logical_name, count_key = resource_spec
        metadata = resource.metadata or {}
        entry: dict[str, Any] = {
            "entry_keys": _sorted_string_list(metadata.get("entry_keys")),
            "count": _safe_int(metadata.get(count_key), default=0),
        }
        if logical_name == "reminders":
            entry["status_counts"] = _status_counts(metadata.get("status_counts"))
        keys[logical_name] = entry
    return keys


def compute_state_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compute a JSON-safe key/count delta between extracted state snapshots."""
    delta: dict[str, Any] = {}
    changed = False
    for resource_name in _STATE_RESOURCES:
        before_entry = _entry_mapping(before.get(resource_name))
        after_entry = _entry_mapping(after.get(resource_name))
        before_keys = set(_sorted_string_list(before_entry.get("entry_keys")))
        after_keys = set(_sorted_string_list(after_entry.get("entry_keys")))
        added_keys = sorted(after_keys - before_keys)
        removed_keys = sorted(before_keys - after_keys)
        resource_delta: dict[str, Any] = {
            "added_keys": added_keys,
            "removed_keys": removed_keys,
            "count_before": _safe_int(before_entry.get("count"), default=0),
            "count_after": _safe_int(after_entry.get("count"), default=0),
        }
        if resource_name == "reminders":
            resource_delta["status_counts_before"] = _status_counts(
                before_entry.get("status_counts")
            )
            resource_delta["status_counts_after"] = _status_counts(
                after_entry.get("status_counts")
            )
        delta[resource_name] = resource_delta
        changed = changed or bool(added_keys or removed_keys)
    delta["changed"] = changed
    return delta


def read_world_snapshots(
    store: EvalStore,
    episode_id: str,
) -> tuple[EvalResourceSnapshot | None, EvalResourceSnapshot | None]:
    """Read captured before/after snapshots, returning None for unavailable files."""
    return (
        _read_world_snapshot(store, episode_id, "world_before"),
        _read_world_snapshot(store, episode_id, "world_after"),
    )


def compute_episode_state_delta(store: EvalStore, episode_id: str) -> dict[str, Any] | None:
    """Compute a captured episode state delta when both snapshots are available."""
    before_snapshot, after_snapshot = read_world_snapshots(store, episode_id)
    if before_snapshot is None or after_snapshot is None:
        return None
    return compute_state_delta(
        extract_state_keys(before_snapshot),
        extract_state_keys(after_snapshot),
    )


def _read_world_snapshot(
    store: EvalStore,
    episode_id: str,
    snapshot_name: str,
) -> EvalResourceSnapshot | None:
    try:
        path = (store.root / "states" / episode_id / f"{snapshot_name}.json").resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if not _is_relative_to(path, store.root):
        return None
    try:
        return EvalResourceSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError, ValueError):
        return None


def _empty_state_entry(resource_name: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"entry_keys": [], "count": 0}
    if resource_name == "reminders":
        entry["status_counts"] = {}
    return entry


def _entry_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sorted_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted(item for item in value if isinstance(item, str))


def _status_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): _safe_int(count, default=0)
        for key, count in sorted(value.items(), key=lambda item: str(item[0]))
    }


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True

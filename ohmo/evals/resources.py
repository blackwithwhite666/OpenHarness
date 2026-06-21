"""Metadata-only resource snapshots for ohmo eval episodes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any

from openharness.evals import EvalResource, EvalResourceSnapshot, EvalStore
from openharness.utils.fs import atomic_write_text

from ohmo.workspace import (
    get_attachments_dir,
    get_bootstrap_path,
    get_contacts_path,
    get_gateway_config_path,
    get_gateway_restart_notice_path,
    get_groups_dir,
    get_identity_path,
    get_memory_dir,
    get_memory_index_path,
    get_plugins_dir,
    get_reminders_path,
    get_sessions_dir,
    get_skills_dir,
    get_soul_path,
    get_state_path,
    get_user_path,
    get_workspace_root,
)

_RESOURCE_ID_SAFE = re.compile(r"[^A-Za-z0-9_.:-]+")
_DIRECTORY_AGGREGATE_ENTRY_LIMIT = 5000
_SNAPSHOT_PHASES = frozenset({"world_before", "world_after"})
_TODO_ITEM_RE = re.compile(r"^\s*[-*]\s+(?:\[[ xX]\]\s+)?(.+?)\s*$")


@dataclass(frozen=True)
class ResourceSnapshotWrite:
    """Summary returned after writing a resource snapshot manifest."""

    manifest: EvalResourceSnapshot
    path: Path
    relative_path: str
    resource_count: int
    local_resource_count: int
    tool_count: int


_WorkspacePathFactory = Callable[[str | Path | None], Path]

_LOCAL_RESOURCES: tuple[tuple[str, str, _WorkspacePathFactory, str], ...] = (
    ("soul_md", "local_file", get_soul_path, "markdown"),
    ("user_md", "local_file", get_user_path, "markdown"),
    ("identity_md", "local_file", get_identity_path, "markdown"),
    ("bootstrap_md", "local_file", get_bootstrap_path, "markdown"),
    ("memory_dir", "local_directory", get_memory_dir, "directory"),
    ("memory_index", "local_file", get_memory_index_path, "markdown"),
    ("sessions_dir", "local_directory", get_sessions_dir, "directory"),
    ("attachments_dir", "local_directory", get_attachments_dir, "directory"),
    ("skills_dir", "local_directory", get_skills_dir, "directory"),
    ("plugins_dir", "local_directory", get_plugins_dir, "directory"),
    ("groups_dir", "local_directory", get_groups_dir, "directory"),
    ("state_json", "local_file", get_state_path, "json"),
    ("gateway_json", "local_file", get_gateway_config_path, "json"),
    ("gateway_restart_notice_json", "local_file", get_gateway_restart_notice_path, "json"),
    ("reminders_json", "local_file", get_reminders_path, "json_count"),
    ("contacts_json", "local_file", get_contacts_path, "json_count"),
)


def build_ohmo_resource_snapshot(
    *,
    episode_id: str,
    workspace: str | Path | None,
    bundle: Any | None = None,
) -> EvalResourceSnapshot:
    """Build a metadata-only manifest for known ohmo resources and runtime tools."""
    resources = [
        *_local_resources(workspace),
        *_runtime_tool_resources(bundle),
    ]
    return EvalResourceSnapshot(episode_id=episode_id, resources=resources)


def write_ohmo_resource_snapshot(
    *,
    store: EvalStore,
    episode_id: str,
    workspace: str | Path | None,
    bundle: Any | None = None,
    phase: str = "world_before",
) -> ResourceSnapshotWrite:
    """Write the episode resource snapshot under ``states/<episode_id>/``."""
    if phase not in _SNAPSHOT_PHASES:
        raise ValueError(f"unknown snapshot phase: {phase!r}")
    manifest = build_ohmo_resource_snapshot(
        episode_id=episode_id,
        workspace=workspace,
        bundle=bundle,
    )
    path = store.root / "states" / episode_id / f"{phase}.json"
    atomic_write_text(path, manifest.model_dump_json(indent=2) + "\n")

    relative_path = path.relative_to(store.root).as_posix()
    tool_count = sum(1 for resource in manifest.resources if resource.kind == "runtime_tool")
    local_resource_count = sum(
        1 for resource in manifest.resources if resource.kind.startswith("local_")
    )
    return ResourceSnapshotWrite(
        manifest=manifest,
        path=path,
        relative_path=relative_path,
        resource_count=len(manifest.resources),
        local_resource_count=local_resource_count,
        tool_count=tool_count,
    )


def _local_resources(workspace: str | Path | None) -> list[EvalResource]:
    root = get_workspace_root(workspace)
    resources = [
        _filesystem_resource(
            resource_id=f"ohmo.workspace.{name}",
            kind=kind,
            name=name,
            path=resolver(root),
            workspace_root=root,
            profile=profile,
        )
        for name, kind, resolver, profile in _LOCAL_RESOURCES
    ]
    todos_dir = root / "todos"
    if todos_dir.exists():
        resources.append(
            _filesystem_resource(
                resource_id="ohmo.workspace.todos_dir",
                kind="local_directory",
                name="todos_dir",
                path=todos_dir,
                workspace_root=root,
                profile="directory",
            )
        )
    return resources


def _filesystem_resource(
    *,
    resource_id: str,
    kind: str,
    name: str,
    path: Path,
    workspace_root: Path,
    profile: str,
) -> EvalResource:
    absolute_path = path.expanduser().resolve()
    metadata: dict[str, Any] = {
        "scope": "ohmo_workspace",
        "profile": profile,
    }
    try:
        stat_result = absolute_path.lstat()
    except FileNotFoundError:
        metadata["parse_status"] = "missing" if profile.startswith("json") else "not_applicable"
        return EvalResource(
            resource_id=resource_id,
            kind=kind,
            name=name,
            path=_logical_path(absolute_path, workspace_root),
            exists=False,
            metadata=metadata,
        )

    actual_kind = _actual_kind(stat_result.st_mode)
    metadata["actual_kind"] = actual_kind
    if actual_kind == "directory":
        metadata.update(_directory_aggregate(absolute_path, stat_result))
        metadata["parse_status"] = "not_applicable"
    elif actual_kind == "file":
        metadata["file_count"] = 1
        metadata["total_size_bytes"] = stat_result.st_size
        metadata["newest_mtime_ns"] = stat_result.st_mtime_ns
        metadata.update(_file_parse_metadata(absolute_path, profile))
    else:
        metadata["file_count"] = 0
        metadata["total_size_bytes"] = stat_result.st_size
        metadata["newest_mtime_ns"] = stat_result.st_mtime_ns
        metadata["parse_status"] = "not_applicable"
    if actual_kind in {"directory", "file"}:
        metadata.update(_state_key_metadata(name, absolute_path))

    return EvalResource(
        resource_id=resource_id,
        kind=kind,
        name=name,
        path=_logical_path(absolute_path, workspace_root),
        exists=True,
        size_bytes=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        metadata=_json_safe_mapping(metadata),
    )


def _directory_aggregate(path: Path, root_stat: object) -> dict[str, Any]:
    file_count = 0
    dir_count = 0
    total_size_bytes = 0
    newest_mtime_ns = int(getattr(root_stat, "st_mtime_ns"))
    visited_count = 0
    truncated = False
    pending_dirs = [path]

    while pending_dirs and visited_count < _DIRECTORY_AGGREGATE_ENTRY_LIMIT:
        current_dir = pending_dirs.pop()
        try:
            entries = os.scandir(current_dir)
        except OSError:
            continue

        with entries:
            for entry in entries:
                if visited_count >= _DIRECTORY_AGGREGATE_ENTRY_LIMIT:
                    truncated = True
                    break
                visited_count += 1
                try:
                    child_stat = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                newest_mtime_ns = max(newest_mtime_ns, child_stat.st_mtime_ns)
                if stat.S_ISDIR(child_stat.st_mode):
                    dir_count += 1
                    pending_dirs.append(Path(entry.path))
                elif stat.S_ISREG(child_stat.st_mode):
                    file_count += 1
                    total_size_bytes += child_stat.st_size
        if truncated:
            break

    if pending_dirs and visited_count >= _DIRECTORY_AGGREGATE_ENTRY_LIMIT:
        truncated = True

    return {
        "file_count": file_count,
        "dir_count": dir_count,
        "total_size_bytes": total_size_bytes,
        "newest_mtime_ns": newest_mtime_ns,
        "truncated": truncated,
        "entry_limit": _DIRECTORY_AGGREGATE_ENTRY_LIMIT,
        "visited_count": visited_count,
    }


def _file_parse_metadata(path: Path, profile: str) -> dict[str, Any]:
    if profile == "json":
        return _json_parse_metadata(path, include_count=False)
    if profile == "json_count":
        return _json_parse_metadata(path, include_count=True)
    return {"parse_status": "not_applicable"}


def _json_parse_metadata(path: Path, *, include_count: bool) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"parse_status": "error", "parse_error_type": type(exc).__name__}

    metadata: dict[str, Any] = {
        "parse_status": "ok",
        "json_type": _json_type_name(raw),
    }
    if include_count and isinstance(raw, (dict, list)):
        metadata["record_count"] = len(raw)
    return metadata


def _state_key_metadata(name: str, path: Path) -> dict[str, Any]:
    """Return metadata-only state keys for resources with structured private state."""
    try:
        if name == "reminders_json":
            return _reminders_state_key_metadata(path)
        if name == "memory_dir":
            return _memory_state_key_metadata(path)
        if name == "todos_dir":
            return _todos_state_key_metadata(path)
    except Exception:
        return {"state_keys_status": "error"}
    return {}


def _stable_key(parts: list[Any]) -> str:
    encoded = json.dumps(
        parts,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _reminders_state_key_metadata(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return {}

    entry_keys: set[str] = set()
    status_counts: dict[str, int] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        entry_keys.add(
            _stable_key(
                [
                    _mapping_field(item, "mode"),
                    _mapping_field(item, "tz"),
                    _mapping_field(item, "dtstart"),
                    _mapping_field(item, "rrule"),
                    _mapping_field(item, "channel"),
                    _mapping_field(item, "chat_id"),
                ]
            )
        )
        status = _mapping_field(item, "status")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "entry_keys": sorted(entry_keys),
        "status_counts": dict(sorted(status_counts.items())),
    }


def _mapping_field(item: Mapping[str, Any], key: str) -> str:
    value = item.get(key)
    return "" if value is None else str(value)


def _memory_state_key_metadata(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {}
    entry_keys = sorted(
        child.stem for child in path.iterdir() if child.is_file() and child.suffix == ".md"
    )
    return {"entry_keys": entry_keys, "entry_count": len(entry_keys)}


def _todos_state_key_metadata(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {}

    entry_keys: set[str] = set()
    entry_count = 0
    for item_path in sorted(path.glob("*.md")):
        if not item_path.is_file():
            continue
        for line in item_path.read_text(encoding="utf-8").splitlines():
            match = _TODO_ITEM_RE.match(line)
            if match is None:
                continue
            normalized = " ".join(match.group(1).strip().split()).lower()
            if not normalized:
                continue
            entry_count += 1  # Total parsed items before hash de-duplication.
            entry_keys.add(_stable_key([normalized]))

    return {"entry_keys": sorted(entry_keys), "entry_count": entry_count}


def _runtime_tool_resources(bundle: Any | None) -> list[EvalResource]:
    registry = getattr(bundle, "tool_registry", None)
    if registry is None:
        return []
    list_tools = getattr(registry, "list_tools", None)
    if not callable(list_tools):
        return []
    tools = list(list_tools())
    tools.sort(key=lambda tool: str(getattr(tool, "name", type(tool).__name__) or ""))
    return [_runtime_tool_resource(tool) for tool in tools]


def _runtime_tool_resource(tool: Any) -> EvalResource:
    name = str(getattr(tool, "name", "") or type(tool).__name__)
    description = str(getattr(tool, "description", "") or "")
    input_schema = _tool_input_schema(tool)
    safe_schema = _json_safe(input_schema)
    metadata: dict[str, Any] = {
        "description": description,
        "input_schema_hash": _schema_hash(safe_schema),
    }
    if _is_json_safe(input_schema):
        metadata["input_schema"] = input_schema
    return EvalResource(
        resource_id=f"ohmo.runtime_tool.{_safe_resource_id(name)}",
        kind="runtime_tool",
        name=name,
        exists=True,
        metadata=_json_safe_mapping(metadata),
    )


def _tool_input_schema(tool: Any) -> Any:
    to_api_schema = getattr(tool, "to_api_schema", None)
    if callable(to_api_schema):
        schema = to_api_schema()
        if isinstance(schema, Mapping):
            return schema.get("input_schema", {})

    input_model = getattr(tool, "input_model", None)
    model_json_schema = getattr(input_model, "model_json_schema", None)
    if callable(model_json_schema):
        return model_json_schema()

    input_schema = getattr(tool, "input_schema", None)
    return input_schema if input_schema is not None else {}


def _schema_hash(schema: Any) -> str:
    encoded = json.dumps(
        schema,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _logical_path(path: Path, workspace_root: Path) -> str:
    root = workspace_root.expanduser().resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _actual_kind(mode: int) -> str:
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "other"


def _json_type_name(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if value is None:
        return "null"
    return type(value).__name__


def _safe_resource_id(value: str) -> str:
    return _RESOURCE_ID_SAFE.sub("_", value).strip("._:-") or "tool"


def _is_json_safe(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    safe = _json_safe(value)
    if not isinstance(safe, dict):
        return {"value": safe}
    return safe


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return repr(value)

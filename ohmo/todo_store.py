"""Per-session, Markdown-backed storage for OHMO todo snapshots.

The model-facing contract is a complete typed snapshot, while the pointer and
per-list files remain compatible with the older ``new_list`` implementation.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")
_SAFE_LIST_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_TODO_LINE = re.compile(r"^\s*-\s+\[([ xX~!])\]\s+(.*?)\s*$")
_BLOCKED_REASON_LINE = re.compile(r"^\s{2,}(?:>\s*)?blocked_reason:\s*(.*?)\s*$")

TODO_STATUSES = ("pending", "in_progress", "completed", "blocked")
MAX_TODOS = 100
MAX_TODO_CONTENT_LENGTH = 500
MAX_BLOCKED_REASON_LENGTH = 500

TodoSnapshot = list[dict[str, str]]


def _safe(component: str) -> str:
    """Make a session id safe to embed in a filename."""
    return _UNSAFE.sub("_", component) or "default"


def _is_safe_list_id(value: object) -> bool:
    return isinstance(value, str) and _SAFE_LIST_ID.fullmatch(value) is not None


def canonicalize_text(value: str) -> str:
    """Apply the one text normalization used by input and Markdown parsing."""
    return " ".join(unicodedata.normalize("NFKC", value).split())


def canonicalize_todos(items: Iterable[Mapping[str, object]]) -> TodoSnapshot:
    """Validate and canonicalize an iterable of raw todo mappings."""
    result: TodoSnapshot = []
    identities: set[str] = set()
    in_progress = 0

    for index, raw in enumerate(items):
        if index >= MAX_TODOS:
            raise ValueError(f"todos may contain at most {MAX_TODOS} items")
        if not isinstance(raw, Mapping):
            raise TypeError(f"todo {index} must be an object")

        content = raw.get("content")
        if not isinstance(content, str):
            raise TypeError(f"todo {index} content must be a string")
        content = canonicalize_text(content)
        if not content:
            raise ValueError(f"todo {index} content must not be empty")
        if len(content) > MAX_TODO_CONTENT_LENGTH:
            raise ValueError(
                f"todo {index} content exceeds {MAX_TODO_CONTENT_LENGTH} characters"
            )

        status = raw.get("status")
        if status not in TODO_STATUSES:
            allowed = ", ".join(TODO_STATUSES)
            raise ValueError(f"todo {index} status must be one of: {allowed}")
        if status == "in_progress":
            in_progress += 1
            if in_progress > 1:
                raise ValueError("at most one todo may be in_progress")

        reason = raw.get("blocked_reason")
        if status == "blocked":
            if not isinstance(reason, str):
                raise ValueError("blocked todo requires blocked_reason")
            reason = canonicalize_text(reason)
            if not reason:
                raise ValueError("blocked todo requires a non-empty blocked_reason")
            if len(reason) > MAX_BLOCKED_REASON_LENGTH:
                raise ValueError(
                    "blocked_reason exceeds "
                    f"{MAX_BLOCKED_REASON_LENGTH} characters"
                )
        elif reason is not None:
            raise ValueError("blocked_reason is only allowed for blocked todos")

        identity = content.casefold()
        if identity in identities:
            raise ValueError(f"duplicate todo content: {content!r}")
        identities.add(identity)

        item = {"content": content, "status": status}
        if status == "blocked":
            item["blocked_reason"] = reason
        result.append(item)

    if len(result) > MAX_TODOS:
        raise ValueError(f"todos may contain at most {MAX_TODOS} items")
    return result


def _status_marker(status: str) -> str:
    return {
        "pending": " ",
        "in_progress": "~",
        "completed": "x",
        "blocked": "!",
    }[status]


def render_todo_markdown(todos: Iterable[Mapping[str, object]]) -> str:
    """Render a canonical, human-readable UTF-8 Markdown checklist."""
    canonical = canonicalize_todos(todos)
    lines = ["# TODO"]
    for item in canonical:
        lines.append(f"- [{_status_marker(item['status'])}] {item['content']}")
        if item["status"] == "blocked":
            lines.append(f"  blocked_reason: {item['blocked_reason']}")
    return "\n".join(lines) + "\n"


def _raw_todos_from_markdown(text: str) -> list[dict[str, str]]:
    """Read both canonical and legacy checklist markers."""
    raw: list[dict[str, str]] = []
    pending_blocked: dict[str, str] | None = None
    for line in text.splitlines():
        match = _TODO_LINE.match(line)
        if match:
            marker, content = match.groups()
            marker = marker.lower()
            status = {
                " ": "pending",
                "~": "in_progress",
                "x": "completed",
                "!": "blocked",
            }[marker]
            item = {"content": content, "status": status}
            raw.append(item)
            pending_blocked = item if status == "blocked" else None
            continue

        if pending_blocked is not None:
            reason_match = _BLOCKED_REASON_LINE.match(line)
            if reason_match:
                pending_blocked["blocked_reason"] = reason_match.group(1)
                pending_blocked = None

    return raw


def parse_todo_markdown(text: str) -> TodoSnapshot:
    """Parse legacy or canonical Markdown into a canonical todo snapshot."""
    return canonicalize_todos(_raw_todos_from_markdown(text))


class TodoStore:
    """Resolve active per-session files and atomically replace their snapshots."""

    def __init__(self, workspace: str | Path):
        self._dir = Path(workspace) / "todos"
        self._pointer_path = self._dir / "active.json"

    @property
    def dir(self) -> Path:
        return self._dir

    def _load_pointers(self) -> dict[str, str]:
        try:
            data = json.loads(self._pointer_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _atomic_write(self, path: Path, content: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=self._dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            Path(temp_name).replace(path)
        except BaseException:
            try:
                Path(temp_name).unlink()
            except OSError:
                pass
            raise

    def _archive_bytes(self, path: Path, content: bytes) -> Path:
        """Copy legacy bytes to an adjacent, never-overwritten archive."""
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.legacy.", dir=self._dir)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())

            suffix = 1
            while True:
                name = f"{path.name}.legacy" if suffix == 1 else f"{path.name}.legacy-{suffix}"
                archive_path = self._dir / name
                try:
                    os.link(temp_path, archive_path)
                except FileExistsError:
                    suffix += 1
                    continue
                temp_path.unlink()
                return archive_path
        except BaseException:
            try:
                temp_path.unlink()
            except OSError:
                pass
            raise

    def _save_pointers(self, pointers: dict[str, str]) -> None:
        self._atomic_write(
            self._pointer_path,
            json.dumps(pointers, ensure_ascii=False, indent=2) + "\n",
        )

    def list_id(self, session_id: str) -> str:
        """Return the active list id for a session."""
        sid = _safe(session_id)
        pointer = self._load_pointers().get(sid)
        return pointer if _is_safe_list_id(pointer) else sid

    def active_path(self, session_id: str) -> Path:
        """Return the active file path and ensure the storage directory exists."""
        self._dir.mkdir(parents=True, exist_ok=True)
        return self._dir / f"{self.list_id(session_id)}.md"

    def read_snapshot(self, session_id: str) -> tuple[TodoSnapshot, bool]:
        """Return ``(snapshot, is_canonical_file)`` for the active list."""
        path = self.active_path(session_id)
        if not path.exists():
            return [], True
        text = path.read_text(encoding="utf-8")
        snapshot = parse_todo_markdown(text)
        return snapshot, text == render_todo_markdown(snapshot)

    def replace_snapshot(self, session_id: str, todos: Iterable[Mapping[str, object]]) -> bool:
        """Atomically write a complete snapshot and return semantic ``changed``."""
        canonical = canonicalize_todos(todos)
        path = self.active_path(session_id)
        if not path.exists():
            self._atomic_write(path, render_todo_markdown(canonical))
            return bool(canonical)

        old_bytes = path.read_bytes()
        try:
            old_text = old_bytes.decode("utf-8")
            existing = parse_todo_markdown(old_text)
            is_canonical = old_text == render_todo_markdown(existing)
        except (UnicodeError, TypeError, ValueError):
            existing = None
            is_canonical = False

        changed = existing != canonical
        if changed or not is_canonical:
            if not is_canonical:
                self._archive_bytes(path, old_bytes)
            self._atomic_write(path, render_todo_markdown(canonical))
        return changed

    def new_list(self, session_id: str) -> Path:
        """Rotate to an empty internal list while preserving the old archive."""
        sid = _safe(session_id)
        self._dir.mkdir(parents=True, exist_ok=True)
        existing = {p.stem for p in self._dir.glob(f"{sid}*.md")}
        n = 2
        while f"{sid}-{n}" in existing:
            n += 1
        new_id = f"{sid}-{n}"
        pointers = self._load_pointers()
        pointers[sid] = new_id
        self._save_pointers(pointers)
        path = self._dir / f"{new_id}.md"
        self._atomic_write(path, render_todo_markdown([]))
        return path

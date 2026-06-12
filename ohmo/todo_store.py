"""Per-session to-do list storage for ohmo.

Replaces the single shared ``<cwd>/TODO.md`` — which leaked one chat's todos into
every other chat (they all ran with the same cwd) — with **one markdown file per
list**, plus a pointer from each session to its active list.

- Files live in ``<workspace>/todos/<list-id>.md``.
- ``list-id`` defaults to the **session_id**, so every conversation gets its own
  file and ``/new`` (which mints a fresh session_id) starts clean while the old
  file is kept on disk.
- ``new_list`` rotates to ``<session_id>-2``, ``-3`` … and moves the pointer, so
  the agent can start a clean list for an unrelated task *mid-conversation*
  without losing the previous list's file.
- The pointer (session_id → list-id) is persisted in ``todos/active.json`` so a
  rotation survives a gateway restart.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def _safe(component: str) -> str:
    """Make a session_id safe to embed in a filename (session_id is normally
    12 hex chars, but never trust it blindly)."""
    return _UNSAFE.sub("_", component) or "default"


class TodoStore:
    """Resolves the active to-do file for a session and rotates lists."""

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

    def _save_pointers(self, pointers: dict[str, str]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._pointer_path.with_name("active.json.tmp")
        tmp.write_text(json.dumps(pointers, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._pointer_path)

    def list_id(self, session_id: str) -> str:
        """The active list-id for a session — its session_id unless ``new_list``
        rotated it to a suffixed id."""
        sid = _safe(session_id)
        return self._load_pointers().get(sid, sid)

    def active_path(self, session_id: str) -> Path:
        """Path to the session's active list file (the ``todos`` dir is ensured)."""
        self._dir.mkdir(parents=True, exist_ok=True)
        return self._dir / f"{self.list_id(session_id)}.md"

    def new_list(self, session_id: str) -> Path:
        """Rotate this session to a fresh, empty list — keeping the previous
        file — and repoint it. Returns the new active path."""
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
        path.write_text("# TODO\n", encoding="utf-8")
        return path

"""Per-chat live-location store for chat channels.

Telegram (and other platforms) deliver a *live* location as an initial message
with ``live_period`` set, followed by a stream of ``edited_message`` updates that
carry the same ``message_id`` and refreshed coordinates. Feeding every edit into
the agent would burn tokens, spam the chat, and risk session poisoning, so edits
are recorded here **silently** and surfaced into the next real user turn (or on
demand) instead of becoming turns of their own.

The store is intentionally tiny and file-based: one JSON record per chat, written
atomically. Reads transparently drop expired shares (``now > expires_at``).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


class LiveLocationStore:
    """File-backed latest-known live location, keyed by chat id."""

    def __init__(self, directory: Path):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, chat_id: str) -> Path:
        safe = "".join(c for c in str(chat_id) if c.isalnum() or c in "-_") or "chat"
        return self._dir / f"{safe}.json"

    def update(
        self,
        chat_id: str,
        *,
        latitude: float,
        longitude: float,
        expires_at: float,
        heading: int | None = None,
        horizontal_accuracy: float | None = None,
        message_id: int | None = None,
        updated_at: float | None = None,
    ) -> None:
        """Record the latest coordinates for *chat_id* (atomic replace)."""
        record: dict[str, Any] = {
            "chat_id": str(chat_id),
            "latitude": latitude,
            "longitude": longitude,
            "heading": heading,
            "horizontal_accuracy": horizontal_accuracy,
            "message_id": message_id,
            "updated_at": time.time() if updated_at is None else updated_at,
            "expires_at": expires_at,
        }
        path = self._path(chat_id)
        fd, tmp = tempfile.mkstemp(dir=str(self._dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(record, fh)
            os.replace(tmp, path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)

    def get(self, chat_id: str, *, now: float | None = None) -> dict[str, Any] | None:
        """Return the latest non-expired record for *chat_id*, else ``None``."""
        now = time.time() if now is None else now
        path = self._path(chat_id)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        expires_at = record.get("expires_at")
        if expires_at is not None and now > expires_at:
            return None
        return record

    def clear(self, chat_id: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            self._path(chat_id).unlink()

"""Per-chat last-known-location store for chat channels.

A chat platform may deliver a location as a static pin, a venue, or a *live*
share that streams ``edited_message`` movement updates. We don't want any of
these to spawn an agent turn (an hour of live edits would be dozens of model
calls); instead every inbound location silently overwrites the chat's last known
location here, and it is injected into the next real user turn as context — only
if one exists and its update is no more than seven days old. Records persist until
replaced even after they become too old for prompt injection.

The store is tiny and file-based: one JSON record per chat, written atomically.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


class LastLocationStore:
    """File-backed last-known location, keyed by chat id."""

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
        source: str | None = None,
        label: str | None = None,
        horizontal_accuracy: float | None = None,
        expires_at: float | None = None,
        updated_at: float | None = None,
    ) -> None:
        """Overwrite the last known location for *chat_id* (atomic replace).

        ``expires_at`` (epoch seconds) is the live-share expiry when known; it is
        retained and may be surfaced in an eligible prompt, but never used to drop
        the record. The transport separately limits prompt injection by
        ``updated_at`` age without deleting stored data."""
        record: dict[str, Any] = {
            "chat_id": str(chat_id),
            "latitude": latitude,
            "longitude": longitude,
            "source": source,
            "label": label,
            "horizontal_accuracy": horizontal_accuracy,
            "expires_at": expires_at,
            "updated_at": time.time() if updated_at is None else updated_at,
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

    def get(self, chat_id: str) -> dict[str, Any] | None:
        """Return the last known location for *chat_id*, or ``None`` if unset."""
        path = self._path(chat_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    def clear(self, chat_id: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            self._path(chat_id).unlink()

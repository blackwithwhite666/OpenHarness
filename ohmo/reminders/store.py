"""Persistent reminder store (atomic write + file lock).

Mirrors :mod:`openharness.services.cron`: every mutation runs load -> mutate ->
save under an exclusive file lock, and ``_save`` uses ``atomic_write_text`` so a
crash never replaces a valid ``reminders.json`` with a partial one. A single
:class:`ReminderStore` instance is shared by the per-session tools and the
scheduler; an external ``asyncio.Lock`` guards the in-process concurrency on top
of the cross-process file lock.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from openharness.utils.file_lock import exclusive_file_lock
from openharness.utils.fs import atomic_write_text

from ohmo.reminders.model import Reminder
from ohmo.workspace import get_reminders_path

logger = logging.getLogger(__name__)


class ReminderStore:
    """File-backed CRUD for :class:`Reminder` records."""

    def __init__(self, workspace: str | Path | None = None) -> None:
        self._workspace = workspace

    def _path(self) -> Path:
        return get_reminders_path(self._workspace)

    def _lock_path(self) -> Path:
        path = self._path()
        return path.with_suffix(path.suffix + ".lock")

    def load(self) -> list[Reminder]:
        """Read + parse ``reminders.json``. Missing or corrupt -> ``[]`` (+ warn)."""
        path = self._path()
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("ohmo reminders store unreadable path=%s error=%s", path, exc)
            return []
        if not isinstance(raw, list):
            logger.warning("ohmo reminders store is not a list path=%s", path)
            return []
        reminders: list[Reminder] = []
        for item in raw:
            try:
                reminders.append(Reminder.model_validate(item))
            except ValidationError as exc:
                logger.warning("ohmo reminders store skipped invalid record error=%s", exc)
        return reminders

    def _save(self, reminders: list[Reminder]) -> None:
        """Persist atomically. The caller MUST hold the file lock."""
        atomic_write_text(
            self._path(),
            json.dumps([r.model_dump() for r in reminders], indent=2) + "\n",
        )

    def add(self, reminder: Reminder) -> None:
        with exclusive_file_lock(self._lock_path()):
            reminders = self.load()
            reminders.append(reminder)
            self._save(reminders)

    def get(self, reminder_id: str) -> Reminder | None:
        for reminder in self.load():
            if reminder.id == reminder_id:
                return reminder
        return None

    def list_for_chat(
        self, channel: str, chat_id: str, *, status: str = "active"
    ) -> list[Reminder]:
        matches = [
            reminder
            for reminder in self.load()
            if reminder.channel == channel
            and reminder.chat_id == chat_id
            and (status is None or reminder.status == status)
        ]
        matches.sort(key=lambda r: r.next_fire_at)
        return matches

    def count_active_for_chat(self, channel: str, chat_id: str) -> int:
        return len(self.list_for_chat(channel, chat_id, status="active"))

    def update(self, reminder: Reminder) -> bool:
        with exclusive_file_lock(self._lock_path()):
            reminders = self.load()
            for index, existing in enumerate(reminders):
                if existing.id == reminder.id:
                    reminders[index] = reminder
                    self._save(reminders)
                    return True
        return False

    def cancel(self, reminder_id: str) -> bool:
        return self.set_status(reminder_id, "done")

    def set_status(self, reminder_id: str, status: str) -> bool:
        with exclusive_file_lock(self._lock_path()):
            reminders = self.load()
            for reminder in reminders:
                if reminder.id == reminder_id:
                    reminder.status = status
                    self._save(reminders)
                    return True
        return False

    def mark_fired(
        self, reminder_id: str, *, next_fire_at: float | None, fired_at: float
    ) -> bool:
        """Idempotency-critical write: persist firing BEFORE delivery.

        Sets ``last_fired_at`` + bumps ``fire_count``; when ``next_fire_at`` is
        ``None`` the reminder is exhausted (one-shot or rule end) -> ``done``,
        otherwise ``next_fire_at`` advances and the reminder stays active.
        """
        with exclusive_file_lock(self._lock_path()):
            reminders = self.load()
            for reminder in reminders:
                if reminder.id == reminder_id:
                    reminder.last_fired_at = fired_at
                    reminder.fire_count += 1
                    if next_fire_at is None:
                        reminder.status = "done"
                    else:
                        reminder.next_fire_at = next_fire_at
                    self._save(reminders)
                    return True
        return False

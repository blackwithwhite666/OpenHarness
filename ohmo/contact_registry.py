"""Persistent registry of messageable ohmo contacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path

from openharness.utils.fs import atomic_write_text

from ohmo.workspace import get_contacts_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContactRecord:
    channel: str
    chat_id: str
    user_id: str | None
    username: str | None
    first_name: str | None
    display_name: str | None
    first_seen: str
    last_seen: str
    message_count: int


class ContactStore:
    """File-backed contact registry.

    The store deliberately keeps no in-memory cache: bridge writes and tool
    reads may happen through separate instances in the same process.
    """

    def __init__(self, workspace: str | Path | None = None) -> None:
        self._workspace = workspace

    def record_inbound(
        self,
        *,
        channel: str,
        chat_id: str,
        user_id: str | None = None,
        username: str | None = None,
        first_name: str | None = None,
        display_name: str | None = None,
    ) -> None:
        """Upsert a contact from an inbound message."""
        channel_value = str(channel).strip()
        chat_id_value = str(chat_id).strip()
        if not channel_value or not chat_id_value:
            return

        records = self._load()
        key = _key(channel_value, chat_id_value)
        existing = records.get(key)
        now = _now_iso_after(existing.last_seen if existing else None)

        cleaned_user_id = _clean_optional(user_id)
        cleaned_username = _clean_username(username)
        cleaned_first_name = _clean_optional(first_name)
        cleaned_display_name = _clean_optional(display_name)

        if existing is None:
            records[key] = ContactRecord(
                channel=channel_value,
                chat_id=chat_id_value,
                user_id=cleaned_user_id,
                username=cleaned_username,
                first_name=cleaned_first_name,
                display_name=cleaned_display_name,
                first_seen=now,
                last_seen=now,
                message_count=1,
            )
        else:
            records[key] = ContactRecord(
                channel=existing.channel,
                chat_id=existing.chat_id,
                user_id=cleaned_user_id or existing.user_id,
                username=cleaned_username or existing.username,
                first_name=cleaned_first_name or existing.first_name,
                display_name=cleaned_display_name or existing.display_name,
                first_seen=existing.first_seen,
                last_seen=now,
                message_count=existing.message_count + 1,
            )
        self._save(records)

    def list(self) -> list[ContactRecord]:
        """Return all contacts sorted by most recent inbound message."""
        contacts = list(self._load().values())

        def _sort_key(record: ContactRecord):
            try:
                return datetime.fromisoformat(record.last_seen)
            except (TypeError, ValueError):
                return datetime.min.replace(tzinfo=timezone.utc)

        contacts.sort(key=_sort_key, reverse=True)
        return contacts

    def get(self, channel: str, chat_id: str) -> ContactRecord | None:
        return self._load().get(_key(str(channel).strip(), str(chat_id).strip()))

    def resolve(
        self,
        query: str,
        *,
        channel: str = "telegram",
        fuzzy: bool = False,
    ) -> list[ContactRecord]:
        """Resolve a free-form recipient to candidate contacts on ``channel``."""
        normalized = _normalize_query(query)
        if not normalized:
            return []

        contacts = [record for record in self.list() if record.channel == channel]
        if _is_numeric_query(normalized):
            matches = [
                record
                for record in contacts
                if record.chat_id == normalized or record.user_id == normalized
            ]
            if matches:
                return matches

        query_lower = normalized.lower()

        matches = [
            record
            for record in contacts
            if record.username is not None and record.username.lower() == query_lower
        ]
        if matches:
            return matches

        matches = [
            record
            for record in contacts
            if _matches_exact(record.first_name, query_lower)
            or _matches_exact(record.display_name, query_lower)
        ]
        if matches:
            return matches

        if not fuzzy:
            return []

        return [
            record
            for record in contacts
            if _contains(record.username, query_lower)
            or _contains(record.first_name, query_lower)
            or _contains(record.display_name, query_lower)
        ]

    def _path(self) -> Path:
        return get_contacts_path(self._workspace)

    def _load(self) -> dict[str, ContactRecord]:
        path = self._path()
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("ohmo contacts store unreadable path=%s error=%s", path, exc)
            return {}
        if not isinstance(raw, dict):
            logger.warning("ohmo contacts store is not an object path=%s", path)
            return {}

        records: dict[str, ContactRecord] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            record = _record_from_mapping(value)
            if record is not None:
                records[_key(record.channel, record.chat_id)] = record
        return records

    def _save(self, records: dict[str, ContactRecord]) -> None:
        payload = {
            key: asdict(record)
            for key, record in sorted(records.items(), key=lambda item: item[0])
        }
        atomic_write_text(
            self._path(),
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )


def _key(channel: str, chat_id: str) -> str:
    return f"{channel}:{chat_id}"


def _clean_optional(value: object | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _clean_username(value: object | None) -> str | None:
    cleaned = _clean_optional(value)
    if cleaned is None:
        return None
    if cleaned.startswith("@"):
        cleaned = cleaned[1:].strip()
    return cleaned or None


def _normalize_query(query: str) -> str:
    normalized = str(query or "").strip()
    if normalized.startswith("@"):
        normalized = normalized[1:].strip()
    return normalized


def _is_numeric_query(value: str) -> bool:
    if value.startswith("-"):
        value = value[1:]
    return bool(value) and value.isdigit()


def _matches_exact(value: str | None, query_lower: str) -> bool:
    return value is not None and value.lower() == query_lower


def _contains(value: str | None, query_lower: str) -> bool:
    return value is not None and query_lower in value.lower()


def _now_iso_after(previous: str | None = None) -> str:
    now = datetime.now(timezone.utc)
    if previous:
        try:
            previous_dt = datetime.fromisoformat(previous)
            if now <= previous_dt:
                now = previous_dt.astimezone(timezone.utc) + timedelta(microseconds=1)
        except (TypeError, ValueError):
            pass
    return now.isoformat()


def _record_from_mapping(data: dict[str, object]) -> ContactRecord | None:
    channel = _clean_optional(data.get("channel"))
    chat_id = _clean_optional(data.get("chat_id"))
    first_seen = _clean_optional(data.get("first_seen"))
    last_seen = _clean_optional(data.get("last_seen"))
    if not channel or not chat_id or not first_seen or not last_seen:
        return None
    try:
        message_count = int(data.get("message_count") or 0)
    except (TypeError, ValueError):
        return None
    if message_count < 0:
        return None
    return ContactRecord(
        channel=channel,
        chat_id=chat_id,
        user_id=_clean_optional(data.get("user_id")),
        username=_clean_username(data.get("username")),
        first_name=_clean_optional(data.get("first_name")),
        display_name=_clean_optional(data.get("display_name")),
        first_seen=first_seen,
        last_seen=last_seen,
        message_count=message_count,
    )

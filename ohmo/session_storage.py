"""Session persistence for ``ohmo``."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import stat
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, sanitize_conversation_messages
from openharness.engine.messages import AttachmentRefBlock
from openharness.services.session_backend import SessionBackend
from openharness.services.session_storage import (
    _persistable_tool_metadata,
    _sanitize_snapshot_payload,
)
from openharness.utils.fs import atomic_write_text

from ohmo.attachment_store import AttachmentStore
from ohmo.workspace import get_sessions_dir, get_work_dir


def _snapshot_message_dict(message: ConversationMessage) -> dict[str, Any]:
    """Serialize a conversation message, retaining gateway-only ref provenance."""
    result = message.model_dump(mode="json")
    for index, block in enumerate(message.content):
        if isinstance(block, AttachmentRefBlock) and block.source_provenance is not None:
            result["content"][index]["source_provenance"] = block.source_provenance
    return result


def get_session_dir(workspace: str | Path | None = None) -> Path:
    """Return the ohmo sessions directory."""
    session_dir = get_sessions_dir(workspace)
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _session_key_token(session_key: str) -> str:
    return hashlib.sha1(session_key.encode("utf-8")).hexdigest()[:12]


def get_session_work_dir(session_key: str, workspace: str | Path | None = None) -> Path:
    """Per-chat scratch dir used as the agent cwd for unbound chats.

    Isolates each chat's transient output (diagrams, downloads, scratch) instead
    of sharing the workspace root across every chat. Lazily created; wiped on
    ``/new`` and reaped by the startup TTL sweep.
    """
    work = get_work_dir(workspace) / _session_key_token(session_key)
    work.mkdir(parents=True, exist_ok=True)
    return work


def clear_session_work_dir(session_key: str, workspace: str | Path | None = None) -> None:
    """Remove a chat's work dir — called on ``/new`` for a clean slate.

    The dir is recreated lazily on the next write, so nothing breaks if the chat
    continues."""
    shutil.rmtree(get_work_dir(workspace) / _session_key_token(session_key), ignore_errors=True)


def reap_stale_work_dirs(workspace: str | Path | None = None, max_age_s: float = 7 * 86400) -> int:
    """Remove per-chat work dirs untouched for ``max_age_s`` (default 7 days).

    Run once at gateway start so abandoned chats' scratch dirs don't accumulate.
    Returns the number reaped. Stale-but-live chats simply get a fresh dir on
    their next write."""
    root = get_work_dir(workspace)
    if not root.is_dir():
        return 0
    cutoff = time.time() - max_age_s
    reaped = 0
    for child in root.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                reaped += 1
        except OSError:
            continue
    return reaped


def _session_key_latest_path(workspace: str | Path | None, session_key: str) -> Path:
    session_dir = get_session_dir(workspace)
    token = _session_key_token(session_key)
    return session_dir / f"latest-{token}.json"


def clear_session_key(workspace: str | Path | None, session_key: str) -> None:
    """Drop the per-session-key 'latest' pointer so the next message for this
    session starts a brand-new conversation (used by /new). The historical
    session-<id>.json snapshots are left on disk; only the 'resume here' pointer
    is removed."""
    try:
        _session_key_latest_path(workspace, session_key).unlink()
    except FileNotFoundError:
        pass


def save_session_snapshot(
    *,
    cwd: str | Path,
    workspace: str | Path | None = None,
    model: str,
    system_prompt: str,
    messages: list[ConversationMessage],
    usage: UsageSnapshot,
    session_id: str | None = None,
    session_key: str | None = None,
    tool_metadata: dict[str, object] | None = None,
) -> Path:
    """Persist the latest ohmo session snapshot."""
    session_dir = get_session_dir(workspace)
    sid = session_id or uuid4().hex[:12]
    now = time.time()
    attachment_store = AttachmentStore(workspace)
    messages = attachment_store.externalize_messages(
        sanitize_conversation_messages(messages)
    )
    attachment_store.assert_externalized(messages)
    summary = ""
    for msg in messages:
        if msg.role == "user" and msg.text.strip():
            summary = msg.text.strip()[:80]
            break

    payload = {
        "app": "ohmo",
        "session_id": sid,
        "session_key": session_key,
        "cwd": str(Path(cwd).resolve()),
        "model": model,
        "system_prompt": system_prompt,
        "messages": [_snapshot_message_dict(message) for message in messages],
        "usage": usage.model_dump(),
        "tool_metadata": _persistable_tool_metadata(tool_metadata),
        "created_at": now,
        "summary": summary,
        "message_count": len(messages),
    }
    data = json.dumps(payload, indent=2) + "\n"
    latest_path = session_dir / "latest.json"
    atomic_write_text(latest_path, data)
    if session_key:
        atomic_write_text(_session_key_latest_path(workspace, session_key), data)
    session_path = session_dir / f"session-{sid}.json"
    atomic_write_text(session_path, data)
    return latest_path


def _externalize_snapshot_payload(
    payload: dict[str, Any],
    workspace: str | Path | None,
) -> dict[str, Any]:
    # The generic engine serializer intentionally drops gateway-private fields.
    # Validate/sanitize the public message structure here, then use the Ohmo
    # snapshot serializer so attachment provenance survives a normal read.
    raw_messages = payload.get("messages", [])
    if not isinstance(raw_messages, list):
        return _sanitize_snapshot_payload(payload)
    sanitized = _sanitize_snapshot_payload({**payload, "messages": []})
    messages = sanitize_conversation_messages(
        [ConversationMessage.model_validate(item) for item in raw_messages]
    )
    store = AttachmentStore(workspace)
    messages = store.externalize_messages(messages)
    store.assert_externalized(messages)
    sanitized = dict(sanitized)
    sanitized["messages"] = [_snapshot_message_dict(message) for message in messages]
    sanitized["message_count"] = len(messages)
    return sanitized


def load_latest(workspace: str | Path | None = None) -> dict[str, Any] | None:
    path = get_session_dir(workspace) / "latest.json"
    if not path.exists():
        return None
    return _externalize_snapshot_payload(
        json.loads(path.read_text(encoding="utf-8")), workspace
    )


def load_latest_for_session_key(workspace: str | Path | None, session_key: str) -> dict[str, Any] | None:
    path = _session_key_latest_path(workspace, session_key)
    if path.exists():
        return _externalize_snapshot_payload(
            json.loads(path.read_text(encoding="utf-8")), workspace
        )
    return None


def load_bounded_latest_for_session_key(
    workspace: str | Path | None,
    session_key: str,
    *,
    max_bytes: int = 64 * 1024 * 1024,
    max_messages: int | None = None,
) -> dict[str, Any] | None:
    """Read only the exact session-key snapshot under explicit size limits."""
    if not session_key or len(session_key) > 512 or not 1 <= max_bytes <= 64 * 1024 * 1024:
        raise ValueError("invalid bounded snapshot query")
    if max_messages is not None and not 1 <= max_messages <= 10000:
        raise ValueError("invalid snapshot message bound")
    path = get_sessions_dir(workspace) / f"latest-{_session_key_token(session_key)}.json"
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= max_bytes:
            raise ValueError("session snapshot is not a bounded regular file")
        data = bytearray()
        while len(data) <= max_bytes:
            chunk = os.read(fd, min(65536, max_bytes + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(fd)
        if (
            len(data) != before.st_size
            or len(data) > max_bytes
            or (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size)
            != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size)
        ):
            raise ValueError("session snapshot changed during read")
    finally:
        os.close(fd)
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("session snapshot JSON is invalid") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("app") != "ohmo"
        or payload.get("session_key") != session_key
        or not isinstance(payload.get("session_id"), str)
        or not payload["session_id"]
        or not isinstance(payload.get("messages"), list)
        or (max_messages is not None and len(payload["messages"]) > max_messages)
    ):
        raise ValueError("session snapshot identity or message bounds are invalid")
    return payload


class _SnapshotJSONReader:
    """Incrementally decode JSON values from a stable snapshot file."""

    def __init__(self, stream):
        self.stream = stream
        self.buffer = ""
        self.position = 0
        self.decoder = json.JSONDecoder(object_pairs_hook=_unique_json_pairs)
        self.eof = False

    def _fill(self) -> None:
        # A larger fixed chunk prevents repeatedly rescanning a long JSON text
        # token while still bounding the traversal reader's scratch buffer.
        chunk = self.stream.read(4 * 1024 * 1024)
        if chunk:
            self.buffer = self.buffer[self.position:] + chunk
            self.position = 0
        else:
            self.eof = True

    def whitespace(self) -> None:
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer) or self.eof:
                return
            self._fill()

    def punctuation(self, expected: str) -> None:
        self.whitespace()
        if self.position >= len(self.buffer) or self.buffer[self.position] != expected:
            raise ValueError("session snapshot JSON is invalid")
        self.position += 1

    def value(self):
        self.whitespace()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
                self.position = end
                return value
            except json.JSONDecodeError as exc:
                if self.eof:
                    raise ValueError("session snapshot JSON is invalid") from exc
                self._fill()

    def messages(self) -> list[dict[str, Any]]:
        self.punctuation("[")
        selected: list[dict[str, Any]] = []
        self.whitespace()
        if self.position < len(self.buffer) and self.buffer[self.position] == "]":
            self.position += 1
            return selected
        while True:
            value = self.value()
            if not isinstance(value, dict):
                raise ValueError("session snapshot message is invalid")
            content = value.get("content")
            if isinstance(content, list) and any(
                isinstance(block, dict)
                and block.get("type") in {"attachment_ref", "image"}
                for block in content
            ):
                selected.append(value)
            self.whitespace()
            if self.position < len(self.buffer) and self.buffer[self.position] == ",":
                self.position += 1
                continue
            self.punctuation("]")
            return selected


def _unique_json_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate session snapshot JSON key")
        result[key] = value
    return result


def load_camera_attachment_snapshot(
    workspace: str | Path | None, session_key: str
) -> dict[str, Any] | None:
    """Read the exact configured snapshot, retaining only attachment messages.

    Ordinary dialogue is traversed and discarded one message at a time; there
    is no all-time text or snapshot-byte ceiling. JSON tail and stable-file
    identity are checked before returning a complete projection.
    """
    if not session_key or len(session_key) > 512:
        raise ValueError("invalid camera snapshot query")
    path = get_sessions_dir(workspace) / f"latest-{_session_key_token(session_key)}.json"
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise ValueError("session snapshot is not a regular file")
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            fd = -1
            reader = _SnapshotJSONReader(stream)
            reader.punctuation("{")
            fields: dict[str, Any] = {}
            messages = None
            reader.whitespace()
            while reader.position < len(reader.buffer) and reader.buffer[reader.position] != "}":
                key = reader.value()
                if not isinstance(key, str) or key in fields:
                    raise ValueError("session snapshot top-level key is invalid")
                reader.punctuation(":")
                value = reader.messages() if key == "messages" else reader.value()
                fields[key] = value if key != "messages" else True
                if key == "messages":
                    messages = value
                reader.whitespace()
                if reader.position < len(reader.buffer) and reader.buffer[reader.position] == ",":
                    reader.position += 1
                    continue
                reader.punctuation("}")
                break
            reader.whitespace()
            if reader.position < len(reader.buffer) or not reader.eof:
                # Force EOF so trailing bytes are never mistaken for completion.
                while not reader.eof:
                    reader._fill()
                reader.whitespace()
            if reader.position != len(reader.buffer) or messages is None:
                raise ValueError("session snapshot has an invalid trailing JSON value")
            after = os.fstat(stream.fileno())
            current_path = os.stat(path, follow_symlinks=False)
        if (
            (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns, before.st_size)
            != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns, after.st_size)
            or (before.st_dev, before.st_ino)
            != (current_path.st_dev, current_path.st_ino)
        ):
            raise ValueError("session snapshot changed during traversal")
    finally:
        if fd >= 0:
            os.close(fd)
    if (
        fields.get("app") != "ohmo"
        or fields.get("session_key") != session_key
        or not isinstance(fields.get("session_id"), str)
        or not fields["session_id"]
    ):
        raise ValueError("session snapshot identity is invalid")
    snapshot_identity = hashlib.sha256(
        f"{before.st_dev}:{before.st_ino}:{before.st_mtime_ns}:{before.st_ctime_ns}:{before.st_size}".encode()
    ).hexdigest()
    return {
        "session_id": fields["session_id"],
        "session_key": session_key,
        "snapshot_identity": snapshot_identity,
        "messages": messages,
    }


def list_snapshots(workspace: str | Path | None = None, limit: int = 20) -> list[dict[str, Any]]:
    session_dir = get_session_dir(workspace)
    sessions: list[dict[str, Any]] = []
    for path in sorted(session_dir.glob("session-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        sessions.append(
            {
                "session_id": data.get("session_id", path.stem.replace("session-", "")),
                "summary": data.get("summary", ""),
                "message_count": data.get("message_count", len(data.get("messages", []))),
                "model": data.get("model", ""),
                "created_at": data.get("created_at", path.stat().st_mtime),
            }
        )
        if len(sessions) >= limit:
            break
    return sessions


def load_by_id(workspace: str | Path | None, session_id: str) -> dict[str, Any] | None:
    path = get_session_dir(workspace) / f"session-{session_id}.json"
    if path.exists():
        return _externalize_snapshot_payload(
            json.loads(path.read_text(encoding="utf-8")), workspace
        )
    latest = load_latest(workspace)
    if latest and (latest.get("session_id") == session_id or session_id == "latest"):
        return latest
    return None


def export_session_markdown(
    *,
    cwd: str | Path,
    workspace: str | Path | None = None,
    messages: list[ConversationMessage],
) -> Path:
    path = get_session_dir(workspace) / "transcript.md"
    parts = ["# ohmo Session Transcript"]
    for message in messages:
        parts.append(f"\n## {message.role.capitalize()}\n")
        text = message.text.strip()
        if text:
            parts.append(text)
    atomic_write_text(path, "\n".join(parts).strip() + "\n")
    return path


class OhmoSessionBackend(SessionBackend):
    """Session backend rooted in ``.ohmo/sessions``."""

    def __init__(self, workspace: str | Path | None = None) -> None:
        self._workspace = workspace
        self._attachment_store = AttachmentStore(workspace)

    @property
    def attachment_store(self) -> AttachmentStore:
        """Return the store shared by snapshot and active-turn boundaries."""
        return self._attachment_store

    def get_session_dir(self, cwd: str | Path) -> Path:
        return get_session_dir(self._workspace)

    def save_snapshot(
        self,
        *,
        cwd: str | Path,
        model: str,
        system_prompt: str,
        messages: list[ConversationMessage],
        usage: UsageSnapshot,
        session_id: str | None = None,
        session_key: str | None = None,
        tool_metadata: dict[str, object] | None = None,
    ) -> Path:
        return save_session_snapshot(
            cwd=cwd,
            workspace=self._workspace,
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            usage=usage,
            session_id=session_id,
            session_key=session_key,
            tool_metadata=tool_metadata,
        )

    def load_latest(self, cwd: str | Path) -> dict[str, Any] | None:
        return load_latest(self._workspace)

    def list_snapshots(self, cwd: str | Path, limit: int = 20) -> list[dict[str, Any]]:
        return list_snapshots(self._workspace, limit=limit)

    def load_by_id(self, cwd: str | Path, session_id: str) -> dict[str, Any] | None:
        return load_by_id(self._workspace, session_id)

    def load_latest_for_session_key(self, session_key: str) -> dict[str, Any] | None:
        return load_latest_for_session_key(self._workspace, session_key)

    def load_bounded_latest_for_session_key(self, session_key: str) -> dict[str, Any] | None:
        return load_bounded_latest_for_session_key(self._workspace, session_key)

    def load_camera_attachment_snapshot(self, session_key: str) -> dict[str, Any] | None:
        return load_camera_attachment_snapshot(self._workspace, session_key)

    def clear_session_key(self, session_key: str) -> None:
        clear_session_key(self._workspace, session_key)

    def export_markdown(
        self,
        *,
        cwd: str | Path,
        messages: list[ConversationMessage],
    ) -> Path:
        return export_session_markdown(cwd=cwd, workspace=self._workspace, messages=messages)

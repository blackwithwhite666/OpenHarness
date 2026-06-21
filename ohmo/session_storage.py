"""Session persistence for ``ohmo``."""

from __future__ import annotations

import json
import hashlib
import shutil
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, sanitize_conversation_messages
from openharness.services.session_backend import SessionBackend
from openharness.services.session_storage import (
    _persistable_tool_metadata,
    _sanitize_snapshot_payload,
)
from openharness.utils.fs import atomic_write_text

from ohmo.workspace import get_sessions_dir, get_work_dir


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
    messages = sanitize_conversation_messages(messages)
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
        "messages": [message.model_dump(mode="json") for message in messages],
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


def load_latest(workspace: str | Path | None = None) -> dict[str, Any] | None:
    path = get_session_dir(workspace) / "latest.json"
    if not path.exists():
        return None
    return _sanitize_snapshot_payload(json.loads(path.read_text(encoding="utf-8")))


def load_latest_for_session_key(workspace: str | Path | None, session_key: str) -> dict[str, Any] | None:
    path = _session_key_latest_path(workspace, session_key)
    if path.exists():
        return _sanitize_snapshot_payload(json.loads(path.read_text(encoding="utf-8")))
    return None


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
        return _sanitize_snapshot_payload(json.loads(path.read_text(encoding="utf-8")))
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

    def clear_session_key(self, session_key: str) -> None:
        clear_session_key(self._workspace, session_key)

    def export_markdown(
        self,
        *,
        cwd: str | Path,
        messages: list[ConversationMessage],
    ) -> Path:
        return export_session_markdown(cwd=cwd, workspace=self._workspace, messages=messages)

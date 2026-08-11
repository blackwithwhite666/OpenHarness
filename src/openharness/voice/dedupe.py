"""Restart-durable, bounded deduplication for downloaded voice ASR."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openharness.voice.transcription import TranscriptionError

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1
_DEFAULT_MAX_ENTRIES = 512
_DEFAULT_TTL_SECONDS = 30 * 24 * 60 * 60
_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass
class _InflightTranscription:
    task: asyncio.Task[str]
    waiters: int = 0


def _dedupe_key_from_hash(chat_id: object, message_id: object, file_hash: str) -> str:
    encoded = json.dumps(
        [str(chat_id), str(message_id), file_hash],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def voice_dedupe_key(chat_id: object, message_id: object, audio_bytes: bytes) -> str:
    """Return a deterministic digest of exactly the required key tuple."""
    file_hash = hashlib.sha256(audio_bytes).hexdigest()
    return _dedupe_key_from_hash(chat_id, message_id, file_hash)


async def voice_dedupe_key_for_path(
    chat_id: object, message_id: object, path: str | Path
) -> str:
    """Return the same key as :func:`voice_dedupe_key` without blocking the loop."""
    file_hash = await asyncio.to_thread(_sha256_path, Path(path))
    return _dedupe_key_from_hash(chat_id, message_id, file_hash)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as audio_file:
        while chunk := audio_file.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


class VoiceTranscriptionDedupe:
    """Coalesce in-flight keys and persist completed results atomically."""

    def __init__(
        self,
        state_dir: Path | None,
        *,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._state_dir = Path(state_dir) if state_dir is not None else None
        self._path: Path | None = None
        if self._state_dir is not None:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(self._state_dir, 0o700)
            self._path = self._state_dir / "voice_transcriptions.json"
        self._max_entries = max_entries
        self._ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._lock = asyncio.Lock()
        self._inflight: dict[str, _InflightTranscription] = {}
        self._completed: dict[str, dict[str, Any]] = self._load()
        self._prune()

    async def transcribe(
        self,
        key: str,
        operation: Callable[[], Awaitable[str]],
    ) -> str:
        async with self._lock:
            cached = self._completed.get(key)
            if cached is not None:
                if self._clock() - float(cached["completed_at"]) <= self._ttl_seconds:
                    return _cached_result(cached)
                self._completed.pop(key, None)
                self._persist_safely()
            entry = self._inflight.get(key)
            if entry is None:
                entry = _InflightTranscription(asyncio.create_task(self._run(key, operation)))
                self._inflight[key] = entry
            entry.waiters += 1
        try:
            return await asyncio.shield(entry.task)
        finally:
            await self._release_waiter(key, entry)

    async def _release_waiter(self, key: str, entry: _InflightTranscription) -> None:
        cancel_task = False
        async with self._lock:
            entry.waiters -= 1
            if entry.waiters == 0 and not entry.task.done():
                if self._inflight.get(key) is entry:
                    self._inflight.pop(key, None)
                entry.task.cancel()
                cancel_task = True
        if cancel_task or entry.task.done():
            with contextlib.suppress(BaseException):
                await entry.task

    async def _run(self, key: str, operation: Callable[[], Awaitable[str]]) -> str:
        try:
            result = await operation()
        except asyncio.CancelledError:
            raise
        except TranscriptionError as transcription_error:
            self._completed[key] = {
                "completed_at": self._clock(),
                "ok": False,
                "provider": transcription_error.provider,
                "error_class": transcription_error.error_class,
                "retryable": transcription_error.retryable,
                "status_code": transcription_error.status_code,
            }
            self._persist_safely()
            raise
        except Exception as exc:
            typed_error = TranscriptionError(
                "transcription provider failed",
                provider="unknown",
                error_class="provider_failure",
                retryable=False,
            )
            self._completed[key] = {
                "completed_at": self._clock(),
                "ok": False,
                "provider": typed_error.provider,
                "error_class": typed_error.error_class,
                "retryable": typed_error.retryable,
                "status_code": typed_error.status_code,
            }
            self._persist_safely()
            raise typed_error from exc
        else:
            self._completed[key] = {
                "completed_at": self._clock(),
                "ok": True,
                "text": result,
            }
            self._persist_safely()
            return result
        finally:
            entry = self._inflight.get(key)
            if entry is not None and entry.task is asyncio.current_task():
                self._inflight.pop(key, None)

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._path is None:
            return {}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("invalid cache payload")
            entries = payload.get("entries")
            if payload.get("version") != _CACHE_VERSION or not isinstance(entries, dict):
                raise ValueError("invalid cache version")
            return {
                str(key): value
                for key, value in entries.items()
                if isinstance(key, str) and _valid_record(value)
            }
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            logger.warning("telegram voice transcription cache unreadable; ignoring cache")
            return {}

    def _prune(self) -> None:
        now = self._clock()
        fresh = {
            key: value
            for key, value in self._completed.items()
            if now - float(value["completed_at"]) <= self._ttl_seconds
        }
        if len(fresh) > self._max_entries:
            keys = sorted(
                fresh,
                key=lambda key: float(fresh[key]["completed_at"]),
                reverse=True,
            )[: self._max_entries]
            fresh = {key: fresh[key] for key in keys}
        changed = len(fresh) != len(self._completed) or set(fresh) != set(self._completed)
        self._completed = fresh
        if changed:
            self._persist_safely()

    def _persist_safely(self) -> None:
        if self._path is None or self._state_dir is None:
            return
        self._prune_in_memory()
        payload = json.dumps(
            {"version": _CACHE_VERSION, "entries": self._completed},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        fd: int | None = None
        temp_path: str | None = None
        try:
            fd, temp_path = tempfile.mkstemp(prefix=".voice-transcriptions.", dir=self._state_dir)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as output:
                fd = None
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp_path, self._path)
            temp_path = None
            os.chmod(self._path, 0o600)
        except OSError:
            logger.warning("telegram voice transcription cache persistence unavailable")
        finally:
            if fd is not None:
                os.close(fd)
            if temp_path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(temp_path)

    def _prune_in_memory(self) -> None:
        now = self._clock()
        self._completed = {
            key: value
            for key, value in self._completed.items()
            if now - float(value["completed_at"]) <= self._ttl_seconds
        }
        if len(self._completed) > self._max_entries:
            keys = sorted(
                self._completed,
                key=lambda key: float(self._completed[key]["completed_at"]),
                reverse=True,
            )[: self._max_entries]
            self._completed = {key: self._completed[key] for key in keys}


def _valid_record(value: object) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("completed_at"), (int, float)):
        return False
    if value.get("ok") is True:
        return isinstance(value.get("text"), str)
    return (
        value.get("ok") is False
        and isinstance(value.get("provider"), str)
        and len(value["provider"]) <= 64
        and isinstance(value.get("error_class"), str)
        and bool(value["error_class"])
        and len(value["error_class"]) <= 64
        and isinstance(value.get("retryable"), bool)
        and (
            value.get("status_code") is None
            or (
                isinstance(value.get("status_code"), int)
                and not isinstance(value.get("status_code"), bool)
                and 100 <= value["status_code"] <= 599
            )
        )
    )


def _cached_result(record: dict[str, Any]) -> str:
    if record.get("ok") is True:
        return str(record["text"])
    raise TranscriptionError(
        "cached transcription failure",
        provider=str(record.get("provider", "unknown")),
        error_class=str(record.get("error_class", "unknown")),
        retryable=bool(record.get("retryable", False)),
        status_code=record.get("status_code"),
    )

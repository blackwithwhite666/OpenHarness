"""Wire primitives and capability authentication for the memory service."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import secrets
import stat
import struct
import time
from pathlib import Path
from typing import Any, Mapping

from ohmo.memory_backend import MemoryHit
from ohmo.memory_store import MemoryEntry, MemoryOpResult

_FRAME_HEADER = struct.Struct("!I")
_MAX_FRAME_BYTES = 4 * 1024 * 1024


class MemoryServiceProtocolError(ValueError):
    """Raised when a peer sends a malformed or oversized frame."""


def memory_entry_to_dict(entry: MemoryEntry) -> dict[str, object]:
    """Serialize a memory entry into JSON-compatible values."""
    return {
        "name": entry.name,
        "slug": entry.slug,
        "title": entry.title,
        "content": entry.content,
        "path": str(entry.path),
    }


def memory_entry_from_dict(payload: Mapping[str, object]) -> MemoryEntry:
    """Reconstruct a memory entry from a wire payload."""
    return MemoryEntry(
        name=_required_str(payload, "name"),
        slug=_required_str(payload, "slug"),
        title=_required_str(payload, "title"),
        content=_required_str(payload, "content"),
        path=Path(_required_str(payload, "path")),
    )


def memory_op_result_to_dict(result: MemoryOpResult) -> dict[str, object]:
    """Serialize a mutation result, including optional overflow entries."""
    entries = result.entries
    return {
        "ok": result.ok,
        "message": result.message,
        "entries": None if entries is None else [memory_entry_to_dict(entry) for entry in entries],
    }


def memory_op_result_from_dict(payload: Mapping[str, object]) -> MemoryOpResult:
    """Reconstruct a mutation result from a wire payload."""
    ok = payload.get("ok")
    if not isinstance(ok, bool):
        raise MemoryServiceProtocolError("memory operation result has invalid ok")
    raw_entries = payload.get("entries")
    if raw_entries is None:
        entries = None
    elif isinstance(raw_entries, list):
        entries = tuple(memory_entry_from_dict(_mapping(item)) for item in raw_entries)
    else:
        raise MemoryServiceProtocolError("memory operation result has invalid entries")
    return MemoryOpResult(
        ok=ok,
        message=_required_str(payload, "message"),
        entries=entries,
    )


def memory_hit_to_dict(hit: MemoryHit) -> dict[str, object]:
    """Serialize one ranked memory search hit."""
    return {
        "name": hit.name,
        "title": hit.title,
        "snippet": hit.snippet,
        "rank": hit.rank,
    }


def memory_hit_from_dict(payload: Mapping[str, object]) -> MemoryHit:
    """Reconstruct one ranked memory search hit."""
    rank = payload.get("rank")
    if not isinstance(rank, int) or isinstance(rank, bool):
        raise MemoryServiceProtocolError("memory hit has invalid rank")
    return MemoryHit(
        name=_required_str(payload, "name"),
        title=_required_str(payload, "title"),
        snippet=_required_str(payload, "snippet"),
        rank=rank,
    )


def mint_capability(secret: bytes, op: str, ttl_s: int) -> str:
    """Mint a short-lived operation-bound HMAC capability token."""
    if not secret:
        raise ValueError("capability secret must not be empty")
    if not isinstance(op, str) or not op or "|" in op:
        raise ValueError("capability operation must be a non-empty string without '|'")
    if not isinstance(ttl_s, int) or isinstance(ttl_s, bool):
        raise TypeError("capability ttl must be an integer")

    nonce = secrets.token_hex(16)
    expiry = int(time.time()) + ttl_s
    payload = f"{op}|{nonce}|{expiry}".encode("utf-8")
    signature = hmac.new(secret, payload, hashlib.sha256).digest()
    return f"{_b64encode(payload)}.{_b64encode(signature)}"


def verify_capability(
    secret: bytes,
    token: str,
    op: str,
    *,
    now: float | None = None,
) -> bool:
    """Validate a capability's signature, operation binding, and expiry."""
    if not secret or not isinstance(token, str) or not isinstance(op, str):
        return False
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        payload = _b64decode(encoded_payload)
        supplied_signature = _b64decode(encoded_signature)
        expected_signature = hmac.new(secret, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(supplied_signature, expected_signature):
            return False
        token_op, nonce, raw_expiry = payload.decode("utf-8").split("|", 2)
        if not nonce or not hmac.compare_digest(token_op, op):
            return False
        expiry = int(raw_expiry)
    except (UnicodeDecodeError, ValueError, TypeError):
        return False
    current_time = time.time() if now is None else now
    return current_time < expiry


def load_secret_file(secret_file: str | Path) -> bytes:
    """Read a non-empty capability secret from an exact-mode 0600 file."""
    path = Path(secret_file).expanduser()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open memory service secret file {path}: {error}") from error
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"memory service secret file must be regular: {path}")
        mode = stat.S_IMODE(metadata.st_mode)
        if mode != 0o600:
            raise ValueError(
                f"memory service secret file must have mode 0600, got {mode:04o}: {path}"
            )
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            secret = handle.read()
    finally:
        if fd >= 0:
            os.close(fd)
    if not secret:
        raise ValueError(f"memory service secret file is empty: {path}")
    return secret


def encode_frame(payload: Mapping[str, object]) -> bytes:
    """Encode one length-prefixed JSON object."""
    try:
        body = json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MemoryServiceProtocolError(f"frame is not JSON serializable: {error}") from error
    if len(body) > _MAX_FRAME_BYTES:
        raise MemoryServiceProtocolError("frame exceeds maximum size")
    return _FRAME_HEADER.pack(len(body)) + body


async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Read and decode one length-prefixed JSON object."""
    header = await reader.readexactly(_FRAME_HEADER.size)
    (length,) = _FRAME_HEADER.unpack(header)
    if length > _MAX_FRAME_BYTES:
        raise MemoryServiceProtocolError("frame exceeds maximum size")
    try:
        payload = json.loads((await reader.readexactly(length)).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MemoryServiceProtocolError(f"invalid JSON frame: {error}") from error
    if not isinstance(payload, dict):
        raise MemoryServiceProtocolError("frame payload must be a JSON object")
    return payload


async def write_frame(
    writer: asyncio.StreamWriter,
    payload: Mapping[str, object],
) -> None:
    """Write and drain one length-prefixed JSON object."""
    writer.write(encode_frame(payload))
    await writer.drain()


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise MemoryServiceProtocolError("expected a JSON object")
    return value


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise MemoryServiceProtocolError(f"field {key!r} must be a string")
    return value


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.b64decode(value + padding, altchars=b"-_", validate=True)


__all__ = [
    "MemoryServiceProtocolError",
    "encode_frame",
    "load_secret_file",
    "memory_entry_from_dict",
    "memory_entry_to_dict",
    "memory_hit_from_dict",
    "memory_hit_to_dict",
    "memory_op_result_from_dict",
    "memory_op_result_to_dict",
    "mint_capability",
    "read_frame",
    "verify_capability",
    "write_frame",
]

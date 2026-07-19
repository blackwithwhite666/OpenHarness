"""Async MemoryBackend client for the local memory-service socket."""

from __future__ import annotations

import asyncio
import builtins
import contextlib
from pathlib import Path
from typing import Mapping

from ohmo.memory_backend import MemoryBackend, MemoryHit
from ohmo.memory_service.protocol import (
    MemoryServiceProtocolError,
    load_secret_file,
    memory_entry_from_dict,
    memory_hit_from_dict,
    memory_op_result_from_dict,
    mint_capability,
    read_frame,
    write_frame,
)
from ohmo.memory_store import MemoryEntry, MemoryOpResult


class MemoryServiceError(RuntimeError):
    """Raised when the memory service rejects or malforms an RPC."""


class MemoryServiceClient(MemoryBackend):
    """Thin per-request client for the capability-authorized AF_UNIX service."""

    def __init__(
        self,
        socket_path: str | Path,
        secret_file: str | Path,
        *,
        capability_ttl_s: int = 30,
    ) -> None:
        if capability_ttl_s <= 0:
            raise ValueError("capability_ttl_s must be positive")
        self._socket_path = Path(socket_path).expanduser()
        self._secret = load_secret_file(secret_file)
        self._capability_ttl_s = capability_ttl_s

    async def list(self) -> list[MemoryEntry]:
        result = await self._request("list", {})
        return [memory_entry_from_dict(_mapping(item)) for item in _list(result)]

    async def get(self, name: str) -> MemoryEntry | None:
        result = await self._request("get", {"name": name})
        if result is None:
            return None
        return memory_entry_from_dict(_mapping(result))

    async def record_use(self, name: str) -> None:
        result = await self._request("record_use", {"name": name})
        if result is not None:
            raise MemoryServiceProtocolError("record_use result must be null")

    async def search(self, query: str, top_k: int) -> builtins.list[MemoryHit]:
        result = await self._request("search", {"query": query, "top_k": top_k})
        return [memory_hit_from_dict(_mapping(item)) for item in _list(result)]

    async def add(self, title: str, content: str) -> MemoryOpResult:
        result = await self._request("add", {"title": title, "content": content})
        return memory_op_result_from_dict(_mapping(result))

    async def update(
        self,
        name: str,
        content: str,
        *,
        title: str | None = None,
    ) -> MemoryOpResult:
        result = await self._request(
            "update",
            {"name": name, "content": content, "title": title},
        )
        return memory_op_result_from_dict(_mapping(result))

    async def remove(self, name: str) -> MemoryOpResult:
        result = await self._request("remove", {"name": name})
        return memory_op_result_from_dict(_mapping(result))

    async def render_prompt(self, budget: int | None = None) -> str:
        result = await self._request("render_prompt", {"budget": budget})
        if not isinstance(result, str):
            raise MemoryServiceProtocolError("render_prompt result must be a string")
        return result

    async def append_turn(self, role: str, text: str) -> None:
        result = await self._request("append_turn", {"role": role, "text": text})
        if result is not None:
            raise MemoryServiceProtocolError("append_turn result must be null")

    async def _request(self, op: str, args: Mapping[str, object]) -> object:
        token = mint_capability(self._secret, op, self._capability_ttl_s)
        reader, writer = await asyncio.open_unix_connection(path=str(self._socket_path))
        try:
            await write_frame(writer, {"token": token, "op": op, "args": dict(args)})
            response = await read_frame(reader)
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError, BrokenPipeError):
                await writer.wait_closed()

        ok = response.get("ok")
        if ok is not True:
            error = response.get("error")
            message = error if isinstance(error, str) else "malformed error response"
            raise MemoryServiceError(message)
        if "result" not in response:
            raise MemoryServiceProtocolError("successful response is missing result")
        return response["result"]


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise MemoryServiceProtocolError("expected a JSON object")
    return value


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise MemoryServiceProtocolError("expected a JSON array")
    return value


__all__ = ["MemoryServiceClient", "MemoryServiceError"]

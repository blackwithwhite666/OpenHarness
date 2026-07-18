"""Async storage seam for ohmo personal memory."""

from __future__ import annotations

import asyncio
import builtins
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from ohmo.memory import load_memory_prompt as load_ohmo_memory_prompt
from ohmo.memory_store import MemoryEntry, MemoryOpResult, MemoryStore
from ohmo.memory_tool import _search_memory

if TYPE_CHECKING:
    from ohmo.gateway.models import GatewayConfig


@dataclass(frozen=True)
class MemoryHit:
    """One ranked memory-search result."""

    name: str
    title: str
    snippet: str
    rank: int


class MemoryBackend(Protocol):
    """Storage-neutral async interface for model-facing memory operations."""

    async def list(self) -> list[MemoryEntry]: ...

    async def get(self, name: str) -> MemoryEntry | None: ...

    async def search(self, query: str, top_k: int) -> builtins.list[MemoryHit]: ...

    async def add(self, title: str, content: str) -> MemoryOpResult: ...

    async def update(self, name: str, content: str) -> MemoryOpResult: ...

    async def remove(self, name: str) -> MemoryOpResult: ...

    async def render_prompt(self, budget: int | None = None) -> str: ...

    async def append_turn(self, role: str, text: str) -> None: ...


class FileMemoryBackend(MemoryBackend):
    """Async adapter over the existing workspace-scoped file memory."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def list(self) -> list[MemoryEntry]:
        return await asyncio.to_thread(self._store.list)

    async def get(self, name: str) -> MemoryEntry | None:
        return await asyncio.to_thread(self._store.get, name)

    async def search(self, query: str, top_k: int) -> builtins.list[MemoryHit]:
        result = await _search_memory(query, top_k)
        if result.is_error:
            return []

        names = result.metadata.get("memory_search_hits")
        lines = result.output.splitlines()[1:]
        if not isinstance(names, list) or len(names) != len(lines):
            return []

        hits: list[MemoryHit] = []
        for rank, (name, line) in enumerate(zip(names, lines, strict=True), start=1):
            if not isinstance(name, str):
                return []
            prefix = f"- {name} (score "
            if not line.startswith(prefix):
                return []
            _, separator, snippet = line[len(prefix) :].partition("): ")
            if not separator:
                return []
            entry = await asyncio.to_thread(self._store.get, name)
            hits.append(
                MemoryHit(
                    name=name,
                    title=entry.title if entry is not None else name,
                    snippet=snippet,
                    rank=rank,
                )
            )
        return hits

    async def add(self, title: str, content: str) -> MemoryOpResult:
        return await asyncio.to_thread(self._store.add, title, content)

    async def update(self, name: str, content: str) -> MemoryOpResult:
        return await asyncio.to_thread(self._store.update, name, content)

    async def remove(self, name: str) -> MemoryOpResult:
        return await asyncio.to_thread(self._store.remove, name)

    async def render_prompt(self, budget: int | None = None) -> str:
        prompt = await asyncio.to_thread(
            load_ohmo_memory_prompt,
            self._store._workspace,
            max_chars=budget,
        )
        return cast(str, prompt)

    async def append_turn(self, role: str, text: str) -> None:
        del role, text


def make_memory_backend(
    cfg: GatewayConfig,
    workspace: str | Path | None,
) -> MemoryBackend:
    """Build the configured workspace-scoped memory backend."""
    if cfg.memory_backend == "file":
        return FileMemoryBackend(MemoryStore(workspace))
    if cfg.memory_backend == "honcho":
        raise NotImplementedError("honcho memory backend not built in Phase 0")
    raise ValueError(f"unsupported memory backend: {cfg.memory_backend!r}")

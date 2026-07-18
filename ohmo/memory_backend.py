"""Async storage seam for ohmo personal memory."""

from __future__ import annotations

import asyncio
import builtins
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from ohmo.memory import (
    DEFAULT_MEMORY_INJECT_CHARS,
    _MEMORY_ENTRY_RENDER_CHARS,
    load_memory_prompt as load_ohmo_memory_prompt,
)
from ohmo.memory_catalog import CatalogRecord, MemoryCatalog
from ohmo.memory_store import MemoryEntry, MemoryOpResult, MemoryStore
from ohmo.memory_tool import _search_memory
from ohmo.threat_patterns import scan_for_threats
from ohmo.workspace import get_memory_dir

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


class CatalogMemoryBackend(MemoryBackend):
    """Async adapter over the workspace-scoped SQLite memory catalog."""

    def __init__(
        self,
        catalog: MemoryCatalog,
        workspace: str | Path | None,
    ) -> None:
        self._catalog = catalog
        self._memory_dir = get_memory_dir(workspace)

    def _entry(self, record: CatalogRecord) -> MemoryEntry:
        return MemoryEntry(
            name=f"{record.slug}.md",
            slug=record.slug,
            title=record.title,
            content=record.content,
            path=self._memory_dir / f"{record.slug}.md",
        )

    async def list(self) -> list[MemoryEntry]:
        records = await asyncio.to_thread(self._catalog.list, include_archived=False)
        return [self._entry(record) for record in records]

    async def get(self, name: str) -> MemoryEntry | None:
        record = await asyncio.to_thread(self._catalog.get, name)
        return self._entry(record) if record is not None else None

    async def search(self, query: str, top_k: int) -> builtins.list[MemoryHit]:
        records = await asyncio.to_thread(self._catalog.search, query, top_k)
        return [
            MemoryHit(
                name=f"{record.slug}.md",
                title=record.title,
                snippet=_content_excerpt(record.content),
                rank=rank,
            )
            for rank, record in enumerate(records, start=1)
        ]

    async def add(self, title: str, content: str) -> MemoryOpResult:
        result = await asyncio.to_thread(self._catalog.add, title, content, source="curated")
        return self._result(result)

    async def update(self, name: str, content: str) -> MemoryOpResult:
        result = await asyncio.to_thread(self._catalog.update, name, content)
        return self._result(result)

    async def remove(self, name: str) -> MemoryOpResult:
        return await asyncio.to_thread(self._catalog.remove, name)

    async def render_prompt(self, budget: int | None = None) -> str:
        return await asyncio.to_thread(self._render_prompt, budget)

    def _result(self, result: MemoryOpResult) -> MemoryOpResult:
        if result.entries is None:
            return result
        records = cast(tuple[CatalogRecord, ...], result.entries)
        return MemoryOpResult(
            ok=result.ok,
            message=result.message,
            entries=tuple(self._entry(record) for record in records),
        )

    def _render_prompt(self, budget: int | None) -> str:
        records = self._catalog.list(include_archived=False)
        lines = [
            "# ohmo Memory",
            f"- Personal memory directory: {self._memory_dir}",
            "- Use this memory for stable user preferences and durable personal context.",
            "- Curate it with the `memory` tool (add/update/remove/list/get) — do NOT write "
            'memory files by hand. Save DECLARATIVE facts ("User prefers UTC"), not '
            "self-instructions; skip transient progress, raw data dumps (paths/listings), and secrets.",
        ]

        if records:
            index_lines = [
                "# Memory Index",
                *(f"- [{record.title}]({record.slug}.md)" for record in records),
            ][:200]
            safe_index_lines = [
                "[BLOCKED: index line contained a threat pattern]"
                if scan_for_threats(line, scope="all")
                else line
                for line in index_lines
            ]
            lines.extend(["", "## MEMORY.md", "```md", *safe_index_lines, "```"])

        render_budget = budget if budget is not None else _inject_char_budget()
        used = 0
        shown = 0
        for index, record in enumerate(records):
            content = record.content.strip()
            if not content:
                continue
            findings = scan_for_threats(content, scope="all")
            name = f"{record.slug}.md"
            if findings:
                body = (
                    f"[BLOCKED: {name} contained threat pattern(s): "
                    f"{', '.join(findings)}. Removed from the prompt; use the memory tool "
                    f"(action='get'/'remove') to inspect or delete it.]"
                )
            else:
                body = content[:_MEMORY_ENTRY_RENDER_CHARS]

            if shown > 0 and used + len(body) > render_budget:
                remaining = sum(1 for item in records[index:] if item.content.strip())
                if remaining:
                    lines.append("")
                    lines.append(
                        f"_({remaining} more memory "
                        f"entr{'y' if remaining == 1 else 'ies'} in the index above — "
                        "read one with memory(action='get', name='<name>'))._"
                    )
                break

            lines.extend(["", f"## {name}", "```md", body, "```"])
            self._catalog.record_use(record.slug)
            used += len(body)
            shown += 1

        return "\n".join(lines)

    async def append_turn(self, role: str, text: str) -> None:
        del role, text


def _content_excerpt(content: str) -> str:
    snippet = " ".join(content.split())
    if len(snippet) > 240:
        return snippet[:237].rstrip() + "..."
    return snippet


def _inject_char_budget() -> int:
    raw = os.environ.get("OHMO_MEMORY_INJECT_CHARS")
    if raw is None:
        return DEFAULT_MEMORY_INJECT_CHARS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MEMORY_INJECT_CHARS
    return value if value > 0 else DEFAULT_MEMORY_INJECT_CHARS


def make_memory_backend(
    cfg: GatewayConfig,
    workspace: str | Path | None,
) -> MemoryBackend:
    """Build the configured workspace-scoped memory backend."""
    if cfg.memory_backend == "file":
        return FileMemoryBackend(MemoryStore(workspace))
    if cfg.memory_backend == "catalog":
        return CatalogMemoryBackend(MemoryCatalog(workspace), workspace)
    if cfg.memory_backend == "honcho":
        raise NotImplementedError("honcho memory backend not built in Phase 0")
    raise ValueError(f"unsupported memory backend: {cfg.memory_backend!r}")

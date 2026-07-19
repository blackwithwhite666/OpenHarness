"""Async storage seam for ohmo personal memory."""

from __future__ import annotations

import asyncio
import builtins
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Protocol, cast

from ohmo.memory import (
    DEFAULT_MEMORY_INJECT_CHARS,
    _MEMORY_ENTRY_RENDER_CHARS,
    ensure_catalog_migrated,
    load_memory_prompt as load_ohmo_memory_prompt,
)
from ohmo.memory_catalog import CatalogRecord, MemoryCatalog
from ohmo.memory_store import MemoryEntry, MemoryOpResult, MemoryStore
from ohmo.memory_tool import _search_memory
from ohmo.threat_patterns import scan_for_threats
from ohmo.workspace import get_memory_dir

if TYPE_CHECKING:
    from ohmo.gateway.models import GatewayConfig
    from ohmo.memory_service.honcho_client import HonchoClient


logger = logging.getLogger(__name__)

_DERIVED_RECALL_HEADING = "## Recalled (honcho, derived — may be imperfect)"
_DERIVED_RECALL_PROVENANCE = (
    "_The catalog memory above is curated; these additive hits are derived from conversations._"
)
_DERIVED_RECALL_TOP_K = 10


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


from ohmo.memory_service.shadow import (  # noqa: E402 - avoids package import cycle
    SHADOW_COMPARISON_LOG_FILENAME,
    append_shadow_record,
    build_shadow_record,
    shadow_report as shadow_report,
)


class ShadowMemoryBackend(MemoryBackend):
    """Catalog backend with off-path Honcho comparison and optional learning."""

    def __init__(
        self,
        base: CatalogMemoryBackend,
        honcho_client: HonchoClient | None = None,
        *,
        observer: str = "ohmo-curated",
        derived_observer: str = "ohmo",
        observed: str = "owner",
        conversation_learning: bool = False,
        session: str = "ohmo",
        assistant_peer: str = "ohmo",
        comparison_log_path: str | Path | None = None,
        is_owner: bool = True,
    ) -> None:
        self._base = base
        self._honcho_client = honcho_client if is_owner else None
        self._observer = observer
        self._derived_observer = derived_observer
        self._observed = observed
        self._conversation_learning = conversation_learning
        self._session = session
        self._assistant_peer = assistant_peer
        self._comparison_log_path = (
            Path(comparison_log_path)
            if comparison_log_path is not None
            else base._memory_dir / SHADOW_COMPARISON_LOG_FILENAME
        )
        self._pending: set[asyncio.Task[None]] = set()
        self._log_lock = asyncio.Lock()
        self._ingest_lock = asyncio.Lock()

    async def list(self) -> list[MemoryEntry]:
        return await self._base.list()

    async def get(self, name: str) -> MemoryEntry | None:
        return await self._base.get(name)

    async def search(self, query: str, top_k: int) -> builtins.list[MemoryHit]:
        started = perf_counter()
        catalog_hits = await self._base.search(query, top_k)
        catalog_latency_ms = (perf_counter() - started) * 1_000
        if self._honcho_client is not None:
            task = asyncio.create_task(
                self._compare(
                    query=query,
                    top_k=top_k,
                    catalog_hits=catalog_hits,
                    catalog_latency_ms=catalog_latency_ms,
                ),
                name="ohmo-shadow-recall",
            )
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)
        return catalog_hits

    async def add(self, title: str, content: str) -> MemoryOpResult:
        return await self._base.add(title, content)

    async def update(self, name: str, content: str) -> MemoryOpResult:
        return await self._base.update(name, content)

    async def remove(self, name: str) -> MemoryOpResult:
        return await self._base.remove(name)

    async def render_prompt(self, budget: int | None = None) -> str:
        return await self._base.render_prompt(budget)

    async def derived_recall_block(
        self,
        query: str,
        *,
        budget: int,
        timeout: float,
    ) -> str | None:
        """Render bounded derived hits, or omit them when Honcho is unavailable.

        ``query`` is the latest user-turn text supplied by the gateway. This
        keeps recall relevant to the submitted turn without adding a second
        Honcho working-representation read to the prompt path.
        """
        honcho_client = self._honcho_client
        query = query.strip()
        if honcho_client is None or not query or budget <= 0 or timeout <= 0:
            return None

        try:
            hits = await asyncio.wait_for(
                honcho_client.query_conclusions(
                    query,
                    observer=self._derived_observer,
                    observed=self._observed,
                    top_k=_DERIVED_RECALL_TOP_K,
                ),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning("ohmo visible Honcho recall timed out")
            return None
        except Exception:  # noqa: BLE001 - derived recall is additive only
            logger.warning("ohmo visible Honcho recall failed", exc_info=True)
            return None

        prefix = f"{_DERIVED_RECALL_HEADING}\n{_DERIVED_RECALL_PROVENANCE}"
        if len(prefix) >= budget:
            return None

        lines = [prefix]
        for hit in hits:
            content = getattr(hit, "content", None)
            if not isinstance(content, str):
                continue
            content = " ".join(content.split())
            if not content or scan_for_threats(content, scope="all"):
                continue

            available = budget - len("\n".join(lines)) - len("\n- ")
            if available <= 0:
                break
            if len(content) > available:
                if available <= 3:
                    break
                content = content[: available - 3].rstrip() + "..."
            lines.append(f"- {content}")
            if len("\n".join(lines)) >= budget:
                break

        return "\n".join(lines) if len(lines) > 1 else None

    async def append_turn(self, role: str, text: str) -> None:
        honcho_client = self._honcho_client
        if not self._conversation_learning or honcho_client is None:
            await self._base.append_turn(role, text)
            return

        peer_id = {"user": self._observed, "assistant": self._assistant_peer}.get(role)
        if peer_id is None:
            return
        task = asyncio.create_task(
            self._ingest_turn(role=role, text=text, peer_id=peer_id),
            name="ohmo-conversation-learning",
        )
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def await_pending(self) -> None:
        """Drain all shadow work scheduled before or while this call runs."""
        while self._pending:
            await asyncio.gather(*tuple(self._pending), return_exceptions=True)

    async def _compare(
        self,
        *,
        query: str,
        top_k: int,
        catalog_hits: builtins.list[MemoryHit],
        catalog_latency_ms: float,
    ) -> None:
        honcho_client = self._honcho_client
        if honcho_client is None:
            return
        try:
            started = perf_counter()
            honcho_hits = await honcho_client.query_conclusions(
                query,
                observer=self._observer,
                observed=self._observed,
                top_k=top_k,
            )
            honcho_latency_ms = (perf_counter() - started) * 1_000
            record = build_shadow_record(
                query=query,
                catalog_hits=[(hit.name, hit.rank, hit.snippet) for hit in catalog_hits],
                honcho_hits=honcho_hits,
                catalog_latency_ms=catalog_latency_ms,
                honcho_latency_ms=honcho_latency_ms,
            )
            async with self._log_lock:
                await asyncio.to_thread(
                    append_shadow_record,
                    self._comparison_log_path,
                    record,
                )
        except Exception:  # noqa: BLE001 - shadow failures never reach the model path
            logger.warning("ohmo shadow recall comparison failed", exc_info=True)

    async def _ingest_turn(self, *, role: str, text: str, peer_id: str) -> None:
        honcho_client = self._honcho_client
        if honcho_client is None:
            return
        try:
            async with self._ingest_lock:
                await honcho_client.create_messages(
                    self._session,
                    [
                        {
                            "content": text,
                            "peer_id": peer_id,
                            "metadata": {"role": role},
                        }
                    ],
                )
        except Exception:  # noqa: BLE001 - learning failures never reach the turn path
            logger.warning("ohmo conversation learning ingestion failed", exc_info=True)


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
        return CatalogMemoryBackend(ensure_catalog_migrated(workspace), workspace)
    if cfg.memory_backend == "shadow":
        base = CatalogMemoryBackend(ensure_catalog_migrated(workspace), workspace)
        honcho_client = None
        if (
            cfg.owner_principals
            and cfg.honcho_base_url
            and cfg.honcho_api_key
            and cfg.honcho_workspace
        ):
            from ohmo.memory_service.honcho_client import HonchoClient

            honcho_client = HonchoClient(
                cfg.honcho_base_url,
                cfg.honcho_api_key,
                cfg.honcho_workspace,
            )
        return ShadowMemoryBackend(
            base,
            honcho_client=honcho_client,
            conversation_learning=cfg.conversation_learning,
            comparison_log_path=(get_memory_dir(workspace) / SHADOW_COMPARISON_LOG_FILENAME),
            is_owner=bool(cfg.owner_principals),
        )
    if cfg.memory_backend == "service":
        if not cfg.memory_service_socket or not cfg.memory_service_secret_file:
            raise ValueError(
                "service memory backend requires memory_service_socket and "
                "memory_service_secret_file"
            )
        from ohmo.memory_service.client import MemoryServiceClient

        return MemoryServiceClient(
            cfg.memory_service_socket,
            cfg.memory_service_secret_file,
        )
    if cfg.memory_backend == "honcho":
        raise NotImplementedError("honcho memory backend not built in Phase 0")
    raise ValueError(f"unsupported memory backend: {cfg.memory_backend!r}")

"""Async storage seam for ohmo personal memory."""

from __future__ import annotations

import asyncio
import builtins
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Mapping, Protocol, Sequence, cast

import numpy as np
import httpx
from numpy.typing import NDArray

from openharness.untrusted import UNTRUSTED_BANNER

from ohmo.memory import (
    DEFAULT_MEMORY_INJECT_CHARS,
    _MEMORY_ENTRY_RENDER_CHARS,
    ensure_catalog_migrated,
    load_memory_prompt as load_ohmo_memory_prompt,
)
from ohmo.memory_catalog import CatalogRecord, MemoryCatalog
from ohmo.memory_store import MemoryEntry, MemoryOpResult, MemoryStore, slugify
from ohmo.memory_tool import _search_memory
from ohmo.threat_patterns import scan_for_threats
from ohmo.workspace import get_memory_dir

if TYPE_CHECKING:
    from openharness.evals.embeddings import EmbeddingClient

    from ohmo.gateway.models import GatewayConfig
    from ohmo.memory_service.honcho_client import HonchoClient


logger = logging.getLogger(__name__)

_DERIVED_RECALL_HEADING = "## Recalled (honcho, derived — may be imperfect)"
_DERIVED_RECALL_PROVENANCE = (
    "_The catalog memory above is curated; these additive hits are derived from conversations._"
)
_DERIVED_RECALL_TOP_K = 10
_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
_DEFAULT_EMBEDDING_TIMEOUT_SECONDS = 2.0
_EMBEDDING_BACKFILL_LIMIT = 128
_SEMANTIC_SIMILARITY_FLOOR = 0.5


@dataclass(frozen=True)
class MemoryHit:
    """One ranked memory-search result."""

    name: str
    title: str
    snippet: str
    rank: int


@dataclass(frozen=True, slots=True)
class TenantHonchoBinding:
    """One private tenant's scoped Honcho credentials and person peer."""

    tenant_id: str
    base_url: str
    api_key: str
    workspace: str
    observed_peer: str
    session: str = "ohmo"


@dataclass(frozen=True, slots=True)
class ConversationAppendReceipt:
    """Durable ids for one ordered user/assistant Honcho exchange."""

    user_message_id: str
    assistant_message_id: str
    user_client_op_id: str
    assistant_client_op_id: str

    @property
    def user_id(self) -> str:
        return self.user_message_id

    @property
    def assistant_id(self) -> str:
        return self.assistant_message_id


class NutritionReconciliationError(RuntimeError):
    """Honcho contained no unambiguous trusted result for a retry."""


def resolve_tenant_honcho_binding(
    cfg: GatewayConfig,
    tenant_id: str,
) -> TenantHonchoBinding | None:
    """Resolve a complete tenant binding without borrowing another tenant's values."""
    tenant_id = tenant_id.strip()
    if not tenant_id or not cfg.honcho_base_url:
        return None

    raw_binding = cfg.tenant_honcho.get(tenant_id)
    if raw_binding is None:
        if tenant_id != "owner" or not cfg.owner_principals:
            return None
        workspace = cfg.honcho_workspace
        api_key = cfg.honcho_api_key
        observed_peer = "owner"
        session = "ohmo"
    else:
        workspace = raw_binding.get("workspace")
        api_key = raw_binding.get("api_key")
        observed_peer = raw_binding.get("observed_peer", tenant_id)
        session = raw_binding.get("session", "ohmo")

    if not all(
        isinstance(value, str) and value.strip()
        for value in (workspace, api_key, observed_peer, session)
    ):
        return None
    assert isinstance(workspace, str)
    assert isinstance(api_key, str)
    assert isinstance(observed_peer, str)
    assert isinstance(session, str)
    return TenantHonchoBinding(
        tenant_id=tenant_id,
        base_url=cfg.honcho_base_url,
        api_key=api_key,
        workspace=workspace.strip(),
        observed_peer=observed_peer.strip(),
        session=session.strip(),
    )


class MemoryBackend(Protocol):
    """Storage-neutral async interface for model-facing memory operations."""

    async def list(self) -> list[MemoryEntry]: ...

    async def get(self, name: str) -> MemoryEntry | None: ...

    async def record_use(self, name: str) -> None: ...

    async def search(self, query: str, top_k: int) -> builtins.list[MemoryHit]: ...

    async def add(self, title: str, content: str) -> MemoryOpResult: ...

    async def update(
        self,
        name: str,
        content: str,
        *,
        title: str | None = None,
    ) -> MemoryOpResult: ...

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

    async def record_use(self, name: str) -> None:
        await asyncio.to_thread(self._store.record_use, name)

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

    async def update(
        self,
        name: str,
        content: str,
        *,
        title: str | None = None,
    ) -> MemoryOpResult:
        return await asyncio.to_thread(self._store.update, name, content, title=title)

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
        *,
        tenant_id: str = "owner",
        shared_tenant_id: str | None = None,
        embedder: EmbeddingClient | None = None,
        owns_embedder: bool = False,
        model: str = _DEFAULT_EMBEDDING_MODEL,
        embedding_timeout: float = _DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    ) -> None:
        self._catalog = catalog
        self._tenant_id = tenant_id.strip()
        if not self._tenant_id:
            raise ValueError("a catalog memory tenant id is required")
        self._shared_tenant_id = (
            shared_tenant_id.strip() if shared_tenant_id is not None else None
        )
        if self._shared_tenant_id == "":
            raise ValueError("a shared catalog memory tenant id cannot be empty")
        if self._shared_tenant_id == self._tenant_id:
            raise ValueError("private and shared catalog memory tenant ids must differ")
        self._catalog.ensure_tenant(self._tenant_id, "private")
        if self._shared_tenant_id is not None:
            self._catalog.ensure_tenant(self._shared_tenant_id, "shared")
        self._memory_dir = get_memory_dir(workspace)
        self._embedder = embedder
        self._owns_embedder = owns_embedder and embedder is not None
        self._embedding_model = model.strip()
        self._embedding_timeout = embedding_timeout

    async def aclose(self) -> None:
        """Close a factory-owned embedder without making shutdown fragile."""
        if not self._owns_embedder:
            return
        self._owns_embedder = False
        close = getattr(self._embedder, "aclose", None)
        if not callable(close):
            return
        try:
            await close()
        except Exception:
            logger.debug("catalog semantic embedder close failed", exc_info=True)

    def _entry(self, record: CatalogRecord, *, shared: bool = False) -> MemoryEntry:
        return MemoryEntry(
            name=f"{record.slug}.md",
            slug=record.slug,
            title=f"[shared] {record.title}" if shared else record.title,
            content=record.content,
            path=self._memory_dir / f"{record.slug}.md",
        )

    async def list(self) -> list[MemoryEntry]:
        private_records = await asyncio.to_thread(
            self._catalog.list,
            self._tenant_id,
            include_archived=False,
        )
        entries = [self._entry(record) for record in private_records]
        if self._shared_tenant_id is not None:
            shared_records = await asyncio.to_thread(
                self._catalog.list,
                self._shared_tenant_id,
                include_archived=False,
            )
            entries.extend(self._entry(record, shared=True) for record in shared_records)
        return entries

    async def get(self, name: str) -> MemoryEntry | None:
        record = await asyncio.to_thread(self._catalog.get, self._tenant_id, name)
        return self._entry(record) if record is not None else None

    async def record_use(self, name: str) -> None:
        await asyncio.to_thread(self._catalog.record_use, self._tenant_id, name)

    async def search(self, query: str, top_k: int) -> builtins.list[MemoryHit]:
        """Rank FTS hits first, then semantic-only hits by cosine and slug."""
        records = await self._search_tenant(self._tenant_id, query, top_k)
        labeled_records = [(record, False) for record in records]
        if self._shared_tenant_id is not None and len(labeled_records) < top_k:
            shared_records = await self._search_tenant(
                self._shared_tenant_id,
                query,
                top_k - len(labeled_records),
                backfill_embeddings=False,
            )
            labeled_records.extend((record, True) for record in shared_records)
        return [
            MemoryHit(
                name=f"{record.slug}.md",
                title=f"[shared] {record.title}" if shared else record.title,
                snippet=_content_excerpt(record.content),
                rank=rank,
            )
            for rank, (record, shared) in enumerate(labeled_records[:top_k], start=1)
        ]

    async def _search_tenant(
        self,
        tenant_id: str,
        query: str,
        top_k: int,
        *,
        backfill_embeddings: bool = True,
    ) -> builtins.list[CatalogRecord]:
        fts_records = await asyncio.to_thread(self._catalog.search, tenant_id, query, top_k)
        records = fts_records
        if self._embedder is not None and self._embedding_model and query.strip() and top_k > 0:
            try:
                semantic_records = await self._semantic_records(
                    tenant_id,
                    query,
                    top_k,
                    backfill_embeddings=backfill_embeddings,
                )
            except Exception:
                # FTS is authoritative: semantic failures must not alter its exact result.
                logger.debug("catalog semantic search failed open to FTS", exc_info=True)
            else:
                # Ranking rule: unchanged FTS order first, followed by semantic-only
                # records in descending cosine order (slug breaks ties).
                seen = {record.slug for record in fts_records}
                records = [*fts_records]
                for record in semantic_records:
                    if record.slug in seen:
                        continue
                    seen.add(record.slug)
                    records.append(record)
                records = records[:top_k]
        return records

    async def add(self, title: str, content: str) -> MemoryOpResult:
        result = await asyncio.to_thread(
            self._catalog.add,
            self._tenant_id,
            title,
            content,
            source="curated",
        )
        if result.ok and result.message.startswith("Saved memory "):
            await self._best_effort_embed_name(slugify(title))
        return self._result(result)

    async def update(
        self,
        name: str,
        content: str,
        *,
        title: str | None = None,
    ) -> MemoryOpResult:
        result = await asyncio.to_thread(
            self._catalog.update,
            self._tenant_id,
            name,
            content,
            title=title,
        )
        if result.ok:
            await self._best_effort_embed_name(name)
        return self._result(result)

    async def remove(self, name: str) -> MemoryOpResult:
        return await asyncio.to_thread(self._catalog.remove, self._tenant_id, name)

    async def _best_effort_embed_name(self, name: str) -> None:
        if self._embedder is None or not self._embedding_model:
            return
        try:
            record = await asyncio.to_thread(self._catalog.get, self._tenant_id, name)
        except Exception:
            logger.debug("catalog embed-on-write lookup failed open", exc_info=True)
            return
        if record is not None:
            await self._best_effort_embed(record)

    async def _best_effort_embed(self, record: CatalogRecord) -> None:
        if (
            self._embedder is None
            or not self._embedding_model
            or record.archive_status != "active"
        ):
            return
        try:
            vectors = await self._embed_texts([_embedding_text(record)])
            await asyncio.to_thread(
                self._catalog.store_embedding,
                self._tenant_id,
                record.slug,
                self._embedding_model,
                vectors[0].tolist(),
                record.generation,
            )
        except Exception:
            logger.debug("catalog embed-on-write failed open", exc_info=True)

    async def _semantic_records(
        self,
        tenant_id: str,
        query: str,
        top_k: int,
        *,
        backfill_embeddings: bool,
    ) -> builtins.list[CatalogRecord]:
        active_records = await asyncio.to_thread(
            self._catalog.list,
            tenant_id,
            include_archived=False,
        )
        if not active_records:
            return []

        embeddings = await asyncio.to_thread(self._catalog.get_embeddings, tenant_id)
        missing = [
            record
            for record in active_records
            if not _embedding_is_current(
                embeddings.get(record.slug),
                model=self._embedding_model,
                generation=record.generation,
            )
        ][:_EMBEDDING_BACKFILL_LIMIT]
        if not backfill_embeddings:
            missing = []
        if missing:
            vectors = await self._embed_texts([_embedding_text(record) for record in missing])
            for record, vector in zip(missing, vectors, strict=True):
                stored = await asyncio.to_thread(
                    self._catalog.store_embedding,
                    tenant_id,
                    record.slug,
                    self._embedding_model,
                    vector.tolist(),
                    record.generation,
                )
                if stored:
                    embeddings[record.slug] = (
                        vector.astype(float).tolist(),
                        self._embedding_model,
                        record.generation,
                    )

        query_vector = (await self._embed_texts([query]))[0]
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm == 0.0:
            return []

        scored: builtins.list[tuple[float, CatalogRecord]] = []
        for record in active_records:
            embedding = embeddings.get(record.slug)
            if not _embedding_is_current(
                embedding,
                model=self._embedding_model,
                generation=record.generation,
            ):
                continue
            assert embedding is not None
            vector = np.asarray(embedding[0], dtype=np.float32)
            if vector.shape != query_vector.shape:
                continue
            denominator = query_norm * float(np.linalg.norm(vector))
            if denominator == 0.0:
                continue
            similarity = float(np.dot(query_vector, vector) / denominator)
            if similarity >= _SEMANTIC_SIMILARITY_FLOOR:
                scored.append((similarity, record))

        scored.sort(key=lambda item: (-item[0], item[1].slug))
        return [record for _, record in scored[:top_k]]

    async def _embed_texts(self, texts: builtins.list[str]) -> builtins.list[NDArray[np.float32]]:
        if self._embedder is None:
            raise RuntimeError("catalog semantic embedder is not configured")
        response = await asyncio.wait_for(
            self._embedder.embed(
                texts,
                return_dense=True,
                return_sparse=False,
                batch_size=len(texts),
            ),
            timeout=self._embedding_timeout,
        )
        return _dense_vectors(response, expected_count=len(texts))

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
        private_records = self._catalog.list(self._tenant_id, include_archived=False)
        labeled_records = [(record, False) for record in private_records]
        if self._shared_tenant_id is not None:
            shared_records = self._catalog.list(
                self._shared_tenant_id,
                include_archived=False,
            )
            labeled_records.extend((record, True) for record in shared_records)
        lines = [
            "# ohmo Memory",
            f"- Personal memory directory: {self._memory_dir}",
            "- Use this memory for stable user preferences and durable personal context.",
            "- Curate it with the `memory` tool (add/update/remove/list/get) — do NOT write "
            'memory files by hand. Save DECLARATIVE facts ("User prefers UTC"), not '
            "self-instructions; skip transient progress, raw data dumps (paths/listings), and secrets.",
        ]

        if labeled_records:
            index_lines = [
                "# Memory Index",
                *(
                    f"- [{record.title}]({record.slug}.md)"
                    f"{' [shared]' if shared else ''}"
                    for record, shared in labeled_records
                ),
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
        for index, (record, shared) in enumerate(labeled_records):
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
                remaining = sum(
                    1 for item, _ in labeled_records[index:] if item.content.strip()
                )
                if remaining:
                    lines.append("")
                    lines.append(
                        f"_({remaining} more memory "
                        f"entr{'y' if remaining == 1 else 'ies'} in the index above — "
                        "read one with memory(action='get', name='<name>'))._"
                    )
                break

            heading = f"## {name}{' [shared]' if shared else ''}"
            lines.extend(["", heading, "```md", body, "```"])
            if not shared:
                self._catalog.record_use(self._tenant_id, record.slug)
            used += len(body)
            shown += 1

        return "\n".join(lines)

    async def append_turn(self, role: str, text: str) -> None:
        del role, text

    def judge_store(self) -> CatalogJudgeStore:
        """Return a synchronous, tenant-bound view for the legacy judge seam."""
        return CatalogJudgeStore(self)


class CatalogJudgeStore:
    """MemoryStore-compatible view used by the synchronous judge helpers.

    The judge predates the async backend seam. Keeping this small adapter at the
    catalog boundary lets it read the private+shared union while every mutation
    remains bound to the private tenant.
    """

    def __init__(self, backend: CatalogMemoryBackend) -> None:
        self._backend = backend
        self._catalog = backend._catalog
        self._tenant_id = backend._tenant_id
        self._shared_tenant_id = backend._shared_tenant_id
        self._store_char_budget = backend._catalog._store_char_budget

    def _dir(self) -> Path:
        return self._backend._memory_dir / ".judge" / self._tenant_id

    def _entry(self, record: CatalogRecord, *, shared: bool = False) -> MemoryEntry:
        entry = self._backend._entry(record, shared=shared)
        return MemoryEntry(
            name=entry.name,
            slug=entry.slug,
            title=entry.title,
            content=entry.content,
            path=self._dir() / entry.name,
        )

    def list(self) -> builtins.list[MemoryEntry]:
        private_records = self._catalog.list(self._tenant_id, include_archived=False)
        entries = [self._entry(record) for record in private_records]
        if self._shared_tenant_id is not None:
            shared_records = self._catalog.list(
                self._shared_tenant_id,
                include_archived=False,
            )
            entries.extend(self._entry(record, shared=True) for record in shared_records)
        return entries

    def get(self, name: str) -> MemoryEntry | None:
        record = self._catalog.get(self._tenant_id, name)
        return self._entry(record) if record is not None else None

    def add(self, title: str, content: str) -> MemoryOpResult:
        return self._result(
            self._catalog.add(
                self._tenant_id,
                title,
                content,
                source="curated",
            )
        )

    def update(
        self,
        name: str,
        content: str,
        *,
        title: str | None = None,
    ) -> MemoryOpResult:
        return self._result(
            self._catalog.update(
                self._tenant_id,
                name,
                content,
                title=title,
            )
        )

    def remove(self, name: str) -> MemoryOpResult:
        return self._catalog.remove(self._tenant_id, name)

    def total_chars(self) -> int:
        return sum(
            len(record.content)
            for record in self._catalog.list(
                self._tenant_id,
                include_archived=False,
            )
        )

    def _result(self, result: MemoryOpResult) -> MemoryOpResult:
        if result.entries is None:
            return result
        records = cast(tuple[CatalogRecord, ...], result.entries)
        return MemoryOpResult(
            ok=result.ok,
            message=result.message,
            entries=tuple(self._entry(record) for record in records),
        )


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

    async def record_use(self, name: str) -> None:
        await self._base.record_use(name)

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

    async def update(
        self,
        name: str,
        content: str,
        *,
        title: str | None = None,
    ) -> MemoryOpResult:
        return await self._base.update(name, content, title=title)

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
            started = perf_counter()
            hits = await asyncio.wait_for(
                honcho_client.query_conclusions(
                    query,
                    observer=self._derived_observer,
                    observed=self._observed,
                    top_k=_DERIVED_RECALL_TOP_K,
                ),
                timeout=timeout,
            )
            honcho_latency_ms = (perf_counter() - started) * 1_000
        except TimeoutError:
            logger.warning("ohmo visible Honcho recall timed out")
            return None
        except Exception:  # noqa: BLE001 - derived recall is additive only
            logger.warning("ohmo visible Honcho recall failed", exc_info=True)
            return None

        if hits:
            task = asyncio.create_task(
                self._shadow_compare_derived(
                    query=query,
                    honcho_hits=hits,
                    honcho_latency_ms=honcho_latency_ms,
                ),
                name="ohmo-derived-shadow",
            )
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)

        prefix = (
            f"{_DERIVED_RECALL_HEADING}\n"
            f"{_DERIVED_RECALL_PROVENANCE}\n"
            f"{UNTRUSTED_BANNER}"
        )
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

    async def append_exchange(
        self,
        user_text: str,
        assistant_text: str,
        *,
        user_metadata: Mapping[str, object],
        assistant_metadata: Mapping[str, object],
        durable: bool = False,
        trusted_nutrition: bool = False,
        nutrition: bool | None = None,
    ) -> ConversationAppendReceipt | None:
        if nutrition is not None:
            trusted_nutrition = trusted_nutrition or nutrition
        honcho_client = self._honcho_client
        if not self._conversation_learning or honcho_client is None:
            await self._base.append_turn("user", user_text)
            await self._base.append_turn("assistant", assistant_text)
            if durable or trusted_nutrition:
                raise NutritionReconciliationError("durable nutrition ingestion is unavailable")
            return None

        user_metadata_mapping = dict(user_metadata)
        assistant_metadata_mapping = dict(assistant_metadata)
        user_metadata_mapping["role"] = "user"
        assistant_metadata_mapping["role"] = "assistant"
        is_trusted_nutrition = trusted_nutrition or bool(
            assistant_metadata_mapping.get("_nutrition_trusted")
        )
        if is_trusted_nutrition:
            _validate_trusted_nutrition_metadata(
                user_metadata_mapping,
                assistant_metadata_mapping,
            )
        if durable or is_trusted_nutrition:
            return await self._append_exchange_durable(
                honcho_client,
                user_text=user_text,
                assistant_text=assistant_text,
                user_metadata=user_metadata_mapping,
                assistant_metadata=assistant_metadata_mapping,
            )
        task = asyncio.create_task(
            self._ingest_exchange(
                user_text=user_text,
                assistant_text=assistant_text,
                user_metadata=user_metadata_mapping,
                assistant_metadata=assistant_metadata_mapping,
            ),
            name="ohmo-conversation-learning-exchange",
        )
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return None

    async def _append_exchange_durable(
        self,
        honcho_client: HonchoClient,
        *,
        user_text: str,
        assistant_text: str,
        user_metadata: Mapping[str, object],
        assistant_metadata: Mapping[str, object],
    ) -> ConversationAppendReceipt:
        user_op = _required_client_op_id(user_metadata, "user")
        assistant_op = _required_client_op_id(assistant_metadata, "assistant")
        async with self._ingest_lock:
            existing_assistant = await self._find_unique_operation(
                honcho_client, assistant_op, expected_role="assistant"
            )
            existing_user = await self._find_unique_operation(
                honcho_client, user_op, expected_role="user"
            )
            if existing_assistant is not None:
                if existing_user is None:
                    raise NutritionReconciliationError(
                        "assistant operation exists without its paired user message"
                    )
                return ConversationAppendReceipt(
                    user_message_id=existing_user.id,
                    assistant_message_id=existing_assistant.id,
                    user_client_op_id=user_op,
                    assistant_client_op_id=assistant_op,
                )

            messages: list[dict[str, object]] = []
            if existing_user is None:
                messages.append(
                    {"content": user_text, "peer_id": self._observed, "metadata": dict(user_metadata)}
                )
            messages.append(
                {
                    "content": assistant_text,
                    "peer_id": self._assistant_peer,
                    "metadata": dict(assistant_metadata),
                }
            )
            try:
                created = await honcho_client.create_messages(self._session, messages)
            except (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException):
                # The server may have committed the batch before the client
                # timed out. Reconcile the stable assistant operation before
                # allowing a retry to append anything.
                created_assistant = await self._find_unique_operation(
                    honcho_client, assistant_op, expected_role="assistant"
                )
                created_user = await self._find_unique_operation(
                    honcho_client, user_op, expected_role="user"
                )
                if created_assistant is None or created_user is None:
                    raise
                return ConversationAppendReceipt(
                    user_message_id=created_user.id,
                    assistant_message_id=created_assistant.id,
                    user_client_op_id=user_op,
                    assistant_client_op_id=assistant_op,
                )

            expected_count = len(messages)
            if len(created) != expected_count:
                raise NutritionReconciliationError("Honcho returned an incomplete message batch")
            by_role = {str(item.metadata.get("role")): item for item in created}
            user_message = existing_user or by_role.get("user")
            assistant_message = by_role.get("assistant")
            if user_message is None or assistant_message is None:
                raise NutritionReconciliationError("Honcho response omitted a paired message")
            return ConversationAppendReceipt(
                user_message_id=user_message.id,
                assistant_message_id=assistant_message.id,
                user_client_op_id=user_op,
                assistant_client_op_id=assistant_op,
            )

    async def _find_unique_operation(
        self,
        honcho_client: HonchoClient,
        client_op_id: str,
        *,
        expected_role: str,
    ):
        matches = await honcho_client.find_messages_by_client_op_id(
            self._session, client_op_id
        )
        if len(matches) > 1:
            raise NutritionReconciliationError(
                f"ambiguous Honcho operation {client_op_id!r}"
            )
        if not matches:
            return None
        message = matches[0]
        if (
            message.metadata.get("client_op_id") != client_op_id
            or message.metadata.get("role") != expected_role
        ):
            raise NutritionReconciliationError(
                f"malformed Honcho operation {client_op_id!r}"
            )
        return message

    async def await_pending(self) -> None:
        """Drain all shadow work scheduled before or while this call runs."""
        while self._pending:
            pending = tuple(self._pending)
            await asyncio.gather(*pending, return_exceptions=True)
            self._pending.difference_update(pending)

    async def aclose(self) -> None:
        """Close resources owned by the catalog backend."""
        await self._base.aclose()

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

    async def _shadow_compare_derived(
        self,
        *,
        query: str,
        honcho_hits: Sequence[object],
        honcho_latency_ms: float,
    ) -> None:
        """Log a catalog-vs-Honcho shadow record for the per-turn derived path.

        Reuses the Honcho hits already fetched by ``derived_recall_block`` (so no
        second Honcho query) and only adds a local catalog search off the reply
        path.
        """
        if self._honcho_client is None:
            return
        try:
            started = perf_counter()
            catalog_hits = await self._base.search(query, _DERIVED_RECALL_TOP_K)
            catalog_latency_ms = (perf_counter() - started) * 1_000
            record = build_shadow_record(
                query=query,
                catalog_hits=[
                    (hit.name, hit.rank, hit.snippet) for hit in catalog_hits
                ],
                honcho_hits=honcho_hits,  # type: ignore[arg-type]
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
            logger.warning("ohmo derived shadow comparison failed", exc_info=True)

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

    async def _ingest_exchange(
        self,
        *,
        user_text: str,
        assistant_text: str,
        user_metadata: Mapping[str, object],
        assistant_metadata: Mapping[str, object],
    ) -> None:
        honcho_client = self._honcho_client
        if honcho_client is None:
            return
        try:
            messages = [
                {
                    "content": user_text,
                    "peer_id": self._observed,
                    "metadata": user_metadata,
                },
                {
                    "content": assistant_text,
                    "peer_id": self._assistant_peer,
                    "metadata": assistant_metadata,
                },
            ]
            async with self._ingest_lock:
                await honcho_client.create_messages(
                    self._session,
                    messages,
                )
        except Exception:  # noqa: BLE001 - learning failures never reach the turn path
            logger.warning(
                "ohmo conversation learning exchange ingestion failed",
                exc_info=True,
            )


def _embedding_text(record: CatalogRecord) -> str:
    return f"{record.title}\n{record.content}"


_TRUSTED_NUTRITION_KEYS = frozenset(
    {
        "_nutrition_trusted",
        "ingest_source",
        "confirmation_required",
        "candidate_id",
        "nutrition_phase",
        "client_op_id",
    }
)


def _required_client_op_id(metadata: Mapping[str, object], role: str) -> str:
    value = metadata.get("client_op_id")
    if not isinstance(value, str) or not value.strip():
        raise NutritionReconciliationError(f"{role} client_op_id is required")
    return value.strip()


def _validate_trusted_nutrition_metadata(
    user_metadata: Mapping[str, object],
    assistant_metadata: Mapping[str, object],
) -> None:
    if user_metadata.get("_nutrition_trusted") is not True:
        raise NutritionReconciliationError("nutrition provenance is not coordinator-trusted")
    if assistant_metadata.get("_nutrition_trusted") is not True:
        raise NutritionReconciliationError("nutrition assistant provenance is not coordinator-trusted")
    if user_metadata.get("ingest_source") != "dropbox_camera" or assistant_metadata.get(
        "ingest_source"
    ) != "dropbox_camera":
        raise NutritionReconciliationError("unsupported nutrition ingest source")
    if user_metadata.get("tenant_id") != "marina" or assistant_metadata.get("tenant_id") != "marina":
        raise NutritionReconciliationError("nutrition provenance is not Marina-bound")
    principal = user_metadata.get("source_principal")
    if not isinstance(principal, str) or not principal.startswith("telegram:"):
        raise NutritionReconciliationError("nutrition provenance has no Telegram principal")
    if user_metadata.get("confirmation_required") is not True or assistant_metadata.get(
        "confirmation_required"
    ) is not True:
        raise NutritionReconciliationError("nutrition confirmation is required")
    candidate = user_metadata.get("candidate_id")
    if not isinstance(candidate, str) or candidate != assistant_metadata.get("candidate_id"):
        raise NutritionReconciliationError("nutrition candidate identity mismatch")
    phase = user_metadata.get("nutrition_phase")
    if phase != "estimation" or assistant_metadata.get("nutrition_phase") != phase:
        raise NutritionReconciliationError("nutrition phase must be estimation")
    if assistant_metadata.get("client_op_id") != f"{candidate}:meal-observation:v1":
        raise NutritionReconciliationError("nutrition assistant operation is not candidate-bound")
    if user_metadata.get("client_op_id") == assistant_metadata.get("client_op_id"):
        raise NutritionReconciliationError("paired nutrition operation ids must differ")


def _embedding_is_current(
    embedding: tuple[Sequence[float], str, int] | None,
    *,
    model: str,
    generation: int,
) -> bool:
    return embedding is not None and embedding[1] == model and embedding[2] == generation


def _dense_vectors(
    response: dict[str, object],
    *,
    expected_count: int,
) -> list[NDArray[np.float32]]:
    count = response.get("count")
    if count is not None and count != expected_count:
        raise RuntimeError("embedding response count does not match input texts")
    dense = response.get("dense")
    if not isinstance(dense, list) or len(dense) != expected_count:
        raise RuntimeError("embedding response dense vector count does not match input texts")

    vectors: list[NDArray[np.float32]] = []
    dimensions: int | None = None
    for value in dense:
        if not isinstance(value, list):
            raise RuntimeError("embedding response vectors must be lists")
        vector = np.asarray(value, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise RuntimeError("embedding vectors must be non-empty finite one-dimensional lists")
        if dimensions is None:
            dimensions = int(vector.size)
        elif vector.size != dimensions:
            raise RuntimeError("embedding vector dimensions changed within one response")
        vectors.append(vector)
    return vectors


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
    *,
    tenant_id: str = "owner",
    shared_tenant_id: str | None = None,
) -> MemoryBackend:
    """Build the configured workspace-scoped memory backend."""
    if cfg.memory_backend == "file":
        return FileMemoryBackend(MemoryStore(workspace))
    if cfg.memory_backend == "catalog":
        return _make_catalog_memory_backend(
            cfg,
            workspace,
            tenant_id=tenant_id,
            shared_tenant_id=shared_tenant_id,
        )
    if cfg.memory_backend == "shadow":
        base = _make_catalog_memory_backend(
            cfg,
            workspace,
            tenant_id=tenant_id,
            shared_tenant_id=shared_tenant_id,
        )
        return make_tenant_shadow_backend(
            cfg,
            base,
            workspace,
            tenant_id=tenant_id,
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


def _make_catalog_memory_backend(
    cfg: GatewayConfig,
    workspace: str | Path | None,
    *,
    tenant_id: str = "owner",
    shared_tenant_id: str | None = None,
) -> CatalogMemoryBackend:
    catalog = ensure_catalog_migrated(workspace)
    if not cfg.semantic_search:
        return CatalogMemoryBackend(
            catalog,
            workspace,
            tenant_id=tenant_id,
            shared_tenant_id=shared_tenant_id,
        )

    from openharness.evals.inference import InferenceClient

    embedder = (
        InferenceClient(cfg.inference_url)
        if cfg.inference_url is not None
        else InferenceClient.from_env()
    )
    if cfg.embedding_model is None:
        return CatalogMemoryBackend(
            catalog,
            workspace,
            tenant_id=tenant_id,
            shared_tenant_id=shared_tenant_id,
            embedder=embedder,
            owns_embedder=True,
        )
    return CatalogMemoryBackend(
        catalog,
        workspace,
        tenant_id=tenant_id,
        shared_tenant_id=shared_tenant_id,
        embedder=embedder,
        owns_embedder=True,
        model=cfg.embedding_model,
    )


def make_tenant_shadow_backend(
    cfg: GatewayConfig,
    base: CatalogMemoryBackend,
    workspace: str | Path | None,
    *,
    tenant_id: str,
) -> ShadowMemoryBackend:
    """Wrap a scoped catalog with only that tenant's Honcho binding."""
    binding = resolve_tenant_honcho_binding(cfg, tenant_id)
    honcho_client = None
    observed_peer = tenant_id
    session = "ohmo"
    if binding is not None:
        from ohmo.memory_service.honcho_client import HonchoClient

        honcho_client = HonchoClient(
            binding.base_url,
            binding.api_key,
            binding.workspace,
        )
        observed_peer = binding.observed_peer
        session = binding.session
    return ShadowMemoryBackend(
        base,
        honcho_client=honcho_client,
        observed=observed_peer,
        conversation_learning=cfg.conversation_learning,
        session=session,
        comparison_log_path=(get_memory_dir(workspace) / SHADOW_COMPARISON_LOG_FILENAME),
        is_owner=honcho_client is not None,
    )

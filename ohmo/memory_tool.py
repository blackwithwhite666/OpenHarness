"""Model-callable ``memory`` tool for the ohmo gateway.

Gives the agent a disciplined way to curate its own durable memory — the layer
Hermes calls "agent self-curation". It accepts either the legacy file store or
the configured async memory backend, keeping tool writes and prompt recall on
the same workspace-scoped storage path.

Replaces ad-hoc ``write_file`` into ``memory/`` (which hit the ASCII-slug clobber
bug and had no bounds/dedup). The tool's description carries the collect/refuse
policy so the model saves the right things.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

from ohmo.memory_store import MemoryOpResult, MemoryStore

if TYPE_CHECKING:
    from ohmo.memory_backend import MemoryBackend

_DOCUMENT_SEARCH_CLI = os.environ.get("OHMO_DOCUMENT_SEARCH_CLI", "document_search-cli")
_MEMORY_SEARCH_COLLECTIONS = os.environ.get("OHMO_MEMORY_SEARCH_COLLECTIONS", "memory,archive")
_READ_ACTIONS = {"list", "get", "search"}


class OhmoMemoryToolInput(BaseModel):
    action: str = Field(
        description=(
            "One of: 'add' (new durable fact), 'update' (replace an existing "
            "entry's content), 'remove' (delete an entry), 'list' (titles + sizes), "
            "'get' (read one entry's full text), 'search' (semantic recall)."
        )
    )
    title: str = Field(
        default="",
        description=(
            "For action='add': a short title that also names the entry file. "
            "For action='update': optional new index label for the entry."
        ),
    )
    name: str = Field(
        default="",
        description="For action='get'/'update'/'remove': the entry name/slug (from 'list').",
    )
    content: str = Field(
        default="",
        description="For action='add'/'update': the memory body. A DECLARATIVE fact, not a self-instruction.",
    )
    query: str = Field(default="", description="For action='search': the natural-language recall query.")
    top_k: int = Field(default=5, description="Search result count (clamped to 1..10).")


class OhmoMemoryTool(BaseTool):
    name = "memory"
    description = (
        "Persist and curate durable personal memory across chats. SAVE proactively when "
        "you learn something lasting: user preferences & communication style, stable "
        "environment facts (hosts, key endpoints, tools), project/workflow conventions, "
        "corrections & workarounds, stable identities of people/services. DO NOT save: "
        "transient progress ('fixed X today', run logs), raw data dumps (file/artifact "
        "paths, listings), web-searchable trivia, secrets/tokens (keep the name, never the "
        "value), or anything already in soul.md / user.md. Write DECLARATIVE facts ('User "
        "prefers UTC timestamps'), not self-instructions ('always use UTC'). Entries are "
        "bounded; if the store is full, the tool tells you to consolidate (update/remove) "
        "in the same turn. Prefer this tool over writing memory files by hand. "
        "actions: add(title, content) · update(name, content) · remove(name) · list · get(name) "
        "· search(query) — semantically recall durable memory + archive when a fact is not visible."
    )
    input_model = OhmoMemoryToolInput

    def __init__(self, store: MemoryStore | MemoryBackend) -> None:
        self._store = store

    def is_read_only(self, arguments: BaseModel) -> bool:
        return getattr(arguments, "action", "").strip().lower() in _READ_ACTIONS

    async def execute(
        self, arguments: OhmoMemoryToolInput, context: ToolExecutionContext
    ) -> ToolResult:
        action = (arguments.action or "").strip().lower()

        if action == "list":
            entries = (
                self._store.list()
                if isinstance(self._store, MemoryStore)
                else await self._store.list()
            )
            if not entries:
                return ToolResult(output="Memory is empty.")
            total_chars = sum(len(entry.content) for entry in entries)
            lines = [f"{len(entries)} memory entries ({total_chars:,} chars):"]
            lines += [f"- {e.name} — {e.title} ({len(e.content):,} chars)" for e in entries]
            return ToolResult(output="\n".join(lines))

        if action == "get":
            if not arguments.name.strip():
                return ToolResult(output="Provide 'name' for action='get'.", is_error=True)
            entry = (
                self._store.get(arguments.name)
                if isinstance(self._store, MemoryStore)
                else await self._store.get(arguments.name)
            )
            if entry is None:
                return ToolResult(output=f"No memory entry {arguments.name!r}.", is_error=True)
            # The agent pulled this fact → it was useful. Bump its access count so
            # load_memory_prompt injects it ahead of cold entries next time.
            if isinstance(self._store, MemoryStore):
                self._store.record_use(entry.name)
            else:
                await self._store.record_use(entry.name)
            return ToolResult(
                output=f"# {entry.title} ({entry.name})\n\n{entry.content}",
                metadata={"memory_used": entry.name},
            )

        if action == "search":
            if not arguments.query.strip():
                return ToolResult(output="Provide 'query' for action='search'.", is_error=True)
            if not isinstance(self._store, MemoryStore):
                hits = await self._store.search(
                    arguments.query,
                    max(1, min(arguments.top_k, 10)),
                )
                lines = [f"{len(hits)} memory hits for {arguments.query!r}:"]
                lines.extend(f"- {hit.name} (rank {hit.rank}): {hit.snippet}" for hit in hits)
                return ToolResult(
                    output="\n".join(lines),
                    metadata={"memory_search_hits": [hit.name for hit in hits]},
                )
            return await _search_memory(arguments.query, max(1, min(arguments.top_k, 10)))

        if action == "add":
            result = (
                self._store.add(arguments.title, arguments.content)
                if isinstance(self._store, MemoryStore)
                else await self._store.add(arguments.title, arguments.content)
            )
            return self._result(result)

        if action == "update":
            if not arguments.name.strip():
                return ToolResult(output="Provide 'name' for action='update'.", is_error=True)
            result = (
                self._store.update(
                    arguments.name,
                    arguments.content,
                    title=arguments.title or None,
                )
                if isinstance(self._store, MemoryStore)
                else await self._store.update(
                    arguments.name,
                    arguments.content,
                    title=arguments.title or None,
                )
            )
            return self._result(result)

        if action == "remove":
            if not arguments.name.strip():
                return ToolResult(output="Provide 'name' for action='remove'.", is_error=True)
            result = (
                self._store.remove(arguments.name)
                if isinstance(self._store, MemoryStore)
                else await self._store.remove(arguments.name)
            )
            return self._result(result)

        return ToolResult(
            output=f"Unknown action {action!r}. Use add | update | remove | list | get | search.",
            is_error=True,
        )

    @staticmethod
    def _result(result: MemoryOpResult) -> ToolResult:
        output = result.message
        if result.entries is not None:
            listing = "\n".join(
                f"- {e.name} — {e.title} ({len(e.content):,} chars)" for e in result.entries
            )
            output = f"{output}\n\nCurrent entries:\n{listing}" if listing else output
        return ToolResult(output=output, is_error=not result.ok)


async def _search_memory(query: str, top_k: int) -> ToolResult:
    try:
        process = await asyncio.create_subprocess_exec(
            _DOCUMENT_SEARCH_CLI,
            "search",
            query,
            "--collection",
            _MEMORY_SEARCH_COLLECTIONS,
            "--top-k",
            str(top_k),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return ToolResult(output="Memory search is unavailable: document_search CLI not found.", is_error=True)
    except Exception:
        return ToolResult(output="Memory search could not be started.", is_error=True)

    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=30)
    except asyncio.TimeoutError:
        with suppress(Exception):
            process.kill()
            await process.wait()
        return ToolResult(output="Memory search timed out after 30 seconds.", is_error=True)
    except Exception:
        return ToolResult(output="Memory search failed while reading results.", is_error=True)

    if process.returncode != 0:
        return ToolResult(output=f"Memory search failed with exit status {process.returncode}.", is_error=True)

    try:
        hits = json.loads(stdout.decode("utf-8"))
        if not isinstance(hits, list):
            raise ValueError("search output is not a list")
        lines = [f"{len(hits)} memory hits for {query!r}:"]
        names: list[str] = []
        for hit in hits:
            source_path = hit["source_path"]
            score = float(hit["score"])
            snippet = hit["snippet"]
            if not isinstance(source_path, str) or not isinstance(snippet, str):
                raise ValueError("invalid search hit")
            path = Path(source_path)
            name = path.stem if path.name.lower().endswith(".md") else path.name
            compact_snippet = " ".join(snippet.split())
            if len(compact_snippet) > 240:
                compact_snippet = compact_snippet[:237].rstrip() + "..."
            names.append(name)
            lines.append(f"- {name} (score {score:.2f}): {compact_snippet}")
    except Exception:
        return ToolResult(output="Memory search returned invalid JSON results.", is_error=True)

    return ToolResult(output="\n".join(lines), metadata={"memory_search_hits": names})

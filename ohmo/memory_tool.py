"""Model-callable ``memory`` tool for the ohmo gateway.

Gives the agent a disciplined way to curate its own durable memory — the layer
Hermes calls "agent self-curation". Backed by :class:`ohmo.memory_store.MemoryStore`
(workspace-scoped, shared across this owner's chats), so entries land in the same
``~/.ohmo/memory/`` files the system prompt injects and ``/memory`` reads.

Replaces ad-hoc ``write_file`` into ``memory/`` (which hit the ASCII-slug clobber
bug and had no bounds/dedup). The tool's description carries the collect/refuse
policy so the model saves the right things.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

from ohmo.memory_store import MemoryOpResult, MemoryStore

_READ_ACTIONS = {"list", "get"}


class OhmoMemoryToolInput(BaseModel):
    action: str = Field(
        description=(
            "One of: 'add' (new durable fact), 'update' (replace an existing "
            "entry's content), 'remove' (delete an entry), 'list' (titles + sizes), "
            "'get' (read one entry's full text)."
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
        "actions: add(title, content) · update(name, content) · remove(name) · list · get(name)."
    )
    input_model = OhmoMemoryToolInput

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def is_read_only(self, arguments: BaseModel) -> bool:
        return getattr(arguments, "action", "").strip().lower() in _READ_ACTIONS

    async def execute(
        self, arguments: OhmoMemoryToolInput, context: ToolExecutionContext
    ) -> ToolResult:
        action = (arguments.action or "").strip().lower()

        if action == "list":
            entries = self._store.list()
            if not entries:
                return ToolResult(output="Memory is empty.")
            lines = [f"{len(entries)} memory entries ({self._store.total_chars():,} chars):"]
            lines += [f"- {e.name} — {e.title} ({len(e.content):,} chars)" for e in entries]
            return ToolResult(output="\n".join(lines))

        if action == "get":
            if not arguments.name.strip():
                return ToolResult(output="Provide 'name' for action='get'.", is_error=True)
            entry = self._store.get(arguments.name)
            if entry is None:
                return ToolResult(output=f"No memory entry {arguments.name!r}.", is_error=True)
            # The agent pulled this fact → it was useful. Bump its access count so
            # load_memory_prompt injects it ahead of cold entries next time.
            self._store.record_use(entry.name)
            return ToolResult(
                output=f"# {entry.title} ({entry.name})\n\n{entry.content}",
                metadata={"memory_used": entry.name},
            )

        if action == "add":
            return self._result(self._store.add(arguments.title, arguments.content))

        if action == "update":
            if not arguments.name.strip():
                return ToolResult(output="Provide 'name' for action='update'.", is_error=True)
            return self._result(
                self._store.update(arguments.name, arguments.content, title=arguments.title or None)
            )

        if action == "remove":
            if not arguments.name.strip():
                return ToolResult(output="Provide 'name' for action='remove'.", is_error=True)
            return self._result(self._store.remove(arguments.name))

        return ToolResult(
            output=f"Unknown action {action!r}. Use add | update | remove | list | get.",
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

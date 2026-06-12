"""ohmo's session-scoped ``todo_write``.

Routes the to-do list through :class:`~ohmo.todo_store.TodoStore` (one file per
session, keyed by the live ``session_id``) instead of the shared ``<cwd>/TODO.md``,
so chats and ``/new``-separated tasks never share or inherit each other's items.
Adds ``new_list`` to start a clean list for an unrelated task mid-conversation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from pydantic import Field

from openharness.tools.base import ToolExecutionContext, ToolResult
from openharness.tools.todo_write_tool import TodoWriteTool, TodoWriteToolInput

from ohmo.todo_store import TodoStore


class OhmoTodoWriteToolInput(TodoWriteToolInput):
    """Adds ``new_list`` on top of the base add/check/remove/clear_completed."""

    new_list: bool = Field(
        default=False,
        description=(
            "Start a FRESH to-do list for a new, unrelated task. Archives the "
            "current list to its own file and points this chat at a new empty "
            "one — use it instead of carrying a previous task's items forward."
        ),
    )


class OhmoTodoWriteTool(TodoWriteTool):
    """Per-session ``todo_write``: the list lives in a TodoStore file keyed by
    the live ``session_id`` (read at call time, so it tracks ``/new``)."""

    input_model = OhmoTodoWriteToolInput

    def __init__(self, store: TodoStore, get_session_id: Callable[[], str]):
        self._store = store
        self._get_session_id = get_session_id

    def _resolve_path(self, arguments: TodoWriteToolInput, context: ToolExecutionContext) -> Path:
        return self._store.active_path(self._get_session_id())

    async def execute(
        self, arguments: OhmoTodoWriteToolInput, context: ToolExecutionContext
    ) -> ToolResult:
        if getattr(arguments, "new_list", False):
            path = self._store.new_list(self._get_session_id())
            return ToolResult(
                output=f"Started a fresh to-do list ({path.name}); the previous one is archived."
            )
        return await super().execute(arguments, context)

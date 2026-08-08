"""OHMO's model-facing, session-scoped todo snapshot tool."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ohmo.todo_store import (
    MAX_BLOCKED_REASON_LENGTH,
    MAX_TODO_CONTENT_LENGTH,
    MAX_TODOS,
    TodoStore,
    canonicalize_text,
)
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


class OhmoTodoItem(BaseModel):
    """One item in the complete desired todo state."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(max_length=MAX_TODO_CONTENT_LENGTH)
    status: Literal["pending", "in_progress", "completed", "blocked"]
    blocked_reason: str | None = Field(default=None, max_length=MAX_BLOCKED_REASON_LENGTH)

    @field_validator("content", "blocked_reason", mode="before")
    @classmethod
    def _canonicalize_text(cls, value: object) -> object:
        return canonicalize_text(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _validate_blocked_reason(self) -> OhmoTodoItem:
        if self.status == "blocked":
            if not self.blocked_reason:
                raise ValueError("blocked todo requires a non-empty blocked_reason")
        elif self.blocked_reason is not None:
            raise ValueError("blocked_reason is only allowed for blocked todos")
        if not self.content:
            raise ValueError("todo content must not be empty")
        return self


class OhmoTodoWriteToolInput(BaseModel):
    """The only OHMO model-facing todo API: a complete typed snapshot."""

    model_config = ConfigDict(extra="forbid")

    todos: list[OhmoTodoItem] = Field(max_length=MAX_TODOS)

    @model_validator(mode="after")
    def _validate_snapshot(self) -> OhmoTodoWriteToolInput:
        identities: set[str] = set()
        in_progress = 0
        for item in self.todos:
            identity = item.content.casefold()
            if identity in identities:
                raise ValueError(f"duplicate todo content: {item.content!r}")
            identities.add(identity)
            if item.status == "in_progress":
                in_progress += 1
        if in_progress > 1:
            raise ValueError("at most one todo may be in_progress")
        return self

    def canonical_todos(self) -> list[dict[str, str]]:
        return [item.model_dump(mode="json", exclude_none=True) for item in self.todos]


class OhmoTodoWriteTool(BaseTool):
    """Atomically replace the active session's todo list."""

    name = "todo_write"
    description = (
        "Replace the complete OHMO todo snapshot. Submit every item in `todos`; "
        "use pending, in_progress, completed, or blocked (with blocked_reason). "
        "Send todos=[] to clear the list. At most one item may be in_progress."
    )
    input_model = OhmoTodoWriteToolInput

    def __init__(self, store: TodoStore, get_session_id: Callable[[], str]):
        self._store = store
        self._get_session_id = get_session_id

    async def execute(
        self, arguments: OhmoTodoWriteToolInput, context: ToolExecutionContext
    ) -> ToolResult:
        del context
        todos = arguments.canonical_todos()
        try:
            changed = self._store.replace_snapshot(self._get_session_id(), todos)
        except OSError as exc:
            return ToolResult(output=f"Todo storage error: {exc}", is_error=True)
        except (TypeError, ValueError) as exc:
            return ToolResult(output=f"Invalid todo snapshot: {exc}", is_error=True)

        payload = {"todos": todos, "changed": changed}
        return ToolResult(
            output=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            metadata=payload,
        )

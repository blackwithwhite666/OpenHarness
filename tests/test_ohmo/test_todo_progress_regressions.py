"""Focused regression coverage for OHMO todo progress boundaries."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

from ohmo.gateway.runtime import OhmoSessionRuntimePool
from ohmo.todo_store import TodoStore
from ohmo.todo_write_tool import OhmoTodoWriteTool, OhmoTodoWriteToolInput
from openharness.channels.bus.events import InboundMessage
from openharness.engine.stream_events import ToolExecutionCompleted
from openharness.tools.base import ToolExecutionContext


async def test_noop_todo_write_still_renders_the_full_checklist(tmp_path: Path):
    """Characterize the runtime's post-write checklist emission for a no-op."""
    store = TodoStore(tmp_path)
    sid = "noop-progress-01"
    tool = OhmoTodoWriteTool(store, lambda: sid)
    context = ToolExecutionContext(cwd=tmp_path)

    snapshot = OhmoTodoWriteToolInput(todos=[{"content": "Step A", "status": "pending"}])
    await tool.execute(snapshot, context)
    result = await tool.execute(snapshot, context)

    runtime = object.__new__(OhmoSessionRuntimePool)
    runtime._todo_store = store
    event = ToolExecutionCompleted(tool_name="todo_write", output=result.output)
    message = InboundMessage(
        channel="telegram", sender_id="synthetic-user", chat_id="synthetic-chat", content="check"
    )
    updates = [
        update
        async for update in runtime._convert_stream_event(
            event=event,
            bundle=SimpleNamespace(session_id=sid),
            message=message,
            session_key="telegram:synthetic-chat",
            content=message.content,
            reply_parts=[],
        )
    ]

    assert '"changed": false' in result.output
    assert [update.text for update in updates] == ["📋 To-do\n⬜ Step A"]


async def test_completed_snapshot_is_retained_until_successful_finalization(tmp_path: Path):
    """Future runtime seam: cleanup happens after, not during, the todo write."""
    store = TodoStore(tmp_path)
    sid = "finalize-01"
    active = store.active_path(sid)
    active.write_text("# TODO\n- [x] Step A\n", encoding="utf-8")

    # The completed snapshot remains available while the runtime is preparing
    # the final answer.
    assert "- [x] Step A" in active.read_text(encoding="utf-8")

    runtime = object.__new__(OhmoSessionRuntimePool)
    runtime._todo_store = store
    finalizer = getattr(runtime, "_finalize_todo_after_successful_answer", None)
    assert callable(finalizer), (
        "runtime must expose the final-answer boundary that owns todo cleanup"
    )

    result = finalizer(session_id=sid)
    if inspect.isawaitable(result):
        await result

    assert "- [x]" not in store.active_path(sid).read_text(encoding="utf-8")

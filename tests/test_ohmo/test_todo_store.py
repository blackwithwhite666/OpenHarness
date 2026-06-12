"""Per-session to-do storage: file-per-session isolation + new_list rotation.

Regression target (2026-06-12): one shared ``<cwd>/TODO.md`` leaked a finished
task's items into every other chat. Lists must now be per-session_id, with a
pointer so a ``new_list`` rotation survives a restart, and old files kept.
"""

from __future__ import annotations

from pathlib import Path

from openharness.tools.base import ToolExecutionContext
from ohmo.todo_store import TodoStore
from ohmo.todo_write_tool import OhmoTodoWriteTool, OhmoTodoWriteToolInput


# ----- TodoStore -----

def test_active_path_is_per_session(tmp_path: Path):
    store = TodoStore(tmp_path)
    a = store.active_path("aaaaaaaaaaaa")
    b = store.active_path("bbbbbbbbbbbb")
    assert a != b
    assert a == tmp_path / "todos" / "aaaaaaaaaaaa.md"
    assert a.parent.is_dir()  # dir auto-created


def test_new_list_rotates_and_keeps_old(tmp_path: Path):
    store = TodoStore(tmp_path)
    sid = "deadbeef0001"
    first = store.active_path(sid)
    first.write_text("# TODO\n- [ ] old task\n", encoding="utf-8")

    second = store.new_list(sid)
    assert second.name == f"{sid}-2.md"
    assert store.active_path(sid) == second           # pointer moved
    assert first.exists() and "old task" in first.read_text()  # old kept
    assert second.read_text() == "# TODO\n"           # new is empty

    third = store.new_list(sid)
    assert third.name == f"{sid}-3.md"


def test_pointer_survives_a_fresh_store(tmp_path: Path):
    sid = "cafecafecafe"
    rotated = TodoStore(tmp_path).new_list(sid)
    # A brand-new TodoStore (e.g. after a gateway restart) must resolve to the
    # rotated list, not back to the default.
    assert TodoStore(tmp_path).active_path(sid) == rotated


def test_session_id_is_sanitized_for_filename(tmp_path: Path):
    store = TodoStore(tmp_path)
    path = store.active_path("../../etc/passwd")
    assert path.parent == tmp_path / "todos"  # no path traversal
    assert "/" not in path.name


# ----- OhmoTodoWriteTool -----

def _ctx(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(cwd=tmp_path)  # cwd is ignored by the ohmo tool


async def test_tool_writes_to_session_file(tmp_path: Path):
    store = TodoStore(tmp_path)
    session_id = "11112222ABCD"
    tool = OhmoTodoWriteTool(store, lambda: session_id)

    await tool.execute(OhmoTodoWriteToolInput(item="step one"), _ctx(tmp_path))

    target = store.active_path(session_id)
    assert "- [ ] step one" in target.read_text()


async def test_tool_sessions_do_not_share(tmp_path: Path):
    store = TodoStore(tmp_path)
    current = {"sid": "session-A"}
    tool = OhmoTodoWriteTool(store, lambda: current["sid"])

    await tool.execute(OhmoTodoWriteToolInput(item="A-task"), _ctx(tmp_path))
    current["sid"] = "session-B"  # lazy session_id → a different chat
    result = await tool.execute(OhmoTodoWriteToolInput(item="B-task"), _ctx(tmp_path))

    a = store.active_path("session-A").read_text()
    b = store.active_path("session-B").read_text()
    assert "A-task" in a and "B-task" not in a
    assert "B-task" in b and "A-task" not in b
    assert not result.is_error


async def test_tool_new_list_rotates(tmp_path: Path):
    store = TodoStore(tmp_path)
    sid = "rotate-me-01"
    tool = OhmoTodoWriteTool(store, lambda: sid)

    await tool.execute(OhmoTodoWriteToolInput(item="first task"), _ctx(tmp_path))
    res = await tool.execute(OhmoTodoWriteToolInput(new_list=True), _ctx(tmp_path))
    assert "fresh" in res.output.lower()

    # New active list is empty; the old task lives in the archived file.
    assert "first task" not in store.active_path(sid).read_text()
    await tool.execute(OhmoTodoWriteToolInput(item="second task"), _ctx(tmp_path))
    assert "second task" in store.active_path(sid).read_text()
    assert "first task" not in store.active_path(sid).read_text()

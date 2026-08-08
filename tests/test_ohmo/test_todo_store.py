"""OHMO typed todo snapshots and legacy Markdown migration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ohmo.todo_store import TodoStore, parse_todo_markdown, render_todo_markdown
from ohmo.todo_write_tool import OhmoTodoWriteTool, OhmoTodoWriteToolInput
from openharness.tools.base import ToolExecutionContext


def _ctx(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(cwd=tmp_path)


def _input(*items: dict[str, str]) -> OhmoTodoWriteToolInput:
    return OhmoTodoWriteToolInput(todos=list(items))


def test_active_path_is_per_session_and_sanitized(tmp_path: Path):
    store = TodoStore(tmp_path)
    assert store.active_path("aaaaaaaaaaaa") == tmp_path / "todos" / "aaaaaaaaaaaa.md"
    assert store.active_path("../../etc/passwd").parent == tmp_path / "todos"


@pytest.mark.parametrize("pointer", [123, None, "", "../outside", "nested/list", "list.md"])
def test_malformed_pointer_falls_back_to_session_list(tmp_path: Path, pointer: object):
    store = TodoStore(tmp_path)
    store.active_path("safe-session")
    (store.dir / "active.json").write_text(
        json.dumps({"safe-session": pointer}), encoding="utf-8"
    )

    path = store.active_path("safe-session")
    assert path == store.dir / "safe-session.md"
    assert path.parent == store.dir


def test_pointer_accepts_only_safe_list_id(tmp_path: Path):
    store = TodoStore(tmp_path)
    store.active_path("safe-session")
    (store.dir / "active.json").write_text(
        json.dumps({"safe-session": "safe-session-2"}), encoding="utf-8"
    )

    assert store.active_path("safe-session") == store.dir / "safe-session-2.md"


def test_new_list_rotates_and_keeps_archived_file(tmp_path: Path):
    store = TodoStore(tmp_path)
    sid = "deadbeef0001"
    first = store.active_path(sid)
    first.write_text("# TODO\n- [ ] old task\n", encoding="utf-8")

    second = store.new_list(sid)
    assert second.name == f"{sid}-2.md"
    assert store.active_path(sid) == second
    assert first.exists() and "old task" in first.read_text(encoding="utf-8")
    assert second.read_text(encoding="utf-8") == "# TODO\n"

    third = store.new_list(sid)
    assert third.name == f"{sid}-3.md"
    assert first.exists()


def test_pointer_survives_restart(tmp_path: Path):
    sid = "cafecafecafe"
    rotated = TodoStore(tmp_path).new_list(sid)
    assert TodoStore(tmp_path).active_path(sid) == rotated


def test_schema_is_model_facing_snapshot_only():
    schema = OhmoTodoWriteToolInput.model_json_schema()
    assert set(schema["properties"]) == {"todos"}
    assert schema["required"] == ["todos"]
    assert schema["additionalProperties"] is False
    item_schema = schema["$defs"]["OhmoTodoItem"]
    assert set(item_schema["properties"]) == {"content", "status", "blocked_reason"}
    assert item_schema["additionalProperties"] is False

    with pytest.raises(ValidationError):
        OhmoTodoWriteToolInput.model_validate({"todos": [], "checked": True})
    with pytest.raises(ValidationError):
        OhmoTodoWriteToolInput.model_validate({"item": "legacy"})


def test_canonicalizes_content_and_reason():
    snapshot = OhmoTodoWriteToolInput.model_validate(
        {
            "todos": [
                {
                    "content": "  Cafe\u0301\t  with  spaces ",
                    "status": "blocked",
                    "blocked_reason": "  waiting\nfor\u00a0input  ",
                }
            ]
        }
    )
    assert snapshot.canonical_todos() == [
        {
            "content": "Café with spaces",
            "status": "blocked",
            "blocked_reason": "waiting for input",
        }
    ]


def test_markdown_round_trip_preserves_marker_like_special_text():
    todos = [
        {
            "content": "  Fix *markdown* [x] — blocked_reason: not metadata ✅ Привет 世界  ",
            "status": "blocked",
            "blocked_reason": "Reason — blocked_reason: keep this, `code`, #tag, 🚧, café 世界",
        }
    ]

    rendered = render_todo_markdown(todos)

    assert parse_todo_markdown(rendered) == [
        {
            "content": "Fix *markdown* [x] — blocked_reason: not metadata ✅ Привет 世界",
            "status": "blocked",
            "blocked_reason": "Reason — blocked_reason: keep this, `code`, #tag, 🚧, café 世界",
        }
    ]


@pytest.mark.asyncio
async def test_round_trip_output_and_metadata_for_all_statuses(tmp_path: Path):
    store = TodoStore(tmp_path)
    tool = OhmoTodoWriteTool(store, lambda: "unicode-session")
    value = _input(
        {"content": "pending ✅", "status": "pending"},
        {"content": "active", "status": "in_progress"},
        {"content": "done — готово", "status": "completed"},
        {"content": "blocked", "status": "blocked", "blocked_reason": "нужен ввод"},
    )

    result = await tool.execute(value, _ctx(tmp_path))
    payload = json.loads(result.output)
    assert payload == result.metadata
    assert payload["changed"] is True
    assert payload["todos"] == value.canonical_todos()
    assert store.read_snapshot("unicode-session")[0] == payload["todos"]
    assert "- [~] active" in store.active_path("unicode-session").read_text(encoding="utf-8")
    assert "blocked_reason: нужен ввод" in store.active_path("unicode-session").read_text(
        encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_identical_snapshot_is_noop_changed_false(tmp_path: Path):
    store = TodoStore(tmp_path)
    tool = OhmoTodoWriteTool(store, lambda: "noop")
    value = _input({"content": "Step A", "status": "pending"})
    await tool.execute(value, _ctx(tmp_path))
    path = store.active_path("noop")
    before = path.read_bytes()

    result = await tool.execute(value, _ctx(tmp_path))
    assert json.loads(result.output) == {"changed": False, "todos": value.canonical_todos()}
    assert result.metadata == json.loads(result.output)
    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_empty_snapshot_atomically_clears(tmp_path: Path):
    store = TodoStore(tmp_path)
    tool = OhmoTodoWriteTool(store, lambda: "clear")
    await tool.execute(_input({"content": "remove me", "status": "completed"}), _ctx(tmp_path))
    result = await tool.execute(OhmoTodoWriteToolInput(todos=[]), _ctx(tmp_path))

    assert json.loads(result.output) == {"changed": True, "todos": []}
    assert store.active_path("clear").read_text(encoding="utf-8") == "# TODO\n"
    assert not list(store.dir.glob("*.tmp"))


@pytest.mark.parametrize(
    "raw",
    [
        {"todos": [{"content": "", "status": "pending"}]},
        {"todos": [{"content": "x", "status": "not-a-status"}]},
        {"todos": [{"content": "x", "status": "pending", "nested": {"x": 1}}]},
        {"todos": [{"content": "x", "status": "blocked"}]},
        {
            "todos": [
                {"content": "x", "status": "blocked", "blocked_reason": " "},
            ]
        },
        {
            "todos": [
                {"content": "x", "status": "pending", "blocked_reason": "why"},
            ]
        },
        {
            "todos": [
                {"content": "x", "status": "in_progress"},
                {"content": "y", "status": "in_progress"},
            ]
        },
        {
            "todos": [
                {"content": "Step A", "status": "pending"},
                {"content": " step\u00a0a ", "status": "completed"},
            ]
        },
        {"todos": [{"content": "x" * 501, "status": "pending"}]},
        {
            "todos": [
                {"content": "x", "status": "blocked", "blocked_reason": "r" * 501}
            ]
        },
        {
            "todos": [
                {"content": f"step-{index}", "status": "pending"}
                for index in range(101)
            ]
        },
    ],
)
def test_snapshot_invariants_are_rejected(raw: dict[str, object]):
    with pytest.raises(ValidationError):
        OhmoTodoWriteToolInput.model_validate(raw)


@pytest.mark.asyncio
async def test_legacy_active_file_migrates_on_semantic_noop_and_archives_survive(
    tmp_path: Path,
):
    store = TodoStore(tmp_path)
    sid = "legacy"
    active = store.active_path(sid)
    legacy = "# TODO\n- [ ] Cafe\u0301  task\n- [X] finished ✅\n"
    active.write_text(legacy, encoding="utf-8")
    archived = store.dir / "legacy-archive.md"
    archived.write_text("# TODO\n- [X] old archive\n", encoding="utf-8")

    tool = OhmoTodoWriteTool(store, lambda: sid)
    result = await tool.execute(
        _input(
            {"content": "Café task", "status": "pending"},
            {"content": "finished ✅", "status": "completed"},
        ),
        _ctx(tmp_path),
    )

    assert json.loads(result.output)["changed"] is False
    assert active.read_text(encoding="utf-8") != legacy
    assert active.read_text(encoding="utf-8") == (
        "# TODO\n- [ ] Café task\n- [x] finished ✅\n"
    )
    assert archived.read_text(encoding="utf-8") == "# TODO\n- [X] old archive\n"
    backups = sorted(store.dir.glob("legacy.md.legacy*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == legacy.encode("utf-8")

    second = await tool.execute(
        _input(
            {"content": "Café task", "status": "pending"},
            {"content": "finished ✅", "status": "completed"},
        ),
        _ctx(tmp_path),
    )
    assert json.loads(second.output)["changed"] is False
    assert sorted(store.dir.glob("legacy.md.legacy*")) == backups


@pytest.mark.parametrize(
    "legacy",
    [
        "# TODO\n- [ ] Cafe\u0301 task\n- [x] CAFÉ  task\n",
        "# TODO\n- [!] blocked without its reason\n",
        "# TODO\n" + "".join(f"- [ ] step-{index}\n" for index in range(101)),
    ],
)
@pytest.mark.asyncio
async def test_valid_replacement_recovers_invalid_legacy_without_loss(
    tmp_path: Path, legacy: str
):
    store = TodoStore(tmp_path)
    sid = "recover"
    active = store.active_path(sid)
    active.write_bytes(legacy.encode("utf-8"))
    existing_archive = store.dir / "recover.md.legacy"
    existing_archive.write_bytes(b"do not touch\n")

    tool = OhmoTodoWriteTool(store, lambda: sid)
    replacement = _input({"content": "Recovered", "status": "pending"})
    result = await tool.execute(replacement, _ctx(tmp_path))

    assert json.loads(result.output) == {
        "changed": True,
        "todos": replacement.canonical_todos(),
    }
    assert active.read_text(encoding="utf-8") == "# TODO\n- [ ] Recovered\n"
    assert existing_archive.read_bytes() == b"do not touch\n"
    backups = sorted(store.dir.glob("recover.md.legacy*"))
    assert len(backups) == 2
    assert any(path.read_bytes() == legacy.encode("utf-8") for path in backups)

    second = await tool.execute(replacement, _ctx(tmp_path))
    assert json.loads(second.output)["changed"] is False
    assert sorted(store.dir.glob("recover.md.legacy*")) == backups


def test_failed_legacy_migration_keeps_active_bytes(tmp_path: Path, monkeypatch):
    store = TodoStore(tmp_path)
    sid = "crash-safe"
    active = store.active_path(sid)
    legacy = b"# TODO\n- [!] blocked without its reason\n"
    active.write_bytes(legacy)

    def fail_replacement(path: Path, content: str) -> None:
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(store, "_atomic_write", fail_replacement)

    with pytest.raises(OSError, match="simulated replacement failure"):
        store.replace_snapshot(sid, [{"content": "Recovered", "status": "pending"}])

    assert active.read_bytes() == legacy
    backups = sorted(store.dir.glob("crash-safe.md.legacy*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == legacy


@pytest.mark.asyncio
async def test_session_isolation_and_restart_read(tmp_path: Path):
    store = TodoStore(tmp_path)
    current = {"sid": "session-A"}
    tool = OhmoTodoWriteTool(store, lambda: current["sid"])
    await tool.execute(_input({"content": "A", "status": "pending"}), _ctx(tmp_path))
    current["sid"] = "session-B"
    await tool.execute(_input({"content": "B", "status": "completed"}), _ctx(tmp_path))

    assert "A" in store.active_path("session-A").read_text(encoding="utf-8")
    assert "B" not in store.active_path("session-A").read_text(encoding="utf-8")
    assert "B" in TodoStore(tmp_path).active_path("session-B").read_text(encoding="utf-8")

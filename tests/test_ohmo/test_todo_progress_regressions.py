"""Focused regression coverage for OHMO todo progress boundaries."""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from ohmo.gateway.runtime import OhmoSessionRuntimePool
from ohmo.todo_store import TodoStore
from ohmo.todo_write_tool import OhmoTodoWriteTool, OhmoTodoWriteToolInput
from openharness.channels.bus.events import InboundMessage
from openharness.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from openharness.engine.query import MaxTurnsExceeded
from openharness.engine.stream_events import (
    AssistantTextDelta,
    ErrorEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from openharness.tools.base import ToolExecutionContext, ToolRegistry


async def test_noop_todo_write_emits_no_progress_event(tmp_path: Path):
    """An unchanged canonical snapshot is a strict runtime no-op."""
    store = TodoStore(tmp_path)
    sid = "noop-progress-01"
    tool = OhmoTodoWriteTool(store, lambda: sid)
    context = ToolExecutionContext(cwd=tmp_path)

    snapshot = OhmoTodoWriteToolInput(todos=[{"content": "Step A", "status": "pending"}])
    _ = await tool.execute(snapshot, context)
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
    assert updates == []


@pytest.mark.parametrize(
    "metadata",
    [
        {"changed": False, "todos": [{"content": "A", "status": "pending"}]},
        {"changed": True, "todos": [{"content": "A", "status": "unknown"}]},
    ],
)
async def test_todo_write_unchanged_or_malformed_metadata_emits_no_event(
    tmp_path: Path, metadata: dict
):
    runtime = object.__new__(OhmoSessionRuntimePool)
    event = ToolExecutionCompleted(
        tool_name="todo_write", output="provider payload", metadata=metadata
    )
    updates = [
        update
        async for update in runtime._convert_stream_event(
            event=event,
            bundle=SimpleNamespace(session_id="session"),
            message=InboundMessage(channel="telegram", sender_id="u", chat_id="c", content="x"),
            session_key="telegram:c",
            content="x",
            reply_parts=[],
        )
    ]
    assert updates == []


async def test_changed_todo_write_emits_canonical_typed_event_without_provider_text(
    tmp_path: Path,
):
    runtime = object.__new__(OhmoSessionRuntimePool)
    event = ToolExecutionCompleted(
        tool_name="todo_write",
        output='{"todos": [{"content": "A", "status": "pending"}]}',
        metadata={
            "changed": True,
            "todos": [
                {"content": "A", "status": "pending"},
                {"content": "B", "status": "blocked", "blocked_reason": "user"},
            ],
        },
    )
    updates = [
        update
        async for update in runtime._convert_stream_event(
            event=event,
            bundle=SimpleNamespace(session_id="session"),
            message=InboundMessage(channel="telegram", sender_id="u", chat_id="c", content="x"),
            session_key="telegram:c",
            content="x",
            reply_parts=[],
        )
    ]
    assert len(updates) == 1
    assert updates[0].text == ""
    assert updates[0].metadata["progress_event"] == {
        "kind": "todo",
        "todos": event.metadata["todos"],
        "changed": True,
        "session_id": "session",
    }


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


def test_runtime_prompt_injects_live_canonical_snapshot(tmp_path: Path):
    store = TodoStore(tmp_path)
    sid = "prompt-session"
    store.replace_snapshot(
        sid,
        [{"content": "Ship it", "status": "blocked", "blocked_reason": "waiting"}],
    )
    runtime = object.__new__(OhmoSessionRuntimePool)
    runtime._todo_store = store
    prompt = runtime._append_todo_runtime_section(SimpleNamespace(session_id=sid), "base")

    assert "# Trusted OHMO Todo Runtime State" in prompt
    assert "session_id: prompt-session" in prompt
    assert json.loads(prompt.split("active_todos_json: ", 1)[1].split("\n", 1)[0]) == [
        {"blocked_reason": "waiting", "content": "Ship it", "status": "blocked"}
    ]


def _lifecycle_runtime(
    store: TodoStore,
    sid: str,
    actions: list[str],
    save_calls=None,
    *,
    initial_action: str = "candidate",
    continue_action: str = "candidate",
):
    class FakeEngine:
        def __init__(self):
            self.messages = []
            self.tool_metadata = {}
            self.max_turns = 3
            self.internal_prompts: list[str] = []
            self.internal_calls = 0
            self.system_prompts: list[str] = []
            self.tool_execution_count = 0

        def set_system_prompt(self, prompt):
            self.system_prompts.append(prompt)

        async def submit_message(self, _message):
            if initial_action == "partial_error":
                yield AssistantTextDelta(text="partial response")
                yield ErrorEvent(message="provider failed")
                return
            self.messages.append(
                ConversationMessage(role="assistant", content=[TextBlock(text="candidate")])
            )
            yield AssistantTextDelta(text="candidate")

        async def submit_internal_message(self, prompt):
            self.internal_prompts.append(prompt)
            action = actions[self.internal_calls]
            self.internal_calls += 1
            if action == "complete":
                store.replace_snapshot(sid, [{"content": "Ship it", "status": "completed"}])
                self.messages.append(
                    ConversationMessage(role="assistant", content=[TextBlock(text="accepted")])
                )
                yield AssistantTextDelta(text="accepted")
            elif action == "tool_pending":
                internal_message = ConversationMessage.from_user_text(prompt)
                self.messages.append(internal_message)
                self.messages.append(
                    ConversationMessage(
                        role="assistant",
                        content=[
                            ToolUseBlock(id="toolu_reconcile_once", name="side_effect", input={})
                        ],
                    )
                )
                self.messages.append(
                    ConversationMessage(
                        role="user",
                        content=[
                            ToolResultBlock(
                                tool_use_id="toolu_reconcile_once", content="side effect done"
                            )
                        ],
                    )
                )
                self.messages.append(
                    ConversationMessage(
                        role="assistant", content=[TextBlock(text="candidate after tool")]
                    )
                )
                self.tool_execution_count += 1
                yield ToolExecutionStarted(
                    tool_name="side_effect", tool_input={}, tool_call_id="toolu_reconcile_once"
                )
                yield ToolExecutionCompleted(
                    tool_name="side_effect",
                    output="side effect done",
                    tool_call_id="toolu_reconcile_once",
                )
                yield AssistantTextDelta(text="candidate after tool")
                self.messages.remove(internal_message)
            elif action == "complete_empty":
                store.replace_snapshot(sid, [{"content": "Ship it", "status": "completed"}])
            elif action == "error":
                yield ErrorEvent(message="provider failed")
            elif action == "max_turns":
                raise MaxTurnsExceeded(1)
            elif action == "tool_error":
                yield ToolExecutionCompleted(
                    tool_name="todo_write",
                    output="todo storage failed",
                    is_error=True,
                    metadata={"changed": False, "todos": []},
                )
            else:
                yield AssistantTextDelta(text="still working")

        async def continue_pending(self, **_kwargs):
            if continue_action == "partial_max_turns":
                yield AssistantTextDelta(text="partial continued response")
                raise MaxTurnsExceeded(1)
            yield AssistantTextDelta(text="continued candidate")

    engine = FakeEngine()
    runtime = object.__new__(OhmoSessionRuntimePool)
    runtime._todo_store = store
    runtime._runtime_system_prompt = lambda *_args, **_kwargs: _resolved("system")
    runtime._maybe_schedule_memory_judge = lambda *_args, **_kwargs: None

    async def save_snapshot(*_args, **_kwargs):
        if save_calls is not None:
            save_calls.append(list(engine.messages))

    runtime._save_snapshot = save_snapshot
    runtime._append_conversation_turn = _append_conversation_turn
    bundle = SimpleNamespace(
        engine=engine,
        session_id=sid,
        enforce_max_turns=False,
        current_settings=lambda: SimpleNamespace(model="test-model"),
    )
    message = InboundMessage(channel="telegram", sender_id="user", chat_id="chat", content="do it")
    return runtime, bundle, message


async def _resolved(value):
    return value


async def _append_conversation_turn(**_kwargs):
    return None


async def test_pending_plan_reconciles_once_and_publishes_only_accepted_final(tmp_path: Path):
    store = TodoStore(tmp_path)
    sid = "reconcile-once"
    store.replace_snapshot(sid, [{"content": "Ship it", "status": "pending"}])
    runtime, bundle, message = _lifecycle_runtime(store, sid, ["complete"])

    updates = [
        update
        async for update in runtime._stream_engine_message(
            bundle=bundle,
            message=message,
            session_key="telegram:chat",
            user_prompt=message.content,
            user_message=message.content,
            turn_ctx=SimpleNamespace(),
            memory_scope=None,
        )
    ]

    assert [update.text for update in updates if update.kind == "final"] == ["accepted"]
    assert bundle.engine.internal_calls == 1
    assert "Ship it" in bundle.engine.internal_prompts[0]


async def test_successful_reconciliation_saves_accepted_history_once(tmp_path: Path):
    store = TodoStore(tmp_path)
    sid = "reconcile-save-once"
    store.replace_snapshot(sid, [{"content": "Ship it", "status": "pending"}])
    saves = []
    runtime, bundle, message = _lifecycle_runtime(store, sid, ["complete"], saves)

    updates = [
        update
        async for update in runtime._stream_engine_message(
            bundle=bundle,
            message=message,
            session_key="telegram:chat",
            user_prompt=message.content,
            user_message=message.content,
            turn_ctx=SimpleNamespace(),
            memory_scope=None,
        )
    ]

    assert len(saves) == 1
    assert [message.text for message in saves[0]] == ["accepted"]
    assert [update.text for update in updates if update.kind == "final"] == ["accepted"]


async def test_pending_plan_reconciliation_is_bounded_and_actionable(tmp_path: Path):
    store = TodoStore(tmp_path)
    sid = "reconcile-bound"
    store.replace_snapshot(sid, [{"content": "Ship it", "status": "pending"}])
    runtime, bundle, message = _lifecycle_runtime(store, sid, ["pending", "pending"])

    updates = [
        update
        async for update in runtime._stream_engine_message(
            bundle=bundle,
            message=message,
            session_key="telegram:chat",
            user_prompt=message.content,
            user_message=message.content,
            turn_ctx=SimpleNamespace(),
            memory_scope=None,
        )
    ]

    assert bundle.engine.internal_calls == 2
    assert not [update for update in updates if update.kind == "final"]
    errors = [update for update in updates if update.kind == "error"]
    assert len(errors) == 1
    assert "Ship it" in errors[0].text


@pytest.mark.parametrize("failure_action", ["error", "tool_error"])
async def test_reconciliation_provider_or_tool_error_does_not_fake_completion(
    tmp_path: Path, failure_action: str
):
    store = TodoStore(tmp_path)
    sid = "reconcile-error"
    store.replace_snapshot(sid, [{"content": "Ship it", "status": "pending"}])
    runtime, bundle, message = _lifecycle_runtime(store, sid, [failure_action])

    updates = [
        update
        async for update in runtime._stream_engine_message(
            bundle=bundle,
            message=message,
            session_key="telegram:chat",
            user_prompt=message.content,
            user_message=message.content,
            turn_ctx=SimpleNamespace(),
            memory_scope=None,
        )
    ]

    assert bundle.engine.internal_calls == 1
    assert not [update for update in updates if update.kind == "final"]
    assert store.read_snapshot(sid)[0][0]["status"] == "pending"
    assert not any(
        (update.metadata.get("progress_event") or {}).get("kind") == "todo" for update in updates
    )


async def test_completed_cleanup_waits_for_final_yield_resume_and_aclose_keeps_plan(
    tmp_path: Path,
):
    store = TodoStore(tmp_path)
    sid = "cleanup-boundary"
    store.replace_snapshot(sid, [{"content": "Ship it", "status": "completed"}])
    runtime, bundle, message = _lifecycle_runtime(store, sid, [])

    stream = runtime._stream_engine_message(
        bundle=bundle,
        message=message,
        session_key="telegram:chat",
        user_prompt=message.content,
        user_message=message.content,
        turn_ctx=SimpleNamespace(),
        memory_scope=None,
    )
    while True:
        update = await anext(stream)
        if update.kind == "final":
            assert store.read_snapshot(sid)[0][0]["status"] == "completed"
            await stream.aclose()
            break

    assert store.read_snapshot(sid)[0][0]["status"] == "completed"


async def test_fully_consumed_successful_final_archives_completed_plan(tmp_path: Path):
    store = TodoStore(tmp_path)
    sid = "cleanup-success"
    store.replace_snapshot(sid, [{"content": "Ship it", "status": "completed"}])
    runtime, bundle, message = _lifecycle_runtime(store, sid, [])

    updates = [
        update
        async for update in runtime._stream_engine_message(
            bundle=bundle,
            message=message,
            session_key="telegram:chat",
            user_prompt=message.content,
            user_message=message.content,
            turn_ctx=SimpleNamespace(),
            memory_scope=None,
        )
    ]

    assert [update.text for update in updates if update.kind == "final"] == ["candidate"]
    assert store.read_snapshot(sid)[0] == []
    cleanup = [
        update
        for update in updates
        if (update.metadata.get("progress_event") or {}).get("kind") == "todo"
    ]
    assert len(cleanup) == 1
    assert cleanup[0].metadata["progress_event"]["todos"] == []


async def test_blocked_final_keeps_panel_state_and_emits_no_cleanup_event(tmp_path: Path):
    store = TodoStore(tmp_path)
    sid = "cleanup-blocked"
    store.replace_snapshot(
        sid, [{"content": "Need input", "status": "blocked", "blocked_reason": "user"}]
    )
    runtime, bundle, message = _lifecycle_runtime(store, sid, [])

    updates = [
        update
        async for update in runtime._stream_engine_message(
            bundle=bundle,
            message=message,
            session_key="telegram:chat",
            user_prompt=message.content,
            user_message=message.content,
            turn_ctx=SimpleNamespace(),
            memory_scope=None,
        )
    ]

    assert [update.text for update in updates if update.kind == "final"] == ["candidate"]
    assert store.read_snapshot(sid)[0][0]["status"] == "blocked"
    assert not any(
        (update.metadata.get("progress_event") or {}).get("kind") == "todo" for update in updates
    )


async def test_cleanup_failure_is_logged_once_and_keeps_recoverable_plan(tmp_path: Path, caplog):
    store = TodoStore(tmp_path)
    sid = "cleanup-failure"
    store.replace_snapshot(sid, [{"content": "Ship it", "status": "completed"}])
    runtime, bundle, message = _lifecycle_runtime(store, sid, [])

    def fail_cleanup(*, session_id: str):
        del session_id
        raise OSError("pointer write failed")

    runtime._finalize_todo_after_successful_answer = fail_cleanup
    with caplog.at_level(logging.WARNING, logger="ohmo.gateway.runtime"):
        updates = [
            update
            async for update in runtime._stream_engine_message(
                bundle=bundle,
                message=message,
                session_key="telegram:chat",
                user_prompt=message.content,
                user_message=message.content,
                turn_ctx=SimpleNamespace(),
                memory_scope=None,
            )
        ]

    assert [update.text for update in updates if update.kind == "final"] == ["candidate"]
    assert not any(
        (update.metadata.get("progress_event") or {}).get("kind") == "todo" for update in updates
    )
    assert store.read_snapshot(sid)[0][0]["status"] == "completed"
    assert sum("ohmo.todo.cleanup_failure" in record.getMessage() for record in caplog.records) == 1


async def test_blocked_plans_are_not_archived_and_reinject_after_fresh_store(tmp_path: Path):
    store = TodoStore(tmp_path)
    for sid, todos in (
        (
            "blocked-only",
            [{"content": "Need input", "status": "blocked", "blocked_reason": "user"}],
        ),
        (
            "completed-blocked",
            [
                {"content": "Done", "status": "completed"},
                {"content": "Need input", "status": "blocked", "blocked_reason": "user"},
            ],
        ),
    ):
        store.replace_snapshot(sid, todos)
        runtime = object.__new__(OhmoSessionRuntimePool)
        runtime._todo_store = store
        assert runtime._finalize_todo_after_successful_answer(session_id=sid) is False
        fresh = TodoStore(tmp_path)
        fresh_runtime = object.__new__(OhmoSessionRuntimePool)
        fresh_runtime._todo_store = fresh
        prompt = fresh_runtime._append_todo_runtime_section(SimpleNamespace(session_id=sid), "base")
        assert "Need input" in prompt


async def test_next_turn_reads_latest_atomic_snapshot(tmp_path: Path):
    store = TodoStore(tmp_path)
    sid = "next-turn"
    store.replace_snapshot(sid, [{"content": "Old", "status": "pending"}])
    runtime = object.__new__(OhmoSessionRuntimePool)
    runtime._todo_store = store
    first = runtime._append_todo_runtime_section(SimpleNamespace(session_id=sid), "base")
    assert '"Old"' in first

    fresh = TodoStore(tmp_path)
    fresh.replace_snapshot(sid, [{"content": "New", "status": "pending"}])
    second = runtime._append_todo_runtime_section(SimpleNamespace(session_id=sid), "base")
    assert '"New"' in second
    assert '"Old"' not in second


def test_history_replacement_does_not_lose_durable_todo_snapshot(tmp_path: Path):
    store = TodoStore(tmp_path)
    sid = "compaction-survives"
    store.replace_snapshot(sid, [{"content": "Keep this", "status": "pending"}])
    runtime = object.__new__(OhmoSessionRuntimePool)
    runtime._todo_store = store
    bundle = SimpleNamespace(session_id=sid, engine=SimpleNamespace())
    bundle.engine.load_messages = lambda messages: setattr(bundle.engine, "messages", messages)
    bundle.engine.load_messages([ConversationMessage.from_user_text("compacted history")])

    fresh_runtime = object.__new__(OhmoSessionRuntimePool)
    fresh_runtime._todo_store = TodoStore(tmp_path)
    prompt = fresh_runtime._append_todo_runtime_section(bundle, "base")

    assert "compacted history" not in prompt
    assert "Keep this" in prompt


async def test_max_turns_does_not_fake_completion_or_clean_unresolved_plan(tmp_path: Path):
    store = TodoStore(tmp_path)
    sid = "reconcile-max-turns"
    store.replace_snapshot(sid, [{"content": "Ship it", "status": "pending"}])
    runtime, bundle, message = _lifecycle_runtime(store, sid, ["max_turns"])

    updates = [
        update
        async for update in runtime._stream_engine_message(
            bundle=bundle,
            message=message,
            session_key="telegram:chat",
            user_prompt=message.content,
            user_message=message.content,
            turn_ctx=SimpleNamespace(),
            memory_scope=None,
        )
    ]

    assert not [update for update in updates if update.kind == "final"]
    assert "unresolved work" in next(update.text for update in updates if update.kind == "error")
    assert store.read_snapshot(sid)[0][0]["status"] == "pending"


@pytest.mark.parametrize("status", ["pending", "completed"])
async def test_partial_provider_error_never_publishes_or_cleans_plan(tmp_path: Path, status: str):
    store = TodoStore(tmp_path)
    sid = f"partial-provider-error-{status}"
    store.replace_snapshot(sid, [{"content": "Ship it", "status": status}])
    runtime, bundle, message = _lifecycle_runtime(store, sid, [], initial_action="partial_error")

    updates = [
        update
        async for update in runtime._stream_engine_message(
            bundle=bundle,
            message=message,
            session_key="telegram:chat",
            user_prompt=message.content,
            user_message=message.content,
            turn_ctx=SimpleNamespace(),
            memory_scope=None,
        )
    ]

    assert not [update for update in updates if update.kind == "final"]
    assert [update.text for update in updates if update.kind == "error"] == ["provider failed"]
    assert store.read_snapshot(sid)[0][0]["status"] == status


@pytest.mark.parametrize("status", ["pending", "completed"])
async def test_partial_continue_max_turns_never_publishes_or_cleans_plan(
    tmp_path: Path, status: str
):
    store = TodoStore(tmp_path)
    sid = f"partial-continue-max-{status}"
    store.replace_snapshot(sid, [{"content": "Ship it", "status": status}])
    runtime, bundle, message = _lifecycle_runtime(
        store, sid, [], continue_action="partial_max_turns"
    )
    result = SimpleNamespace(
        refresh_runtime=False,
        message=None,
        submit_prompt=None,
        submit_model=None,
        continue_pending=True,
        continue_turns=None,
    )

    updates = [
        update
        async for update in runtime._stream_command_result(
            bundle=bundle,
            message=message,
            session_key="telegram:chat",
            user_prompt=message.content,
            result=result,
            turn_ctx=SimpleNamespace(),
            memory_scope=None,
            todo_lifecycle=True,
        )
    ]

    assert not [update for update in updates if update.kind == "final"]
    assert any("max_turns" in update.text for update in updates if update.kind == "error")
    assert store.read_snapshot(sid)[0][0]["status"] == status


async def test_reconciliation_keeps_tool_trace_without_repeating_work(tmp_path: Path):
    store = TodoStore(tmp_path)
    sid = "reconcile-tool-trace"
    store.replace_snapshot(sid, [{"content": "Ship it", "status": "pending"}])
    saves = []
    runtime, bundle, message = _lifecycle_runtime(store, sid, ["tool_pending", "complete"], saves)

    updates = [
        update
        async for update in runtime._stream_engine_message(
            bundle=bundle,
            message=message,
            session_key="telegram:chat",
            user_prompt=message.content,
            user_message=message.content,
            turn_ctx=SimpleNamespace(),
            memory_scope=None,
        )
    ]

    assert bundle.engine.tool_execution_count == 1
    assert bundle.engine.internal_calls == 2
    assert len(bundle.engine.internal_prompts) == 2
    assert not any(
        message.role == "user" and message.text in bundle.engine.internal_prompts
        for message in bundle.engine.messages
    )
    assert (
        sum(
            isinstance(block, ToolUseBlock) and block.id == "toolu_reconcile_once"
            for message in bundle.engine.messages
            for block in message.content
        )
        == 1
    )
    assert (
        sum(
            isinstance(block, ToolResultBlock) and block.tool_use_id == "toolu_reconcile_once"
            for message in bundle.engine.messages
            for block in message.content
        )
        == 1
    )
    assert not any(message.text == "candidate after tool" for message in bundle.engine.messages)
    assert sum(message.text == "accepted" for message in bundle.engine.messages) == 1
    assert [message.text for message in saves[0] if message.text == "accepted"] == ["accepted"]
    assert [update.text for update in updates if update.kind == "final"] == ["accepted"]


async def test_continue_pending_model_final_is_guarded(tmp_path: Path):
    store = TodoStore(tmp_path)
    sid = "continue-guard"
    store.replace_snapshot(sid, [{"content": "Ship it", "status": "pending"}])
    runtime, bundle, message = _lifecycle_runtime(store, sid, ["complete"])
    result = SimpleNamespace(
        refresh_runtime=False,
        message=None,
        submit_prompt=None,
        submit_model=None,
        continue_pending=True,
        continue_turns=None,
    )

    updates = [
        update
        async for update in runtime._stream_command_result(
            bundle=bundle,
            message=message,
            session_key="telegram:chat",
            user_prompt=message.content,
            result=result,
            turn_ctx=SimpleNamespace(),
            memory_scope=None,
            todo_lifecycle=True,
        )
    ]

    assert [update.text for update in updates if update.kind == "final"] == ["accepted"]
    assert "continued candidate" not in [
        update.text for update in updates if update.kind == "final"
    ]


async def test_todo_prompt_read_failure_is_explicit_and_fail_closed(tmp_path: Path):
    class BrokenStore(TodoStore):
        def read_snapshot(self, _session_id):
            raise OSError("snapshot unavailable")

    runtime = object.__new__(OhmoSessionRuntimePool)
    runtime._todo_store = BrokenStore(tmp_path)
    bundle = SimpleNamespace(session_id="broken")
    prompt = runtime._append_todo_runtime_section(bundle, "base")

    assert "TODO_STATE_READ_ERROR" in prompt
    assert "active_todos_json: <UNAVAILABLE>" in prompt
    assert "active_todos_json: []" not in prompt


async def test_todo_prompt_read_failure_logs_once_and_blocks_model_turn(tmp_path: Path, caplog):
    class BrokenStore(TodoStore):
        def read_snapshot(self, _session_id):
            raise OSError("snapshot unavailable")

    class FakeEngine:
        def __init__(self):
            self.messages = []
            self.submit_calls = 0

        def set_system_prompt(self, _prompt):
            return None

        async def submit_message(self, _message):
            self.submit_calls += 1
            yield AssistantTextDelta(text="must not be published")

    runtime = object.__new__(OhmoSessionRuntimePool)
    runtime._todo_store = BrokenStore(tmp_path)
    runtime._runtime_system_prompt = lambda bundle, *_args, **_kwargs: _broken_prompt(
        runtime, bundle
    )
    bundle = SimpleNamespace(session_id="broken-turn", engine=FakeEngine())
    message = InboundMessage(
        channel="telegram", sender_id="user", chat_id="chat", content="continue"
    )

    with caplog.at_level(logging.WARNING, logger="ohmo.gateway.runtime"):
        await runtime._runtime_system_prompt(bundle, "continue")
        await runtime._runtime_system_prompt(bundle, "continue")
        updates = [
            update
            async for update in runtime._stream_engine_message(
                bundle=bundle,
                message=message,
                session_key="telegram:chat",
                user_prompt=message.content,
                user_message=message.content,
                turn_ctx=SimpleNamespace(),
                memory_scope=None,
            )
        ]

    assert bundle.engine.submit_calls == 0
    assert not [update for update in updates if update.kind == "final"]
    assert any(
        "No model response was accepted or published" in update.text
        for update in updates
        if update.kind == "error"
    )
    assert (
        sum("ohmo.todo.prompt.read_failure" in record.getMessage() for record in caplog.records)
        == 1
    )


async def test_resolved_without_final_has_actionable_missing_final_error(tmp_path: Path):
    store = TodoStore(tmp_path)
    sid = "missing-final"
    store.replace_snapshot(sid, [{"content": "Ship it", "status": "pending"}])
    runtime, bundle, message = _lifecycle_runtime(store, sid, ["complete_empty"])

    updates = [
        update
        async for update in runtime._stream_engine_message(
            bundle=bundle,
            message=message,
            session_key="telegram:chat",
            user_prompt=message.content,
            user_message=message.content,
            turn_ctx=SimpleNamespace(),
            memory_scope=None,
        )
    ]

    assert not [update for update in updates if update.kind == "final"]
    error_text = next(update.text for update in updates if update.kind == "error")
    assert "accepted final response is missing" in error_text


async def test_runtime_noop_logging_has_one_lifecycle_event(tmp_path: Path, caplog):
    store = TodoStore(tmp_path)
    sid = "log-noop"
    tool = OhmoTodoWriteTool(store, lambda: sid)
    context = ToolExecutionContext(cwd=tmp_path)
    snapshot = OhmoTodoWriteToolInput(todos=[{"content": "Step", "status": "pending"}])
    _ = await tool.execute(snapshot, context)
    result = await tool.execute(snapshot, context)
    runtime = object.__new__(OhmoSessionRuntimePool)
    runtime._todo_store = store
    with caplog.at_level(logging.INFO, logger="ohmo.gateway.runtime"):
        updates = [
            update
            async for update in runtime._convert_stream_event(
                event=ToolExecutionCompleted(
                    tool_name="todo_write", output=result.output, metadata=result.metadata
                ),
                bundle=SimpleNamespace(session_id=sid),
                message=InboundMessage(
                    channel="telegram", sender_id="user", chat_id="chat", content="check"
                ),
                session_key="telegram:chat",
                content="check",
                reply_parts=[],
            )
        ]

    assert updates == []
    assert sum("ohmo.todo.write.noop" in record.getMessage() for record in caplog.records) == 1


async def test_errored_todo_write_changed_false_is_not_logged_as_noop(tmp_path: Path, caplog):
    runtime = object.__new__(OhmoSessionRuntimePool)
    with caplog.at_level(logging.INFO, logger="ohmo.gateway.runtime"):
        updates = [
            update
            async for update in runtime._convert_stream_event(
                event=ToolExecutionCompleted(
                    tool_name="todo_write",
                    output="todo storage failed",
                    is_error=True,
                    metadata={"changed": False, "todos": []},
                ),
                bundle=SimpleNamespace(session_id="errored-noop"),
                message=InboundMessage(
                    channel="telegram", sender_id="user", chat_id="chat", content="check"
                ),
                session_key="telegram:chat",
                content="check",
                reply_parts=[],
            )
        ]

    assert updates == []
    assert not any("ohmo.todo.write.noop" in record.getMessage() for record in caplog.records)


async def test_blocked_noop_does_not_log_new_blocked_lifecycle_event(tmp_path: Path, caplog):
    store = TodoStore(tmp_path)
    sid = "log-blocked-noop"
    tool = OhmoTodoWriteTool(store, lambda: sid)
    context = ToolExecutionContext(cwd=tmp_path)
    snapshot = OhmoTodoWriteToolInput(
        todos=[{"content": "Need input", "status": "blocked", "blocked_reason": "user"}]
    )
    first_result = await tool.execute(snapshot, context)
    result = await tool.execute(snapshot, context)
    runtime = object.__new__(OhmoSessionRuntimePool)
    runtime._todo_store = store

    with caplog.at_level(logging.INFO, logger="ohmo.gateway.runtime"):
        _ = [
            update
            async for update in runtime._convert_stream_event(
                event=ToolExecutionCompleted(
                    tool_name="todo_write",
                    output=result.output,
                    metadata=result.metadata,
                ),
                bundle=SimpleNamespace(session_id=sid),
                message=InboundMessage(
                    channel="telegram", sender_id="user", chat_id="chat", content="check"
                ),
                session_key="telegram:chat",
                content="check",
                reply_parts=[],
            )
        ]

    assert sum("ohmo.todo.write.noop" in record.getMessage() for record in caplog.records) == 1
    assert not any(
        "ohmo.todo.blocked.persisted" in record.getMessage() for record in caplog.records
    )

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="ohmo.gateway.runtime"):
        _ = [
            update
            async for update in runtime._convert_stream_event(
                event=ToolExecutionCompleted(
                    tool_name="todo_write",
                    output=first_result.output,
                    metadata=first_result.metadata,
                ),
                bundle=SimpleNamespace(session_id=sid),
                message=InboundMessage(
                    channel="telegram", sender_id="user", chat_id="chat", content="check"
                ),
                session_key="telegram:chat",
                content="check",
                reply_parts=[],
            )
        ]
    assert (
        sum("ohmo.todo.blocked.persisted" in record.getMessage() for record in caplog.records) == 1
    )


async def _broken_prompt(runtime, bundle):
    return runtime._append_todo_runtime_section(bundle, "base")


@pytest.mark.parametrize("status", ["pending", "completed"])
async def test_stream_message_synthetic_turn_suspends_todo_lifecycle(
    tmp_path: Path, monkeypatch, status: str
):
    workspace = tmp_path / "workspace"
    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    sid = f"synthetic-turn-{status}"
    pool._todo_store.replace_snapshot(sid, [{"content": "Keep work", "status": status}])
    registry = ToolRegistry()
    registry.register(OhmoTodoWriteTool(pool._todo_store, lambda: sid))
    observations: dict[str, object] = {"prompts": [], "tool_visible": None, "guards": 0}

    class FakeEngine:
        def __init__(self):
            self.messages = []
            self.tool_metadata = {}

        def set_system_prompt(self, prompt):
            observations["prompts"].append(prompt)

        async def submit_message(self, _message):
            observations["tool_visible"] = registry.get("todo_write") is not None
            yield AssistantTextDelta(text="synthetic reply")

    bundle = SimpleNamespace(
        engine=FakeEngine(),
        tool_registry=registry,
        session_id=sid,
        cwd=str(tmp_path),
        commands=SimpleNamespace(lookup=lambda _raw: None),
    )

    async def fake_get_bundle(*_args, **_kwargs):
        return bundle

    async def fake_prompt(_bundle, _latest, **kwargs):
        observations.setdefault("include_todo", []).append(kwargs["include_todo"])
        if kwargs["include_todo"]:
            return pool._append_todo_runtime_section(_bundle, "base")
        return "base"

    async def no_op(*_args, **_kwargs):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime._evals_capture_enabled", lambda _config: False)
    monkeypatch.setattr(pool, "get_bundle", fake_get_bundle)
    monkeypatch.setattr(pool, "_cwd_for_message", lambda _message, _key: str(tmp_path))
    monkeypatch.setattr(pool, "_runtime_system_prompt", fake_prompt)
    monkeypatch.setattr(pool, "_configure_turn_memory_surfaces", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pool, "_resolve_turn_memory_scope", lambda _ctx: None)
    monkeypatch.setattr(pool, "_bind_session_owner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pool, "_set_group_request_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pool, "_restore_group_request_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pool, "_clear_reminder_context", lambda _bundle: None)
    monkeypatch.setattr(pool, "_save_snapshot", no_op)
    monkeypatch.setattr(pool, "_append_conversation_turn", no_op)
    monkeypatch.setattr(pool, "_maybe_schedule_memory_judge", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        pool,
        "_guard_todo_final",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("synthetic guard invoked")),
    )

    message = InboundMessage(
        channel="telegram",
        sender_id="user",
        chat_id="chat",
        content="reminder",
        metadata={"_synthetic": True},
    )
    updates = [u async for u in pool.stream_message(message, "telegram:chat")]

    assert any(update.kind == "final" for update in updates)
    assert observations["tool_visible"] is False
    assert observations["include_todo"] == [False, False]
    assert all(
        "Trusted OHMO Todo Runtime State" not in prompt for prompt in observations["prompts"]
    )
    assert pool._todo_store.read_snapshot(sid)[0][0]["status"] == status
    assert registry.get("todo_write") is not None

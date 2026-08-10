"""Compact progress on Telegram: a turn's ``_collapse`` events fold into one
spinner-animated status message, edited in place and deleted when the final
answer is sent."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from telegram.error import BadRequest, RetryAfter

from openharness.channels.bus.events import OutboundMessage
from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.telegram import (
    _SPINNER_FRAMES,
    TelegramChannel,
    _compact_step_line,
)
from openharness.config.schema import TelegramConfig


class FakeBot:
    def __init__(self, *, retry_after_once: bool = False, not_modified: bool = False):
        self.calls: list[tuple[str, dict]] = []
        self._next_id = 1000
        self._retry_after_once = retry_after_once
        self._not_modified = not_modified

    async def send_message(self, **kwargs):
        self.calls.append(("send_message", kwargs))
        self._next_id += 1
        return SimpleNamespace(message_id=self._next_id)

    async def edit_message_text(self, **kwargs):
        self.calls.append(("edit_message_text", kwargs))
        if self._retry_after_once:
            self._retry_after_once = False
            raise RetryAfter(3)
        if self._not_modified:
            raise BadRequest("Message is not modified")

    async def delete_message(self, **kwargs):
        self.calls.append(("delete_message", kwargs))

    async def send_chat_action(self, **kwargs):
        self.calls.append(("send_chat_action", kwargs))

    def count(self, name: str) -> int:
        return sum(1 for c, _ in self.calls if c == name)


def _channel(bot: FakeBot) -> TelegramChannel:
    channel = TelegramChannel(TelegramConfig(token="token", reply_to_message=True), MessageBus())
    channel._app = SimpleNamespace(bot=bot)
    return channel


def _progress(chat_id: str, text: str) -> OutboundMessage:
    return OutboundMessage(
        channel="telegram",
        chat_id=chat_id,
        content=text,
        metadata={"_progress": True, "_collapse": True},
    )


def _tool_progress(
    chat_id: str,
    text: str,
    *,
    tool_name: str,
    tool_call_id: str,
    display_label: str,
    phase: str,
    status: str,
) -> OutboundMessage:
    return OutboundMessage(
        channel="telegram",
        chat_id=chat_id,
        content=text,
        metadata={
            "_progress": True,
            "_collapse": True,
            "progress_event": {
                "kind": "tool",
                "tool": tool_name,
                "tool_call_id": tool_call_id,
                "display_label": display_label,
                "phase": phase,
                "status": status,
            },
        },
    )


def _todo_progress(
    chat_id: str,
    text: str,
    *,
    snapshot: dict,
    changed: bool,
) -> OutboundMessage:
    return OutboundMessage(
        channel="telegram",
        chat_id=chat_id,
        content=text,
        metadata={
            "_progress": True,
            "_collapse": True,
            "progress_event": {
                "kind": "todo",
                "snapshot": snapshot,
                "changed": changed,
            },
        },
    )


def _final(chat_id: str, text: str) -> OutboundMessage:
    return OutboundMessage(channel="telegram", chat_id=chat_id, content=text, metadata={})


def _debug_todo(chat_id: str, todos: list[dict], *, changed: bool = True, session_id: str = "plan-1") -> OutboundMessage:
    return OutboundMessage(
        channel="telegram",
        chat_id=chat_id,
        content="",
        metadata={
            "_progress": True,
            "progress_event": {
                "kind": "todo",
                "todos": todos,
                "changed": changed,
                "session_id": session_id,
            },
        },
    )


def _kill_anim(channel: TelegramChannel, chat_id: str) -> None:
    st = channel._status.get(chat_id)
    if st and st.anim is not None:
        st.anim.cancel()


def test_compact_step_line_takes_first_nonempty_trimmed():
    assert _compact_step_line("\n🛠️ Bash — a1b2\n<args>\n") == "🛠️ Bash — a1b2"
    assert _compact_step_line("x" * 500).__len__() <= 160


@pytest.mark.asyncio
async def test_first_event_creates_one_status_then_events_coalesce():
    bot = FakeBot()
    ch = _channel(bot)

    await ch.send(_progress("42", "🤔 Думаю…"))
    _kill_anim(ch, "42")
    # Exactly one message created, no per-event edits from send() itself.
    assert bot.count("send_message") == 1
    status = ch._status["42"]
    assert status.message_id == 1001
    assert list(status.lines) == ["🤔 Думаю…"]
    # Spinner frame present in the created text.
    created_text = bot.calls[0][1]["text"]
    assert _SPINNER_FRAMES[0] in created_text

    # A second + third event append lines, mark dirty, but do NOT send/edit.
    await ch.send(_progress("42", "🛠️ Bash — a1b2"))
    await ch.send(_progress("42", "🧠 Свожу результат"))
    assert bot.count("send_message") == 1
    assert bot.count("edit_message_text") == 0
    assert status.dirty is True
    assert list(status.lines)[-1] == "🧠 Свожу результат"


@pytest.mark.asyncio
async def test_final_deletes_status_then_sends_answer():
    bot = FakeBot()
    ch = _channel(bot)

    await ch.send(_progress("42", "🤔 Думаю…"))
    assert "42" in ch._status

    await ch.send(_final("42", "Готовый ответ"))
    # Status message deleted, state cleared, and the answer sent as a new message.
    assert bot.count("delete_message") == 1
    assert bot.calls[-1][0] == "send_message"
    assert bot.calls[-1][1]["text"] == "Готовый ответ" or "Готовый" in bot.calls[-1][1]["text"]
    assert "42" not in ch._status


@pytest.mark.asyncio
async def test_verbose_chat_untouched_no_status():
    bot = FakeBot()
    ch = _channel(bot)
    # No _collapse → normal per-event message, no status, no delete.
    await ch.send(
        OutboundMessage(
            channel="telegram",
            chat_id="42",
            content="🛠️ Bash",
            metadata={"_progress": True},
        )
    )
    assert ch._status == {}
    assert bot.count("send_message") == 1
    assert bot.count("delete_message") == 0


@pytest.mark.asyncio
async def test_edit_status_backs_off_on_retry_after():
    bot = FakeBot(retry_after_once=True)
    ch = _channel(bot)
    await ch.send(_progress("42", "step"))
    status = ch._status["42"]
    _kill_anim(ch, "42")
    before = status.tick
    await ch._edit_status(42, status, "next")  # raises RetryAfter internally, swallowed
    assert status.tick > before  # tick backed off, no exception propagated


@pytest.mark.asyncio
async def test_edit_status_swallows_not_modified():
    bot = FakeBot(not_modified=True)
    ch = _channel(bot)
    await ch.send(_progress("42", "step"))
    status = ch._status["42"]
    _kill_anim(ch, "42")
    await ch._edit_status(42, status, "same")  # BadRequest 'not modified' swallowed
    assert bot.count("edit_message_text") == 1


@pytest.mark.asyncio
async def test_anim_edits_advance_the_spinner():
    bot = FakeBot()
    ch = _channel(bot)
    # Drive one manual edit through the real render path.
    from openharness.channels.impl.telegram import _CompactStatus

    st = _CompactStatus(message_id=1001)
    st.lines.append("шаг")
    st.spinner_idx = 1
    await ch._edit_status(1001, st, ch._render_status(st))
    text = bot.calls[-1][1]["text"]
    assert _SPINNER_FRAMES[1] in text
    assert "шаг" in text


@pytest.mark.asyncio
async def test_tool_start_and_completion_share_one_compact_row():
    """Future contract: structured tool events update one correlated row."""
    bot = FakeBot()
    ch = _channel(bot)

    await ch.send(
        _tool_progress(
            "424242",
            "provider-native start payload",
            tool_name="bash",
            tool_call_id="call-1234",
            display_label="Run command",
            phase="started",
            status="running",
        )
    )
    assert list(ch._status["424242"].lines) == ["Run command ⏳"]
    _kill_anim(ch, "424242")
    await ch.send(
        _tool_progress(
            "424242",
            "provider-native completion payload",
            tool_name="bash",
            tool_call_id="call-1234",
            display_label="Run command",
            phase="completed",
            status="succeeded",
        )
    )

    assert list(ch._status["424242"].lines) == ["Run command ✅"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "label", "expected"),
    [
        ("failed", "Failing tool", "Failing tool ❌"),
        ("cancelled", "Stopped tool", "Stopped tool ⏹️"),
    ],
)
async def test_tool_terminal_statuses_use_redacted_unicode_rows(status, label, expected):
    bot = FakeBot()
    ch = _channel(bot)

    await ch.send(
        _tool_progress(
            "424242",
            "secret args and output",
            tool_name="private_tool",
            tool_call_id="call-terminal",
            display_label=label,
            phase="started",
            status="running",
        )
    )
    _kill_anim(ch, "424242")
    await ch.send(
        _tool_progress(
            "424242",
            "private output hash deadbeef",
            tool_name="private_tool",
            tool_call_id="call-terminal",
            display_label=label,
            phase="completed",
            status=status,
        )
    )

    assert list(ch._status["424242"].lines) == [expected]
    assert "secret" not in ch._render_status(ch._status["424242"])
    assert "call-terminal" not in ch._render_status(ch._status["424242"])
    assert "deadbeef" not in ch._render_status(ch._status["424242"])


@pytest.mark.asyncio
async def test_concurrent_tool_rows_keep_insertion_order():
    bot = FakeBot()
    ch = _channel(bot)

    for call_id, label in (("call-a", "First"), ("call-b", "Second")):
        await ch.send(
            _tool_progress(
                "424242",
                "technical payload",
                tool_name="tool",
                tool_call_id=call_id,
                display_label=label,
                phase="started",
                status="running",
            )
        )
    _kill_anim(ch, "424242")
    await ch.send(
        _tool_progress(
            "424242",
            "technical payload",
            tool_name="tool",
            tool_call_id="call-b",
            display_label="Second",
            phase="completed",
            status="succeeded",
        )
    )

    assert list(ch._status["424242"].lines) == ["First ⏳", "Second ✅"]


@pytest.mark.asyncio
async def test_mixed_progress_keeps_ordinary_tail_and_tool_positions():
    bot = FakeBot()
    ch = _channel(bot)

    for text in ("old one", "old two", "old three"):
        await ch.send(_progress("424242", text))
    for call_id, label in (("call-a", "First tool"), ("call-b", "Second tool")):
        await ch.send(
            _tool_progress(
                "424242",
                "private payload",
                tool_name="tool",
                tool_call_id=call_id,
                display_label=label,
                phase="started",
                status="running",
            )
        )
    await ch.send(_progress("424242", "fresh one"))
    await ch.send(_progress("424242", "fresh two"))
    _kill_anim(ch, "424242")

    await ch.send(
        _tool_progress(
            "424242",
            "private completion payload",
            tool_name="tool",
            tool_call_id="call-b",
            display_label="Second tool",
            phase="completed",
            status="succeeded",
        )
    )

    assert list(ch._status["424242"].lines) == [
        "old three",
        "First tool ⏳",
        "Second tool ✅",
        "fresh one",
        "fresh two",
    ]


@pytest.mark.asyncio
async def test_duplicate_terminal_and_late_start_are_idempotent():
    bot = FakeBot()
    ch = _channel(bot)

    terminal = _tool_progress(
        "424242",
        "technical payload",
        tool_name="tool",
        tool_call_id="call-a",
        display_label="Stable label",
        phase="completed",
        status="succeeded",
    )
    await ch.send(terminal)
    _kill_anim(ch, "424242")
    await ch.send(terminal)
    await ch.send(
        _tool_progress(
            "424242",
            "late start payload",
            tool_name="tool",
            tool_call_id="call-a",
            display_label="Changed label",
            phase="started",
            status="running",
        )
    )

    assert list(ch._status["424242"].lines) == ["Stable label ✅"]


@pytest.mark.asyncio
async def test_terminal_before_start_stays_terminal_and_invalid_ids_do_not_merge():
    bot = FakeBot()
    ch = _channel(bot)

    await ch.send(
        _tool_progress(
            "424242",
            "terminal payload",
            tool_name="tool",
            tool_call_id="call-a",
            display_label="Already done",
            phase="completed",
            status="succeeded",
        )
    )
    _kill_anim(ch, "424242")
    for label in ("No id one", "No id two"):
        await ch.send(
            _tool_progress(
                "424242",
                "untrusted payload",
                tool_name="tool",
                tool_call_id="",
                display_label=label,
                phase="started",
                status="running",
            )
        )
    await ch.send(
        _tool_progress(
            "424242",
            "late start payload",
            tool_name="tool",
            tool_call_id="call-a",
            display_label="Already done",
            phase="started",
            status="running",
        )
    )

    assert list(ch._status["424242"].lines) == [
        "Already done ✅",
        "No id one ⏳",
        "No id two ⏳",
    ]


@pytest.mark.asyncio
async def test_structured_event_is_ignored_by_detailed_mode():
    bot = FakeBot()
    ch = _channel(bot)

    await ch.send(
        OutboundMessage(
            channel="telegram",
            chat_id="424242",
            content="🛠️ Tool — technical arguments",
            metadata={
                "_progress": True,
                "progress_event": {
                    "kind": "tool",
                    "tool": "tool",
                    "tool_call_id": "call-debug",
                    "display_label": "Human label",
                    "phase": "started",
                    "status": "running",
                },
            },
        )
    )

    assert ch._status == {}
    assert bot.calls[-1][0] == "send_message"
    assert bot.calls[-1][1]["text"] == "🛠️ Tool — technical arguments"


@pytest.mark.asyncio
async def test_repeated_todo_snapshots_update_one_compact_panel():
    """Future contract: canonical todo snapshots honor ``changed`` semantics."""
    bot = FakeBot()
    ch = _channel(bot)
    pending = {"todos": [{"content": "Step A", "status": "pending"}]}
    completed = {"todos": [{"content": "Step A", "status": "completed"}]}

    await ch.send(
        _todo_progress(
            "424242",
            "provider-native todo payload 1",
            snapshot=pending,
            changed=True,
        )
    )
    _kill_anim(ch, "424242")
    await ch.send(
        _todo_progress(
            "424242",
            "provider-native todo payload 2",
            snapshot=completed,
            changed=True,
        )
    )
    await ch.send(
        _todo_progress(
            "424242",
            "stale display text must not create a third row",
            snapshot=completed,
            changed=False,
        )
    )

    assert list(ch._status["424242"].lines) == []
    assert ch._status["424242"].todo_text == "📋 To-do\n✅ Step A"


@pytest.mark.asyncio
async def test_compact_todo_empty_removes_only_todo_section_and_final_clears_status():
    bot = FakeBot()
    ch = _channel(bot)
    todo = [{"content": "Plan", "status": "pending"}]
    await ch.send(_progress("42", "ordinary"))
    await ch.send(_todo_progress("42", "", snapshot={"todos": todo}, changed=True))
    assert ch._status["42"].todo_text == "📋 To-do\n⬜ Plan"
    await ch.send(_todo_progress("42", "stale", snapshot={"todos": todo}, changed=False))
    assert ch._status["42"].todo_text == "📋 To-do\n⬜ Plan"
    await ch.send(_todo_progress("42", "", snapshot={"todos": []}, changed=True))
    assert "42" in ch._status
    assert ch._status["42"].todo_text is None
    assert list(ch._status["42"].lines) == ["ordinary"]

    await ch.send(_final("42", "answer"))
    assert "42" not in ch._status


@pytest.mark.asyncio
async def test_compact_todo_only_empty_snapshot_deletes_status_once():
    bot = FakeBot()
    ch = _channel(bot)
    todo = [{"content": "Plan", "status": "pending"}]

    await ch.send(_todo_progress("42", "", snapshot={"todos": todo}, changed=True))
    _kill_anim(ch, "42")
    await ch.send(_todo_progress("42", "", snapshot={"todos": []}, changed=True))

    assert "42" not in ch._status
    assert bot.count("delete_message") == 1


@pytest.mark.asyncio
async def test_compact_todo_rows_are_typed_bounded_and_coexist_with_tools():
    bot = FakeBot()
    ch = _channel(bot)
    rows = [
        {"content": "pending", "status": "pending"},
        {"content": "working", "status": "in_progress"},
        {"content": "done", "status": "completed"},
        {"content": "blocked", "status": "blocked", "blocked_reason": "waiting for user"},
    ] + [{"content": f"long-{i}-" + "x" * 500, "status": "pending"} for i in range(20)]
    await ch.send(_progress("42", "ordinary"))
    await ch.send(
        _tool_progress(
            "42", "payload", tool_name="bash", tool_call_id="call", display_label="Bash",
            phase="started", status="running",
        )
    )
    await ch.send(_todo_progress("42", "", snapshot={"todos": rows}, changed=True))
    text = ch._render_status(ch._status["42"])
    assert len(text) <= 4000
    assert "⬜ pending" in text
    assert "⏳ working" in text
    assert "✅ done" in text
    assert "⛔ blocked — waiting for user" in text
    assert "Bash ⏳" in text
    assert "ordinary" in text


@pytest.mark.asyncio
async def test_debug_todo_panel_sends_once_then_edits_same_message_and_coalesces():
    bot = FakeBot()
    ch = _channel(bot)
    first = [{"content": "A", "status": "pending"}]
    second = [{"content": "A", "status": "completed"}]
    await ch.send(_debug_todo("42", first))
    await ch._flush_todo_panels("42")
    assert bot.count("send_message") == 1
    panel_id = ch._todo_panels["42"].message_id

    await ch.send(_debug_todo("42", first, changed=False))
    await ch._flush_todo_panels("42")
    assert bot.count("edit_message_text") == 0

    await ch.send(_debug_todo("42", second))
    await ch.send(_debug_todo("42", [{"content": "A", "status": "completed"}, {"content": "B", "status": "pending"}]))
    await ch._flush_todo_panels("42")
    assert bot.count("send_message") == 1
    assert bot.count("edit_message_text") == 1
    assert bot.calls[-1][1]["message_id"] == panel_id
    assert "B" in bot.calls[-1][1]["text"]


@pytest.mark.asyncio
async def test_debug_todo_panel_blocked_long_list_and_empty_cleanup():
    bot = FakeBot()
    ch = _channel(bot)
    rows = [{"content": "blocked", "status": "blocked", "blocked_reason": "need input"}]
    rows.extend({"content": str(i) * 500, "status": "pending"} for i in range(20))
    await ch.send(_debug_todo("42", rows))
    await ch._flush_todo_panels("42")
    assert len(bot.calls[-1][1]["text"]) <= 4000
    assert "⛔ blocked — need input" in bot.calls[-1][1]["text"]
    await ch.send(_debug_todo("42", []))
    await ch._flush_todo_panels("42")
    assert bot.count("delete_message") == 1
    assert "42" not in ch._todo_panels


@pytest.mark.asyncio
async def test_debug_todo_blocked_panel_flushes_before_final_and_remains_tracked():
    bot = FakeBot()
    ch = _channel(bot)

    await ch.send(
        _debug_todo(
            "42",
            [{"content": "Need input", "status": "blocked", "blocked_reason": "user"}],
        )
    )
    await ch.send(_final("42", "answer"))

    assert bot.calls[0][0] == "send_message"
    assert bot.calls[-1][0] == "send_message"
    assert bot.calls[-1][1]["text"] == "answer"
    assert "42" in ch._todo_panels
    assert ch._todo_panels["42"].message_id is not None


@pytest.mark.asyncio
async def test_debug_todo_empty_cleanup_flushes_before_final_and_removes_state():
    bot = FakeBot()
    ch = _channel(bot)

    await ch.send(_debug_todo("42", [{"content": "A", "status": "pending"}]))
    await ch._flush_todo_panels("42")
    await ch.send(_debug_todo("42", []))
    await ch.send(_final("42", "answer"))

    assert [name for name, _ in bot.calls] == [
        "send_message",
        "delete_message",
        "send_message",
    ]
    assert bot.calls[-1][1]["text"] == "answer"
    assert "42" not in ch._todo_panels


@pytest.mark.asyncio
async def test_debug_todo_new_session_deletes_old_panel_before_sending_new_one():
    bot = FakeBot()
    ch = _channel(bot)

    await ch.send(_debug_todo("42", [{"content": "old", "status": "pending"}], session_id="old"))
    await ch._flush_todo_panels("42")
    old_message_id = ch._todo_panels["42"].message_id

    await ch.send(_debug_todo("42", [{"content": "new", "status": "pending"}], session_id="new"))
    await ch._flush_todo_panels("42")

    assert [name for name, _ in bot.calls] == [
        "send_message",
        "delete_message",
        "send_message",
    ]
    assert bot.calls[1][1]["message_id"] == old_message_id
    assert ch._todo_panels["42"].plan_id == "new"
    assert ch._todo_panels["42"].message_id != old_message_id


@pytest.mark.asyncio
async def test_debug_todo_writer_is_joined_and_cannot_mutate_after_shutdown():
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingBot(FakeBot):
        async def send_message(self, **kwargs):
            self.calls.append(("send_message", kwargs))
            started.set()
            await release.wait()
            self._next_id += 1
            return SimpleNamespace(message_id=self._next_id)

    bot = BlockingBot()
    ch = _channel(bot)
    ch._app.updater = SimpleNamespace(stop=lambda: asyncio.sleep(0))
    ch._app.stop = lambda: asyncio.sleep(0)
    ch._app.shutdown = lambda: asyncio.sleep(0)

    await ch.send(_debug_todo("42", [{"content": "A", "status": "pending"}]))
    await started.wait()
    task = ch._todo_panels["42"].task
    await ch.stop()
    await asyncio.sleep(0)

    assert task is not None and task.done()
    assert ch._todo_panels == {}
    assert bot.count("send_message") == 1


@pytest.mark.asyncio
async def test_debug_todo_panel_recreates_lost_message_without_duplicates():
    class LostBot(FakeBot):
        def __init__(self):
            super().__init__()
            self.lost_once = True

        async def edit_message_text(self, **kwargs):
            self.calls.append(("edit_message_text", kwargs))
            if self.lost_once:
                self.lost_once = False
                raise BadRequest("Message to edit not found")

    bot = LostBot()
    ch = _channel(bot)
    await ch.send(_debug_todo("42", [{"content": "A", "status": "pending"}]))
    await ch._flush_todo_panels("42")
    first_id = ch._todo_panels["42"].message_id
    await ch.send(_debug_todo("42", [{"content": "A", "status": "completed"}]))
    await ch._flush_todo_panels("42")
    assert bot.count("send_message") == 2
    assert bot.count("edit_message_text") == 1
    assert ch._todo_panels["42"].message_id != first_id


@pytest.mark.asyncio
async def test_debug_todo_panel_handles_retry_after_and_not_modified():
    class RetryBot(FakeBot):
        def __init__(self, error):
            super().__init__()
            self.error = error

        async def edit_message_text(self, **kwargs):
            self.calls.append(("edit_message_text", kwargs))
            if self.error is not None:
                error, self.error = self.error, None
                raise error

    bot = RetryBot(RetryAfter(0))
    ch = _channel(bot)
    await ch.send(_debug_todo("42", [{"content": "A", "status": "pending"}]))
    await ch._flush_todo_panels("42")
    await ch.send(_debug_todo("42", [{"content": "A", "status": "completed"}]))
    await ch._flush_todo_panels("42")
    assert ch._todo_panels["42"].message_id is not None

    not_modified_bot = RetryBot(BadRequest("Message is not modified"))
    not_modified = _channel(not_modified_bot)
    await not_modified.send(_debug_todo("42", [{"content": "A", "status": "pending"}]))
    await not_modified._flush_todo_panels("42")
    await not_modified.send(_debug_todo("42", [{"content": "A", "status": "completed"}]))
    await not_modified._flush_todo_panels("42")
    assert not_modified._todo_panels["42"].message_id is not None


@pytest.mark.asyncio
async def test_debug_todo_panel_does_not_recreate_after_unknown_edit_error():
    class UnknownBot(FakeBot):
        def __init__(self):
            super().__init__()
            self.fail = True

        async def edit_message_text(self, **kwargs):
            self.calls.append(("edit_message_text", kwargs))
            if self.fail:
                self.fail = False
                raise RuntimeError("network status unknown")

    bot = UnknownBot()
    ch = _channel(bot)
    await ch.send(_debug_todo("42", [{"content": "A", "status": "pending"}]))
    await ch._flush_todo_panels("42")
    await ch.send(_debug_todo("42", [{"content": "A", "status": "completed"}]))
    await ch._flush_todo_panels("42")
    await ch.send(_debug_todo("42", [{"content": "A", "status": "blocked", "blocked_reason": "user"}]))
    await ch._flush_todo_panels("42")
    assert bot.count("send_message") == 1
    assert bot.count("edit_message_text") == 2


# ---------------------------------------------------------------------------
# Unified cancelled/terminal progress event (bead agents-playgroud-98i)
# ---------------------------------------------------------------------------


def _cancelled_progress(
    chat_id: str,
    *,
    reason: str = "replaced by a newer user message",
) -> OutboundMessage:
    # Mirrors the bridge's production metadata: a cancelled notice is NOT
    # ``_progress`` — it is durable, so the dispatcher retries it on RetryAfter.
    return OutboundMessage(
        channel="telegram",
        chat_id=chat_id,
        content="",
        metadata={
            "_collapse": True,
            "progress_event": {"kind": "cancelled", "reason": reason},
        },
    )


@pytest.mark.asyncio
async def test_cancelled_event_edits_live_status_to_terminal_and_removes_state():
    """In quiet mode with a live compact status, a cancelled event must stop
    the animator, edit the status once to a terminal phrase, and remove it
    from active state — without sending a standalone notice."""
    bot = FakeBot()
    ch = _channel(bot)
    await ch.send(_progress("424242", "🤔 Думаю…"))
    assert "424242" in ch._status
    status = ch._status["424242"]
    message_id = status.message_id
    _kill_anim(ch, "424242")

    await ch.send(_cancelled_progress("424242"))

    # Status removed from active state, animator cancelled.
    assert "424242" not in ch._status
    # No new send_message — only an edit of the existing status.
    edits = [c for c in bot.calls if c[0] == "edit_message_text"]
    assert len(edits) >= 1
    assert edits[-1][1]["message_id"] == message_id
    # The terminal edit must contain a stop marker, not a spinner.
    assert "⏹" in edits[-1][1]["text"]
    assert _SPINNER_FRAMES[0] not in edits[-1][1]["text"]


@pytest.mark.asyncio
async def test_cancelled_event_without_live_status_sends_standalone_notice():
    """In quiet mode when there is no live compact status (verbose mode or
    already cleared), a cancelled event sends at most one standalone notice."""
    bot = FakeBot()
    ch = _channel(bot)
    # No prior progress — no live status.
    assert "424242" not in ch._status

    await ch.send(_cancelled_progress("424242"))

    sends = [c for c in bot.calls if c[0] == "send_message"]
    assert len(sends) == 1
    assert "⏹" in sends[0][1]["text"]
    # No compact status was created.
    assert "424242" not in ch._status


@pytest.mark.asyncio
async def test_cancelled_then_new_turn_creates_fresh_status():
    """After a cancelled event removes the status, the next turn's first
    progress event must create a fresh status (new message_id)."""
    bot = FakeBot()
    ch = _channel(bot)
    await ch.send(_progress("424242", "first turn"))
    old_message_id = ch._status["424242"].message_id
    _kill_anim(ch, "424242")

    await ch.send(_cancelled_progress("424242"))
    assert "424242" not in ch._status

    await ch.send(_progress("424242", "second turn"))
    assert "424242" in ch._status
    assert ch._status["424242"].message_id != old_message_id
    _kill_anim(ch, "424242")


@pytest.mark.asyncio
async def test_idle_expiry_does_not_render_terminal_stop_text():
    """Idle expiry must not assert 'stopped' while the runtime can still be
    active. It may stop animation but only an explicit lifecycle event may
    render terminal cancellation."""
    bot = FakeBot()
    ch = _channel(bot)
    await ch.send(_progress("424242", "working"))
    status = ch._status["424242"]
    message_id = status.message_id

    edits_before = bot.count("edit_message_text")
    # Force idle: set last_event far in the past.
    status.last_event = time.monotonic() - 999.0
    # Run one animator tick (sleep + check + break).
    await ch._compact_anim("424242", 424242)
    await asyncio.sleep(0.01)

    # Any edit that happened during idle expiry must NOT contain a stop marker.
    for _, kwargs in bot.calls:
        if kwargs.get("message_id") == message_id and "text" in kwargs:
            assert "⏹" not in kwargs["text"], "idle expiry must not claim stopped"


@pytest.mark.asyncio
async def test_cancelled_event_cancels_running_animator():
    """The cancelled event must cancel a still-running animator task so it
    cannot keep editing after terminal cancellation."""
    bot = FakeBot()
    ch = _channel(bot)
    await ch.send(_progress("424242", "working"))
    status = ch._status["424242"]
    anim = status.anim
    assert anim is not None and not anim.done()

    await ch.send(_cancelled_progress("424242"))
    await asyncio.sleep(0.01)
    assert anim.done()


@pytest.mark.asyncio
async def test_cancelled_standalone_notice_send_failure_propagates():
    """With no live compact status, the standalone cancelled notice is a real
    send — a Telegram rejection (e.g. RetryAfter) must NOT be swallowed into a
    false success. It must propagate so ChannelManager can retry (RetryAfter)
    or invoke the failure hook."""
    class _RetryAfterSendBot(FakeBot):
        async def send_message(self, **kwargs):
            raise RetryAfter(0.1)

    bot = _RetryAfterSendBot()
    ch = _channel(bot)

    with pytest.raises(RetryAfter):
        await ch.send(_cancelled_progress("424242"))

    assert "424242" not in ch._status


@pytest.mark.asyncio
async def test_cancelled_live_status_edit_stays_best_effort():
    """With a live compact status, the terminal edit remains best-effort: an
    edit failure must not raise (and must not fall back to a duplicate
    standalone notice)."""
    class _FailingEditBot(FakeBot):
        async def edit_message_text(self, **kwargs):
            raise BadRequest("message to edit not found")

    bot = _FailingEditBot()
    ch = _channel(bot)
    await ch.send(_progress("424242", "working"))
    _kill_anim(ch, "424242")

    await ch.send(_cancelled_progress("424242"))  # must not raise

    assert "424242" not in ch._status
    # No duplicate standalone notice after the failed edit.
    assert bot.count("send_message") == 1  # only the initial status create

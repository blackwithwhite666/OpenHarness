"""Compact progress on Telegram: a turn's ``_collapse`` events fold into one
spinner-animated status message, edited in place and deleted when the final
answer is sent."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from types import SimpleNamespace

import pytest
from telegram.error import BadRequest, RetryAfter

import openharness.channels.impl.telegram as telegram_impl
from openharness.channels.bus.events import OutboundMessage
from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.telegram import (
    _COMPACT_HEADERS_RU,
    _COMPACT_HEARTBEAT_S,
    _SPINNER_FRAMES,
    TelegramChannel,
    _compact_step_line,
    _CompactStatus,
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
    purpose: str | None = None,
) -> OutboundMessage:
    event = {
        "kind": "tool",
        "tool": tool_name,
        "tool_call_id": tool_call_id,
        "display_label": display_label,
        "phase": phase,
        "status": status,
    }
    if purpose is not None:
        event["purpose"] = purpose
    return OutboundMessage(
        channel="telegram",
        chat_id=chat_id,
        content=text,
        metadata={
            "_progress": True,
            "_collapse": True,
            "progress_event": event,
        },
    )


def _inference_progress(chat_id: str, state: str = "active") -> OutboundMessage:
    return OutboundMessage(
        channel="telegram",
        chat_id=chat_id,
        content="",
        metadata={
            "_progress": True,
            "_collapse": True,
            "progress_event": {"kind": "inference", "state": state},
        },
    )


def _tool_rows(channel: TelegramChannel, chat_id: str) -> list[tuple[str, str]]:
    """The visible tool rows (label, state) in insertion order."""
    status = channel._status[chat_id]
    return [
        (label, state)
        for label, state, _terminal in status.tool_rows.values()
    ]


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


def test_compact_headers_ru_are_the_approved_phrase_inventory():
    assert _COMPACT_HEADERS_RU == (
        "Думаю. Прошу не мешать.",
        "Бессвязиц не бывает.",
        "Есть идея.",
        "Вот об этом я сейчас и думаю.",
        "Работать! Работать!",
    )


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
    # Spinner frame + a stable per-turn Russian header in the created text.
    created_text = bot.calls[0][1]["text"]
    assert _SPINNER_FRAMES[0] in created_text
    assert created_text.split(" ", 1)[1] in _COMPACT_HEADERS_RU

    # Content-only events keep the turn alive but change nothing visible, so
    # they trigger neither sends nor edits.
    await ch.send(_progress("42", "🛠️ Bash — a1b2"))
    await ch.send(_progress("42", "🧠 Свожу результат"))
    assert bot.count("send_message") == 1
    assert bot.count("edit_message_text") == 0
    assert status.dirty is False


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
    st = _CompactStatus(message_id=1001, header="Думаю. Прошу не мешать.", inference_active=True)
    st.spinner_idx = 1
    await ch._edit_status(1001, st, ch._render_status(st))
    text = bot.calls[-1][1]["text"]
    assert _SPINNER_FRAMES[1] in text
    assert "Думаю. Прошу не мешать." in text
    assert "Размышляю…" in text


@pytest.mark.asyncio
async def test_tool_start_and_completion_share_one_compact_row():
    """Structured tool events update one row correlated by tool_call_id."""
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
    assert _tool_rows(ch, "424242") == [("Run command", "running")]
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

    assert _tool_rows(ch, "424242") == [("Run command", "success")]
    assert "Run command ✅" in ch._render_status(ch._status["424242"])


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

    rendered = ch._render_status(ch._status["424242"])
    assert expected in rendered
    assert "secret" not in rendered
    assert "call-terminal" not in rendered
    assert "deadbeef" not in rendered


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

    assert _tool_rows(ch, "424242") == [("First", "running"), ("Second", "success")]


@pytest.mark.asyncio
async def test_content_only_events_keep_turn_alive_but_never_render():
    """Quiet mode renders no ordinary lines: narration/status texts only keep
    the live status fresh; the visible state is header + tools + todo."""
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
    _kill_anim(ch, "424242")

    status = ch._status["424242"]
    rendered = ch._render_status(status)
    assert "old one" not in rendered
    assert "fresh one" not in rendered
    assert "First tool ⏳" in rendered
    assert "Second tool ⏳" in rendered


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

    assert _tool_rows(ch, "424242") == [("Stable label", "success")]


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

    assert _tool_rows(ch, "424242") == [
        ("Already done", "success"),
        ("No id one", "running"),
        ("No id two", "running"),
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

    assert list(ch._status["424242"].tool_rows) == []
    assert ch._status["424242"].todo_text == "📋 To-do\n✅ Step A"


@pytest.mark.asyncio
async def test_compact_todo_empty_removes_todo_section_and_final_clears_status():
    bot = FakeBot()
    ch = _channel(bot)
    todo = [{"content": "Plan", "status": "pending"}]
    await ch.send(_inference_progress("42"))
    await ch.send(_todo_progress("42", "", snapshot={"todos": todo}, changed=True))
    assert ch._status["42"].todo_text == "📋 To-do\n⬜ Plan"
    await ch.send(_todo_progress("42", "stale", snapshot={"todos": todo}, changed=False))
    assert ch._status["42"].todo_text == "📋 To-do\n⬜ Plan"
    await ch.send(_todo_progress("42", "", snapshot={"todos": []}, changed=True))
    # The todo block disappears while the turn (inference) is still active.
    assert "42" in ch._status
    assert ch._status["42"].todo_text is None
    rendered = ch._render_status(ch._status["42"])
    assert "📋" not in rendered

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
    # Content-only lines are not rendered; blocks are blank-line separated.
    assert "ordinary" not in text
    head, tool_block, todo_block = text.split("\n\n", 2)
    assert head.split(" ", 1)[1] in _COMPACT_HEADERS_RU
    assert tool_block == "Bash ⏳"
    assert todo_block.startswith("📋 To-do\n")


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
async def test_old_idle_status_remains_active_until_explicit_teardown(monkeypatch):
    """A stale progress event must not stop a still-live compact animator."""
    bot = FakeBot()
    ch = _channel(bot)
    await ch.send(_progress("424242", "working"))
    status = ch._status["424242"]
    monkeypatch.setattr(telegram_impl, "_COMPACT_HEARTBEAT_S", 0.02)
    status.tick = 0.005
    status.last_edit = time.monotonic() - 999.0
    status.last_event = time.monotonic() - 999.0
    anim = asyncio.create_task(ch._compact_anim("424242", 424242))
    status.anim = anim
    await asyncio.sleep(0.05)

    assert "424242" in ch._status
    assert bot.count("edit_message_text") >= 2

    await ch.send(_final("424242", "answer"))
    assert anim.done()


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


# ---------------------------------------------------------------------------
# Three-block compact progress contract (bead agents-playgroud-fe2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_three_block_render_exact_snapshot_and_blank_line_separators():
    ch = _channel(FakeBot())
    st = _CompactStatus(message_id=1, header="Есть идея.")
    st.tool_rows["call-1"] = ("Проверяю расписание поездов", "running", False)
    st.tool_rows["call-2"] = ("Смотрю цены", "success", True)
    st.todo_text = "📋 To-do\n⬜ Купить билет"

    rendered = ch._render_status(st)

    assert rendered == (
        f"{_SPINNER_FRAMES[0]} Есть идея."
        "\n\nПроверяю расписание поездов ⏳\nСмотрю цены ✅"
        "\n\n📋 To-do\n⬜ Купить билет"
    )


@pytest.mark.asyncio
async def test_render_has_no_empty_todo_block_and_no_empty_middle_block():
    ch = _channel(FakeBot())
    st = _CompactStatus(message_id=1, header="Работать! Работать!")

    # Header only: no inference line, no todo block, no trailing separators.
    assert ch._render_status(st) == f"{_SPINNER_FRAMES[0]} Работать! Работать!"

    st.todo_text = "📋 To-do\n✅ Готово"
    assert ch._render_status(st) == (
        f"{_SPINNER_FRAMES[0]} Работать! Работать!\n\n📋 To-do\n✅ Готово"
    )


@pytest.mark.asyncio
async def test_header_is_stable_within_a_turn_and_varies_between_turns():
    bot = FakeBot()
    ch = _channel(bot)

    await ch.send(_progress("42", "turn one"))
    _kill_anim(ch, "42")
    status = ch._status["42"]
    first = status.header
    assert first in _COMPACT_HEADERS_RU
    # Stable across spinner frames and re-renders within the same turn.
    assert ch._render_status(status).split("\n", 1)[0] == f"{_SPINNER_FRAMES[0]} {first}"
    status.spinner_idx = 4
    assert ch._render_status(status).split("\n", 1)[0] == f"{_SPINNER_FRAMES[4]} {first}"

    # Subsequent turns never instantly repeat the previous header.
    seen = {first}
    for _ in range(3):
        await ch.send(_final("42", "answer"))
        await ch.send(_progress("42", "next turn"))
        _kill_anim(ch, "42")
        current = ch._status["42"].header
        assert current in _COMPACT_HEADERS_RU
        seen.add(current)
    assert len(seen) > 1


@pytest.mark.asyncio
async def test_inference_event_renders_exactly_razmyshlyayu_until_first_tool():
    bot = FakeBot()
    ch = _channel(bot)

    await ch.send(_inference_progress("424242", "active"))
    _kill_anim(ch, "424242")
    status = ch._status["424242"]
    rendered = ch._render_status(status)
    blocks = rendered.split("\n\n")
    assert len(blocks) == 2
    assert blocks[1] == "Размышляю…"

    # A tool activity replaces the inference line with the purpose row.
    await ch.send(
        _tool_progress(
            "424242",
            "payload",
            tool_name="bash",
            tool_call_id="call-1",
            display_label="Bash",
            phase="started",
            status="running",
            purpose="Проверяю логи",
        )
    )
    rendered = ch._render_status(ch._status["424242"])
    assert "Размышляю…" not in rendered
    assert "Проверяю логи ⏳" in rendered


@pytest.mark.asyncio
async def test_inference_idle_event_hides_the_inference_line():
    bot = FakeBot()
    ch = _channel(bot)
    await ch.send(_inference_progress("424242", "active"))
    _kill_anim(ch, "424242")
    await ch.send(_inference_progress("424242", "idle"))
    rendered = ch._render_status(ch._status["424242"])
    assert "Размышляю…" not in rendered
    assert "\n\n" not in rendered


@pytest.mark.asyncio
async def test_malformed_inference_event_is_a_strict_noop():
    bot = FakeBot()
    ch = _channel(bot)
    msg = OutboundMessage(
        channel="telegram",
        chat_id="424242",
        content="",
        metadata={
            "_progress": True,
            "_collapse": True,
            "progress_event": {"kind": "inference", "state": "exploding"},
        },
    )
    await ch.send(msg)
    assert ch._status == {}
    assert bot.count("send_message") == 0


@pytest.mark.asyncio
async def test_purpose_is_validated_and_truncated_to_twenty_words():
    bot = FakeBot()
    ch = _channel(bot)
    long_purpose = " ".join(f"слово{i}" for i in range(30))
    await ch.send(
        _tool_progress(
            "424242",
            "payload",
            tool_name="bash",
            tool_call_id="call-1",
            display_label="Bash",
            phase="started",
            status="running",
            purpose=long_purpose,
        )
    )
    _kill_anim(ch, "424242")
    (label, state), = _tool_rows(ch, "424242")
    assert state == "running"
    assert label.endswith("…")
    assert len(label.rstrip("…").split()) == 20


@pytest.mark.asyncio
async def test_purpose_multiline_uses_first_meaningful_line_only():
    bot = FakeBot()
    ch = _channel(bot)
    await ch.send(
        _tool_progress(
            "424242",
            "payload",
            tool_name="bash",
            tool_call_id="call-1",
            display_label="Bash",
            phase="started",
            status="running",
            purpose="\n\n  Первая   строка   с   пробелами  \nВторая строка не должна попасть",
        )
    )
    _kill_anim(ch, "424242")
    (label, _), = _tool_rows(ch, "424242")
    assert label == "Первая строка с пробелами"


@pytest.mark.asyncio
async def test_missing_or_invalid_purpose_falls_back_to_safe_tool_label():
    bot = FakeBot()
    ch = _channel(bot)
    for call_id, purpose in (("call-1", None), ("call-2", 123), ("call-3", "  \n  ")):
        event = {
            "kind": "tool",
            "tool": "read_file",
            "tool_call_id": call_id,
            "display_label": "Read file",
            "phase": "started",
            "status": "running",
        }
        if purpose is not None:
            event["purpose"] = purpose
        await ch.send(
            OutboundMessage(
                channel="telegram",
                chat_id="424242",
                content="payload",
                metadata={"_progress": True, "_collapse": True, "progress_event": event},
            )
        )
    _kill_anim(ch, "424242")
    assert _tool_rows(ch, "424242") == [("Read file", "running")] * 3


@pytest.mark.asyncio
async def test_tool_rows_are_bounded_to_last_n_and_late_completion_does_not_resurrect():
    bot = FakeBot()
    ch = _channel(bot)  # default compact_tool_rows=3

    for i in range(4):
        await ch.send(
            _tool_progress(
                "424242",
                "payload",
                tool_name="tool",
                tool_call_id=f"call-{i}",
                display_label=f"Tool {i}",
                phase="started",
                status="running",
            )
        )
    _kill_anim(ch, "424242")
    # The oldest row was evicted; only the last 3 remain visible.
    assert _tool_rows(ch, "424242") == [
        ("Tool 1", "running"),
        ("Tool 2", "running"),
        ("Tool 3", "running"),
    ]

    # A late completion for the evicted call must not resurrect its row.
    await ch.send(
        _tool_progress(
            "424242",
            "payload",
            tool_name="tool",
            tool_call_id="call-0",
            display_label="Tool 0",
            phase="completed",
            status="succeeded",
        )
    )
    assert _tool_rows(ch, "424242") == [
        ("Tool 1", "running"),
        ("Tool 2", "running"),
        ("Tool 3", "running"),
    ]
    rendered = ch._render_status(ch._status["424242"])
    assert "Tool 0" not in rendered


@pytest.mark.asyncio
async def test_tool_row_limit_is_configurable():
    bot = FakeBot()
    channel = TelegramChannel(
        TelegramConfig(token="token", compact_tool_rows=2), MessageBus()
    )
    channel._app = SimpleNamespace(bot=bot)

    for i in range(3):
        await channel.send(
            _tool_progress(
                "9",
                "payload",
                tool_name="tool",
                tool_call_id=f"call-{i}",
                display_label=f"Tool {i}",
                phase="started",
                status="running",
            )
        )
    _kill_anim(channel, "9")
    assert _tool_rows(channel, "9") == [("Tool 1", "running"), ("Tool 2", "running")]


@pytest.mark.asyncio
async def test_completion_after_eviction_of_another_call_updates_its_own_row():
    bot = FakeBot()
    ch = _channel(bot)
    for i in range(4):
        await ch.send(
            _tool_progress(
                "424242", "payload", tool_name="tool", tool_call_id=f"call-{i}",
                display_label=f"Tool {i}", phase="started", status="running",
            )
        )
    _kill_anim(ch, "424242")
    await ch.send(
        _tool_progress(
            "424242", "payload", tool_name="tool", tool_call_id="call-3",
            display_label="Tool 3", phase="completed", status="failed",
        )
    )
    assert _tool_rows(ch, "424242")[-1] == ("Tool 3", "failure")


@pytest.mark.asyncio
async def test_animator_skips_edits_without_changes_and_heartbeats_at_configured_cadence(monkeypatch):
    bot = FakeBot()
    ch = _channel(bot)
    assert _COMPACT_HEARTBEAT_S == 3.0
    monkeypatch.setattr(telegram_impl, "_COMPACT_HEARTBEAT_S", 0.02)
    await ch.send(_progress("42", "turn start"))
    status = ch._status["42"]
    _kill_anim(ch, "42")
    status.tick = 0.01
    anim = asyncio.create_task(ch._compact_anim("42", 42))
    # Keep the channel's animator-restart check satisfied so it neither spawns
    # a replacement loop nor resets the fast test tick.
    status.anim = anim
    try:
        await asyncio.sleep(0.05)
        # No content change since creation → recurring spinner-only edits.
        edits_before_change = bot.count("edit_message_text")
        assert edits_before_change >= 2

        # A content change is coalesced into exactly one edit.
        await ch.send(
            _tool_progress(
                "42", "payload", tool_name="bash", tool_call_id="call-1",
                display_label="Bash", phase="started", status="running",
            )
        )
        await asyncio.sleep(0.05)
        edits_after_change = bot.count("edit_message_text")
        assert edits_after_change > edits_before_change

        # Still no changes → recurring spinner-only edits at the configured
        # heartbeat cadence.
        await asyncio.sleep(0.05)
        assert bot.count("edit_message_text") > edits_after_change
    finally:
        anim.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await anim


@pytest.mark.asyncio
async def test_compact_edit_failures_warn_except_not_modified(caplog):
    class ErrorBot(FakeBot):
        def __init__(self):
            super().__init__()
            self.error: BaseException | None = None

        async def edit_message_text(self, **kwargs):
            self.calls.append(("edit_message_text", kwargs))
            if self.error is not None:
                raise self.error

    bot = ErrorBot()
    ch = _channel(bot)
    await ch.send(_progress("42", "turn start"))
    status = ch._status["42"]
    _kill_anim(ch, "42")

    with caplog.at_level(logging.WARNING, logger=telegram_impl.__name__):
        bot.error = RetryAfter(0)
        await ch._edit_status(42, status, "retry")
        bot.error = BadRequest("message to edit not found")
        await ch._edit_status(42, status, "failure")

    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert any("rate limited" in message for message in warning_messages)
    assert any("edit failed" in message for message in warning_messages)
    warning_count = len(caplog.records)
    bot.error = BadRequest("Message is not modified")
    await ch._edit_status(42, status, "same")
    assert len(caplog.records) == warning_count


@pytest.mark.asyncio
async def test_final_teardown_stays_prompt_under_the_new_cadence():
    bot = FakeBot()
    ch = _channel(bot)
    await ch.send(_progress("42", "turn start"))
    status = ch._status["42"]
    anim = status.anim
    await ch.send(_final("42", "answer"))
    assert "42" not in ch._status
    assert anim is not None and anim.done()
    assert bot.calls[-1][0] == "send_message"
    assert bot.count("delete_message") == 1


# ---------------------------------------------------------------------------
# Runtime coercion of compact_tool_rows (validator-bypass safe path)
# ---------------------------------------------------------------------------
# build_channel_manager_config uses model_copy(update=...), which skips
# TelegramConfig validators. The channel must coerce+clamp any malformed value
# to 1..10 (default 3) and never crash on startup.


def _bypassed_config(compact_tool_rows: object) -> TelegramConfig:
    """Mirror the gateway's validator-bypassing model_copy(update=...) path."""
    return TelegramConfig().model_copy(update={"compact_tool_rows": compact_tool_rows})


def test_compact_tool_row_limit_default_is_three():
    ch = _channel(FakeBot())
    assert ch._compact_tool_row_limit == 3


def test_compact_tool_row_limit_valid_value_is_preserved():
    ch = TelegramChannel(
        TelegramConfig(token="token", compact_tool_rows=5), MessageBus()
    )
    assert ch._compact_tool_row_limit == 5


def test_compact_tool_row_limit_bypassed_zero_clamps_to_default():
    # 0 is below the 1..10 range → fall back to the default 3, not 1.
    ch = TelegramChannel(_bypassed_config(0), MessageBus())
    assert ch._compact_tool_row_limit == 3


def test_compact_tool_row_limit_bypassed_hundred_is_clamped_to_ten():
    ch = TelegramChannel(_bypassed_config(100), MessageBus())
    assert ch._compact_tool_row_limit == 10


def test_compact_tool_row_limit_bypassed_string_is_default_not_crash():
    # A non-numeric string bypassed through model_copy would crash int();
    # startup must not fail and the limit falls back to 3.
    ch = TelegramChannel(_bypassed_config("not-a-number"), MessageBus())
    assert ch._compact_tool_row_limit == 3


def test_compact_tool_row_limit_bypassed_numeric_string_is_coerced():
    ch = TelegramChannel(_bypassed_config("7"), MessageBus())
    assert ch._compact_tool_row_limit == 7


def test_compact_tool_row_limit_bypassed_none_is_default():
    ch = TelegramChannel(_bypassed_config(None), MessageBus())
    assert ch._compact_tool_row_limit == 3


def test_compact_tool_row_limit_bypassed_true_is_default_not_one():
    # bool is not a usable row count → default 3 (not int(True)==1).
    ch = TelegramChannel(_bypassed_config(True), MessageBus())
    assert ch._compact_tool_row_limit == 3


def test_compact_tool_row_limit_bypassed_false_is_default():
    ch = TelegramChannel(_bypassed_config(False), MessageBus())
    assert ch._compact_tool_row_limit == 3


def test_compact_tool_row_limit_bypassed_list_is_default():
    ch = TelegramChannel(_bypassed_config([1, 2, 3]), MessageBus())
    assert ch._compact_tool_row_limit == 3


def test_compact_tool_row_limit_bypassed_out_of_range_string_is_clamped():
    ch = TelegramChannel(_bypassed_config("42"), MessageBus())
    assert ch._compact_tool_row_limit == 10

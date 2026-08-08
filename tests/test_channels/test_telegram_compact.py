"""Compact progress on Telegram: a turn's ``_collapse`` events fold into one
spinner-animated status message, edited in place and deleted when the final
answer is sent."""

from __future__ import annotations

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

    assert list(ch._status["424242"].lines) == ["📋 To-do\n✅ Step A"]

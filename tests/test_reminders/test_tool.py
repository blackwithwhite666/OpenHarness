"""Tool behavior tests: create / list / cancel."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from openharness.tools.base import ToolExecutionContext

from ohmo.reminders.store import ReminderStore
from ohmo.reminders.tool import (
    RemindCancelInput,
    RemindCancelTool,
    RemindCreateInput,
    RemindCreateTool,
    RemindListInput,
    RemindListTool,
)

MSK = timezone(timedelta(hours=3))


def _ctx(metadata: dict | None, tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(cwd=tmp_path, metadata=metadata or {})


def _reminder_ctx(
    *,
    chat_id: str = "100",
    sender_id: str = "42",
    chat_type: str = "private",
    is_group: bool = False,
) -> dict:
    return {
        "ohmo_reminder_ctx": {
            "channel": "telegram",
            "chat_id": chat_id,
            "session_key": f"telegram:{chat_id}",
            "sender_id": sender_id,
            "chat_type": chat_type,
            "is_group": is_group,
            "tz": "Europe/Moscow",
        }
    }


def _future_iso(hours: int = 2) -> str:
    return (datetime.now(MSK) + timedelta(hours=hours)).isoformat()


async def test_create_requires_context(tmp_path: Path) -> None:
    tool = RemindCreateTool(ReminderStore(), asyncio.Lock(), default_tz="Europe/Moscow", max_per_chat=50)
    result = await tool.execute(
        RemindCreateInput(summary="hi", dtstart=_future_iso()),
        _ctx({}, tmp_path),
    )
    assert result.is_error


async def test_create_persists_and_returns_fire_times(tmp_path: Path) -> None:
    store = ReminderStore()
    tool = RemindCreateTool(store, asyncio.Lock(), default_tz="Europe/Moscow", max_per_chat=50)
    result = await tool.execute(
        RemindCreateInput(summary="standup", dtstart=_future_iso(), rrule="FREQ=DAILY"),
        _ctx(_reminder_ctx(), tmp_path),
    )
    assert not result.is_error
    assert "Current time:" in result.output
    assert "Next fire times:" in result.output
    assert store.count_active_for_chat("telegram", "100") == 1


async def test_create_past_dtstart_oneshot_errors(tmp_path: Path) -> None:
    tool = RemindCreateTool(ReminderStore(), asyncio.Lock(), default_tz="Europe/Moscow", max_per_chat=50)
    past = (datetime.now(MSK) - timedelta(hours=1)).isoformat()
    result = await tool.execute(
        RemindCreateInput(summary="late", dtstart=past),
        _ctx(_reminder_ctx(), tmp_path),
    )
    assert result.is_error
    assert "past" in result.output.lower()


async def test_create_bad_rrule_errors(tmp_path: Path) -> None:
    tool = RemindCreateTool(ReminderStore(), asyncio.Lock(), default_tz="Europe/Moscow", max_per_chat=50)
    result = await tool.execute(
        RemindCreateInput(summary="x", dtstart=_future_iso(), rrule="NOT-AN-RRULE"),
        _ctx(_reminder_ctx(), tmp_path),
    )
    assert result.is_error


async def test_create_quota_enforced(tmp_path: Path) -> None:
    store = ReminderStore()
    lock = asyncio.Lock()
    tool = RemindCreateTool(store, lock, default_tz="Europe/Moscow", max_per_chat=2)
    for _ in range(2):
        ok = await tool.execute(
            RemindCreateInput(summary="x", dtstart=_future_iso()),
            _ctx(_reminder_ctx(), tmp_path),
        )
        assert not ok.is_error
    overflow = await tool.execute(
        RemindCreateInput(summary="x", dtstart=_future_iso()),
        _ctx(_reminder_ctx(), tmp_path),
    )
    assert overflow.is_error
    assert "limit" in overflow.output.lower()


async def test_list_scoped_to_chat(tmp_path: Path) -> None:
    store = ReminderStore()
    lock = asyncio.Lock()
    create = RemindCreateTool(store, lock, default_tz="Europe/Moscow", max_per_chat=50)
    await create.execute(
        RemindCreateInput(summary="chat-100", dtstart=_future_iso()),
        _ctx(_reminder_ctx(chat_id="100"), tmp_path),
    )
    await create.execute(
        RemindCreateInput(summary="chat-200", dtstart=_future_iso()),
        _ctx(_reminder_ctx(chat_id="200"), tmp_path),
    )
    list_tool = RemindListTool(store, lock, default_tz="Europe/Moscow")
    result = await list_tool.execute(RemindListInput(), _ctx(_reminder_ctx(chat_id="100"), tmp_path))
    assert "chat-100" in result.output
    assert "chat-200" not in result.output


async def test_cancel_creator_only_in_group(tmp_path: Path) -> None:
    # Telegram-realistic context: the channel emits ``is_group`` (bool) and NEVER
    # ``chat_type``. The ACL must key off ``is_group`` — keying off ``chat_type``
    # alone made this dead code on Telegram (any group member could cancel).
    store = ReminderStore()
    lock = asyncio.Lock()
    create = RemindCreateTool(store, lock, default_tz="Europe/Moscow", max_per_chat=50)
    res = await create.execute(
        RemindCreateInput(summary="owned", dtstart=_future_iso()),
        _ctx(_reminder_ctx(chat_id="500", sender_id="owner", is_group=True), tmp_path),
    )
    assert not res.is_error
    reminder_id = store.list_for_chat("telegram", "500")[0].id

    cancel = RemindCancelTool(store, lock)
    # A different group member cannot cancel.
    denied = await cancel.execute(
        RemindCancelInput(id=reminder_id),
        _ctx(_reminder_ctx(chat_id="500", sender_id="intruder", is_group=True), tmp_path),
    )
    assert denied.is_error
    assert store.get(reminder_id).status == "active"

    # The creator can.
    allowed = await cancel.execute(
        RemindCancelInput(id=reminder_id),
        _ctx(_reminder_ctx(chat_id="500", sender_id="owner", is_group=True), tmp_path),
    )
    assert not allowed.is_error
    assert store.get(reminder_id).status == "done"


async def test_cancel_creator_only_feishu_chat_type(tmp_path: Path) -> None:
    # Feishu emits ``chat_type='group'`` (no ``is_group``). The ACL must still
    # apply via the chat_type fallback.
    store = ReminderStore()
    lock = asyncio.Lock()
    create = RemindCreateTool(store, lock, default_tz="Europe/Moscow", max_per_chat=50)
    await create.execute(
        RemindCreateInput(summary="owned", dtstart=_future_iso()),
        _ctx(_reminder_ctx(chat_id="700", sender_id="owner", chat_type="group"), tmp_path),
    )
    reminder_id = store.list_for_chat("telegram", "700")[0].id

    cancel = RemindCancelTool(store, lock)
    denied = await cancel.execute(
        RemindCancelInput(id=reminder_id),
        _ctx(_reminder_ctx(chat_id="700", sender_id="intruder", chat_type="group"), tmp_path),
    )
    assert denied.is_error
    assert store.get(reminder_id).status == "active"


async def test_cancel_wrong_chat_errors(tmp_path: Path) -> None:
    store = ReminderStore()
    lock = asyncio.Lock()
    create = RemindCreateTool(store, lock, default_tz="Europe/Moscow", max_per_chat=50)
    await create.execute(
        RemindCreateInput(summary="x", dtstart=_future_iso()),
        _ctx(_reminder_ctx(chat_id="100"), tmp_path),
    )
    reminder_id = store.list_for_chat("telegram", "100")[0].id
    cancel = RemindCancelTool(store, lock)
    result = await cancel.execute(
        RemindCancelInput(id=reminder_id),
        _ctx(_reminder_ctx(chat_id="999"), tmp_path),
    )
    assert result.is_error

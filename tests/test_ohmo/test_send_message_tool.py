"""Tests for the ohmo Telegram send tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from openharness.channels.bus.events import OutboundMessage
from openharness.tools.base import ToolExecutionContext

from ohmo.contact_registry import ContactStore
from ohmo.gateway.send_message_tool import (
    SendTelegramMessageInput,
    SendTelegramMessageTool,
)


def _ctx(tmp_path: Path, **send_ctx) -> ToolExecutionContext:
    ctx = {"sender_id": "1|owner", "first_name": "Owner", "username": "owner"}
    ctx.update(send_ctx)
    return ToolExecutionContext(cwd=tmp_path, metadata={"ohmo_send_ctx": ctx})


@pytest.mark.asyncio
async def test_send_telegram_message_success(tmp_path: Path):
    published: list[OutboundMessage] = []

    async def send_outbound(message: OutboundMessage) -> None:
        published.append(message)

    store = ContactStore(tmp_path)
    store.record_inbound(
        channel="telegram",
        chat_id="123",
        user_id="456",
        username="alice",
        first_name="Alice",
    )
    tool = SendTelegramMessageTool(store, send_outbound)

    result = await tool.execute(
        SendTelegramMessageInput(recipient="@alice", text="Hello Alice"),
        _ctx(tmp_path),
    )

    assert not result.is_error
    assert result.metadata == {"recipient_chat_id": "123"}
    assert len(published) == 1
    message = published[0]
    assert message.channel == "telegram"
    assert message.chat_id == "123"
    assert "Hello Alice" in message.content
    assert message.metadata == {
        "_session_key": "telegram:123",
        "_origin": "send_telegram_message",
    }


@pytest.mark.asyncio
async def test_send_telegram_message_unknown_recipient_errors(tmp_path: Path):
    published: list[OutboundMessage] = []

    async def send_outbound(message: OutboundMessage) -> None:
        published.append(message)

    store = ContactStore(tmp_path)
    store.record_inbound(channel="telegram", chat_id="123", username="alice")
    tool = SendTelegramMessageTool(store, send_outbound)

    result = await tool.execute(
        SendTelegramMessageInput(recipient="nobody", text="Hello"),
        _ctx(tmp_path),
    )

    assert result.is_error
    assert "known Telegram contacts" in result.output
    assert published == []


@pytest.mark.asyncio
async def test_send_telegram_message_ambiguous_recipient_errors(tmp_path: Path):
    published: list[OutboundMessage] = []

    async def send_outbound(message: OutboundMessage) -> None:
        published.append(message)

    store = ContactStore(tmp_path)
    store.record_inbound(
        channel="telegram",
        chat_id="111",
        username="alice_one",
        first_name="Alice",
    )
    store.record_inbound(
        channel="telegram",
        chat_id="222",
        username="alice_two",
        first_name="Alice",
    )
    tool = SendTelegramMessageTool(store, send_outbound)

    result = await tool.execute(
        SendTelegramMessageInput(recipient="alice", text="Hello"),
        _ctx(tmp_path),
    )

    assert result.is_error
    assert "Ambiguous recipient 'alice'; matches:" in result.output
    assert published == []


@pytest.mark.asyncio
async def test_send_telegram_message_empty_text_errors(tmp_path: Path):
    published: list[OutboundMessage] = []

    async def send_outbound(message: OutboundMessage) -> None:
        published.append(message)

    store = ContactStore(tmp_path)
    store.record_inbound(channel="telegram", chat_id="123", username="alice")
    tool = SendTelegramMessageTool(store, send_outbound)

    result = await tool.execute(
        SendTelegramMessageInput(recipient="@alice", text="   "),
        _ctx(tmp_path),
    )

    assert result.is_error
    assert result.output == "Refusing to send an empty message."
    assert published == []


@pytest.mark.asyncio
async def test_send_telegram_message_requires_sender_ctx(tmp_path: Path):
    published: list[OutboundMessage] = []

    async def send_outbound(message: OutboundMessage) -> None:
        published.append(message)

    store = ContactStore(tmp_path)
    store.record_inbound(channel="telegram", chat_id="123", username="alice")
    tool = SendTelegramMessageTool(store, send_outbound)

    result = await tool.execute(
        SendTelegramMessageInput(recipient="@alice", text="Hello"),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.is_error
    assert "sender context" in result.output or "live chat" in result.output
    assert published == []


@pytest.mark.asyncio
async def test_send_telegram_message_refuses_non_human_sender(tmp_path: Path):
    published: list[OutboundMessage] = []

    async def send_outbound(message: OutboundMessage) -> None:
        published.append(message)

    store = ContactStore(tmp_path)
    store.record_inbound(channel="telegram", chat_id="123", username="alice")
    tool = SendTelegramMessageTool(store, send_outbound)

    result = await tool.execute(
        SendTelegramMessageInput(recipient="@alice", text="Hello"),
        ToolExecutionContext(
            cwd=tmp_path,
            metadata={
                "ohmo_send_ctx": {
                    "sender_id": "__scheduler__",
                    "username": "",
                    "first_name": "",
                    "display_name": "",
                }
            },
        ),
    )

    assert result.is_error
    assert "human" in result.output or "reminder" in result.output
    assert published == []


@pytest.mark.asyncio
async def test_send_telegram_message_signs_with_sender(tmp_path: Path):
    published: list[OutboundMessage] = []

    async def send_outbound(message: OutboundMessage) -> None:
        published.append(message)

    store = ContactStore(tmp_path)
    store.record_inbound(channel="telegram", chat_id="123", username="alice")
    tool = SendTelegramMessageTool(store, send_outbound)

    result = await tool.execute(
        SendTelegramMessageInput(recipient="@alice", text="Hello Alice"),
        _ctx(tmp_path, first_name="Dmitrii", username="blackwithwhite"),
    )

    assert not result.is_error
    assert len(published) == 1
    content = published[0].content
    assert "Hello Alice" in content
    assert "Dmitrii" in content
    assert "@blackwithwhite" in content
    assert "через бота" in content


@pytest.mark.asyncio
async def test_send_telegram_message_fuzzy_match_suggests_without_sending(tmp_path: Path):
    published: list[OutboundMessage] = []

    async def send_outbound(message: OutboundMessage) -> None:
        published.append(message)

    store = ContactStore(tmp_path)
    store.record_inbound(channel="telegram", chat_id="123", username="alexandra")
    tool = SendTelegramMessageTool(store, send_outbound)

    result = await tool.execute(
        SendTelegramMessageInput(recipient="alex", text="Hello"),
        _ctx(tmp_path),
    )

    assert result.is_error
    assert "Did you mean" in result.output
    assert published == []

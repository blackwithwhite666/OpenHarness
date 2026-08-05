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
async def test_send_telegram_message_signs_from_reminder_creator(tmp_path: Path):
    # A reminder turn surfaces the creator as send_ctx.sender_id in the Telegram
    # "<id>|<username>" form (no username/first_name keys, since the scheduler
    # only stored the creator's sender_id). The tool must still sign + send,
    # signing as @<username> — not refuse for lack of a human sender.
    published: list[OutboundMessage] = []

    async def send_outbound(message: OutboundMessage) -> None:
        published.append(message)

    store = ContactStore(tmp_path)
    store.record_inbound(channel="telegram", chat_id="123", username="alice")
    tool = SendTelegramMessageTool(store, send_outbound)

    result = await tool.execute(
        SendTelegramMessageInput(recipient="@alice", text="Hello Alice"),
        ToolExecutionContext(
            cwd=tmp_path,
            metadata={
                "ohmo_send_ctx": {
                    "sender_id": "42|valeria",
                    "username": "",
                    "first_name": "",
                    "display_name": "",
                }
            },
        ),
    )

    assert not result.is_error
    assert len(published) == 1
    content = published[0].content
    assert "Hello Alice" in content
    assert "@valeria" in content
    assert "через бота" in content


def _bound_reminder_ctx(tmp_path: Path, **overrides) -> ToolExecutionContext:
    """ohmo_send_ctx as the runtime builds it for a recipient-bound synthetic
    reminder turn: creator as signer + the fixed recipient + reminder id."""
    ctx = {
        "sender_id": "42|valeria",
        "username": "",
        "first_name": "",
        "display_name": "",
        "fixed_recipient_chat_id": "200",
        "fixed_recipient_principal": "200",
        "fixed_recipient_label": "Marina @marina",
        "reminder_id": "r1",
    }
    ctx.update(overrides)
    return ToolExecutionContext(cwd=tmp_path, metadata={"ohmo_send_ctx": ctx})


@pytest.mark.asyncio
async def test_scheduled_send_goes_to_fixed_recipient_signed_with_reminder_id(tmp_path: Path):
    published: list[OutboundMessage] = []

    async def send_outbound(message: OutboundMessage) -> None:
        published.append(message)

    store = ContactStore(tmp_path)
    store.record_inbound(
        channel="telegram",
        chat_id="200",
        user_id="200",
        username="marina",
        first_name="Marina",
    )
    tool = SendTelegramMessageTool(store, send_outbound)

    result = await tool.execute(
        SendTelegramMessageInput(recipient="@marina", text="Your morning digest"),
        _bound_reminder_ctx(tmp_path),
    )

    assert not result.is_error
    assert len(published) == 1
    message = published[0]
    assert message.chat_id == "200"
    # Signed as the reminder's creator.
    assert "@valeria" in message.content
    assert "через бота" in message.content
    # Carries the reminder id so a real Telegram delivery failure can pause it.
    assert message.metadata["_reminder_id"] == "r1"
    assert message.metadata["_origin"] == "send_telegram_message"


@pytest.mark.asyncio
async def test_scheduled_send_accepts_exact_fixed_recipient_label(tmp_path: Path):
    published: list[OutboundMessage] = []

    async def send_outbound(message: OutboundMessage) -> None:
        published.append(message)

    store = ContactStore(tmp_path)
    store.record_inbound(
        channel="telegram",
        chat_id="200",
        user_id="200",
        username="marina",
        first_name="Marina",
    )
    tool = SendTelegramMessageTool(store, send_outbound)

    result = await tool.execute(
        SendTelegramMessageInput(recipient="Marina @marina", text="Condition met"),
        _bound_reminder_ctx(tmp_path),
    )

    assert not result.is_error
    assert result.metadata == {"recipient_chat_id": "200"}
    assert [message.chat_id for message in published] == ["200"]


@pytest.mark.asyncio
async def test_scheduled_send_fixed_recipient_label_mismatch_does_not_bypass_resolution(
    tmp_path: Path,
):
    published: list[OutboundMessage] = []

    async def send_outbound(message: OutboundMessage) -> None:
        published.append(message)

    store = ContactStore(tmp_path)
    store.record_inbound(
        channel="telegram",
        chat_id="200",
        user_id="200",
        username="marina",
        first_name="Marina",
    )
    tool = SendTelegramMessageTool(store, send_outbound)

    result = await tool.execute(
        SendTelegramMessageInput(recipient="Marina (@marina)", text="Condition met"),
        _bound_reminder_ctx(tmp_path),
    )

    assert result.is_error
    assert "Unknown recipient" in result.output
    assert published == []


@pytest.mark.asyncio
async def test_scheduled_send_rejects_stale_fixed_contact(tmp_path: Path):
    published: list[OutboundMessage] = []

    async def send_outbound(message: OutboundMessage) -> None:
        published.append(message)

    store = ContactStore(tmp_path)
    store.record_inbound(
        channel="telegram",
        chat_id="200",
        user_id="200",
        username="marina",
        first_name="Maria",
    )
    tool = SendTelegramMessageTool(store, send_outbound)

    result = await tool.execute(
        SendTelegramMessageInput(recipient="Marina @marina", text="Condition met"),
        _bound_reminder_ctx(tmp_path),
    )

    assert result.is_error
    assert "label" in result.output
    assert published == []


@pytest.mark.asyncio
async def test_scheduled_send_rejects_changed_fixed_principal(tmp_path: Path):
    published: list[OutboundMessage] = []

    async def send_outbound(message: OutboundMessage) -> None:
        published.append(message)

    store = ContactStore(tmp_path)
    store.record_inbound(
        channel="telegram",
        chat_id="200",
        user_id="999",
        username="marina",
        first_name="Marina",
    )
    tool = SendTelegramMessageTool(store, send_outbound)

    result = await tool.execute(
        SendTelegramMessageInput(recipient="Marina @marina", text="Condition met"),
        _bound_reminder_ctx(tmp_path),
    )

    assert result.is_error
    assert "principal" in result.output
    assert published == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("recipient", "first_name", "expected_error"),
    [
        pytest.param("@marina", "Maria", "label", id="changed_label_alias"),
        pytest.param("999", "Marina", "principal", id="changed_principal_alias"),
    ],
)
async def test_scheduled_send_rejects_changed_fixed_identity_via_alias(
    tmp_path: Path,
    recipient: str,
    first_name: str,
    expected_error: str,
):
    published: list[OutboundMessage] = []

    async def send_outbound(message: OutboundMessage) -> None:
        published.append(message)

    store = ContactStore(tmp_path)
    store.record_inbound(
        channel="telegram",
        chat_id="200",
        user_id="999" if expected_error == "principal" else "200",
        username="marina",
        first_name=first_name,
    )
    tool = SendTelegramMessageTool(store, send_outbound)

    result = await tool.execute(
        SendTelegramMessageInput(recipient=recipient, text="Condition met"),
        _bound_reminder_ctx(tmp_path),
    )

    assert result.is_error
    assert expected_error in result.output
    assert published == []


@pytest.mark.asyncio
async def test_scheduled_send_to_another_known_contact_is_rejected(tmp_path: Path):
    published: list[OutboundMessage] = []

    async def send_outbound(message: OutboundMessage) -> None:
        published.append(message)

    store = ContactStore(tmp_path)
    store.record_inbound(
        channel="telegram",
        chat_id="200",
        user_id="200",
        username="marina",
        first_name="Marina",
    )
    store.record_inbound(
        channel="telegram",
        chat_id="123",
        user_id="456",
        username="alice",
        first_name="Alice",
    )
    tool = SendTelegramMessageTool(store, send_outbound)

    result = await tool.execute(
        SendTelegramMessageInput(recipient="@alice", text="hi Alice"),
        _bound_reminder_ctx(tmp_path),
    )

    assert result.is_error
    assert "bound" in result.output
    assert published == []


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

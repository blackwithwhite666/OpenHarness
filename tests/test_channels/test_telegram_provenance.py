"""Trusted Telegram receive/source chronology and forward provenance."""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.telegram import TelegramChannel
from openharness.config.schema import TelegramConfig


class _DownloadedFile:
    async def download_to_drive(self, path: str) -> None:
        Path(path).write_bytes(b"forwarded-photo")


class _Bot:
    async def get_file(self, file_id: str) -> _DownloadedFile:
        assert file_id == "photo-file-id"
        return _DownloadedFile()


def _channel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TelegramChannel:
    monkeypatch.setenv("OPENHARNESS_CHANNEL_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENHARNESS_CHANNEL_MEDIA_DIR", str(tmp_path / "media"))
    channel = TelegramChannel(
        TelegramConfig(token="token", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = SimpleNamespace(bot=_Bot())
    monkeypatch.setattr(channel, "_start_typing", lambda *_args, **_kwargs: None)
    return channel


def _user() -> SimpleNamespace:
    return SimpleNamespace(
        id=116870365,
        username="current_sender",
        first_name="Current sender",
        is_bot=False,
    )


def _message(**changes: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "message_id": 10,
        "chat_id": 116870365,
        "chat": SimpleNamespace(type="private"),
        "text": None,
        "caption": None,
        "photo": None,
        "voice": None,
        "audio": None,
        "document": None,
        "location": None,
        "venue": None,
        "media_group_id": None,
        "reply_to_message": None,
        "forward_origin": None,
        "date": datetime(2026, 7, 31, 22, 30, tzinfo=timezone(timedelta(hours=3))),
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _update(message: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        effective_user=_user(),
        effective_message=message,
        edited_message=None,
        message=message,
    )


@pytest.mark.asyncio
async def test_plain_telegram_message_publishes_receive_timestamp_and_non_forward_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = _channel(tmp_path, monkeypatch)
    message = _message(text="hello")

    await channel._on_message(_update(message), None)
    inbound = await channel.bus.consume_inbound()

    assert inbound.timestamp is message.date
    assert inbound.content == "hello"
    assert inbound.metadata["received_at"] == "2026-07-31T19:30:00+00:00"
    assert inbound.metadata["is_forwarded"] is False
    assert inbound.metadata["source_message_at"] is None


@pytest.mark.asyncio
async def test_forwarded_photo_preserves_media_and_only_safe_forward_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = _channel(tmp_path, monkeypatch)
    origin_date = datetime(2026, 7, 30, 9, 15, tzinfo=timezone(timedelta(hours=-4)))
    origin = SimpleNamespace(
        date=origin_date,
        sender_user=SimpleNamespace(id=987654321, username="original_author"),
        sender_user_name="Original Author",
        chat=SimpleNamespace(id=-10012345, title="Original Secret Chat"),
        message_id=777,
    )
    photo = SimpleNamespace(
        file_id="photo-file-id",
        file_unique_id="photo-unique-id",
        mime_type="image/jpeg",
    )
    message = _message(caption="lunch photo", photo=[photo], forward_origin=origin)

    await channel._on_message(_update(message), None)
    inbound = await channel.bus.consume_inbound()

    assert inbound.timestamp is message.date
    assert inbound.metadata["received_at"] == "2026-07-31T19:30:00+00:00"
    assert inbound.metadata["is_forwarded"] is True
    assert inbound.metadata["source_message_at"] == "2026-07-30T13:15:00+00:00"
    assert len(inbound.media) == 1
    assert Path(inbound.media[0]).read_bytes() == b"forwarded-photo"
    assert inbound.content == f"lunch photo\n[image: {inbound.media[0]}]"

    rendered_metadata = json.dumps(inbound.metadata)
    assert "987654321" not in rendered_metadata
    assert "original_author" not in rendered_metadata
    assert "Original Author" not in rendered_metadata
    assert "-10012345" not in rendered_metadata
    assert "Original Secret Chat" not in rendered_metadata
    assert "forward_origin" not in inbound.metadata
    assert "forward_from" not in inbound.metadata
    assert "forward_sender_name" not in inbound.metadata


@pytest.mark.asyncio
async def test_media_group_keeps_first_identity_and_earliest_receive_and_source_times(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = _channel(tmp_path, monkeypatch)
    first = _message(
        message_id=20,
        caption="first",
        media_group_id="album-1",
        date=datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc),
        forward_origin=SimpleNamespace(
            date=datetime(2026, 7, 30, 10, 0, tzinfo=timezone.utc)
        ),
    )
    second = _message(
        message_id=21,
        caption="second",
        media_group_id="album-1",
        date=datetime(2026, 7, 31, 11, 0, tzinfo=timezone.utc),
        forward_origin=SimpleNamespace(
            date=datetime(2026, 7, 30, 9, 0, tzinfo=timezone.utc)
        ),
    )

    await channel._on_message(_update(first), None)
    await channel._on_message(_update(second), None)

    key = "116870365:album-1"
    task = channel._media_group_tasks.pop(key)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await channel._flush_media_group(key)
    inbound = await channel.bus.consume_inbound()

    assert inbound.content == "first\nsecond"
    assert inbound.metadata["message_id"] == 20
    assert inbound.timestamp == second.date
    assert inbound.metadata["received_at"] == "2026-07-31T11:00:00+00:00"
    assert inbound.metadata["is_forwarded"] is True
    assert inbound.metadata["source_message_at"] == "2026-07-30T09:00:00+00:00"

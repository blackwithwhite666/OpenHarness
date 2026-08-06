from types import SimpleNamespace

from telegram import InputMediaDocument, InputMediaPhoto, ReplyParameters

import pytest

from openharness.channels.bus.events import OutboundMessage
from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.telegram import TelegramChannel
from openharness.config.schema import TelegramConfig


class FakeBot:
    def __init__(self, *, fail_media_group: bool = False):
        self.fail_media_group = fail_media_group
        self.calls: list[tuple[str, dict]] = []

    async def send_media_group(self, **kwargs):
        self.calls.append(("send_media_group", kwargs))
        if self.fail_media_group:
            raise RuntimeError("album failed")

    async def send_photo(self, **kwargs):
        self.calls.append(("send_photo", kwargs))

    async def send_video(self, **kwargs):
        self.calls.append(("send_video", kwargs))

    async def send_document(self, **kwargs):
        self.calls.append(("send_document", kwargs))

    async def send_audio(self, **kwargs):
        self.calls.append(("send_audio", kwargs))

    async def send_voice(self, **kwargs):
        self.calls.append(("send_voice", kwargs))

    async def send_message(self, **kwargs):
        self.calls.append(("send_message", kwargs))


def _channel(bot: FakeBot) -> TelegramChannel:
    channel = TelegramChannel(
        TelegramConfig(token="token", reply_to_message=True),
        MessageBus(),
    )
    channel._app = SimpleNamespace(bot=bot)
    return channel


def _paths(tmp_path, *names: str) -> list[str]:
    paths: list[str] = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"media")
        paths.append(str(path))
    return paths


async def _send(channel: TelegramChannel, media: list[str], *, message_id: int = 42) -> None:
    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            media=media,
            metadata={"message_id": message_id},
        )
    )


@pytest.mark.asyncio
async def test_send_groups_multiple_photos_into_one_album(tmp_path):
    bot = FakeBot()
    channel = _channel(bot)

    await _send(channel, _paths(tmp_path, "a.jpg", "b.jpg", "c.jpg"))

    assert [name for name, _ in bot.calls] == ["send_media_group"]
    media = bot.calls[0][1]["media"]
    assert len(media) == 3
    assert all(isinstance(item, InputMediaPhoto) for item in media)


@pytest.mark.asyncio
async def test_send_chunks_photo_albums_at_ten_and_replies_once(tmp_path):
    bot = FakeBot()
    channel = _channel(bot)

    await _send(channel, _paths(tmp_path, *(f"p{i}.jpg" for i in range(12))), message_id=99)

    groups = [kwargs for name, kwargs in bot.calls if name == "send_media_group"]
    assert [len(group["media"]) for group in groups] == [10, 2]
    assert isinstance(groups[0]["reply_parameters"], ReplyParameters)
    assert groups[0]["reply_parameters"].message_id == 99
    assert groups[1]["reply_parameters"] is None


@pytest.mark.asyncio
async def test_send_single_photo_uses_single_send(tmp_path):
    bot = FakeBot()
    channel = _channel(bot)

    await _send(channel, _paths(tmp_path, "one.jpg"))

    assert [name for name, _ in bot.calls] == ["send_photo"]


@pytest.mark.asyncio
async def test_long_photo_prompt_uses_generic_media_plus_text_path(tmp_path):
    bot = FakeBot()
    channel = _channel(bot)
    image = _paths(tmp_path, "one.jpg")

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="x" * 1025,
            media=image,
            buttons=["Да", "Нет"],
        )
    )

    assert [name for name, _ in bot.calls] == ["send_photo", "send_message"]
    assert "caption" not in bot.calls[0][1]
    assert len(bot.calls[1][1]["text"]) == 1025


@pytest.mark.asyncio
async def test_send_keeps_voice_individual_and_groups_photos(tmp_path):
    bot = FakeBot()
    channel = _channel(bot)

    await _send(channel, _paths(tmp_path, "clip.ogg", "a.jpg", "b.jpg"))

    assert [name for name, _ in bot.calls] == ["send_voice", "send_media_group"]
    assert len(bot.calls[1][1]["media"]) == 2
    assert all(isinstance(item, InputMediaPhoto) for item in bot.calls[1][1]["media"])


@pytest.mark.asyncio
async def test_send_groups_documents_separately_from_photos(tmp_path):
    bot = FakeBot()
    channel = _channel(bot)

    await _send(channel, _paths(tmp_path, "a.jpg", "b.jpg", "one.pdf", "two.pdf"))

    assert [name for name, _ in bot.calls] == ["send_media_group", "send_media_group"]
    assert all(isinstance(item, InputMediaPhoto) for item in bot.calls[0][1]["media"])
    assert all(isinstance(item, InputMediaDocument) for item in bot.calls[1][1]["media"])


@pytest.mark.asyncio
async def test_send_media_group_failure_falls_back_to_individual_sends(tmp_path):
    bot = FakeBot(fail_media_group=True)
    channel = _channel(bot)

    await _send(channel, _paths(tmp_path, "a.jpg", "b.jpg", "c.jpg"))

    assert [name for name, _ in bot.calls] == [
        "send_media_group",
        "send_photo",
        "send_photo",
        "send_photo",
    ]

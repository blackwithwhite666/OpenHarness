"""Durable visible Telegram voice transcripts (bead agents-playgroud-kvg).

Voice/audio messages are transcribed by an injectable async transcriber BEFORE
the inbound turn is published. The exact transcript travels in
``InboundMessage.content`` (retaining the media path) and is ALSO sent as a
durable ``📝 ...`` Telegram reply explicitly linked to the original voice
message through trusted channel-owned metadata, so compact/final cleanup can
never delete or edit it.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from openharness.channels.bus.events import OutboundMessage
from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.telegram import TelegramChannel
from openharness.config.schema import TelegramConfig
from openharness.voice import (
    SubprocessVoiceTranscriber,
    TranscriptionError,
)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_voice_transcription_config_defaults_are_disabled_and_fail_closed():
    config = TelegramConfig()
    assert config.voice_transcription_enabled is False
    assert config.voice_transcription_argv == []
    assert config.voice_transcription_timeout_seconds > 0


def test_voice_transcription_enabled_requires_non_empty_argv():
    with pytest.raises(ValidationError):
        TelegramConfig(voice_transcription_enabled=True, voice_transcription_argv=[])
    with pytest.raises(ValidationError):
        TelegramConfig(
            voice_transcription_enabled=True, voice_transcription_argv=["  ", ""]
        )


def test_voice_transcription_timeout_is_bounded():
    with pytest.raises(ValidationError):
        TelegramConfig(voice_transcription_timeout_seconds=0)
    with pytest.raises(ValidationError):
        TelegramConfig(voice_transcription_timeout_seconds=-1)
    with pytest.raises(ValidationError):
        TelegramConfig(voice_transcription_timeout_seconds=99999)


def test_voice_transcription_valid_config_round_trips_through_telegram_config():
    config = TelegramConfig(
        voice_transcription_enabled=True,
        voice_transcription_argv=["elevenlabs-cli", "asr", "--timestamps", "none", "--json"],
        voice_transcription_timeout_seconds=60.0,
    )
    assert config.voice_transcription_enabled is True
    assert config.voice_transcription_argv[-1] == "--json"
    assert config.voice_transcription_timeout_seconds == 60.0


def test_subprocess_transcriber_rejects_empty_argv():
    with pytest.raises(ValueError):
        SubprocessVoiceTranscriber([])
    with pytest.raises(ValueError):
        SubprocessVoiceTranscriber(["ok", ""])


# ---------------------------------------------------------------------------
# Subprocess transcriber behaviour (real child processes)
# ---------------------------------------------------------------------------


def _script_transcriber(script: str, *, timeout: float = 10.0) -> SubprocessVoiceTranscriber:
    return SubprocessVoiceTranscriber(
        [sys.executable, "-c", script], timeout_seconds=timeout
    )


@pytest.mark.asyncio
async def test_subprocess_transcriber_success_appends_path_as_separate_argv(tmp_path):
    target = tmp_path / "voice sample.ogg"
    target.write_bytes(b"ogg")
    # The child echoes back the LAST argv element it received as the transcript.
    transcriber = _script_transcriber(
        "import json, sys; print(json.dumps({'text': sys.argv[-1]}))"
    )

    text = await transcriber.transcribe(str(target))

    assert text == str(target)


@pytest.mark.asyncio
async def test_subprocess_transcriber_nonzero_exit_is_an_explicit_failure(tmp_path):
    target = tmp_path / "v.ogg"
    target.write_bytes(b"ogg")
    transcriber = _script_transcriber(
        "import sys; sys.stderr.write('boom'); sys.exit(3)"
    )

    with pytest.raises(TranscriptionError):
        await transcriber.transcribe(str(target))


@pytest.mark.asyncio
async def test_subprocess_transcriber_timeout_kills_the_child(tmp_path):
    target = tmp_path / "v.ogg"
    target.write_bytes(b"ogg")
    transcriber = _script_transcriber("import time; time.sleep(30)", timeout=0.2)

    started = time.monotonic()
    with pytest.raises(TranscriptionError):
        await transcriber.transcribe(str(target))
    # The child must be killed, not left running for its full 30s sleep.
    assert time.monotonic() - started < 10


@pytest.mark.asyncio
async def test_subprocess_transcriber_invalid_json_is_an_explicit_failure(tmp_path):
    target = tmp_path / "v.ogg"
    target.write_bytes(b"ogg")
    transcriber = _script_transcriber("print('not json at all')")

    with pytest.raises(TranscriptionError):
        await transcriber.transcribe(str(target))


@pytest.mark.asyncio
async def test_subprocess_transcriber_empty_text_is_an_explicit_failure(tmp_path):
    target = tmp_path / "v.ogg"
    target.write_bytes(b"ogg")
    transcriber = _script_transcriber("import json; print(json.dumps({'text': '   '}))")

    with pytest.raises(TranscriptionError):
        await transcriber.transcribe(str(target))


# ---------------------------------------------------------------------------
# Telegram voice flow
# ---------------------------------------------------------------------------


class _DownloadedFile:
    async def download_to_drive(self, path: str) -> None:
        Path(path).write_bytes(b"voice-bytes")


class _Bot:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.edited: list[dict] = []
        self.deleted: list[dict] = []
        self._next_id = 5000

    async def get_file(self, file_id: str) -> _DownloadedFile:
        return _DownloadedFile()

    async def send_message(self, **kwargs):
        self.sent.append(kwargs)
        self._next_id += 1
        return SimpleNamespace(message_id=self._next_id)

    async def edit_message_text(self, **kwargs):
        self.edited.append(kwargs)

    async def delete_message(self, **kwargs):
        self.deleted.append(kwargs)

    async def send_chat_action(self, **kwargs):
        return None


class _FakeTranscriber:
    def __init__(self, result: str | Exception) -> None:
        self.result = result
        self.calls: list[str] = []

    async def transcribe(self, path: str) -> str:
        self.calls.append(path)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transcriber=None,
    *,
    reply_to_message: bool = False,
) -> tuple[TelegramChannel, _Bot]:
    monkeypatch.setenv("OPENHARNESS_CHANNEL_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENHARNESS_CHANNEL_MEDIA_DIR", str(tmp_path / "media"))
    bot = _Bot()
    channel = TelegramChannel(
        TelegramConfig(token="token", allow_from=["*"], reply_to_message=reply_to_message),
        MessageBus(),
        transcriber=transcriber,
    )
    channel._app = SimpleNamespace(bot=bot)
    monkeypatch.setattr(channel, "_start_typing", lambda *_args, **_kwargs: None)
    return channel, bot


def _user() -> SimpleNamespace:
    return SimpleNamespace(id=42, username="user", first_name="User", is_bot=False)


def _voice_message(message_id: int, unique: str) -> SimpleNamespace:
    return SimpleNamespace(
        message_id=message_id,
        chat_id=42,
        chat=SimpleNamespace(type="private"),
        text=None,
        caption=None,
        photo=None,
        voice=SimpleNamespace(
            file_id=f"file-id-{unique}",
            file_unique_id=unique,
            mime_type="audio/ogg",
        ),
        audio=None,
        document=None,
        location=None,
        venue=None,
        media_group_id=None,
        reply_to_message=None,
        forward_origin=None,
        date=datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
    )


def _update(message: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        effective_user=_user(),
        effective_message=message,
        edited_message=None,
        message=message,
    )


@pytest.mark.asyncio
async def test_voice_transcribed_before_inbound_with_path_and_durable_reply(
    tmp_path, monkeypatch
):
    transcriber = _FakeTranscriber("привет, это точный транскрипт")
    channel, _bot = _channel(tmp_path, monkeypatch, transcriber)

    await channel._on_message(_update(_voice_message(101, "voice-a")), None)

    # The transcript reply is a durable outbound message linked to the voice.
    reply = await channel.bus.consume_outbound()
    assert reply.content == "📝 привет, это точный транскрипт"
    assert reply.metadata["_voice_transcript_message_id"] == 101
    assert "_progress" not in reply.metadata
    assert "_collapse" not in reply.metadata

    inbound = await channel.bus.consume_inbound()
    path = transcriber.calls[0]
    assert f"[voice: {path}]" in inbound.content
    assert "[transcription: привет, это точный транскрипт]" in inbound.content
    assert path in inbound.media


@pytest.mark.asyncio
async def test_voice_transcription_failure_keeps_path_and_sends_one_actionable_notice(
    tmp_path, monkeypatch
):
    transcriber = _FakeTranscriber(TranscriptionError("exit 3"))
    channel, _bot = _channel(tmp_path, monkeypatch, transcriber)

    await channel._on_message(_update(_voice_message(102, "voice-b")), None)

    notice = await channel.bus.consume_outbound()
    assert notice.metadata["_voice_transcript_message_id"] == 102
    assert "📝" not in notice.content
    assert notice.content.strip()  # exactly one durable, actionable indication

    inbound = await channel.bus.consume_inbound()
    assert f"[voice: {transcriber.calls[0]}]" in inbound.content
    assert "[transcription:" not in inbound.content


@pytest.mark.asyncio
async def test_voice_without_transcriber_is_fail_closed_and_silent(tmp_path, monkeypatch):
    channel, _bot = _channel(tmp_path, monkeypatch, transcriber=None)

    await channel._on_message(_update(_voice_message(103, "voice-c")), None)

    inbound = await channel.bus.consume_inbound()
    assert inbound.content.startswith("[voice: ")
    assert channel.bus.outbound_size == 0


@pytest.mark.asyncio
async def test_voice_burst_each_message_gets_own_reply_and_ordered_transcripts(
    tmp_path, monkeypatch
):
    transcripts = iter(["первый голос", "второй голос"])

    class _BurstTranscriber:
        async def transcribe(self, path: str) -> str:
            return next(transcripts)

    channel, _bot = _channel(tmp_path, monkeypatch, _BurstTranscriber())

    await channel._on_message(_update(_voice_message(201, "voice-1")), None)
    await channel._on_message(_update(_voice_message(202, "voice-2")), None)

    reply1 = await channel.bus.consume_outbound()
    reply2 = await channel.bus.consume_outbound()
    inbound1 = await channel.bus.consume_inbound()
    inbound2 = await channel.bus.consume_inbound()

    assert reply1.content == "📝 первый голос"
    assert reply1.metadata["_voice_transcript_message_id"] == 201
    assert reply2.content == "📝 второй голос"
    assert reply2.metadata["_voice_transcript_message_id"] == 202
    assert "[transcription: первый голос]" in inbound1.content
    assert "voice-1" in inbound1.content
    assert "[transcription: второй голос]" in inbound2.content
    assert "voice-2" in inbound2.content


# ---------------------------------------------------------------------------
# Outbound delivery of the transcript reply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcript_reply_is_linked_even_when_reply_to_message_is_disabled(
    tmp_path, monkeypatch
):
    channel, bot = _channel(tmp_path, monkeypatch, object(), reply_to_message=False)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="42",
            content="📝 текст",
            metadata={"_voice_transcript": True, "_voice_transcript_message_id": 777},
        )
    )

    assert bot.sent[-1]["reply_parameters"].message_id == 777


@pytest.mark.asyncio
async def test_long_transcript_uses_normal_safe_splitting(tmp_path, monkeypatch):
    channel, bot = _channel(tmp_path, monkeypatch, object())
    transcript = "📝 " + ("длинный фрагмент расшифровки " * 400)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="42",
            content=transcript,
            metadata={"_voice_transcript": True, "_voice_transcript_message_id": 7},
        )
    )

    assert len(bot.sent) > 1
    assert all(len(call["text"]) <= 4000 for call in bot.sent)
    # Only the first chunk is reply-linked to the voice message.
    assert bot.sent[0]["reply_parameters"].message_id == 7
    assert bot.sent[1]["reply_parameters"] is None


@pytest.mark.asyncio
async def test_transcript_reply_survives_compact_teardown_and_final_cleanup(
    tmp_path, monkeypatch
):
    channel, bot = _channel(tmp_path, monkeypatch, object())

    # A live compact status from an in-flight turn.
    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="42",
            content="working",
            metadata={"_progress": True, "_collapse": True},
        )
    )
    status_id = channel._status["42"].message_id

    receipt = await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="42",
            content="📝 транскрипт",
            metadata={"_voice_transcript": True, "_voice_transcript_message_id": 55},
        )
    )
    (transcript_id,) = receipt.native_message_ids
    # The transcript send must NOT tear down the in-flight turn's live status.
    assert "42" in channel._status

    # Final answer: the compact status is torn down; the transcript is not.
    await channel.send(OutboundMessage(channel="telegram", chat_id="42", content="ответ", metadata={}))

    assert "42" not in channel._status
    assert [call["message_id"] for call in bot.deleted] == [status_id]
    assert all(call.get("message_id") != transcript_id for call in bot.edited)

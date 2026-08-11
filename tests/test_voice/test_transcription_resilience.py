"""Focused subprocess, retry, and Telegram voice dedupe tests."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from types import SimpleNamespace

import pytest

from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.manager import ChannelManager
from openharness.config.schema import Config
from openharness.voice import (
    RetryingVoiceTranscriber,
    SubprocessVoiceTranscriber,
    TranscriptionError,
    VoiceTranscriptionDedupe,
    voice_dedupe_key,
    voice_dedupe_key_for_path,
)


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
@pytest.mark.parametrize(
    "status_code,retryable,error_class",
    [(429, True, "rate_limit"), (500, True, "upstream")],
)
async def test_subprocess_transcriber_trusted_elevenlabs_error_contract(
    tmp_path, status_code, retryable, error_class
):
    target = tmp_path / "v.ogg"
    target.write_bytes(b"ogg")
    payload = {
        "schema_version": 1,
        "provider": "elevenlabs",
        "ok": False,
        "error_class": error_class,
        "retryable": retryable,
        "status_code": status_code,
        "message": "safe provider message",
    }
    script = "import json,sys; sys.stderr.write(json.dumps(" + repr(payload) + ")); sys.exit(1)"
    with pytest.raises(TranscriptionError) as raised:
        await _script_transcriber(script).transcribe(str(target))
    assert raised.value.provider == "elevenlabs"
    assert raised.value.error_class == error_class
    assert raised.value.retryable is retryable
    assert raised.value.status_code == status_code


@pytest.mark.asyncio
async def test_subprocess_transcriber_malformed_error_contract_is_not_retryable(tmp_path):
    target = tmp_path / "v.ogg"
    target.write_bytes(b"ogg")
    secret = "api-secret-in-stderr"
    script = f"import sys; sys.stderr.write({secret!r}); sys.exit(1)"
    with pytest.raises(TranscriptionError) as raised:
        await _script_transcriber(script).transcribe(str(target))
    assert raised.value.error_class == "invalid_error_contract"
    assert raised.value.retryable is False
    assert secret not in str(raised.value)


@pytest.mark.asyncio
async def test_subprocess_transcriber_timeout_kills_the_child(tmp_path):
    target = tmp_path / "v.ogg"
    target.write_bytes(b"ogg")
    transcriber = _script_transcriber("import time; time.sleep(30)", timeout=0.2)

    started = time.monotonic()
    with pytest.raises(TranscriptionError) as raised:
        await transcriber.transcribe(str(target))
    assert raised.value.error_class == "timeout"
    assert raised.value.retryable is True
    # The child must be killed, not left running for its full 30s sleep.
    assert time.monotonic() - started < 10


@pytest.mark.asyncio
async def test_subprocess_transcriber_cancellation_kills_the_child(tmp_path):
    target = tmp_path / "v.ogg"
    target.write_bytes(b"ogg")
    transcriber = _script_transcriber("import time; time.sleep(30)", timeout=30)
    task = asyncio.create_task(transcriber.transcribe(str(target)))
    await asyncio.sleep(0.1)
    task.cancel()
    started = time.monotonic()
    with pytest.raises(asyncio.CancelledError):
        await task
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


@pytest.mark.asyncio
async def test_retryable_failure_then_success_runs_twice_and_logs_no_transcript(caplog):
    class _RetryThenSuccess:
        provider = "elevenlabs"

        def __init__(self):
            self.calls = 0

        async def transcribe(self, path: str) -> str:
            self.calls += 1
            if self.calls == 1:
                raise TranscriptionError(
                    "secret provider text",
                    provider="elevenlabs",
                    error_class="rate_limit",
                    retryable=True,
                    status_code=429,
                )
            return "exact transcript secret"

    provider = _RetryThenSuccess()
    async def no_sleep(_: float) -> None:
        return None

    with caplog.at_level(logging.INFO):
        result = await RetryingVoiceTranscriber(
            provider, sleep=no_sleep, jitter=lambda: 0.0
        ).transcribe("audio-path")
    assert result == "exact transcript secret"
    assert provider.calls == 2
    assert "exact transcript secret" not in caplog.text
    assert "secret provider text" not in caplog.text


@pytest.mark.asyncio
async def test_non_retryable_and_exhausted_retry_call_counts():
    class _Failing:
        def __init__(self, error):
            self.error = error
            self.calls = 0

        async def transcribe(self, path: str) -> str:
            self.calls += 1
            raise self.error

    auth = _Failing(TranscriptionError("auth", error_class="auth"))
    with pytest.raises(TranscriptionError):
        await RetryingVoiceTranscriber(auth, sleep=lambda _: asyncio.sleep(0)).transcribe("p")
    assert auth.calls == 1

    exhausted = _Failing(TranscriptionError("429", error_class="rate_limit", retryable=True))
    with pytest.raises(TranscriptionError):
        await RetryingVoiceTranscriber(
            exhausted, sleep=lambda _: asyncio.sleep(0), jitter=lambda: 0.0
        ).transcribe("p")
    assert exhausted.calls == 2


@pytest.mark.asyncio
async def test_retry_total_budget_stops_before_extra_attempt():
    now = [0.0]
    calls = 0

    async def operation(path: str) -> str:
        nonlocal calls
        calls += 1
        now[0] += 0.6
        raise TranscriptionError("busy", error_class="busy", retryable=True)

    async def sleep(delay: float) -> None:
        now[0] += delay

    with pytest.raises(TranscriptionError):
        await RetryingVoiceTranscriber(
            SimpleNamespace(transcribe=operation),
            total_budget_seconds=1.0,
            base_backoff_seconds=0.5,
            sleep=sleep,
            clock=lambda: now[0],
            jitter=lambda: 0.0,
        ).transcribe("p")
    assert calls == 1


@pytest.mark.asyncio
async def test_voice_dedupe_coalesces_persists_and_distinguishes_key(tmp_path):
    cache = VoiceTranscriptionDedupe(tmp_path)
    key = voice_dedupe_key("chat", 1, b"same")
    calls = 0
    release = asyncio.Event()

    async def operation() -> str:
        nonlocal calls
        calls += 1
        await release.wait()
        return "cached transcript"

    first_task = asyncio.create_task(cache.transcribe(key, operation))
    second_task = asyncio.create_task(cache.transcribe(key, operation))
    await asyncio.sleep(0)
    release.set()
    first, second = await asyncio.gather(first_task, second_task)
    assert first == second == "cached transcript"
    assert calls == 1

    fresh = VoiceTranscriptionDedupe(tmp_path)
    fresh_calls = 0

    async def should_not_run() -> str:
        nonlocal fresh_calls
        fresh_calls += 1
        return "wrong"

    assert await fresh.transcribe(key, should_not_run) == "cached transcript"
    assert fresh_calls == 0

    changed_calls: list[str] = []

    async def changed_hash_operation() -> str:
        changed_calls.append("hash")
        return "changed hash transcript"

    async def changed_message_operation() -> str:
        changed_calls.append("message")
        return "changed message transcript"

    changed_hash_key = voice_dedupe_key("chat", 1, b"changed")
    changed_message_key = voice_dedupe_key("chat", 2, b"same")
    assert await fresh.transcribe(changed_hash_key, changed_hash_operation) == (
        "changed hash transcript"
    )
    assert await fresh.transcribe(changed_message_key, changed_message_operation) == (
        "changed message transcript"
    )
    assert changed_calls == ["hash", "message"]


@pytest.mark.asyncio
async def test_voice_dedupe_path_key_matches_bytes_key(tmp_path):
    audio = b"voice bytes" * 100_000
    path = tmp_path / "voice.ogg"
    path.write_bytes(audio)

    assert await voice_dedupe_key_for_path("chat", 1, path) == voice_dedupe_key(
        "chat", 1, audio
    )


@pytest.mark.asyncio
async def test_voice_dedupe_cancels_operation_when_sole_waiter_is_cancelled():
    cache = VoiceTranscriptionDedupe(None)
    key = voice_dedupe_key("chat", 1, b"cancel")
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def operation() -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "unreachable"

    waiter = asyncio.create_task(cache.transcribe(key, operation))
    await started.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert cancelled.is_set()
    assert await cache.transcribe(key, lambda: asyncio.sleep(0, result="fresh")) == "fresh"


@pytest.mark.asyncio
async def test_voice_dedupe_cancellation_keeps_operation_for_another_waiter():
    cache = VoiceTranscriptionDedupe(None)
    key = voice_dedupe_key("chat", 1, b"shared")
    started = asyncio.Event()
    release = asyncio.Event()
    cancelled = False
    calls = 0

    async def operation() -> str:
        nonlocal calls, cancelled
        calls += 1
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled = True
            raise
        return "shared transcript"

    first = asyncio.create_task(cache.transcribe(key, operation))
    second = asyncio.create_task(cache.transcribe(key, operation))
    await started.wait()
    while cache._inflight[key].waiters < 2:
        await asyncio.sleep(0)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not cancelled
    assert not second.done()

    release.set()
    assert await second == "shared transcript"
    assert calls == 1


@pytest.mark.asyncio
async def test_voice_dedupe_persists_typed_failures_and_ignores_corrupt_cache(tmp_path):
    key = voice_dedupe_key("chat", 7, b"failure")
    error = TranscriptionError(
        "secret failure details",
        provider="elevenlabs",
        error_class="auth",
        retryable=False,
        status_code=401,
    )
    cache = VoiceTranscriptionDedupe(tmp_path)

    async def fail() -> str:
        raise error

    with pytest.raises(TranscriptionError) as first:
        await cache.transcribe(key, fail)
    assert first.value.status_code == 401

    fresh = VoiceTranscriptionDedupe(tmp_path)
    calls = 0

    async def should_not_run() -> str:
        nonlocal calls
        calls += 1
        return "wrong"

    with pytest.raises(TranscriptionError) as cached:
        await fresh.transcribe(key, should_not_run)
    assert cached.value.error_class == "auth"
    assert cached.value.status_code == 401
    assert calls == 0

    cache_path = tmp_path / "voice_transcriptions.json"
    cache_path.write_text("not json", encoding="utf-8")
    recovered = VoiceTranscriptionDedupe(tmp_path)
    assert await recovered.transcribe(key, should_not_run) == "wrong"
    assert calls == 1


def test_manager_wires_retrying_elevenlabs_transcriber(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENHARNESS_CHANNEL_STATE_DIR", str(tmp_path / "state"))
    config = Config(
        channels={
            "telegram": {
                "enabled": True,
                "token": "token",
                "voice_transcription_enabled": True,
                "voice_transcription_argv": ["elevenlabs-cli", "asr", "--json"],
            }
        }
    )
    manager = ChannelManager(config, MessageBus())
    transcriber = manager.channels["telegram"]._transcriber
    assert isinstance(transcriber, RetryingVoiceTranscriber)
    assert isinstance(transcriber._transcriber, SubprocessVoiceTranscriber)


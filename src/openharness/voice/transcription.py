"""Bounded ElevenLabs voice transcription and deterministic retry support.

The CLI writes ``{"text": "..."}`` on successful stdout. A non-zero exit is
accepted only when stderr is the validated ElevenLabs ``schema_version=1``
error object; prose stderr is never used to infer retryability.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, cast, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_ATTEMPTS = 2
MAX_ATTEMPTS = 3
DEFAULT_BASE_BACKOFF_SECONDS = 0.5
MAX_BASE_BACKOFF_SECONDS = 30.0
DEFAULT_TOTAL_BUDGET_SECONDS = 90.0
MAX_TOTAL_BUDGET_SECONDS = 900.0
_DEFAULT_MAX_OUTPUT_BYTES = 4 * 1024 * 1024
_KILL_GRACE_SECONDS = 5.0
_ERROR_CLASS_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class TranscriptionError(RuntimeError):
    """A safe, typed transcription failure.

    The optional fields deliberately have defaults so existing callers that
    construct ``TranscriptionError("message")`` remain compatible.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "unknown",
        error_class: str = "unknown",
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.error_class = error_class
        self.retryable = retryable
        self.status_code = status_code


@runtime_checkable
class VoiceTranscriber(Protocol):
    """Async transcription of one downloaded audio file into plain text."""

    async def transcribe(self, path: str) -> str:
        """Return the transcript or raise :class:`TranscriptionError`."""
        ...


def _error(
    message: str,
    *,
    error_class: str,
    retryable: bool = False,
    provider: str = "elevenlabs",
    status_code: int | None = None,
) -> TranscriptionError:
    return TranscriptionError(
        message,
        provider=provider,
        error_class=error_class,
        retryable=retryable,
        status_code=status_code,
    )


def _parse_elevenlabs_error(stderr: bytes) -> TranscriptionError:
    """Parse only the documented ElevenLabs JSON error contract.

    Anything outside the contract is intentionally treated as an opaque,
    non-retryable failure. In particular, neither prose nor exit codes are
    used to infer retryability.
    """
    try:
        payload = json.loads(stderr.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return _error(
            "transcription returned an invalid error contract",
            error_class="invalid_error_contract",
        )
    if not isinstance(payload, dict):
        return _error(
            "transcription returned an invalid error contract",
            error_class="invalid_error_contract",
        )
    error_class = payload.get("error_class")
    provider = payload.get("provider")
    retryable = payload.get("retryable")
    message = payload.get("message")
    status_code = payload.get("status_code")
    valid_status = status_code is None or (
        isinstance(status_code, int)
        and not isinstance(status_code, bool)
        and 100 <= status_code <= 599
    )
    if (
        payload.get("schema_version") != 1
        or provider != "elevenlabs"
        or payload.get("ok") is not False
        or not isinstance(error_class, str)
        or not _ERROR_CLASS_RE.fullmatch(error_class)
        or not isinstance(retryable, bool)
        or not isinstance(message, str)
        or not message.strip()
        or len(message) > 4096
        or not valid_status
    ):
        return _error(
            "transcription returned an invalid error contract",
            error_class="invalid_error_contract",
        )
    return _error(
        message,
        error_class=error_class,
        retryable=retryable,
        provider=provider,
        status_code=status_code,
    )


class SubprocessVoiceTranscriber:
    """Transcribe by running a configured ElevenLabs CLI argv."""

    provider = "elevenlabs"

    def __init__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        argv = [str(part) for part in argv]
        if not argv or any(not part.strip() for part in argv):
            raise ValueError("transcription argv must be a non-empty list of non-empty strings")
        if not 0 < float(timeout_seconds) <= MAX_TIMEOUT_SECONDS:
            raise ValueError(
                f"transcription timeout must be in (0, {MAX_TIMEOUT_SECONDS}] seconds"
            )
        if int(max_output_bytes) <= 0:
            raise ValueError("transcription max output must be positive")
        self._argv = argv
        self._timeout = float(timeout_seconds)
        self._max_output_bytes = int(max_output_bytes)

    async def transcribe(self, path: str) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._argv,
                path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except asyncio.CancelledError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise _error(
                "transcription process could not be started",
                error_class="spawn_failure",
            ) from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                self._communicate_bounded(proc),
                timeout=self._timeout,
            )
        except asyncio.CancelledError:
            await self._terminate_child(proc)
            raise
        except asyncio.TimeoutError as exc:
            await self._terminate_child(proc)
            raise _error(
                "transcription timed out",
                error_class="timeout",
                retryable=True,
            ) from exc
        except TranscriptionError:
            await self._terminate_child(proc)
            raise

        if proc.returncode != 0:
            raise _parse_elevenlabs_error(stderr)
        try:
            payload = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise _error(
                "transcription returned invalid JSON",
                error_class="invalid_success_json",
            ) from exc
        if not isinstance(payload, dict):
            raise _error(
                "transcription returned an invalid success contract",
                error_class="invalid_success_contract",
            )
        text = payload.get("text")
        if not isinstance(text, str):
            raise _error(
                "transcription returned an invalid success contract",
                error_class="invalid_success_contract",
            )
        if not text.strip():
            raise _error(
                "transcription returned no text",
                error_class="empty_transcript",
            )
        return text.strip()

    async def _communicate_bounded(
        self, proc: asyncio.subprocess.Process
    ) -> tuple[bytes, bytes]:
        if proc.stdout is None or proc.stderr is None:
            raise _error("transcription pipes are unavailable", error_class="spawn_failure")
        results = await asyncio.gather(
            self._read_bounded(proc.stdout, self._max_output_bytes, "stdout"),
            self._read_bounded(proc.stderr, max(self._max_output_bytes // 16, 4096), "stderr"),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        await proc.wait()
        return cast(bytes, results[0]), cast(bytes, results[1])

    @staticmethod
    async def _read_bounded(
        stream: asyncio.StreamReader, limit: int, what: str
    ) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise _error(
                    f"transcription {what} exceeded its output bound",
                    error_class="oversized_output",
                )
        return b"".join(chunks)

    @staticmethod
    async def _terminate_child(proc: asyncio.subprocess.Process) -> None:
        """Kill and reap the child so timeout/cancellation never leaks it."""
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
            await asyncio.wait_for(proc.wait(), timeout=_KILL_GRACE_SECONDS)


Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]
Jitter = Callable[[], float]


class RetryingVoiceTranscriber:
    """Retry one voice transcriber under a single wall-clock budget."""

    provider = "elevenlabs"

    def __init__(
        self,
        transcriber: VoiceTranscriber,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        base_backoff_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS,
        total_budget_seconds: float = DEFAULT_TOTAL_BUDGET_SECONDS,
        sleep: Sleep = asyncio.sleep,
        clock: Clock = time.monotonic,
        jitter: Jitter = random.random,
    ) -> None:
        if not 1 <= int(max_attempts) <= MAX_ATTEMPTS:
            raise ValueError(f"transcription max attempts must be in [1, {MAX_ATTEMPTS}]")
        if not 0 <= float(base_backoff_seconds) <= MAX_BASE_BACKOFF_SECONDS:
            raise ValueError(
                f"transcription base backoff must be in [0, {MAX_BASE_BACKOFF_SECONDS}] seconds"
            )
        if not 0 < float(total_budget_seconds) <= MAX_TOTAL_BUDGET_SECONDS:
            raise ValueError(
                f"transcription total budget must be in (0, {MAX_TOTAL_BUDGET_SECONDS}] seconds"
            )
        self._transcriber = transcriber
        self._max_attempts = int(max_attempts)
        self._base_backoff = float(base_backoff_seconds)
        self._total_budget = float(total_budget_seconds)
        self._sleep = sleep
        self._clock = clock
        self._jitter = jitter

    async def transcribe(self, path: str) -> str:
        started = self._clock()
        last_error: TranscriptionError | None = None
        for attempt in range(1, self._max_attempts + 1):
            remaining = self._total_budget - (self._clock() - started)
            if remaining <= 0:
                break
            attempt_started = self._clock()
            try:
                result = await asyncio.wait_for(
                    self._transcriber.transcribe(path), timeout=remaining
                )
            except asyncio.CancelledError:
                logger.info(
                    "voice_asr_attempt provider=%s attempt=%d latency=%.3f result=cancelled "
                    "error_class=cancelled retryable=false",
                    getattr(self._transcriber, "provider", "elevenlabs"),
                    attempt,
                    max(0.0, self._clock() - attempt_started),
                )
                raise
            except asyncio.TimeoutError:
                error = _error("transcription timed out", error_class="timeout", retryable=True)
                last_error = error
                self._log_attempt(attempt, attempt_started, error)
            except TranscriptionError as transcription_error:
                last_error = transcription_error
                self._log_attempt(attempt, attempt_started, transcription_error)
                if not transcription_error.retryable:
                    raise
            except Exception as exc:
                typed_error = _error(
                    "transcription provider failed",
                    error_class="provider_failure",
                )
                last_error = typed_error
                self._log_attempt(attempt, attempt_started, typed_error)
                raise typed_error from exc
            else:
                logger.info(
                    "voice_asr_attempt provider=%s attempt=%d latency=%.3f result=success "
                    "error_class=none retryable=false",
                    getattr(self._transcriber, "provider", "elevenlabs"),
                    attempt,
                    max(0.0, self._clock() - attempt_started),
                )
                return result

            if attempt >= self._max_attempts or last_error is None:
                break
            remaining = self._total_budget - (self._clock() - started)
            delay = min(self._base_backoff * (2 ** (attempt - 1)), 30.0)
            delay += min(max(0.0, self._jitter()), 1.0) * self._base_backoff
            delay = min(delay, max(0.0, remaining))
            if delay <= 0:
                break
            await self._sleep(delay)

        if last_error is not None:
            raise last_error
        raise _error(
            "transcription total budget exhausted",
            error_class="timeout",
            retryable=True,
        )

    def _log_attempt(
        self, attempt: int, started: float, error: TranscriptionError
    ) -> None:
        logger.info(
            "voice_asr_attempt provider=%s attempt=%d latency=%.3f result=error "
            "error_class=%s retryable=%s",
            error.provider,
            attempt,
            max(0.0, self._clock() - started),
            error.error_class,
            error.retryable,
        )

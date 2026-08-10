"""Injectable async voice transcription with a bounded subprocess backend.

The production wiring runs a local ASR CLI (e.g. ``elevenlabs-cli asr
--timestamps none --json``) as a child process: no shell, no string
interpolation — the audio path is appended as one separate argv element. The
child prints a JSON object with a ``text`` field on stdout. Every failure mode
(nonzero exit, timeout, invalid JSON, empty text, oversized output) is an
explicit :class:`TranscriptionError`; the child is always reaped on timeout or
cancellation.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_TIMEOUT_SECONDS = 600.0
_DEFAULT_MAX_OUTPUT_BYTES = 4 * 1024 * 1024
_KILL_GRACE_SECONDS = 5.0


class TranscriptionError(RuntimeError):
    """A transcription attempt failed in a well-defined, actionable way."""


@runtime_checkable
class VoiceTranscriber(Protocol):
    """Async transcription of one downloaded audio file into plain text."""

    async def transcribe(self, path: str) -> str:
        """Return the transcript for the audio file at ``path``.

        Implementations raise :class:`TranscriptionError` on failure; they
        never return an empty/whitespace transcript.
        """
        ...


class SubprocessVoiceTranscriber:
    """Transcribe by running a configured argv as a subprocess.

    ``argv`` is the full command template (program + flags); the audio file
    path is appended as the final, separate argv element. Output is bounded,
    the runtime is bounded by ``timeout_seconds``, and the child is killed and
    reaped on timeout or task cancellation.
    """

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
        self._argv = argv
        self._timeout = float(timeout_seconds)
        self._max_output_bytes = int(max_output_bytes)

    async def transcribe(self, path: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            *self._argv,
            path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
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
            raise TranscriptionError(
                f"transcription timed out after {self._timeout:g}s"
            ) from exc
        except TranscriptionError:
            await self._terminate_child(proc)
            raise

        if proc.returncode != 0:
            snippet = stderr.decode("utf-8", errors="replace").strip()[:200]
            raise TranscriptionError(
                f"transcription exited with code {proc.returncode}"
                + (f": {snippet}" if snippet else "")
            )
        try:
            payload = json.loads(stdout.decode("utf-8", errors="replace"))
        except ValueError as exc:
            raise TranscriptionError("transcription returned invalid JSON") from exc
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise TranscriptionError("transcription returned no text")
        return text.strip()

    async def _communicate_bounded(
        self, proc: asyncio.subprocess.Process
    ) -> tuple[bytes, bytes]:
        results = await asyncio.gather(
            self._read_bounded(proc.stdout, self._max_output_bytes, "stdout"),
            self._read_bounded(proc.stderr, max(self._max_output_bytes // 16, 4096), "stderr"),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        await proc.wait()
        return results[0], results[1]

    @staticmethod
    async def _read_bounded(stream, limit: int, what: str) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise TranscriptionError(
                    f"transcription {what} exceeded the {limit}-byte bound"
                )
        return b"".join(chunks)

    @staticmethod
    async def _terminate_child(proc: asyncio.subprocess.Process) -> None:
        """Kill and reap the child so timeout/cancellation never leaks it."""
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
            await asyncio.wait_for(proc.wait(), timeout=_KILL_GRACE_SECONDS)

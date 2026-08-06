"""Atomic, compare-and-replace persistence for consumer result sidecars."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None

from .models import NutritionResultSidecar, ResultState


class NutritionResultStore:
    """Persist one result sidecar without ever replacing it with partial JSON."""

    MAX_HISTORY = 32

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> NutritionResultSidecar | None:
        if not self.path.exists():
            return None
        return NutritionResultSidecar.model_validate_json(self.path.read_text(encoding="utf-8"))

    def compare_and_replace(
        self,
        value: NutritionResultSidecar,
        *,
        expected_revision: int | None = None,
        crash_after_flush: bool = False,
    ) -> NutritionResultSidecar:
        """Validate and atomically publish ``value`` if its predecessor matches.

        ``expected_revision`` is a compare-and-replace guard.  The temporary is
        created beside the destination and fsynced before ``os.replace``.  Any
        error before the replace leaves the previous valid sidecar untouched.
        """
        with self._file_lock():
            return self._compare_and_replace_unlocked(
                value,
                expected_revision=expected_revision,
                crash_after_flush=crash_after_flush,
            )

    def _compare_and_replace_unlocked(
        self,
        value: NutritionResultSidecar,
        *,
        expected_revision: int | None,
        crash_after_flush: bool,
    ) -> NutritionResultSidecar:
        value = NutritionResultSidecar.model_validate(value.model_dump(mode="json"))
        current = self.load()
        current_revision = current.revision if current else 0
        if expected_revision is not None and current_revision != expected_revision:
            raise RuntimeError(
                f"stale nutrition sidecar revision: expected {expected_revision}, "
                f"found {current_revision}"
            )
        if value.revision != current_revision + 1:
            raise ValueError(
                f"sidecar revision must advance from {current_revision} to "
                f"{current_revision + 1}, got {value.revision}"
            )
        if current is not None:
            if value.candidate_id != current.candidate_id:
                raise ValueError("candidate_id is immutable")
            if value.recipient != current.recipient:
                raise ValueError("recipient binding is immutable")
            self._validate_transition(current.state, value.state)
            if value.confirmation_operation_id != current.confirmation_operation_id:
                raise ValueError("confirmation operation id is immutable")
            if value.meal_observation_operation_id != current.meal_observation_operation_id:
                raise ValueError("meal operation id is immutable")
        elif value.state not in {ResultState.published, ResultState.discovered}:
            raise ValueError("a new sidecar must start at discovered or published")

        if len(value.state_history) > self.MAX_HISTORY:
            raise ValueError("state history exceeds bounded limit")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if crash_after_flush:
                raise OSError("simulated crash after sidecar flush")
            os.replace(temporary, self.path)
            self._fsync_directory()
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return value

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(f".{self.path.name}.lock")
        with lock_path.open("a+") as lock:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _validate_transition(previous: ResultState, current: ResultState) -> None:
        legal: dict[ResultState, set[ResultState]] = {
            ResultState.discovered: {ResultState.classified, ResultState.retryable_error},
            ResultState.classified: {ResultState.published, ResultState.completed, ResultState.retryable_error},
            ResultState.published: {ResultState.prompt_sending, ResultState.pending_confirmation, ResultState.retryable_error},
            ResultState.prompt_sending: {
                ResultState.pending_confirmation,
                ResultState.retryable_error,
                ResultState.delivery_unknown,
                ResultState.dead_letter,
            },
            # Returning to prompt_sending is an explicit operator replay of an
            # ambiguous delivery, never an automatic retry.
            ResultState.delivery_unknown: {
                ResultState.published,
                ResultState.prompt_sending,
                ResultState.dead_letter,
            },
            ResultState.pending_confirmation: {ResultState.confirmed, ResultState.declined, ResultState.retryable_error},
            ResultState.confirmed: {
                ResultState.estimated,
                ResultState.retryable_error,
                ResultState.dead_letter,
            },
            ResultState.estimated: {
                ResultState.completed,
                ResultState.retryable_error,
                ResultState.dead_letter,
            },
            ResultState.retryable_error: {
                ResultState.discovered,
                ResultState.classified,
                ResultState.published,
                ResultState.prompt_sending,
                ResultState.pending_confirmation,
                ResultState.confirmed,
                ResultState.estimated,
                ResultState.retryable_error,
                ResultState.dead_letter,
            },
            ResultState.declined: {ResultState.completed},
            ResultState.dead_letter: {
                ResultState.discovered,
                ResultState.published,
                ResultState.prompt_sending,
                ResultState.pending_confirmation,
                ResultState.confirmed,
            },
            ResultState.completed: set(),
        }
        if current not in legal.get(previous, set()):
            raise ValueError(f"illegal nutrition state transition: {previous} -> {current}")

    def _fsync_directory(self) -> None:
        directory_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

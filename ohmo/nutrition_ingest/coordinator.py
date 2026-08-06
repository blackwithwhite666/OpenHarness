"""Serialized, filesystem-backed Marina confirmation coordinator."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from openharness.channels.bus.events import (
    InboundMessage,
    OutboundDeliveryReceipt,
    OutboundMessage,
)

from .models import NutritionResultSidecar, RecipientBinding, ResultState, StageAttempt, StateHistoryEntry
from .metrics import NutritionMetrics
from .prompts import build_post_confirmation_prompt
from .sidecars import NutritionResultStore
from .trust import COORDINATOR_TRUST_TOKEN
from .watcher import NutritionArtifactScanner, ReadyNutritionArtifact


_PRINCIPAL_RE = re.compile(r"^[1-9][0-9]*$")
_YES = {"да"}
_NO = {"нет"}
logger = logging.getLogger(__name__)


class NutritionCoordinatorError(RuntimeError):
    """A fail-closed coordinator state or binding error."""


class NutritionIngestCoordinator:
    """Own the single global confirmation slot and durable state machine."""

    def __init__(
        self,
        config,
        *,
        scanner: NutritionArtifactScanner | None = None,
        publish_outbound: Callable[[OutboundMessage], Awaitable[None] | None] | None = None,
        runtime_pool: object | None = None,
        estimate: Callable[[InboundMessage], Awaitable[object] | object] | None = None,
        metrics: NutritionMetrics | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        if config.enabled:
            if not _PRINCIPAL_RE.fullmatch(config.principal):
                raise NutritionCoordinatorError("nutrition principal must be positive numeric")
            if str(config.chat_id) != str(config.principal):
                raise NutritionCoordinatorError("nutrition chat must equal the principal")
            if config.session_key != f"telegram:{config.principal}":
                raise NutritionCoordinatorError("nutrition session is not bound to the private chat")
            if config.synchronized_root is None:
                raise NutritionCoordinatorError("nutrition root is required")
        self._scanner = scanner or NutritionArtifactScanner(config.synchronized_root or Path("."))
        self._publish_outbound = publish_outbound
        self._runtime_pool = runtime_pool
        self._estimate = estimate
        self._metrics = metrics or NutritionMetrics()
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._lock = asyncio.Lock()
        self._running = False

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def root(self) -> Path:
        if self.config.synchronized_root is None:
            raise NutritionCoordinatorError("nutrition root is not configured")
        return Path(self.config.synchronized_root).expanduser().resolve()

    def _recipient(self) -> RecipientBinding:
        return RecipientBinding(
            channel="telegram",
            principal=self.config.principal,
            chat_id=str(self.config.chat_id),
            session_key=self.config.session_key,
            tenant_id="marina",
        )

    def _store(self, artifact: ReadyNutritionArtifact) -> NutritionResultStore:
        return NutritionResultStore(artifact.directory / "result.json")

    def _new_sidecar(self, artifact: ReadyNutritionArtifact) -> NutritionResultSidecar:
        at = self._now()
        return NutritionResultSidecar(
            candidate_id=artifact.candidate_id,
            revision=1,
            state=ResultState.published,
            state_history=[StateHistoryEntry(revision=1, state=ResultState.published, at=at)],
            recipient=self._recipient(),
            confirmation_operation_id=f"{artifact.candidate_id}:confirm:v1",
            meal_observation_operation_id=f"{artifact.candidate_id}:meal-observation:v1",
        )

    def _advance(
        self,
        store: NutritionResultStore,
        current: NutritionResultSidecar,
        state: ResultState,
        *,
        error: str | None = None,
        **updates: Any,
    ) -> NutritionResultSidecar:
        history = list(current.state_history)
        history.append(
            StateHistoryEntry(
                revision=current.revision + 1,
                state=state,
                at=self._now(),
                error=error,
            )
        )
        value = current.model_copy(
            update={
                "revision": current.revision + 1,
                "state": state,
                "state_history": history[-32:],
                **updates,
            }
        )
        return store.compare_and_replace(value, expected_revision=current.revision)

    async def poll_once(self) -> list[str]:
        """Discover candidates and publish at most one confirmation prompt."""
        if not self.enabled:
            return []
        async with self._lock:
            artifacts = self._scanner.scan_ready()
            for artifact in artifacts:
                store = self._store(artifact)
                current = store.load()
                if current is None:
                    current = store.compare_and_replace(self._new_sidecar(artifact))
                    self._metrics.verified_candidate()
                elif current.state == ResultState.prompt_sending:
                    # A restart cannot distinguish Telegram acceptance from a
                    # lost receipt.  Never resend an ambiguous prompt.
                    current = self._advance(store, current, ResultState.delivery_unknown)
                    self._metrics.delivery_unknown()
                elif current.state == ResultState.pending_confirmation:
                    self._metrics.duplicate_suppression("prompt")

            pending = self._pending_artifact(artifacts)
            if pending is None:
                return [artifact.candidate_id for artifact in artifacts]
            artifact, sidecar = pending
            if sidecar.state in {ResultState.confirmed, ResultState.estimated} or (
                sidecar.state == ResultState.retryable_error
                and sidecar.consumption_status == "consumed"
            ):
                if sidecar.state == ResultState.retryable_error and not self._retry_eligible(
                    sidecar, stage="estimation"
                ):
                    return [item.candidate_id for item in artifacts]
                try:
                    await self._estimate_candidate(artifact, sidecar)
                except Exception as exc:  # noqa: BLE001 - one candidate cannot kill polling
                    current = self._store(artifact).load() or sidecar
                    if current.state in {
                        ResultState.confirmed,
                        ResultState.estimated,
                        ResultState.retryable_error,
                    } and current.consumption_status == "consumed":
                        self._record_failure(self._store(artifact), current, stage="estimation", error=exc)
                return [item.candidate_id for item in artifacts]
            if sidecar.state not in {ResultState.published, ResultState.retryable_error}:
                return [item.candidate_id for item in artifacts]
            if sidecar.state == ResultState.retryable_error and not self._retry_eligible(
                sidecar, stage="prompt"
            ):
                # The head candidate remains the serialized queue barrier while
                # its bounded retry backoff is pending.  Never let a later
                # artifact overtake it.
                return [item.candidate_id for item in artifacts]
            await self._publish_prompt(artifact, sidecar)
            return [item.candidate_id for item in artifacts]

    def _retry_eligible(self, sidecar: NutritionResultSidecar, *, stage: str) -> bool:
        """Return whether the next retry may run according to its last attempt."""
        attempts = [attempt for attempt in sidecar.attempts if attempt.stage == stage]
        if not attempts:
            return True
        last_attempt = attempts[-1]
        finished_at = last_attempt.finished_at or last_attempt.started_at
        exponent = max(last_attempt.attempt - 1, 0)
        delay_seconds = min(
            float(self.config.retry_backoff_seconds) * (2**exponent),
            3600.0,
        )
        return self._now() >= finished_at + timedelta(seconds=delay_seconds)

    def _pending_artifact(
        self, artifacts: list[ReadyNutritionArtifact]
    ) -> tuple[ReadyNutritionArtifact, NutritionResultSidecar] | None:
        for artifact in artifacts:
            sidecar = self._store(artifact).load()
            if sidecar is None:
                continue
            if sidecar.state in {
                ResultState.prompt_sending,
                ResultState.delivery_unknown,
                ResultState.pending_confirmation,
                ResultState.confirmed,
                ResultState.estimated,
            }:
                return artifact, sidecar
        for artifact in artifacts:
            sidecar = self._store(artifact).load()
            if sidecar is not None and sidecar.state in {
                ResultState.published,
                ResultState.retryable_error,
            }:
                return artifact, sidecar
        return None

    async def _publish_prompt(
        self, artifact: ReadyNutritionArtifact, sidecar: NutritionResultSidecar
    ) -> None:
        store = self._store(artifact)
        operation_id = sidecar.confirmation_operation_id
        sending = self._advance(store, sidecar, ResultState.prompt_sending)
        message = OutboundMessage(
            channel="telegram",
            chat_id=str(self.config.chat_id),
            content="Вы это съели?",
            media=[str(artifact.image_path)],
            buttons=["Да", "Нет"],
            metadata={
                "_trusted_outbound_operation_id": operation_id,
                "_nutrition_confirmation": True,
                "_nutrition_candidate_id": artifact.candidate_id,
                "_nutrition_principal": self.config.principal,
                "_nutrition_chat_id": str(self.config.chat_id),
                "_nutrition_session_key": self.config.session_key,
                "_nutrition_phase": "confirmation",
            },
        )
        try:
            if self._publish_outbound is None:
                raise NutritionCoordinatorError("nutrition outbound publisher is unavailable")
            result = self._publish_outbound(message)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            self._record_failure(store, sending, stage="prompt", error=exc)

    async def on_send_success(
        self, message: OutboundMessage, receipt: OutboundDeliveryReceipt | None
    ) -> bool:
        if not self.enabled or not message.metadata.get("_nutrition_confirmation"):
            return False
        candidate_id = str(message.metadata.get("_nutrition_candidate_id") or "")
        artifact = self._find_artifact(candidate_id)
        if artifact is None:
            return False
        store = self._store(artifact)
        current = store.load()
        if current is None or current.state != ResultState.prompt_sending:
            return False
        if receipt is None or not self._valid_receipt(message, receipt):
            self._record_failure(
                store,
                current,
                stage="prompt",
                error=RuntimeError("nutrition confirmation delivery returned no valid receipt"),
                ambiguous=True,
            )
            return False
        self._advance(
            store,
            current,
            ResultState.pending_confirmation,
            prompt_message_id=receipt.native_message_ids[0],
        )
        return True

    async def on_send_failure(self, message: OutboundMessage, error: BaseException) -> bool:
        if not self.enabled or not message.metadata.get("_nutrition_confirmation"):
            return False
        artifact = self._find_artifact(str(message.metadata.get("_nutrition_candidate_id") or ""))
        if artifact is None:
            return False
        store = self._store(artifact)
        current = store.load()
        if current is None or current.state != ResultState.prompt_sending:
            return False
        self._record_failure(store, current, stage="prompt", error=error, ambiguous=True)
        return True

    def _record_failure(
        self,
        store: NutritionResultStore,
        current: NutritionResultSidecar,
        *,
        stage: str,
        error: BaseException,
        ambiguous: bool = False,
    ) -> NutritionResultSidecar:
        attempts = [item for item in current.attempts if item.stage == stage]
        attempt = len(attempts) + 1
        entry = StageAttempt(stage=stage, attempt=attempt, started_at=self._now(), finished_at=self._now(), error=str(error)[:1024])
        next_state = (
            ResultState.dead_letter
            if attempt >= (self.config.max_prompt_attempts if stage == "prompt" else self.config.max_estimation_attempts)
            else (ResultState.delivery_unknown if ambiguous else ResultState.retryable_error)
        )
        if next_state == ResultState.delivery_unknown:
            self._metrics.delivery_unknown()
        elif next_state == ResultState.dead_letter:
            self._metrics.dead_letter(stage, self._error_class(error))
        else:
            self._metrics.retry(stage, self._error_class(error))
        return self._advance(
            store,
            current,
            next_state,
            error=str(error)[:1024],
            attempts=[*current.attempts, entry][-32:],
        )

    def _valid_receipt(self, message: OutboundMessage, receipt: OutboundDeliveryReceipt) -> bool:
        native_ids = getattr(receipt, "native_message_ids", ())
        if not isinstance(native_ids, tuple):
            return False
        native_id = native_ids[0] if len(native_ids) == 1 else None
        return (
            getattr(receipt, "channel", None) == "telegram"
            and str(getattr(receipt, "chat_id", "")) == str(self.config.chat_id)
            and getattr(receipt, "outbound_operation_id", None)
            == message.metadata.get("_trusted_outbound_operation_id")
            and len(native_ids) == 1
            and isinstance(native_id, (int, str))
            and not isinstance(native_id, bool)
            and bool(str(native_id).strip())
            and (not isinstance(native_id, int) or native_id > 0)
        )

    def _find_artifact(self, candidate_id: str) -> ReadyNutritionArtifact | None:
        return next(
            (item for item in self._scanner.scan_ready() if item.candidate_id == candidate_id),
            None,
        )

    async def handle_inbound(self, message: InboundMessage) -> bool:
        """Intercept only bound Marina confirmation traffic."""
        if not self.enabled or message.channel != "telegram":
            return False
        principal = str(message.sender_id).split("|", 1)[0].strip()
        if principal != self.config.principal or str(message.chat_id) != str(self.config.chat_id):
            return False
        if message.session_key != self.config.session_key:
            return False
        pending = self._pending_confirmations()
        if len(pending) == 0:
            return False
        if len(pending) != 1:
            await self._clarify(message)
            return True
        artifact, sidecar = pending[0]
        native_id = message.metadata.get("native_message_id")
        callback = bool(message.metadata.get("callback_query"))
        if callback and str(native_id) != str(sidecar.prompt_message_id):
            await self._clarify(message)
            return True
        answer = message.content.strip().casefold()
        if answer not in _YES | _NO:
            await self._clarify(message)
            return True
        if callback and str(native_id) != str(sidecar.prompt_message_id):
            await self._clarify(message)
            return True
        if answer in _NO:
            self._metrics.confirmation("declined")
            self._record_pending_latency(sidecar)
            store = self._store(artifact)
            declined = self._advance(store, sidecar, ResultState.declined, consumption_status="not_consumed", reply_message_id=message.metadata.get("message_id"))
            completed = self._advance(store, declined, ResultState.completed)
            self._record_end_to_end_latency(completed)
            await self._ack(message, "Понял, не записываю этот снимок как съеденное.")
            return True
        store = self._store(artifact)
        self._metrics.confirmation("accepted")
        self._record_pending_latency(sidecar)
        confirmed = self._advance(store, sidecar, ResultState.confirmed, consumption_status="consumed", reply_message_id=message.metadata.get("message_id"))
        try:
            await self._estimate_candidate(artifact, confirmed)
        except Exception as exc:
            current = store.load() or confirmed
            if current.state == ResultState.confirmed:
                self._record_failure(store, current, stage="estimation", error=exc)
        return True

    def _pending_confirmations(self) -> list[tuple[ReadyNutritionArtifact, NutritionResultSidecar]]:
        pending = []
        for artifact in self._scanner.scan_ready():
            current = self._store(artifact).load()
            if current is not None and current.state == ResultState.pending_confirmation:
                pending.append((artifact, current))
        return pending

    async def _ack(self, message: InboundMessage, content: str) -> None:
        if self._publish_outbound is None:
            return
        result = self._publish_outbound(
            OutboundMessage(channel="telegram", chat_id=str(message.chat_id), content=content)
        )
        if asyncio.iscoroutine(result):
            await result

    async def _clarify(self, message: InboundMessage) -> None:
        await self._ack(message, "Пожалуйста, ответьте кнопкой «Да» или «Нет» для текущего фото.")

    async def _estimate_candidate(
        self, artifact: ReadyNutritionArtifact, sidecar: NutritionResultSidecar
    ) -> None:
        metadata = artifact.manifest.exif.model_dump(mode="json")
        # The manifest model has no GPS fields; construct a bounded copy rather
        # than passing arbitrary EXIF through to the model.
        prompt = build_post_confirmation_prompt(
            candidate_id=artifact.candidate_id,
            exif=metadata,
        )
        synthetic = InboundMessage(
            channel="telegram",
            sender_id="__nutrition_ingest__",
            chat_id=str(self.config.chat_id),
            content=prompt,
            media=[str(artifact.image_path)],
            session_key_override=self.config.session_key,
            metadata={
                "_synthetic": True,
                "_nutrition_trusted": True,
                "_nutrition_trust_token": COORDINATOR_TRUST_TOKEN,
                "_nutrition_candidate_id": artifact.candidate_id,
                "_nutrition_client_op_id": sidecar.meal_observation_operation_id,
                "_nutrition_phase": "estimation",
                "_nutrition_principal": self.config.principal,
                "_nutrition_tenant_id": "marina",
                "_nutrition_chat_id": str(self.config.chat_id),
                "_nutrition_session_key": self.config.session_key,
                "_nutrition_exif": metadata,
            },
        )
        if self._estimate is not None:
            result = self._estimate(synthetic)
            if asyncio.iscoroutine(result):
                result = await result
            assistant_id = None
            if isinstance(result, str) and result:
                assistant_id = result
            elif isinstance(result, dict):
                value = result.get("assistant_message_id") or result.get("honcho_message_id")
                if isinstance(value, str) and value:
                    assistant_id = value
            if not isinstance(assistant_id, str) or not assistant_id.strip():
                raise NutritionCoordinatorError("injected estimator returned no Honcho assistant id")
            self._complete_after_estimation(artifact, sidecar, assistant_id)
            return
        elif self._runtime_pool is not None:
            stream = self._runtime_pool.stream_message(synthetic, self.config.session_key)
            final_metadata: dict[str, object] = {}
            async for update in stream:
                if getattr(update, "kind", None) == "final":
                    final_metadata = dict(getattr(update, "metadata", {}) or {})
            assistant_id = final_metadata.get("_trusted_nutrition_assistant_message_id")
            if not isinstance(assistant_id, str) or not assistant_id:
                raise NutritionCoordinatorError("trusted estimation returned no Honcho assistant id")
            self._complete_after_estimation(artifact, sidecar, assistant_id)
            return
        raise NutritionCoordinatorError("nutrition estimator/runtime is unavailable")

    def _complete_after_estimation(
        self, artifact: ReadyNutritionArtifact, sidecar: NutritionResultSidecar, assistant_id: str | None
    ) -> None:
        if not isinstance(assistant_id, str) or not assistant_id.strip():
            raise NutritionCoordinatorError("estimation completion requires a Honcho assistant id")
        store = self._store(artifact)
        current = store.load() or sidecar
        if current.state in {ResultState.confirmed, ResultState.retryable_error}:
            current = self._advance(store, current, ResultState.estimated)
        elif current.state == ResultState.completed:
            self._metrics.duplicate_suppression("observation")
            return
        if current.state == ResultState.estimated:
            completed = self._advance(store, current, ResultState.completed, emitted_honcho_message_id=assistant_id)
            self._record_end_to_end_latency(completed)

    def _record_end_to_end_latency(self, sidecar: NutritionResultSidecar) -> None:
        published = next((item for item in sidecar.state_history if item.state == ResultState.published), None)
        terminal = sidecar.state_history[-1] if sidecar.state_history else None
        if published is not None and terminal is not None:
            self._metrics.end_to_end_latency(max(0.0, (terminal.at - published.at).total_seconds()))

    def _record_pending_latency(self, sidecar: NutritionResultSidecar) -> None:
        published = next((item for item in sidecar.state_history if item.state == ResultState.published), None)
        if published is not None:
            self._metrics.pending_latency(max(0.0, (self._now() - published.at).total_seconds()))

    @staticmethod
    def _error_class(error: BaseException) -> str:
        text = str(error).lower()
        if "hash" in text or "integrity" in text or "mismatch" in text:
            return "integrity"
        if "receipt" in text or "ack" in text or "delivery" in text:
            return "acknowledgement"
        if "timeout" in text or "transport" in text or "unavailable" in text:
            return "transport"
        return "runtime"

    def request_replay(self, candidate_id: str) -> bool:
        """Atomically enqueue an operator-approved replay for the service."""
        if not self.enabled:
            return False
        artifact = self._find_artifact(candidate_id)
        if artifact is None:
            return False
        store = self._store(artifact)
        current = store.load()
        if current is None or current.state not in {ResultState.delivery_unknown, ResultState.dead_letter}:
            return False
        if current.state == ResultState.dead_letter:
            if current.consumption_status == "consumed":
                target = ResultState.confirmed
            else:
                target = ResultState.published
        else:
            target = ResultState.published
        try:
            self._advance(store, current, target)
        except RuntimeError:
            return False
        return True

    async def replay(self, candidate_id: str) -> bool:
        """Compatibility wrapper; the CLI must not perform channel/runtime work."""
        return self.request_replay(candidate_id)

    async def run(self) -> None:
        self._running = True
        try:
            while self._running:
                try:
                    await self.poll_once()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - keep the coordinator alive after one bad candidate
                    logger.exception("nutrition ingest poll failed; will retry")
                await asyncio.sleep(self.config.poll_interval_seconds)
        except asyncio.CancelledError:
            raise

    def stop(self) -> None:
        self._running = False

    def status(self) -> dict[str, object]:
        items = []
        for artifact in self._scanner.scan_ready():
            sidecar = self._store(artifact).load()
            if sidecar is not None:
                items.append({"state": sidecar.state.value, "candidate": sidecar.candidate_id})
        return {"enabled": self.enabled, "pending": sum(item["state"] == "pending_confirmation" for item in items), "items": items}


__all__ = ["NutritionCoordinatorError", "NutritionIngestCoordinator"]

"""Channel manager for coordinating chat channels."""

from __future__ import annotations

import asyncio
import datetime
import logging
import math
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any


from openharness.channels.bus.events import OutboundDeliveryReceipt, OutboundMessage
from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.base import BaseChannel
from openharness.config.schema import Config

logger = logging.getLogger(__name__)

SendFailureHook = Callable[[OutboundMessage, BaseException], Awaitable[None] | None]
SendSuccessHook = Callable[
    [OutboundMessage, OutboundDeliveryReceipt | None], Awaitable[None] | None
]

# RetryAfter handling constants (bead agents-playgroud-axd).
_MAX_RETRY_AFTER_ATTEMPTS = 3  # bounded attempts for durable messages
_RETRY_AFTER_MARGIN = 0.5  # small safety margin on top of the server delay


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Duck-typed detection of a RetryAfter-style error.

    Works with ``telegram.error.RetryAfter`` (which exposes ``retry_after``)
    and any future channel error that follows the same protocol, without
    coupling the provider-neutral dispatcher to a specific SDK.

    Accepts a positive number of seconds or a positive
    ``datetime.timedelta`` (python-telegram-bot is migrating
    ``RetryAfter.retry_after`` from ``float`` to ``timedelta``). Zero,
    negative, non-finite or malformed values are rejected.
    """
    raw = getattr(exc, "retry_after", None)
    if isinstance(raw, bool):
        return None
    if isinstance(raw, datetime.timedelta):
        seconds = raw.total_seconds()
    elif isinstance(raw, (int, float)):
        seconds = float(raw)
    else:
        return None
    if not math.isfinite(seconds) or seconds <= 0:
        return None
    return seconds


class ChannelManager:
    """
    Manages chat channels and coordinates message routing.

    Responsibilities:
    - Initialize enabled channels (Telegram, WhatsApp, etc.)
    - Start/stop channels
    - Route outbound messages
    """

    def __init__(
        self,
        config: Config,
        bus: MessageBus,
        on_send_failure: SendFailureHook | None = None,
        on_send_success: SendSuccessHook | None = None,
    ):
        self.config = config
        self.bus = bus
        self.channels: dict[str, BaseChannel] = {}
        self._dispatch_task: asyncio.Task | None = None
        # Optional hook invoked when a channel.send() raises. The bus only
        # enqueues (publish_* never raises), so a real send failure — e.g. a
        # Telegram Forbidden/blocked — surfaces ONLY here. Lets a producer
        # (e.g. the reminder scheduler) react to a failed delivery it queued.
        self._on_send_failure = on_send_failure
        self._on_send_success = on_send_success
        # Per-chat delivery queues + worker tasks: FIFO within a chat,
        # concurrent across chats — a RetryAfter backoff sleep in one chat
        # never blocks another chat's deliveries. Workers are created lazily
        # on first enqueue and cancelled deterministically when the dispatcher
        # stops. LIMITATION: there is no restart-durable outbox — messages
        # still queued when the process stops are dropped.
        self._chat_queues: dict[str, asyncio.Queue[tuple[BaseChannel, OutboundMessage]]] = {}
        self._delivery_workers: dict[str, asyncio.Task[None]] = {}

        self._init_channels()

    @staticmethod
    def _is_durable(msg: OutboundMessage) -> bool:
        """Durable = user-visible final/command/error/interruption delivery.

        Progress/status traffic (``_progress`` metadata) is NOT durable: it may
        be coalesced or dropped on RetryAfter and must never starve a final.
        """
        return not msg.metadata.get("_progress", False)

    def _init_channels(self) -> None:
        """Initialize channels based on config."""

        # Telegram channel
        if self.config.channels.telegram.enabled:
            try:
                from openharness.channels.impl.telegram import TelegramChannel
                telegram_config = self.config.channels.telegram
                transcriber = None
                if telegram_config.voice_transcription_enabled:
                    from openharness.voice.transcription import SubprocessVoiceTranscriber
                    try:
                        transcriber = SubprocessVoiceTranscriber(
                            telegram_config.voice_transcription_argv,
                            timeout_seconds=telegram_config.voice_transcription_timeout_seconds,
                        )
                    except ValueError as voice_error:
                        # Fail closed: an unusable ASR command disables
                        # transcription rather than breaking the channel.
                        logger.warning(
                            "Telegram voice transcription disabled: %s", voice_error
                        )
                self.channels["telegram"] = TelegramChannel(
                    telegram_config,
                    self.bus,
                    transcriber=transcriber,
                )
                logger.info("Telegram channel enabled")
            except ImportError as e:
                logger.warning("Telegram channel not available: %s", e)

        # WhatsApp channel
        if self.config.channels.whatsapp.enabled:
            try:
                from openharness.channels.impl.whatsapp import WhatsAppChannel
                self.channels["whatsapp"] = WhatsAppChannel(
                    self.config.channels.whatsapp, self.bus
                )
                logger.info("WhatsApp channel enabled")
            except ImportError as e:
                logger.warning("WhatsApp channel not available: %s", e)

        # Discord channel
        if self.config.channels.discord.enabled:
            try:
                from openharness.channels.impl.discord import DiscordChannel
                self.channels["discord"] = DiscordChannel(
                    self.config.channels.discord, self.bus
                )
                logger.info("Discord channel enabled")
            except ImportError as e:
                logger.warning("Discord channel not available: %s", e)

        # Feishu channel
        if self.config.channels.feishu.enabled:
            try:
                from openharness.channels.impl.feishu import FeishuChannel
                self.channels["feishu"] = FeishuChannel(
                    self.config.channels.feishu, self.bus
                )
                logger.info("Feishu channel enabled")
            except ImportError as e:
                logger.warning("Feishu channel not available: %s", e)

        # Mochat channel
        if self.config.channels.mochat.enabled:
            try:
                from openharness.channels.impl.mochat import MochatChannel

                self.channels["mochat"] = MochatChannel(
                    self.config.channels.mochat, self.bus
                )
                logger.info("Mochat channel enabled")
            except ImportError as e:
                logger.warning("Mochat channel not available: %s", e)

        # DingTalk channel
        if self.config.channels.dingtalk.enabled:
            try:
                from openharness.channels.impl.dingtalk import DingTalkChannel
                self.channels["dingtalk"] = DingTalkChannel(
                    self.config.channels.dingtalk, self.bus
                )
                logger.info("DingTalk channel enabled")
            except ImportError as e:
                logger.warning("DingTalk channel not available: %s", e)

        # Email channel
        if self.config.channels.email.enabled:
            try:
                from openharness.channels.impl.email import EmailChannel
                self.channels["email"] = EmailChannel(
                    self.config.channels.email, self.bus
                )
                logger.info("Email channel enabled")
            except ImportError as e:
                logger.warning("Email channel not available: %s", e)

        # Slack channel
        if self.config.channels.slack.enabled:
            try:
                from openharness.channels.impl.slack import SlackChannel
                self.channels["slack"] = SlackChannel(
                    self.config.channels.slack, self.bus
                )
                logger.info("Slack channel enabled")
            except ImportError as e:
                logger.warning("Slack channel not available: %s", e)

        # QQ channel
        if self.config.channels.qq.enabled:
            try:
                from openharness.channels.impl.qq import QQChannel
                self.channels["qq"] = QQChannel(
                    self.config.channels.qq,
                    self.bus,
                )
                logger.info("QQ channel enabled")
            except ImportError as e:
                logger.warning("QQ channel not available: %s", e)

        # Matrix channel
        if self.config.channels.matrix.enabled:
            try:
                from openharness.channels.impl.matrix import MatrixChannel
                self.channels["matrix"] = MatrixChannel(
                    self.config.channels.matrix,
                    self.bus,
                )
                logger.info("Matrix channel enabled")
            except ImportError as e:
                logger.warning("Matrix channel not available: %s", e)

        self._validate_allow_from()

    def _validate_allow_from(self) -> None:
        for name, ch in self.channels.items():
            if getattr(ch.config, "allow_from", None) == []:
                logger.warning(
                    '%s channel has empty allow_from; remote access is denied until an operator explicitly adds allowed identities or chooses ["*"].',
                    name,
                )

    async def _start_channel(self, name: str, channel: BaseChannel) -> None:
        """Start a channel and log any exceptions."""
        try:
            await channel.start()
        except Exception as e:
            setattr(channel, "last_error", str(e))
            logger.exception("Failed to start channel %s", name)

    async def start_all(self) -> None:
        """Start all channels and the outbound dispatcher."""
        if not self.channels:
            logger.warning("No channels enabled")
            return

        # Start outbound dispatcher
        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())

        # Start channels
        tasks = []
        for name, channel in self.channels.items():
            logger.info("Starting %s channel...", name)
            tasks.append(asyncio.create_task(self._start_channel(name, channel)))

        # Wait for all to complete (they should run forever)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self) -> None:
        """Stop all channels and the dispatcher."""
        logger.info("Stopping all channels...")

        # Stop dispatcher (its shutdown also cancels the delivery workers).
        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
        await self._cancel_delivery_workers()

        # Stop all channels
        for name, channel in self.channels.items():
            try:
                await channel.stop()
                logger.info("Stopped %s channel", name)
            except Exception as e:
                logger.error("Error stopping %s: %s", name, e)

    async def _dispatch_outbound(self) -> None:
        """Route outbound messages to per-chat delivery workers."""
        logger.info("Outbound dispatcher started")

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(
                        self.bus.consume_outbound(),
                        timeout=1.0
                    )

                    progress_event = msg.metadata.get("progress_event")
                    telegram_todo = (
                        msg.channel == "telegram"
                        and isinstance(progress_event, dict)
                        and progress_event.get("kind") == "todo"
                    )
                    if msg.metadata.get("_progress") and not msg.metadata.get("_collapse") and not telegram_todo:
                        if msg.metadata.get("_tool_hint") and not self.config.channels.send_tool_hints:
                            continue
                        if not msg.metadata.get("_tool_hint") and not self.config.channels.send_progress:
                            continue

                    channel = self.channels.get(msg.channel)
                    if channel:
                        self._enqueue_delivery(channel, msg)
                    else:
                        logger.warning("Unknown channel: %s", msg.channel)

                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
        finally:
            await self._cancel_delivery_workers()

    def _enqueue_delivery(self, channel: BaseChannel, msg: OutboundMessage) -> None:
        """Hand one message to its (channel, chat) queue, spawning the chat's
        delivery worker on first use."""
        chat_key = f"{msg.channel}:{msg.chat_id}"
        queues = getattr(self, "_chat_queues", None)
        if queues is None:
            queues = self._chat_queues = {}
        workers = getattr(self, "_delivery_workers", None)
        if workers is None:
            workers = self._delivery_workers = {}

        queue = queues.setdefault(chat_key, asyncio.Queue())
        worker = workers.get(chat_key)
        if worker is None or worker.done():
            worker = asyncio.create_task(self._delivery_worker(chat_key, queue))
            worker.add_done_callback(partial(self._on_worker_done, chat_key))
            workers[chat_key] = worker
        queue.put_nowait((channel, msg))
        durable = self._is_durable(msg)
        log = logger.info if durable else logger.debug
        log(
            "outbound queued channel=%s chat=%s durable=%s",
            msg.channel,
            msg.chat_id,
            durable,
        )

    async def _delivery_worker(
        self,
        chat_key: str,
        queue: asyncio.Queue[tuple[BaseChannel, OutboundMessage]],
    ) -> None:
        """Drain one chat's queue in FIFO order. A RetryAfter backoff sleep
        here blocks only this chat; other chats' workers deliver
        independently."""
        while True:
            channel, msg = await queue.get()
            await self._send_with_retry(channel, msg, durable=self._is_durable(msg))

    def _on_worker_done(self, chat_key: str, task: asyncio.Task[None]) -> None:
        """Surface unexpected worker crashes instead of leaking them."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "delivery worker crashed chat=%s error=%s",
                chat_key,
                type(exc).__name__,
            )

    async def _cancel_delivery_workers(self) -> None:
        """Cancel all per-chat delivery workers and await them, without
        leaking exceptions. Queued-but-unsent messages are dropped (no
        restart-durable outbox in this pass)."""
        workers = list(getattr(self, "_delivery_workers", {}).values())
        for worker in workers:
            worker.cancel()
        if workers:
            results = await asyncio.gather(*workers, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    logger.error(
                        "delivery worker shutdown error: %s", type(result).__name__
                    )
        if hasattr(self, "_delivery_workers"):
            self._delivery_workers.clear()
        if hasattr(self, "_chat_queues"):
            self._chat_queues.clear()

    async def _send_with_retry(
        self,
        channel: BaseChannel,
        msg: OutboundMessage,
        *,
        durable: bool,
    ) -> None:
        """Attempt to send, retrying bounded times on RetryAfter.

        RetryAfter is safe/explicit: the server did NOT accept the message, so
        retrying cannot produce duplicates. Unknown exceptions are ambiguous
        (the remote send may have succeeded) and are NOT retried — only the
        failure hook is invoked.
        """
        max_attempts = _MAX_RETRY_AFTER_ATTEMPTS if durable else 1
        for attempt in range(1, max_attempts + 1):
            try:
                receipt = await channel.send(msg)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                retry_after = _retry_after_seconds(e)
                if retry_after is not None and attempt < max_attempts:
                    delay = retry_after + _RETRY_AFTER_MARGIN
                    logger.warning(
                        "outbound retrying channel=%s chat=%s durable=%s "
                        "attempt=%d/%d delay=%.1fs",
                        msg.channel,
                        msg.chat_id,
                        durable,
                        attempt,
                        max_attempts,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                if retry_after is not None and not durable:
                    logger.info(
                        "outbound dropped channel=%s chat=%s reason=retry_after",
                        msg.channel,
                        msg.chat_id,
                    )
                else:
                    logger.error(
                        "outbound failed channel=%s chat=%s durable=%s error=%s",
                        msg.channel,
                        msg.chat_id,
                        durable,
                        type(e).__name__,
                    )
                await self._notify_send_failure(msg, e)
                return

            # Nutrition confirmation integrity check (unchanged behavior).
            nutrition_receipt_invalid = msg.metadata.get("_nutrition_confirmation") and (
                receipt is None
                or receipt.channel != msg.channel
                or str(receipt.chat_id) != str(msg.chat_id)
                or receipt.outbound_operation_id
                != msg.metadata.get("_trusted_outbound_operation_id")
                or len(receipt.native_message_ids) != 1
            )
            if nutrition_receipt_invalid:
                await self._notify_send_failure(
                    msg,
                    RuntimeError("nutrition confirmation delivery returned no receipt"),
                )
            else:
                log = logger.info if durable else logger.debug
                log(
                    "outbound delivered channel=%s chat=%s durable=%s attempts=%d",
                    msg.channel,
                    msg.chat_id,
                    durable,
                    attempt,
                )
                await self._notify_send_success(msg, receipt)
            return

    async def _notify_send_failure(self, msg: OutboundMessage, error: BaseException) -> None:
        """Invoke the optional send-failure hook, swallowing hook errors so a
        misbehaving hook can never break the outbound dispatcher loop."""
        if self._on_send_failure is None:
            return
        try:
            result = self._on_send_failure(msg, error)
            if asyncio.iscoroutine(result):
                await result
        except Exception as hook_error:  # noqa: BLE001 — never break dispatch
            logger.error("Send-failure hook raised: %s", hook_error)

    async def _notify_send_success(
        self, msg: OutboundMessage, receipt: OutboundDeliveryReceipt | None
    ) -> None:
        """Invoke the success hook in isolation from dispatch and send errors."""
        hook = getattr(self, "_on_send_success", None)
        if hook is None:
            return
        try:
            result = hook(msg, receipt)
            if asyncio.iscoroutine(result):
                await result
        except Exception as hook_error:  # noqa: BLE001 — never break dispatch
            logger.error("Send-success hook raised: %s", hook_error)

    def get_channel(self, name: str) -> BaseChannel | None:
        """Get a channel by name."""
        return self.channels.get(name)

    def get_status(self) -> dict[str, Any]:
        """Get status of all channels."""
        return {
            name: {
                "enabled": True,
                "running": channel.is_running
            }
            for name, channel in self.channels.items()
        }

    @property
    def enabled_channels(self) -> list[str]:
        """Get list of enabled channel names."""
        return list(self.channels.keys())

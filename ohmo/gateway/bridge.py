"""Gateway bridge connecting channel bus traffic to ohmo runtimes."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import time
from datetime import datetime
from collections.abc import Awaitable, Callable
from pathlib import Path

from ohmo.contact_registry import ContactStore
from ohmo.evals import GatewayEvalRecorder
from ohmo.gateway.camera import (
    CAMERA_AUTHORITY,
    COALESCED_ATTACHMENT_PROVENANCE_AUTHORITY,
    CameraIngress,
    _classify_answer,
)
from ohmo.gateway.config import load_gateway_config, save_gateway_config
from ohmo.gateway.router import session_key_for_message
from ohmo.gateway.runtime import OhmoSessionRuntimePool, _evals_capture_enabled
from ohmo.group_registry import load_managed_group_record
from ohmo.workspace import get_gateway_interrupted_requests_path
from openharness.channels.bus.events import InboundMessage, OutboundMessage
from openharness.channels.bus.queue import MessageBus

logger = logging.getLogger(__name__)


def _content_snippet(text: str, *, limit: int = 160) -> str:
    """Return a single-line preview suitable for logs."""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


_ATTACH_RE = re.compile(r"\[\[\s*attach\s*:\s*([^\]]+?)\s*\]\]", re.IGNORECASE)


def _extract_attachments(
    text: str, base_dir: str | os.PathLike | None = None
) -> tuple[str, list[str]]:
    """Pull ``[[attach: <path>]]`` markers out of an agent reply.

    Returns ``(text_without_markers, [existing_file_paths])``. The agent writes a
    file (e.g. an HTML report) and references it with the marker; the gateway
    strips the marker from the visible text and attaches the file to the outbound
    message (``OutboundMessage.media`` → Telegram ``send_document``). Only paths
    that resolve to an existing readable file are attached; every marker is
    stripped from the text regardless (so a typo'd path never leaks raw).

    A **relative** path resolves against ``base_dir`` — the session's cwd (its
    per-chat work dir) — so the agent can attach a file it just wrote with a plain
    ``[[attach: report.html]]`` instead of spelling out an absolute path. Absolute
    and ``~`` paths are used as-is. Without ``base_dir`` a relative path resolves
    against the process cwd (legacy behavior).
    """
    if not text or "[[" not in text:
        return text, []
    paths: list[str] = []
    seen: set[str] = set()
    for match in _ATTACH_RE.finditer(text):
        raw = match.group(1).strip().strip("'\"")
        path = os.path.expanduser(raw)
        if base_dir and not os.path.isabs(path):
            path = os.path.join(str(base_dir), path)
        if path and path not in seen and os.path.isfile(path):
            seen.add(path)
            paths.append(path)
    clean = _ATTACH_RE.sub("", text).strip()
    return clean, paths


_ASK_RE = re.compile(r"\[\[\s*ask\s*:\s*([^\]]+?)\s*\]\]", re.IGNORECASE)


def _extract_ask(text: str) -> tuple[str, str, list[str]]:
    """Pull a ``[[ask: <question> | <option> | <option> …]]`` marker out of an
    agent reply (the Telegram counterpart of Claude Code's AskUserQuestion).

    Returns ``(text_without_marker, question, options)``. The agent ends its
    reply with the marker to offer tappable answer buttons; the gateway strips
    it, shows ``question`` above the buttons, and a tap sends the chosen option
    back as the user's next message. Only the FIRST marker is honored (one
    question per reply). With fewer than 2 options the marker is still stripped
    but no buttons are produced — a "question" with no real choices isn't a
    button prompt.
    """
    if not text or "[[" not in text:
        return text, "", []
    match = _ASK_RE.search(text)
    if not match:
        return text, "", []
    clean = _ASK_RE.sub("", text).strip()
    parts = [p.strip() for p in match.group(1).split("|")]
    parts = [p for p in parts if p]
    if len(parts) < 3:  # need a question + at least two options
        return clean, "", []
    question, *options = parts
    return clean, question, options[:8]  # Telegram keyboards: keep it sane


def _format_gateway_error(exc: Exception, *, camera_context: bool = False) -> str:
    """Return a short, user-facing gateway error message."""
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    if "claude oauth refresh failed" in lowered:
        return (
            "[ohmo gateway error] Claude subscription auth refresh failed. "
            "Run `oh auth claude-login` again or switch the gateway profile."
        )
    if "claude oauth refresh token is invalid or expired" in lowered:
        return (
            "[ohmo gateway error] Claude subscription token is expired. "
            "Run `claude auth login`, then `oh auth claude-login`, or switch the gateway profile."
        )
    if "auth source not found" in lowered or "access token" in lowered:
        return (
            "[ohmo gateway error] Authentication is not configured for the current "
            "gateway profile. Run `oh setup` or `ohmo config`."
        )
    if (
        "api key" in lowered
        or "oauth" in lowered
        or re.search(r"\b(?:auth|authenticat\w*|authoriz\w*|credential\w*|unauthori[sz]ed|401)\b", lowered)
    ):
        return (
            "[ohmo gateway error] Authentication failed for the current gateway "
            "profile. Check `oh auth status` and `ohmo config`."
        )
    if camera_context and isinstance(exc, ValueError) and lowered.startswith("camera "):
        return "Не удалось подтвердить запись этой порции."
    return f"[ohmo gateway error] {message}"


class OhmoGatewayBridge:
    """Consume inbound messages and publish assistant replies."""

    def __init__(
        self,
        *,
        bus: MessageBus,
        runtime_pool: OhmoSessionRuntimePool,
        restart_gateway: Callable[[object, str], Awaitable[None] | None] | None = None,
        workspace: str | Path | None = None,
        feishu_group_policy: str = "open",
        message_coalesce_window: float = 0.0,
        message_coalesce_media_window: float = 0.0,
        message_coalesce_max: int = 20,
        contact_store: ContactStore | None = None,
        debug_progress_chats: list[str] | None = None,
        camera_ingress: CameraIngress | None = None,
    ) -> None:
        self._bus = bus
        self._runtime_pool = runtime_pool
        self._restart_gateway = restart_gateway
        self._workspace = workspace
        self._feishu_group_policy = _normalize_feishu_group_policy(feishu_group_policy)
        self._running = False
        self._session_tasks: dict[str, asyncio.Task[None]] = {}
        self._session_cancel_reasons: dict[str, str] = {}
        self._coalesce_window = float(message_coalesce_window)
        self._coalesce_media_window = float(message_coalesce_media_window)
        self._coalesce_max = int(message_coalesce_max)
        self._pending: dict[str, list[InboundMessage]] = {}
        self._pending_deadline: dict[str, float] = {}
        # In-flight dispatched turns, so a shutdown can record what it interrupts.
        self._inflight: dict[str, InboundMessage] = {}
        self._contact_store = contact_store
        # Telegram progress is quiet by default.  Only this exact chat-ID
        # allowlist opts into detailed progress; /debug and /quiet mutate it.
        self._debug_chats: set[str] = {str(c) for c in (debug_progress_chats or [])}
        self._camera_ingress = camera_ingress

    async def run(self) -> None:
        self._running = True
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._bus.consume_inbound(), timeout=self._next_flush_timeout()
                )
            except asyncio.TimeoutError:
                # Flush tick: a buffered burst whose debounce window elapsed.
                await self._flush_due()
                continue
            except asyncio.CancelledError:
                break

            if not self._should_process_message(message):
                logger.info(
                    "ohmo inbound ignored channel=%s chat_id=%s sender_id=%s reason=feishu_group_policy policy=%s content=%r",
                    message.channel,
                    message.chat_id,
                    message.sender_id,
                    self._feishu_group_policy,
                    _content_snippet(message.content),
                )
                continue

            if self._camera_ingress is not None and message.sender_id != "__camera__":
                self._camera_ingress.process_real_inbound(message)
                if message.metadata.get("_camera_classifier_feedback") is CAMERA_AUTHORITY:
                    # Owner-authorized classifier feedback is journaled by the
                    # Camera ingress and recorded from the native receipt only.
                    # It never enters the model or nutrition turn path.
                    self._record_camera_feedback_export(message)
                    continue

            session_key = session_key_for_message(message)
            logger.info(
                "ohmo inbound received channel=%s chat_id=%s sender_id=%s session_key=%s content=%r",
                message.channel,
                message.chat_id,
                message.sender_id,
                session_key,
                _content_snippet(message.content),
            )
            self._record_contact(message)

            stripped = message.content.strip()
            group_args = _parse_group_command(message.content)
            is_synthetic = (
                bool(message.metadata.get("_synthetic"))
                or message.sender_id == "__scheduler__"
                or message.metadata.get("_camera_authority") is CAMERA_AUTHORITY
                or message.metadata.get("_camera_unbound") is CAMERA_AUTHORITY
            )
            is_special = (
                stripped in ("/stop", "/restart", "/new", "/clear", "/debug", "/quiet", "/verbose")
                or group_args is not None
                or is_synthetic
            )

            if is_special:
                is_control = stripped in (
                    "/stop",
                    "/restart",
                    "/new",
                    "/clear",
                    "/debug",
                    "/quiet",
                    "/verbose",
                )
                # Dispatch any buffered plain messages first (arrival order, no
                # loss), THEN handle the special message verbatim. Control
                # commands stop/reset the session, so cancelling the just-flushed
                # turn is benign — flush without waiting. Synthetic-reminder and
                # /group paths dispatch their OWN turn right after, which would
                # else cancel the not-yet-started buffered task before its stream
                # runs (data loss); wait for that turn to finish first.
                await self._flush_pending(session_key, wait=not is_control)
                if stripped == "/stop":
                    await self._handle_stop(message, session_key)
                    continue
                if stripped == "/restart":
                    await self._handle_restart(message, session_key)
                    continue
                if stripped in ("/new", "/clear"):
                    await self._handle_new(message, session_key)
                    continue
                if stripped in ("/debug", "/quiet", "/verbose"):
                    await self._handle_progress_toggle(
                        message,
                        session_key,
                        enable_debug=stripped in ("/debug", "/verbose"),
                    )
                    continue
                if group_args is not None:
                    prepared = await self._prepare_group_prompt_message(
                        message, session_key, group_args
                    )
                    if prepared is None:
                        continue
                    message = prepared
                    session_key = session_key_for_message(message)
                if (
                    message.sender_id == "__camera__"
                    and message.metadata.get("_camera_authority") is CAMERA_AUTHORITY
                ):
                    # A Camera photo may arrive while the configured account's
                    # ordinary turn is running. Do not cancel it just to analyze a photo.
                    active = self._session_tasks.get(session_key)
                    if active is not None and not active.done():
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await asyncio.shield(active)
                await self._dispatch(message, session_key)
                await self._flush_due()
                continue

            effective_window = self._effective_coalesce_window(message, session_key)
            if effective_window <= 0:
                # OFF switch: behave exactly like the pre-coalescer code.
                await self._dispatch(message, session_key)
                await self._flush_due()
                continue

            buffer = self._pending.setdefault(session_key, [])
            buffer.append(message)
            self._pending_deadline[session_key] = time.monotonic() + effective_window
            if len(buffer) >= self._coalesce_max:
                await self._flush_pending(session_key)
            await self._flush_due()

    def _effective_coalesce_window(self, message: InboundMessage, session_key: str) -> float:
        has_media = bool(message.media) or any(
            buffered.media for buffered in self._pending.get(session_key, ())
        )
        window = self._coalesce_media_window if has_media else self._coalesce_window
        return window if window > 0 else self._coalesce_window

    def _record_contact(self, message: InboundMessage) -> None:
        if self._contact_store is None:
            return
        md = message.metadata or {}
        if md.get("_synthetic") or message.sender_id == "__scheduler__":
            return
        is_group = bool(md.get("is_group")) or str(md.get("chat_type") or "").strip().lower() in {
            "group",
            "supergroup",
            "channel",
            "chat",
            "room",
        }
        if is_group:
            return
        try:
            self._contact_store.record_inbound(
                channel=message.channel,
                chat_id=str(message.chat_id),
                user_id=(str(md["user_id"]) if md.get("user_id") is not None else None),
                username=(str(md["username"]) if md.get("username") else None),
                first_name=(str(md["first_name"]) if md.get("first_name") else None),
                display_name=(
                    str(md["sender_display_name"]) if md.get("sender_display_name") else None
                ),
            )
        except Exception:
            logger.warning(
                "ohmo contact record failed channel=%s chat_id=%s",
                message.channel,
                message.chat_id,
                exc_info=True,
            )

    def _record_camera_feedback_export(self, message: InboundMessage) -> str | None:
        """Capture accepted native classifier feedback without starting a model turn."""
        metadata = message.metadata or {}
        receipt = metadata.get("_camera_feedback_receipt")
        options = metadata.get("native_keyboard_options")
        index = metadata.get("native_keyboard_selected_index")
        label = metadata.get("native_keyboard_selected_label")
        callback_id = metadata.get("callback_query_id")
        candidate_id = metadata.get("_camera_ingress_callback_candidate")
        camera_config = getattr(getattr(self._runtime_pool, "_gateway_config", None),
                                "camera_ingress", None)
        bundles = getattr(self._runtime_pool, "_bundles", None)
        workspace = getattr(self._runtime_pool, "_workspace", None)
        session_key = session_key_for_message(message)
        bundle = bundles.get(session_key) if isinstance(bundles, dict) else None
        if (
            not isinstance(receipt, dict)
            or metadata.get("_camera_ingress_callback_eligible") is not True
            or camera_config is None
            or not getattr(camera_config, "enabled", False)
            or message.channel != "telegram"
            or str(message.chat_id) != str(getattr(camera_config, "chat_id", ""))
            or message.sender_id.split("|", 1)[0] != str(getattr(camera_config, "principal", ""))
            or receipt.get("kind") != "not_food"
            or receipt.get("candidate_id") != candidate_id
            or receipt.get("tenant_id") != getattr(camera_config, "tenant_id", None)
            or receipt.get("owner_principal") != f"telegram:{getattr(camera_config, 'principal', '')}"
            or type(receipt.get("native_photo_id")) is not int
            or receipt.get("native_photo_id") <= 0
            or receipt.get("native_message_id") != str(metadata.get("native_message_id"))
            or receipt.get("source_message_id") != str(metadata.get("message_id"))
            or receipt.get("callback_query_id") != callback_id
            or receipt.get("operation_id") != f"{candidate_id}:classifier_feedback:{callback_id}"
            or metadata.get("callback_query") is not True
            or not isinstance(options, list)
            or type(index) is not int or not 0 <= index < len(options)
            or options[index] != "Это не еда" or label != "Это не еда"
            or metadata.get("callback_data") != f"ask:{index}"
            or metadata.get("native_keyboard_reflection_confirmed") is not True
            or "✅ Это не еда" not in str(metadata.get("native_keyboard_reflection") or "")
            or bundle is None
            or not isinstance(workspace, (str, Path))
            or not _evals_capture_enabled(getattr(self._runtime_pool, "_gateway_config", None))
        ):
            return None
        session_id = getattr(bundle, "session_id", None)
        if not isinstance(session_id, str) or not session_id:
            return None
        try:
            bound_receipt = dict(receipt)
            bound_receipt["gateway_session_id"] = session_id
            metadata["_camera_feedback_receipt"] = bound_receipt
            recorder = GatewayEvalRecorder.start(
                workspace=workspace,
                bundle=bundle,
                message=message,
                session_key=session_key,
                user_text=message.content,
                user_goal="Owner marked the Camera image as not food.",
            )
            recorder.record_gateway_final(
                text=str(metadata["native_keyboard_reflection"]),
                metadata={"camera_feedback_operation_id": receipt["operation_id"]},
            )
            recorder.finish(status="completed")
            return recorder.episode_id
        except Exception:
            logger.warning("Camera classifier feedback eval capture failed", exc_info=True)
            return None

    async def _dispatch(self, message: InboundMessage, session_key: str) -> None:
        await self._interrupt_session(
            session_key,
            reason="replaced by a newer user message",
            notify=self._build_interrupt_notice(message, session_key),
        )
        task = asyncio.create_task(
            self._process_message(message, session_key),
            name=f"ohmo-session:{session_key}",
        )
        self._session_tasks[session_key] = task
        self._inflight[session_key] = message
        task.add_done_callback(lambda finished, key=session_key: self._cleanup_task(key, finished))

    def _is_quiet_telegram(self, message: InboundMessage) -> bool:
        """Whether this chat uses compact (quiet) progress on Telegram."""
        return message.channel == "telegram" and str(message.chat_id) not in self._debug_chats

    def _build_interrupt_notice(self, message: InboundMessage, session_key: str) -> OutboundMessage:
        """Build the cancellation notice for the previous task.

        In quiet Telegram mode: a structured ``cancelled`` collapse event so
        the compact status is the single owner — it is edited once to a
        terminal phrase instead of sending a standalone message.

        In verbose mode or on non-Telegram channels: a standalone notice
        (no ``_progress`` so it is durable and not subject to progress
        drop logic).
        """
        if self._is_quiet_telegram(message):
            return OutboundMessage(
                channel=message.channel,
                chat_id=message.chat_id,
                content="",
                metadata={
                    "_collapse": True,
                    "_session_key": session_key,
                    "progress_event": {
                        "kind": "cancelled",
                        "reason": "replaced by a newer user message",
                    },
                },
            )
        return OutboundMessage(
            channel=message.channel,
            chat_id=message.chat_id,
            content="⏹️ Остановил предыдущую задачу, перехожу к новому сообщению.",
            metadata={"_session_key": session_key},
        )

    def _next_flush_timeout(self) -> float:
        if not self._pending_deadline:
            return 1.0
        now = time.monotonic()
        remaining = min(deadline - now for deadline in self._pending_deadline.values())
        return max(0.05, min(remaining, 1.0))

    async def _flush_due(self) -> None:
        now = time.monotonic()
        for session_key, deadline in list(self._pending_deadline.items()):
            if now >= deadline:
                # Isolate each flush: a dispatch failure for one session must not
                # kill the run loop or strand other sessions' buffered messages.
                # ``_flush_pending`` already popped the buffer+deadline, so a
                # faulted session is not retried in a tight loop.
                try:
                    await self._flush_pending(session_key)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("ohmo reminder flush failed session_key=%s", session_key)

    async def _flush_pending(self, session_key: str, *, wait: bool = False) -> None:
        buffer = self._pending.pop(session_key, None)
        self._pending_deadline.pop(session_key, None)
        if not buffer:
            return
        message = _coalesce(buffer)
        await self._dispatch(message, session_key)
        if wait:
            # Run the just-dispatched buffered turn to completion before
            # returning, so a special handler that dispatches its own turn next
            # does not cancel this not-yet-started task and drop the buffered
            # messages. Shield it so awaiting here never cancels the turn; its
            # own failures are already logged inside _process_message.
            task = self._session_tasks.get(session_key)
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.shield(task)

    def stop(self) -> None:
        self._running = False
        self._persist_interrupted_requests()
        for session_key, task in list(self._session_tasks.items()):
            self._session_cancel_reasons[session_key] = "gateway stopping"
            task.cancel()

    def _persist_interrupted_requests(self) -> None:
        """Record in-flight + buffered user requests so a restart can recover.

        A SIGTERM (``systemctl restart``) or crash cancels the active turn and
        drops any coalesce-buffered messages, none of which Telegram redelivers
        (the update was already acked). Persist them so startup can tell the user
        their message was interrupted instead of silently losing it.
        """
        pending: list[InboundMessage] = list(self._inflight.values())
        for buffered in self._pending.values():
            pending.extend(buffered)
        records: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for msg in pending:
            md = msg.metadata or {}
            if md.get("_synthetic") or msg.sender_id == "__scheduler__":
                continue
            key = (msg.channel, str(msg.chat_id), msg.content)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "channel": msg.channel,
                    "chat_id": str(msg.chat_id),
                    "content": msg.content,
                    "session_key": session_key_for_message(msg),
                }
            )
        if not records:
            return
        try:
            path = get_gateway_interrupted_requests_path(self._workspace)
            path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            logger.info(
                "ohmo gateway persisted %d interrupted request(s) for restart recovery",
                len(records),
            )
        except Exception:  # noqa: BLE001 - best effort during shutdown
            logger.exception("ohmo gateway failed to persist interrupted requests")

    async def _handle_stop(self, message, session_key: str) -> None:
        stopped = await self._interrupt_session(
            session_key,
            reason="stopped by user command",
        )
        if not stopped:
            await self._publish_command_reply(message, session_key, "Сейчас нет активной задачи.")
            return
        if self._is_quiet_telegram(message):
            await self._bus.publish_outbound(
                OutboundMessage(
                    channel=message.channel,
                    chat_id=message.chat_id,
                    content="",
                    metadata={
                        "_collapse": True,
                        "_session_key": session_key,
                        "progress_event": {
                            "kind": "cancelled",
                            "reason": "stopped by user command",
                        },
                    },
                )
            )
        else:
            await self._publish_command_reply(message, session_key, "⏹️ Остановил текущую задачу.")

    async def _handle_new(self, message, session_key: str) -> None:
        """/new (alias /clear): cancel the in-flight turn and HARD-reset the
        session so the next message starts a fresh conversation.

        Previously /new was not a recognized command — it fell through to the
        model, which only *said* "Контекст сброшен" while the accumulated history
        (and session_id) stayed intact. So context never actually reset and the
        model kept anchoring on its own past mistakes.
        """
        await self._interrupt_session(session_key, reason="reset by /new")
        await self._runtime_pool.reset_session(session_key)
        await self._publish_command_reply(
            message, session_key, "🧹 Контекст сброшен — начинаю новую сессию."
        )

    async def _handle_progress_toggle(
        self, message, session_key: str, *, enable_debug: bool
    ) -> None:
        """Add or remove this Telegram chat from the detailed-progress allowlist."""
        chat_id = str(message.chat_id)
        if enable_debug:
            self._debug_chats.add(chat_id)
            reply = "🔊 Подробный прогресс включён для этого чата."
        else:
            self._debug_chats.discard(chat_id)
            reply = "🔇 Тихий прогресс включён для этого чата — покажу один статус со спиннером."
        try:
            self._persist_progress_overrides()
        except Exception:
            logger.exception("ohmo failed to persist progress overrides chat_id=%s", chat_id)
        await self._publish_command_reply(message, session_key, reply)

    def _persist_progress_overrides(self) -> None:
        """Round-trip gateway.json, updating the canonical progress allowlist."""
        config = load_gateway_config(self._workspace)
        config.debug_progress_chats = sorted(self._debug_chats)
        save_gateway_config(config, self._workspace)

    async def _handle_restart(self, message, session_key: str) -> None:
        await self._interrupt_session(
            session_key,
            reason="restarting gateway by user command",
        )
        await self._bus.publish_outbound(
            OutboundMessage(
                channel=message.channel,
                chat_id=message.chat_id,
                content="🔄 正在重启 gateway，马上回来。\nRestarting the gateway now. I'll be back in a moment.",
                metadata={"_session_key": session_key},
            )
        )
        if self._restart_gateway is not None:
            result = self._restart_gateway(message, session_key)
            if asyncio.iscoroutine(result):
                await result

    async def _prepare_group_prompt_message(
        self,
        message,
        session_key: str,
        args: str,
    ) -> InboundMessage | None:
        """Convert a private /group command into an agent task."""
        if message.channel != "feishu":
            await self._publish_command_reply(
                message,
                session_key,
                "/group 当前只支持飞书。\n/group is currently only available for Feishu.",
            )
            return None

        chat_type = str(message.metadata.get("chat_type") or "").strip().lower()
        is_private = chat_type in {"p2p", "private", "im", "direct"} or (
            not chat_type and str(message.chat_id) == str(message.sender_id)
        )
        if not is_private:
            await self._publish_command_reply(
                message,
                session_key,
                "请在和 ohmo 的私聊里使用 /group 创建新群。\nUse /group in a private chat with ohmo to create a new group.",
            )
            return None

        metadata = dict(message.metadata)
        metadata["_ohmo_group_command"] = True
        metadata["_ohmo_group_raw_request"] = args
        prompt = _build_group_agent_prompt(args)
        return InboundMessage(
            channel=message.channel,
            sender_id=message.sender_id,
            chat_id=message.chat_id,
            content=prompt,
            timestamp=message.timestamp,
            media=list(message.media),
            metadata=metadata,
            session_key_override=message.session_key_override,
        )

    async def _publish_command_reply(self, message, session_key: str, content: str) -> None:
        await self._bus.publish_outbound(
            OutboundMessage(
                channel=message.channel,
                chat_id=message.chat_id,
                content=content,
                metadata={"_session_key": session_key},
            )
        )

    async def _interrupt_session(
        self,
        session_key: str,
        *,
        reason: str,
        notify: OutboundMessage | None = None,
    ) -> bool:
        task = self._session_tasks.get(session_key)
        if task is None or task.done():
            return False
        self._session_cancel_reasons[session_key] = reason
        task.cancel()
        if notify is not None:
            await self._bus.publish_outbound(notify)
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=3.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        return True

    async def _process_message(self, message, session_key: str) -> None:
        # Preserve thread metadata only for shared chats. Feishu p2p replies
        # should stay as normal private messages, not topic replies.
        inbound_meta = {k: message.metadata[k] for k in ("thread_id",) if k in message.metadata}
        chat_type = str(message.metadata.get("chat_type") or "").lower()
        if chat_type == "group" or inbound_meta.get("thread_id"):
            if "message_id" in message.metadata:
                inbound_meta["message_id"] = message.metadata["message_id"]
        # Collapse this turn's progress into one live status message? Read the
        # exact Telegram chat-ID per-turn so a prior /quiet or /debug is in effect.
        chat_id = str(message.chat_id)
        collapse = message.channel == "telegram" and (
            chat_id not in self._debug_chats
            or message.metadata.get("_camera_authority") is CAMERA_AUTHORITY
        )
        try:
            reply = ""
            final_media: list[str] = []
            final_metadata: dict[str, object] = {}
            delivered_assistant_updates: set[str] = set()
            stream_error = False
            async for update in self._runtime_pool.stream_message(message, session_key):
                if update.kind == "final":
                    reply = update.text
                    final_media = list(
                        getattr(update, "media", None)
                        or (update.metadata or {}).get("_media")
                        or []
                    )
                    final_metadata = dict(update.metadata or {})
                    final_metadata.pop("_collapse", None)
                    continue
                ephemeral_assistant_update = collapse and update.kind == "assistant_update"
                if (
                    update.kind == "assistant_update"
                    and update.text.strip()
                    and not ephemeral_assistant_update
                ):
                    delivered_assistant_updates.add(update.text.strip())
                if update.kind == "error":
                    stream_error = True
                progress_event = (update.metadata or {}).get("progress_event")
                has_structured_progress = (
                    isinstance(progress_event, dict)
                    and progress_event.get("kind") in {"tool", "inference"}
                ) or (
                    message.channel == "telegram"
                    and isinstance(progress_event, dict)
                    and progress_event.get("kind") == "todo"
                )
                if not update.text and not has_structured_progress:
                    continue
                logger.info(
                    "ohmo outbound update channel=%s chat_id=%s session_key=%s kind=%s content=%r",
                    message.channel,
                    message.chat_id,
                    session_key,
                    update.kind,
                    _content_snippet(update.text),
                )
                update_meta = {**inbound_meta, **(update.metadata or {})}
                if collapse and update.kind != "error":
                    # Runtime errors must use Telegram's normal send path: it
                    # clears any live compact status and renders the error.
                    update_meta["_collapse"] = True
                if ephemeral_assistant_update:
                    # In quiet Telegram, public pre-tool narration is part of
                    # the compact status rather than a durable assistant reply.
                    update_meta["_progress"] = True
                await self._bus.publish_outbound(
                    OutboundMessage(
                        channel=message.channel,
                        chat_id=message.chat_id,
                        content=update.text,
                        media=list(
                            getattr(update, "media", None)
                            or (update.metadata or {}).get("_media")
                            or []
                        ),
                        metadata=update_meta,
                    )
                )
        except asyncio.CancelledError:
            logger.info(
                "ohmo session interrupted channel=%s chat_id=%s session_key=%s reason=%s",
                message.channel,
                message.chat_id,
                session_key,
                self._session_cancel_reasons.get(session_key, "cancelled"),
            )
            raise
        except Exception as exc:  # pragma: no cover - gateway failure path
            logger.exception(
                "ohmo gateway failed to process inbound message channel=%s chat_id=%s sender_id=%s session_key=%s content=%r",
                message.channel,
                message.chat_id,
                message.sender_id,
                session_key,
                _content_snippet(message.content),
            )
            camera_error_context = (
                message.metadata.get("_camera_authority") is CAMERA_AUTHORITY
                and message.metadata.get("_camera_answer") == "yes"
            )
            reply = _format_gateway_error(exc, camera_context=camera_error_context)
        if not reply:
            if delivered_assistant_updates and not stream_error:
                await self._bus.publish_outbound(
                    OutboundMessage(
                        channel=message.channel,
                        chat_id=message.chat_id,
                        content="⚠️ Ответ не завершён; выше показан частичный результат.",
                        metadata={**inbound_meta, "_session_key": session_key, "_partial": True},
                    )
                )
            logger.info(
                "ohmo inbound finished without final reply channel=%s chat_id=%s session_key=%s",
                message.channel,
                message.chat_id,
                session_key,
            )
            return
        duplicate_assistant_update = reply.strip() in delivered_assistant_updates
        # Resolve a relative [[attach:]] path against the session's cwd (its
        # per-chat work dir) so the agent can attach a file it wrote with a plain
        # name. getattr keeps the bridge resilient to pool stubs lacking session_cwd.
        cwd_fn = getattr(self._runtime_pool, "session_cwd", None)
        base_dir = cwd_fn(message, session_key) if callable(cwd_fn) else None
        content, media = _extract_attachments(reply, base_dir=base_dir)
        content, question, options = _extract_ask(content)
        initial_camera_prompt = (
            message.metadata.get("_camera_authority") is CAMERA_AUTHORITY
            and message.metadata.get("_camera_initial_prompt") is CAMERA_AUTHORITY
        )
        if initial_camera_prompt and not options:
            # A model response without a bounded ask still gets the existing
            # caption's explicit consumption question and native choices.
            options = ["Я это съел", "Нет, не ел"]
        if initial_camera_prompt and options:
            # The admitted manifest, not the model's wording, decides whether
            # the classifier feedback choice is present.
            decision = message.metadata.get("_camera_classifier_decision")
            filtered_options = []
            seen_options = set()
            for option in options:
                normalized = option.strip().casefold()
                if option == "Это не еда" or normalized in seen_options:
                    continue
                seen_options.add(normalized)
                filtered_options.append(option)
            if decision == "ambiguous":
                # Keep the model's first-party choices, but reserve one native
                # slot for the correction that closes the loop on ambiguity.
                if not filtered_options:
                    filtered_options = ["Я это съел(а)", "Нет, не ел(а)"]
                options = filtered_options[:7]
                options.append("Это не еда")
            else:
                options = filtered_options
                if len(options) < 2:
                    if not options:
                        options = ["Я это съел(а)", "Нет, не ел(а)"]
                    else:
                        alternative = (
                            "Я это съел(а)"
                            if _classify_answer(options[0], anchored=False) == "no"
                            else "Нет, не ел(а)"
                        )
                        if alternative.casefold() != options[0].strip().casefold():
                            options.append(alternative)
        if options:
            # Show the question above the buttons (the visible text may already
            # carry context; append the question so the choices read clearly).
            content = (content + ("\n\n" if content else "") + question).strip()
        # Attachments can surface twice for the same file: the runtime final-reply
        # fallback (_extract_final_reply_media) matches a bare absolute image path,
        # while _extract_attachments matches that same path inside its
        # [[attach: ...]] marker. Concatenating both would attach — and Telegram
        # would send — the file twice. Dedup (order-preserving) so an image
        # referenced by an absolute [[attach:]] path is delivered exactly once.
        final_media_paths = list(dict.fromkeys([*final_media, *media]))
        if initial_camera_prompt:
            # The original photo already has a confirmed native message. Its
            # file must never flow into the normal media-send branch, including
            # after an edit failure or when the native receipt is unusable.
            def same_original_file(path: str) -> bool:
                try:
                    resolved = Path(path).resolve(strict=False)
                    return any(resolved == Path(original).resolve(strict=False)
                               for original in message.media or [])
                except (OSError, RuntimeError, TypeError, ValueError):
                    return path in set(message.media or [])

            final_media_paths = [path for path in final_media_paths if not same_original_file(path)]
        if duplicate_assistant_update and not final_media_paths and not options:
            logger.info(
                "ohmo final duplicates already-delivered assistant update channel=%s chat_id=%s session_key=%s",
                message.channel,
                message.chat_id,
                session_key,
            )
            return
        if duplicate_assistant_update:
            # The user already saw the text, but a repeated final can still
            # carry a file or interactive controls that must be delivered.
            content = question if options else ""
        logger.info(
            "ohmo outbound final channel=%s chat_id=%s session_key=%s media=%d buttons=%d content=%r",
            message.channel,
            message.chat_id,
            session_key,
            len(final_media_paths),
            len(options),
            _content_snippet(content),
        )
        # Reply-thread the FINAL answer under the user's message on Telegram
        # (gated by the channel's reply_to_message). Scoped to the final send —
        # it goes via send_message, where reply_parameters works. Progress is
        # left untouched: carrying message_id there would route it to the draft
        # API, which this non-business bot can't use.
        final_meta = {**inbound_meta, **final_metadata, "_session_key": session_key}
        if message.metadata.get("_camera_authority") is CAMERA_AUTHORITY:
            final_meta["_camera_candidate_id"] = message.metadata["_camera_candidate_id"]
            final_meta["_camera_authority"] = CAMERA_AUTHORITY
            final_meta["_camera_final"] = CAMERA_AUTHORITY
            photo_id = message.metadata.get("_camera_photo_id")
            if initial_camera_prompt and type(photo_id) is int and photo_id > 0:
                final_meta["_camera_photo_id"] = photo_id
                final_meta["_camera_caption"] = message.metadata.get("_camera_caption")
                final_meta["_camera_edit_existing_photo"] = CAMERA_AUTHORITY
                final_meta["_camera_initial_prompt"] = CAMERA_AUTHORITY
            final_meta.pop("_camera_turn_id", None)
            turn_id = message.metadata.get("_camera_turn_id")
            if isinstance(turn_id, str):
                final_meta["_camera_turn_id"] = turn_id
        if message.channel == "telegram" and "message_id" in message.metadata:
            final_meta["message_id"] = message.metadata["message_id"]
        await self._bus.publish_outbound(
            OutboundMessage(
                channel=message.channel,
                chat_id=message.chat_id,
                content=content,
                media=final_media_paths,
                buttons=options,
                metadata=final_meta,
            )
        )

    def _cleanup_task(
        self,
        session_key: str,
        task: asyncio.Task[None],
    ) -> None:
        current = self._session_tasks.get(session_key)
        if current is task:
            self._session_tasks.pop(session_key, None)
            self._inflight.pop(session_key, None)
        self._session_cancel_reasons.pop(session_key, None)

    def _should_process_message(self, message: InboundMessage) -> bool:
        # Scheduler-originated synthetic reminders bypass the channel-layer ACL
        # by design: they come from a stored, owner-created reminder, not from an
        # arbitrary group member. Without this, an agentic reminder for a Feishu
        # group under a mention/managed group_policy would be silently dropped at
        # fire time (and the occurrence already consumed by mark_fired).
        if message.metadata.get("_synthetic"):
            return True
        if message.channel != "feishu":
            return True
        chat_type = str(message.metadata.get("chat_type") or "").strip().lower()
        if chat_type != "group":
            return True
        policy = self._feishu_group_policy
        if policy == "open":
            return True
        mentioned = _message_mentions_bot(message)
        if policy == "mention":
            return mentioned
        if policy == "managed":
            return self._is_managed_feishu_group(message.chat_id)
        if policy == "managed_or_mention":
            return mentioned or self._is_managed_feishu_group(message.chat_id)
        return mentioned

    def _is_managed_feishu_group(self, chat_id: str) -> bool:
        try:
            return (
                load_managed_group_record(
                    workspace=self._workspace,
                    channel="feishu",
                    chat_id=chat_id,
                )
                is not None
            )
        except Exception:
            logger.exception("failed to load ohmo managed group metadata chat_id=%s", chat_id)
            return False


def _parse_group_command(content: str) -> str | None:
    stripped = content.strip()
    parts = stripped.split(maxsplit=1)
    if not parts or parts[0] != "/group":
        return None
    if len(parts) == 1:
        return ""
    return parts[1].strip()


def _coalesce(messages: list[InboundMessage]) -> InboundMessage:
    """Merge a burst of same-session plain messages into one turn.

    A single message passes through unchanged. For multiple, text is joined in
    arrival order; ``media`` (URLs/paths) is concatenated so nothing is dropped;
    channel/chat/sender/metadata are taken from the LAST message so the reply
    threads under the most recent one. All messages share a ``session_key``
    (which encodes sender for shared chats), so senders are never merged.
    """
    if len(messages) == 1:
        return messages[0]
    last = messages[-1]
    content = "\n\n".join(m.content for m in messages)
    media: list[str] = []
    media_sources: list[dict[str, object]] = []
    for m in messages:
        media.extend(m.media)
        source_id = m.metadata.get("message_id") if isinstance(m.metadata, dict) else None
        source_time = (
            m.metadata.get("received_at")
            if m.channel == "telegram" and "received_at" in m.metadata
            else m.timestamp
        )
        received_at = None
        if isinstance(source_time, datetime) and source_time.tzinfo is not None:
            received_at = source_time.isoformat()
        elif isinstance(source_time, str):
            try:
                parsed = datetime.fromisoformat(source_time.strip())
            except ValueError:
                parsed = None
            if parsed is not None and parsed.tzinfo is not None:
                received_at = parsed.isoformat()
        for _ in m.media:
            media_sources.append({
                "source_message_id": str(source_id) if source_id is not None else None,
                "received_at": received_at,
            })
    metadata = dict(last.metadata)
    metadata["_coalesced_media_sources"] = media_sources
    metadata["_coalesced_media_provenance_authority"] = (
        COALESCED_ATTACHMENT_PROVENANCE_AUTHORITY
    )
    return InboundMessage(
        channel=last.channel,
        sender_id=last.sender_id,
        chat_id=last.chat_id,
        content=content,
        timestamp=last.timestamp,
        media=media,
        metadata=metadata,
        session_key_override=last.session_key_override,
    )


def _build_group_agent_prompt(raw_request: str) -> str:
    request = raw_request.strip() or "(user did not provide details)"
    return (
        "The user invoked `/group` from a Feishu private chat.\n"
        "Your task is to create a dedicated Feishu group for this request.\n\n"
        "Use the `ohmo_create_feishu_group` tool exactly once if you can infer a safe group name. "
        "You, the model, must decide the final `name`, optional `repo`, and optional `cwd` from the user's "
        "natural-language request and available local context. If the cwd is not obvious, inspect the filesystem "
        "before calling the tool. If there is not enough information to choose safely, ask one concise clarification "
        "instead of calling the tool. Do not create the group via bash or direct API calls.\n\n"
        f"User /group request:\n{request}"
    )


def _normalize_feishu_group_policy(value: str | None) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "all": "open",
        "always": "open",
        "always_reply": "open",
        "managed_mention": "managed_or_mention",
        "managed_or_at": "managed_or_mention",
        "at": "mention",
        "mentions": "mention",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {"open", "mention", "managed", "managed_or_mention"}:
        return normalized
    return "managed_or_mention"


def _message_mentions_bot(message: InboundMessage) -> bool:
    value = message.metadata.get("mentions_bot")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False

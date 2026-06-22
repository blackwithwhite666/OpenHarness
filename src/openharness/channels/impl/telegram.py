"""Telegram channel implementation using python-telegram-bot."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time

from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaAudio,
    InputMediaDocument,
    InputMediaPhoto,
    InputMediaVideo,
    ReplyParameters,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from openharness.channels.bus.events import OutboundMessage
from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.base import BaseChannel, resolve_channel_state_dir
from openharness.channels.live_location import LiveLocationStore
from openharness.config.schema import TelegramConfig
from openharness.utils.helpers import split_message

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE_LEN = 4000  # Telegram message character limit

_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")

_REPLY_QUOTE_MAX = 500  # cap the quoted antecedent inlined into the agent prompt


def _reply_context(reply) -> tuple[str, dict]:
    """Build an inline quote prefix + metadata from a replied-to message.

    Telegram delivers a reply (``reply_to_message``) without inlining the quoted
    message, so a bare follow-up like "а тут?" reaches the agent with no
    antecedent. Surfacing the quoted text + author lets the agent resolve the
    reference instead of hallucinating. Returns ``("", {})`` when there is no
    reply.
    """
    if reply is None:
        return "", {}
    author_user = getattr(reply, "from_user", None)
    if author_user is None:
        author = "unknown"
    elif getattr(author_user, "is_bot", False):
        author = "you (the bot)"
    else:
        author = (
            getattr(author_user, "first_name", None)
            or getattr(author_user, "username", None)
            or "user"
        )
    quoted = (getattr(reply, "text", None) or getattr(reply, "caption", None) or "").strip()
    if not quoted:
        venue = getattr(reply, "venue", None)
        loc = getattr(reply, "location", None)
        if venue is not None:
            quoted = _format_venue(venue)
        elif loc is not None:
            quoted = _format_location(loc)
    if not quoted:
        for attr, label in (
            ("photo", "photo"),
            ("voice", "voice"),
            ("audio", "audio"),
            ("document", "file"),
            ("sticker", "sticker"),
            ("video", "video"),
        ):
            if getattr(reply, attr, None):
                quoted = f"[{label}]"
                break
    if not quoted:
        quoted = "[no text]"
    if len(quoted) > _REPLY_QUOTE_MAX:
        quoted = quoted[:_REPLY_QUOTE_MAX] + "…"
    prefix = f'[In reply to {author}: "{quoted}"]'
    meta = {
        "reply_to_message_id": getattr(reply, "message_id", None),
        "reply_to_text": quoted,
    }
    return prefix, meta


def _media_filename(media_file, ext: str) -> str:
    """Collision-free on-disk name for a downloaded Telegram media file.

    Use ``file_unique_id`` (stable, distinct per file), NOT ``file_id[:16]``:
    Telegram ``file_id``s within a chat share a long common prefix, so the old
    16-char truncation mapped every voice note to a handful of names — a burst
    of voices overwrote each other on disk and only the last survived.
    """
    stem = getattr(media_file, "file_unique_id", None) or getattr(media_file, "file_id", "")
    stem = re.sub(r"[^A-Za-z0-9_-]", "", stem)[:48] or "media"
    return f"{stem}{ext}"


def _fmt_coord(lat: float, lon: float) -> str:
    return f"{lat:.5f}, {lon:.5f}"


def _maps_link(lat: float, lon: float) -> str:
    # A plain maps URL the agent can echo back or open via the maps skill.
    return f"https://maps.yandex.ru/?pt={lon:.5f},{lat:.5f}&z=16&l=map"


def _format_location(loc) -> str:
    """A static location pin -> inline text the agent can act on."""
    parts = [f"[location: {_fmt_coord(loc.latitude, loc.longitude)}"]
    acc = getattr(loc, "horizontal_accuracy", None)
    if acc:
        parts.append(f", ±{int(acc)}m")
    parts.append(f"; {_maps_link(loc.latitude, loc.longitude)}]")
    return "".join(parts)


def _format_venue(venue) -> str:
    loc = venue.location
    coord = _fmt_coord(loc.latitude, loc.longitude) if loc else "?"
    title = venue.title or "venue"
    address = venue.address or ""
    body = f"«{title}»"
    if address:
        body += f", {address}"
    return f"[venue: {body} ({coord})]"


def _live_expires_at(message, loc) -> float | None:
    """Epoch seconds when this live share stops updating, or ``None`` if static."""
    live_period = getattr(loc, "live_period", None)
    if not live_period:
        return None
    base = None
    if getattr(message, "date", None) is not None:
        with contextlib.suppress(Exception):
            base = message.date.timestamp()
    if base is None:
        base = time.time()
    return base + float(live_period)


def _format_live_started(loc, expires_at: float | None) -> str:
    body = f"[live location started: {_fmt_coord(loc.latitude, loc.longitude)}"
    if expires_at is not None:
        mins = max(0, int((expires_at - time.time()) // 60))
        body += f"; updates for ~{mins} min"
    body += f"; {_maps_link(loc.latitude, loc.longitude)}]"
    return body


def _format_live_current(record: dict, now: float) -> str:
    lat, lon = record["latitude"], record["longitude"]
    ago = max(0, int(now - record.get("updated_at", now)))
    body = f"[current live location: {_fmt_coord(lat, lon)}; updated {ago}s ago"
    exp = record.get("expires_at")
    if exp is not None:
        left = max(0, int((exp - now) // 60))
        body += f", ~{left} min left"
    body += f"; {_maps_link(lat, lon)}]"
    return body


def _split_table_row(line: str) -> list[str]:
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [cell.strip() for cell in inner.split("|")]


def _is_table_separator(line: str) -> bool:
    if not _TABLE_ROW_RE.match(line):
        return False
    cells = _split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-+:?", cell) for cell in cells)


def _render_aligned_table(rows: list[list[str]]) -> str:
    """Markdown table rows -> a monospace, column-aligned block (Telegram has no
    <table>; a left raw ``| a | b |`` reads as garbage)."""
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    widths = [max(len(r[c]) for r in rows) for c in range(ncols)]

    def fmt(r: list[str]) -> str:
        return " | ".join(r[c].ljust(widths[c]) for c in range(ncols)).rstrip()

    sep = "-+-".join("-" * widths[c] for c in range(ncols))
    return "\n".join([fmt(rows[0]), sep, *(fmt(r) for r in rows[1:])])


def _convert_md_tables(text: str, save_block) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if (
            i + 1 < len(lines)
            and _TABLE_ROW_RE.match(lines[i])
            and not _is_table_separator(lines[i])
            and _is_table_separator(lines[i + 1])
        ):
            rows = [_split_table_row(lines[i])]
            j = i + 2
            while (
                j < len(lines)
                and _TABLE_ROW_RE.match(lines[j])
                and not _is_table_separator(lines[j])
            ):
                rows.append(_split_table_row(lines[j]))
                j += 1
            out.append(save_block(_render_aligned_table(rows)))
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def _markdown_to_telegram_html(text: str) -> str:
    """
    Convert markdown to Telegram-safe HTML.
    """
    if not text:
        return ""

    # 1. Extract and protect code blocks (preserve content from other processing)
    code_blocks: list[str] = []
    def save_code_block(m: re.Match) -> str:
        code_blocks.append(m.group(1))
        return f"\x00CB{len(code_blocks) - 1}\x00"

    text = re.sub(r'```[\w]*\n?([\s\S]*?)```', save_code_block, text)

    # 1b. Markdown tables -> aligned monospace block (protected like a code block)
    def save_table(aligned: str) -> str:
        code_blocks.append(aligned)
        return f"\x00CB{len(code_blocks) - 1}\x00"

    text = _convert_md_tables(text, save_table)

    # 2. Extract and protect inline code
    inline_codes: list[str] = []
    def save_inline_code(m: re.Match) -> str:
        inline_codes.append(m.group(1))
        return f"\x00IC{len(inline_codes) - 1}\x00"

    text = re.sub(r'`([^`]+)`', save_inline_code, text)

    # 3. Headers # Title -> just the title text
    text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)

    # 4. Blockquotes > text -> just the text (before HTML escaping)
    text = re.sub(r'^>\s*(.*)$', r'\1', text, flags=re.MULTILINE)

    # 5. Escape HTML special characters
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 6. Links [text](url) - must be before bold/italic to handle nested cases
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)

    # 7. Bold **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)

    # 8. Italic _text_ (avoid matching inside words like some_var_name)
    text = re.sub(r'(?<![a-zA-Z0-9])_([^_]+)_(?![a-zA-Z0-9])', r'<i>\1</i>', text)

    # 9. Strikethrough ~~text~~
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)

    # 10. Bullet lists - item -> • item
    text = re.sub(r'^[-*]\s+', '• ', text, flags=re.MULTILINE)

    # 11. Restore inline code with HTML tags
    for i, code in enumerate(inline_codes):
        # Escape HTML in code content
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00IC{i}\x00", f"<code>{escaped}</code>")

    # 12. Restore code blocks with HTML tags
    for i, code in enumerate(code_blocks):
        # Escape HTML in code content
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00CB{i}\x00", f"<pre><code>{escaped}</code></pre>")

    return text


class TelegramChannel(BaseChannel):
    """
    Telegram channel using long polling.

    Simple and reliable - no webhook/public IP needed.
    """

    name = "telegram"

    # Commands registered with Telegram's command menu
    BOT_COMMANDS = [
        BotCommand("start", "Start the bot"),
        BotCommand("new", "Start a new conversation"),
        BotCommand("stop", "Stop the current task"),
        BotCommand("help", "Show available commands"),
    ]

    def __init__(
        self,
        config: TelegramConfig,
        bus: MessageBus,
        groq_api_key: str = "",
    ):
        super().__init__(config, bus)
        self.config: TelegramConfig = config
        self.groq_api_key = groq_api_key
        self._app: Application | None = None
        self._chat_ids: dict[str, int] = {}  # Map sender_id to chat_id for replies
        self._typing_tasks: dict[str, asyncio.Task] = {}  # chat_id -> typing loop task
        self._media_group_buffers: dict[str, dict] = {}
        self._media_group_tasks: dict[str, asyncio.Task] = {}
        # Latest known live location per chat (updated silently from edited_message
        # live-location updates; surfaced into the next real turn, never its own).
        self._live_store = LiveLocationStore(
            resolve_channel_state_dir(self.name, "live_location")
        )

    async def start(self) -> None:
        """Start the Telegram bot with long polling."""
        if not self.config.token:
            logger.error("Telegram bot token not configured")
            return

        self._running = True

        # Build the application with larger connection pool to avoid pool-timeout on long runs
        req = HTTPXRequest(connection_pool_size=16, pool_timeout=5.0, connect_timeout=30.0, read_timeout=30.0)
        builder = Application.builder().token(self.config.token).request(req).get_updates_request(req)
        if self.config.proxy:
            builder = builder.proxy(self.config.proxy).get_updates_proxy(self.config.proxy)
        self._app = builder.build()
        self._app.add_error_handler(self._on_error)

        # Add command handlers
        self._app.add_handler(CommandHandler("start", self._on_start))
        self._app.add_handler(CommandHandler("new", self._forward_command))
        self._app.add_handler(CommandHandler("help", self._on_help))

        # Add message handler for text, photos, voice, documents, and geo
        # (LOCATION also matches edited_message live-location updates — handled
        # silently in _on_message; VENUE messages also carry a .location).
        self._app.add_handler(
            MessageHandler(
                (
                    filters.TEXT
                    | filters.PHOTO
                    | filters.VOICE
                    | filters.AUDIO
                    | filters.Document.ALL
                    | filters.LOCATION
                    | filters.VENUE
                )
                & ~filters.COMMAND,
                self._on_message
            )
        )

        # Inline-button taps (the `[[ask: …]]` quick-reply keyboard).
        self._app.add_handler(CallbackQueryHandler(self._on_callback))

        logger.info("Starting Telegram bot (polling mode)...")

        # Initialize and start polling
        await self._app.initialize()
        await self._app.start()

        # Get bot info and register command menu
        bot_info = await self._app.bot.get_me()
        logger.info("Telegram bot @%s connected", bot_info.username)

        try:
            await self._app.bot.set_my_commands(self.BOT_COMMANDS)
            logger.debug("Telegram bot commands registered")
        except Exception as e:
            logger.warning("Failed to register bot commands: %s", e)

        # Start polling (this runs until stopped)
        await self._app.updater.start_polling(
            allowed_updates=["message", "callback_query"],
            drop_pending_updates=True  # Ignore old messages on startup
        )

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        self._running = False

        # Cancel all typing indicators
        for chat_id in list(self._typing_tasks):
            self._stop_typing(chat_id)

        for task in self._media_group_tasks.values():
            task.cancel()
        self._media_group_tasks.clear()
        self._media_group_buffers.clear()

        if self._app:
            logger.info("Stopping Telegram bot...")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            self._app = None

    @staticmethod
    def _build_keyboard(buttons: list[str]) -> InlineKeyboardMarkup | None:
        """One vertical inline button per ``[[ask: …]]`` option. The callback
        carries the index; the chosen label is recovered from the keyboard on
        tap (so no option text has to be squeezed into 64-byte callback_data)."""
        options = [b for b in (buttons or []) if b and b.strip()]
        if not options:
            return None
        return InlineKeyboardMarkup(
            [[InlineKeyboardButton(text=opt[:60], callback_data=f"ask:{i}")]
             for i, opt in enumerate(options)]
        )

    @staticmethod
    def _get_media_type(path: str) -> str:
        """Guess media type from file extension."""
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext in ("jpg", "jpeg", "png", "gif", "webp"):
            return "photo"
        if ext in ("mp4", "mov", "m4v", "webm", "avi", "mkv"):
            return "video"
        if ext == "ogg":
            return "voice"
        if ext in ("mp3", "m4a", "wav", "aac"):
            return "audio"
        return "document"

    @classmethod
    def _partition_media(cls, paths: list[str]) -> tuple[dict[str, list[str]], list[str]]:
        buckets = {
            "photo_video": [],
            "document": [],
            "audio": [],
            "voice": [],
        }
        order: list[str] = []
        for path in paths:
            media_type = cls._get_media_type(path)
            if media_type in ("photo", "video"):
                bucket = "photo_video"
            elif media_type == "audio":
                bucket = "audio"
            elif media_type == "voice":
                bucket = "voice"
            else:
                bucket = "document"
            if not buckets[bucket]:
                order.append(bucket)
            buckets[bucket].append(path)
        return buckets, order

    @staticmethod
    def _chunked(paths: list[str], size: int) -> list[list[str]]:
        return [paths[i:i + size] for i in range(0, len(paths), size)]

    @classmethod
    def _build_input_media(cls, bucket: str, media_path: str, media_file):
        if bucket == "photo_video":
            media_type = cls._get_media_type(media_path)
            if media_type == "video":
                return InputMediaVideo(media=media_file)
            return InputMediaPhoto(media=media_file)
        if bucket == "document":
            return InputMediaDocument(media=media_file)
        if bucket == "audio":
            return InputMediaAudio(media=media_file)
        raise ValueError(f"unsupported media group bucket: {bucket}")

    async def _send_single_media(
        self,
        *,
        chat_id: int,
        media_path: str,
        reply_parameters: ReplyParameters | None,
    ) -> ReplyParameters | None:
        try:
            media_type = self._get_media_type(media_path)
            if media_type == "photo":
                sender = self._app.bot.send_photo
            elif media_type == "video":
                sender = self._app.bot.send_video
            elif media_type == "voice":
                sender = self._app.bot.send_voice
            elif media_type == "audio":
                sender = self._app.bot.send_audio
            else:
                sender = self._app.bot.send_document
            param = media_type if media_type in ("photo", "video", "voice", "audio") else "document"
            with open(media_path, "rb") as f:
                await sender(
                    chat_id=chat_id,
                    **{param: f},
                    reply_parameters=reply_parameters,
                )
            return None
        except Exception as e:
            filename = media_path.rsplit("/", 1)[-1]
            logger.error("Failed to send media %s: %s", media_path, e)
            try:
                await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=f"[Failed to send: {filename}]",
                    reply_parameters=reply_parameters,
                )
                return None
            except Exception as fallback_error:
                logger.error("Failed to send media failure notice for %s: %s", media_path, fallback_error)
                return reply_parameters

    async def _send_media_group_batch(
        self,
        *,
        chat_id: int,
        bucket: str,
        media_paths: list[str],
        reply_parameters: ReplyParameters | None,
    ) -> ReplyParameters | None:
        if len(media_paths) == 1:
            return await self._send_single_media(
                chat_id=chat_id,
                media_path=media_paths[0],
                reply_parameters=reply_parameters,
            )
        try:
            with contextlib.ExitStack() as stack:
                media = [
                    self._build_input_media(bucket, path, stack.enter_context(open(path, "rb")))
                    for path in media_paths
                ]
                await self._app.bot.send_media_group(
                    chat_id=chat_id,
                    media=media,
                    reply_parameters=reply_parameters,
                )
            return None
        except Exception as e:
            filenames = ", ".join(path.rsplit("/", 1)[-1] for path in media_paths)
            logger.error("Failed to send media group %s: %s", filenames, e)
            for media_path in media_paths:
                reply_parameters = await self._send_single_media(
                    chat_id=chat_id,
                    media_path=media_path,
                    reply_parameters=reply_parameters,
                )
            return reply_parameters

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Telegram."""
        if not self._app:
            logger.warning("Telegram bot not running")
            return

        # Only stop typing indicator for final responses
        if not msg.metadata.get("_progress", False):
            self._stop_typing(msg.chat_id)

        try:
            chat_id = int(msg.chat_id)
        except ValueError:
            logger.error("Invalid chat_id: %s", msg.chat_id)
            return

        reply_params = None
        if getattr(self.config, "reply_to_message", False):
            reply_to_message_id = msg.metadata.get("message_id")
            if reply_to_message_id:
                reply_params = ReplyParameters(
                    message_id=reply_to_message_id,
                    allow_sending_without_reply=True
                )

        reply_params_for_next_send = reply_params

        # Send media files
        media_paths = list(msg.media or [])
        if media_paths:
            buckets, bucket_order = self._partition_media(media_paths)
            for bucket in bucket_order:
                if bucket == "voice":
                    for media_path in buckets[bucket]:
                        reply_params_for_next_send = await self._send_single_media(
                            chat_id=chat_id,
                            media_path=media_path,
                            reply_parameters=reply_params_for_next_send,
                        )
                    continue
                for batch in self._chunked(buckets[bucket], 10):
                    reply_params_for_next_send = await self._send_media_group_batch(
                        chat_id=chat_id,
                        bucket=bucket,
                        media_paths=batch,
                        reply_parameters=reply_params_for_next_send,
                    )

        # Send text content
        if msg.content and msg.content != "[empty message]":
            is_progress = msg.metadata.get("_progress", False)
            draft_id = msg.metadata.get("message_id")
            keyboard = self._build_keyboard(msg.buttons)  # [[ask: …]] quick-reply buttons
            chunks = split_message(msg.content, TELEGRAM_MAX_MESSAGE_LEN)

            for ci, chunk in enumerate(chunks):
                # Attach the keyboard only to the LAST chunk (buttons sit under
                # the whole message). Progress drafts never carry buttons.
                markup = keyboard if ci == len(chunks) - 1 else None
                try:
                    html = _markdown_to_telegram_html(chunk)
                    if is_progress and draft_id:
                        await self._app.bot.send_message_draft(
                            chat_id=chat_id,
                            draft_id=draft_id,
                            text=html,
                            parse_mode="HTML"
                        )
                    else:
                        await self._app.bot.send_message(
                            chat_id=chat_id,
                            text=html,
                            parse_mode="HTML",
                            reply_parameters=reply_params_for_next_send,
                            reply_markup=markup,
                        )
                        reply_params_for_next_send = None
                except Exception as e:
                    logger.warning("HTML parse failed, falling back to plain text: %s", e)
                    try:
                        if is_progress and draft_id:
                            await self._app.bot.send_message_draft(
                                chat_id=chat_id,
                                draft_id=draft_id,
                                text=chunk
                            )
                        else:
                            await self._app.bot.send_message(
                                chat_id=chat_id,
                                text=chunk,
                                reply_parameters=reply_params_for_next_send,
                                reply_markup=markup,
                            )
                            reply_params_for_next_send = None
                    except Exception as e2:
                        logger.error("Error sending Telegram message: %s", e2)

    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not update.message or not update.effective_user:
            return

        user = update.effective_user
        await update.message.reply_text(
            f"👋 Hi {user.first_name}! I'm nanobot.\n\n"
            "Send me a message and I'll respond!\n"
            "Type /help to see available commands."
        )

    async def _on_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command, bypassing ACL so all users can access it."""
        if not update.message:
            return
        await update.message.reply_text(
            "🐈 nanobot commands:\n"
            "/new — Start a new conversation\n"
            "/stop — Stop the current task\n"
            "/help — Show available commands"
        )

    @staticmethod
    def _sender_id(user) -> str:
        """Build sender_id with username for allowlist matching."""
        sid = str(user.id)
        return f"{sid}|{user.username}" if user.username else sid

    async def _forward_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Forward slash commands to the bus for unified handling in AgentLoop."""
        if not update.message or not update.effective_user:
            return
        await self._handle_message(
            sender_id=self._sender_id(update.effective_user),
            chat_id=str(update.message.chat_id),
            content=update.message.text,
        )

    async def _on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle an inline-button tap from a ``[[ask: …]]`` keyboard: ack it,
        reflect the pick + strip the keyboard, and feed the chosen label back as
        the user's next message (so the agent continues on its next turn). ACL is
        applied downstream, same as a typed message."""
        query = update.callback_query
        if not query:
            return
        try:
            await query.answer()  # stop the button's loading spinner
        except Exception as e:  # noqa: BLE001
            logger.debug("callback answer failed: %s", e)

        data = query.data or ""
        message = query.message
        user = update.effective_user
        if not data.startswith("ask:") or message is None or user is None:
            return
        try:
            idx = int(data.split(":", 1)[1])
        except ValueError:
            return

        # Recover the chosen label from the message's own keyboard.
        option = None
        markup = getattr(message, "reply_markup", None)
        if markup:
            flat = [btn for row in markup.inline_keyboard for btn in row]
            if 0 <= idx < len(flat):
                option = flat[idx].text
        if not option:
            return

        chat_id = message.chat_id
        # Reflect the pick + remove the keyboard so it can't be tapped twice.
        try:
            base = message.text_html if message.text else ""
            picked = option.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            new_text = (base + f"\n\n✅ {picked}").strip() if base else f"✅ {picked}"
            await query.edit_message_text(text=new_text, parse_mode="HTML")
        except Exception as e:  # noqa: BLE001 — best-effort; at least drop the keyboard
            logger.debug("callback edit failed: %s", e)
            try:
                await query.edit_message_reply_markup(reply_markup=None)
            except Exception:  # noqa: BLE001
                pass

        sender_id = self._sender_id(user)
        self._chat_ids[sender_id] = chat_id
        self._start_typing(str(chat_id))
        await self._handle_message(
            sender_id=sender_id,
            chat_id=str(chat_id),
            content=option,
            metadata={
                "message_id": message.message_id,
                "user_id": user.id,
                "username": user.username,
                "first_name": user.first_name,
                "is_group": message.chat.type != "private",
            },
        )

    def _record_live_location(self, message) -> None:
        """Persist a live-location update silently (no agent turn).

        Telegram streams live-location movement as ``edited_message`` updates with
        the same ``message_id``. We stash the latest coordinates per chat so a
        later "where am I / what's nearby" turn can use them, without spawning a
        model turn per edit (which over an hour would be dozens of calls)."""
        loc = getattr(message, "location", None)
        if loc is None:
            return
        expires_at = _live_expires_at(message, loc)
        if expires_at is None:
            # An edit that dropped live_period: treat as the share ending.
            return
        self._live_store.update(
            str(message.chat_id),
            latitude=loc.latitude,
            longitude=loc.longitude,
            expires_at=expires_at,
            heading=getattr(loc, "heading", None),
            horizontal_accuracy=getattr(loc, "horizontal_accuracy", None),
            message_id=getattr(message, "message_id", None),
        )
        logger.info(
            "telegram live-location update chat_id=%s coord=%s (silent, no turn)",
            message.chat_id, _fmt_coord(loc.latitude, loc.longitude),
        )

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages (text, photos, voice, documents, geo)."""
        if not update.effective_user:
            return

        # Edited messages: we only act on live-location movement (recorded
        # silently). Every other edit is ignored, as before.
        if update.message is None:
            edited = update.edited_message
            if edited is not None and getattr(edited, "location", None) is not None:
                self._record_live_location(edited)
            return

        message = update.message
        user = update.effective_user
        chat_id = message.chat_id
        sender_id = self._sender_id(user)

        # Store chat_id for replies
        self._chat_ids[sender_id] = chat_id

        # Build content from text and/or media
        content_parts = []
        media_paths = []

        # Text content
        if message.text:
            content_parts.append(message.text)
        if message.caption:
            content_parts.append(message.caption)

        # Handle media files
        media_file = None
        media_type = None

        if message.photo:
            media_file = message.photo[-1]  # Largest photo
            media_type = "image"
        elif message.voice:
            media_file = message.voice
            media_type = "voice"
        elif message.audio:
            media_file = message.audio
            media_type = "audio"
        elif message.document:
            media_file = message.document
            media_type = "file"

        # Geo: venue (carries its own .location) takes precedence over a bare pin.
        # A live-location *start* (live_period set) is recorded so its subsequent
        # silent edits extend the same share; a static pin is one-shot turn text.
        if message.venue is not None:
            content_parts.append(_format_venue(message.venue))
        elif message.location is not None:
            loc = message.location
            if getattr(loc, "live_period", None):
                self._record_live_location(message)
                content_parts.append(
                    _format_live_started(loc, _live_expires_at(message, loc))
                )
            else:
                content_parts.append(_format_location(loc))

        # Download media if present
        if media_file and self._app:
            file_path = None
            try:
                file = await self._app.bot.get_file(media_file.file_id)
                ext = self._get_extension(media_type, getattr(media_file, 'mime_type', None))

                # Save to workspace/media/ under a collision-free name so a burst
                # of voices does not overwrite each other (see _media_filename).
                from openharness.channels.impl.base import resolve_channel_media_dir
                media_dir = resolve_channel_media_dir(self.name)

                file_path = media_dir / _media_filename(media_file, ext)
                await file.download_to_drive(str(file_path))
                media_paths.append(str(file_path))
            except Exception as e:
                logger.error("Failed to download media: %s", e)
                content_parts.append(f"[{media_type}: download failed]")
                file_path = None

            if file_path is not None:
                # Transcription is best-effort and SEPARATE from the download:
                # a missing local STT must not be reported as "download failed".
                transcription = None
                if media_type in ("voice", "audio"):
                    try:
                        from openharness.providers.transcription import GroqTranscriptionProvider  # noqa: F401
                        transcriber = GroqTranscriptionProvider(api_key=self.groq_api_key)
                        transcription = await transcriber.transcribe(file_path)
                    except Exception as e:
                        logger.info("Local transcription unavailable for %s: %s", media_type, e)
                if transcription:
                    logger.info("Transcribed %s: %s...", media_type, transcription[:50])
                    content_parts.append(f"[transcription: {transcription}]")
                else:
                    # Carry the per-file path so a coalesced burst stays
                    # individually addressable (each voice → its own path).
                    content_parts.append(f"[{media_type}: {file_path}]")
                logger.debug("Downloaded %s to %s", media_type, file_path)

        # If a live-location share is active for this chat, surface the latest
        # known coordinates so a follow-up like "what's nearby?" can be answered.
        if message.venue is None and message.location is None:
            live = self._live_store.get(str(chat_id))
            if live:
                content_parts.append(_format_live_current(live, time.time()))

        content = "\n".join(content_parts) if content_parts else "[empty message]"

        # Surface the replied-to message so a bare follow-up ("а тут?") carries
        # its antecedent into the agent prompt instead of arriving context-free.
        reply_prefix, reply_meta = _reply_context(getattr(message, "reply_to_message", None))
        if reply_prefix:
            content = f"{reply_prefix}\n{content}"

        logger.debug("Telegram message from %s: %s...", sender_id, content[:50])

        str_chat_id = str(chat_id)

        # Telegram media groups: buffer briefly, forward as one aggregated turn.
        if media_group_id := getattr(message, "media_group_id", None):
            key = f"{str_chat_id}:{media_group_id}"
            if key not in self._media_group_buffers:
                self._media_group_buffers[key] = {
                    "sender_id": sender_id, "chat_id": str_chat_id,
                    "contents": [], "media": [],
                    "metadata": {
                        "message_id": message.message_id, "user_id": user.id,
                        "username": user.username, "first_name": user.first_name,
                        "is_group": message.chat.type != "private",
                        **reply_meta,
                    },
                }
                self._start_typing(str_chat_id)
            buf = self._media_group_buffers[key]
            if content and content != "[empty message]":
                buf["contents"].append(content)
            buf["media"].extend(media_paths)
            if key not in self._media_group_tasks:
                self._media_group_tasks[key] = asyncio.create_task(self._flush_media_group(key))
            return

        # Start typing indicator before processing
        self._start_typing(str_chat_id)

        # Forward to the message bus
        await self._handle_message(
            sender_id=sender_id,
            chat_id=str_chat_id,
            content=content,
            media=media_paths,
            metadata={
                "message_id": message.message_id,
                "user_id": user.id,
                "username": user.username,
                "first_name": user.first_name,
                "is_group": message.chat.type != "private",
                **reply_meta,
            }
        )

    async def _flush_media_group(self, key: str) -> None:
        """Wait briefly, then forward buffered media-group as one turn."""
        try:
            await asyncio.sleep(0.6)
            if not (buf := self._media_group_buffers.pop(key, None)):
                return
            content = "\n".join(buf["contents"]) or "[empty message]"
            await self._handle_message(
                sender_id=buf["sender_id"], chat_id=buf["chat_id"],
                content=content, media=list(dict.fromkeys(buf["media"])),
                metadata=buf["metadata"],
            )
        finally:
            self._media_group_tasks.pop(key, None)

    def _start_typing(self, chat_id: str) -> None:
        """Start sending 'typing...' indicator for a chat."""
        # Cancel any existing typing task for this chat
        self._stop_typing(chat_id)
        self._typing_tasks[chat_id] = asyncio.create_task(self._typing_loop(chat_id))

    def _stop_typing(self, chat_id: str) -> None:
        """Stop the typing indicator for a chat."""
        task = self._typing_tasks.pop(chat_id, None)
        if task and not task.done():
            task.cancel()

    async def _typing_loop(self, chat_id: str) -> None:
        """Repeatedly send 'typing' action until cancelled."""
        try:
            while self._app:
                await self._app.bot.send_chat_action(chat_id=int(chat_id), action="typing")
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("Typing indicator stopped for %s: %s", chat_id, e)

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log polling / handler errors instead of silently swallowing them."""
        logger.error("Telegram error: %s", context.error)

    def _get_extension(self, media_type: str, mime_type: str | None) -> str:
        """Get file extension based on media type."""
        if mime_type:
            ext_map = {
                "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
                "audio/ogg": ".ogg", "audio/mpeg": ".mp3", "audio/mp4": ".m4a",
            }
            if mime_type in ext_map:
                return ext_map[mime_type]

        type_map = {"image": ".jpg", "voice": ".ogg", "audio": ".mp3", "file": ""}
        return type_map.get(media_type, "")

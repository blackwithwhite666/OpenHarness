"""Bounded adapter for a person-origin Telegram message entering real Ohmo."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from openharness.channels.bus.events import InboundMessage


def person_source_message(
    *,
    sender_id: str,
    chat_id: str,
    source_message_id: str,
    sent_at: datetime,
    text: str = "",
    photo_path: str | Path | None = None,
) -> InboundMessage:
    """Build the ordinary production inbound type without adding Camera claims.

    ``sent_at`` is the channel-provided message time. This adapter carries it
    unchanged; runtime provenance and any later save decision remain Ohmo's.
    The caller must supply a chat ID known to be private; this classification
    does not grant family access or authorize a nutrition append.
    No reply/thread/callback fields are set, and no nutrition metadata is added.
    """
    if not all(
        value.isdecimal() for value in (sender_id.strip(), chat_id.strip(), source_message_id.strip())
    ):
        raise ValueError("person-origin input requires numeric sender, chat, and source message IDs")
    if sent_at.tzinfo is None or sent_at.utcoffset() is None:
        raise ValueError("person-origin input requires an aware sent time")
    if len(text) > 4000:
        raise ValueError("person-origin text exceeds the 4000-character input bound")
    if not text.strip() and photo_path is None:
        raise ValueError("person-origin input requires text or a photo")
    media: list[str] = []
    if photo_path is not None:
        path = Path(photo_path)
        if not path.is_file() or path.suffix.casefold() not in {".jpg", ".jpeg"}:
            raise ValueError("person-origin photo must be an existing JPEG path")
        if not 4 <= path.stat().st_size <= 10 * 1024 * 1024:
            raise ValueError("person-origin JPEG must be between 4 bytes and 10 MiB")
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(path) as image:
                if image.format != "JPEG":
                    raise ValueError("person-origin photo must be a JPEG")
                image.load()
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError("person-origin photo must be a decodable JPEG") from exc
        media.append(str(path))
    return InboundMessage(
        channel="telegram",
        sender_id=sender_id,
        chat_id=chat_id,
        content=text,
        timestamp=sent_at,
        media=media,
        metadata={
            "message_id": source_message_id,
            "chat_type": "private",
            "is_group": False,
        },
    )


def source_run_status(deliveries: list[dict[str, object]]) -> dict[str, object]:
    """Report a save only when the real runtime returned a durable event ID."""
    ids = [
        item.get("event_id")
        for item in deliveries
        if isinstance(item.get("event_id"), str) and item["event_id"]
    ]
    return {
        "saved": len(ids) == 1,
        "event_id": ids[0] if len(ids) == 1 else None,
        "event_ids": ids,
        "ambiguous": len(ids) > 1,
    }

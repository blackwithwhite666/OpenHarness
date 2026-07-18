"""Trusted per-turn identity derived by the ohmo gateway."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from openharness.channels.bus.events import InboundMessage

_PRIVATE_CHAT_TYPES = frozenset({"p2p", "private", "im", "direct"})
_SHARED_CHAT_TYPES = frozenset({"group", "supergroup", "chat", "channel", "room"})
_FORWARD_METADATA_KEYS = (
    "forward_origin",
    "forward_from",
    "forward_sender_name",
    "forwarded_from",
)


@dataclass(frozen=True)
class TurnContext:
    """Immutable gateway-authenticated identity and chat scope for one turn."""

    principal: str
    is_owner: bool
    is_private: bool
    channel: str
    chat_id: str
    session_id: str


def canonical_principal(channel: str, sender_id: str) -> str:
    """Return the channel's stable user identifier.

    Telegram sender IDs may append a mutable username after ``|``; only the
    immutable numeric prefix is trusted. Other channel adapters already expose
    their stable platform user ID as ``sender_id``, so it is preserved (apart
    from surrounding whitespace).
    """
    sender = str(sender_id).strip()
    if str(channel).strip().lower() == "telegram":
        return sender.split("|", 1)[0].strip()
    return sender


def is_private_message(message: InboundMessage) -> bool:
    """Return whether the channel positively identifies a direct private chat.

    Unknown or absent metadata is deliberately not treated as private. A
    forwarded payload is also not a private turn even when it arrived through
    a direct chat, because its original audience/sender context is different.
    """
    metadata = message.metadata
    if not metadata or _is_forwarded(metadata):
        return False

    chat_type = str(metadata.get("chat_type") or "").strip().lower()
    if metadata.get("is_group") is True or chat_type in _SHARED_CHAT_TYPES:
        return False

    channel = str(message.channel).strip().lower()
    if channel in {"telegram", "whatsapp"}:
        return metadata.get("is_group") is False
    return chat_type in _PRIVATE_CHAT_TYPES


def build_turn_context(
    message: InboundMessage,
    *,
    session_id: str,
    owner_principals: Iterable[str] = (),
) -> TurnContext:
    """Build the immutable identity context for one inbound gateway turn."""
    principal = canonical_principal(message.channel, message.sender_id)
    owners = {str(owner).strip() for owner in owner_principals}
    return TurnContext(
        principal=principal,
        is_owner=principal in owners,
        is_private=is_private_message(message),
        channel=str(message.channel),
        chat_id=str(message.chat_id),
        session_id=str(session_id),
    )


def _is_forwarded(metadata: Mapping[str, object]) -> bool:
    if metadata.get("is_forwarded") is True or metadata.get("forwarded") is True:
        return True
    if str(metadata.get("msg_type") or "").strip().lower() == "merge_forward":
        return True
    return any(metadata.get(key) not in (None, "", False) for key in _FORWARD_METADATA_KEYS)

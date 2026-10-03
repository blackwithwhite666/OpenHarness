"""Resolve a loaded historical attachment to one trusted conversation source."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime

from openharness.channels.bus.events import InboundMessage
from openharness.engine.messages import AttachmentRefBlock, ConversationMessage

from ohmo.gateway.turn_context import TurnContext, canonical_principal, is_private_message


def _event_id_from_provenance(provenance: Mapping[str, object], source_id: str) -> str | None:
    channel = provenance.get("channel")
    chat_id = provenance.get("chat_id")
    principal = provenance.get("principal")
    if not all(isinstance(value, str) and value for value in (channel, chat_id, principal)):
        return None
    prefix = f"{channel.strip().lower()}:"
    if not principal.startswith(prefix):
        return None
    canonical = canonical_principal(channel, principal[len(prefix):])
    seed = "\x00".join((channel.strip().lower(), chat_id, canonical, source_id)).encode("utf-8")
    return f"ohmo-event-{hashlib.sha256(seed).hexdigest()}"


def resolve_selected_photo_source(
    *,
    attachment_id: str,
    history: list[ConversationMessage],
    message: InboundMessage,
    turn_ctx: TurnContext,
    session_key: str,
    gateway_session_id: str,
    tenant_id: str,
    authorized_participant: bool,
) -> dict[str, object] | None:
    """Return gateway-stamped identity only for one verified private source."""
    if (
        not authorized_participant
        or turn_ctx.channel != "telegram"
        or not turn_ctx.is_private
        or turn_ctx.is_forwarded
        or turn_ctx.session_id != gateway_session_id
        or message.channel != "telegram"
        or not is_private_message(message)
        or message.sender_id == "__camera__"
        or str(message.chat_id) != str(turn_ctx.chat_id)
        or message.metadata.get("is_forwarded") is True
        or message.metadata.get("is_group") is True
    ):
        return None
    source_principal = f"telegram:{canonical_principal('telegram', turn_ctx.principal)}"
    matches: dict[tuple[str, str], tuple[Mapping[str, object], str]] = {}
    for historical in history:
        for block in historical.content:
            if not isinstance(block, AttachmentRefBlock) or block.attachment_id != attachment_id:
                continue
            provenance = block.source_provenance
            if not isinstance(provenance, Mapping):
                continue
            if (
                provenance.get("channel") != "telegram"
                or provenance.get("principal") != source_principal
                or provenance.get("chat_id") != str(message.chat_id)
                or provenance.get("session_key") != session_key
                or provenance.get("gateway_session_id") != gateway_session_id
                or provenance.get("is_group") is not False
                or provenance.get("is_forwarded") is not False
                or provenance.get("timestamp_authority") != "inbound_event_timestamp"
            ):
                continue
            source_id = provenance.get("source_message_id")
            received_at = provenance.get("received_at")
            if not isinstance(source_id, str) or not source_id.strip() or not isinstance(received_at, str):
                continue
            try:
                source_time = datetime.fromisoformat(received_at.replace("Z", "+00:00"))
            except ValueError:
                continue
            if source_time.tzinfo is None or source_time.utcoffset() is None:
                continue
            consumed_occurrences = provenance.get("consumed_occurrences")
            if isinstance(consumed_occurrences, list) and consumed_occurrences:
                for occurrence in consumed_occurrences:
                    if not isinstance(occurrence, Mapping):
                        continue
                    receipt_id = occurrence.get("receipt_event_id")
                    client_op_id = occurrence.get("client_op_id")
                    consumed_append_id = occurrence.get("append_source_message_id")
                    if (
                        isinstance(receipt_id, str) and receipt_id.strip()
                        and isinstance(client_op_id, str) and client_op_id.strip()
                        and isinstance(consumed_append_id, str) and consumed_append_id.strip()
                    ):
                        matches[(source_id.strip(), consumed_append_id.strip())] = (
                            provenance, received_at
                        )
                continue
            append_id = provenance.get("append_source_message_id")
            if not isinstance(append_id, str) or not append_id.strip():
                if historical.event_id != _event_id_from_provenance(provenance, source_id):
                    continue
                append_id = source_id
            matches[(source_id.strip(), append_id.strip())] = (provenance, received_at)
    if len(matches) != 1:
        return None
    (source_id, append_id), (provenance, received_at) = next(iter(matches.items()))
    return {
        "schema_version": 1,
        "tenant_id": tenant_id,
        "source_principal": source_principal,
        "gateway_session_id": gateway_session_id,
        "source_message_id": source_id,
        "append_source_message_id": append_id,
        "is_private": True,
        "is_forwarded": False,
        "is_group": False,
        "attachment_id": attachment_id,
        "chat_id": str(message.chat_id),
        "session_key": session_key,
        "received_at": received_at,
    }

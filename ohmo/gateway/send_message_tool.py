"""Model-callable Telegram send tool for ohmo gateway."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from openharness.channels.bus.events import OutboundMessage
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

from ohmo.contact_registry import ContactRecord, ContactStore


class SendTelegramMessageInput(BaseModel):
    recipient: str = Field(
        description=(
            "Who to message: a Telegram @username, a known contact's name, or a numeric chat_id. "
            "MUST be a known contact (someone who has already written to the bot)."
        )
    )
    text: str = Field(description="The message text to send (plain text / light markdown).")


class SendTelegramMessageTool(BaseTool):
    name = "send_telegram_message"
    description = (
        "Send a Telegram message to another person on the user's behalf, relayed by "
        "the bot. Anyone allowed to chat with the bot may use it, but it can only "
        "reach known contacts: people who have already written to ohmo; it refuses "
        "unknown recipients. The message is automatically signed with the sender's "
        "name, so do not add your own signature or \"via bot\" note. Confirm the "
        "recipient and exact text with the user before sending unless the user "
        "already gave both explicitly; this is an outward-facing action, so never "
        "send half-baked or speculative messages and never invent a chat_id. "
        "`recipient` accepts an @username, a known contact's name, or a numeric "
        "chat_id; `text` is the body. On success the message is queued to Telegram; "
        "a delivery failure (e.g. the recipient blocked the bot) is logged "
        "server-side, not returned here."
    )
    input_model = SendTelegramMessageInput

    def __init__(
        self,
        contact_store: ContactStore,
        send_outbound: Callable[[OutboundMessage], Awaitable[None]],
    ) -> None:
        self._contact_store = contact_store
        self._send_outbound = send_outbound

    def is_read_only(self, arguments: BaseModel) -> bool:
        del arguments
        return False

    async def execute(
        self, arguments: SendTelegramMessageInput, context: ToolExecutionContext
    ) -> ToolResult:
        send_ctx = context.metadata.get("ohmo_send_ctx")
        if not isinstance(send_ctx, dict):
            return ToolResult(
                output="No sender context; this tool can only be used from a live chat.",
                is_error=True,
            )
        sender_label = _sender_label(send_ctx)
        if sender_label is None:
            return ToolResult(
                output=(
                    "Refusing to send: this turn has no identifiable human sender to sign "
                    "the message (e.g. a background or reminder turn). Messages can only be "
                    "sent from a live human chat."
                ),
                is_error=True,
            )
        if not arguments.text.strip():
            return ToolResult(
                output="Refusing to send an empty message.",
                is_error=True,
            )
        # Trusted fixed scheduled recipient (surfaced by the runtime only for a
        # scheduler-stamped recipient-bound reminder turn). When present, the
        # delivery target is pinned: any resolved recipient whose chat_id
        # differs is rejected — the model cannot widen the audience.
        fixed_chat_id = str(send_ctx.get("fixed_recipient_chat_id") or "").strip()
        fixed_principal = str(send_ctx.get("fixed_recipient_principal") or "").strip()
        fixed_label = str(send_ctx.get("fixed_recipient_label") or "").strip()
        reminder_id = str(send_ctx.get("reminder_id") or "").strip()

        fixed_contact = None
        if fixed_chat_id and fixed_label and arguments.recipient == fixed_label:
            fixed_contact = self._contact_store.get("telegram", fixed_chat_id)
            if fixed_contact is None:
                return ToolResult(
                    output=(
                        "Refusing to send: the fixed reminder recipient is no longer "
                        "a known Telegram contact."
                    ),
                    is_error=True,
                )
        matches = (
            [fixed_contact]
            if fixed_contact is not None
            else self._contact_store.resolve(arguments.recipient, channel="telegram")
        )
        if len(matches) == 1:
            contact = matches[0]
            fixed_contact_error = _fixed_contact_validation_error(
                contact,
                fixed_chat_id=fixed_chat_id,
                fixed_principal=fixed_principal,
                fixed_label=fixed_label,
            )
            if fixed_contact_error is not None:
                return fixed_contact_error
            metadata = {
                "_session_key": f"telegram:{contact.chat_id}",
                "_origin": "send_telegram_message",
            }
            if reminder_id:
                # Lets the channel dispatcher pause the reminder when the actual
                # Telegram delivery fails (e.g. the recipient blocked the bot).
                metadata["_reminder_id"] = reminder_id
            await self._send_outbound(
                OutboundMessage(
                    channel="telegram",
                    chat_id=contact.chat_id,
                    content=_signed_text(arguments.text, sender_label),
                    metadata=metadata,
                )
            )
            return ToolResult(
                output=(
                    f"Message queued for delivery to {_success_name(contact)} "
                    f"(chat_id={contact.chat_id})."
                ),
                metadata={"recipient_chat_id": contact.chat_id},
            )
        if len(matches) > 1:
            return ToolResult(
                output=_ambiguous_recipient_output(arguments.recipient, matches),
                is_error=True,
            )

        suggestions = self._contact_store.resolve(
            arguments.recipient,
            channel="telegram",
            fuzzy=True,
        )
        return ToolResult(
            output=_unknown_recipient_output(
                arguments.recipient,
                self._contact_store.list(),
                suggestions,
            ),
            is_error=True,
        )


def _unknown_recipient_output(
    query: str,
    contacts: list[ContactRecord],
    suggestions: list[ContactRecord],
) -> str:
    telegram_contacts = [contact for contact in contacts if contact.channel == "telegram"][:10]
    lines = [
        (
            f"Unknown recipient {query!r}. The bot can only message known Telegram "
            "contacts: people who have already written to ohmo."
        )
    ]
    if suggestions:
        lines.append("Did you mean:")
        lines.extend(f"- {_format_contact(contact)}" for contact in suggestions[:10])
        return "\n".join(lines)
    if telegram_contacts:
        lines.append("Known Telegram contacts:")
        lines.extend(f"- {_format_contact(contact)}" for contact in telegram_contacts)
    else:
        lines.append("No known Telegram contacts yet.")
    return "\n".join(lines)


def _ambiguous_recipient_output(query: str, matches: list[ContactRecord]) -> str:
    lines = [f"Ambiguous recipient {query!r}; matches:"]
    lines.extend(f"- {_format_contact(contact)}" for contact in matches)
    lines.append("Please specify which contact to message.")
    return "\n".join(lines)


def _format_contact(contact: ContactRecord) -> str:
    parts: list[str] = []
    if contact.username:
        parts.append(f"@{contact.username}")
    if contact.first_name:
        parts.append(contact.first_name)
    elif contact.display_name:
        parts.append(contact.display_name)
    parts.append(f"chat_id={contact.chat_id}")
    return " · ".join(parts)


def _contact_label(contact: ContactRecord) -> str:
    """Return the stable label used when a reminder binds a contact."""
    parts: list[str] = []
    if contact.first_name:
        parts.append(contact.first_name)
    elif contact.display_name:
        parts.append(contact.display_name)
    if contact.username:
        parts.append(f"@{contact.username}")
    return " ".join(parts) or contact.chat_id


def _contact_principal(contact: ContactRecord) -> str:
    return (contact.user_id or contact.chat_id).strip()


def _fixed_contact_validation_error(
    contact: ContactRecord,
    *,
    fixed_chat_id: str,
    fixed_principal: str,
    fixed_label: str,
) -> ToolResult | None:
    """Validate the current contact identity for a scheduled fixed target.

    The model may address the contact through any supported alias, so this
    check must run after resolution as well as on the exact stored label. A
    changed contact record must not become sendable merely because an alias
    still resolves to the pinned chat id.
    """
    if fixed_chat_id and str(contact.chat_id) != fixed_chat_id:
        return ToolResult(
            output=(
                "Refusing to send: this scheduled reminder is bound to "
                f"{fixed_label or 'a fixed recipient'} "
                f"(chat_id={fixed_chat_id}) and cannot message anyone else."
            ),
            is_error=True,
        )
    if fixed_label and _contact_label(contact) != fixed_label:
        return ToolResult(
            output=(
                "Refusing to send: the fixed reminder recipient label no "
                "longer matches the pinned Telegram contact."
            ),
            is_error=True,
        )
    if fixed_principal and _contact_principal(contact) != fixed_principal:
        return ToolResult(
            output=(
                "Refusing to send: the fixed reminder recipient principal "
                "no longer matches the pinned Telegram contact."
            ),
            is_error=True,
        )
    return None


def _success_name(contact: ContactRecord) -> str:
    if contact.first_name:
        return contact.first_name
    if contact.username:
        return f"@{contact.username}"
    return contact.chat_id


def _sender_label(send_ctx: dict) -> str | None:
    """Human-readable label for whoever asked the bot to send the message,
    or None when the turn has no identifiable human sender (e.g. a background
    or reminder turn whose sender_id is the scheduler sentinel).
    """
    name = str(send_ctx.get("first_name") or "").strip() or str(
        send_ctx.get("display_name") or ""
    ).strip()
    username = str(send_ctx.get("username") or "").strip().lstrip("@")
    if name and username:
        return f"{name} (@{username})"
    if username:
        return f"@{username}"
    if name:
        return name
    sid = str(send_ctx.get("sender_id") or "").strip()
    if "|" in sid:  # Telegram sender_id is "<id>|<username>"
        uname = sid.partition("|")[2].strip()
        if uname:
            return f"@{uname}"
    return None


def _signed_text(text: str, sender_label: str) -> str:
    return f"{text}\n\n— {sender_label} (отправлено через бота)"

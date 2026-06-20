"""Inbound reply-context extraction for the Telegram channel."""

from types import SimpleNamespace

from openharness.channels.impl.telegram import _REPLY_QUOTE_MAX, _reply_context


def _reply(*, text=None, caption=None, from_user=None, message_id=42, **media):
    return SimpleNamespace(
        text=text,
        caption=caption,
        from_user=from_user,
        message_id=message_id,
        photo=media.get("photo"),
        voice=media.get("voice"),
        audio=media.get("audio"),
        document=media.get("document"),
        sticker=media.get("sticker"),
        video=media.get("video"),
    )


def test_no_reply_returns_empty():
    assert _reply_context(None) == ("", {})


def test_text_reply_inlines_quote_and_author():
    user = SimpleNamespace(is_bot=False, first_name="Дарья", username="dshatko")
    prefix, meta = _reply_context(_reply(text="поставь 1-1 с Дарьей", from_user=user))
    assert prefix == '[In reply to Дарья: "поставь 1-1 с Дарьей"]'
    assert meta["reply_to_message_id"] == 42
    assert meta["reply_to_text"] == "поставь 1-1 с Дарьей"


def test_reply_to_bot_message_labels_author_as_bot():
    bot = SimpleNamespace(is_bot=True, first_name="ohmo", username="SkunkLipinBot")
    prefix, _ = _reply_context(_reply(text="Поставил в очередь 1-1", from_user=bot))
    assert prefix.startswith('[In reply to you (the bot): "')


def test_media_reply_without_text_uses_media_label():
    user = SimpleNamespace(is_bot=False, first_name="Дима", username=None)
    prefix, meta = _reply_context(_reply(from_user=user, photo=[object()]))
    assert prefix == '[In reply to Дима: "[photo]"]'
    assert meta["reply_to_text"] == "[photo]"


def test_long_quote_is_truncated():
    user = SimpleNamespace(is_bot=False, first_name="X", username=None)
    prefix, meta = _reply_context(_reply(text="a" * (_REPLY_QUOTE_MAX + 50), from_user=user))
    assert meta["reply_to_text"].endswith("…")
    assert len(meta["reply_to_text"]) == _REPLY_QUOTE_MAX + 1  # cap + ellipsis
    assert "…" in prefix

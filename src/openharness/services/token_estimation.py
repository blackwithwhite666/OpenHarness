"""Simple token estimation utilities."""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Estimate tokens from plain text using a UTF-8 *byte* heuristic.

    The classic ~4-chars-per-token rule of thumb holds for ASCII English, but
    badly under-counts denser scripts: Cyrillic tokenizes at roughly 2 chars
    per token, CJK at ~1. Counting characters (``len(text) // 4``) therefore
    under-estimates a Russian conversation ~2x — which let sessions blow past
    the context window before auto-compaction ever fired.

    Counting UTF-8 *bytes* instead self-corrects: ASCII is 1 byte/char (so
    English is unchanged at ~4 chars/token), Cyrillic is 2 bytes/char (→ ~2
    chars/token), CJK ~3. This errs slightly toward over-estimation, which is
    the safe direction for a context-window guard.
    """
    if not text:
        return 0
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


def estimate_message_tokens(messages: list[str]) -> int:
    """Estimate tokens for a collection of message strings."""
    return sum(estimate_tokens(message) for message in messages)

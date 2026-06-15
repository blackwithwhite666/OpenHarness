"""Session-end hook to extract and persist local environment rules."""

from __future__ import annotations

import logging
import os

from openharness.engine.messages import ConversationMessage
from openharness.personalization.extractor import (
    extract_facts_from_text,
    facts_to_rules_markdown,
)
from openharness.personalization.rules import (
    load_facts,
    merge_facts,
    save_facts,
    save_local_rules,
)

log = logging.getLogger(__name__)


def _personalization_enabled() -> bool:
    """Opt-in flag for the legacy regex local-rules harvester (default OFF).

    The regex extractor accumulated junk — every tool-artifact path, unvalidated
    "IP"-like strings, any 5 numbers as a "cron" — into an unbounded
    ``~/.openharness/local_rules/rules.md`` that grew to ~97k tokens and
    dominated the system prompt. It also duplicates ohmo's curated memory
    (soul.md / user.md / ~/.ohmo/memory + the memory tool). Disabled by default;
    set ``OPENHARNESS_PERSONALIZATION=1`` to re-enable.
    """
    return os.environ.get("OPENHARNESS_PERSONALIZATION", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def update_rules_from_session(messages: list[ConversationMessage]) -> int:
    """Extract local facts from session messages and update rules.

    Called at session end. Returns the number of new facts extracted (0 when the
    harvester is disabled, which is the default — see ``_personalization_enabled``).

    Args:
        messages: The conversation messages from the session.

    Returns:
        Number of new facts found and persisted.
    """
    if not _personalization_enabled():
        return 0

    # Collect all text from messages
    all_text = []
    for msg in messages:
        for block in msg.content:
            text = getattr(block, "text", None) or getattr(block, "content", None) or ""
            if isinstance(text, str) and text:
                all_text.append(text)

    if not all_text:
        return 0

    combined = "\n".join(all_text)
    new_facts = extract_facts_from_text(combined)
    if not new_facts:
        return 0

    # Merge with existing
    existing = load_facts()
    merged = merge_facts(existing, new_facts)
    save_facts(merged)

    # Regenerate rules markdown
    rules_md = facts_to_rules_markdown(merged["facts"])
    if rules_md:
        save_local_rules(rules_md)

    new_count = len(merged["facts"]) - len(existing.get("facts", []))
    log.info(
        "Personalization: %d new facts extracted (%d total)",
        max(new_count, 0),
        len(merged["facts"]),
    )
    return max(new_count, 0)

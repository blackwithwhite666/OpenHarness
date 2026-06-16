"""Background self-improvement memory judge (P2).

After a turn (on a cadence), an off-hot-path pass asks the model whether anything
durable from the conversation should be saved/updated/removed, and applies the
result through :class:`ohmo.memory_store.MemoryStore` — so every write still goes
through P0 discipline (unicode slug, dedup, char bounds) and P1 safety (threat
scan). Mirrors Hermes' ``_spawn_background_review`` but as a single ``stream_message``
call that emits a structured op-list (no forked agent / tool loop), Mem0-shaped
ADD / UPDATE. Autonomous REMOVE is intentionally excluded — deletion stays a
human / foreground decision (the Cursor auto-memory cautionary tale).

Enabled by default; set ``OHMO_MEMORY_JUDGE=0`` (or false/no/off) to disable.
Tune cadence with ``OHMO_MEMORY_JUDGE_INTERVAL`` (turns). Precision rests on the
conservative prompt, add/update-only ops (no autonomous delete), and the fact
that every write still passes the disciplined store's dedup/bounds/safety gates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field

from openharness.api.client import ApiMessageCompleteEvent, ApiMessageRequest
from openharness.engine.messages import ConversationMessage

from ohmo.memory_store import MemoryStore

log = logging.getLogger(__name__)

# Run every N user turns *within one session*. Hermes uses 10, but ohmo is a
# Telegram bot whose sessions are short bursts that rarely reach 10 turns (and a
# gateway restart / `/new` resets the per-session counter), so 10 meant the judge
# almost never fired. 3 makes it fire within a realistic chat. Off-hot-path, so
# the extra LLM calls don't affect latency. Override with OHMO_MEMORY_JUDGE_INTERVAL.
DEFAULT_JUDGE_INTERVAL = 3
_MAX_OPS = 5  # cap ops applied per run
_MAX_TRANSCRIPT_CHARS = 8000  # trim the conversation fed to the judge
_MAX_MEMORY_CTX_CHARS = 4000  # trim the current-memory context fed to the judge
_JUDGE_MAX_TOKENS = 800
_DEFAULT_TIMEOUT = 30.0
_FALSE = {"0", "false", "no", "off"}

JUDGE_SYSTEM_PROMPT = (
    "You curate an AI assistant's long-term memory about its owner. Review the recent "
    "conversation and the CURRENT MEMORY, then decide if anything DURABLE should change.\n\n"
    "SAVE (add) only lasting, declarative facts: stable user preferences & communication "
    "style, environment facts, project/workflow conventions, corrections & workarounds, "
    "stable identities of people/services. UPDATE an existing entry when the conversation "
    "revises it (use update to merge or shrink an overlapping entry). Do NOT delete entries "
    "— removal is a human decision.\n\n"
    "DO NOT save: transient progress ('fixed X today', run logs), raw data dumps "
    "(file/artifact paths, listings), web-searchable trivia, secrets/tokens, one-off task "
    "narratives, or anything already in CURRENT MEMORY (avoid duplicates). Write DECLARATIVE "
    "facts ('User prefers UTC'), not self-instructions ('always use UTC'). Be conservative — "
    "most turns need NO change.\n\n"
    "Respond with ONLY a JSON object, no prose, no markdown fences:\n"
    '{"ops": [{"action": "add", "title": "...", "content": "..."}, '
    '{"action": "update", "name": "<entry-name>", "content": "...", "title": "..."}], '
    '"reason": "<short>"}\n'
    'If nothing is worth changing, respond exactly: {"ops": [], "reason": "nothing to save"}'
)


@dataclass
class JudgeOutcome:
    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    reason: str = ""
    raw: str = ""


def judge_enabled() -> bool:
    """On by default; set OHMO_MEMORY_JUDGE=0/false/no/off to disable."""
    return os.environ.get("OHMO_MEMORY_JUDGE", "").strip().lower() not in _FALSE


def judge_interval() -> int:
    raw = os.environ.get("OHMO_MEMORY_JUDGE_INTERVAL")
    if raw is None:
        return DEFAULT_JUDGE_INTERVAL
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_JUDGE_INTERVAL
    return value if value > 0 else DEFAULT_JUDGE_INTERVAL


def _render_transcript(messages: list[ConversationMessage]) -> str:
    """Recent conversation as text, newest-biased, trimmed to a char budget."""
    rendered: list[str] = []
    total = 0
    for msg in reversed(messages):
        role = getattr(msg, "role", "?")
        text = (getattr(msg, "text", "") or "").strip()
        if not text:
            continue
        chunk = f"{role}: {text[:1000]}"
        if total + len(chunk) > _MAX_TRANSCRIPT_CHARS:
            break
        rendered.append(chunk)
        total += len(chunk)
    return "\n\n".join(reversed(rendered))


def _render_current_memory(store: MemoryStore) -> str:
    entries = store.list()
    if not entries:
        return "(memory is empty)"
    lines: list[str] = []
    total = 0
    for e in entries:
        line = f"- {e.name} | {e.title}: {e.content[:200]}"
        if total + len(line) > _MAX_MEMORY_CTX_CHARS:
            lines.append(f"… (+{len(entries) - len(lines)} more entries)")
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def parse_judge_ops(text: str) -> tuple[list[dict], str]:
    """Tolerantly parse the judge's JSON. Returns (ops, reason); ([], reason) on junk."""
    if not text:
        return [], ""
    cleaned = text.strip()
    # strip ```json … ``` fences if present
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace:
            cleaned = brace.group(0)
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError):
        return [], ""
    if not isinstance(data, dict):
        return [], ""
    ops = data.get("ops")
    reason = str(data.get("reason", "") or "")
    if not isinstance(ops, list):
        return [], reason
    return [op for op in ops if isinstance(op, dict)], reason


def apply_judge_ops(store: MemoryStore, ops: list[dict], *, max_ops: int = _MAX_OPS) -> JudgeOutcome:
    """Apply ops through the disciplined store (each write is dedup/bounds/scan gated)."""
    outcome = JudgeOutcome()
    for op in ops[:max_ops]:
        action = str(op.get("action", "")).strip().lower()
        try:
            if action == "add":
                r = store.add(str(op.get("title", "")), str(op.get("content", "")))
            elif action == "update":
                r = store.update(
                    str(op.get("name", "")),
                    str(op.get("content", "")),
                    title=(str(op["title"]) if op.get("title") else None),
                )
            elif action == "remove":
                # Autonomous deletion is intentionally NOT applied — removal stays a
                # human / foreground (model tool, /memory) decision. The judge can only
                # add or update; this avoids the auto-memory loop silently shrinking the
                # corpus (the Cursor cautionary tale).
                outcome.skipped.append(f"remove {op.get('name', '?')}: not auto-applied (human-only)")
                continue
            else:
                outcome.skipped.append(f"unknown action {action!r}")
                continue
        except Exception as exc:  # defensive — never let one op break the run
            outcome.skipped.append(f"{action} error: {exc}")
            continue
        label = op.get("title") or op.get("name") or "?"
        (outcome.applied if r.ok else outcome.skipped).append(f"{action} {label}: {r.message}")
    return outcome


async def _complete(api_client, model: str, system: str, user: str, *, timeout: float) -> str:
    request = ApiMessageRequest(
        model=model,
        messages=[ConversationMessage.from_user_text(user)],
        system_prompt=system,
        max_tokens=_JUDGE_MAX_TOKENS,
        tools=[],
    )

    async def _collect() -> str:
        text = ""
        async for event in api_client.stream_message(request):
            if isinstance(event, ApiMessageCompleteEvent):
                text = event.message.text
        return text

    return (await asyncio.wait_for(_collect(), timeout=timeout)).strip()


async def run_memory_judge(
    *,
    api_client,
    model: str,
    messages: list[ConversationMessage],
    store: MemoryStore,
    timeout: float = _DEFAULT_TIMEOUT,
    max_ops: int = _MAX_OPS,
) -> JudgeOutcome:
    """One-shot judge: review transcript + current memory, apply proposed ops.

    Best-effort: any failure returns an empty outcome rather than raising.
    """
    transcript = _render_transcript(messages)
    if not transcript:
        return JudgeOutcome(reason="empty transcript")
    user_prompt = (
        f"# Recent conversation\n{transcript}\n\n"
        f"# Current memory\n{_render_current_memory(store)}\n\n"
        "Decide the memory ops per your instructions. JSON only."
    )
    try:
        raw = await _complete(api_client, model, JUDGE_SYSTEM_PROMPT, user_prompt, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — best-effort (incl. asyncio.TimeoutError)
        log.warning("memory judge call failed: %s", exc)
        return JudgeOutcome(reason=f"call failed: {exc}")
    ops, reason = parse_judge_ops(raw)
    outcome = apply_judge_ops(store, ops, max_ops=max_ops)
    outcome.reason = reason
    outcome.raw = raw
    return outcome

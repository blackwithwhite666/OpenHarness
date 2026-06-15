"""Threat-pattern library for scanning content that enters the system prompt.

Ported from Hermes' ``tools/threat_patterns.py`` (single source of truth for
prompt-injection / promptware / exfiltration patterns). Used by the ohmo memory
store to scan entries write-time (refuse) and snapshot-time (replace with a
placeholder), so a poisoned memory entry — typed, or written on disk by a
compromised tool / sister session — cannot inject into the always-on prompt.

Pattern philosophy: organized by ATTACK CLASS, each ``(regex, pattern_id, scope)``.
Scope is cumulative: ``all`` ⊂ ``context`` ⊂ ``strict``.

- ``"all"``     — classic prompt injection + exfiltration (minimal false positives)
- ``"context"`` — + promptware / C2 / role-play (broader)
- ``"strict"``  — + persistence / SSH backdoor / exfil-URL / config tamper /
  hardcoded secrets. Memory writes + snapshots use ``"strict"``.

Patterns use ``(?:\\w+\\s+)*`` between key tokens to defeat filler-word bypass,
and anchor on attack-specific vocabulary rather than bossy English (which is too
common in legitimate instruction files).
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# Each entry: (regex, pattern_id, scope). scope ∈ {"all", "context", "strict"}
_PATTERNS: List[Tuple[str, str, str]] = [
    # ── Classic prompt injection (applies everywhere) ────────────────
    (r'ignore\s+(?:\w+\s+)*(previous|all|above|prior)\s+(?:\w+\s+)*instructions', "prompt_injection", "all"),
    (r'system\s+prompt\s+override', "sys_prompt_override", "all"),
    (r'disregard\s+(?:\w+\s+)*(your|all|any)\s+(?:\w+\s+)*(instructions|rules|guidelines)', "disregard_rules", "all"),
    (r'act\s+as\s+(if|though)\s+(?:\w+\s+)*you\s+(?:\w+\s+)*(have\s+no|don\'t\s+have)\s+(?:\w+\s+)*(restrictions|limits|rules)', "bypass_restrictions", "all"),
    (r'<!--[^>]*(?:ignore|override|system|secret|hidden)[^>]*-->', "html_comment_injection", "all"),
    (r'<\s*div\s+style\s*=\s*["\'][\s\S]*?display\s*:\s*none', "hidden_div", "all"),
    (r'translate\s+.*\s+into\s+.*\s+and\s+(execute|run|eval)', "translate_execute", "all"),
    (r'do\s+not\s+(?:\w+\s+)*tell\s+(?:\w+\s+)*the\s+user', "deception_hide", "all"),

    # ── Role-play / identity hijack (context) ────────────────────────
    (r'you\s+are\s+(?:\w+\s+)*now\s+(?:a|an|the)\s+', "role_hijack", "context"),
    (r'pretend\s+(?:\w+\s+)*(you\s+are|to\s+be)\s+', "role_pretend", "context"),
    (r'output\s+(?:\w+\s+)*(system|initial)\s+prompt', "leak_system_prompt", "context"),
    (r'(respond|answer|reply)\s+without\s+(?:\w+\s+)*(restrictions|limitations|filters|safety)', "remove_filters", "context"),
    (r'you\s+have\s+been\s+(?:\w+\s+)*(updated|upgraded|patched)\s+to', "fake_update", "context"),
    (r'\bname\s+yourself\s+\w+', "identity_override", "context"),

    # ── C2 / Brainworm-style promptware (context) ────────────────────
    (r'register\s+(as\s+)?a?\s*node', "c2_node_registration", "context"),
    (r'(heartbeat|beacon|check[\s\-]?in)\s+(to|with)\s+', "c2_heartbeat", "context"),
    (r'pull\s+(down\s+)?(?:new\s+)?task(?:ing|s)?\b', "c2_task_pull", "context"),
    (r'connect\s+to\s+the\s+network\b', "c2_network_connect", "context"),
    (r'you\s+must\s+(?:\w+\s+){0,3}(register|connect|report|beacon)\b', "forced_action", "context"),
    (r'only\s+use\s+one[\s\-]?liners?\b', "anti_forensic_oneliner", "context"),
    (r'never\s+(?:\w+\s+)*(?:create|write)\s+(?:\w+\s+)*(?:script|file)\s+(?:\w+\s+)*disk', "anti_forensic_disk", "context"),
    (r'unset\s+\w*(?:CLAUDE|CODEX|HERMES|AGENT|OPENAI|ANTHROPIC)\w*', "env_var_unset_agent", "context"),

    # ── Known C2 / red-team framework names (context) ────────────────
    (r'\b(?:praxis|cobalt\s*strike|sliver|havoc|mythic|metasploit|brainworm)\b', "known_c2_framework", "context"),
    (r'\bc2\s+(?:server|channel|infrastructure|beacon)\b', "c2_explicit", "context"),
    (r'\bcommand\s+and\s+control\b', "c2_explicit_long", "context"),

    # ── Exfiltration via curl/wget/cat with secrets (applies everywhere) ──
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl", "all"),
    (r'wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_wget", "all"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets", "all"),
    (r'(send|post|upload|transmit)\s+.*\s+(to|at)\s+https?://', "send_to_url", "strict"),
    (r'(include|output|print|share)\s+(?:\w+\s+)*(conversation|chat\s+history|previous\s+messages|full\s+context|entire\s+context)', "context_exfil", "strict"),

    # ── Persistence / SSH backdoor (strict) ──────────────────────────
    (r'authorized_keys', "ssh_backdoor", "strict"),
    (r'\$HOME/\.ssh|\~/\.ssh', "ssh_access", "strict"),
    # ohmo-specific secret files (adapted from Hermes' hermes_env): reading the
    # gateway/Codex/worfalomey credential stores is exfiltration behavior.
    (r'\.codex/auth\.json|\.worfalomey_oauth\.json|\.ohmo/gateway\.json', "ohmo_secret_files", "strict"),
    # Tampering with agent instruction / persona / memory files.
    (r'(update|modify|edit|write|change|append|add\s+to)\s+.*(?:AGENTS\.md|CLAUDE\.md|\.cursorrules|\.clinerules|soul\.md|user\.md|MEMORY\.md)', "agent_config_mod", "strict"),
    (r'(update|modify|edit|write|change|append|add\s+to)\s+.*\.ohmo/(gateway\.json|soul\.md|user\.md|BOOTSTRAP\.md)', "ohmo_config_mod", "strict"),

    # ── Hardcoded secrets ────────────────────────────────────────────
    (r'(?:api[_-]?key|token|secret|password)\s*[=:]\s*["\'][A-Za-z0-9+/=_-]{20,}', "hardcoded_secret", "strict"),
]

# Invisible / bidirectional unicode used in injection attacks: zero-width chars,
# BOM, bidi embeds/overrides, directional isolates, invisible math operators.
# Built from explicit code points (no literal invisible chars in source).
INVISIBLE_CHARS = frozenset(
    chr(cp)
    for cp in (
        0x200B, 0x200C, 0x200D,  # zero-width space / non-joiner / joiner
        0x2060,                  # word joiner
        0x2062, 0x2063, 0x2064,  # invisible times / separator / plus
        0xFEFF,                  # zero-width no-break space (BOM)
        0x202A, 0x202B, 0x202C, 0x202D, 0x202E,  # bidi embeds / overrides
        0x2066, 0x2067, 0x2068, 0x2069,          # directional isolates
    )
)

_COMPILED: dict[str, List[Tuple[re.Pattern, str]]] = {}


def _compile() -> None:
    """Compile pattern sets per scope. Cumulative: all ⊂ context ⊂ strict."""
    global _COMPILED
    if _COMPILED:
        return
    all_p: List[Tuple[re.Pattern, str]] = []
    context_p: List[Tuple[re.Pattern, str]] = []
    strict_p: List[Tuple[re.Pattern, str]] = []
    for pattern, pid, scope in _PATTERNS:
        entry = (re.compile(pattern, re.IGNORECASE), pid)
        if scope == "all":
            all_p.append(entry); context_p.append(entry); strict_p.append(entry)
        elif scope == "context":
            context_p.append(entry); strict_p.append(entry)
        elif scope == "strict":
            strict_p.append(entry)
        else:
            raise ValueError(f"threat_patterns: unknown scope {scope!r} for {pid!r}")
    _COMPILED = {"all": all_p, "context": context_p, "strict": strict_p}


_compile()


def scan_for_threats(content: str, scope: str = "strict") -> List[str]:
    """Return matched pattern IDs (+ ``invisible_unicode_U+XXXX``) in ``content``.

    Empty content -> []. ``scope`` selects the cumulative pattern set
    (all/context/strict). Memory uses ``strict``.
    """
    if not content:
        return []
    findings: List[str] = []
    for ch in set(content) & INVISIBLE_CHARS:
        findings.append(f"invisible_unicode_U+{ord(ch):04X}")
    patterns = _COMPILED.get(scope)
    if patterns is None:
        raise ValueError(f"scan_for_threats: unknown scope {scope!r}")
    for compiled, pid in patterns:
        if compiled.search(content):
            findings.append(pid)
    return findings


def first_threat_message(content: str, scope: str = "strict") -> Optional[str]:
    """Human-readable block message for the first finding, or None if clean."""
    findings = scan_for_threats(content, scope=scope)
    if not findings:
        return None
    pid = findings[0]
    if pid.startswith("invisible_unicode_"):
        codepoint = pid.replace("invisible_unicode_", "")
        return f"Blocked: content contains invisible unicode character {codepoint} (possible injection)."
    return (
        f"Blocked: content matches threat pattern '{pid}'. Memory is injected into "
        f"the system prompt and must not contain injection or exfiltration payloads."
    )


__all__ = ["INVISIBLE_CHARS", "scan_for_threats", "first_threat_message"]

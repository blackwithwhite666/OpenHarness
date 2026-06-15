"""Disciplined backend for ohmo personal memory.

Wraps the existing on-disk layout (one ``<slug>.md`` per entry under
``~/.ohmo/memory/`` + a ``MEMORY.md`` index of ``- [Title](slug.md)`` links) with
the curation discipline ported from Hermes' ``tools/memory_tool.py``:

- **Unicode-safe slugs** — the legacy ASCII-only slug (``[^a-zA-Z0-9]+``) collapsed
  every Cyrillic title to ``memory`` so all RU memories clobbered one file. Here
  slugs keep Unicode word chars, so ``"Тренировки"`` → ``тренировки.md``.
- **No silent clobber** — ``add`` refuses to overwrite an existing slug with
  different content (suggests ``update``); identical content is a no-op.
- **Dedup** — exact-duplicate content (any slug) is rejected.
- **Bounds + error-on-overflow** — a per-entry char cap and a whole-store char
  budget; an over-budget ``add`` returns a structured error that tells the agent
  to consolidate *this turn* and lists the current entries, instead of silently
  growing the corpus.
- **Index consistency** — the link line is upserted/removed atomically with the
  file, and the title label is kept correct on ``update`` (the legacy naive
  substring dedup left stale labels).

The renderer (``ohmo.memory.load_memory_prompt``) and the ``/memory`` slash
command keep reading the same files, so this store is drop-in compatible.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from ohmo.threat_patterns import first_threat_message
from ohmo.workspace import get_memory_dir, get_memory_index_path

# Defaults are env-tunable. The per-entry cap matches the 4000-char body
# truncation in load_memory_prompt, so any entry that fits is injected whole when
# it is among the first files. The store budget keeps the corpus curated (it is
# NOT a prompt-size guard — only the first 5 bodies + index are ever injected —
# but it forces consolidation before memory sprawls).
DEFAULT_ENTRY_CHAR_LIMIT = 4000
DEFAULT_STORE_CHAR_BUDGET = 24000
_INDEX_HEADER = "# Memory Index"
_MAX_SLUG_LEN = 80
# The index basename (MEMORY.md) is reserved — never treat it as an entry. Compared
# case-insensitively so a "Memory"-titled entry can't collide with it on a
# case-insensitive filesystem (macOS).
_RESERVED_NAMES = {"memory.md"}

# Matches an index link line: "- [Title](slug.md)" (tolerant of bullet/space).
_LINK_RE = re.compile(r"^\s*[-*]\s*\[(?P<title>.*?)\]\((?P<name>[^)]+)\)\s*$")


def slugify(title: str) -> str:
    """Filesystem-safe slug that preserves Unicode word characters.

    ``\\w`` is Unicode-aware for ``str`` patterns in Python 3, so Cyrillic /
    accented letters survive (``"Café заметки"`` → ``café_заметки``) instead of
    collapsing to a single fallback slug. Empty/symbol-only titles fall back to
    ``"note"`` (not ``"memory"`` — that would collide with the reserved index).
    """
    slug = re.sub(r"[^\w]+", "_", title.strip().lower()).strip("_")
    # Fallback is "note", NOT "memory", so a degenerate title can't collide with
    # the reserved MEMORY.md index basename.
    return (slug or "note")[:_MAX_SLUG_LEN]


@dataclass(frozen=True)
class MemoryEntry:
    name: str  # filename, e.g. "timezone.md"
    slug: str  # "timezone"
    title: str  # index label (falls back to slug if no index line)
    content: str
    path: Path


@dataclass(frozen=True)
class MemoryOpResult:
    ok: bool
    message: str
    entries: tuple[MemoryEntry, ...] | None = None  # populated on overflow


class MemoryStore:
    """Workspace-scoped (NOT session-scoped) disciplined memory store."""

    def __init__(
        self,
        workspace: str | Path | None = None,
        *,
        entry_char_limit: int | None = None,
        store_char_budget: int | None = None,
    ) -> None:
        self._workspace = workspace
        self._entry_char_limit = entry_char_limit if entry_char_limit is not None else _env_int(
            "OHMO_MEMORY_ENTRY_CHARS", DEFAULT_ENTRY_CHAR_LIMIT
        )
        self._store_char_budget = store_char_budget if store_char_budget is not None else _env_int(
            "OHMO_MEMORY_STORE_CHARS", DEFAULT_STORE_CHAR_BUDGET
        )

    # -- paths --------------------------------------------------------------
    def _dir(self) -> Path:
        return get_memory_dir(self._workspace)

    def _index_path(self) -> Path:
        return get_memory_index_path(self._workspace)

    # -- reads --------------------------------------------------------------
    def _index_titles(self) -> dict[str, str]:
        """Map ``slug.md`` -> title label from MEMORY.md."""
        index = self._index_path()
        titles: dict[str, str] = {}
        if index.exists():
            for line in index.read_text(encoding="utf-8", errors="replace").splitlines():
                m = _LINK_RE.match(line)
                if m:
                    titles[m.group("name").strip()] = m.group("title").strip()
        return titles

    def entry_paths(self) -> list[Path]:
        """Entry file paths — excludes the MEMORY.md index and symlinks (a symlink
        planted in the dir must not be read/injected into the system prompt)."""
        memory_dir = self._dir()
        if not memory_dir.exists():
            return []
        files = [
            p
            for p in memory_dir.glob("*.md")
            if p.name.lower() not in _RESERVED_NAMES and not p.is_symlink()
        ]
        return sorted(files)

    def list(self) -> list[MemoryEntry]:
        titles = self._index_titles()
        entries: list[MemoryEntry] = []
        for path in self.entry_paths():
            content = path.read_text(encoding="utf-8", errors="replace").strip()
            entries.append(
                MemoryEntry(
                    name=path.name,
                    slug=path.stem,
                    title=titles.get(path.name, path.stem),
                    content=content,
                    path=path,
                )
            )
        return entries

    def get(self, name: str) -> MemoryEntry | None:
        """Resolve NAME → NAME, NAME.md, slugify(NAME).md, with a containment guard."""
        path = self._resolve_path(name)
        if path is None or not path.exists():
            return None
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        return MemoryEntry(
            name=path.name,
            slug=path.stem,
            title=self._index_titles().get(path.name, path.stem),
            content=content,
            path=path,
        )

    def _resolve_path(self, name: str) -> Path | None:
        memory_dir = self._dir().resolve()
        raw = (name or "").strip()
        if not raw:
            return None
        candidates = [raw, raw if raw.endswith(".md") else f"{raw}.md", f"{slugify(raw)}.md"]
        for cand in candidates:
            candidate = (memory_dir / cand)
            try:
                resolved = candidate.resolve()
            except (OSError, RuntimeError):
                continue
            # Path-traversal guard: must stay inside the memory dir.
            # Path-traversal guard + never resolve to the reserved index file.
            if resolved.parent != memory_dir or resolved.name.lower() in _RESERVED_NAMES:
                continue
            if resolved.exists():
                return resolved
        # Fall back to the slug path even if it does not exist (for callers that
        # only need the intended location), still containment- + reserve-checked.
        target = (memory_dir / f"{slugify(raw)}.md").resolve()
        if target.parent != memory_dir or target.name.lower() in _RESERVED_NAMES:
            return None
        return target

    def total_chars(self) -> int:
        return sum(len(e.content) for e in self.list())

    # -- writes -------------------------------------------------------------
    def add(self, title: str, content: str) -> MemoryOpResult:
        content = (content or "").strip()
        title = (title or "").strip()
        if not title:
            return MemoryOpResult(False, "A title is required.")
        if not content:
            return MemoryOpResult(False, "Content cannot be empty.")

        # Safety scan before anything reaches disk / the system prompt.
        threat = first_threat_message(f"{title}\n{content}", scope="strict")
        if threat:
            return MemoryOpResult(False, threat)

        limit = self._entry_char_limit
        if len(content) > limit:
            return MemoryOpResult(
                False,
                f"Entry is {len(content):,} chars, over the {limit:,}-char per-entry limit. "
                f"Split it into focused entries or shorten it.",
            )

        slug = slugify(title)
        name = f"{slug}.md"
        if name.lower() in _RESERVED_NAMES:
            return MemoryOpResult(
                False, "That title is reserved for the memory index — choose a more specific title."
            )
        existing = self.list()

        # Exact-duplicate content anywhere → no-op success (mirror Hermes).
        for e in existing:
            if e.content == content:
                return MemoryOpResult(True, f"Already remembered (matches {e.name}); nothing added.")

        # Slug collision with different content → refuse clobber, suggest update.
        for e in existing:
            if e.slug == slug:
                return MemoryOpResult(
                    False,
                    f"An entry {name!r} already exists with different content. "
                    f"Use action='update' (name={slug!r}) to change it, or choose a more "
                    f"specific title so it gets its own file.",
                )

        # Store budget → overflow: tell the agent to consolidate THIS turn.
        new_total = self.total_chars() + len(content)
        if new_total > self._store_char_budget:
            return MemoryOpResult(
                False,
                f"Memory at {self.total_chars():,}/{self._store_char_budget:,} chars. "
                f"Adding {title!r} ({len(content):,} chars) would exceed the budget. "
                f"Consolidate now — use action='update' to merge overlapping entries into "
                f"shorter ones, or action='remove' to drop stale/less-important ones (see "
                f"entries below), then retry this add — all in this turn.",
                entries=tuple(existing),
            )

        memory_dir = self._dir()
        memory_dir.mkdir(parents=True, exist_ok=True)
        (memory_dir / name).write_text(content + "\n", encoding="utf-8")
        self._upsert_index(name, title)
        return MemoryOpResult(True, f"Saved memory {name}.")

    def update(self, name: str, content: str, *, title: str | None = None) -> MemoryOpResult:
        content = (content or "").strip()
        if not content:
            return MemoryOpResult(False, "Content cannot be empty.")
        threat = first_threat_message(f"{title or ''}\n{content}", scope="strict")
        if threat:
            return MemoryOpResult(False, threat)
        path = self._resolve_path(name)
        if path is None or not path.exists():
            return MemoryOpResult(False, f"No memory entry {name!r}. Use action='add' to create it.")

        limit = self._entry_char_limit
        if len(content) > limit:
            return MemoryOpResult(
                False,
                f"Entry is {len(content):,} chars, over the {limit:,}-char per-entry limit. "
                f"Shorten it or split into focused entries.",
            )

        # Budget check on the delta (the new content replaces the old).
        old_len = len(path.read_text(encoding="utf-8", errors="replace").strip())
        new_total = self.total_chars() - old_len + len(content)
        if new_total > self._store_char_budget:
            return MemoryOpResult(
                False,
                f"Memory would be {new_total:,}/{self._store_char_budget:,} chars after this update. "
                f"Trim this entry or remove stale ones first (see entries below), then retry — this turn.",
                entries=tuple(self.list()),
            )

        path.write_text(content + "\n", encoding="utf-8")
        if title and title.strip():
            self._upsert_index(path.name, title.strip())
        return MemoryOpResult(True, f"Updated memory {path.name}.")

    def remove(self, name: str) -> MemoryOpResult:
        path = self._resolve_path(name)
        if path is None or not path.exists():
            return MemoryOpResult(False, f"No memory entry {name!r}.")
        filename = path.name
        path.unlink(missing_ok=True)
        self._drop_index(filename)
        return MemoryOpResult(True, f"Removed memory {filename}.")

    def add_legacy(self, title: str, content: str) -> Path:
        """Permissive writer for the ``/memory`` slash command + CLI.

        Unlike :meth:`add`, it always writes and returns the path (no dedup or
        budget refusal — a human asked for it), but it is unicode-safe and never
        clobbers the reserved ``MEMORY.md`` index (a reserved slug falls back to
        ``<slug>_note``), and the index link is upserted per-line (correct label
        on re-add). This keeps ``/memory`` on the same disciplined index path as
        the model tool, eliminating the legacy substring-dedup / clobber bugs.
        """
        title = (title or "").strip()
        content = (content or "").strip()
        threat = first_threat_message(f"{title}\n{content}", scope="strict")
        if threat:
            raise ValueError(threat)
        slug = slugify(title)
        if f"{slug}.md".lower() in _RESERVED_NAMES:
            slug = f"{slug}_note"
        name = f"{slug}.md"
        memory_dir = self._dir()
        memory_dir.mkdir(parents=True, exist_ok=True)
        path = memory_dir / name
        path.write_text(content + "\n", encoding="utf-8")
        self._upsert_index(name, title or slug)
        return path

    # -- index management ---------------------------------------------------
    def _read_index_lines(self) -> list[str]:
        index = self._index_path()
        if index.exists():
            return index.read_text(encoding="utf-8", errors="replace").splitlines()
        return [_INDEX_HEADER]

    def _write_index_lines(self, lines: list[str]) -> None:
        index = self._index_path()
        index.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(lines).rstrip() + "\n"
        index.write_text(text, encoding="utf-8")

    def _upsert_index(self, name: str, title: str) -> None:
        link = f"- [{title}]({name})"
        lines = self._read_index_lines()
        for i, line in enumerate(lines):
            m = _LINK_RE.match(line)
            if m and m.group("name").strip() == name:
                lines[i] = link  # replace (keeps label correct on update)
                self._write_index_lines(lines)
                return
        lines.append(link)
        self._write_index_lines(lines)

    def _drop_index(self, name: str) -> None:
        kept: list[str] = []
        for line in self._read_index_lines():
            m = _LINK_RE.match(line)
            if m and m.group("name").strip() == name:
                continue
            kept.append(line)
        self._write_index_lines(kept)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

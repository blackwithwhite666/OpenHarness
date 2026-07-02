"""Response-level completion cache for the query-engine eval inner loop.

The query-engine eval already replays *tools* deterministically. The only
remaining source of cost and run-to-run flap is the agent model's turns (and the
trajectory judge's votes). This module memoizes model *completions* keyed by the
full request — model + system prompt + tool schemas + params + message prefix —
so a re-run whose prompts are unchanged serves every turn from disk with zero
model calls and zero non-determinism.

It is the same idea as ``args_then_order`` fixture matching, one level up: there
we replay tool outputs, here we replay model completions.

Semantics:

* A key maps to an *ordered list* of recorded completions. The Nth lookup of a
  key during a run returns the Nth recorded completion. This lets identical
  requests that are *meant* to be sampled repeatedly — the judge's N majority
  votes — replay each distinct sample in order instead of collapsing to one.
* A lookup is a HIT only when its position is backed by a completion that was
  loaded from disk *before this run started*. Completions recorded during the
  current run never count as hits, so a cold run over an empty cache reports 0%
  hit and simply records, while a warm run over an unchanged suite reports
  ~100%.

The cache measures *reproducibility against a recording*, not fresh model
capability: keying on ``model`` means a model change invalidates automatically,
and a periodic cold run refreshes the recording.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiStreamEvent,
    SupportsStreamingMessages,
)
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, TextBlock

CACHE_FORMAT_VERSION = 1


def request_cache_key(request: ApiMessageRequest) -> str:
    """Return a stable hash of everything that determines a completion.

    Two requests share a key iff the model would see identical input: model,
    system prompt, token budget, effort, tool schemas, and the full message
    prefix. List order (messages, tools) is preserved — it is part of what the
    model sees — while dict key order is normalized via ``sort_keys``.
    """
    payload = {
        "v": CACHE_FORMAT_VERSION,
        "model": request.model,
        "system_prompt": request.system_prompt,
        "max_tokens": request.max_tokens,
        "effort": request.effort,
        "tools": request.tools,
        "messages": [message.model_dump(mode="json") for message in request.messages],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _serialize_event(event: ApiMessageCompleteEvent) -> dict[str, Any]:
    return {
        "message": event.message.model_dump(mode="json"),
        "usage": event.usage.model_dump(mode="json"),
        "stop_reason": event.stop_reason,
    }


def _deserialize_event(data: dict[str, Any]) -> ApiMessageCompleteEvent:
    return ApiMessageCompleteEvent(
        message=ConversationMessage.model_validate(data["message"]),
        usage=UsageSnapshot.model_validate(data.get("usage", {})),
        stop_reason=data.get("stop_reason"),
    )


class CompletionCache:
    """Disk-backed ordered-list-per-key store of model completions."""

    def __init__(
        self, path: str | Path | None = None, *, strict_offline: bool = False
    ) -> None:
        self._path = Path(path).expanduser().resolve() if path is not None else None
        # Strict/offline: on a miss, never call the wrapped model — the wrapper
        # yields an empty stub so the turn ends. Used in CI (no model access) and
        # to measure hit-rate cheaply: a prompt/code change that invalidates the
        # cache surfaces as misses instead of silently going live.
        self.strict_offline = strict_offline
        # key -> completions recorded across all runs (loaded + this run)
        self._store: dict[str, list[dict[str, Any]]] = {}
        # key -> number of completions that existed on disk before this run
        self._preloaded_len: dict[str, int] = {}
        # key -> next lookup position for THIS run (reset per process)
        self._positions: dict[str, int] = {}
        # key -> highest position served (hit) this run, for save_pruned()
        self._used_positions: dict[str, int] = {}
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()
        if self._path is not None and self._path.exists():
            self._load()

    def _load(self) -> None:
        assert self._path is not None
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        entries = raw.get("entries", {}) if isinstance(raw, dict) else {}
        for key, completions in entries.items():
            if isinstance(completions, list):
                self._store[key] = list(completions)
                self._preloaded_len[key] = len(completions)

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def total(self) -> int:
        return self._hits + self._misses

    @property
    def hit_rate(self) -> float:
        total = self.total
        return self._hits / total if total else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": self.total,
            "hit_rate": round(self.hit_rate, 6),
            "keys": len(self._store),
        }

    def lookup(self, key: str) -> ApiMessageCompleteEvent | None:
        """Return the next recorded completion for ``key``, or ``None`` on miss.

        Every call advances the per-key position so repeated identical requests
        (e.g. the judge's majority votes) walk the recorded list. Only positions
        backed by the pre-run recording count as hits.
        """
        with self._lock:
            position = self._positions.get(key, 0)
            self._positions[key] = position + 1
            if position < self._preloaded_len.get(key, 0):
                self._hits += 1
                self._used_positions[key] = max(
                    self._used_positions.get(key, -1), position
                )
                data = self._store[key][position]
            else:
                self._misses += 1
                data = None
        if data is None:
            return None
        return _deserialize_event(data)

    def record(self, key: str, event: ApiMessageCompleteEvent) -> None:
        with self._lock:
            self._store.setdefault(key, []).append(_serialize_event(event))

    def _write(self, path: Path, entries: dict[str, list[dict[str, Any]]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": CACHE_FORMAT_VERSION, "entries": entries}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)

    def save(self) -> None:
        if self._path is None:
            return
        self._write(self._path, self._store)

    def save_pruned(self, path: str | Path) -> dict[str, int]:
        """Write a minimal cache with only the completions served this run.

        Drops keys never looked up (stale/other-run recordings) and, per used
        key, keeps completions only up to the highest position actually served.
        The result replays the same trajectory this run took, so a re-run against
        it reproduces the same hit-rate — the committable artifact for CI. Returns
        {keys, completions} counts.
        """
        pruned = {
            key: self._store[key][: self._used_positions[key] + 1]
            for key in sorted(self._used_positions)
        }
        self._write(Path(path).expanduser().resolve(), pruned)
        return {
            "keys": len(pruned),
            "completions": sum(len(v) for v in pruned.values()),
        }


class CachingApiClient:
    """Wrap a streaming client, serving recorded completions on cache hits.

    On a hit the wrapped client is never called: the single cached
    ``ApiMessageCompleteEvent`` is yielded (text deltas are a UI concern and are
    not needed to drive the engine or the judge). On a miss the wrapped client
    streams live, the terminal completion is recorded, and every event passes
    through unchanged.
    """

    def __init__(self, inner: SupportsStreamingMessages, cache: CompletionCache) -> None:
        self._inner = inner
        self._cache = cache

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        key = request_cache_key(request)
        cached = self._cache.lookup(key)
        if cached is not None:
            yield cached
            return
        if self._cache.strict_offline:
            # Offline: don't touch the model. Yield an empty stub so the turn
            # ends cleanly (the case truncates at the first miss). The miss is
            # already counted in lookup().
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant", content=[TextBlock(text="")]
                ),
                usage=UsageSnapshot(),
                stop_reason="cache_miss_strict",
            )
            return
        recorded = False
        async for event in self._inner.stream_message(request):
            if isinstance(event, ApiMessageCompleteEvent) and not recorded:
                # Record exactly one completion per call so the ordered list
                # stays 1:1 with lookups on replay.
                self._cache.record(key, event)
                recorded = True
            yield event


class NullApiClient:
    """A model client that refuses to be called.

    In strict/offline replay (``--cache-strict``) the model is never invoked: hits
    come from the cache and misses yield a stub. So the underlying client is only
    a placeholder — using this one means an offline run needs no API auth, and a
    real call (a bug) fails loudly instead of silently going live.
    """

    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        raise RuntimeError(
            "model call in strict-offline eval (cache miss with no model available)"
        )
        yield  # pragma: no cover - makes this an async generator

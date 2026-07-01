from __future__ import annotations

from pathlib import Path

import pytest

from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
)
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, TextBlock
from openharness.evals import CachingApiClient, CompletionCache, request_cache_key


def _req(
    text: str,
    *,
    model: str = "m",
    system: str | None = "s",
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
) -> ApiMessageRequest:
    return ApiMessageRequest(
        model=model,
        messages=[ConversationMessage.from_user_text(text)],
        system_prompt=system,
        max_tokens=max_tokens,
        tools=tools or [],
    )


def _complete(text: str) -> ApiMessageCompleteEvent:
    return ApiMessageCompleteEvent(
        message=ConversationMessage(role="assistant", content=[TextBlock(text=text)]),
        usage=UsageSnapshot(input_tokens=3, output_tokens=5),
    )


class _RecordingClient:
    """Fake inner client: yields a scripted completion per call, counts calls."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0

    async def stream_message(self, request: ApiMessageRequest):
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        yield ApiTextDeltaEvent(text=reply[:1])
        yield _complete(reply)


# ---- request_cache_key -------------------------------------------------------


def test_key_is_stable_for_equal_requests():
    assert request_cache_key(_req("hello")) == request_cache_key(_req("hello"))


def test_key_changes_with_message_content():
    assert request_cache_key(_req("a")) != request_cache_key(_req("b"))


def test_key_ignores_tool_dict_ordering_but_not_list_order():
    a = _req("x", tools=[{"a": 1, "b": 2}])
    b = _req("x", tools=[{"b": 2, "a": 1}])
    assert request_cache_key(a) == request_cache_key(b)


def test_key_changes_with_model_and_system_and_tokens():
    base = _req("x")
    assert request_cache_key(base) != request_cache_key(_req("x", model="other"))
    assert request_cache_key(base) != request_cache_key(_req("x", system="other"))
    assert request_cache_key(base) != request_cache_key(_req("x", max_tokens=8))


def test_key_changes_with_message_order():
    two = ["one", "two"]
    forward = ApiMessageRequest(
        model="m",
        messages=[ConversationMessage.from_user_text(t) for t in two],
    )
    reverse = ApiMessageRequest(
        model="m",
        messages=[ConversationMessage.from_user_text(t) for t in reversed(two)],
    )
    assert request_cache_key(forward) != request_cache_key(reverse)


# ---- CompletionCache ---------------------------------------------------------


def test_cold_lookup_misses_then_records():
    cache = CompletionCache()
    assert cache.lookup("k") is None  # nothing preloaded
    cache.record("k", _complete("v"))
    # Recorded-this-run entries never count as hits.
    assert cache.hits == 0
    assert cache.misses == 1


def test_ordered_list_per_key_replays_votes_in_order(tmp_path: Path):
    path = tmp_path / "c.json"
    cold = CompletionCache(path)
    key = "judgekey"
    # Cold: three identical requests (majority votes) all miss and record.
    for text in ("pass", "fail", "pass"):
        assert cold.lookup(key) is None
        cold.record(key, _complete(text))
    cold.save()

    warm = CompletionCache(path)
    got = [warm.lookup(key) for _ in range(3)]
    assert [e.message.text for e in got] == ["pass", "fail", "pass"]
    assert warm.hits == 3 and warm.misses == 0
    # A fourth lookup exhausts the preloaded list -> miss.
    assert warm.lookup(key) is None
    assert warm.misses == 1


def test_persistence_round_trip_turns_records_into_hits(tmp_path: Path):
    path = tmp_path / "c.json"
    cold = CompletionCache(path)
    cold.lookup("k")  # miss
    cold.record("k", _complete("cached"))
    cold.save()

    warm = CompletionCache(path)
    event = warm.lookup("k")
    assert event is not None and event.message.text == "cached"
    assert warm.hit_rate == 1.0
    assert warm.stats()["keys"] == 1


def test_stats_shape():
    cache = CompletionCache()
    cache.lookup("k")
    stats = cache.stats()
    assert set(stats) == {"hits", "misses", "total", "hit_rate", "keys"}
    assert stats["misses"] == 1 and stats["total"] == 1


# ---- CachingApiClient --------------------------------------------------------


@pytest.mark.asyncio
async def test_miss_calls_inner_records_and_passes_through(tmp_path: Path):
    cache = CompletionCache(tmp_path / "c.json")
    inner = _RecordingClient(["live-answer"])
    client = CachingApiClient(inner, cache)

    events = [ev async for ev in client.stream_message(_req("q"))]

    assert inner.calls == 1  # inner was invoked
    assert any(isinstance(e, ApiTextDeltaEvent) for e in events)  # deltas pass through
    complete = [e for e in events if isinstance(e, ApiMessageCompleteEvent)]
    assert complete and complete[0].message.text == "live-answer"
    assert cache.misses == 1


@pytest.mark.asyncio
async def test_cold_then_warm_is_full_hit(tmp_path: Path):
    path = tmp_path / "c.json"

    # Cold run: empty cache -> inner called, completion recorded, then saved.
    cold_cache = CompletionCache(path)
    cold_inner = _RecordingClient(["A1"])
    cold_client = CachingApiClient(cold_inner, cold_cache)
    async for _ in cold_client.stream_message(_req("turn")):
        pass
    cold_cache.save()
    assert cold_inner.calls == 1
    assert cold_cache.hit_rate == 0.0

    # Warm run: reload -> served from disk, inner never called.
    warm_cache = CompletionCache(path)
    warm_inner = _RecordingClient(["SHOULD-NOT-BE-USED"])
    warm_client = CachingApiClient(warm_inner, warm_cache)
    events = [ev async for ev in warm_client.stream_message(_req("turn"))]

    assert warm_inner.calls == 0  # cache hit: inner untouched
    complete = [e for e in events if isinstance(e, ApiMessageCompleteEvent)]
    assert complete and complete[0].message.text == "A1"
    assert warm_cache.hit_rate == 1.0


@pytest.mark.asyncio
async def test_strict_offline_miss_yields_stub_without_calling_inner(tmp_path: Path):
    cache = CompletionCache(tmp_path / "c.json", strict_offline=True)
    inner = _RecordingClient(["SHOULD-NOT-BE-CALLED"])
    client = CachingApiClient(inner, cache)

    events = [ev async for ev in client.stream_message(_req("q"))]

    assert inner.calls == 0  # strict: never goes to the model on a miss
    complete = [e for e in events if isinstance(e, ApiMessageCompleteEvent)]
    assert complete and complete[0].message.text == ""
    assert complete[0].stop_reason == "cache_miss_strict"
    assert cache.misses == 1 and cache.hits == 0


def test_save_pruned_drops_unserved_keys(tmp_path: Path):
    full = tmp_path / "full.json"
    seed = CompletionCache(full)
    seed.record("kA", _complete("a0"))
    seed.record("kA", _complete("a1"))
    seed.record("kB", _complete("b0"))  # never looked up below
    seed.save()

    warm = CompletionCache(full)
    assert warm.lookup("kA").message.text == "a0"
    assert warm.lookup("kA").message.text == "a1"

    out = tmp_path / "pruned.json"
    stats = warm.save_pruned(out)
    assert stats == {"keys": 1, "completions": 2}  # kB dropped

    reloaded = CompletionCache(out)
    assert reloaded.lookup("kA").message.text == "a0"
    assert reloaded.lookup("kB") is None  # not in the pruned artifact


def test_save_pruned_trims_positions_past_the_last_hit(tmp_path: Path):
    full = tmp_path / "full.json"
    seed = CompletionCache(full)
    seed.record("k", _complete("v0"))
    seed.record("k", _complete("v1"))  # recorded but not served below
    seed.save()

    warm = CompletionCache(full)
    warm.lookup("k")  # position 0 only

    out = tmp_path / "pruned.json"
    assert warm.save_pruned(out) == {"keys": 1, "completions": 1}  # v1 trimmed

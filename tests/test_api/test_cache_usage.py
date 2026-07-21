from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from openharness.api.codex_client import _usage_from_response
from openharness.api.usage import UsageSnapshot
from openharness.evals import EvalEpisode, EvalStore

from ohmo.evals.recorder import GatewayEvalRecorder


def test_usage_snapshot_cache_fields_default_to_zero() -> None:
    usage = UsageSnapshot()

    assert usage.cached_input_tokens == 0
    assert usage.cache_write_input_tokens == 0


def test_codex_usage_parses_cached_input_tokens() -> None:
    usage = _usage_from_response(
        {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 80},
            }
        }
    )

    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.cached_input_tokens == 80


def test_codex_usage_defaults_cached_input_tokens_when_details_are_missing() -> None:
    usage = _usage_from_response(
        {"usage": {"input_tokens": 100, "output_tokens": 20}}
    )

    assert usage.cached_input_tokens == 0


def test_record_model_call_writes_cached_input_tokens(tmp_path: Path) -> None:
    episode_id = "cache-usage"
    store = EvalStore(tmp_path)
    store.append_episode(
        EvalEpisode(
            episode_id=episode_id,
            source="gateway",
            app="ohmo",
            session_id="session-1",
            user_text="hello",
        )
    )
    recorder = GatewayEvalRecorder(store=store, episode_id=episode_id)
    event = SimpleNamespace(
        usage=UsageSnapshot(
            input_tokens=1,
            output_tokens=1,
            cached_input_tokens=7,
        )
    )

    recorder.record_model_call(event, model="gpt-prod")

    [recorded] = list(store.iter_events(episode_id))
    assert recorded.payload["cached_input_tokens"] == 7

from __future__ import annotations

from pathlib import Path

from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, TextBlock
from openharness.engine.stream_events import AssistantTurnComplete
from openharness.evals import EvalEpisode

from ohmo.evals import GatewayEvalRecorder, get_eval_store


def test_gateway_eval_recorder_record_model_call_writes_tokens(tmp_path: Path) -> None:
    store = get_eval_store(tmp_path)
    episode_id = "ep-recorder"
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
    event = AssistantTurnComplete(
        message=ConversationMessage(
            role="assistant",
            content=[TextBlock(text="done")],
        ),
        usage=UsageSnapshot(input_tokens=12, output_tokens=5),
    )

    recorder.record_model_call(event, model="gpt-prod")

    [recorded] = list(store.iter_events(episode_id))
    assert recorded.kind == "model_call"
    assert recorded.payload == {
        "model": "gpt-prod",
        "input_tokens": 12,
        "output_tokens": 5,
    }

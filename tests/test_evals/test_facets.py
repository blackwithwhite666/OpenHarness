from __future__ import annotations

from pathlib import Path

from openharness.evals import EvalEpisode, EvalEvent, EvalStore, collect_text_facets


def test_collect_text_facets_uses_private_text_only_as_transient_input(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    store.append_episode(
        EvalEpisode(
            episode_id="ep-1",
            source="gateway",
            app="ohmo",
            session_id="session-1",
            user_goal="Find the private launch plan",
            user_text="Please check Project Mango secretly",
            metadata={"secret": "metadata must not enter facets"},
        )
    )
    store.append_event(
        EvalEvent(
            episode_id="ep-1",
            kind="inbound_message",
            payload={
                "user_goal": "Find the private launch plan",
                "user_text": "Please check Project Mango secretly",
            },
        )
    )
    store.append_event(
        EvalEvent(
            episode_id="ep-1",
            kind="tool_started",
            tool_name="web_fetch",
            tool_call_id="tool-1",
            payload={
                "input_summary": "fetch https://private.example/launch",
                "input": {"url": "https://private.example/launch"},
            },
        )
    )
    store.append_event(
        EvalEvent(
            episode_id="ep-1",
            kind="tool_completed",
            tool_name="web_fetch",
            tool_call_id="tool-1",
            payload={
                "output_summary": "private launch result",
                "output": "the full private launch result body",
            },
        )
    )

    facets = collect_text_facets(store)

    assert [item.facet.facet_kind for item in facets] == [
        "user_goal",
        "user_request",
        "tool_input",
        "tool_output",
    ]
    assert [item.text for item in facets] == [
        "Find the private launch plan",
        "Please check Project Mango secretly",
        "fetch https://private.example/launch",
        "private launch result",
    ]
    serialized_facets = "\n".join(item.facet.model_dump_json() for item in facets)
    for sensitive_fragment in (
        "Find the private launch plan",
        "Please check Project Mango secretly",
        "https://private.example/launch",
        "private launch result",
        "the full private launch result body",
        "metadata must not enter facets",
    ):
        assert sensitive_fragment not in serialized_facets
    assert all(len(item.facet.text_hash) == 64 for item in facets)
    assert all(item.facet.text_length == len(item.text) for item in facets)

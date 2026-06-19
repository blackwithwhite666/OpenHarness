from __future__ import annotations

import json
from pathlib import Path

from openharness.evals import (
    EvalEpisode,
    EvalEvent,
    EvalResource,
    EvalResourceSnapshot,
    EvalStore,
    build_usage_graph,
    write_usage_graph,
)


def test_usage_graph_builds_metadata_only_graph_and_handles_bad_snapshots(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        "ep-one",
        events=[
            EvalEvent(
                episode_id="ep-one",
                kind="inbound_message",
                payload={"user_text": "private user text", "session_key": "telegram:secret"},
            ),
            EvalEvent(
                episode_id="ep-one",
                kind="resource_snapshot",
                payload={
                    "path": "states/ep-one/resource_snapshot.json",
                    "resource_count": 2,
                    "cwd": "/Users/private/project",
                },
            ),
            EvalEvent(
                episode_id="ep-one",
                kind="tool_started",
                tool_name="web_fetch",
                tool_call_id="tool-1",
                payload={"input": {"url": "https://private.example/secret"}},
            ),
            EvalEvent(
                episode_id="ep-one",
                kind="tool_completed",
                tool_name="web_fetch",
                tool_call_id="tool-1",
                payload={"output": "private tool output"},
            ),
            EvalEvent(
                episode_id="ep-one",
                kind="engine_error",
                is_error=True,
                payload={"message": "private stack detail"},
            ),
        ],
    )
    _write_snapshot(
        store,
        "ep-one",
        resources=[
            EvalResource(
                resource_id="ohmo.workspace.memory_dir",
                kind="local_directory",
                name="memory_dir",
                path="memory/Private_Project.md",
                exists=True,
                metadata={"file_names": ["Private_Project.md"]},
            ),
            EvalResource(
                resource_id="ohmo.runtime_tool.web_fetch",
                kind="runtime_tool",
                name="web_fetch",
                exists=True,
                metadata={"input_schema": {"properties": {"url": {"type": "string"}}}},
            ),
        ],
    )

    _add_episode(
        store,
        "ep-two",
        events=[
            EvalEvent(
                episode_id="ep-two",
                kind="inbound_message",
                payload={"user_text": "second private message"},
            ),
            EvalEvent(
                episode_id="ep-two",
                kind="resource_snapshot",
                payload={"path": "states/ep-two/missing_snapshot.json"},
            ),
            EvalEvent(
                episode_id="ep-two",
                kind="tool_started",
                tool_name="web_fetch",
                tool_call_id="tool-2",
                payload={"input": {"text": "private outbound text"}},
            ),
            EvalEvent(
                episode_id="ep-two",
                kind="tool_completed",
                tool_name="web_fetch",
                tool_call_id="tool-2",
                payload={"output": "sent privately"},
            ),
            EvalEvent(
                episode_id="ep-two",
                kind="engine_error",
                is_error=True,
                payload={"message": "same path motif"},
            ),
        ],
    )

    graph = build_usage_graph(store)
    nodes = {node.node_id: node for node in graph.nodes}
    edges = {(edge.source, edge.target, edge.kind): edge for edge in graph.edges}

    assert graph.metadata["episode_count"] == 2
    assert graph.metadata["event_count"] == 10
    assert graph.metadata["resource_snapshot_missing_count"] == 1
    assert graph.metadata["resource_snapshot_corrupt_count"] == 0
    assert graph.metadata["warnings"] == [
        {"kind": "resource_snapshot_missing", "episode_id": "ep-two"}
    ]

    assert nodes["episode:ep-one"].count == 1
    assert nodes["event_kind:inbound_message"].count == 2
    assert nodes["event_kind:engine_error"].count == 2
    assert nodes["event_kind:resource_snapshot_missing"].count == 1
    assert nodes["tool:web_fetch"].count == 2
    assert nodes["resource:ohmo.workspace.memory_dir"].metadata == {
        "resource_kind": "local_directory"
    }
    assert nodes["resource:ohmo.runtime_tool.web_fetch"].metadata == {
        "resource_kind": "runtime_tool"
    }

    assert edges[("episode:ep-one", "event_kind:resource_snapshot", "has_event")].count == 1
    assert edges[("episode:ep-two", "event_kind:resource_snapshot_missing", "has_event")].count == 1
    assert (
        edges[("event_kind:inbound_message", "event_kind:resource_snapshot", "next_event")].count
        == 2
    )
    assert edges[("episode:ep-one", "tool:web_fetch", "uses_tool")].count == 1
    assert edges[("episode:ep-two", "tool:web_fetch", "uses_tool")].count == 1
    assert (
        edges[("episode:ep-one", "resource:ohmo.workspace.memory_dir", "has_resource")].count
        == 1
    )
    assert (
        edges[("episode:ep-one", "resource:ohmo.runtime_tool.web_fetch", "has_resource")].count
        == 1
    )

    episode_motifs = {motif.episode_id: motif for motif in graph.episode_motifs}
    assert episode_motifs["ep-one"].event_kind_path == [
        "inbound_message",
        "resource_snapshot",
        "tool_started",
        "tool_completed",
        "engine_error",
    ]
    assert episode_motifs["ep-one"].tool_path == ["web_fetch"]
    assert episode_motifs["ep-two"].tool_path == ["web_fetch"]
    assert len(graph.motifs) == 1
    assert graph.motifs[0].count == 2
    assert graph.motifs[0].episode_ids == ["ep-one", "ep-two"]

    output_path = write_usage_graph(store, graph)
    assert output_path == store.root / "graph" / "usage_graph.json"
    serialized = output_path.read_text(encoding="utf-8")
    assert json.loads(serialized)["metadata"]["episode_count"] == 2
    for sensitive_fragment in (
        "private user text",
        "telegram:secret",
        "/Users/private/project",
        "https://private.example/secret",
        "private tool output",
        "private stack detail",
        "Private_Project.md",
        "memory/Private_Project.md",
        "missing_snapshot.json",
        "private outbound text",
        "sent privately",
    ):
        assert sensitive_fragment not in serialized


def _add_episode(store: EvalStore, episode_id: str, *, events: list[EvalEvent]) -> None:
    store.append_episode(
        EvalEpisode(
            episode_id=episode_id,
            source="gateway",
            app="ohmo",
            session_id=f"session-{episode_id}",
            user_goal="private goal",
            user_text="private text",
            metadata={
                "session_key": "telegram:secret",
                "cwd": "/Users/private/project",
            },
        )
    )
    for event in events:
        store.append_event(event)


def _write_snapshot(
    store: EvalStore,
    episode_id: str,
    *,
    resources: list[EvalResource],
) -> None:
    path = store.root / "states" / episode_id / "resource_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        EvalResourceSnapshot(episode_id=episode_id, resources=resources).model_dump_json(),
        encoding="utf-8",
    )

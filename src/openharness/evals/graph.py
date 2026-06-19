"""Offline metadata-only usage graph builder for eval episodes."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from openharness.evals.models import (
    EvalEpisode,
    EvalEvent,
    EvalResourceSnapshot,
    EvalUsageGraph,
    EvalUsageGraphEdge,
    EvalUsageGraphEpisodeMotif,
    EvalUsageGraphMotif,
    EvalUsageGraphNode,
)
from openharness.evals.store import EvalStore
from openharness.utils.fs import atomic_write_text


def build_usage_graph(store: EvalStore) -> EvalUsageGraph:
    """Build a metadata-only graph from captured eval episodes and events."""
    builder = _UsageGraphBuilder()
    episode_count = 0
    event_count = 0

    for episode_id in store.list_episode_ids():
        episode = store.get_episode(episode_id)
        if episode is None:
            continue
        episode_count += 1
        events = list(store.iter_events(episode_id))
        event_count += len(events)
        builder.add_episode(store, episode, events)

    return builder.build(episode_count=episode_count, event_count=event_count)


def write_usage_graph(
    store: EvalStore,
    graph: EvalUsageGraph | None = None,
    filename: str = "usage_graph.json",
) -> Path:
    """Write an eval usage graph below ``store.root/graph`` and return its path."""
    output_path = _graph_output_path(store, filename)
    payload = graph or build_usage_graph(store)
    atomic_write_text(output_path, payload.model_dump_json(indent=2) + "\n")
    return output_path


class _UsageGraphBuilder:
    def __init__(self) -> None:
        self._nodes: dict[str, EvalUsageGraphNode] = {}
        self._edges: dict[tuple[str, str, str], EvalUsageGraphEdge] = {}
        self._episode_motifs: list[EvalUsageGraphEpisodeMotif] = []
        self._motifs: dict[tuple[tuple[str, ...], tuple[str, ...]], EvalUsageGraphMotif] = {}
        self._warnings: list[dict[str, str]] = []
        self._warning_counts: Counter[str] = Counter()

    def add_episode(
        self,
        store: EvalStore,
        episode: EvalEpisode,
        events: list[EvalEvent],
    ) -> None:
        episode_node_id = _node_id("episode", episode.episode_id)
        self._add_node(
            episode_node_id,
            "episode",
            metadata={
                "source": episode.source,
                "app": episode.app,
                "status": episode.status,
            },
        )

        event_kind_path = [event.kind for event in events]
        tool_path = self._tool_path(events)
        self._add_episode_motif(
            episode_id=episode.episode_id,
            event_kind_path=event_kind_path,
            tool_path=tool_path,
        )

        for event in events:
            event_kind_node_id = _node_id("event_kind", event.kind)
            self._add_node(event_kind_node_id, "event_kind")
            self._add_edge(episode_node_id, event_kind_node_id, "has_event")

            if event.kind == "resource_snapshot":
                self._add_resource_snapshot(store, episode_node_id, episode.episode_id, event)

        for current_event, next_event in zip(events, events[1:]):
            self._add_edge(
                _node_id("event_kind", current_event.kind),
                _node_id("event_kind", next_event.kind),
                "next_event",
            )

        for tool_name in tool_path:
            tool_node_id = _node_id("tool", tool_name)
            self._add_node(tool_node_id, "tool")
            self._add_edge(episode_node_id, tool_node_id, "uses_tool")

    def build(self, *, episode_count: int, event_count: int) -> EvalUsageGraph:
        nodes = sorted(self._nodes.values(), key=lambda node: node.node_id)
        edges = sorted(
            self._edges.values(),
            key=lambda edge: (edge.kind, edge.source, edge.target),
        )
        motifs = sorted(
            self._motifs.values(),
            key=lambda motif: (motif.event_kind_path, motif.tool_path, motif.episode_ids),
        )
        metadata: dict[str, Any] = {
            "episode_count": episode_count,
            "event_count": event_count,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "resource_snapshot_missing_count": self._warning_counts[
                "resource_snapshot_missing"
            ],
            "resource_snapshot_corrupt_count": self._warning_counts[
                "resource_snapshot_corrupt"
            ],
        }
        if self._warnings:
            metadata["warnings"] = self._warnings

        return EvalUsageGraph(
            nodes=nodes,
            edges=edges,
            episode_motifs=self._episode_motifs,
            motifs=motifs,
            metadata=metadata,
        )

    def _add_node(
        self,
        node_id: str,
        node_type: str,
        *,
        count: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        node = self._nodes.get(node_id)
        if node is None:
            self._nodes[node_id] = EvalUsageGraphNode(
                node_id=node_id,
                node_type=node_type,
                count=count,
                metadata=metadata or {},
            )
            return
        node.count += count
        if metadata:
            for key, value in metadata.items():
                node.metadata.setdefault(key, value)

    def _add_edge(
        self,
        source: str,
        target: str,
        kind: str,
        *,
        count: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        key = (source, target, kind)
        edge = self._edges.get(key)
        if edge is None:
            self._edges[key] = EvalUsageGraphEdge(
                source=source,
                target=target,
                kind=kind,
                count=count,
                metadata=metadata or {},
            )
            return
        edge.count += count
        if metadata:
            for metadata_key, value in metadata.items():
                edge.metadata.setdefault(metadata_key, value)

    def _add_episode_motif(
        self,
        *,
        episode_id: str,
        event_kind_path: list[str],
        tool_path: list[str],
    ) -> None:
        self._episode_motifs.append(
            EvalUsageGraphEpisodeMotif(
                episode_id=episode_id,
                event_kind_path=event_kind_path,
                tool_path=tool_path,
            )
        )
        key = (tuple(event_kind_path), tuple(tool_path))
        motif = self._motifs.get(key)
        if motif is None:
            self._motifs[key] = EvalUsageGraphMotif(
                event_kind_path=event_kind_path,
                tool_path=tool_path,
                count=1,
                episode_ids=[episode_id],
            )
            return
        motif.count += 1
        motif.episode_ids.append(episode_id)

    def _add_resource_snapshot(
        self,
        store: EvalStore,
        episode_node_id: str,
        episode_id: str,
        event: EvalEvent,
    ) -> None:
        snapshot = _read_resource_snapshot(store, event)
        if isinstance(snapshot, str):
            self._add_snapshot_warning(snapshot, episode_node_id, episode_id)
            return

        for resource in snapshot.resources:
            resource_node_id = _node_id("resource", resource.resource_id)
            self._add_node(
                resource_node_id,
                "resource",
                metadata={"resource_kind": resource.kind},
            )
            self._add_edge(episode_node_id, resource_node_id, "has_resource")

    def _add_snapshot_warning(
        self,
        warning_kind: str,
        episode_node_id: str,
        episode_id: str,
    ) -> None:
        self._warning_counts[warning_kind] += 1
        self._warnings.append({"kind": warning_kind, "episode_id": episode_id})
        warning_node_id = _node_id("event_kind", warning_kind)
        self._add_node(warning_node_id, "event_kind")
        self._add_edge(episode_node_id, warning_node_id, "has_event")

    def _tool_path(self, events: list[EvalEvent]) -> list[str]:
        path: list[str] = []
        seen_calls: set[tuple[str, str]] = set()
        started_calls: set[tuple[str, str]] = set()
        for index, event in enumerate(events):
            if not event.tool_name:
                continue
            call_key = (
                event.tool_name,
                event.tool_call_id or f"event-{index}",
            )
            if event.kind == "tool_started":
                started_calls.add(call_key)
            elif event.tool_call_id and call_key in started_calls:
                continue
            if call_key in seen_calls:
                continue
            seen_calls.add(call_key)
            path.append(event.tool_name)
        return path


def _read_resource_snapshot(store: EvalStore, event: EvalEvent) -> EvalResourceSnapshot | str:
    path_value = event.payload.get("path")
    if not isinstance(path_value, str) or not path_value:
        return "resource_snapshot_missing"

    path = (store.root / path_value).resolve()
    if not _is_relative_to(path, store.root):
        return "resource_snapshot_corrupt"
    try:
        return EvalResourceSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "resource_snapshot_missing"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError):
        return "resource_snapshot_corrupt"


def _graph_output_path(store: EvalStore, filename: str) -> Path:
    graph_dir = store.root / "graph"
    output_path = (graph_dir / filename).resolve()
    if not _is_relative_to(output_path, graph_dir.resolve()):
        raise ValueError("usage graph filename must stay under store.root/graph")
    return output_path


def _node_id(node_type: str, key: str) -> str:
    return f"{node_type}:{key}"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True

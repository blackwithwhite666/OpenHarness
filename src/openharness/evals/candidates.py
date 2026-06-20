"""Candidate and draft-case mining for eval/data-flywheel episodes."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from openharness.evals.facets import collect_text_facets
from openharness.evals.models import (
    EvalCaseCandidate,
    EvalCaseDraft,
    EvalEvent,
    EvalPackManifest,
)
from openharness.evals.store import EvalStore
from openharness.evals.tool_labels import effective_tool_path
from openharness.utils.fs import atomic_write_text


@dataclass(frozen=True)
class EvalPackWrite:
    """Summary returned after writing a candidate or case draft pack."""

    manifest: EvalPackManifest
    manifest_path: Path
    records_path: Path
    manifest_relative_path: str
    records_relative_path: str


_INPUT_FACET_KINDS = {"user_goal", "user_request"}
_EXPECTED_FACET_KINDS = {"assistant_final", "gateway_error", "tool_output"}


def build_case_candidates(store: EvalStore) -> list[EvalCaseCandidate]:
    """Build metadata-only case candidates from captured episodes."""
    facets_by_episode = _facet_ids_by_episode(store)
    embedded_facet_ids = store.list_embedding_facet_ids()
    candidates: list[EvalCaseCandidate] = []
    for episode_id in store.list_episode_ids():
        episode = store.get_episode(episode_id)
        if episode is None:
            continue
        events = list(store.iter_events(episode_id))
        event_kind_path = [event.kind for event in events]
        tool_path = _tool_path(events)
        capability_path = effective_tool_path(events)
        facet_ids = facets_by_episode.get(episode_id, [])
        embedded_facet_count = sum(1 for facet_id in facet_ids if facet_id in embedded_facet_ids)
        signals = _signals(
            events,
            tool_path,
            embedded_facet_count=embedded_facet_count,
        )
        candidate_kind = _candidate_kind(signals)
        score = _score(signals, tool_path)
        candidates.append(
            EvalCaseCandidate(
                candidate_id=_stable_id("candidate", episode_id, candidate_kind),
                episode_id=episode_id,
                candidate_kind=candidate_kind,
                score=score,
                signals=signals,
                facet_ids=facet_ids,
                event_kind_path=event_kind_path,
                tool_path=tool_path,
                capability_path=capability_path,
                metadata={
                    "source": episode.source,
                    "app": episode.app,
                    "status": episode.status,
                    "event_count": len(events),
                    "tool_count": len(tool_path),
                    "error_count": sum(1 for event in events if event.is_error),
                    "facet_count": len(facet_ids),
                    "embedded_facet_count": embedded_facet_count,
                    "graph_motif_key": _motif_key(event_kind_path, tool_path),
                },
            )
        )
    return sorted(
        candidates,
        key=lambda candidate: (-candidate.score, candidate.episode_id, candidate.candidate_id),
    )


def build_case_drafts(
    store: EvalStore,
    candidates: Sequence[EvalCaseCandidate] | None = None,
) -> list[EvalCaseDraft]:
    """Build draft eval cases from mined candidates for later human review."""
    source_candidates = list(candidates) if candidates is not None else build_case_candidates(store)
    facets = collect_text_facets(store)
    facet_kind_by_id = {item.facet.facet_id: item.facet.facet_kind for item in facets}
    drafts: list[EvalCaseDraft] = []
    for candidate in source_candidates:
        input_facet_ids = [
            facet_id
            for facet_id in candidate.facet_ids
            if facet_kind_by_id.get(facet_id) in _INPUT_FACET_KINDS
        ]
        expected_facet_ids = [
            facet_id
            for facet_id in candidate.facet_ids
            if facet_kind_by_id.get(facet_id) in _EXPECTED_FACET_KINDS
        ]
        drafts.append(
            EvalCaseDraft(
                case_id=_stable_id("case", candidate.candidate_id, candidate.candidate_kind),
                candidate_id=candidate.candidate_id,
                episode_id=candidate.episode_id,
                case_kind=candidate.candidate_kind,
                input_facet_ids=input_facet_ids,
                expected_facet_ids=expected_facet_ids,
                tool_names=candidate.tool_path,
                capability_path=candidate.capability_path,
                rubric=_rubric(candidate),
                review_status="draft",
                metadata={
                    "candidate_score": candidate.score,
                    "signals": candidate.signals,
                    "event_count": candidate.metadata.get("event_count", 0),
                },
            )
        )
    return drafts


def write_candidate_pack(
    store: EvalStore,
    candidates: Sequence[EvalCaseCandidate] | None = None,
    *,
    records_filename: str = "candidates.jsonl",
    manifest_filename: str = "candidate_manifest.json",
) -> EvalPackWrite:
    """Write mined candidates under ``store.root/candidates``."""
    records = list(candidates) if candidates is not None else build_case_candidates(store)
    return _write_pack(
        store=store,
        directory_name="candidates",
        pack_kind="case_candidates",
        records=records,
        records_filename=records_filename,
        manifest_filename=manifest_filename,
    )


def write_case_draft_pack(
    store: EvalStore,
    drafts: Sequence[EvalCaseDraft] | None = None,
    *,
    candidates: Sequence[EvalCaseCandidate] | None = None,
    records_filename: str = "case_drafts.jsonl",
    manifest_filename: str = "case_manifest.json",
) -> EvalPackWrite:
    """Write draft cases under ``store.root/cases``."""
    records = list(drafts) if drafts is not None else build_case_drafts(store, candidates)
    return _write_pack(
        store=store,
        directory_name="cases",
        pack_kind="case_drafts",
        records=records,
        records_filename=records_filename,
        manifest_filename=manifest_filename,
    )


def _write_pack(
    *,
    store: EvalStore,
    directory_name: str,
    pack_kind: str,
    records: Sequence[BaseModel],
    records_filename: str,
    manifest_filename: str,
) -> EvalPackWrite:
    records_path = _pack_output_path(store, directory_name, records_filename)
    manifest_path = _pack_output_path(store, directory_name, manifest_filename)
    records_relative_path = records_path.relative_to(store.root).as_posix()
    _write_records(records_path, records)
    manifest = EvalPackManifest(
        pack_kind=pack_kind,
        records_path=records_relative_path,
        record_count=len(records),
        metadata={"privacy": "metadata_only"},
    )
    atomic_write_text(manifest_path, manifest.model_dump_json(indent=2) + "\n")
    return EvalPackWrite(
        manifest=manifest,
        manifest_path=manifest_path,
        records_path=records_path,
        manifest_relative_path=manifest_path.relative_to(store.root).as_posix(),
        records_relative_path=records_relative_path,
    )


def _facet_ids_by_episode(store: EvalStore) -> dict[str, list[str]]:
    facets: dict[str, list[str]] = {}
    for item in collect_text_facets(store):
        facets.setdefault(item.facet.episode_id, []).append(item.facet.facet_id)
    return facets


def _signals(
    events: Sequence[EvalEvent],
    tool_path: Sequence[str],
    *,
    embedded_facet_count: int,
) -> list[str]:
    signals: list[str] = []
    if any(event.is_error for event in events):
        signals.append("has_error")
    if tool_path:
        signals.append("uses_tools")
    if any(event.kind == "resource_snapshot" for event in events):
        signals.append("has_resource_snapshot")
    if any(event.kind == "gateway_final" for event in events):
        signals.append("has_final_response")
    if embedded_facet_count:
        signals.append("has_embeddings")
    if len({event.kind for event in events}) > 1 or tool_path:
        signals.append("has_graph_motif")
    if not signals:
        signals.append("conversation_only")
    return signals


def _candidate_kind(signals: Sequence[str]) -> str:
    if "has_error" in signals:
        return "error_recovery"
    if "uses_tools" in signals:
        return "tool_workflow"
    return "conversation_replay"


def _score(signals: Sequence[str], tool_path: Sequence[str]) -> float:
    score = 1.0
    if "has_error" in signals:
        score += 5.0
    if "uses_tools" in signals:
        score += 2.0
    if "has_resource_snapshot" in signals:
        score += 1.0
    if "has_final_response" in signals:
        score += 1.0
    if "has_embeddings" in signals:
        score += 0.5
    if "has_graph_motif" in signals:
        score += 0.5
    return score + min(len(tool_path), 3) * 0.25


def _tool_path(events: Sequence[EvalEvent]) -> list[str]:
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


def _rubric(candidate: EvalCaseCandidate) -> list[str]:
    rubric = [
        "preserve the original user intent referenced by input facets",
        "return an outcome consistent with expected facets after review",
    ]
    if candidate.tool_path:
        rubric.append("use an equivalent tool strategy when tools are required")
    if candidate.candidate_kind == "error_recovery":
        rubric.append("handle the error path without hiding the failure")
    return rubric


def _motif_key(event_kind_path: Sequence[str], tool_path: Sequence[str]) -> str:
    return "|".join(event_kind_path) + "::" + "|".join(tool_path)


def _write_records(path: Path, records: Sequence[BaseModel]) -> None:
    lines = [record.model_dump_json() for record in records]
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    atomic_write_text(path, payload)


def _pack_output_path(store: EvalStore, directory_name: str, filename: str) -> Path:
    output_dir = store.root / directory_name
    output_path = (output_dir / filename).resolve()
    if not _is_relative_to(output_path, output_dir.resolve()):
        raise ValueError(f"{directory_name} output filename must stay under store.root/{directory_name}")
    return output_path


def _stable_id(prefix: str, *parts: str) -> str:
    raw = "\n".join(parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:24]}"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True

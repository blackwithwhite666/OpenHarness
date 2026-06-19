"""Text facet extraction for eval/data-flywheel mining."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from openharness.evals.models import EvalEpisode, EvalEvent, EvalTextFacet
from openharness.evals.store import EvalStore


@dataclass(frozen=True)
class EvalTextFacetInput:
    """A text facet plus the normalized text used only for embedding calls."""

    facet: EvalTextFacet
    text: str


_EPISODE_TEXT_FIELDS: tuple[tuple[str, str], ...] = (
    ("user_goal", "user_goal"),
    ("user_text", "user_request"),
)

_EVENT_PAYLOAD_TEXT_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "inbound_message": (
        ("user_goal", "user_goal"),
        ("user_text", "user_request"),
    ),
    "tool_started": (("input_summary", "tool_input"),),
    "tool_completed": (("output_summary", "tool_output"),),
    "gateway_final": (("text", "assistant_final"),),
    "gateway_error": (("text", "gateway_error"),),
}


def collect_text_facets(store: EvalStore) -> list[EvalTextFacetInput]:
    """Collect normalized text facets without storing raw text in the facet model."""
    inputs: list[EvalTextFacetInput] = []
    seen: set[tuple[str, str, str]] = set()

    for episode_id in store.list_episode_ids():
        episode = store.get_episode(episode_id)
        if episode is None:
            continue
        _add_episode_facets(inputs, seen, episode)
        for event_index, event in enumerate(store.iter_events(episode_id)):
            _add_event_facets(inputs, seen, event_index, event)

    return inputs


def _add_episode_facets(
    inputs: list[EvalTextFacetInput],
    seen: set[tuple[str, str, str]],
    episode: EvalEpisode,
) -> None:
    for field_name, facet_kind in _EPISODE_TEXT_FIELDS:
        _append_facet(
            inputs,
            seen,
            episode_id=episode.episode_id,
            facet_kind=facet_kind,
            source_path=f"episode.{field_name}",
            value=getattr(episode, field_name),
            metadata={
                "source": episode.source,
                "app": episode.app,
                "privacy": episode.privacy,
                "status": episode.status,
            },
        )


def _add_event_facets(
    inputs: list[EvalTextFacetInput],
    seen: set[tuple[str, str, str]],
    event_index: int,
    event: EvalEvent,
) -> None:
    fields = _EVENT_PAYLOAD_TEXT_FIELDS.get(event.kind, ())
    for field_name, facet_kind in fields:
        _append_facet(
            inputs,
            seen,
            episode_id=event.episode_id,
            facet_kind=facet_kind,
            source_path=f"events/{event_index:06d}.{event.kind}.payload.{field_name}",
            value=event.payload.get(field_name),
            metadata={
                "event_kind": event.kind,
                "tool_name": event.tool_name or "",
                "is_error": event.is_error,
            },
        )


def _append_facet(
    inputs: list[EvalTextFacetInput],
    seen: set[tuple[str, str, str]],
    *,
    episode_id: str,
    facet_kind: str,
    source_path: str,
    value: Any,
    metadata: dict[str, Any],
) -> None:
    text = _normalize_text(value)
    if not text:
        return
    text_hash = _hash_text(text)
    dedupe_key = (episode_id, facet_kind, text_hash)
    if dedupe_key in seen:
        return
    seen.add(dedupe_key)
    facet_id = _facet_id(episode_id, source_path, text_hash)
    inputs.append(
        EvalTextFacetInput(
            facet=EvalTextFacet(
                facet_id=facet_id,
                episode_id=episode_id,
                facet_kind=facet_kind,
                source_path=source_path,
                text_hash=text_hash,
                text_length=len(text),
                metadata=metadata,
            ),
            text=text,
        )
    )


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _facet_id(episode_id: str, source_path: str, text_hash: str) -> str:
    raw = f"{episode_id}\n{source_path}\n{text_hash}".encode("utf-8")
    return f"facet:{hashlib.sha256(raw).hexdigest()[:24]}"

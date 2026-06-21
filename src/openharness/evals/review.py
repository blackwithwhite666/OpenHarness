"""Review and promotion flow for eval case drafts."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Mapping, TypeVar

from pydantic import BaseModel, ValidationError

from openharness.evals.candidates import EvalPackWrite
from openharness.evals.models import EvalCaseDraft, EvalGoldCase, EvalPackManifest
from openharness.evals.state import compute_episode_state_delta
from openharness.evals.store import EvalStore
from openharness.utils.fs import atomic_write_text

ModelT = TypeVar("ModelT", bound=BaseModel)


def read_case_drafts(
    store: EvalStore,
    *,
    records_filename: str = "case_drafts.jsonl",
) -> list[EvalCaseDraft]:
    """Read case drafts from ``store.root/cases``."""
    return _read_jsonl_models(
        EvalCaseDraft,
        _cases_output_path(store, records_filename),
        missing_ok=False,
    )


def read_gold_cases(
    store: EvalStore,
    *,
    records_filename: str = "gold_cases.jsonl",
) -> list[EvalGoldCase]:
    """Read existing promoted gold cases from ``store.root/cases``."""
    return _read_jsonl_models(
        EvalGoldCase,
        _cases_output_path(store, records_filename),
        missing_ok=True,
    )


def promote_case_drafts(
    store: EvalStore,
    *,
    case_ids: Sequence[str] | None = None,
    reviewer: str = "",
    review_metadata_by_case: Mapping[str, Mapping[str, object]] | None = None,
    draft_records_filename: str = "case_drafts.jsonl",
    records_filename: str = "gold_cases.jsonl",
    manifest_filename: str = "gold_manifest.json",
) -> EvalPackWrite:
    """Promote selected draft cases into an idempotent reviewed gold pack."""
    drafts = read_case_drafts(store, records_filename=draft_records_filename)
    selected = select_case_drafts(drafts, case_ids)
    existing = {gold.case_id: gold for gold in read_gold_cases(store, records_filename=records_filename)}
    review_metadata = review_metadata_by_case or {}
    for draft in selected:
        candidate = _gold_from_draft(
            draft,
            store=store,
            reviewer=reviewer,
            review_metadata=review_metadata.get(draft.case_id),
        )
        current = existing.get(draft.case_id)
        if current is not None and _gold_content_fingerprint(current) == _gold_content_fingerprint(
            candidate
        ):
            continue
        existing[draft.case_id] = candidate

    records = sorted(existing.values(), key=lambda gold: (gold.episode_id, gold.case_id))
    records_path = _cases_output_path(store, records_filename)
    manifest_path = _cases_output_path(store, manifest_filename)
    records_relative_path = records_path.relative_to(store.root).as_posix()
    _write_records(records_path, records)
    manifest = EvalPackManifest(
        pack_kind="gold_cases",
        records_path=records_relative_path,
        record_count=len(records),
        metadata={
            "privacy": "metadata_only",
            "promoted_count": len(selected),
            "review_status": "approved",
        },
    )
    atomic_write_text(manifest_path, manifest.model_dump_json(indent=2) + "\n")
    return EvalPackWrite(
        manifest=manifest,
        manifest_path=manifest_path,
        records_path=records_path,
        manifest_relative_path=manifest_path.relative_to(store.root).as_posix(),
        records_relative_path=records_relative_path,
    )


def select_case_drafts(
    drafts: Sequence[EvalCaseDraft],
    case_ids: Sequence[str] | None,
) -> list[EvalCaseDraft]:
    """Select draft cases by ids, or all drafts when ``case_ids`` is None."""
    _ensure_unique_case_ids(drafts)
    if case_ids is None:
        return list(drafts)
    requested = list(dict.fromkeys(case_ids))
    if not requested:
        raise ValueError("case_ids must not be empty")
    by_id = {draft.case_id: draft for draft in drafts}
    missing = [case_id for case_id in requested if case_id not in by_id]
    if missing:
        raise ValueError(f"case draft not found: {', '.join(missing)}")
    return [by_id[case_id] for case_id in requested]


def _gold_from_draft(
    draft: EvalCaseDraft,
    *,
    store: EvalStore,
    reviewer: str,
    review_metadata: Mapping[str, object] | None = None,
) -> EvalGoldCase:
    metadata = {
        "source_review_status": draft.review_status,
        "candidate_score": draft.metadata.get("candidate_score", 0),
        "signals": draft.metadata.get("signals", []),
        "event_count": draft.metadata.get("event_count", 0),
    }
    if review_metadata:
        metadata.update(dict(review_metadata))
    state_delta = compute_episode_state_delta(store, draft.episode_id)
    if state_delta is not None:
        metadata["state_delta"] = state_delta
    else:
        metadata.pop("state_delta", None)
    return EvalGoldCase(
        gold_case_id=_stable_id("gold", draft.case_id),
        case_id=draft.case_id,
        candidate_id=draft.candidate_id,
        episode_id=draft.episode_id,
        case_kind=draft.case_kind,
        input_facet_ids=draft.input_facet_ids,
        expected_facet_ids=draft.expected_facet_ids,
        tool_names=draft.tool_names,
        capability_path=draft.capability_path,
        rubric=draft.rubric,
        scorer=draft.scorer,
        review_status="approved",
        reviewer=reviewer,
        metadata=metadata,
    )


def _ensure_unique_case_ids(drafts: Sequence[EvalCaseDraft]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for draft in drafts:
        if draft.case_id in seen:
            duplicates.append(draft.case_id)
        seen.add(draft.case_id)
    if duplicates:
        raise ValueError(f"duplicate case draft ids: {', '.join(sorted(set(duplicates)))}")


def _gold_content_fingerprint(gold: EvalGoldCase) -> str:
    comparable = gold.model_copy(update={"promoted_at": None, "reviewer": ""}).model_dump(
        mode="json"
    )
    return hashlib.sha256(repr(sorted(comparable.items())).encode("utf-8")).hexdigest()


def _read_jsonl_models(
    model_type: type[ModelT],
    path: Path,
    *,
    missing_ok: bool,
) -> list[ModelT]:
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        if missing_ok:
            return []
        raise

    records = []
    for line_number, line in enumerate(raw_lines, 1):
        if not line.strip():
            continue
        try:
            records.append(model_type.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"invalid {path.name} row {line_number}") from exc
    return records


def _write_records(path: Path, records: Sequence[BaseModel]) -> None:
    lines = [record.model_dump_json() for record in records]
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    atomic_write_text(path, payload)


def _cases_output_path(store: EvalStore, filename: str) -> Path:
    output_dir = store.root / "cases"
    output_path = (output_dir / filename).resolve()
    if not _is_relative_to(output_path, output_dir.resolve()):
        raise ValueError("case output filename must stay under store.root/cases")
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

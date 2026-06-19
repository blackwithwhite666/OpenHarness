"""Ohmo helpers for reviewing and promoting eval case drafts."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

from openharness.evals import (
    EvalPackWrite,
    promote_case_drafts,
    read_case_drafts,
    read_gold_cases,
    select_case_drafts,
)
from openharness.evals.models import EvalCaseDraft
from openharness.utils.fs import atomic_write_text

from ohmo.evals.adapter import get_eval_store


@dataclass(frozen=True)
class OhmoEvalReviewItem:
    """Metadata-only row shown by ``ohmo evals review``."""

    case_id: str
    case_kind: str
    episode_id: str
    review_status: str
    input_facet_count: int
    expected_facet_count: int
    tool_names: list[str]


@dataclass(frozen=True)
class OhmoEvalReviewResult:
    """Summary of draft cases selected for review output."""

    total_count: int
    shown: list[OhmoEvalReviewItem]


@dataclass(frozen=True)
class OhmoEvalPromoteResult:
    """Summary returned by Ohmo eval case promotion."""

    promoted_count: int
    total_draft_count: int
    remaining_unpromoted_count: int
    selected_case_ids: list[str]
    dry_run: bool
    manifest_path: Path
    records_path: Path
    write: EvalPackWrite | None = None


@dataclass(frozen=True)
class OhmoEvalReviewManifestWrite:
    """Summary returned after writing a metadata-only review manifest."""

    path: Path
    relative_path: str
    total_count: int
    shown_count: int


def review_ohmo_eval_case_drafts(
    *,
    workspace: str | Path | None = None,
    case_id: str | None = None,
    limit: int = 20,
) -> OhmoEvalReviewResult:
    """Return metadata-only draft case rows for CLI review."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    store = get_eval_store(workspace)
    drafts = read_case_drafts(store)
    selected = select_case_drafts(drafts, [case_id] if case_id else None)
    shown = selected if case_id else selected[:limit]
    return OhmoEvalReviewResult(
        total_count=len(drafts),
        shown=[_review_item(draft) for draft in shown],
    )


def write_ohmo_eval_review_manifest(
    *,
    workspace: str | Path | None = None,
    case_id: str | None = None,
    limit: int = 20,
    filename: str = "review_manifest.json",
) -> OhmoEvalReviewManifestWrite:
    """Write a metadata-only batch review manifest under ``evals/cases``."""
    result = review_ohmo_eval_case_drafts(
        workspace=workspace,
        case_id=case_id,
        limit=limit,
    )
    store = get_eval_store(workspace)
    path = _cases_output_path(store.root, filename)
    payload = {
        "schema_version": 1,
        "manifest_kind": "case_draft_review",
        "total_count": result.total_count,
        "shown_count": len(result.shown),
        "items": [_review_item_payload(item) for item in result.shown],
        "metadata": {
            "privacy": "metadata_only",
            "case_id": case_id or "",
            "limit": limit,
        },
    }
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )
    return OhmoEvalReviewManifestWrite(
        path=path,
        relative_path=path.relative_to(store.root).as_posix(),
        total_count=result.total_count,
        shown_count=len(result.shown),
    )


def promote_ohmo_eval_case_drafts(
    *,
    workspace: str | Path | None = None,
    case_ids: list[str] | None = None,
    promote_all: bool = False,
    manifest_filename: str | None = None,
    dry_run: bool = False,
    reviewer: str = "",
) -> OhmoEvalPromoteResult:
    """Promote selected draft cases into the reviewed gold pack."""
    if manifest_filename and (promote_all or case_ids):
        raise ValueError("choose either manifest or case_ids/promote_all")
    if promote_all and case_ids:
        raise ValueError("choose either case_ids or promote_all")
    if not manifest_filename and not promote_all and not case_ids:
        raise ValueError("choose case_ids, promote_all, or manifest")

    store = get_eval_store(workspace)
    drafts = read_case_drafts(store)
    review_metadata_by_case: dict[str, dict[str, object]] = {}
    if manifest_filename:
        manifest_selection = _approved_cases_from_review_manifest(
            store.root,
            manifest_filename,
        )
        case_ids = manifest_selection.case_ids
        review_metadata_by_case = manifest_selection.review_metadata_by_case

    selected = select_case_drafts(drafts, None if promote_all else case_ids)
    selected_case_ids = [draft.case_id for draft in selected]
    existing_gold_case_ids = {gold.case_id for gold in read_gold_cases(store)}
    promoted_case_ids = existing_gold_case_ids | set(selected_case_ids)
    remaining_unpromoted_count = sum(
        1 for draft in drafts if draft.case_id not in promoted_case_ids
    )
    manifest_path = store.root / "cases" / "gold_manifest.json"
    records_path = store.root / "cases" / "gold_cases.jsonl"

    if dry_run:
        return OhmoEvalPromoteResult(
            promoted_count=len(selected),
            total_draft_count=len(drafts),
            remaining_unpromoted_count=remaining_unpromoted_count,
            selected_case_ids=selected_case_ids,
            dry_run=True,
            manifest_path=manifest_path,
            records_path=records_path,
        )

    write = promote_case_drafts(
        store,
        case_ids=None if promote_all else selected_case_ids,
        reviewer=reviewer,
        review_metadata_by_case=review_metadata_by_case,
    )
    return OhmoEvalPromoteResult(
        promoted_count=len(selected),
        total_draft_count=len(drafts),
        remaining_unpromoted_count=remaining_unpromoted_count,
        selected_case_ids=selected_case_ids,
        dry_run=False,
        manifest_path=write.manifest_path,
        records_path=write.records_path,
        write=write,
    )


def _review_item(draft: EvalCaseDraft) -> OhmoEvalReviewItem:
    return OhmoEvalReviewItem(
        case_id=draft.case_id,
        case_kind=draft.case_kind,
        episode_id=draft.episode_id,
        review_status=draft.review_status,
        input_facet_count=len(draft.input_facet_ids),
        expected_facet_count=len(draft.expected_facet_ids),
        tool_names=draft.tool_names,
    )


def _review_item_payload(item: OhmoEvalReviewItem) -> dict[str, object]:
    return {
        "case_id": item.case_id,
        "case_kind": item.case_kind,
        "decision": "pending",
        "episode_id": item.episode_id,
        "review_status": item.review_status,
        "reviewer": "",
        "comment": "",
        "input_facet_count": item.input_facet_count,
        "expected_facet_count": item.expected_facet_count,
        "tool_names": item.tool_names,
    }


@dataclass(frozen=True)
class _ReviewManifestSelection:
    case_ids: list[str]
    review_metadata_by_case: dict[str, dict[str, object]]


def _approved_cases_from_review_manifest(
    store_root: Path,
    filename: str,
) -> _ReviewManifestSelection:
    path = _cases_output_path(store_root, filename)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid review manifest: {path.name}") from exc
    if not isinstance(payload, dict) or payload.get("manifest_kind") != "case_draft_review":
        raise ValueError("review manifest must have manifest_kind=case_draft_review")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("review manifest must contain an items list")

    case_ids: list[str] = []
    metadata_by_case: dict[str, dict[str, object]] = {}
    seen: set[str] = set()
    for index, raw_item in enumerate(raw_items, 1):
        if not isinstance(raw_item, dict):
            raise ValueError(f"review manifest item {index} must be an object")
        case_id = str(raw_item.get("case_id") or "").strip()
        if not case_id:
            raise ValueError(f"review manifest item {index} is missing case_id")
        if case_id in seen:
            raise ValueError(f"duplicate review manifest case_id: {case_id}")
        seen.add(case_id)
        decision = _normalize_review_decision(raw_item.get("decision", "pending"))
        if decision != "approved":
            continue
        case_ids.append(case_id)
        comment = str(raw_item.get("comment") or "")
        item_reviewer = str(raw_item.get("reviewer") or "").strip()
        metadata_by_case[case_id] = {
            "review_decision": decision,
            "review_manifest": path.name,
            "reviewer": item_reviewer,
            "review_comment_hash": _hash_text(comment) if comment else "",
            "review_comment_length": len(comment),
        }
    if not case_ids:
        raise ValueError("review manifest has no approved cases")
    return _ReviewManifestSelection(
        case_ids=case_ids,
        review_metadata_by_case=metadata_by_case,
    )


def _normalize_review_decision(value: object) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "approve": "approved",
        "approved": "approved",
        "yes": "approved",
        "reject": "rejected",
        "rejected": "rejected",
        "no": "rejected",
        "skip": "pending",
        "pending": "pending",
        "": "pending",
    }
    try:
        return aliases[normalized]
    except KeyError:
        raise ValueError(f"unknown review decision: {value}") from None


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _cases_output_path(store_root: Path, filename: str) -> Path:
    output_dir = store_root / "cases"
    output_path = (output_dir / filename).resolve()
    if not _is_relative_to(output_path, output_dir.resolve()):
        raise ValueError("review manifest filename must stay under store.root/cases")
    return output_path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True

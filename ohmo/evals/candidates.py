"""Ohmo entrypoints for mining eval candidates and draft cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openharness.evals import EvalPackWrite, build_case_candidates
from openharness.evals.candidates import write_candidate_pack, write_case_draft_pack

from ohmo.evals.adapter import get_eval_store


@dataclass(frozen=True)
class OhmoEvalMineWrite:
    """Summary returned after writing Ohmo eval candidate and case draft packs."""

    candidates: EvalPackWrite
    cases: EvalPackWrite


def write_ohmo_eval_mine(
    *,
    workspace: str | Path | None = None,
) -> OhmoEvalMineWrite:
    """Mine candidates and draft cases from captured Ohmo eval episodes."""
    store = get_eval_store(workspace)
    candidates = build_case_candidates(store)
    candidate_write = write_candidate_pack(store, candidates)
    case_write = write_case_draft_pack(store, candidates=candidates)
    return OhmoEvalMineWrite(candidates=candidate_write, cases=case_write)

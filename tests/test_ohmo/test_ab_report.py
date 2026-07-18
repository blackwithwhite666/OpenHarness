"""Tests for the cross-backend memory recall report and gate."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ohmo.evals.memory import report
from ohmo.evals.memory.benchmark import load_cases
from ohmo.evals.memory.provisioning import ProvisionedBackend
from ohmo.evals.memory.recall_judge import CaseScore
from ohmo.evals.memory.report import (
    BackendSummary,
    SweepResult,
    evaluate_gate,
    render_report,
    run_sweep,
)
from ohmo.memory_backend import FileMemoryBackend, MemoryHit

CASES_PATH = (
    Path(__file__).parents[2] / "ohmo" / "evals" / "memory" / "cases" / "memory_recall_v1.jsonl"
)


class FakeCompleter:
    """Answer from the supplied memory block without making any network calls."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if prompt.startswith(("SEMANTIC USE CHECK", "SEMANTIC GROUNDING CHECK")):
            return "NO"
        match = re.search(r"<memory>\n(.*?)\n</memory>", prompt, re.DOTALL)
        return match.group(1) if match is not None else "I don't know."


async def test_run_sweep_summarizes_isolated_backends_and_tears_them_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = load_cases(CASES_PATH)
    provisioned_backends: list[ProvisionedBackend] = []
    real_provision = report.provision_backend

    async def tracking_provision(*args: object, **kwargs: object) -> ProvisionedBackend:
        provisioned = await real_provision(*args, **kwargs)  # type: ignore[arg-type]
        provisioned_backends.append(provisioned)
        _install_isolated_file_search(provisioned, monkeypatch)
        return provisioned

    monkeypatch.setattr(report, "provision_backend", tracking_provision)

    result = await run_sweep(cases, ["file", "catalog"], complete=FakeCompleter())

    assert isinstance(result, SweepResult)
    assert set(result.by_kind) == {"file", "catalog"}
    assert set(result.raw_scores) == {"file", "catalog"}
    for kind in ("file", "catalog"):
        summary = result.by_kind[kind]
        assert isinstance(summary, BackendSummary)
        assert summary.kind == kind
        assert summary.n_cases == len(cases)
        assert len(result.raw_scores[kind]) == len(cases)
        assert all(isinstance(score, CaseScore) for score in result.raw_scores[kind])
        assert set(summary.per_category) == {case.category for case in cases}
        assert summary.mean_latency_ms >= 0.0

    workspaces = [provisioned.workspace for provisioned in provisioned_backends]
    assert len(workspaces) == len(cases) * 2
    assert len(set(workspaces)) == len(workspaces)
    assert all(provisioned._torn_down for provisioned in provisioned_backends)
    assert all(not workspace.exists() for workspace in workspaces)


def test_evaluate_gate_passes_when_catalog_meets_file_baseline() -> None:
    sweep = _sweep(file_score=0.75, catalog_score=0.80, file_grounded=0.90, catalog_grounded=0.90)

    gate = evaluate_gate(sweep)

    assert gate.passed is True
    assert gate.reasons == []
    assert gate.deltas["catalog.mean_score"] == pytest.approx(0.05)
    assert gate.deltas["catalog.grounded_rate"] == pytest.approx(0.0)


def test_evaluate_gate_fails_when_catalog_recall_is_worse() -> None:
    sweep = _sweep(file_score=0.80, catalog_score=0.70, file_grounded=0.90, catalog_grounded=0.95)

    gate = evaluate_gate(sweep)

    assert gate.passed is False
    assert any("catalog recall score is worse" in reason for reason in gate.reasons)
    assert gate.deltas["catalog.mean_score"] == pytest.approx(-0.10)


def test_evaluate_gate_fails_on_catalog_grounding_regression() -> None:
    sweep = _sweep(file_score=0.80, catalog_score=0.80, file_grounded=0.95, catalog_grounded=0.85)

    gate = evaluate_gate(sweep)

    assert gate.passed is False
    assert any("catalog grounding regressed" in reason for reason in gate.reasons)
    assert gate.deltas["catalog.grounded_rate"] == pytest.approx(-0.10)


def test_render_report_contains_both_backends_breakdown_and_gate() -> None:
    sweep = _sweep(file_score=0.75, catalog_score=0.80, file_grounded=1.0, catalog_grounded=1.0)
    gate = evaluate_gate(sweep)

    rendered = render_report(sweep, gate)

    assert "| Backend | Cases | Mean score | Recall | Use | Grounded" in rendered
    assert "| file |" in rendered
    assert "| catalog |" in rendered
    assert "write-early/recall-late" in rendered
    assert "GATE: PASS" in rendered


async def test_cli_writes_report_and_returns_nonzero_on_gate_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep = _sweep(file_score=0.80, catalog_score=0.70, file_grounded=1.0, catalog_grounded=1.0)
    client = _FakeApiClient()

    async def fake_run_sweep(*args: object, **kwargs: object) -> SweepResult:
        del args, kwargs
        return sweep

    monkeypatch.setattr(report, "load_settings", _FakeSettings)
    monkeypatch.setattr(report, "resolve_api_client_from_settings", lambda _settings: client)
    monkeypatch.setattr(report, "run_sweep", fake_run_sweep)
    output = tmp_path / "report.md"
    args = report._build_parser().parse_args(
        [
            "--cases",
            str(CASES_PATH),
            "--kinds",
            "file,catalog",
            "--out",
            str(output),
        ]
    )

    exit_code = await report._run_cli(args)

    assert exit_code == 1
    assert output.read_text(encoding="utf-8") == render_report(sweep, evaluate_gate(sweep))
    assert "GATE: FAIL" in output.read_text(encoding="utf-8")
    assert client.closed is True


class _FakeSettings:
    model = "fake-model"

    def merge_cli_overrides(self, **_overrides: object) -> _FakeSettings:
        return self

    def materialize_active_profile(self) -> _FakeSettings:
        return self


class _FakeApiClient:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _sweep(
    *,
    file_score: float,
    catalog_score: float,
    file_grounded: float,
    catalog_grounded: float,
) -> SweepResult:
    return SweepResult(
        by_kind={
            "file": _summary("file", file_score, file_grounded),
            "catalog": _summary("catalog", catalog_score, catalog_grounded),
        }
    )


def _summary(kind: str, score: float, grounded: float) -> BackendSummary:
    return BackendSummary(
        kind=kind,
        n_cases=2,
        mean_score=score,
        recall_rate=score,
        use_rate=score,
        grounded_rate=grounded,
        per_category={"write-early/recall-late": score},
        mean_latency_ms=2.5,
    )


def _install_isolated_file_search(
    provisioned: ProvisionedBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not isinstance(provisioned.backend, FileMemoryBackend):
        return
    backend = provisioned.backend

    async def search(query: str, top_k: int) -> list[MemoryHit]:
        query_terms = set(re.findall(r"\w+", query.casefold()))
        matches = []
        for entry in await backend.list():
            entry_terms = set(re.findall(r"\w+", f"{entry.title} {entry.content}".casefold()))
            if query_terms <= entry_terms:
                matches.append(entry)
        return [
            MemoryHit(
                name=entry.name,
                title=entry.title,
                snippet=entry.content,
                rank=rank,
            )
            for rank, entry in enumerate(matches[:top_k], start=1)
        ]

    monkeypatch.setattr(backend, "search", search)

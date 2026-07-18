"""Tests for the focused multi-turn memory-recall benchmark."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ohmo.evals.memory.benchmark import MemoryCase, Turn, load_cases, run_case
from ohmo.evals.memory.provisioning import BackendKind, ProvisionedBackend, provision_backend
from ohmo.memory_backend import FileMemoryBackend, MemoryHit

CASES_PATH = (
    Path(__file__).parents[2] / "ohmo" / "evals" / "memory" / "cases" / "memory_recall_v1.jsonl"
)


def test_load_cases_validates_memory_recall_dataset() -> None:
    cases = load_cases(CASES_PATH)

    assert len(cases) >= 8
    assert {case.category for case in cases} == {
        "write-early/recall-late",
        "update",
        "remove",
        "distractor",
        "no-fabrication",
    }
    assert len({case.id for case in cases}) == len(cases)
    assert all(isinstance(case, MemoryCase) for case in cases)
    assert all(case.turns and all(isinstance(turn, Turn) for turn in case.turns) for case in cases)
    assert all(any(turn.kind == "query" for turn in case.turns) for case in cases)


def test_load_cases_reports_malformed_case_with_line_and_field(tmp_path: Path) -> None:
    malformed = {
        "id": "bad-case",
        "category": "update",
        "seed_entries": [],
        "turns": [{"kind": "replace", "name": "old", "content": "new"}],
    }
    path = tmp_path / "malformed.jsonl"
    path.write_text(json.dumps(malformed) + "\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"malformed\.jsonl:1: invalid memory case: case\.turns\[0\]\.kind",
    ):
        load_cases(path)


@pytest.mark.parametrize("kind", ["catalog", "file"])
@pytest.mark.parametrize(
    ("case_id", "expected_category"),
    [
        ("recall_luggage_tag", "write-early/recall-late"),
        ("update_desk_height", "update"),
        ("remove_bicycle_lock", "remove"),
        ("no_fabrication_cabin_code", "no-fabrication"),
    ],
)
async def test_run_case_observes_recall_mutations_and_no_fabrication(
    kind: BackendKind,
    case_id: str,
    expected_category: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(case_id)
    provisioned = await provision_backend(
        kind,
        run="memory-benchmark",
        case=case.id,
        sample=0,
        seed_entries=case.seed_entries,
    )

    try:
        _install_isolated_file_search(provisioned, monkeypatch)
        [observation] = await run_case(case, provisioned)

        assert case.category == expected_category
        if case.category in {"write-early/recall-late", "update"}:
            assert observation.expected_hit
            assert all(observation.expected_hit.values())
        if case.category in {"update", "remove", "no-fabrication"}:
            assert observation.must_not_leak
            assert not any(observation.must_not_leak.values())
        if case.category == "no-fabrication":
            assert observation.surfaced == []
            assert observation.surfaced_content == []
            assert observation.expected_hit == {}
    finally:
        await provisioned.teardown()


async def test_run_case_selects_specific_entry_among_distractors() -> None:
    case = _case("distractor_airport_sign")
    provisioned = await provision_backend(
        "catalog",
        run="memory-benchmark",
        case=case.id,
        sample=0,
        seed_entries=case.seed_entries,
    )

    try:
        [observation] = await run_case(case, provisioned)

        assert observation.surfaced == ["bob_airport_pickup.md"]
        assert observation.expected_hit == {"GOLDEN MAPLE": True}
        assert observation.must_not_leak == {
            "SILVER PINE": False,
            "COPPER BIRCH": False,
        }
    finally:
        await provisioned.teardown()


async def test_cases_use_independent_provisioned_backends() -> None:
    first_case = _case("recall_luggage_tag")
    second_case = _case("recall_veterinarian_pin")
    first = await provision_backend(
        "catalog",
        run="memory-isolation",
        case=first_case.id,
        sample=0,
        seed_entries=first_case.seed_entries,
    )
    second = await provision_backend(
        "catalog",
        run="memory-isolation",
        case=second_case.id,
        sample=0,
        seed_entries=second_case.seed_entries,
    )

    try:
        [first_observation] = await run_case(first_case, first)
        [second_observation] = await run_case(second_case, second)

        assert first.workspace != second.workspace
        assert first_observation.expected_hit == {"ORCHID-482": True}
        assert second_observation.expected_hit == {"LUMEN-93": True}
        assert await first.backend.search("LUMEN", 8) == []
        assert await second.backend.search("ORCHID", 8) == []
        assert not any("LUMEN-93" in entry.content for entry in await first.backend.list())
        assert not any("ORCHID-482" in entry.content for entry in await second.backend.list())
    finally:
        await first.teardown()
        await second.teardown()


def _case(case_id: str) -> MemoryCase:
    return next(case for case in load_cases(CASES_PATH) if case.id == case_id)


def _install_isolated_file_search(
    provisioned: ProvisionedBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep file-backend tests local instead of using the machine-wide search index."""
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

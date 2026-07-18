"""Multi-turn cases and runner for focused memory-recall evaluations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ohmo.evals.memory.provisioning import ProvisionedBackend

TurnKind = Literal["add", "update", "remove", "query"]


@dataclass(frozen=True)
class Turn:
    """One write or recall operation in a memory benchmark case."""

    kind: TurnKind
    title: str | None = None
    content: str | None = None
    name: str | None = None
    query: str | None = None
    expected_recall: list[str] = field(default_factory=list)
    must_not_recall: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryCase:
    """An isolated initial memory state and its ordered turn script."""

    id: str
    category: str
    seed_entries: list[tuple[str, str]]
    turns: list[Turn]


@dataclass(frozen=True)
class QueryObservation:
    """The entries and assertion phrases observed at one query turn."""

    query: str
    surfaced: list[str]
    surfaced_content: list[str]
    expected_hit: dict[str, bool]
    must_not_leak: dict[str, bool]


def load_cases(path: str | Path) -> list[MemoryCase]:
    """Load and validate newline-delimited memory benchmark cases from ``path``."""
    source = Path(path)
    cases: list[MemoryCase] = []
    seen_ids: set[str] = set()

    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {error.msg}") from error
            try:
                case = _parse_case(payload)
            except ValueError as error:
                raise ValueError(f"{source}:{line_number}: invalid memory case: {error}") from error
            if case.id in seen_ids:
                raise ValueError(
                    f"{source}:{line_number}: invalid memory case: duplicate id {case.id!r}"
                )
            seen_ids.add(case.id)
            cases.append(case)

    return cases


async def run_case(
    case: MemoryCase,
    provisioned: ProvisionedBackend,
    *,
    top_k: int = 8,
) -> list[QueryObservation]:
    """Apply ``case`` in order and record backend recall at every query turn."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    backend = provisioned.backend
    observations: list[QueryObservation] = []
    for turn_number, turn in enumerate(case.turns, start=1):
        if turn.kind == "add":
            result = await backend.add(
                _turn_text(case, turn_number, "title", turn.title),
                _turn_text(case, turn_number, "content", turn.content),
            )
            _require_write_success(case, turn_number, turn.kind, result.ok, result.message)
        elif turn.kind == "update":
            result = await backend.update(
                _turn_text(case, turn_number, "name", turn.name),
                _turn_text(case, turn_number, "content", turn.content),
            )
            _require_write_success(case, turn_number, turn.kind, result.ok, result.message)
        elif turn.kind == "remove":
            result = await backend.remove(_turn_text(case, turn_number, "name", turn.name))
            _require_write_success(case, turn_number, turn.kind, result.ok, result.message)
        elif turn.kind == "query":
            query = _turn_text(case, turn_number, "query", turn.query)
            hits = await backend.search(query, top_k)
            surfaced = [hit.name for hit in hits]
            surfaced_content: list[str] = []
            for hit in hits:
                entry = await backend.get(hit.name)
                if entry is not None:
                    surfaced_content.append(entry.content)

            observations.append(
                QueryObservation(
                    query=query,
                    surfaced=surfaced,
                    surfaced_content=surfaced_content,
                    expected_hit={
                        phrase: _is_surfaced(phrase, surfaced_content)
                        for phrase in turn.expected_recall
                    },
                    must_not_leak={
                        phrase: _is_surfaced(phrase, surfaced_content)
                        for phrase in turn.must_not_recall
                    },
                )
            )
        else:
            raise ValueError(
                f"memory case {case.id!r} turn {turn_number} has invalid kind {turn.kind!r}"
            )

    return observations


def _parse_case(payload: object) -> MemoryCase:
    data = _object(payload, "case")
    _keys(data, required={"id", "category", "seed_entries", "turns"}, where="case")
    case_id = _text(data["id"], "case.id")
    category = _text(data["category"], "case.category")

    raw_seeds = _list(data["seed_entries"], "case.seed_entries")
    seed_entries: list[tuple[str, str]] = []
    for index, raw_seed in enumerate(raw_seeds):
        where = f"case.seed_entries[{index}]"
        if not isinstance(raw_seed, list) or len(raw_seed) != 2:
            raise ValueError(f"{where} must be a [title, content] pair")
        seed_entries.append((_text(raw_seed[0], f"{where}[0]"), _text(raw_seed[1], f"{where}[1]")))

    raw_turns = _list(data["turns"], "case.turns")
    if not raw_turns:
        raise ValueError("case.turns must contain at least one turn")
    turns = [_parse_turn(raw_turn, index) for index, raw_turn in enumerate(raw_turns)]
    return MemoryCase(
        id=case_id,
        category=category,
        seed_entries=seed_entries,
        turns=turns,
    )


def _parse_turn(payload: object, index: int) -> Turn:
    where = f"case.turns[{index}]"
    data = _object(payload, where)
    kind = data.get("kind")
    if kind not in {"add", "update", "remove", "query"}:
        raise ValueError(f"{where}.kind must be one of add, update, remove, query")

    required_by_kind = {
        "add": {"kind", "title", "content"},
        "update": {"kind", "name", "content"},
        "remove": {"kind", "name"},
        "query": {"kind", "query", "expected_recall", "must_not_recall"},
    }
    required = required_by_kind[kind]
    _keys(data, required=required, where=where)

    if kind == "add":
        return Turn(
            kind="add",
            title=_text(data["title"], f"{where}.title"),
            content=_text(data["content"], f"{where}.content"),
        )
    if kind == "update":
        return Turn(
            kind="update",
            name=_text(data["name"], f"{where}.name"),
            content=_text(data["content"], f"{where}.content"),
        )
    if kind == "remove":
        return Turn(kind="remove", name=_text(data["name"], f"{where}.name"))
    return Turn(
        kind="query",
        query=_text(data["query"], f"{where}.query"),
        expected_recall=_string_list(data["expected_recall"], f"{where}.expected_recall"),
        must_not_recall=_string_list(data["must_not_recall"], f"{where}.must_not_recall"),
    )


def _object(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{where} must be an object")
    return value


def _list(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a list")
    return value


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _string_list(value: object, where: str) -> list[str]:
    values = _list(value, where)
    return [_text(item, f"{where}[{index}]") for index, item in enumerate(values)]


def _keys(data: dict[str, object], *, required: set[str], where: str) -> None:
    missing = required - data.keys()
    if missing:
        raise ValueError(f"{where} is missing field(s): {', '.join(sorted(missing))}")
    unexpected = data.keys() - required
    if unexpected:
        raise ValueError(f"{where} has unexpected field(s): {', '.join(sorted(unexpected))}")


def _turn_text(
    case: MemoryCase,
    turn_number: int,
    field_name: str,
    value: str | None,
) -> str:
    if value is None or not value.strip():
        raise ValueError(
            f"memory case {case.id!r} turn {turn_number} requires non-empty {field_name}"
        )
    return value


def _require_write_success(
    case: MemoryCase,
    turn_number: int,
    kind: TurnKind,
    ok: bool,
    message: str,
) -> None:
    if not ok:
        raise RuntimeError(f"memory case {case.id!r} turn {turn_number} {kind} failed: {message}")


def _is_surfaced(phrase: str, contents: list[str]) -> bool:
    return any(phrase in content for content in contents)


__all__ = ["MemoryCase", "QueryObservation", "Turn", "load_cases", "run_case"]

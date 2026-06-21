from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from openharness.evals import (
    EVAL_EXECUTION_SCORERS,
    EvalExecutorResult,
    EvalResource,
    EvalResourceSnapshot,
    EvalStore,
    StateOracleV1,
    compute_episode_state_delta,
    compute_state_delta,
    extract_state_keys,
    read_world_snapshots,
    resolve_execution_scorer,
)


def test_compute_state_delta_reports_added_keys_and_noop():
    before = extract_state_keys(
        _snapshot(
            "ep-1",
            reminders=("aaaaaaaaaaaaaaaa",),
            status_counts={"pending": 1},
        )
    )
    after = extract_state_keys(
        _snapshot(
            "ep-1",
            reminders=("aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"),
            status_counts={"pending": 2},
        )
    )

    delta = compute_state_delta(before, after)

    assert delta["changed"] is True
    assert delta["reminders"]["added_keys"] == ["bbbbbbbbbbbbbbbb"]
    assert delta["reminders"]["removed_keys"] == []
    assert delta["reminders"]["count_before"] == 1
    assert delta["reminders"]["count_after"] == 2
    assert delta["reminders"]["status_counts_before"] == {"pending": 1}
    assert delta["reminders"]["status_counts_after"] == {"pending": 2}
    assert delta["memory"]["added_keys"] == []
    assert delta["todos"]["added_keys"] == []

    noop_delta = compute_state_delta(before, before)

    assert noop_delta["changed"] is False
    assert noop_delta["reminders"]["added_keys"] == []
    assert noop_delta["reminders"]["removed_keys"] == []


def test_read_world_snapshots_returns_none_for_absent_files(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")

    assert read_world_snapshots(store, "missing-episode") == (None, None)


def test_state_oracle_passes_matching_captured_delta(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _write_snapshot(
        store,
        "ep-1",
        "world_before",
        _snapshot("ep-1", reminders=("aaaaaaaaaaaaaaaa",)),
    )
    _write_snapshot(
        store,
        "ep-1",
        "world_after",
        _snapshot("ep-1", reminders=("aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb")),
    )
    expected = compute_episode_state_delta(store, "ep-1")

    out = StateOracleV1().score(
        context=_oracle_context(store, "ep-1", {"state_delta": expected}),
        executor_result=EvalExecutorResult(),
    )

    assert out.passed is True
    assert out.score == 1.0
    assert out.scorer_name == "state_oracle_v1"
    assert out.metadata["check.world_after_captured"] is True
    assert out.metadata["check.state_changed"] is True
    assert out.metadata["check.state_delta_matches_gold"] is True
    assert out.metadata["observed_added_key_count"] == 1
    assert out.metadata["observed_delta"] == expected
    assert out.metadata["expected_delta"] == expected


def test_state_oracle_fails_when_world_after_missing(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _write_snapshot(
        store,
        "ep-1",
        "world_before",
        _snapshot("ep-1", reminders=("aaaaaaaaaaaaaaaa",)),
    )

    out = StateOracleV1().score(
        context=_oracle_context(store, "ep-1", {}),
        executor_result=EvalExecutorResult(),
    )

    assert out.passed is False
    assert out.metadata["check.world_after_captured"] is False
    assert out.metadata["check.state_changed"] is False
    assert out.metadata["check.state_delta_matches_gold"] is True
    assert out.metadata["observed_delta"] is None


def test_state_oracle_fails_when_captured_state_did_not_change(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    unchanged_snapshot = _snapshot("ep-1", reminders=("aaaaaaaaaaaaaaaa",))
    _write_snapshot(store, "ep-1", "world_before", unchanged_snapshot)
    _write_snapshot(store, "ep-1", "world_after", unchanged_snapshot)
    claimed_delta = compute_state_delta(
        extract_state_keys(unchanged_snapshot),
        extract_state_keys(
            _snapshot(
                "ep-1",
                reminders=("aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"),
            )
        ),
    )

    out = StateOracleV1().score(
        context=_oracle_context(store, "ep-1", {"state_delta": claimed_delta}),
        executor_result=EvalExecutorResult(),
    )

    assert out.passed is False
    assert out.metadata["check.world_after_captured"] is True
    assert out.metadata["check.state_changed"] is False
    assert out.metadata["check.state_delta_matches_gold"] is False


def test_state_oracle_is_registered_without_exact_tool_sequence_requirement():
    scorer = resolve_execution_scorer("state_oracle_v1")

    assert scorer is EVAL_EXECUTION_SCORERS["state_oracle_v1"]
    assert isinstance(scorer, StateOracleV1)
    assert scorer.requires_exact_tool_sequence is False


def _oracle_context(
    store: EvalStore,
    episode_id: str,
    metadata: dict[str, object],
) -> SimpleNamespace:
    return SimpleNamespace(
        store=store,
        episode=SimpleNamespace(episode_id=episode_id),
        case=SimpleNamespace(metadata=metadata),
    )


def _write_snapshot(
    store: EvalStore,
    episode_id: str,
    phase: str,
    snapshot: EvalResourceSnapshot,
) -> None:
    path = store.root / "states" / episode_id / f"{phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(snapshot.model_dump_json(), encoding="utf-8")


def _snapshot(
    episode_id: str,
    *,
    reminders: tuple[str, ...] | None = (),
    memory: tuple[str, ...] | None = (),
    todos: tuple[str, ...] | None = (),
    status_counts: dict[str, int] | None = None,
) -> EvalResourceSnapshot:
    resources = []
    if reminders is not None:
        resources.append(
            _resource(
                "reminders_json",
                entry_keys=reminders,
                count_key="record_count",
                status_counts=status_counts or {"pending": len(reminders)},
            )
        )
    if memory is not None:
        resources.append(
            _resource("memory_dir", entry_keys=memory, count_key="entry_count")
        )
    if todos is not None:
        resources.append(_resource("todos_dir", entry_keys=todos, count_key="entry_count"))
    return EvalResourceSnapshot(episode_id=episode_id, resources=resources)


def _resource(
    name: str,
    *,
    entry_keys: tuple[str, ...],
    count_key: str,
    status_counts: dict[str, int] | None = None,
) -> EvalResource:
    metadata = {
        "entry_keys": list(entry_keys),
        count_key: len(entry_keys),
    }
    if status_counts is not None:
        metadata["status_counts"] = status_counts
    return EvalResource(
        resource_id=f"resource:{name}",
        kind="local_file" if name == "reminders_json" else "local_directory",
        name=name,
        exists=True,
        metadata=metadata,
    )

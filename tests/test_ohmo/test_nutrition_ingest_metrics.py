from __future__ import annotations

import pytest

from ohmo.nutrition_ingest.metrics import NutritionMetrics


def test_nutrition_metrics_are_aggregate_and_bounded() -> None:
    metrics = NutritionMetrics()
    metrics.verified_candidate()
    metrics.confirmation("accepted")
    metrics.retry("estimation", "transport")
    metrics.dead_letter("prompt", "acknowledgement")
    metrics.delivery_unknown()
    metrics.duplicate_suppression("prompt")
    metrics.duplicate_suppression("observation")
    metrics.pending_latency(2.5)
    metrics.end_to_end_latency(4.0)

    snapshot = metrics.snapshot()
    assert snapshot["pending_latency"] == {"count": 1, "sum_seconds": 2.5, "max_seconds": 2.5}
    assert snapshot["end_to_end_latency"] == {"count": 1, "sum_seconds": 4.0, "max_seconds": 4.0}
    rendered = repr(snapshot)
    for forbidden_value in (
        "dropbox-camera-v1-secret",
        "/private/camera/photo.jpg",
        "reply prose",
        "@private-user",
        "123456789",
    ):
        assert forbidden_value not in rendered


@pytest.mark.parametrize(
    ("method", "value"),
    [
        ("confirmation", "reply-content"),
        ("duplicate_suppression", "candidate-id"),
        ("retry", "candidate-id"),
    ],
)
def test_nutrition_metrics_reject_unbounded_labels(method: str, value: str) -> None:
    metrics = NutritionMetrics()
    with pytest.raises(ValueError):
        if method == "retry":
            metrics.retry(value, "runtime")
        else:
            getattr(metrics, method)(value)


def test_nutrition_latency_summaries_have_fixed_size_after_many_observations() -> None:
    metrics = NutritionMetrics()
    for index in range(10_000):
        metrics.pending_latency(index / 10)
        metrics.end_to_end_latency(index / 5)

    snapshot = metrics.snapshot()
    assert snapshot["pending_latency"] == {
        "count": 10_000,
        "sum_seconds": pytest.approx(4_999_500.0),
        "max_seconds": 999.9,
    }
    assert snapshot["end_to_end_latency"] == {
        "count": 10_000,
        "sum_seconds": pytest.approx(9_999_000.0),
        "max_seconds": 1_999.8,
    }
    assert "sample" not in repr(snapshot).lower()


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), float("-inf"), -1, "1"])
def test_nutrition_metrics_reject_invalid_latency(value: object) -> None:
    metrics = NutritionMetrics()
    with pytest.raises(ValueError):
        metrics.pending_latency(value)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Stage latency summaries
# ---------------------------------------------------------------------------

_STAGES = (
    "publish_to_receipt",
    "pending_confirmation",
    "confirmation_to_estimation",
    "publish_to_terminal",
)


def test_stage_latency_records_bounded_summaries() -> None:
    metrics = NutritionMetrics()
    metrics.stage_latency("publish_to_receipt", 2.0)
    metrics.stage_latency("publish_to_receipt", 4.0)
    metrics.stage_latency("pending_confirmation", 122.0)
    metrics.stage_latency("confirmation_to_estimation", 51.0)
    metrics.stage_latency("publish_to_terminal", 175.0)

    snapshot = metrics.snapshot()
    stage = snapshot["stage_latencies"]
    assert stage["publish_to_receipt"] == {
        "count": 2,
        "sum_seconds": 6.0,
        "max_seconds": 4.0,
    }
    assert stage["pending_confirmation"] == {
        "count": 1,
        "sum_seconds": 122.0,
        "max_seconds": 122.0,
    }
    assert stage["confirmation_to_estimation"] == {
        "count": 1,
        "sum_seconds": 51.0,
        "max_seconds": 51.0,
    }
    assert stage["publish_to_terminal"] == {
        "count": 1,
        "sum_seconds": 175.0,
        "max_seconds": 175.0,
    }


def test_stage_latency_rejects_unknown_stage() -> None:
    metrics = NutritionMetrics()
    with pytest.raises(ValueError):
        metrics.stage_latency("candidate-secret", 1.0)
    with pytest.raises(ValueError):
        metrics.stage_latency("", 1.0)


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), -1, "1"])
def test_stage_latency_rejects_invalid_value(value: object) -> None:
    metrics = NutritionMetrics()
    with pytest.raises(ValueError):
        metrics.stage_latency("publish_to_receipt", value)  # type: ignore[arg-type]


def test_stage_latency_snapshot_excludes_unknown_keys() -> None:
    metrics = NutritionMetrics()
    metrics.stage_latency("publish_to_receipt", 1.0)
    snapshot = metrics.snapshot()
    assert set(snapshot["stage_latencies"]).issubset(set(_STAGES))


# ---------------------------------------------------------------------------
# State / alert gauges
# ---------------------------------------------------------------------------

def test_set_state_gauges_replaces_wholesale() -> None:
    metrics = NutritionMetrics()
    metrics.set_state_gauges(
        {"pending_confirmation": 1, "completed": 2},
        {"prompt_without_receipt": 0},
    )
    assert metrics.snapshot()["state_gauges"] == {
        "completed": 2,
        "pending_confirmation": 1,
    }
    assert metrics.snapshot()["alert_gauges"] == {"prompt_without_receipt": 0}

    # Second call replaces, does not accumulate.
    metrics.set_state_gauges({"delivery_unknown": 1}, {"dead_letter": 1})
    assert metrics.snapshot()["state_gauges"] == {"delivery_unknown": 1}
    assert metrics.snapshot()["alert_gauges"] == {"dead_letter": 1}


def test_set_state_gauges_rejects_unknown_keys() -> None:
    metrics = NutritionMetrics()
    with pytest.raises(ValueError):
        metrics.set_state_gauges({"candidate-secret": 1}, {})
    with pytest.raises(ValueError):
        metrics.set_state_gauges({}, {"free-form-alert": 1})


@pytest.mark.parametrize("value", [True, -1, "1", 1.5])
def test_set_state_gauges_rejects_invalid_counts(value: object) -> None:
    metrics = NutritionMetrics()
    with pytest.raises(ValueError):
        metrics.set_state_gauges({"pending_confirmation": value}, {})  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        metrics.set_state_gauges({}, {"dead_letter": value})  # type: ignore[dict-item]


def test_snapshot_privacy_assertions() -> None:
    """No identifier or free-form text may appear anywhere in the snapshot."""
    metrics = NutritionMetrics()
    metrics.stage_latency("publish_to_receipt", 2.0)
    metrics.set_state_gauges({"pending_confirmation": 1}, {})

    rendered = repr(metrics.snapshot())
    for forbidden in (
        "dropbox-camera-v1-",
        "/private/camera/",
        "@private-user",
        "123456789",
        "reply prose",
        "candidate-secret",
        "free-form",
    ):
        assert forbidden not in rendered

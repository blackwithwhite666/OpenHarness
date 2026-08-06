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

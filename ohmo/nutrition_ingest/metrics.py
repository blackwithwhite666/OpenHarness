"""Bounded, privacy-safe nutrition-ingest observability.

This module deliberately stores only aggregate event dimensions.  Candidate
identity, filesystem names, message content, and recipient bindings never
become metric labels.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from numbers import Real

_EVENTS = frozenset(
    {
        "verified_candidate",
        "confirmation",
        "retry",
        "dead_letter",
        "delivery_unknown",
        "duplicate_suppression",
    }
)
_STAGES = frozenset({"scan", "dedup", "prompt", "estimation", "observation"})
_ERROR_CLASSES = frozenset({"", "runtime", "integrity", "transport", "acknowledgement"})
_CONFIRMATION_OUTCOMES = frozenset({"accepted", "declined", "non_food"})
_DUPLICATE_OPERATIONS = frozenset({"dedup", "prompt", "observation"})

# Fixed, low-cardinality latency stage names.  Adding a new stage here is the
# only way to observe a latency summary; arbitrary strings are rejected.
_LATENCY_STAGES = frozenset(
    {
        "publish_to_receipt",
        "pending_confirmation",
        "confirmation_to_estimation",
        "publish_to_terminal",
    }
)

# Fixed state and alert-condition gauge keys.  These prevent arbitrary
# identifiers from leaking through the gauge interface.  The state key set
# mirrors the full ``ResultState`` enum so every reachable sidecar state is
# representable.
_STATE_GAUGE_KEYS = frozenset(
    {
        "discovered",
        "classified",
        "published",
        "prompt_sending",
        "delivery_unknown",
        "pending_confirmation",
        "confirmed",
        "declined",
        "non_food",
        "estimated",
        "completed",
        "retryable_error",
        "dead_letter",
        "skipped",
        "seen",
    }
)
_ALERT_GAUGE_KEYS = frozenset(
    {
        "prompt_without_receipt",
        "pending_confirmation_over_slo",
        "estimation_over_slo",
        "delivery_unknown",
        "retryable_error",
        "dead_letter",
    }
)


@dataclass(slots=True)
class _LatencySummary:
    count: int = 0
    sum_seconds: float = 0.0
    max_seconds: float = 0.0

    def observe(self, seconds: Real) -> None:
        if isinstance(seconds, bool) or not isinstance(seconds, Real):
            raise ValueError("nutrition latency must be a finite real number")
        value = float(seconds)
        if not math.isfinite(value) or value < 0:
            raise ValueError("nutrition latency must be finite and non-negative")
        self.count += 1
        self.sum_seconds += value
        self.max_seconds = max(self.max_seconds, value)

    def snapshot(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "sum_seconds": self.sum_seconds,
            "max_seconds": self.max_seconds,
        }


@dataclass(slots=True)
class NutritionMetrics:
    """Aggregate counters and fixed-memory latency summaries."""

    _events: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    _pending_latency: _LatencySummary = field(default_factory=_LatencySummary)
    _end_to_end_latency: _LatencySummary = field(default_factory=_LatencySummary)
    _stage_latencies: dict[str, _LatencySummary] = field(default_factory=dict)
    _state_gauges: dict[str, int] = field(default_factory=dict)
    _alert_gauges: dict[str, int] = field(default_factory=dict)

    def _record(self, event: str, stage: str = "", error_class: str = "") -> None:
        if event not in _EVENTS:
            raise ValueError("unsupported nutrition metric event")
        if stage not in _STAGES and stage not in _DUPLICATE_OPERATIONS and stage != "":
            raise ValueError("unsupported nutrition metric stage")
        if error_class not in _ERROR_CLASSES and error_class not in _CONFIRMATION_OUTCOMES:
            raise ValueError("unsupported nutrition metric error class")
        self._events[(event, stage, error_class)] += 1

    def verified_candidate(self) -> None:
        self._record("verified_candidate", "scan")

    def confirmation(self, outcome: str) -> None:
        if outcome not in _CONFIRMATION_OUTCOMES:
            raise ValueError("unsupported nutrition confirmation outcome")
        self._record("confirmation", "prompt", outcome)

    def retry(self, stage: str, error_class: str) -> None:
        self._record("retry", stage, error_class)

    def dead_letter(self, stage: str, error_class: str) -> None:
        self._record("dead_letter", stage, error_class)

    def delivery_unknown(self, error_class: str = "acknowledgement") -> None:
        self._record("delivery_unknown", "prompt", error_class)

    def duplicate_suppression(self, operation: str) -> None:
        if operation not in _DUPLICATE_OPERATIONS:
            raise ValueError("unsupported nutrition duplicate operation")
        self._record("duplicate_suppression", operation)

    def pending_latency(self, seconds: float) -> None:
        self._pending_latency.observe(seconds)

    def end_to_end_latency(self, seconds: float) -> None:
        self._end_to_end_latency.observe(seconds)

    def stage_latency(self, stage: str, seconds: float) -> None:
        """Observe a bounded latency for a fixed, named pipeline stage."""
        if stage not in _LATENCY_STAGES:
            raise ValueError("unsupported nutrition latency stage")
        self._stage_latencies.setdefault(stage, _LatencySummary()).observe(seconds)

    def set_state_gauges(
        self,
        state_counts: dict[str, int],
        alert_counts: dict[str, int],
    ) -> None:
        """Replace current-state and alert-condition gauges wholesale.

        Keys must belong to the fixed low-cardinality enums so that no
        candidate identifier, path, or free-form text can leak as a gauge
        label.
        """
        validated_states: dict[str, int] = {}
        for key, value in state_counts.items():
            if key not in _STATE_GAUGE_KEYS:
                raise ValueError("unsupported nutrition state gauge key")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("nutrition state gauge must be a non-negative integer")
            validated_states[key] = value
        validated_alerts: dict[str, int] = {}
        for key, value in alert_counts.items():
            if key not in _ALERT_GAUGE_KEYS:
                raise ValueError("unsupported nutrition alert gauge key")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("nutrition alert gauge must be a non-negative integer")
            validated_alerts[key] = value
        self._state_gauges = validated_states
        self._alert_gauges = validated_alerts

    def snapshot(self) -> dict[str, object]:
        """Return test/debug data containing no unbounded or identifying labels."""
        return {
            "events": {
                f"{event}|{stage}|{error_class}": count
                for (event, stage, error_class), count in sorted(self._events.items())
            },
            "pending_latency": self._pending_latency.snapshot(),
            "end_to_end_latency": self._end_to_end_latency.snapshot(),
            "stage_latencies": {
                stage: self._stage_latencies[stage].snapshot()
                for stage in sorted(self._stage_latencies)
                if stage in _LATENCY_STAGES
            },
            "state_gauges": dict(sorted(self._state_gauges.items())),
            "alert_gauges": dict(sorted(self._alert_gauges.items())),
        }


__all__ = ["NutritionMetrics"]

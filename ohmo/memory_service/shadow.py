"""Comparison logging and reporting for off-path Honcho shadow recall."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeAlias

from ohmo.memory_service.honcho_client import Conclusion

CatalogShadowHit: TypeAlias = tuple[str, int, str]
ShadowRecord: TypeAlias = dict[str, object]

SHADOW_COMPARISON_LOG_FILENAME = "shadow-comparison.jsonl"


def build_shadow_record(
    *,
    query: str,
    catalog_hits: Sequence[CatalogShadowHit],
    honcho_hits: Sequence[Conclusion],
    catalog_latency_ms: float,
    honcho_latency_ms: float,
) -> ShadowRecord:
    """Build one compact, source-neutral comparison record."""
    catalog_rows = [{"name": name, "rank": rank} for name, rank, _snippet in catalog_hits]
    honcho_rows = [{"id": hit.id, "snippet": _snippet(hit.content)} for hit in honcho_hits]
    rank_pairs = _match_rank_pairs(catalog_hits, honcho_hits)
    overlap = len(rank_pairs)
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "catalog_hits": catalog_rows,
        "honcho_hits": honcho_rows,
        "catalog_latency_ms": round(max(0.0, catalog_latency_ms), 3),
        "honcho_latency_ms": round(max(0.0, honcho_latency_ms), 3),
        "overlap": overlap,
        "catalog_only": len(catalog_hits) - overlap,
        "honcho_only": len(honcho_hits) - overlap,
        "rank_corr": _rank_correlation(rank_pairs),
    }


def append_shadow_record(path: str | Path, record: Mapping[str, object]) -> None:
    """Append one JSONL record, creating its memory directory when needed."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with resolved.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def shadow_report(path: str | Path) -> str:
    """Render aggregate completeness, ranking, and latency from a shadow log."""
    records = _read_records(Path(path))
    if not records:
        return "# Shadow memory comparison\n\nNo comparison records."

    catalog_latencies = _numbers(records, "catalog_latency_ms")
    honcho_latencies = _numbers(records, "honcho_latency_ms")
    rank_correlations = _numbers(records, "rank_corr")

    overlap_rates: list[float] = []
    completeness_rates: list[float] = []
    catalog_total = 0
    honcho_total = 0
    catalog_only_total = 0.0
    honcho_only_total = 0.0
    for record in records:
        catalog_count = _list_length(record.get("catalog_hits"))
        honcho_count = _list_length(record.get("honcho_hits"))
        overlap = min(
            _number(record.get("overlap"), default=0.0),
            float(catalog_count),
            float(honcho_count),
        )
        union = catalog_count + honcho_count - overlap
        if union > 0:
            overlap_rates.append(overlap / union)
        if catalog_count > 0:
            completeness_rates.append(overlap / catalog_count)
        catalog_total += catalog_count
        honcho_total += honcho_count
        catalog_only_total += _number(
            record.get("catalog_only"),
            default=max(0.0, catalog_count - overlap),
        )
        honcho_only_total += _number(
            record.get("honcho_only"),
            default=max(0.0, honcho_count - overlap),
        )

    catalog_only_rate = catalog_only_total / catalog_total if catalog_total else 0.0
    honcho_only_rate = honcho_only_total / honcho_total if honcho_total else 0.0
    return "\n".join(
        (
            "# Shadow memory comparison",
            "",
            f"- Records: {len(records)}",
            f"- Average catalog latency: {_mean(catalog_latencies):.3f} ms",
            f"- Average Honcho latency: {_mean(honcho_latencies):.3f} ms",
            f"- Mean overlap: {_mean(overlap_rates):.1%}",
            f"- Mean completeness: {_mean(completeness_rates):.1%}",
            f"- Catalog-only rate: {catalog_only_rate:.1%}",
            f"- Honcho-only rate: {honcho_only_rate:.1%}",
            f"- Mean rank correlation: {_format_correlation(rank_correlations)}",
        )
    )


def _match_rank_pairs(
    catalog_hits: Sequence[CatalogShadowHit],
    honcho_hits: Sequence[Conclusion],
) -> list[tuple[int, int]]:
    honcho_ranks: dict[str, list[int]] = {}
    for rank, hit in enumerate(honcho_hits, start=1):
        honcho_ranks.setdefault(_normalise(_snippet(hit.content)), []).append(rank)

    pairs: list[tuple[int, int]] = []
    for _name, catalog_rank, snippet in catalog_hits:
        ranks = honcho_ranks.get(_normalise(snippet))
        if ranks:
            pairs.append((catalog_rank, ranks.pop(0)))
    return pairs


def _rank_correlation(pairs: Sequence[tuple[int, int]]) -> float | None:
    if not pairs:
        return None
    if len(pairs) == 1:
        return 1.0
    left = [float(pair[0]) for pair in pairs]
    right = [float(pair[1]) for pair in pairs]
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return 1.0 if left == right else 0.0
    return round(numerator / (left_scale * right_scale), 6)


def _normalise(value: str) -> str:
    return " ".join(value.split()).casefold()


def _snippet(value: str) -> str:
    compact = " ".join(value.split())
    if len(compact) > 240:
        return compact[:237].rstrip() + "..."
    return compact


def _read_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _numbers(records: Sequence[Mapping[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                values.append(number)
    return values


def _number(value: object, *, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return max(0.0, number)
    return default


def _list_length(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _format_correlation(values: Sequence[float]) -> str:
    return f"{_mean(values):.3f}" if values else "n/a"

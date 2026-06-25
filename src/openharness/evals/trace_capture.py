"""Raw eval execution trace artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openharness.utils.fs import atomic_write_text

_DISABLED_VALUES = {"0", "false", "no", "off"}


def write_eval_trace(
    traces_root: Path,
    run_id: str,
    case_id: str,
    sample_index: int,
    *,
    context: Any,
    executor_result: Any,
    scorer_result: Any,
) -> Path | None:
    """Write the raw per-sample eval trace, unless disabled by environment."""
    if _trace_capture_disabled():
        return None

    payload = {
        "case_id": case_id,
        "sample_index": sample_index,
        "prompt": context.primary_prompt,
        "final_text": executor_result.final_text,
        "tool_calls": [
            {
                "tool_name": call.tool_name,
                "input": call.arguments,
                "output": call.output,
                "is_error": call.is_error,
                "started_ms": call.started_ms,
                "ended_ms": call.ended_ms,
            }
            for call in executor_result.tool_calls
        ],
        "judge": {
            "verdict": scorer_result.metadata.get("verdict"),
            "reason": getattr(scorer_result, "raw_reason", None),
        },
        "score": scorer_result.score,
        "passed": scorer_result.passed,
    }
    path = traces_root / run_id / f"{case_id}-{sample_index}.json"
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    return path


def _trace_capture_disabled() -> bool:
    value = os.environ.get("OHMO_EVALS_TRACE_CAPTURE")
    return value is not None and value.strip().lower() in _DISABLED_VALUES

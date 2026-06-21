"""Shared replay fixture argument matching helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def _fixture_input_key(args: Mapping[str, Any] | object) -> str:
    """Hash normalized exact tool args; relaxed/subset policies are future tunables."""
    if not isinstance(args, Mapping):
        return ""
    payload = _canonical_json(_normalize_args(args))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _normalize_args(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_args(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
            if item is not None
        }
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_args(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))

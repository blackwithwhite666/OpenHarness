"""Bounded, offline intake for reviewed native-subscription Padavan results.

The supplied provenance is an assertion, not service authentication. A lead must
compare every session and turn with the actual Padavan result before intake.
"""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ohmo.evals.camera_calibration import A2Vote, Case, JudgeCase, Reference

MAX_PRIVATE_JSON_BYTES = 1_000_000
REPOSITORY = Path(__file__).resolve().parents[2]


def read_private_json(path: Path) -> Any:
    resolved = path.resolve(strict=True)
    if resolved.is_relative_to(REPOSITORY):
        raise ValueError("private input must be outside the repository")
    if not resolved.is_file() or stat.S_IMODE(resolved.stat().st_mode) != 0o600:
        raise ValueError("private input must be a regular mode 0600 file")
    with resolved.open("rb") as source:
        data = source.read(MAX_PRIVATE_JSON_BYTES + 1)
    if len(data) > MAX_PRIVATE_JSON_BYTES:
        raise ValueError("private input exceeds 1 MB")
    try:
        return json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("private input is not valid JSON") from exc


class SubscriptionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Literal["native_subscription_padavan"]
    case_id: str = Field(min_length=1, max_length=128)
    model: Literal["openai/gpt-6.1-sol", "openai/gpt-6-luna"]
    reasoning_effort: Literal["high", "medium"]
    prompt: str = Field(min_length=1, max_length=16_000)
    source_image_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    padavan_session_id: str = Field(min_length=1, max_length=256)
    padavan_turn_id: str = Field(min_length=1, max_length=256)
    response_json: str = Field(min_length=2, max_length=16_000)


class SubscriptionResults:
    def __init__(self, data: Any, cases: list[Case | JudgeCase]):
        if not isinstance(data, list) or not 1 <= len(data) <= 20:
            raise ValueError("subscription results must contain 1..20 entries")
        try:
            self.results = [SubscriptionResult.model_validate(item) for item in data]
        except ValidationError:
            raise ValueError("subscription result schema invalid") from None
        self.cases = cases
        self.index = 0
        self.case_index = -1
        self.luna_sessions: set[str] = set()
        provenances: set[tuple[str, str]] = set()
        for item in self.results:
            key = (item.padavan_session_id, item.padavan_turn_id)
            if key in provenances:
                raise ValueError("reused Padavan session/turn provenance")
            provenances.add(key)
            try:
                if item.model == "openai/gpt-6.1-sol":
                    Reference.model_validate_json(item.response_json)
                else:
                    A2Vote.model_validate_json(item.response_json)
            except ValidationError:
                raise ValueError("subscription response JSON or schema invalid") from None

    async def call(self, model: str, effort: str, prompt: str, image: bytes | None) -> str:
        if self.index >= len(self.results):
            raise ValueError("missing subscription result")
        item = self.results[self.index]
        if model == "openai/gpt-6.1-sol":
            self.case_index += 1
            self.luna_sessions.clear()
        if self.case_index >= len(self.cases):
            raise ValueError("subscription result has no matching case")
        case = self.cases[self.case_index]
        sha = hashlib.sha256(image).hexdigest() if image is not None else None
        if (
            item.case_id != case.case_id
            or item.model != model
            or item.reasoning_effort != effort
            or item.prompt != prompt
            or item.source_image_sha256 != sha
        ):
            raise ValueError("subscription result does not match exact requested call")
        if model == "openai/gpt-6-luna":
            if item.padavan_session_id in self.luna_sessions:
                raise ValueError("Luna votes require distinct Padavan sessions per case")
            self.luna_sessions.add(item.padavan_session_id)
        self.index += 1
        return item.response_json

    def assert_exhausted(self) -> None:
        if self.index != len(self.results):
            raise ValueError("extra unused subscription results")

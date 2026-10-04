from __future__ import annotations

import asyncio
import copy
import json
import re

import pytest

from openharness.evals import DecisionTraceValidationError
from openharness.evals.decision_trace import validate_decision_trace_payload
from openharness.engine.query import _offload_tool_output_if_needed
from openharness.skills.bundled import get_bundled_skills
from openharness.skills.registry import SkillRegistry
from openharness.tools.base import ToolExecutionContext
from openharness.tools.skill_tool import SkillTool, SkillToolInput
from openharness.tools.trace_tool import TraceToolInput
from ohmo.evals.nutrition_trace import validate_trace_finalization_annotations


def test_bundled_calory_trace_example_survives_default_engine_preview(tmp_path, monkeypatch):
    for name in (
        "OPENHARNESS_TOOL_OUTPUT_INLINE_CHARS",
        "OPENHARNESS_TOOL_OUTPUT_PREVIEW_CHARS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "openharness-data"))

    def bundled_only_registry(_cwd, **_kwargs):
        registry = SkillRegistry()
        for skill in get_bundled_skills():
            registry.register(skill)
        return registry

    monkeypatch.setattr("openharness.tools.skill_tool.load_skill_registry", bundled_only_registry)
    result = asyncio.run(
        SkillTool().execute(
            SkillToolInput(name="calory"),
            ToolExecutionContext(cwd=tmp_path),
        )
    )
    assert not result.is_error
    assert len(result.output) > 16_000

    inline, artifact_path = _offload_tool_output_if_needed(
        tool_name="skill",
        tool_use_id="synthetic-calory-contract",
        output=result.output,
    )
    assert artifact_path is not None
    assert artifact_path.is_relative_to(tmp_path)
    assert artifact_path.read_text(encoding="utf-8") == result.output
    assert "[Tool output truncated]" in inline
    assert "Inline preview: first 3000 chars" in inline
    preview_match = re.search(r"\n\nPreview:\n(.*)\Z", inline, re.DOTALL)
    assert preview_match is not None
    preview = preview_match.group(1)
    assert preview == result.output[:3000]

    specimen = re.search(r"```json\n(.*?)\n```", preview, re.DOTALL)
    assert specimen is not None, "complete trace-call specimen must fit in the default preview"
    call = json.loads(specimen.group(1))
    trace_call = TraceToolInput.model_validate(call)
    assert trace_call.kind == "trace_finalization"
    validated_payload = validate_decision_trace_payload(trace_call.kind, trace_call.payload)
    assert validated_payload["schema_version"] == 1
    assert validated_payload["trace_event_id"] == "example-trace-event-1"
    validated = validate_trace_finalization_annotations(validated_payload)
    nutrition = validated["annotations"]["nutrition"]
    assert nutrition["schema_version"] == 2
    assert nutrition["record_type"] == "meal_observation"
    assert nutrition["consumption_status"] == "consumed"
    assert nutrition["meal_at"] is None and nutrition["meal_date"] is None
    assert nutrition["items"] == [
        {
            "name": "vegetable soup",
            "quantity_text": "1 bowl",
            "energy_kcal_min": None,
            "energy_kcal_max": None,
            "energy_kcal_best": 185.0,
        }
    ]

    for required_field in ("schema_version", "trace_event_id"):
        missing_field_payload = copy.deepcopy(trace_call.payload)
        missing_field_payload.pop(required_field)
        with pytest.raises(DecisionTraceValidationError):
            validate_decision_trace_payload(trace_call.kind, missing_field_payload)

    invalid_call = json.loads(specimen.group(1))
    invalid_nutrition = invalid_call["payload"]["annotations"]["nutrition"]
    invalid_nutrition.update(
        {
            "food": "vegetable soup",
            "quantity": {"value": 1, "unit": "bowl"},
            "energy_kcal_estimate": 185,
        }
    )
    with pytest.raises(DecisionTraceValidationError):
        validate_trace_finalization_annotations(invalid_call["payload"])

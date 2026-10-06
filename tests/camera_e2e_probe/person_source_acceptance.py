"""Real-storage person-origin acceptance with scripted synthetic transports.

This verifies persistence and projection mechanics only. The fixture does not
classify images or make nutrition judgments.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from openharness.api.client import ApiMessageCompleteEvent, ApiMessageRequest
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from PIL import Image

from probe_support import create_storage_run_dir


_CASES = {
    "send_time_meal": {
        "sent_at": "2026-10-06T08:30:00+00:00",
        "text": "I ate the food in this picture.",
        "owner_scenario": "A synthetic owner says they ate the pictured meal.",
        "expect_meal_at": "2026-10-06T08:30:00+00:00",
        "expected_kcal": 285,
        "replay": True,
    },
    "explicit_time_meal": {
        "sent_at": "2026-10-06T08:30:00+00:00",
        "text": "I ate the pictured meal at 2026-10-05 18:45 UTC.",
        "owner_scenario": "A synthetic owner gives an explicit meal date and time.",
        "expect_meal_at": "2026-10-05T18:45:00+00:00",
        "expected_kcal": 285,
        "replay": False,
    },
    "unclear_image": {
        "sent_at": "2026-10-06T08:30:00+00:00",
        "text": "",
        "owner_scenario": "The synthetic fixture asks whether the owner ate the unclear image.",
        "expect_meal_at": None,
        "expected_kcal": None,
        "replay": False,
    },
    "nonfood_image": {
        "sent_at": "2026-10-06T08:30:00+00:00",
        "text": "This picture is an ordinary household object.",
        "owner_scenario": "The synthetic fixture treats this case as an obvious nonfood input.",
        "expect_meal_at": None,
        "expected_kcal": None,
        "replay": False,
    },
    "estimate_only": {
        "sent_at": "2026-10-06T08:30:00+00:00",
        "text": "Please estimate calories only; I have not eaten this.",
        "owner_scenario": "The synthetic owner requests an estimate and denies consumption.",
        "expect_meal_at": None,
        "expected_kcal": None,
        "replay": False,
    },
}


class SyntheticPersonSourceApi:
    """Deterministic fixture transport that emits fixed trace or no trace."""

    synthetic = True

    def __init__(self, case_name: str) -> None:
        if case_name not in _CASES:
            raise ValueError(f"unknown E1 person-source case: {case_name}")
        self.case_name = case_name
        self.calls = 0
        self.trace_proposed = False
        self.saw_same_photo_context = False

    async def stream_message(self, request: ApiMessageRequest):
        self.calls += 1
        system_prompt = request.system_prompt or ""
        if self.case_name == "send_time_meal" and "Previously seen user photo" in system_prompt:
            self.saw_same_photo_context = True
            yield ApiMessageCompleteEvent(
                message=ConversationMessage.from_user_text(
                    "I will not create another meal record for the same photo."
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                stop_reason="end_turn",
            )
            return

        if self.case_name == "unclear_image":
            text = "Did you eat this? I need your confirmation before recording a meal."
        elif self.case_name == "nonfood_image":
            text = "This is not food, so I will not record a meal."
        elif self.case_name == "estimate_only":
            text = "I can estimate the pictured food, but I will not record consumption."
        elif self.trace_proposed:
            text = "I recorded the synthetic person-origin meal."
        else:
            self.trace_proposed = True
            spec = _CASES[self.case_name]
            nutrition = {
                "schema_version": 2,
                "record_type": "meal_observation",
                "basis": ["image"],
                "consumption_status": "consumed",
                "is_estimate": True,
                "energy_kcal_best": spec["expected_kcal"],
                "items": [{
                    "name": "synthetic fixture meal",
                    "quantity_text": "one synthetic fixture serving",
                    "energy_kcal_best": spec["expected_kcal"],
                }],
                "assumptions": ["fixed synthetic acceptance fixture; not a food judgment"],
            }
            if self.case_name == "explicit_time_meal":
                nutrition["meal_at"] = spec["expect_meal_at"]
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant",
                    content=[ToolUseBlock(
                        name="trace",
                        input={
                            "kind": "trace_finalization",
                            "payload": {
                                "schema_version": 1,
                                "trace_event_id": f"e1-{self.case_name}",
                                "annotations": {"nutrition": nutrition},
                            },
                        },
                    )],
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                stop_reason="tool_use",
            )
            return

        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant", content=[TextBlock(text=text)],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="end_turn",
        )


def _synthetic_photo(root: Path) -> tuple[Path, str]:
    path = root / "tmp/camera-native-docker/person-source-fixtures/e1-food.jpg"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.exists():
        buffer = io.BytesIO()
        Image.new("RGB", (48, 32), (120, 80, 35)).save(buffer, format="JPEG", quality=85)
        path.write_bytes(buffer.getvalue())
        path.chmod(0o600)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_case(name: str, artifact: dict) -> dict:
    spec = _CASES[name]
    honcho = artifact["honcho"]
    after = honcho["after"]
    projection = artifact["projection"]
    new_ids = projection.get("new_nutrition_event_ids", [])
    if honcho["before_status"] != "complete" or honcho["after_status"] != "complete":
        raise AssertionError(f"{name}: full-scope Honcho snapshots are unavailable")

    if spec["expect_meal_at"] is None:
        if artifact["trajectory"]["save"]["saved"] or artifact["trajectory"]["save"]["event_id"]:
            raise AssertionError(f"{name}: runtime claimed a meal without a consumed event")
        if new_ids or projection.get("reopened_records") or projection.get("reopened_current_meals"):
            raise AssertionError(f"{name}: Telegent contains a newly projected meal")
        if projection.get("status") != "complete_absence":
            raise AssertionError(f"{name}: full-read absence was not established")
        positive_claims = ("recorded the meal", "meal was saved", "saved meal", "logged the meal")
        delivered = artifact["trajectory"]["deliveries"]
        if any(
            item.get("event_id")
            or any(claim in item.get("content", "").casefold() for claim in positive_claims)
            for item in delivered
        ):
            raise AssertionError(f"{name}: no-save case made a positive saved-meal claim")
        return {"case": name, "result": "absence_verified", "new_nutrition_event_ids": []}

    event_id = artifact["trajectory"]["save"]["event_id"]
    if (
        not artifact["trajectory"]["save"]["saved"]
        or not event_id
        or projection.get("status") != "projected"
        or projection.get("store_reopened") is not True
        or new_ids != [event_id]
    ):
        raise AssertionError(f"{name}: expected exactly one runtime-receipted meal event")
    record = projection.get("reopened_event") or {}
    meal = projection.get("reopened_effective_meal") or {}
    if record.get("message_id") != event_id or meal.get("latest_event_id") != event_id:
        raise AssertionError(f"{name}: reopened store does not identify the causal event")
    expected_time = datetime.fromisoformat(spec["expect_meal_at"])
    actual_time = datetime.fromisoformat(meal["meal_at"])
    if actual_time != expected_time:
        raise AssertionError(f"{name}: effective meal time does not match trusted semantics")
    expected_day = expected_time.astimezone(ZoneInfo("UTC")).date().isoformat()
    if projection.get("reopened_effective_meal_date") != expected_day:
        raise AssertionError(f"{name}: effective meal date differs from trusted meal time")
    if meal.get("energy_kcal_best") != spec["expected_kcal"]:
        raise AssertionError(f"{name}: effective meal kcal differs from its causal event")
    honcho_event = next((row for row in after if row.get("id") == event_id), None)
    if honcho_event is None:
        raise AssertionError(f"{name}: exact runtime event is absent from the Honcho full read")
    trace = honcho_event.get("metadata", {}).get("decision_trace", {})
    annotations = trace.get("annotations", {}) if isinstance(trace, dict) else {}
    if not isinstance(annotations.get("nutrition"), dict):
        raise AssertionError(f"{name}: exact Honcho event lacks its nutrition finalizer annotation")
    if spec["replay"]:
        replay = artifact.get("replay") or {}
        if (
            not replay.get("same_source")
            or replay.get("save", {}).get("event_id") is not None
            or replay.get("new_nutrition_event_ids")
            or replay.get("event_count_after_replay") != 1
        ):
            raise AssertionError("same-photo/source replay was not stable")
    return {
        "case": name,
        "result": "one_event_projected_reopened",
        "event_id": event_id,
        "meal_at": actual_time.isoformat(),
        "meal_date": expected_day,
        "energy_kcal_best": meal["energy_kcal_best"],
        "new_nutrition_event_ids": new_ids,
    }


async def run_suite() -> dict:
    root = Path(__file__).resolve().parents[2]
    required = (
        "CAMERA_TELEGENT_WORKTREE", "CAMERA_TELEGENT_SHA", "CAMERA_HONCHO_URL",
    )
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise ValueError("E1 suite missing required settings: " + ", ".join(missing))
    if os.environ.get("CAMERA_TELEGENT_SHA") != "71f09310c433d0f7ce32aba6a5e5db5a60ba3425":
        raise ValueError("E1 suite requires the pinned read-only Telegent source")
    photo, digest = _synthetic_photo(root)
    artifact_root = create_storage_run_dir(root)
    initial_environment = dict(os.environ)
    from run_person_source import main as run_one_case

    summaries = []
    artifacts = []
    try:
        for index, (name, spec) in enumerate(_CASES.items(), start=1):
            os.environ.update({
                "CAMERA_RUN_MODE": "offline",
                "CAMERA_PERSON_SOURCE_CASE": name,
                "CAMERA_SOURCE_KIND": "photo",
                "CAMERA_SOURCE_SENDER_ID": "12345",
                "CAMERA_SOURCE_CHAT_ID": "12345",
                "CAMERA_SOURCE_MESSAGE_ID": str(8100 + index),
                "CAMERA_SOURCE_SENT_AT": spec["sent_at"],
                "CAMERA_SOURCE_TEXT": spec["text"],
                "CAMERA_SOURCE_JPEG": str(photo),
                "CAMERA_SOURCE_SHA256": digest,
                "CAMERA_OWNER_ID": "synthetic_person_origin_owner",
                "CAMERA_USER_SCENARIO": spec["owner_scenario"],
            })
            capture = io.StringIO()
            with redirect_stdout(capture):
                await run_one_case()
            lines = [line for line in capture.getvalue().splitlines() if line.startswith("{")]
            if not lines:
                raise AssertionError(f"{name}: person-source runner emitted no result envelope")
            runner_result = json.loads(lines[-1])
            evidence_path = Path(runner_result["artifact"])
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            summaries.append(_validate_case(name, evidence))
            artifacts.append({"case": name, "path": str(evidence_path)})
    finally:
        os.environ.clear()
        os.environ.update(initial_environment)

    result = {
        "scope": "E1 person-origin persistence mechanics with deterministic synthetic transports",
        "classification_or_model_quality_claim": False,
        "docker_image_digest": "sha256:137c608dc941872a5e942e88e564905f0f2952c6449046a9059033251c609f8f",
        "honcho_url": os.environ["CAMERA_HONCHO_URL"],
        "telegent_source": os.environ["CAMERA_TELEGENT_WORKTREE"],
        "telegent_sha": os.environ["CAMERA_TELEGENT_SHA"],
        "synthetic_photo_sha256": digest,
        "cases": summaries,
        "evidence": artifacts,
    }
    output = artifact_root / "person-source-acceptance.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    output.chmod(0o600)
    print(json.dumps({"acceptance_artifact": str(output), "cases": summaries}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    asyncio.run(run_suite())

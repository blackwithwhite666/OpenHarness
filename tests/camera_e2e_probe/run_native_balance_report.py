"""Opt-in ordinary model follow-up over a retained native Camera conversation.

Run only after run_balance_followup has produced facts.json. The model receives
the historical owner dialogue and an ordinary balance question, never expected
totals. Its wellness tool executes the registered Telegent implementation over
the fresh projection and verifies the production signed call arguments.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from math import isclose
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TELEGENT = Path(os.environ["CAMERA_TELEGENT_WORKTREE"]).resolve()
sys.path.insert(0, str(TELEGENT))

from mcp.server.fastmcp.server import FastMCP  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402
from ohmo.memory_service.honcho_client import HonchoClient  # noqa: E402
from openharness.mcp.client import McpToolCallResult  # noqa: E402
from openharness.mcp.types import McpToolInfo  # noqa: E402
from openharness.tools.mcp_tool import McpToolAdapter  # noqa: E402
from telegent.health_advisor.nutrition.store import NutritionDataStore  # noqa: E402
from telegent.health_advisor.storage import HealthDataStore  # noqa: E402
from telegent.mcp_simple_auth.wellness import register_wellness_tools  # noqa: E402
from telegent.mcp_simple_auth.wellness_delegation import (  # noqa: E402
    META_KEY, WellnessDelegationConfig, verify_delegation,
)
from balance_support import assert_fixture_balance  # noqa: E402
from native_balance_config import (  # noqa: E402
    admit_native_wellness_read, resolve_native_report_clients,
)
from native_balance_guards import (  # noqa: E402
    assert_honcho_report_delta, assert_report_only_traces,
    honcho_row_state, projection_state,
)
from probe_support import (  # noqa: E402
    call_wellness_with_synthetic_self, synthetic_wellness_self_scope,
)


async def run(grade_dir: Path, projection_dir: Path, output_dir: Path, config_dir: Path) -> None:
    from ohmo.gateway.bridge import OhmoGatewayBridge
    from ohmo.gateway.config import save_gateway_config
    from ohmo.gateway.models import GatewayConfig
    from ohmo.gateway.runtime import OhmoSessionRuntimePool
    from ohmo.session_storage import _session_key_latest_path
    from openharness.channels.bus.events import InboundMessage
    from openharness.channels.bus.queue import MessageBus
    from openharness.channels.impl.telegram import TelegramChannel
    from openharness.config.schema import TelegramConfig
    from openharness.ui.runtime import build_runtime as original_build_runtime
    from camera_runtime_support import OfflineTelegramBot, isolated_runtime_loaders
    import ohmo.gateway.runtime as gateway_runtime
    import openharness.ui.runtime as openharness_runtime

    permitted = (ROOT / "tmp" / "full-chain-balance").resolve()
    output_dir = output_dir.resolve()
    if permitted not in output_dir.parents or output_dir.exists():
        raise ValueError("native output must be a fresh task-local directory")
    facts = json.loads((projection_dir / "facts.json").read_text())
    if facts.get("status") != "offline_real_honcho_real_telegent" or facts.get("reopen_replay_stable") is not True:
        raise AssertionError("factual projection has not passed")
    manifest = json.loads((grade_dir / "manifest.json").read_text())
    goal, = manifest["goals"]
    if facts["event_id"] != json.loads((grade_dir / "a1-result.json").read_text())["actual_latest_event_id"]:
        raise AssertionError("native event differs from frozen grade")
    original_workspace = Path(goal["eval_workspace"]).resolve()
    snapshot_path = _session_key_latest_path(original_workspace, "telegram:123")
    snapshot_bytes = snapshot_path.read_bytes()
    if not 0 < len(snapshot_bytes) < 64 * 1024 * 1024:
        raise AssertionError("retained dialogue snapshot is unbounded")
    snapshot = json.loads(snapshot_bytes)
    if (snapshot.get("session_key") != "telegram:123"
            or snapshot.get("session_id") != goal["gateway_session_id"]
            or facts["event_id"] not in snapshot_bytes.decode("utf-8")
            or not snapshot.get("messages")):
        raise AssertionError("retained ordinary dialogue/receipt does not match native event")
    bot_client, _unused_user_client = resolve_native_report_clients(config_dir)
    output_dir.mkdir(mode=0o700, parents=True)
    workspace = output_dir / "workspace"
    workspace.mkdir(mode=0o700)
    fresh_snapshot = _session_key_latest_path(workspace, "telegram:123")
    fresh_snapshot.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fresh_snapshot.write_bytes(snapshot_bytes)
    # No retained photo bytes, auth config, or raw corpus is copied.
    save_gateway_config(GatewayConfig(
        enabled_channels=["telegram"], family_principals={"123": "synthetic_owner"},
        enabled_memory_tenants=("synthetic_owner",), memory_backend="file",
        conversation_learning=False, evals_capture=True,
    ), workspace)
    os.environ["OPENHARNESS_DATA_DIR"] = str(output_dir / "openharness-data")
    os.environ["OPENHARNESS_LOGS_DIR"] = str(output_dir / "openharness-logs")
    os.environ["OHMO_MEMORY_AUTOINDEX"] = "0"
    os.environ["OHMO_MEMORY_JUDGE"] = "0"
    signing_env = {
        "WELLNESS_DELEGATION_SIGNING_KEY": "camera-balance-probe-signing-key-0123456789",
        "WELLNESS_DELEGATION_KID": "camera-balance-probe",
        "WELLNESS_DELEGATION_ISSUER": "camera-balance-probe",
        "WELLNESS_DELEGATION_AUDIENCE": "camera-balance-probe-resource",
        "WELLNESS_DELEGATION_CLIENT_ID": "camera-balance-probe-client",
    }
    prior_signing = {key: os.environ.get(key) for key in signing_env}
    os.environ.update(signing_env)
    signing = WellnessDelegationConfig.from_env()
    db = NutritionDataStore(projection_dir / "nutrition.db")
    health = HealthDataStore(projection_dir / "health.db")
    registry, auth, authorized_read = synthetic_wellness_self_scope("synthetic_owner")
    app = FastMCP(name="camera-balance-native-facts")

    async def get_health_store():
        return health

    async def get_nutrition_store():
        return db

    register_wellness_tools(
        app, read_only_annotations=ToolAnnotations(readOnlyHint=True),
        get_health_store=get_health_store, get_nutrition_store=get_nutrition_store,
        participant_registry=registry, authorization_context=auth,
    )
    tool = next(item for item in await app.list_tools() if item.name == "get_wellness_data")
    calls: list[dict] = []

    class LocalVerifiedManager:
        async def call_tool_result(self, server_name, tool_name, arguments, *, meta=None):
            if (server_name, tool_name) != ("worfalomey", "get_wellness_data"):
                raise AssertionError("unexpected native MCP call")
            principal, _ = verify_delegation(
                (meta or {}).get(META_KEY), arguments, signing,
                authenticated_client_id=signing.client_id,
            )
            if principal != 123:
                raise AssertionError("native wellness actor changed")
            result = await call_wellness_with_synthetic_self(app, auth, authorized_read, arguments)
            payload = result[1]
            calls.append({"arguments": arguments, "payload": payload})
            return McpToolCallResult(output=json.dumps(payload, ensure_ascii=False))

    manager = LocalVerifiedManager()

    async def original_honcho_rows():
        async with HonchoClient(
            os.environ["CAMERA_HONCHO_URL"], "local-auth-disabled", goal["workspace_id"],
        ) as honcho:
            return await honcho.list_messages_in_window(
                goal["session_id"], expected_peer_id="ohmo",
                since=datetime(1970, 1, 1, tzinfo=timezone.utc),
                until=datetime.now(timezone.utc),
            )

    async def injected_build_runtime(*args, **kwargs):
        kwargs.pop("api_client", None)
        bundle = await original_build_runtime(*args, api_client=bot_client, **kwargs)
        bundle.tool_registry.register(McpToolAdapter(manager, McpToolInfo(
            server_name="worfalomey", name="get_wellness_data",
            description=tool.description or "Read current wellness data",
            input_schema=tool.inputSchema,
        )))
        admit_native_wellness_read(bundle)
        return bundle

    original_builder = gateway_runtime.build_runtime
    gateway_runtime.build_runtime = injected_build_runtime
    bus = MessageBus()
    fake_bot = OfflineTelegramBot()
    channel = TelegramChannel(TelegramConfig(token="offline-no-network", allow_from=["123"]), bus)
    from types import SimpleNamespace

    channel._app = SimpleNamespace(bot=fake_bot)
    channel.polling_started = True
    channel._start_typing = lambda _chat_id: None
    channel._stop_typing = lambda _chat_id: None
    pool = None
    try:
        before_rows = await original_honcho_rows()
        before_honcho = honcho_row_state(before_rows)
        before_projection = projection_state(db, "synthetic_owner")
        if (before_honcho.get(facts["event_id"], {}).get("sha256") != facts["honcho_fingerprint"]
                or facts["event_id"] not in before_projection["meal_records"]
                or before_projection["current_meals"].get(facts["meal_id"], {}).get("latest_event_id")
                != facts["event_id"]):
            raise AssertionError("retained Honcho and current Telegent meal differ before report")
        with isolated_runtime_loaders(openharness_runtime):
            pool = OhmoSessionRuntimePool(
                cwd=workspace, workspace=workspace,
                provider_profile="codex", model="gpt-6-luna",
                effort="medium", max_turns=12,
            )
            bridge = OhmoGatewayBridge(bus=bus, runtime_pool=pool)
            bounds = facts["query"]["params"]
            question = (
                "What was my provisional observed calorie balance for the exact "
                "historical 24-hour UTC interval "
                f"{bounds['start']} through {bounds['end']}? "
                "Use my recorded nutrition and observed energy data."
            )
            inbound = InboundMessage(
                channel="telegram", sender_id="123", chat_id="123", content=question,
                timestamp=datetime.now(timezone.utc),
                metadata={"message_id": "balance-report-followup-1", "is_group": False,
                          "chat_type": "private", "_telegram_raw_text": question},
            )
            await bridge._process_message(inbound, "telegram:123")
            delivered = []
            while bus.outbound_size:
                outbound = await bus.consume_outbound()
                receipt = await channel.send(outbound)
                delivered.append({"text": outbound.content, "metadata": outbound.metadata,
                                  "receipt": str(receipt)})
            if not calls:
                raise AssertionError("native report never called real get_wellness_data")
            query = facts["query"]["params"]
            start = datetime.fromisoformat(query["start"])
            end = datetime.fromisoformat(query["end"])
            exact_calls = []
            for call in calls:
                try:
                    totals = assert_fixture_balance(
                        call["payload"], start=start, end=end,
                        event_id=facts["event_id"], device_id="camera-balance-watch",
                    )
                except AssertionError:
                    continue
                params = call["arguments"].get("params", {})
                try:
                    exact = (datetime.fromisoformat(params["start"].replace("Z", "+00:00")) == start
                             and datetime.fromisoformat(params["end"].replace("Z", "+00:00")) == end)
                except (KeyError, TypeError, ValueError):
                    exact = False
                if exact:
                    exact_calls.append(totals)
            expected_totals = (
                facts["intake_kcal"], facts["expenditure_kcal"], facts["observed_balance_kcal"],
            )
            if not any(all(isclose(actual, expected, abs_tol=1e-8)
                           for actual, expected in zip(totals, expected_totals, strict=True))
                       for totals in exact_calls):
                raise AssertionError("native tool call did not read the exact event and energy window")
            new_snapshot = json.loads(fresh_snapshot.read_text())
            new_messages = new_snapshot["messages"][len(snapshot["messages"]):]
            tool_uses = [part for message in new_messages for part in message.get("content", [])
                         if isinstance(part, dict) and part.get("type") == "tool_use"]
            if not any(part.get("name") == "skill"
                       and isinstance(part.get("input"), dict)
                       and part["input"].get("name") == "calory" for part in tool_uses):
                raise AssertionError("native follow-up did not load the production calory skill")
            summaries = assert_report_only_traces(tool_uses)
            if any(item["metadata"].get("nutrition_append_event_id") for item in delivered):
                raise AssertionError("balance report attempted a new meal observation")
            if not any(item["text"] for item in delivered):
                raise AssertionError("native report returned no answer")
            after_rows = await original_honcho_rows()
            honcho_before, honcho_after = assert_honcho_report_delta(before_rows, after_rows)
            after_projection = projection_state(db, "synthetic_owner")
            if after_projection != before_projection:
                raise AssertionError("native report changed canonical current meals or meal records")
            (output_dir / "native-report.json").write_text(json.dumps({
                "event_id": facts["event_id"], "owner_question": question,
                "tool_calls": calls, "exact_totals": exact_calls,
                "delivered": delivered, "accepted_day_summaries": summaries,
                "trace_uses": [part for part in tool_uses if part.get("name") == "trace"],
                "honcho_before": honcho_before, "honcho_after": honcho_after,
                "projection_before": before_projection, "projection_after": after_projection,
                "honcho_scope": "retained food read only; file backend has no fresh Honcho append",
            }, indent=2, ensure_ascii=False, default=str) + "\n")
            print(f"NATIVE REPORT retained at {output_dir / 'native-report.json'}")
    finally:
        if pool is not None:
            await pool.aclose()
        gateway_runtime.build_runtime = original_builder
        health.close()
        db.close()
        for key, prior in prior_signing.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained-grade-dir", type=Path, required=True)
    parser.add_argument("--projection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.retained_grade_dir, args.projection_dir, args.output_dir,
                    args.native_config_dir))

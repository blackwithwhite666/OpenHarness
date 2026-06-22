import asyncio
import contextlib
import logging
from types import SimpleNamespace
from datetime import datetime
import json
from pathlib import Path

import pytest

from openharness.api.client import ApiMessageCompleteEvent
from openharness.api.usage import UsageSnapshot
from openharness.bridge import get_bridge_manager
from openharness.channels.bus.events import InboundMessage
from openharness.channels.bus.queue import MessageBus
from openharness.commands import CommandResult
from openharness.commands.registry import SlashCommand, create_default_command_registry
from openharness.config.settings import PermissionSettings, ProviderProfile, Settings
from openharness.engine.messages import ConversationMessage, ImageBlock, TextBlock, ToolUseBlock
from openharness.engine.query_engine import QueryEngine
from openharness.engine.stream_events import (
    AssistantTextDelta,
    CompactProgressEvent,
    ErrorEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from openharness.memory import add_memory_entry as add_project_memory_entry
from openharness.memory import list_memory_files as list_project_memory_files
from openharness.permissions import PermissionChecker, PermissionMode
from openharness.tools.base import ToolExecutionContext, ToolRegistry

from ohmo.evals import get_eval_store
from ohmo.gateway.bridge import OhmoGatewayBridge, _format_gateway_error
from ohmo.gateway.config import load_gateway_config, save_gateway_config
from ohmo.gateway.group_tool import OhmoCreateFeishuGroupInput, OhmoCreateFeishuGroupTool
from ohmo.gateway.models import GatewayConfig, GatewayState
from ohmo.gateway.provider_commands import handle_gateway_model_command, handle_gateway_provider_command
from ohmo.gateway.runtime import (
    OhmoSessionRuntimePool,
    _build_inbound_user_message,
    _evals_capture_enabled,
    _format_channel_progress,
    _sanitize_group_command_metadata,
    _sanitize_group_command_prompts,
)
from ohmo.gateway.service import OhmoGatewayService, gateway_status, stop_gateway_process
from ohmo.group_registry import load_managed_group_record, save_managed_group_record
from ohmo.memory import add_memory_entry as add_ohmo_memory_entry
from ohmo.memory import list_memory_files as list_ohmo_memory_files
from ohmo.gateway.router import session_key_for_message
from ohmo.session_storage import save_session_snapshot
from ohmo.workspace import (
    get_gateway_interrupted_requests_path,
    get_gateway_restart_notice_path,
    get_skills_dir,
    initialize_workspace,
)


def _single_eval_episode(workspace: Path):
    store = get_eval_store(workspace)
    episode_ids = store.list_episode_ids()
    assert len(episode_ids) == 1
    episode = store.get_episode(episode_ids[0])
    assert episode is not None
    return episode, list(store.iter_events(episode_ids[0]))


def _assert_resource_snapshot_event(
    workspace: Path, episode, event, *, phase: str = "world_before", tool_count: int
):
    assert event.kind == "resource_snapshot"
    assert event.payload["path"] == f"states/{episode.episode_id}/{phase}.json"
    assert event.payload["phase"] == phase
    assert event.payload["resource_count"] >= event.payload["local_resource_count"]
    assert event.payload["local_resource_count"] >= 16
    assert event.payload["tool_count"] == tool_count

    manifest_path = get_eval_store(workspace).root / event.payload["path"]
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["episode_id"] == episode.episode_id
    assert len(manifest["resources"]) == event.payload["resource_count"]
    resource_ids = {resource["resource_id"] for resource in manifest["resources"]}
    assert "ohmo.workspace.soul_md" in resource_ids
    assert "ohmo.workspace.user_md" in resource_ids
    assert "ohmo.workspace.logs_dir" not in resource_ids
    return manifest


def test_gateway_router_uses_thread_and_sender_for_group_when_present():
    message = InboundMessage(
        channel="slack",
        sender_id="u1",
        chat_id="c1",
        content="hello",
        timestamp=datetime.utcnow(),
        metadata={"thread_ts": "t1", "chat_type": "group"},
    )
    assert session_key_for_message(message) == "slack:c1:t1:u1"


def test_gateway_router_keeps_private_chat_scope_for_legacy_sessions():
    message = InboundMessage(
        channel="feishu",
        sender_id="u1",
        chat_id="ou_legacy",
        content="hello",
        timestamp=datetime.utcnow(),
        metadata={"chat_type": "p2p"},
    )
    assert session_key_for_message(message) == "feishu:ou_legacy"


def test_gateway_router_falls_back_to_chat_scope_when_chat_type_unknown():
    message = InboundMessage(
        channel="telegram",
        sender_id="u1",
        chat_id="chat-1",
        content="hello",
        timestamp=datetime.utcnow(),
    )
    assert session_key_for_message(message) == "telegram:chat-1"


def test_gateway_router_separates_senders_in_same_chat_thread():
    first = InboundMessage(
        channel="slack",
        sender_id="alice",
        chat_id="shared-chat",
        content="hello",
        timestamp=datetime.utcnow(),
        metadata={"thread_ts": "thread-1", "chat_type": "group"},
    )
    second = InboundMessage(
        channel="slack",
        sender_id="bob",
        chat_id="shared-chat",
        content="hello",
        timestamp=datetime.utcnow(),
        metadata={"thread_ts": "thread-1", "chat_type": "group"},
    )
    assert session_key_for_message(first) == "slack:shared-chat:thread-1:alice"
    assert session_key_for_message(second) == "slack:shared-chat:thread-1:bob"


def test_gateway_router_separates_senders_in_same_group_without_thread():
    first = InboundMessage(
        channel="feishu",
        sender_id="alice",
        chat_id="oc_shared",
        content="hello",
        timestamp=datetime.utcnow(),
        metadata={"chat_type": "group"},
    )
    second = InboundMessage(
        channel="feishu",
        sender_id="bob",
        chat_id="oc_shared",
        content="hello",
        timestamp=datetime.utcnow(),
        metadata={"chat_type": "group"},
    )
    assert session_key_for_message(first) == "feishu:oc_shared:alice"
    assert session_key_for_message(second) == "feishu:oc_shared:bob"


def test_gateway_error_formats_claude_refresh_failure():
    exc = ValueError("Claude OAuth refresh failed: HTTP Error 400: Bad Request")
    assert "claude-login" in _format_gateway_error(exc)
    assert "Claude subscription auth refresh failed" in _format_gateway_error(exc)


def test_gateway_error_formats_generic_auth_failure():
    exc = ValueError("API key missing for current profile")
    assert "Authentication failed" in _format_gateway_error(exc)


def test_compact_progress_formats_reactive_channel_hint_in_chinese():
    text = _format_channel_progress(
        channel="feishu",
        kind="compact_progress",
        text="",
        session_key="feishu:c1",
        content="帮我继续处理",
        compact_phase="compact_start",
        compact_trigger="reactive",
        attempt=None,
    )
    assert "重试" in text


def test_gateway_status_prefers_live_config_over_stale_state(tmp_path):
    workspace = tmp_path / ".ohmo-home"
    workspace.mkdir()
    (workspace / "gateway.json").write_text(
        json.dumps({"provider_profile": "codex", "enabled_channels": ["feishu"]}) + "\n",
        encoding="utf-8",
    )
    (workspace / "state.json").write_text(
        GatewayState(
            running=False,
            provider_profile="claude-subscription",
            enabled_channels=["feishu"],
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    state = gateway_status(tmp_path, workspace)
    assert state.running is False
    assert state.provider_profile == "codex"
    assert state.enabled_channels == ["feishu"]


def test_stop_gateway_process_kills_matching_workspace_processes(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    workspace.mkdir()
    (workspace / "gateway.json").write_text('{"provider_profile":"codex"}\n', encoding="utf-8")
    (workspace / "gateway.pid").write_text("123\n", encoding="utf-8")

    killed: list[int] = []

    def fake_run(*args, **kwargs):
        class Result:
            stdout = (
                f"123 python -m ohmo gateway run --workspace {workspace}\n"
                f"456 python -m ohmo gateway run --workspace {workspace}\n"
            )

        return Result()

    monkeypatch.setattr("ohmo.gateway.service.subprocess.run", fake_run)
    monkeypatch.setattr("ohmo.gateway.service._pid_is_running", lambda pid: True)
    monkeypatch.setattr("ohmo.gateway.service.os.kill", lambda pid, sig: killed.append(pid))

    assert stop_gateway_process(tmp_path, workspace) is True
    assert killed == [123, 456]


@pytest.mark.asyncio
async def test_runtime_pool_restores_messages_for_private_legacy_session_key(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    save_session_snapshot(
        cwd=tmp_path,
        workspace=workspace,
        model="gpt-5.4",
        system_prompt="system",
        messages=[ConversationMessage.from_user_text("remember private chat")],
        usage=UsageSnapshot(),
        session_id="sess123",
        session_key="feishu:chat-1",
    )

    captured: dict[str, object] = {}

    async def fake_build_runtime(**kwargs):
        captured["restore_messages"] = kwargs.get("restore_messages")
        return SimpleNamespace(
            engine=SimpleNamespace(set_system_prompt=lambda prompt: None, messages=[]),
            session_id="newsession",
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    bundle = await pool.get_bundle("feishu:chat-1")

    assert captured["restore_messages"] is not None
    assert bundle.session_id == "sess123"


@pytest.mark.asyncio
async def test_runtime_pool_uses_managed_group_cwd_binding(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    project = tmp_path / "OpenHarness-new"
    project.mkdir()
    initialize_workspace(workspace)
    save_managed_group_record(
        workspace=workspace,
        channel="feishu",
        chat_id="oc_group",
        owner_open_id="ou_user",
        name="HKUDS/OpenHarness",
        cwd=str(project),
        repo="HKUDS/OpenHarness",
        binding_status="bound",
    )

    captured: dict[str, object] = {}

    async def fake_build_runtime(**kwargs):
        captured["cwd"] = kwargs.get("cwd")
        return SimpleNamespace(
            engine=SimpleNamespace(set_system_prompt=lambda prompt: None, messages=[]),
            session_id="newsession",
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(
        channel="feishu",
        sender_id="ou_user",
        chat_id="oc_group",
        content="hello",
        metadata={"chat_type": "group"},
    )
    bundle = await pool.get_bundle(
        "feishu:oc_group:ou_user",
        cwd=pool._cwd_for_message(message, "feishu:oc_group:ou_user"),
    )

    assert captured["cwd"] == str(project.resolve())
    assert bundle.session_id == "newsession"


@pytest.mark.asyncio
async def test_runtime_pool_restores_messages_for_group_sender_scoped_session_key(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    save_session_snapshot(
        cwd=tmp_path,
        workspace=workspace,
        model="gpt-5.4",
        system_prompt="system",
        messages=[ConversationMessage.from_user_text("remember alice only")],
        usage=UsageSnapshot(),
        session_id="sess123",
        session_key="feishu:chat-1:alice",
    )

    captured: dict[str, object] = {}

    async def fake_build_runtime(**kwargs):
        captured["restore_messages"] = kwargs.get("restore_messages")
        return SimpleNamespace(
            engine=SimpleNamespace(set_system_prompt=lambda prompt: None, messages=[]),
            session_id="newsession",
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    bundle = await pool.get_bundle("feishu:chat-1:alice")

    assert captured["restore_messages"] is not None
    assert bundle.session_id == "sess123"


@pytest.mark.asyncio
async def test_runtime_pool_does_not_restore_other_group_sender_session_key(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    save_session_snapshot(
        cwd=tmp_path,
        workspace=workspace,
        model="gpt-5.4",
        system_prompt="system",
        messages=[ConversationMessage.from_user_text("remember alice only")],
        usage=UsageSnapshot(),
        session_id="sess123",
        session_key="feishu:chat-1:alice",
    )

    captured: dict[str, object] = {}

    async def fake_build_runtime(**kwargs):
        captured["restore_messages"] = kwargs.get("restore_messages")
        return SimpleNamespace(
            engine=SimpleNamespace(set_system_prompt=lambda prompt: None, messages=[]),
            session_id="newsession",
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    bundle = await pool.get_bundle("feishu:chat-1:bob")

    assert captured["restore_messages"] is None
    assert bundle.session_id == "newsession"


@pytest.mark.asyncio
async def test_runtime_pool_splits_per_turn_narration_into_reasoning(tmp_path, monkeypatch):
    """A tool-using turn's interstitial narration is surfaced live as a 🧠
    reasoning message, and the final reply holds ONLY the last (tool-free)
    turn's text — earlier turns' narration is no longer concatenated into the
    final message with no separator."""
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)

    async def fake_build_runtime(**kwargs):
        class FakeEngine:
            messages = []
            total_usage = UsageSnapshot()

            def set_system_prompt(self, prompt):
                return None

            async def submit_message(self, content):
                # turn 1: narration preamble, then a tool call
                yield AssistantTextDelta(text="Проверю через travel-cli.")
                yield ToolExecutionStarted(tool_name="bash", tool_input={"command": "ls"})
                yield ToolExecutionCompleted(tool_name="bash", output="ok", is_error=False)
                # turn 2: the actual answer (no tool call after it)
                yield AssistantTextDelta(text="Готово: купе есть.")

        return SimpleNamespace(
            engine=FakeEngine(),
            cwd=str(tmp_path),
            session_id="sess-multiturn",
            current_settings=lambda: SimpleNamespace(model="gpt-5.5"),
            commands=SimpleNamespace(lookup=lambda raw: None),
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="купе в Тамбов")
    updates = [u async for u in pool.stream_message(message, "telegram:c1")]

    # turn-1 narration is surfaced live as a 🧠 reasoning progress message
    reasoning = [u for u in updates if u.kind == "progress" and "Проверю через travel-cli." in u.text]
    assert reasoning, [(u.kind, u.text) for u in updates]
    assert reasoning[0].text == "🧠 Проверю через travel-cli."

    # the reasoning message precedes the tool hint
    reasoning_idx = updates.index(reasoning[0])
    tool_hint_idx = next(i for i, u in enumerate(updates) if u.kind == "tool_hint")
    assert reasoning_idx < tool_hint_idx

    # the final reply is ONLY the last turn — earlier narration is not glued in
    final = updates[-1]
    assert final.kind == "final"
    assert final.text == "Готово: купе есть."
    assert "Проверю через travel-cli" not in final.text


@pytest.mark.asyncio
async def test_runtime_pool_stream_message_emits_progress_and_tool_hint(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)

    async def fake_build_runtime(**kwargs):
        class FakeEngine:
            messages = []
            total_usage = UsageSnapshot()

            def set_system_prompt(self, prompt):
                return None

            async def submit_message(self, content):
                yield ToolExecutionStarted(tool_name="web_fetch", tool_input={"url": "https://example.com"})
                yield AssistantTextDelta(text="done")

        return SimpleNamespace(
            engine=FakeEngine(),
            cwd=str(tmp_path),
            session_id="sess123",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=SimpleNamespace(lookup=lambda raw: None),
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="check")
    updates = [u async for u in pool.stream_message(message, "feishu:c1")]

    assert updates[0].kind == "progress"
    assert updates[0].text.startswith(("🤔", "🧠", "✨", "🔎", "🪄", "🧩", "📝"))
    assert updates[1].kind == "tool_hint"
    assert updates[1].text.startswith("🛠️ ")
    assert "Web fetch" in updates[1].text  # humanized, mcp__ prefix stripped
    assert "```" in updates[1].text  # args rendered as a code block
    assert updates[-1].kind == "final"
    assert updates[-1].text == "done"


@pytest.mark.asyncio
async def test_runtime_pool_records_eval_episode_for_tool_turn(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)

    async def fake_build_runtime(**kwargs):
        class FakeEngine:
            messages = []
            total_usage = UsageSnapshot()

            def set_system_prompt(self, prompt):
                return None

            async def submit_message(self, content):
                yield ToolExecutionStarted(
                    tool_name="web_fetch",
                    tool_input={"url": "https://example.com", "limit": 3},
                    tool_call_id="toolu_abc123",
                )
                yield ToolExecutionCompleted(
                    tool_name="web_fetch",
                    output="ok",
                    is_error=False,
                    tool_call_id="toolu_abc123",
                )
                yield AssistantTextDelta(text="done")

        return SimpleNamespace(
            engine=FakeEngine(),
            cwd=str(tmp_path),
            session_id="sess-eval-tool",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=SimpleNamespace(lookup=lambda raw: None),
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(
        channel="feishu",
        sender_id="u1",
        chat_id="c1",
        content="check",
        timestamp=datetime(2026, 1, 2, 3, 4, 5),
        metadata={"chat_type": "p2p", "sent_at": datetime(2026, 1, 2, 3, 4, 5)},
    )
    updates = [u async for u in pool.stream_message(message, "feishu:c1")]

    assert updates[-1].kind == "final"
    assert updates[-1].text == "done"

    episode, events = _single_eval_episode(workspace)
    assert episode.source == "gateway"
    assert episode.app == "ohmo"
    assert episode.session_id == "sess-eval-tool"
    assert episode.user_text == "check"
    assert episode.metadata["session_key"] == "feishu:c1"
    assert episode.metadata["cwd"] == str(tmp_path)
    assert episode.metadata["model"] == "gpt-5.4"
    assert episode.metadata["inbound"]["metadata"]["sent_at"] == "2026-01-02T03:04:05"
    assert [event.kind for event in events] == [
        "inbound_message",
        "resource_snapshot",
        "tool_started",
        "tool_completed",
        "gateway_final",
        "resource_snapshot",
        "episode_finished",
    ]
    _assert_resource_snapshot_event(workspace, episode, events[1], tool_count=0)
    _assert_resource_snapshot_event(
        workspace, episode, events[5], phase="world_after", tool_count=0
    )
    assert events[2].tool_name == "web_fetch"
    assert events[2].tool_call_id == "toolu_abc123"
    assert events[2].payload["input"]["url"] == "https://example.com"
    assert "https://example.com" in events[2].payload["input_summary"]
    assert events[3].tool_name == "web_fetch"
    assert events[3].tool_call_id == "toolu_abc123"
    assert events[3].is_error is False
    assert events[3].payload["output"] == "ok"
    assert events[4].payload["text"] == "done"
    assert events[6].payload == {"status": "completed"}


def _install_fake_tool_turn(monkeypatch, tmp_path, *, session_id):
    """Patch build/start runtime with a fake one-tool-then-final turn."""

    async def fake_build_runtime(**kwargs):
        class FakeEngine:
            messages = []
            total_usage = UsageSnapshot()

            def set_system_prompt(self, prompt):
                return None

            async def submit_message(self, content):
                yield ToolExecutionStarted(
                    tool_name="web_fetch",
                    tool_input={"url": "https://example.com"},
                    tool_call_id="toolu_x",
                )
                yield ToolExecutionCompleted(
                    tool_name="web_fetch", output="ok", is_error=False, tool_call_id="toolu_x"
                )
                yield AssistantTextDelta(text="done")

        return SimpleNamespace(
            engine=FakeEngine(),
            cwd=str(tmp_path),
            session_id=session_id,
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=SimpleNamespace(lookup=lambda raw: None),
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)


@pytest.mark.asyncio
async def test_runtime_pool_skips_eval_capture_when_disabled_via_env(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    monkeypatch.setenv("OHMO_EVALS_CAPTURE", "0")
    _install_fake_tool_turn(monkeypatch, tmp_path, session_id="sess-env-off")

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="check")
    updates = [u async for u in pool.stream_message(message, "feishu:c1")]

    # The turn still completes normally...
    assert updates[-1].kind == "final"
    assert updates[-1].text == "done"
    # ...but nothing is captured.
    assert get_eval_store(workspace).list_episode_ids() == []


@pytest.mark.asyncio
async def test_runtime_pool_skips_eval_capture_when_disabled_via_config(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    monkeypatch.delenv("OHMO_EVALS_CAPTURE", raising=False)
    config = load_gateway_config(workspace)
    config.evals_capture = False
    save_gateway_config(config, workspace)
    _install_fake_tool_turn(monkeypatch, tmp_path, session_id="sess-cfg-off")

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="check")
    updates = [u async for u in pool.stream_message(message, "feishu:c1")]

    assert updates[-1].text == "done"
    assert get_eval_store(workspace).list_episode_ids() == []


@pytest.mark.asyncio
async def test_runtime_pool_captures_by_default(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    monkeypatch.delenv("OHMO_EVALS_CAPTURE", raising=False)
    _install_fake_tool_turn(monkeypatch, tmp_path, session_id="sess-default-on")

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="check")
    _ = [u async for u in pool.stream_message(message, "feishu:c1")]

    # Default config has evals_capture=True -> exactly one episode captured.
    assert len(get_eval_store(workspace).list_episode_ids()) == 1


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0", False),
        ("false", False),
        ("FALSE", False),
        ("no", False),
        ("off", False),
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
    ],
)
def test_evals_capture_enabled_env_overrides_config(monkeypatch, raw, expected):
    monkeypatch.setenv("OHMO_EVALS_CAPTURE", raw)
    # Env wins over the config flag in both directions.
    assert _evals_capture_enabled(GatewayConfig(evals_capture=True)) is expected
    assert _evals_capture_enabled(GatewayConfig(evals_capture=False)) is expected


def test_evals_capture_enabled_follows_config_without_env(monkeypatch):
    monkeypatch.delenv("OHMO_EVALS_CAPTURE", raising=False)
    assert _evals_capture_enabled(GatewayConfig(evals_capture=True)) is True
    assert _evals_capture_enabled(GatewayConfig(evals_capture=False)) is False
    # Default config captures.
    assert _evals_capture_enabled(GatewayConfig()) is True


def test_evals_capture_enabled_blank_env_falls_back_to_config(monkeypatch):
    monkeypatch.setenv("OHMO_EVALS_CAPTURE", "   ")
    assert _evals_capture_enabled(GatewayConfig(evals_capture=False)) is False
    assert _evals_capture_enabled(GatewayConfig(evals_capture=True)) is True


@pytest.mark.asyncio
async def test_runtime_pool_records_eval_episode_for_command_only_final(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)

    async def command_handler(args, context):
        assert args == ""
        return CommandResult(message="pong")

    async def fake_build_runtime(**kwargs):
        class FakeEngine:
            messages = []
            total_usage = UsageSnapshot()

            def set_system_prompt(self, prompt):
                return None

        command = SlashCommand("hello", "Say hello", command_handler)
        return SimpleNamespace(
            engine=FakeEngine(),
            cwd=str(tmp_path),
            session_id="sess-eval-command",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=SimpleNamespace(
                lookup=lambda raw: (command, "") if raw == "/hello" else None
            ),
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/hello")
    updates = [u async for u in pool.stream_message(message, "telegram:c1")]

    assert [(update.kind, update.text) for update in updates] == [("final", "pong")]

    episode, events = _single_eval_episode(workspace)
    assert episode.session_id == "sess-eval-command"
    assert episode.user_text == "/hello"
    assert [event.kind for event in events] == [
        "inbound_message",
        "resource_snapshot",
        "gateway_final",
        "resource_snapshot",
        "episode_finished",
    ]
    _assert_resource_snapshot_event(workspace, episode, events[1], tool_count=0)
    _assert_resource_snapshot_event(
        workspace, episode, events[3], phase="world_after", tool_count=0
    )
    assert events[2].payload["text"] == "pong"
    assert events[2].payload["metadata"]["_command"] is True
    assert events[4].payload == {"status": "completed"}


@pytest.mark.asyncio
async def test_runtime_pool_stream_message_formats_auto_compact_status_for_feishu(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)

    async def fake_build_runtime(**kwargs):
        class FakeEngine:
            messages = []
            total_usage = UsageSnapshot()

            def set_system_prompt(self, prompt):
                return None

            async def submit_message(self, content):
                yield CompactProgressEvent(phase="compact_start", trigger="auto")
                yield AssistantTextDelta(text="done")

        return SimpleNamespace(
            engine=FakeEngine(),
            cwd=str(tmp_path),
            session_id="sess123",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=SimpleNamespace(lookup=lambda raw: None),
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="继续")
    updates = [u async for u in pool.stream_message(message, "feishu:c1")]

    assert updates[1].kind == "progress"
    assert updates[1].text.startswith("\U0001f9e0")  # localization-robust: compact-status emoji
    assert "compact" in updates[1].text.lower()
    assert updates[-1].kind == "final"
    assert updates[-1].text == "done"


@pytest.mark.asyncio
async def test_runtime_pool_stream_message_formats_compact_retry_for_feishu(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)

    async def fake_build_runtime(**kwargs):
        class FakeEngine:
            messages = []
            total_usage = UsageSnapshot()

            def set_system_prompt(self, prompt):
                return None

            async def submit_message(self, content):
                yield CompactProgressEvent(phase="compact_retry", trigger="auto", attempt=2, message="retrying")
                yield AssistantTextDelta(text="done")

        return SimpleNamespace(
            engine=FakeEngine(),
            session_id="sess123",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=SimpleNamespace(lookup=lambda raw: None),
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="继续")
    updates = [u async for u in pool.stream_message(message, "feishu:c1")]

    assert updates[1].kind == "progress"
    assert "\U0001f501" in updates[1].text  # retry emoji
    assert "retry" in updates[1].text.lower()


@pytest.mark.asyncio
async def test_runtime_pool_stream_message_formats_compact_hooks_start_for_feishu(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)

    async def fake_build_runtime(**kwargs):
        class FakeEngine:
            messages = []
            total_usage = UsageSnapshot()

            def set_system_prompt(self, prompt):
                return None

            async def submit_message(self, content):
                yield CompactProgressEvent(phase="hooks_start", trigger="auto")
                yield AssistantTextDelta(text="done")

        return SimpleNamespace(
            engine=FakeEngine(),
            session_id="sess123",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=SimpleNamespace(lookup=lambda raw: None),
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="继续")
    updates = [u async for u in pool.stream_message(message, "feishu:c1")]

    assert updates[1].kind == "progress"
    assert "\U0001fae7" in updates[1].text  # compaction hooks-start emoji
    assert "compact" in updates[1].text.lower()


@pytest.mark.asyncio
async def test_runtime_pool_stream_message_uses_english_progress_for_english_input(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)

    async def fake_build_runtime(**kwargs):
        class FakeEngine:
            messages = []
            total_usage = UsageSnapshot()

            def set_system_prompt(self, prompt):
                return None

            async def submit_message(self, content):
                yield ToolExecutionStarted(tool_name="web_fetch", tool_input={"url": "https://example.com"})
                yield AssistantTextDelta(text="done")

        return SimpleNamespace(
            engine=FakeEngine(),
            session_id="sess123",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=SimpleNamespace(lookup=lambda raw: None),
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="can you check this")
    updates = [u async for u in pool.stream_message(message, "feishu:c1")]

    assert updates[0].kind == "progress"
    assert updates[0].text.startswith(("🤔", "🧠", "✨", "🔎", "🪄", "🧩", "📝"))
    assert "Thinking" in updates[0].text or "Working" in updates[0].text or "Looking" in updates[0].text or "Following" in updates[0].text or "Pulling" in updates[0].text
    assert updates[1].kind == "tool_hint"
    assert updates[1].text.startswith("🛠️ Web fetch")


@pytest.mark.asyncio
async def test_runtime_pool_blocks_local_only_commands_from_remote_messages(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    handler_called = False

    async def forbidden_handler(args, context):
        nonlocal handler_called
        handler_called = True
        return CommandResult(message="should not run")

    async def fake_build_runtime(**kwargs):
        class FakeEngine:
            messages = []
            total_usage = UsageSnapshot()

            def set_system_prompt(self, prompt):
                return None

        command = SlashCommand(
            "permissions",
            "Show or update permission mode",
            forbidden_handler,
            remote_invocable=False,
        )
        command.remote_admin_opt_in = True
        return SimpleNamespace(
            engine=FakeEngine(),
            session_id="sess123",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=SimpleNamespace(lookup=lambda raw: (command, "full_auto")),
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="/permissions full_auto")
    updates = [u async for u in pool.stream_message(message, "feishu:c1")]

    assert handler_called is False
    assert updates[-1].kind == "final"
    assert updates[-1].text == "/permissions is only available in the local OpenHarness UI."


@pytest.mark.asyncio
async def test_runtime_pool_blocks_bridge_spawn_from_remote_messages(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    handler_called = False

    async def forbidden_bridge_handler(args, context):
        nonlocal handler_called
        handler_called = True
        return CommandResult(message="spawned")

    async def fake_build_runtime(**kwargs):
        class FakeEngine:
            messages = []
            total_usage = UsageSnapshot()

            def set_system_prompt(self, prompt):
                return None

        command = SlashCommand(
            "bridge",
            "Inspect bridge helpers and spawn bridge sessions",
            forbidden_bridge_handler,
            remote_invocable=False,
            remote_admin_opt_in=True,
        )
        return SimpleNamespace(
            engine=FakeEngine(),
            session_id="sess123",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=SimpleNamespace(lookup=lambda raw: (command, "spawn id")),
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="/bridge spawn id")
    updates = [u async for u in pool.stream_message(message, "feishu:c1")]

    assert handler_called is False
    assert updates[-1].kind == "final"
    assert updates[-1].text == "/bridge is only available in the local OpenHarness UI."


@pytest.mark.asyncio
async def test_runtime_pool_blocks_registered_bridge_spawn_without_shelling_out(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    marker = tmp_path / "remote-bridge-marker.txt"
    payload = f"/bridge spawn printf REMOTE_BRIDGE_EXEC > {marker}"
    registry = create_default_command_registry()
    command, _ = registry.lookup(payload)
    existing_bridge_sessions = {session.session_id for session in get_bridge_manager().list_sessions()}

    assert command is not None
    assert command.name == "bridge"

    async def fake_build_runtime(**kwargs):
        class FakeEngine:
            messages = []
            total_usage = UsageSnapshot()

            def set_system_prompt(self, prompt):
                return None

        return SimpleNamespace(
            engine=FakeEngine(),
            cwd=str(tmp_path),
            session_id="sess123",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=registry,
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content=payload)
    updates = [u async for u in pool.stream_message(message, "feishu:c1")]

    assert updates[-1].kind == "final"
    assert updates[-1].text == "/bridge is only available in the local OpenHarness UI."
    assert {session.session_id for session in get_bridge_manager().list_sessions()} == existing_bridge_sessions
    assert marker.exists() is False



@pytest.mark.asyncio
async def test_runtime_pool_blocks_registered_config_show_without_leaking_secrets(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    registry = create_default_command_registry()
    command, _ = registry.lookup("/config show")

    assert command is not None
    assert command.name == "config"
    assert command.remote_invocable is False

    fake_settings = Settings(
        mcp_servers={
            "internal-http": {
                "type": "http",
                "url": "https://mcp.internal",
                "headers": {"Authorization": "Bearer MCP_FAKE_SECRET"},
            },
        },
        vision={"model": "vision-test", "api_key": "VISION_FAKE_SECRET"},
    )

    async def fake_build_runtime(**kwargs):
        class FakeEngine:
            messages = []
            total_usage = UsageSnapshot()

            def set_system_prompt(self, prompt):
                return None

        return SimpleNamespace(
            engine=FakeEngine(),
            cwd=str(tmp_path),
            session_id="sess123",
            current_settings=lambda: fake_settings,
            commands=registry,
            tool_registry=None,
            app_state=None,
            session_backend=None,
            extra_skill_dirs=(),
            extra_plugin_roots=(),
            hook_summary=lambda: "",
            mcp_summary=lambda: "",
            plugin_summary=lambda: "",
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(channel="slack", sender_id="U_ATTACKER", chat_id="C_SHARED", content="/config show")
    updates = [u async for u in pool.stream_message(message, "slack:C_SHARED:U_ATTACKER")]

    assert updates[-1].kind == "final"
    assert updates[-1].text == "/config is only available in the local OpenHarness UI."
    assert "MCP_FAKE_SECRET" not in updates[-1].text
    assert "VISION_FAKE_SECRET" not in updates[-1].text


@pytest.mark.asyncio
async def test_runtime_pool_memory_command_uses_ohmo_personal_memory(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    registry = create_default_command_registry()

    async def fake_build_runtime(**kwargs):
        class FakeEngine:
            messages = []
            total_usage = UsageSnapshot()

            def set_system_prompt(self, prompt):
                self.system_prompt = prompt

        return SimpleNamespace(
            engine=FakeEngine(),
            cwd=str(tmp_path),
            session_id="sess123",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=registry,
            tool_registry=None,
            app_state=None,
            session_backend=None,
            extra_skill_dirs=(),
            extra_plugin_roots=(),
            hook_summary=lambda: "",
            mcp_summary=lambda: "",
            plugin_summary=lambda: "",
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(
        channel="feishu",
        sender_id="u1",
        chat_id="c1",
        content="/memory add Profile :: prefers concise answers",
    )
    updates = [u async for u in pool.stream_message(message, "feishu:c1")]

    assert updates[-1].text == "Added memory entry profile.md"
    assert [path.name for path in list_ohmo_memory_files(workspace)] == ["profile.md"]
    assert "prefers concise answers" in (workspace / "memory" / "profile.md").read_text(encoding="utf-8")
    assert list_project_memory_files(tmp_path) == []


@pytest.mark.asyncio
async def test_runtime_pool_prompt_excludes_project_memory(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    add_ohmo_memory_entry(workspace, "personal", "ohmo-only personal fact")
    monkeypatch.delenv("CLAUDE_CODE_COORDINATOR_MODE", raising=False)
    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    add_project_memory_entry(tmp_path, "project", "project memory should not leak")

    async def fake_build_runtime(**kwargs):
        class FakeEngine:
            messages = []
            total_usage = UsageSnapshot()

            def set_system_prompt(self, prompt):
                self.system_prompt = prompt

        engine = FakeEngine()
        return SimpleNamespace(
            engine=engine,
            session_id="sess123",
            current_settings=lambda: Settings(system_prompt=kwargs["system_prompt"]),
            commands=create_default_command_registry(),
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    bundle = await pool.get_bundle("feishu:c1", latest_user_prompt="hello")

    assert "ohmo-only personal fact" in bundle.engine.system_prompt
    assert "project memory should not leak" not in bundle.engine.system_prompt


@pytest.mark.asyncio
async def test_runtime_pool_allows_opted_in_remote_admin_commands(tmp_path, monkeypatch, caplog):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    save_gateway_config(
        GatewayConfig(
            provider_profile="codex",
            allow_remote_admin_commands=True,
            allowed_remote_admin_commands=["permissions"],
        ),
        workspace,
    )
    handler_called = False

    async def allowed_handler(args, context):
        nonlocal handler_called
        handler_called = True
        return CommandResult(message=f"ran with {args}")

    async def fake_build_runtime(**kwargs):
        class FakeEngine:
            messages = []
            total_usage = UsageSnapshot()

            def set_system_prompt(self, prompt):
                return None

        command = SlashCommand(
            "permissions",
            "Show or update permission mode",
            allowed_handler,
            remote_invocable=False,
        )
        command.remote_admin_opt_in = True
        return SimpleNamespace(
            engine=FakeEngine(),
            session_id="sess123",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=SimpleNamespace(lookup=lambda raw: (command, "full_auto")),
            hook_summary=lambda: "",
            mcp_summary=lambda: "",
            plugin_summary=lambda: "",
            cwd=str(tmp_path),
            tool_registry=None,
            app_state=None,
            session_backend=None,
            extra_skill_dirs=(),
            extra_plugin_roots=(),
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    with caplog.at_level(logging.WARNING):
        pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
        message = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="/permissions full_auto")
        updates = [u async for u in pool.stream_message(message, "feishu:c1")]

    assert handler_called is True
    assert updates[-1].kind == "final"
    assert updates[-1].text == "ran with full_auto"
    assert "remote administrative command accepted" in caplog.text


@pytest.mark.asyncio
async def test_runtime_pool_includes_media_paths_in_prompt(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    image_path = tmp_path / "example.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01"
        b"\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    report_path = tmp_path / "report.txt"
    report_path.write_text("Quarterly summary\nRevenue up 12%\n", encoding="utf-8")
    captured: dict[str, object] = {}

    async def fake_build_runtime(**kwargs):
        class FakeEngine:
            messages = []
            total_usage = UsageSnapshot()

            def set_system_prompt(self, prompt):
                return None

            async def submit_message(self, content):
                captured["content"] = content
                yield AssistantTextDelta(text="done")

        return SimpleNamespace(
            engine=FakeEngine(),
            session_id="sess123",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=SimpleNamespace(lookup=lambda raw: None),
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(
        channel="feishu",
        sender_id="u1",
        chat_id="c1",
        content="请看这个图片",
        media=[str(image_path), str(report_path)],
    )
    updates = [u async for u in pool.stream_message(message, "feishu:c1")]

    assert updates[-1].text == "done"
    submitted = captured["content"]
    assert isinstance(submitted, ConversationMessage)
    assert any(isinstance(block, ImageBlock) for block in submitted.content)
    text = "".join(block.text for block in submitted.content if isinstance(block, TextBlock))
    assert "[Channel attachments]" in text
    assert f"image: example.png (path: {image_path})" in text
    assert f"file: report.txt (path: {report_path})" in text
    assert "text preview: Quarterly summary Revenue up 12%" in text


@pytest.mark.asyncio
async def test_runtime_pool_retries_with_attachment_summary_when_model_rejects_images(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    image_path = tmp_path / "example.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01"
        b"\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    captured: dict[str, object] = {}

    async def fake_build_runtime(**kwargs):
        class FakeEngine:
            def __init__(self):
                self.messages = []
                self.total_usage = UsageSnapshot()
                self.max_turns = 8

            def set_system_prompt(self, prompt):
                return None

            def load_messages(self, messages):
                self.messages = list(messages)

            async def submit_message(self, content):
                self.messages.append(content)
                yield ErrorEvent(
                    message=(
                        "API error: Error code: 404 - {'error': {'message': "
                        "'No endpoints found that support image input', 'code': 404}}"
                    )
                )

            async def continue_pending(self, *, max_turns=None):
                captured["retry_messages"] = list(self.messages)
                yield AssistantTextDelta(text="done")

        return SimpleNamespace(
            engine=FakeEngine(),
            session_id="sess123",
            current_settings=lambda: SimpleNamespace(model="openrouter/text-only"),
            commands=SimpleNamespace(lookup=lambda raw: None),
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="openrouter")
    message = InboundMessage(
        channel="telegram",
        sender_id="u1",
        chat_id="c1",
        content="帮我看这个图片",
        media=[str(image_path)],
    )
    updates = [u async for u in pool.stream_message(message, "telegram:c1")]

    assert updates[-1].kind == "final"
    assert updates[-1].text == "done"
    assert not any(update.kind == "error" for update in updates)
    assert any(update.metadata.get("_image_fallback") for update in updates)
    retry_messages = captured["retry_messages"]
    assert all(
        not isinstance(block, ImageBlock)
        for item in retry_messages
        for block in item.content
    )
    text = "".join(
        block.text
        for item in retry_messages
        for block in item.content
        if isinstance(block, TextBlock)
    )
    assert "[Channel attachments]" in text
    assert f"image: example.png (path: {image_path})" in text


def test_runtime_pool_includes_group_speaker_context():
    built = _build_inbound_user_message(
        InboundMessage(
            channel="feishu",
            sender_id="ou_123",
            chat_id="oc_group",
            content="请帮我看一下",
            metadata={"chat_type": "group", "sender_display_name": "Tang Jiabin"},
        )
    )
    text = "".join(block.text for block in built.content if isinstance(block, TextBlock))
    assert "[Channel speaker]" in text
    assert "Tang Jiabin" in text
    assert "Sender id: ou_123" in text
    assert "请帮我看一下" in text


@pytest.mark.asyncio
async def test_gateway_bridge_publishes_progress_updates():
    bus = MessageBus()

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            yield SimpleNamespace(kind="progress", text="🤔 想一想…", metadata={"_progress": True, "_session_key": session_key})
            yield SimpleNamespace(kind="tool_hint", text="🛠️ 正在使用 web_fetch: https://example.com", metadata={"_progress": True, "_tool_hint": True, "_session_key": session_key})
            yield SimpleNamespace(kind="final", text="Done", metadata={"_session_key": session_key})

    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=FakeRuntimePool())
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="hi")
        )
        first = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        second = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        third = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    finally:
        bridge.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert first.content.startswith(("🤔", "🧠", "✨", "🔎", "🪄", "🧩", "📝"))
    assert first.metadata["_progress"] is True
    assert second.metadata["_tool_hint"] is True
    assert second.content.startswith("🛠️ ")
    assert "web_fetch" in second.content
    assert third.content == "Done"


@pytest.mark.asyncio
async def test_gateway_bridge_does_not_thread_private_feishu_replies():
    bus = MessageBus()

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            yield SimpleNamespace(kind="progress", text="🤔 想一想…", metadata={"_progress": True})
            yield SimpleNamespace(kind="final", text="Done", metadata={})

    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=FakeRuntimePool())
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(
                channel="feishu",
                sender_id="ou_1",
                chat_id="ou_1",
                content="hi",
                metadata={"chat_type": "p2p", "message_id": "om_private"},
            )
        )
        progress = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        final = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert "message_id" not in progress.metadata
    assert "message_id" not in final.metadata
    assert final.metadata["_session_key"] == "feishu:ou_1"


@pytest.mark.asyncio
async def test_gateway_bridge_threads_private_telegram_replies():
    """Telegram DMs DO carry the inbound message_id (unlike Feishu p2p) so the
    bot's reply threads under the user's message — the user asked for it; the
    actual reply is still gated by the channel's reply_to_message config."""
    bus = MessageBus()

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            yield SimpleNamespace(kind="progress", text="🤔…", metadata={"_progress": True})
            yield SimpleNamespace(kind="final", text="Done", metadata={})

    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=FakeRuntimePool())
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(
                channel="telegram",
                sender_id="116870365|blackwithwhite",
                chat_id="116870365",
                content="который час?",
                metadata={"chat_type": "private", "message_id": 4242},
            )
        )
        progress = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        final = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert final.metadata["message_id"] == 4242  # final threads under the user's message
    assert "message_id" not in progress.metadata  # progress stays a plain message (no draft path)
    assert final.metadata["_session_key"] == "telegram:116870365"


@pytest.mark.asyncio
async def test_gateway_bridge_threads_group_feishu_replies():
    bus = MessageBus()

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            yield SimpleNamespace(kind="progress", text="🤔 想一想…", metadata={"_progress": True})
            yield SimpleNamespace(kind="final", text="Done", metadata={})

    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=FakeRuntimePool())
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(
                channel="feishu",
                sender_id="ou_1",
                chat_id="oc_group",
                content="hi",
                metadata={"chat_type": "group", "message_id": "om_group", "thread_id": "thread_1"},
            )
        )
        progress = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        final = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert progress.metadata["message_id"] == "om_group"
    assert final.metadata["message_id"] == "om_group"
    assert final.metadata["_session_key"] == "feishu:oc_group:thread_1:ou_1"


@pytest.mark.asyncio
async def test_gateway_bridge_ignores_unmentioned_unmanaged_feishu_group(tmp_path):
    bus = MessageBus()
    calls: list[InboundMessage] = []

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            calls.append(message)
            yield SimpleNamespace(kind="final", text="should not happen", metadata={"_session_key": session_key})

    bridge = OhmoGatewayBridge(
        bus=bus,
        runtime_pool=FakeRuntimePool(),
        workspace=tmp_path,
        feishu_group_policy="managed_or_mention",
    )
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(
                channel="feishu",
                sender_id="ou_1",
                chat_id="oc_random",
                content="这个普通群消息不应该触发",
                metadata={"chat_type": "group", "mentions_bot": False},
            )
        )
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(bus.consume_outbound(), timeout=0.1)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert calls == []


@pytest.mark.asyncio
async def test_gateway_bridge_runs_synthetic_reminder_in_feishu_group_under_mention_policy(tmp_path):
    # A scheduler-originated agentic reminder publishes a synthetic InboundMessage
    # into a Feishu group. Even under a mention/managed group_policy (and with no
    # mentions_bot flag) it must still run an agent turn — the synthetic message
    # bypasses the channel-layer ACL by design (it came from a stored, owner-
    # created reminder). Without the bypass the occurrence is silently dropped.
    bus = MessageBus()
    calls: list[InboundMessage] = []

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            calls.append(message)
            yield SimpleNamespace(kind="final", text="reminder ran", metadata={"_session_key": session_key})

    bridge = OhmoGatewayBridge(
        bus=bus,
        runtime_pool=FakeRuntimePool(),
        workspace=tmp_path,
        feishu_group_policy="mention",
    )
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(
                channel="feishu",
                sender_id="__scheduler__",
                chat_id="oc_unmanaged",
                content="send the weather",
                session_key_override="feishu:oc_unmanaged",
                metadata={"chat_type": "group", "_synthetic": True, "_reminder_id": "rem-1"},
            )
        )
        final = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert len(calls) == 1
    assert final.content == "reminder ran"


@pytest.mark.asyncio
async def test_gateway_bridge_processes_managed_feishu_group_without_mention(tmp_path):
    bus = MessageBus()
    save_managed_group_record(
        workspace=tmp_path,
        channel="feishu",
        chat_id="oc_managed",
        owner_open_id="ou_owner",
        name="Managed Group",
    )

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            yield SimpleNamespace(kind="final", text="managed done", metadata={"_session_key": session_key})

    bridge = OhmoGatewayBridge(
        bus=bus,
        runtime_pool=FakeRuntimePool(),
        workspace=tmp_path,
        feishu_group_policy="managed_or_mention",
    )
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(
                channel="feishu",
                sender_id="ou_1",
                chat_id="oc_managed",
                content="不用 @ 也应该处理",
                metadata={"chat_type": "group", "mentions_bot": False},
            )
        )
        final = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert final.content == "managed done"


@pytest.mark.asyncio
async def test_gateway_bridge_processes_mentioned_unmanaged_feishu_group(tmp_path):
    bus = MessageBus()

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            yield SimpleNamespace(kind="final", text="mentioned done", metadata={"_session_key": session_key})

    bridge = OhmoGatewayBridge(
        bus=bus,
        runtime_pool=FakeRuntimePool(),
        workspace=tmp_path,
        feishu_group_policy="managed_or_mention",
    )
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(
                channel="feishu",
                sender_id="ou_1",
                chat_id="oc_random",
                content="@ohmo 帮我看下",
                metadata={"chat_type": "group", "mentions_bot": True},
            )
        )
        final = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert final.content == "mentioned done"


@pytest.mark.asyncio
async def test_gateway_bridge_mention_policy_overrides_managed_feishu_group(tmp_path):
    bus = MessageBus()
    calls: list[InboundMessage] = []
    save_managed_group_record(
        workspace=tmp_path,
        channel="feishu",
        chat_id="oc_managed",
        owner_open_id="ou_owner",
        name="Managed Group",
    )

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            calls.append(message)
            yield SimpleNamespace(kind="final", text="should not happen", metadata={"_session_key": session_key})

    bridge = OhmoGatewayBridge(
        bus=bus,
        runtime_pool=FakeRuntimePool(),
        workspace=tmp_path,
        feishu_group_policy="mention",
    )
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(
                channel="feishu",
                sender_id="ou_1",
                chat_id="oc_managed",
                content="mention policy 下没有 @ 不应该处理",
                metadata={"chat_type": "group", "mentions_bot": False},
            )
        )
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(bus.consume_outbound(), timeout=0.1)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert calls == []


@pytest.mark.asyncio
async def test_gateway_bridge_logs_inbound_and_final(caplog):
    bus = MessageBus()

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            yield SimpleNamespace(kind="progress", text="🤔 想一想…", metadata={"_progress": True, "_session_key": session_key})
            yield SimpleNamespace(kind="final", text="Done", metadata={"_session_key": session_key})

    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=FakeRuntimePool())
    task = asyncio.create_task(bridge.run())
    caplog.set_level(logging.INFO)
    try:
        await bus.publish_inbound(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="please translate this")
        )
        await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    finally:
        bridge.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert "ohmo inbound received" in caplog.text
    assert "ohmo outbound final" in caplog.text
    assert "please translate this" in caplog.text


@pytest.mark.asyncio
async def test_gateway_bridge_stop_command_cancels_current_session():
    bus = MessageBus()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            try:
                yield SimpleNamespace(kind="progress", text="🤔 想一想…", metadata={"_progress": True, "_session_key": session_key})
                await release.wait()
                yield SimpleNamespace(kind="final", text="Done", metadata={"_session_key": session_key})
            except asyncio.CancelledError:
                cancelled.set()
                raise

    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=FakeRuntimePool())
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="long task")
        )
        first = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert first.metadata["_progress"] is True
        await bus.publish_inbound(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="/stop")
        )
        stopped = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert stopped.content.startswith("\u23f9\ufe0f")  # localization-robust: stop notice


@pytest.mark.asyncio
async def test_gateway_bridge_restart_command_requests_gateway_restart():
    bus = MessageBus()
    restarted = asyncio.Event()
    restart_payloads: list[tuple[str, str, str]] = []

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            if False:
                yield

    async def fake_restart(message, session_key: str) -> None:
        restart_payloads.append((message.channel, message.chat_id, session_key))
        restarted.set()

    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=FakeRuntimePool(), restart_gateway=fake_restart)
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="/restart")
        )
        restarting = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        await asyncio.wait_for(restarted.wait(), timeout=1.0)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert restarting.content == (
        "🔄 正在重启 gateway，马上回来。\n"
        "Restarting the gateway now. I'll be back in a moment."
    )
    assert restart_payloads == [("feishu", "c1", "feishu:c1")]


@pytest.mark.asyncio
async def test_gateway_bridge_group_command_routes_to_agent_tool_prompt():
    bus = MessageBus()
    captured: list[tuple[InboundMessage, str]] = []

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            captured.append((message, session_key))
            yield SimpleNamespace(kind="final", text="agent handled group request", metadata={})

    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=FakeRuntimePool())
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(
                channel="feishu",
                sender_id="ou_user",
                chat_id="ou_user",
                content="/group 帮我创建一个群聊专门处理HKUDS/OpenHarness的问题吧，绑定cwd就在~/OpenHarness-new",
                metadata={"chat_type": "p2p", "sender_display_name": "Tang"},
            )
        )
        reply = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert reply.content == "agent handled group request"
    assert len(captured) == 1
    message, session_key = captured[0]
    assert session_key == "feishu:ou_user"
    assert "ohmo_create_feishu_group" in message.content
    assert "HKUDS/OpenHarness" in message.content
    assert "~/OpenHarness-new" in message.content
    assert message.metadata["_ohmo_group_command"] is True
    assert message.metadata["_ohmo_group_raw_request"].startswith("帮我创建一个群聊")


@pytest.mark.asyncio
async def test_ohmo_create_feishu_group_tool_creates_and_binds_metadata(tmp_path):
    project = tmp_path / "OpenHarness-new"
    project.mkdir()
    created: list[tuple[str, str]] = []
    welcomes: list[tuple[str, str, str]] = []

    async def fake_create_group(user_open_id: str, name: str) -> str:
        created.append((user_open_id, name))
        return "oc_project_group"

    async def fake_publish_welcome(chat_id: str, content: str, owner_open_id: str) -> None:
        welcomes.append((chat_id, content, owner_open_id))

    tool = OhmoCreateFeishuGroupTool(
        workspace=tmp_path,
        create_group=fake_create_group,
        publish_group_welcome=fake_publish_welcome,
    )
    result = await tool.execute(
        OhmoCreateFeishuGroupInput(
            name="HKUDS/OpenHarness",
            cwd=str(project),
            repo="HKUDS/OpenHarness",
            reason="The request names the repo and cwd directly.",
        ),
        ToolExecutionContext(
            cwd=tmp_path,
            metadata={
                "ohmo_group_request": {
                    "channel": "feishu",
                    "chat_type": "p2p",
                    "sender_id": "ou_user",
                    "source_chat_id": "ou_user",
                    "source_session_key": "feishu:ou_user",
                    "sender_display_name": "Tang",
                    "raw_request": "帮我创建 OpenHarness 群",
                    "used": False,
                }
            },
        ),
    )

    assert result.is_error is False
    assert created == [("ou_user", "HKUDS/OpenHarness")]
    assert len(welcomes) == 1
    assert welcomes[0][0] == "oc_project_group"
    assert welcomes[0][2] == "ou_user"
    assert "已绑定工作目录" in welcomes[0][1]
    record = load_managed_group_record(workspace=tmp_path, channel="feishu", chat_id="oc_project_group")
    assert record is not None
    assert record["owner_open_id"] == "ou_user"
    assert record["name"] == "HKUDS/OpenHarness"
    assert record["cwd"] == str(project.resolve())
    assert record["repo"] == "HKUDS/OpenHarness"
    assert record["binding_status"] == "bound"


@pytest.mark.asyncio
async def test_agent_loop_can_create_group_via_ohmo_group_tool(tmp_path):
    project = tmp_path / "ClawTeam"
    project.mkdir()
    created: list[tuple[str, str]] = []

    class FakeApiClient:
        def __init__(self):
            self.requests = []
            self.responses = [
                ConversationMessage(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="toolu_group",
                            name="ohmo_create_feishu_group",
                            input={
                                "name": "HKUDS/ClawTeam",
                                "cwd": str(project),
                                "repo": "HKUDS/ClawTeam",
                                "reason": "The user asked for a ClawTeam project group.",
                            },
                        )
                    ],
                ),
                ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text="已创建 HKUDS/ClawTeam 群。")],
                ),
            ]

        async def stream_message(self, request):
            self.requests.append(request)
            yield ApiMessageCompleteEvent(
                message=self.responses.pop(0),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                stop_reason=None,
            )

    async def fake_create_group(user_open_id: str, name: str) -> str:
        created.append((user_open_id, name))
        return "oc_clawteam"

    registry = ToolRegistry()
    registry.register(OhmoCreateFeishuGroupTool(workspace=tmp_path, create_group=fake_create_group))
    client = FakeApiClient()
    engine = QueryEngine(
        api_client=client,
        tool_registry=registry,
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.DEFAULT)),
        cwd=tmp_path,
        model="fake-model",
        system_prompt="system",
        tool_metadata={
            "ohmo_group_request": {
                "channel": "feishu",
                "chat_type": "p2p",
                "sender_id": "ou_user",
                "source_chat_id": "ou_user",
                "source_session_key": "feishu:ou_user",
                "raw_request": "帮我创建一个群聊专门处理HKUDS/ClawTeam的问题吧",
                "used": False,
            }
        },
    )

    events = [event async for event in engine.submit_message("create group")]

    completed = [event for event in events if isinstance(event, ToolExecutionCompleted)]
    assert len(completed) == 1
    assert completed[0].tool_name == "ohmo_create_feishu_group"
    assert completed[0].is_error is False
    assert created == [("ou_user", "HKUDS/ClawTeam")]
    assert "ohmo_create_feishu_group" in {tool["name"] for tool in client.requests[0].tools}
    record = load_managed_group_record(workspace=tmp_path, channel="feishu", chat_id="oc_clawteam")
    assert record is not None
    assert record["cwd"] == str(project.resolve())
    assert record["repo"] == "HKUDS/ClawTeam"


@pytest.mark.asyncio
async def test_ohmo_create_feishu_group_tool_rejects_without_slash_group_context(tmp_path):
    async def fake_create_group(user_open_id: str, name: str) -> str:
        raise AssertionError("tool should reject before creating a group")

    tool = OhmoCreateFeishuGroupTool(workspace=tmp_path, create_group=fake_create_group)
    result = await tool.execute(
        OhmoCreateFeishuGroupInput(name="HKUDS/OpenHarness"),
        ToolExecutionContext(cwd=tmp_path, metadata={}),
    )

    assert result.is_error is True
    assert "only run immediately" in result.output


@pytest.mark.asyncio
async def test_gateway_bridge_group_command_rejects_group_chat(tmp_path):
    bus = MessageBus()

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            raise AssertionError("/group should not enter the agent runtime")

    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=FakeRuntimePool())
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(
                channel="feishu",
                sender_id="ou_user",
                chat_id="oc_existing",
                content="/group New Group",
                metadata={"chat_type": "group"},
            )
        )
        reply = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert "私聊" in reply.content


@pytest.mark.asyncio
async def test_gateway_bridge_group_command_without_details_still_routes_to_agent():
    bus = MessageBus()
    captured: list[InboundMessage] = []

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            del session_key
            captured.append(message)
            yield SimpleNamespace(kind="final", text="need details", metadata={})

    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=FakeRuntimePool())
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(
                channel="feishu",
                sender_id="ou_user",
                chat_id="ou_user",
                content="/group",
                metadata={"chat_type": "p2p"},
            )
        )
        reply = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert reply.content == "need details"
    assert "(user did not provide details)" in captured[0].content


@pytest.mark.asyncio
async def test_gateway_service_request_restart_waits_before_stop(monkeypatch):
    service = object.__new__(OhmoGatewayService)
    service._restart_requested = False
    service._stop_event = asyncio.Event()
    service._workspace = "/tmp/ohmo"

    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr("ohmo.gateway.service.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(
        "ohmo.gateway.service.get_gateway_restart_notice_path",
        lambda workspace: Path("/tmp/restart-notice.json"),
    )
    writes: list[str] = []
    monkeypatch.setattr(
        "pathlib.Path.write_text",
        lambda self, content, encoding=None: writes.append(content) or len(content),
    )

    message = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="/restart")

    await OhmoGatewayService.request_restart(service, message, "feishu:c1")

    assert service._restart_requested is True
    assert service._stop_event.is_set() is True
    assert slept == [0.75]
    assert writes


@pytest.mark.asyncio
async def test_gateway_service_publishes_pending_restart_notice(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    notice_path = get_gateway_restart_notice_path(workspace)
    notice_path.write_text(
        json.dumps(
            {
                "channel": "feishu",
                "chat_id": "chat-1",
                "session_key": "feishu:chat-1",
                "content": "✅ gateway 已经重新连上，可以继续了。\nGateway is back online. We can continue.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    service = object.__new__(OhmoGatewayService)
    service._workspace = workspace
    service._bus = MessageBus()

    async def fake_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr("ohmo.gateway.service.asyncio.sleep", fake_sleep)

    await OhmoGatewayService._publish_pending_restart_notice(service)

    outbound = await asyncio.wait_for(service._bus.consume_outbound(), timeout=1.0)
    assert outbound.content == "✅ gateway 已经重新连上，可以继续了。\nGateway is back online. We can continue."
    assert outbound.chat_id == "chat-1"
    assert not notice_path.exists()


@pytest.mark.asyncio
async def test_gateway_bridge_persists_interrupted_requests_on_stop(tmp_path):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):  # pragma: no cover
            if False:
                yield None

    bridge = OhmoGatewayBridge(
        bus=MessageBus(), runtime_pool=FakeRuntimePool(), workspace=workspace
    )
    # An in-flight dispatched turn, a coalesce-buffered message, and a synthetic
    # reminder (which must be excluded — it is not a user request to recover).
    bridge._inflight["telegram:c1"] = InboundMessage(
        channel="telegram", sender_id="u1", chat_id="c1", content="поставь 1-1 с Дарьей"
    )
    bridge._pending["telegram:c2"] = [
        InboundMessage(channel="telegram", sender_id="u1", chat_id="c2", content="buffered q")
    ]
    bridge._inflight["telegram:sched"] = InboundMessage(
        channel="telegram",
        sender_id="__scheduler__",
        chat_id="c3",
        content="reminder fire",
        metadata={"_synthetic": True},
    )

    bridge.stop()

    records = json.loads(
        get_gateway_interrupted_requests_path(workspace).read_text(encoding="utf-8")
    )
    contents = {r["content"] for r in records}
    assert "поставь 1-1 с Дарьей" in contents
    assert "buffered q" in contents
    assert "reminder fire" not in contents


@pytest.mark.asyncio
async def test_gateway_bridge_persist_skips_when_nothing_in_flight(tmp_path):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    bridge = OhmoGatewayBridge(
        bus=MessageBus(), runtime_pool=object(), workspace=workspace
    )
    bridge.stop()
    assert not get_gateway_interrupted_requests_path(workspace).exists()


@pytest.mark.asyncio
async def test_gateway_service_publishes_interrupted_requests_notice(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    get_gateway_interrupted_requests_path(workspace).write_text(
        json.dumps(
            [
                {
                    "channel": "telegram",
                    "chat_id": "c1",
                    "session_key": "telegram:c1",
                    "content": "поставь 1-1 с Дарьей",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    service = object.__new__(OhmoGatewayService)
    service._workspace = workspace
    service._bus = MessageBus()

    async def fake_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr("ohmo.gateway.service.asyncio.sleep", fake_sleep)

    await OhmoGatewayService._publish_interrupted_requests_notice(service)

    outbound = await asyncio.wait_for(service._bus.consume_outbound(), timeout=1.0)
    assert "перезапустился" in outbound.content
    assert "поставь 1-1 с Дарьей" in outbound.content
    assert outbound.chat_id == "c1"
    assert not get_gateway_interrupted_requests_path(workspace).exists()


@pytest.mark.asyncio
async def test_gateway_bridge_new_message_interrupts_same_session():
    bus = MessageBus()
    first_cancelled = asyncio.Event()
    second_started = asyncio.Event()

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            if message.content == "first":
                try:
                    yield SimpleNamespace(kind="progress", text="🤔 想一想…", metadata={"_progress": True, "_session_key": session_key})
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    first_cancelled.set()
                    raise
            else:
                second_started.set()
                yield SimpleNamespace(kind="final", text="second-done", metadata={"_session_key": session_key})

    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=FakeRuntimePool())
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="first")
        )
        await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        await bus.publish_inbound(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="second")
        )
        interrupted = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        final = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        await asyncio.wait_for(first_cancelled.wait(), timeout=1.0)
        await asyncio.wait_for(second_started.wait(), timeout=1.0)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert interrupted.content.startswith("\u23f9\ufe0f")  # localization-robust: interrupt notice
    assert final.content == "second-done"


@pytest.mark.asyncio
async def test_runtime_pool_logs_session_lifecycle(tmp_path, monkeypatch, caplog):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)

    async def fake_build_runtime(**kwargs):
        class FakeEngine:
            messages = []
            total_usage = UsageSnapshot()

            def set_system_prompt(self, prompt):
                return None

            async def submit_message(self, content):
                yield ToolExecutionStarted(tool_name="web_fetch", tool_input={"url": "https://example.com"})
                yield AssistantTextDelta(text="done")

        return SimpleNamespace(
            engine=FakeEngine(),
            session_id="sess123",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=SimpleNamespace(lookup=lambda raw: None),
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="check")
    caplog.set_level(logging.INFO)
    updates = [u async for u in pool.stream_message(message, "feishu:c1")]

    assert updates[-1].text == "done"
    assert "ohmo runtime processing start" in caplog.text
    assert "ohmo runtime tool start" in caplog.text
    assert "ohmo runtime saved snapshot" in caplog.text
    assert "ohmo runtime processing complete" in caplog.text


def test_gateway_provider_command_uses_ohmo_gateway_profile(tmp_path, monkeypatch):
    workspace = initialize_workspace(tmp_path / ".ohmo-home")
    save_gateway_config(GatewayConfig(provider_profile="kimi-anthropic"), workspace)

    statuses = {
        "codex": {
            "label": "Codex subscription",
            "configured": True,
            "base_url": None,
            "model": "gpt-5.4",
        },
        "kimi-anthropic": {
            "label": "Kimi Anthropic",
            "configured": True,
            "base_url": "https://api.example.test",
            "model": "kimi-k2.5",
        },
    }

    class FakeAuthManager:
        def __init__(self, settings):
            del settings

        def get_profile_statuses(self):
            return statuses

    monkeypatch.setattr("ohmo.gateway.provider_commands.load_settings", lambda: object())
    monkeypatch.setattr("ohmo.gateway.provider_commands.AuthManager", FakeAuthManager)

    text, refresh = handle_gateway_provider_command("list", workspace=workspace)
    assert refresh is False
    assert "ohmo gateway provider profiles:" in text
    assert "* kimi-anthropic [ready]" in text
    assert "  codex [ready]" in text

    text, refresh = handle_gateway_provider_command("codex", workspace=workspace)
    assert refresh is True
    assert "provider_profile set to codex" in text
    assert load_gateway_config(workspace).provider_profile == "codex"


def test_gateway_model_command_updates_selected_gateway_profile(tmp_path, monkeypatch):
    workspace = initialize_workspace(tmp_path / ".ohmo-home")
    save_gateway_config(GatewayConfig(provider_profile="codex"), workspace)
    profile = ProviderProfile(
        label="Codex",
        provider="openai_codex",
        api_format="responses",
        auth_source="codex_subscription",
        default_model="gpt-5.4",
        allowed_models=["gpt-5.4", "gpt-5.5"],
    )
    updates: list[tuple[str, dict[str, object]]] = []

    class FakeAuthManager:
        def __init__(self, settings):
            del settings

        def list_profiles(self):
            return {"codex": profile}

        def update_profile(self, name, **kwargs):
            nonlocal profile
            updates.append((name, kwargs))
            profile = profile.model_copy(update={key: value for key, value in kwargs.items() if value is not None})

    monkeypatch.setattr("ohmo.gateway.provider_commands.load_settings", lambda: object())
    monkeypatch.setattr("ohmo.gateway.provider_commands.AuthManager", FakeAuthManager)

    text, refresh = handle_gateway_model_command("show", workspace=workspace)
    assert refresh is False
    assert "ohmo gateway model: gpt-5.4" in text
    assert "Profile: codex" in text

    text, refresh = handle_gateway_model_command("list", workspace=workspace)
    assert refresh is False
    assert "- gpt-5.4" in text
    assert "- gpt-5.5" in text

    text, refresh = handle_gateway_model_command("gpt-5.5", workspace=workspace)
    assert refresh is True
    assert "model set to gpt-5.5" in text
    assert updates[-1] == ("codex", {"last_model": "gpt-5.5"})


def test_runtime_pool_only_exposes_group_tool_for_group_command_turn(tmp_path):
    async def fake_create_group(user_open_id: str, name: str) -> str:
        del user_open_id, name
        return "oc_group"

    workspace = initialize_workspace(tmp_path / ".ohmo-home")
    pool = OhmoSessionRuntimePool(
        cwd=tmp_path,
        workspace=workspace,
        provider_profile="codex",
        create_feishu_group=fake_create_group,
    )
    bundle = SimpleNamespace(
        engine=SimpleNamespace(tool_metadata={}),
        tool_registry=ToolRegistry(),
    )

    pool._set_group_request_context(
        bundle,
        InboundMessage(channel="feishu", sender_id="u1", chat_id="u1", content="hello"),
        "feishu:u1",
    )
    assert bundle.tool_registry.get("ohmo_create_feishu_group") is None

    previous = pool._set_group_request_context(
        bundle,
        InboundMessage(
            channel="feishu",
            sender_id="u1",
            chat_id="u1",
            content="create",
            metadata={"_ohmo_group_command": True, "_ohmo_group_raw_request": "创建项目群", "chat_type": "p2p"},
        ),
        "feishu:u1",
    )
    assert bundle.tool_registry.get("ohmo_create_feishu_group") is not None
    assert bundle.engine.tool_metadata.get("_suppress_next_user_goal") is True

    pool._restore_group_request_context(bundle, previous)
    assert "ohmo_group_request" not in bundle.engine.tool_metadata
    assert "_suppress_next_user_goal" not in bundle.engine.tool_metadata
    assert bundle.tool_registry.get("ohmo_create_feishu_group") is None


def test_runtime_pool_sanitizes_internal_group_prompt_history():
    messages = _sanitize_group_command_prompts([
        ConversationMessage.from_user_text(
            "The user invoked `/group` from a Feishu private chat.\n"
            "Your task is to create a dedicated Feishu group for this request.\n\n"
            "Use the `ohmo_create_feishu_group` tool exactly once.\n\n"
            "User /group request:\n"
            "帮我创建一个群聊专门处理novix-monorepo的问题，绑定cwd在~/novix-monorepo"
        ),
        ConversationMessage.from_user_text("你现在使用的是什么模型"),
    ])

    first_text = messages[0].text
    assert "Use the `ohmo_create_feishu_group` tool exactly once" not in first_text
    assert "[Handled /group request]" in first_text
    assert "novix-monorepo" in first_text
    assert messages[1].text == "你现在使用的是什么模型"


def test_runtime_pool_sanitizes_internal_group_prompt_metadata():
    internal_prompt = (
        "The user invoked `/group` from a Feishu private chat.\n"
        "Use the `ohmo_create_feishu_group` tool exactly once.\n\n"
        "User /group request:\n"
        "帮我创建一个群聊专门处理novix-monorepo的问题，绑定cwd在~/novix-monorepo"
    )

    metadata = _sanitize_group_command_metadata(
        {
            "task_focus_state": {
                "goal": internal_prompt,
                "recent_goals": [internal_prompt, "你现在使用的是什么模型"],
                "next_step": internal_prompt,
            },
            "recent_work_log": [internal_prompt],
            "mcp_manager": object(),
        }
    )

    focus = metadata["task_focus_state"]
    rendered = "\n".join([focus["goal"], *focus["recent_goals"], focus["next_step"], *metadata["recent_work_log"]])
    assert "Use the `ohmo_create_feishu_group` tool exactly once" not in rendered
    assert "[Handled /group request]" in rendered
    assert "你现在使用的是什么模型" in rendered
    assert "mcp_manager" in metadata


@pytest.mark.asyncio
async def test_runtime_pool_provider_command_refresh_uses_gateway_profile(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    save_gateway_config(GatewayConfig(provider_profile="kimi-anthropic"), workspace)
    build_calls: list[dict[str, object]] = []

    statuses = {
        "codex": {
            "label": "Codex subscription",
            "configured": True,
            "base_url": None,
            "model": "gpt-5.4",
        },
        "kimi-anthropic": {
            "label": "Kimi Anthropic",
            "configured": True,
            "base_url": "https://api.example.test",
            "model": "kimi-k2.5",
        },
    }

    class FakeAuthManager:
        def __init__(self, settings):
            del settings

        def get_profile_statuses(self):
            return statuses

    class FakeEngine:
        def __init__(self):
            self.messages = [ConversationMessage.from_user_text("before")]
            self.total_usage = UsageSnapshot()

        def set_system_prompt(self, prompt):
            del prompt

        async def submit_message(self, content):
            del content
            if False:
                yield None

    async def fake_build_runtime(**kwargs):
        build_calls.append(kwargs)
        return SimpleNamespace(
            engine=FakeEngine(),
            session_id="sess123",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=create_default_command_registry(),
            hook_summary=lambda: "",
            mcp_summary=lambda: "",
            plugin_summary=lambda: "",
            cwd=str(tmp_path),
            tool_registry=None,
            app_state=None,
            session_backend=None,
            extra_skill_dirs=(),
            extra_plugin_roots=(),
            enforce_max_turns=False,
        )

    async def fake_start_runtime(bundle):
        del bundle

    async def fake_close_runtime(bundle):
        del bundle

    monkeypatch.setattr("ohmo.gateway.provider_commands.load_settings", lambda: object())
    monkeypatch.setattr("ohmo.gateway.provider_commands.AuthManager", FakeAuthManager)
    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.close_runtime", fake_close_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="kimi-anthropic")
    message = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="/provider codex")
    updates = [u async for u in pool.stream_message(message, "feishu:c1")]

    assert updates[-1].text.startswith("ohmo gateway provider_profile set to codex")
    assert build_calls[0]["active_profile"] == "kimi-anthropic"
    assert build_calls[1]["active_profile"] == "codex"


@pytest.mark.asyncio
async def test_runtime_pool_stream_message_handles_slash_command_and_refresh_runtime(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    build_calls: list[dict[str, object]] = []
    close_calls: list[str] = []

    class FakeEngine:
        def __init__(self):
            self.messages = [ConversationMessage.from_user_text("before")]
            self.total_usage = UsageSnapshot()
            self.system_prompts: list[str] = []

        def set_system_prompt(self, prompt):
            self.system_prompts.append(prompt)

        async def submit_message(self, content):
            yield AssistantTextDelta(text="done")

    class FakeCommand:
        async def handler(self, args, context):
            assert args == ""
            return CommandResult(message="Permission mode set to plan", refresh_runtime=True)

    async def fake_build_runtime(**kwargs):
        build_calls.append(kwargs)
        engine = FakeEngine()
        return SimpleNamespace(
            engine=engine,
            session_id="sess123",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=SimpleNamespace(lookup=lambda raw: (FakeCommand(), "") if raw == "/plan" else None),
            hook_summary=lambda: "",
            mcp_summary=lambda: "",
            plugin_summary=lambda: "",
            cwd=str(tmp_path),
            tool_registry=None,
            app_state=None,
            session_backend=None,
            extra_skill_dirs=(),
            extra_plugin_roots=(),
            enforce_max_turns=False,
        )

    async def fake_start_runtime(bundle):
        return None

    async def fake_close_runtime(bundle):
        close_calls.append(bundle.session_id)

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.close_runtime", fake_close_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="/plan")
    updates = [u async for u in pool.stream_message(message, "feishu:c1")]

    assert [u.text for u in updates] == ["Permission mode set to plan"]
    assert len(build_calls) == 2
    assert close_calls == ["sess123"]
    assert build_calls[1]["restore_messages"] == [ConversationMessage.from_user_text("before").model_dump(mode="json")]


@pytest.mark.asyncio
async def test_runtime_pool_refresh_runtime_drops_dangling_tool_use_tail(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    build_calls: list[dict[str, object]] = []

    class FakeEngine:
        def __init__(self):
            self.messages = [
                ConversationMessage.from_user_text("before"),
                ConversationMessage(
                    role="assistant",
                    content=[ToolUseBlock(id="write_file:234", name="write_file", input={"path": "x"})],
                ),
            ]
            self.total_usage = UsageSnapshot()

        def set_system_prompt(self, prompt):
            del prompt
            return None

        async def submit_message(self, content):
            del content
            if False:
                yield None

    class FakeCommand:
        async def handler(self, args, context):
            del args, context
            return CommandResult(message="Switched provider profile", refresh_runtime=True)

    async def fake_build_runtime(**kwargs):
        build_calls.append(kwargs)
        return SimpleNamespace(
            engine=FakeEngine(),
            session_id="sess123",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=SimpleNamespace(lookup=lambda raw: (FakeCommand(), "") if raw == "/provider github" else None),
            hook_summary=lambda: "",
            mcp_summary=lambda: "",
            plugin_summary=lambda: "",
            cwd=str(tmp_path),
            tool_registry=None,
            app_state=None,
            session_backend=None,
            extra_skill_dirs=(),
            extra_plugin_roots=(),
            enforce_max_turns=False,
        )

    async def fake_start_runtime(bundle):
        del bundle
        return None

    async def fake_close_runtime(bundle):
        del bundle
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.close_runtime", fake_close_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="/provider github")
    _ = [u async for u in pool.stream_message(message, "feishu:c1")]

    assert len(build_calls) == 2
    assert build_calls[1]["restore_messages"] == [ConversationMessage.from_user_text("before").model_dump(mode="json")]


@pytest.mark.asyncio
async def test_runtime_pool_stream_message_handles_plugin_command_submit_prompt(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    submitted: list[object] = []

    class FakeEngine:
        messages = []
        total_usage = UsageSnapshot()
        model = "gpt-5.4"

        def set_system_prompt(self, prompt):
            return None

        def set_model(self, model):
            self.model = model

        async def submit_message(self, content):
            submitted.append(content)
            yield AssistantTextDelta(text="plugin-done")

    class FakeCommand:
        async def handler(self, args, context):
            assert args == "hello"
            return CommandResult(submit_prompt="plugin expanded prompt")

    async def fake_build_runtime(**kwargs):
        return SimpleNamespace(
            engine=FakeEngine(),
            session_id="sess123",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=SimpleNamespace(lookup=lambda raw: (FakeCommand(), "hello") if raw == "/plugin-cmd hello" else None),
            hook_summary=lambda: "",
            mcp_summary=lambda: "",
            plugin_summary=lambda: "",
            cwd=str(tmp_path),
            tool_registry=None,
            app_state=None,
            session_backend=None,
            extra_skill_dirs=(),
            extra_plugin_roots=(),
            enforce_max_turns=False,
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="/plugin-cmd hello")
    updates = [u async for u in pool.stream_message(message, "feishu:c1")]

    assert submitted == ["plugin expanded prompt"]
    assert updates[-1].text == "plugin-done"


@pytest.mark.asyncio
async def test_runtime_pool_parses_group_slash_command_before_speaker_context(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)

    class FakeEngine:
        messages = []
        total_usage = UsageSnapshot()
        model = "gpt-5.4"

        def set_system_prompt(self, prompt):
            return None

        async def submit_message(self, content):
            raise AssertionError("group /skills should be handled by the command layer")

    async def fake_build_runtime(**kwargs):
        return SimpleNamespace(
            engine=FakeEngine(),
            session_id="sess123",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=create_default_command_registry(),
            hook_summary=lambda: "",
            mcp_summary=lambda: "",
            plugin_summary=lambda: "",
            cwd=str(tmp_path),
            tool_registry=None,
            app_state=None,
            session_backend=None,
            extra_skill_dirs=(),
            extra_plugin_roots=(),
            enforce_max_turns=False,
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(
        channel="feishu",
        sender_id="u1",
        chat_id="c1",
        content="/skills",
        metadata={"chat_type": "group", "sender_display_name": "Tester"},
    )
    updates = [u async for u in pool.stream_message(message, "feishu:c1:u1")]

    assert len(updates) == 1
    assert updates[0].kind == "final"
    assert updates[0].text.startswith("Available skills:")


@pytest.mark.asyncio
async def test_runtime_pool_stream_message_handles_ohmo_skill_slash_command(tmp_path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    skill_dir = get_skills_dir(workspace) / "quick-note"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: Quick Note\n"
        "description: Capture a concise note.\n"
        "---\n\n"
        "# Quick Note\n\n"
        "Summarize this: $ARGUMENTS\n",
        encoding="utf-8",
    )
    submitted: list[object] = []

    class FakeEngine:
        messages = []
        total_usage = UsageSnapshot()
        model = "gpt-5.4"

        def set_system_prompt(self, prompt):
            return None

        def set_model(self, model):
            self.model = model

        async def submit_message(self, content):
            submitted.append(content)
            yield AssistantTextDelta(text="skill-done")

    async def fake_build_runtime(**kwargs):
        return SimpleNamespace(
            engine=FakeEngine(),
            session_id="sess123",
            current_settings=lambda: SimpleNamespace(model="gpt-5.4"),
            commands=SimpleNamespace(lookup=lambda raw: None),
            hook_summary=lambda: "",
            mcp_summary=lambda: "",
            plugin_summary=lambda: "",
            cwd=str(tmp_path),
            tool_registry=None,
            app_state=None,
            session_backend=None,
            extra_skill_dirs=(str(get_skills_dir(workspace)),),
            extra_plugin_roots=(),
            enforce_max_turns=False,
        )

    async def fake_start_runtime(bundle):
        return None

    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    message = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="/quick-note hello")
    updates = [u async for u in pool.stream_message(message, "feishu:c1")]

    assert len(submitted) == 1
    assert isinstance(submitted[0], str)
    assert f"Base directory for this skill: {skill_dir.resolve()}" in submitted[0]
    assert "Summarize this: hello" in submitted[0]
    assert updates[-1].text == "skill-done"


@pytest.mark.asyncio
async def test_reset_session_pops_bundle_and_clears_pointer(tmp_path, monkeypatch):
    import ohmo.gateway.runtime as rt

    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    pool._session_backend.save_snapshot(
        cwd=tmp_path,
        model="gpt-5.5",
        system_prompt="s",
        messages=[ConversationMessage.from_user_text("hi")],
        usage=UsageSnapshot(),
        session_id="sid",
        session_key="telegram:7",
    )
    assert pool._session_backend.load_latest_for_session_key("telegram:7") is not None

    closed = {}

    async def fake_close(bundle):
        closed["bundle"] = bundle

    monkeypatch.setattr(rt, "close_runtime", fake_close)
    sentinel = object()
    pool._bundles["telegram:7"] = sentinel

    had = await pool.reset_session("telegram:7")

    assert had is True
    assert closed["bundle"] is sentinel               # bundle closed in-task
    assert "telegram:7" not in pool._bundles          # dropped from memory
    assert pool._session_backend.load_latest_for_session_key("telegram:7") is None  # pointer gone
    # NB: /new no longer wipes a shared TODO.md — lists are per-session_id, so the
    # next message's fresh session_id starts a clean list (old file kept on disk).
    assert await pool.reset_session("telegram:absent") is False  # unknown session is safe


@pytest.mark.asyncio
async def test_gateway_bridge_new_command_resets_session():
    bus = MessageBus()

    class FakeRuntimePool:
        def __init__(self):
            self.reset_calls = []

        async def reset_session(self, session_key):
            self.reset_calls.append(session_key)
            return True

        async def stream_message(self, message, session_key):
            yield SimpleNamespace(kind="final", text="SHOULD-NOT-RUN", metadata={})

    pool = FakeRuntimePool()
    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=pool)
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/new")
        )
        reply = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert pool.reset_calls            # /new actually invoked reset_session...
    assert "сброшен" in reply.content.lower()  # ...and confirmed, not run through the model


def test_extract_attachments_strips_markers_and_keeps_existing_files(tmp_path):
    from ohmo.gateway.bridge import _extract_attachments

    existing = tmp_path / "report.html"
    existing.write_text("<html></html>")
    missing = tmp_path / "nope.html"
    text = f"Готово, отчёт во вложении.\n[[attach: {existing}]] хвост [[attach:{missing}]]"

    clean, media = _extract_attachments(text)

    assert media == [str(existing)]          # only the existing file is attached
    assert "[[attach" not in clean           # every marker stripped (incl. the missing one)
    assert "Готово" in clean and "хвост" in clean


def test_extract_attachments_passthrough_without_markers():
    from ohmo.gateway.bridge import _extract_attachments

    clean, media = _extract_attachments("just text, no files")
    assert clean == "just text, no files"
    assert media == []


def test_extract_attachments_resolves_relative_against_base_dir(tmp_path):
    from ohmo.gateway.bridge import _extract_attachments

    (tmp_path / "diagram.html").write_text("<html></html>")
    # a relative path resolves against base_dir (the session cwd / per-chat work dir)
    clean, media = _extract_attachments("Готово.\n[[attach: diagram.html]]", base_dir=tmp_path)
    assert media == [str(tmp_path / "diagram.html")]
    assert "[[attach" not in clean
    # without base_dir a relative path is NOT found (resolves against process cwd)
    _clean2, media2 = _extract_attachments("[[attach: diagram.html]]")
    assert media2 == []
    # an absolute path is unaffected by base_dir
    _clean3, media3 = _extract_attachments(
        f"[[attach: {tmp_path / 'diagram.html'}]]", base_dir="/nonexistent"
    )
    assert media3 == [str(tmp_path / "diagram.html")]


def test_extract_attachments_dedupes_same_file(tmp_path):
    from ohmo.gateway.bridge import _extract_attachments

    f = tmp_path / "a.html"
    f.write_text("x")
    clean, media = _extract_attachments(f"[[attach:{f}]] [[attach:  {f} ]]")
    assert media == [str(f)]                 # same path referenced twice -> one attachment
    assert clean == ""


def test_extract_ask_parses_question_and_options():
    from ohmo.gateway.bridge import _extract_ask

    clean, question, options = _extract_ask(
        "Прикинул варианты.\n[[ask: Куда едем? | Питер | Москва | Дома]]"
    )
    assert clean == "Прикинул варианты."          # marker stripped from visible text
    assert question == "Куда едем?"
    assert options == ["Питер", "Москва", "Дома"]


def test_extract_ask_without_marker_is_passthrough():
    from ohmo.gateway.bridge import _extract_ask

    clean, question, options = _extract_ask("just a normal reply")
    assert clean == "just a normal reply"
    assert question == "" and options == []


def test_extract_ask_needs_at_least_two_options():
    from ohmo.gateway.bridge import _extract_ask

    # A "question" with <2 options is no button prompt — strip it, no buttons.
    clean, question, options = _extract_ask("text [[ask: Точно? | Ок]]")
    assert "[[ask" not in clean
    assert options == [] and question == ""


def test_extract_ask_caps_option_count():
    from ohmo.gateway.bridge import _extract_ask

    inner = " | ".join(["Q"] + [f"o{i}" for i in range(12)])
    _, _, options = _extract_ask(f"[[ask: {inner}]]")
    assert len(options) == 8  # keyboard kept sane


def test_build_keyboard_one_button_per_option():
    from openharness.channels.impl.telegram import TelegramChannel

    assert TelegramChannel._build_keyboard([]) is None
    kb = TelegramChannel._build_keyboard(["Питер", "Москва"])
    flat = [b for row in kb.inline_keyboard for b in row]
    assert [b.text for b in flat] == ["Питер", "Москва"]
    assert [b.callback_data for b in flat] == ["ask:0", "ask:1"]


def test_render_todo_checklist(tmp_path):
    from ohmo.gateway.runtime import _render_todo_checklist

    todo = tmp_path / "list.md"  # now takes the path to the session's active list
    assert _render_todo_checklist(None) is None
    assert _render_todo_checklist(todo) is None  # file doesn't exist yet
    todo.write_text("# TODO\n- [ ] step one\n- [x] step two\n")

    out = _render_todo_checklist(todo)
    assert out.startswith("📋 To-do")
    assert "⬜ step one" in out
    assert "✅ step two" in out


def test_speaker_context_direct_chat_carries_telegram_handle():
    from ohmo.gateway.runtime import _build_speaker_context

    msg = InboundMessage(
        channel="telegram",
        sender_id="116870365|blackwithwhite",
        chat_id="116870365",
        content="привет",
        metadata={"chat_type": "private", "username": "blackwithwhite", "first_name": "Дмитрий"},
    )
    ctx = _build_speaker_context(msg)
    assert "Direct message from" in ctx
    assert "Дмитрий" in ctx and "@blackwithwhite" in ctx
    assert "116870365|blackwithwhite" in ctx


def test_speaker_context_group_still_labeled_as_group():
    from ohmo.gateway.runtime import _build_speaker_context

    msg = InboundMessage(
        channel="feishu",
        sender_id="ou_123",
        chat_id="c1",
        content="hi",
        metadata={"chat_type": "group", "sender_display_name": "Tang Jiabin"},
    )
    ctx = _build_speaker_context(msg)
    assert ctx.startswith("[Channel speaker]")
    assert "group chat" in ctx and "Tang Jiabin" in ctx

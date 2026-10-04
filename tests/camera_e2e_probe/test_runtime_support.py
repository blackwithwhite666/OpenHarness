from __future__ import annotations

import pytest
from types import SimpleNamespace

from camera_runtime_support import OfflineCameraBotApi, isolated_runtime_loaders
from openharness.config.settings import ProviderProfile, Settings
from openharness.engine.messages import ConversationMessage, ToolResultBlock, ToolUseBlock
from probe_support import (
    NativeClientPreconditionError,
    native_preflight_and_clients,
    native_profile_clients,
)


class FakeCodexClient:
    pass


def _native_settings() -> Settings:
    profile = ProviderProfile(
        label="Codex subscription",
        provider="openai_codex",
        api_format="openai",
        auth_source="codex_subscription",
        default_model="gpt-6-luna",
    )
    return Settings(
        active_profile="codex",
        profiles={"codex": profile},
        allow_project_skills=False,
        project_skill_dirs=[],
    )


def test_native_profile_resolves_two_distinct_codex_clients_without_fallback():
    clients = iter((FakeCodexClient(), FakeCodexClient()))
    calls = []

    def resolver(settings):
        calls.append(settings)
        return next(clients)

    bot, user = native_profile_clients(
        _native_settings(), resolver=resolver, codex_client_type=FakeCodexClient
    )

    assert len(calls) == 2
    assert bot is not user


@pytest.mark.parametrize(
    "profile",
    [
        ProviderProfile(
            label="OpenRouter",
            provider="openrouter",
            api_format="openai",
            auth_source="openrouter_api_key",
            default_model="gpt-6-luna",
        ),
        ProviderProfile(
            label="Codex subscription",
            provider="openai_codex",
            api_format="openai",
            auth_source="codex_subscription",
            default_model="gpt-5.4",
        ),
        ProviderProfile(
            label="Codex API key",
            provider="openai_codex",
            api_format="openai",
            auth_source="openai_api_key",
            default_model="gpt-6-luna",
        ),
    ],
)
def test_native_profile_rejects_wrong_provider_model_or_auth_before_resolution(profile):
    settings = Settings(active_profile="codex", profiles={"codex": profile})
    calls = []

    with pytest.raises(NativeClientPreconditionError):
        native_profile_clients(
            settings,
            resolver=lambda _settings: calls.append("resolved"),
            codex_client_type=FakeCodexClient,
        )

    assert calls == []


def test_native_profile_rejects_shared_or_non_codex_clients_without_fallback():
    shared = FakeCodexClient()
    results = iter((shared, shared))

    with pytest.raises(NativeClientPreconditionError, match="two distinct"):
        native_profile_clients(
            _native_settings(),
            resolver=lambda _settings: next(results),
            codex_client_type=FakeCodexClient,
        )

    with pytest.raises(NativeClientPreconditionError, match="two distinct"):
        native_profile_clients(
            _native_settings(),
            resolver=lambda _settings: object(),
            codex_client_type=FakeCodexClient,
        )


@pytest.mark.parametrize(
    ("scenario", "source_path", "source_sha", "settings_update"),
    [
        ("", "/missing/source.jpg", "0" * 64, {}),
        ("owner scenario", "/missing/source.jpg", "0" * 64, {}),
        ("owner scenario", "/missing/source.jpg", "0" * 64, {"mcp_servers": {"remote": {}}}),
        ("owner scenario", "/missing/source.jpg", "0" * 64, {"hooks": {"session_start": [{}]}}),
        ("owner scenario", "/missing/source.jpg", "0" * 64, {"enabled_plugins": {"sample": True}}),
    ],
)
def test_native_preflight_rejects_invalid_inputs_before_resolver(
    tmp_path, scenario, source_path, source_sha, settings_update
):
    settings = _native_settings().model_copy(update=settings_update)
    calls = []

    with pytest.raises(NativeClientPreconditionError):
        native_preflight_and_clients(
            settings,
            scenario=scenario,
            source_path=source_path,
            source_sha256=source_sha,
            root=tmp_path,
            resolver=lambda _settings: calls.append("resolved"),
            codex_client_type=FakeCodexClient,
        )

    assert calls == []


def test_native_runtime_guard_blocks_discovered_loader_surfaces_and_restores_them():
    class BuilderModule:
        def load_plugins(self, *_args, **_kwargs):
            return ["discovered plugin"]

        def load_mcp_server_configs(self, *_args, **_kwargs):
            return {"discovered": "server"}

        def load_hook_registry(self, *_args, **_kwargs):
            return "discovered hook"

    runtime = BuilderModule()
    originals = (runtime.load_plugins, runtime.load_mcp_server_configs, runtime.load_hook_registry)
    with isolated_runtime_loaders(runtime):
        assert runtime.load_plugins({}, "/project") == []
        assert runtime.load_mcp_server_configs({}, []) == {}
        assert runtime.load_hook_registry({}, []).summary() == ""
    assert (
        runtime.load_plugins,
        runtime.load_mcp_server_configs,
        runtime.load_hook_registry,
    ) == originals


def test_native_guard_covers_real_prompt_catalog_route_and_restores_after_failure(
    tmp_path, monkeypatch
):
    import openharness.ui.runtime as runtime_module
    from openharness.commands import registry as command_registry
    from openharness.plugins import loader as plugin_loader
    from openharness.prompts import context as prompt_context
    from openharness.skills import loader as skill_loader
    from openharness.skills.bundled import get_bundled_skills

    original_plugin_loader = plugin_loader.load_plugins
    plugin_loader_calls = []

    def observed_original_plugin_loader(*args, **kwargs):
        plugin_loader_calls.append((args, kwargs))
        return original_plugin_loader(*args, **kwargs)

    ambient_catalog_calls = []

    def reject_ambient_catalog(*_args, **_kwargs):
        ambient_catalog_calls.append(True)
        raise AssertionError("native prompt attempted ambient catalog discovery")

    monkeypatch.setattr(plugin_loader, "load_plugins", observed_original_plugin_loader)
    monkeypatch.setattr(plugin_loader, "get_user_plugins_dir", reject_ambient_catalog)
    monkeypatch.setattr(skill_loader, "get_user_skills_dir", reject_ambient_catalog)
    monkeypatch.setattr(prompt_context, "load_local_rules", lambda: "")

    symbols = (
        (runtime_module, "load_plugins"),
        (runtime_module, "load_mcp_server_configs"),
        (runtime_module, "load_hook_registry"),
        (plugin_loader, "load_plugins"),
        (command_registry, "load_plugins"),
        (skill_loader, "load_user_skills"),
    )
    originals = tuple((module, name, getattr(module, name)) for module, name in symbols)
    settings = _native_settings().model_copy(update={"system_prompt": "synthetic prompt"})
    bundled = get_bundled_skills()
    assert bundled
    bundled_name = bundled[0].command_name or bundled[0].name

    with isolated_runtime_loaders(runtime_module):
        prompt = prompt_context.build_runtime_system_prompt(
            settings,
            cwd=tmp_path,
            include_project_memory=False,
        )

    assert "# Available Skills" in prompt
    assert f"**{bundled_name}**" in prompt
    assert plugin_loader_calls == []
    assert ambient_catalog_calls == []
    assert all(getattr(module, name) is original for module, name, original in originals)

    with pytest.raises(RuntimeError, match="forced prompt failure"):
        with isolated_runtime_loaders(runtime_module):
            raise RuntimeError("forced prompt failure")
    assert all(getattr(module, name) is original for module, name, original in originals)


@pytest.mark.asyncio
async def test_user_role_tool_result_finishes_owner_turn_without_reasking_for_photo():
    client = OfflineCameraBotApi()
    messages = [ConversationMessage.from_user_text("Да, я это съел(а)")]

    proposal = [event async for event in client.stream_message(SimpleNamespace(messages=messages))]
    trace_call = next(
        block for block in proposal[0].message.content if isinstance(block, ToolUseBlock)
    )
    messages.extend(
        [
            ConversationMessage(role="assistant", content=[trace_call]),
            ConversationMessage(
                role="user",
                content=[ToolResultBlock(tool_use_id=trace_call.id, content="meal recorded")],
            ),
        ]
    )

    recorded = [event async for event in client.stream_message(SimpleNamespace(messages=messages))]

    response = recorded[0].message.text
    assert "recorded the synthetic Camera meal" in response
    assert "[[ask:" not in response
    assert client.finalization_proposals == 1


@pytest.mark.asyncio
async def test_old_user_role_tool_result_does_not_suppress_fresh_owner_replay_trace():
    client = OfflineCameraBotApi()
    request = SimpleNamespace(
        messages=[
            ConversationMessage.from_user_text("Да, я это съел(а)"),
            ConversationMessage(
                role="assistant",
                content=[ToolUseBlock(name="trace", input={})],
            ),
            ConversationMessage(
                role="user",
                content=[ToolResultBlock(tool_use_id="earlier", content="meal recorded")],
            ),
            ConversationMessage.from_user_text("Да, я это съел(а)"),
        ]
    )

    events = [event async for event in client.stream_message(request)]

    assert any(isinstance(block, ToolUseBlock) for block in events[0].message.content)
    assert client.finalization_proposals == 1

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from camera_runtime_support import (
    OfflineCameraBotApi,
    assert_e5_raw_honcho_row_unchanged,
    e5_raw_honcho_row_fingerprint,
    isolated_runtime_loaders,
    validate_e5_date_source_link,
    validate_e5_denial_receipt,
    validate_e5_post_correction_replay,
    validate_e5_unique_original_event_ids,
)
from openharness.config.settings import ProviderProfile, Settings
from openharness.engine.messages import ConversationMessage, ToolResultBlock, ToolUseBlock
from probe_support import (
    NativeClientPreconditionError,
    native_preflight_and_clients,
    native_profile_clients,
)


class FakeCodexClient:
    pass


def _e5_original_metadata():
    return {
        "tenant_id": "tenant-owner-123",
        "source_principal": "telegram:123",
        "gateway_session_id": "session-owner-123",
        "source_message_id": "77",
        "is_group": False,
        "is_forwarded": False,
    }


def test_e5_date_event_accepts_actual_reply_to_source_binding_without_camera_extras():
    original = _e5_original_metadata()
    date_metadata = {
        **original,
        "source_message_id": "date-message-91",
        "reply_to_source_message_id": "77",
    }

    validate_e5_date_source_link(
        date_metadata=date_metadata,
        original_metadata=original,
        expected_date_source_id="date-message-91",
    )


def test_e5_requires_one_exact_original_event_for_the_reply_source():
    validate_e5_unique_original_event_ids(
        observed_event_ids=["original-meal-1"],
        expected_original_event_id="original-meal-1",
    )


@pytest.mark.parametrize("observed_event_ids", [[], ["original-meal-1", "duplicate-meal"]])
def test_e5_date_source_does_not_pass_when_original_event_is_missing_or_ambiguous(
    observed_event_ids,
):
    with pytest.raises(AssertionError, match="one original runtime observation"):
        validate_e5_unique_original_event_ids(
            observed_event_ids=observed_event_ids,
            expected_original_event_id="original-meal-1",
        )


@pytest.mark.asyncio
async def test_e5_immutability_uses_the_full_raw_honcho_reader_shape():
    from ohmo.memory_service.honcho_client import HonchoClient

    created_at = "2026-10-06T12:00:00+00:00"
    persisted_row = {
        "id": "original-meal-1",
        "peer_id": "ohmo",
        "session_id": "session-owner-123",
        "workspace_id": "workspace-1",
        "created_at": created_at,
        "content": "Persisted assistant meal text",
        "metadata": {
            "gateway_session_id": "session-owner-123",
            "source_message_id": "77",
            "decision_trace": {"annotations": {"nutrition": {
                "record_type": "meal_observation",
                "consumption_status": "consumed",
                "energy_kcal_best": 125,
            }}},
        },
    }

    def handler(request):
        assert request.url.path.endswith("sessions/session-owner-123/messages/list")
        page = int(request.url.params["page"])
        size = int(request.url.params["size"])
        return httpx.Response(200, json={
            "items": [persisted_row], "page": page, "size": size, "pages": 1, "total": 1,
        })

    since = datetime(2026, 10, 6, 11, tzinfo=timezone.utc)
    until = datetime(2026, 10, 6, 13, tzinfo=timezone.utc)
    client = HonchoClient(
        "https://honcho.fixture", "synthetic-token", "workspace-1",
        transport=httpx.MockTransport(handler),
    )
    try:
        metadata_rows = await client.list_recent_message_metadata(
            "session-owner-123", expected_peer_id="ohmo", since=since, until=until,
        )
        assert len(metadata_rows) == 1
        assert not hasattr(metadata_rows[0], "content")

        raw_rows = await client.list_messages_in_window(
            "session-owner-123", expected_peer_id="ohmo", since=since, until=until,
        )
    finally:
        await client.aclose()

    assert len(raw_rows) == 1
    assert raw_rows[0] == persisted_row
    assert raw_rows[0]["content"] == "Persisted assistant meal text"
    assert raw_rows[0]["metadata"] == persisted_row["metadata"]
    assert raw_rows[0]["created_at"] == created_at
    assert e5_raw_honcho_row_fingerprint(raw_rows[0]) == e5_raw_honcho_row_fingerprint(
        dict(raw_rows[0])
    )
    for changed_field, changed_value in (
        ("content", "changed persisted content"),
        ("metadata", {"source_message_id": "different"}),
        ("created_at", "2026-10-06T12:00:01+00:00"),
    ):
        changed = dict(raw_rows[0])
        changed[changed_field] = changed_value
        with pytest.raises(AssertionError, match="immutable original raw Honcho row"):
            assert_e5_raw_honcho_row_unchanged(raw_rows[0], changed)


@pytest.mark.parametrize(
    ("override", "missing"),
    [
        ({"source_principal": "telegram:foreign"}, None),
        ({"gateway_session_id": "foreign-session"}, None),
        ({"tenant_id": "foreign-tenant"}, None),
        ({"reply_to_source_message_id": "999"}, None),
        ({"source_message_id": "foreign-date-message"}, None),
        ({"is_group": True}, None),
        ({"is_forwarded": True}, None),
        ({}, "reply_to_source_message_id"),
        ({}, "source_message_id"),
    ],
)
def test_e5_date_event_rejects_untrusted_or_missing_reply_source(override, missing):
    original = _e5_original_metadata()
    date_metadata = {
        **original,
        "source_message_id": "date-message-91",
        "reply_to_source_message_id": "77",
    }
    date_metadata.update(override)
    if missing:
        date_metadata.pop(missing)

    with pytest.raises(AssertionError, match="trusted owner/source/session"):
        validate_e5_date_source_link(
            date_metadata=date_metadata,
            original_metadata=original,
            expected_date_source_id="date-message-91",
        )


def _e5_denial_fixture():
    original = _e5_original_metadata()
    commit = {
        "kind": "denial",
        "event_id": "assistant-denial-8",
        "source_message_id": "owner-denial-10",
        "client_op_id": "turn-denial:assistant",
        "target_event_id": "original-meal-1",
        "target_source_message_id": "77",
    }
    metadata = {
        **original,
        "source_message_id": "owner-denial-10",
        "reply_to_source_message_id": "77",
        "client_op_id": "turn-denial:assistant",
        "camera_candidate_id": "candidate-123",
        "camera_operation_id": "candidate-123",
        "camera_original_event_id": "original-meal-1",
        "camera_answer_bound": "no",
        "camera_correction_bound": True,
        "decision_trace": {"annotations": {"nutrition": {
            "record_type": "meal_correction",
            "changed_fields": ["consumption_status", "energy_kcal_best"],
            "consumption_status": "not_consumed",
            "energy_kcal_best": 0,
        }}},
    }
    row = {
        "id": "assistant-denial-8",
        "peer_id": "ohmo",
        "session_id": "session-owner-123",
        "workspace_id": "workspace-1",
        "created_at": "2026-10-06T12:01:00+00:00",
        "content": "Persisted denial correction",
        "metadata": metadata,
    }
    return original, commit, row


@pytest.mark.parametrize("outbound_event_id", [None, "assistant-denial-8"])
def test_e5_denial_receipt_uses_actual_durable_event_when_public_id_is_absent(
    outbound_event_id,
):
    original, commit, row = _e5_denial_fixture()

    event_id = validate_e5_denial_receipt(
        correction_commit=commit,
        outbound_event_id=outbound_event_id,
        candidate_id="candidate-123",
        original_event_id="original-meal-1",
        original_metadata=original,
        honcho_row=row,
    )

    assert event_id == "assistant-denial-8"


@pytest.mark.parametrize(
    ("commit_override", "metadata_override", "outbound_event_id"),
    [
        ({"kind": "portion"}, {}, None),
        ({"target_event_id": "foreign-event"}, {}, None),
        ({"target_source_message_id": "999"}, {}, None),
        ({}, {"source_principal": "telegram:foreign"}, None),
        ({}, {"gateway_session_id": "foreign-session"}, None),
        ({}, {"source_message_id": "foreign-source"}, None),
        ({}, {"reply_to_source_message_id": "999"}, None),
        ({}, {"decision_trace": {"annotations": {"nutrition": {
            "record_type": "meal_correction", "changed_fields": ["items"],
            "consumption_status": "not_consumed", "energy_kcal_best": 0,
        }}}}, None),
        ({}, {}, "different-outbound-event"),
        ({}, {}, ""),
    ],
)
def test_e5_denial_receipt_rejects_mismatched_commit_or_honcho_row(
    commit_override, metadata_override, outbound_event_id,
):
    original, commit, row = _e5_denial_fixture()
    commit.update(commit_override)
    row["metadata"].update(metadata_override)

    with pytest.raises(AssertionError, match="durable Honcho correction"):
        validate_e5_denial_receipt(
            correction_commit=commit,
            outbound_event_id=outbound_event_id,
            candidate_id="candidate-123",
            original_event_id="original-meal-1",
            original_metadata=original,
            honcho_row=row,
        )


def test_e5_replay_oracle_tracks_denial_without_resurrecting_original_meal():
    original_commit = {"event_id": "original-meal-1", "client_op_id": "original:assistant"}

    status = validate_e5_post_correction_replay(
        status="Исправление уже записано.",
        delivery_receipt=object(),
        event_id="assistant-denial-8",
        expected_event_id="assistant-denial-8",
        original_commit=original_commit,
        current_commit=original_commit.copy(),
    )

    assert status == "Исправление уже записано."


def test_e5_replay_oracle_accepts_context_replay_when_receipt_and_history_are_verified():
    status = validate_e5_post_correction_replay(
        status="Изменение сохранено; баланс обновляется.",
        delivery_receipt=object(),
        event_id="assistant-context-date-8",
        expected_event_id="assistant-context-date-8",
        original_commit={"event_id": "original-meal-1"},
        current_commit={"event_id": "original-meal-1"},
        required_phrase=None,
    )
    assert status == "Изменение сохранено; баланс обновляется."


@pytest.mark.parametrize(
    ("status", "event_id", "delivery_receipt", "current_commit"),
    [
        ("Запись уже записана.", "assistant-denial-8", object(), {"event_id": "original-meal-1"}),
        ("Исправление уже записано.", "original-meal-1", object(), {"event_id": "original-meal-1"}),
        ("Исправление уже записано.", "assistant-denial-8", None, {"event_id": "original-meal-1"}),
        ("Исправление уже записано.", "assistant-denial-8", object(), {"event_id": "resurrected"}),
    ],
)
def test_e5_replay_oracle_rejects_stale_or_unverified_final_status(
    status, event_id, delivery_receipt, current_commit,
):
    with pytest.raises(AssertionError, match="post-denial replay"):
        validate_e5_post_correction_replay(
            status=status,
            delivery_receipt=delivery_receipt,
            event_id=event_id,
            expected_event_id="assistant-denial-8",
            original_commit={"event_id": "original-meal-1"},
            current_commit=current_commit,
        )


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


def test_native_profile_rejects_non_medium_effort_before_resolution():
    settings = _native_settings().model_copy(update={"effort": "high"})
    with pytest.raises(NativeClientPreconditionError, match="medium reasoning effort"):
        native_profile_clients(
            settings,
            resolver=lambda _settings: pytest.fail("resolver should not run"),
            codex_client_type=FakeCodexClient,
        )


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
async def test_owner_text_without_selected_image_cannot_finalize_meal():
    client = OfflineCameraBotApi()
    messages = [ConversationMessage.from_user_text("Да, я это съел(а)")]

    proposal = [event async for event in client.stream_message(SimpleNamespace(messages=messages))]
    assert not any(isinstance(block, ToolUseBlock) for block in proposal[0].message.content)
    assert "No owned image source" in proposal[0].message.text
    assert client.finalization_proposals == 0


@pytest.mark.asyncio
async def test_old_trace_result_cannot_supply_missing_owned_image():
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

    assert not any(isinstance(block, ToolUseBlock) for block in events[0].message.content)
    assert client.finalization_proposals == 0

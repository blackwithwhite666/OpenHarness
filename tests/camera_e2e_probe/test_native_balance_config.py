"""The follow-up resolves harmless fixture auth from its selected config root."""

from __future__ import annotations

import json
import os

from openharness.api.codex_client import CodexApiClient
from openharness.auth.external import CODEX_PROVIDER
from openharness.auth.storage import ExternalAuthBinding, store_external_binding
from openharness.config.settings import ProviderProfile, Settings
from native_balance_config import resolve_native_report_clients


def test_native_resolver_reads_selected_config_before_creating_output(monkeypatch, tmp_path):
    selected = tmp_path / "selected"
    ambient = tmp_path / "ambient-without-binding"
    selected.mkdir()
    ambient.mkdir()
    source = tmp_path / "fixture-codex-auth.json"
    source.write_text(json.dumps({"tokens": {"access_token": "fixture-only-token"}}))
    settings = Settings(
        active_profile="codex", effort="medium",
        profiles={"codex": ProviderProfile(
            label="Fixture Codex subscription", provider="openai_codex",
            api_format="openai", auth_source="codex_subscription",
            default_model="gpt-6-luna",
        )},
        allow_project_skills=False, project_skill_dirs=[],
    )
    (selected / "settings.json").write_text(settings.model_dump_json())
    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(selected))
    store_external_binding(ExternalAuthBinding(
        provider=CODEX_PROVIDER, source_path=str(source),
        source_kind="codex_auth_json", managed_by="fixture",
    ))
    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(ambient))
    monkeypatch.setenv("OPENHARNESS_PROFILE", "fixture-ambient")

    bot, virtual_user = resolve_native_report_clients(selected)

    assert isinstance(bot, CodexApiClient)
    assert isinstance(virtual_user, CodexApiClient)
    assert bot is not virtual_user
    assert os.environ["OPENHARNESS_CONFIG_DIR"] == str(selected)
    assert list(ambient.iterdir()) == []

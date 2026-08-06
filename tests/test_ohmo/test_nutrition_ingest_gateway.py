from __future__ import annotations

import pytest

from ohmo.gateway.models import GatewayConfig, NutritionIngestConfig


def _enabled_config(tmp_path, **nutrition_overrides):
    nutrition = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
        **nutrition_overrides,
    )
    return GatewayConfig(
        honcho_base_url="https://honcho.test",
        conversation_learning=True,
        family_principals={"123": "marina"},
        enabled_memory_tenants=("marina",),
        tenant_honcho={
            "marina": {
                "workspace": "family-marina",
                "api_key": "runtime-key",
                "session": "marina-session",
                "observed_peer": "marina-peer",
            }
        },
        nutrition_ingest=nutrition,
    )


def test_disabled_nutrition_ingest_is_safe_by_default() -> None:
    config = GatewayConfig()
    assert config.nutrition_ingest.enabled is False


def test_enabled_nutrition_ingest_fails_closed_for_non_numeric_principal(tmp_path) -> None:
    with pytest.raises(ValueError, match="numeric"):
        NutritionIngestConfig(
            enabled=True,
            synchronized_root=tmp_path,
            principal="@marina",
            chat_id="chat",
            session_key="telegram:chat",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"principal": "0", "chat_id": "0", "session_key": "telegram:0"},
        {"principal": "123", "chat_id": "-123", "session_key": "telegram:-123"},
        {"principal": "123", "chat_id": "124", "session_key": "telegram:124"},
        {"principal": "123", "chat_id": "123", "session_key": "telegram:124"},
    ],
)
def test_enabled_nutrition_ingest_requires_one_positive_private_binding(tmp_path, kwargs) -> None:
    with pytest.raises(ValueError):
        NutritionIngestConfig(
            enabled=True,
            synchronized_root=tmp_path,
            **kwargs,
        )


def test_enabled_nutrition_ingest_requires_complete_marina_gateway_binding(tmp_path) -> None:
    with pytest.raises(ValueError, match="marina"):
        GatewayConfig(
            conversation_learning=True,
            family_principals={"123": "dmitry"},
            enabled_memory_tenants=("dmitry",),
            tenant_honcho={
                "dmitry": {
                    "workspace": "family-dmitry",
                    "api_key": "runtime-key",
                    "observed_peer": "dmitry-peer",
                }
            },
            nutrition_ingest=NutritionIngestConfig(
                enabled=True,
                synchronized_root=tmp_path,
                principal="123",
                chat_id="123",
                session_key="telegram:123",
            ),
        )


def test_enabled_nutrition_ingest_rejects_owner_overlap(tmp_path) -> None:
    with pytest.raises(ValueError, match="both owner"):
        _enabled_config_with_owner(tmp_path)


def _enabled_config_with_owner(tmp_path):
    return GatewayConfig(
        honcho_base_url="https://honcho.test",
        conversation_learning=True,
        owner_principals=("123",),
        family_principals={"123": "marina"},
        enabled_memory_tenants=("marina",),
        tenant_honcho={
            "marina": {
                "workspace": "family-marina",
                "api_key": "runtime-key",
                "observed_peer": "marina-peer",
            }
        },
        nutrition_ingest=NutritionIngestConfig(
            enabled=True,
            synchronized_root=tmp_path,
            principal="123",
            chat_id="123",
            session_key="telegram:123",
        ),
    )


def test_enabled_nutrition_ingest_accepts_only_the_marina_binding(tmp_path) -> None:
    config = _enabled_config(tmp_path)
    assert config.nutrition_ingest.session_key == "telegram:123"
    assert config.tenant_honcho["marina"]["session"] == "marina-session"


@pytest.mark.parametrize("session", ["", "with spaces", "slash/name", "x" * 513])
def test_tenant_honcho_session_must_be_a_bounded_identifier(session: str) -> None:
    with pytest.raises(ValueError, match="bounded Honcho identifier"):
        GatewayConfig(
            tenant_honcho={
                "marina": {
                    "workspace": "family-marina",
                    "api_key": "runtime-key",
                    "observed_peer": "marina-peer",
                    "session": session,
                }
            }
        )


def test_tenant_honcho_session_defaults_to_ohmo(tmp_path) -> None:
    config = _enabled_config(tmp_path)
    del config.tenant_honcho["marina"]["session"]
    from ohmo.memory_backend import resolve_tenant_honcho_binding

    binding = resolve_tenant_honcho_binding(config, "marina")
    assert binding is not None
    assert binding.session == "ohmo"


def test_enabled_nutrition_ingest_requires_honcho_base_url(tmp_path) -> None:
    with pytest.raises(ValueError, match="Honcho base URL"):
        GatewayConfig(
            conversation_learning=True,
            family_principals={"123": "marina"},
            enabled_memory_tenants=("marina",),
            tenant_honcho={
                "marina": {
                    "workspace": "family-marina",
                    "api_key": "runtime-key",
                    "observed_peer": "marina-peer",
                }
            },
            nutrition_ingest=NutritionIngestConfig(
                enabled=True,
                synchronized_root=tmp_path,
                principal="123",
                chat_id="123",
                session_key="telegram:123",
            ),
        )


def test_nutrition_status_and_replay_cli_output_is_privacy_safe(monkeypatch, capsys) -> None:
    import ohmo.cli as cli

    class StubCoordinator:
        def __init__(self, _config):
            pass

        def status(self):
            return {
                "enabled": True,
                "pending": 1,
                "items": [{"state": "delivery_unknown", "candidate": "secret-candidate"}],
            }

        def request_replay(self, candidate):
            assert candidate == "operator-selected"
            return True

    monkeypatch.setattr(cli, "load_gateway_config", lambda _workspace: GatewayConfig())
    monkeypatch.setattr(cli, "NutritionIngestCoordinator", StubCoordinator)
    cli.nutrition_status_cmd(None)
    status = capsys.readouterr().out
    assert "secret-candidate" not in status
    assert "delivery_unknown" in status

    cli.nutrition_replay_cmd("operator-selected", None)
    replay = capsys.readouterr().out
    assert replay.strip() == '{"replayed": true}'

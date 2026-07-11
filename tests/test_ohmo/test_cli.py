import json
import logging
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from openharness.api.client import ApiMessageCompleteEvent
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, TextBlock

from ohmo.cli import _build_gateway_logging_handlers, app
from ohmo.memory_judge import load_removal_proposals, save_removal_proposals
from ohmo.memory_store import MemoryStore


class _FakeSettings:
    model = "fake-model"
    active_profile = "fake-profile"

    def merge_cli_overrides(self, *, model=None, active_profile=None):
        settings = _FakeSettings()
        settings.model = model or self.model
        settings.active_profile = active_profile or self.active_profile
        return settings

    def materialize_active_profile(self):
        return self


class _FakeCompletionClient:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.requests = []

    async def stream_message(self, request):
        self.requests.append(request)
        text = self.responses.pop(0) if self.responses else '{"ops":[],"reason":"done"}'
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text=text)]),
            usage=UsageSnapshot(),
        )


def _patch_memory_consolidation_client(monkeypatch, client: _FakeCompletionClient) -> None:
    monkeypatch.setattr("ohmo.cli.load_settings", lambda: _FakeSettings())
    monkeypatch.setattr("ohmo.cli.resolve_api_client_from_settings", lambda settings: client)


def test_ohmo_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "personal-agent app" in result.output
    assert "config" in result.output
    assert "evals" in result.output


def test_ohmo_init_and_doctor(tmp_path: Path):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    result = runner.invoke(app, ["init", "--workspace", str(workspace), "--no-interactive"])
    assert result.exit_code == 0
    assert str(workspace) in result.output

    doctor = runner.invoke(app, ["doctor", "--cwd", str(tmp_path), "--workspace", str(workspace)])
    assert doctor.exit_code == 0
    assert "ohmo doctor:" in doctor.output
    assert "workspace: ok" in doctor.output


def test_ohmo_init_existing_workspace_points_to_config(tmp_path: Path):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    first = runner.invoke(app, ["init", "--workspace", str(workspace), "--no-interactive"])
    assert first.exit_code == 0

    second = runner.invoke(app, ["init", "--workspace", str(workspace), "--no-interactive"])
    assert second.exit_code == 0
    assert "ohmo workspace already exists." in second.output
    assert "Use `ohmo config`" in second.output


def test_ohmo_init_noninteractive_defaults_to_deny_all_remote_access(tmp_path: Path):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    result = runner.invoke(app, ["init", "--workspace", str(workspace), "--no-interactive"])
    assert result.exit_code == 0
    config = json.loads((workspace / "gateway.json").read_text(encoding="utf-8"))
    assert config["channel_configs"] == {}


def test_gateway_logging_handlers_write_gateway_log_file(tmp_path: Path):
    workspace = tmp_path / ".ohmo-home"
    runner = CliRunner()
    result = runner.invoke(app, ["init", "--workspace", str(workspace), "--no-interactive"])
    assert result.exit_code == 0

    handlers = _build_gateway_logging_handlers(workspace, console=True, log_file=True)
    try:
        file_handlers = [handler for handler in handlers if isinstance(handler, logging.FileHandler)]
        console_handlers = [
            handler
            for handler in handlers
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert len(console_handlers) == 1

        record = logging.LogRecord(
            name="ohmo.gateway.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="GATEWAY_LOG_OK",
            args=(),
            exc_info=None,
        )
        file_handlers[0].emit(record)
        file_handlers[0].flush()

        log_path = workspace / "logs" / "gateway.log"
        assert "GATEWAY_LOG_OK" in log_path.read_text(encoding="utf-8")
    finally:
        for handler in handlers:
            handler.close()


def test_ohmo_init_interactive_writes_gateway_config(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    user_input = "\n".join(
        [
            "1",  # provider profile
            "y",  # enable telegram
            "123456",  # allow_from
            "telegram-token",
            "y",  # reply_to_message
            "n",  # slack
            "n",  # discord
            "n",  # feishu
            "y",  # send_progress
            "y",  # send_tool_hints
            "n",  # allow_remote_admin_commands
        ]
    )
    result = runner.invoke(app, ["init", "--workspace", str(workspace)], input=user_input)
    assert result.exit_code == 0
    config = json.loads((workspace / "gateway.json").read_text(encoding="utf-8"))
    assert config["enabled_channels"] == ["telegram"]
    assert config["channel_configs"]["telegram"]["token"] == "telegram-token"
    assert config["channel_configs"]["telegram"]["allow_from"] == ["123456"]


def test_ohmo_init_interactive_allows_blank_allow_from_for_secure_default(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    user_input = "\n".join(
        [
            "1",  # provider profile
            "y",  # enable telegram
            "",   # allow_from -> deny all until explicitly configured
            "telegram-token",
            "y",  # reply_to_message
            "n",  # slack
            "n",  # discord
            "n",  # feishu
            "y",  # send_progress
            "y",  # send_tool_hints
            "n",  # allow_remote_admin_commands
        ]
    )
    result = runner.invoke(app, ["init", "--workspace", str(workspace)], input=user_input)
    assert result.exit_code == 0
    config = json.loads((workspace / "gateway.json").read_text(encoding="utf-8"))
    assert config["channel_configs"]["telegram"]["allow_from"] == []
    assert "Remote access denied until allow_from is configured for: telegram" in result.output


def test_ohmo_init_interactive_writes_feishu_gateway_config(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    user_input = "\n".join(
        [
            "1",         # provider profile
            "n",         # telegram
            "n",         # slack
            "n",         # discord
            "y",         # feishu
            "feishu-user-1",         # allow_from
            "1",         # domain -> Feishu (China)
            "cli_app",   # app_id
            "cli_secret",# app_secret
            "enc_key",   # encrypt_key
            "verify_me", # verification_token
            "OK",        # react_emoji
            "1",         # group_policy -> managed_or_mention
            "ohmo,openclaw", # bot_names
            "",          # bot_open_id
            "y",         # send_progress
            "n",         # send_tool_hints
            "n",         # allow_remote_admin_commands
        ]
    )
    result = runner.invoke(app, ["init", "--workspace", str(workspace)], input=user_input)
    assert result.exit_code == 0
    config = json.loads((workspace / "gateway.json").read_text(encoding="utf-8"))
    assert config["enabled_channels"] == ["feishu"]
    assert config["channel_configs"]["feishu"]["domain"] == "https://open.feishu.cn"
    assert config["channel_configs"]["feishu"]["app_id"] == "cli_app"
    assert config["channel_configs"]["feishu"]["app_secret"] == "cli_secret"
    assert config["channel_configs"]["feishu"]["encrypt_key"] == "enc_key"
    assert config["channel_configs"]["feishu"]["verification_token"] == "verify_me"
    assert config["channel_configs"]["feishu"]["react_emoji"] == "OK"
    assert config["channel_configs"]["feishu"]["group_policy"] == "managed_or_mention"
    assert config["channel_configs"]["feishu"]["bot_names"] == ["ohmo", "openclaw"]


def test_ohmo_config_interactive_can_restart_gateway(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    runner.invoke(app, ["init", "--workspace", str(workspace), "--no-interactive"])
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("ohmo.cli.gateway_status", lambda cwd, workspace: type("State", (), {"running": True})())
    monkeypatch.setattr("ohmo.cli.stop_gateway_process", lambda cwd, workspace: True)
    monkeypatch.setattr("ohmo.cli.start_gateway_process", lambda cwd, workspace: 4321)
    user_input = "\n".join(
        [
            "4",          # provider profile -> codex
            "n",          # telegram
            "n",          # slack
            "n",          # discord
            "y",          # feishu
            "feishu-user-1",          # allow_from
            "2",          # domain -> Lark (International)
            "cli_app",    # app_id
            "cli_secret", # app_secret
            "",           # encrypt_key
            "verify_me",  # verification_token
            "OK",         # react_emoji
            "1",          # group_policy -> managed_or_mention
            "ohmo,openclaw", # bot_names
            "",           # bot_open_id
            "y",          # send_progress
            "y",          # send_tool_hints
            "n",          # allow_remote_admin_commands
            "y",          # restart gateway
        ]
    )
    result = runner.invoke(app, ["config", "--workspace", str(workspace)], input=user_input)
    assert result.exit_code == 0
    assert "ohmo gateway restarted (pid=4321)" in result.output
    config = json.loads((workspace / "gateway.json").read_text(encoding="utf-8"))
    assert config["provider_profile"] == "codex"
    assert config["enabled_channels"] == ["feishu"]
    assert config["channel_configs"]["feishu"]["domain"] == "https://open.larksuite.com"


def test_ohmo_config_keeps_existing_channel_when_not_reconfigured(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    runner.invoke(app, ["init", "--workspace", str(workspace), "--no-interactive"])
    gateway_path = workspace / "gateway.json"
    config = json.loads(gateway_path.read_text(encoding="utf-8"))
    config["enabled_channels"] = ["feishu"]
    config["channel_configs"]["feishu"] = {
        "allow_from": ["feishu-user-1"],
        "app_id": "old_app",
        "app_secret": "old_secret",
        "encrypt_key": "",
        "verification_token": "old_verify",
        "react_emoji": "OK",
    }
    gateway_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("ohmo.cli.gateway_status", lambda cwd, workspace: type("State", (), {"running": False})())
    user_input = "\n".join(
        [
            "4",  # provider profile -> codex
            "n",  # telegram
            "n",  # slack
            "n",  # discord
            "n",  # reconfigure feishu? keep existing
            "y",  # send_progress
            "y",  # send_tool_hints
            "n",  # allow_remote_admin_commands
        ]
    )
    result = runner.invoke(app, ["config", "--workspace", str(workspace)], input=user_input)
    assert result.exit_code == 0
    updated = json.loads(gateway_path.read_text(encoding="utf-8"))
    assert updated["enabled_channels"] == ["feishu"]
    assert updated["channel_configs"]["feishu"]["app_id"] == "old_app"
    assert updated["channel_configs"]["feishu"]["app_secret"] == "old_secret"


def test_ohmo_memory_list_outputs_sizes_and_budget(tmp_path: Path):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    store = MemoryStore(workspace)
    store.add("Timezone", "UTC")

    result = runner.invoke(app, ["memory", "list", "--workspace", str(workspace)])

    assert result.exit_code == 0
    assert "name | title | size" in result.output
    assert "timezone.md | Timezone | 3" in result.output
    assert "total: 3/24000" in result.output


def test_ohmo_memory_proposals_and_prune_apply(tmp_path: Path):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    store = MemoryStore(workspace)
    store.add("Timezone", "UTC")
    store.add("Style", "Concise")
    save_removal_proposals(
        store,
        [
            {"name": "timezone.md", "reason": "duplicated"},
            {"name": "missing.md", "reason": "stale"},
        ],
    )

    proposals = runner.invoke(app, ["memory", "proposals", "--workspace", str(workspace)])
    assert proposals.exit_code == 0
    assert "timezone.md | duplicated | 3" in proposals.output
    assert "missing.md" not in proposals.output

    result = runner.invoke(
        app,
        ["memory", "prune", "--workspace", str(workspace), "--apply", "timezone.md"],
    )

    assert result.exit_code == 0
    assert "Removed timezone.md (3 chars)" in result.output
    assert "Freed 3 chars." in result.output
    assert store.get("timezone.md") is None
    assert store.get("style.md") is not None
    assert load_removal_proposals(store) == []


def test_ohmo_memory_prune_all_proposed(tmp_path: Path):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    store = MemoryStore(workspace)
    store.add("A", "aaa")
    store.add("B", "bbb")
    save_removal_proposals(
        store,
        [
            {"name": "a.md", "reason": "stale"},
            {"name": "b.md", "reason": "stale"},
        ],
    )

    result = runner.invoke(app, ["memory", "prune", "--workspace", str(workspace), "--all-proposed"])

    assert result.exit_code == 0
    assert "Removed a.md (3 chars)" in result.output
    assert "Removed b.md (3 chars)" in result.output
    assert store.list() == []
    assert load_removal_proposals(store) == []


def test_ohmo_memory_prune_dismiss_keeps_entry(tmp_path: Path):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    store = MemoryStore(workspace)
    store.add("Timezone", "UTC")
    save_removal_proposals(store, [{"name": "timezone.md", "reason": "duplicated"}])

    result = runner.invoke(
        app,
        ["memory", "prune", "--workspace", str(workspace), "--dismiss", "timezone.md"],
    )

    assert result.exit_code == 0
    assert "Dismissed proposals: timezone.md" in result.output
    assert store.get("timezone.md") is not None
    assert load_removal_proposals(store) == []


def test_ohmo_memory_consolidate_dry_run_prints_without_changing_store(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    store = MemoryStore(workspace)
    store.add("Work prefs", "User uses UTC. User prefers concise replies.")
    store.add("Reply prefs", "User uses UTC. User likes tables.")
    before = store.total_chars()
    client = _FakeCompletionClient(
        json.dumps(
            {
                "ops": [
                    {
                        "action": "consolidate",
                        "names": ["work_prefs.md", "reply_prefs.md"],
                        "into": "work_prefs.md",
                        "title": "Preferences",
                        "content": "User uses UTC, prefers concise replies, and likes tables.",
                    }
                ],
                "reason": "overlap",
            }
        )
    )
    _patch_memory_consolidation_client(monkeypatch, client)

    result = runner.invoke(
        app,
        ["memory", "consolidate", "--workspace", str(workspace), "--dry-run"],
    )

    assert result.exit_code == 0
    assert "Proposed consolidations:" in result.output
    assert "work_prefs.md, reply_prefs.md -> work_prefs.md" in result.output
    assert "projected freed" in result.output
    assert store.total_chars() == before
    assert store.get("reply_prefs.md") is not None


def test_ohmo_memory_consolidate_apply_shrinks_store(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    store = MemoryStore(workspace)
    store.add("Work prefs", "User uses UTC. User prefers concise replies.")
    store.add("Reply prefs", "User uses UTC. User likes tables.")
    before = store.total_chars()
    client = _FakeCompletionClient(
        json.dumps(
            {
                "ops": [
                    {
                        "action": "consolidate",
                        "names": ["work_prefs.md", "reply_prefs.md"],
                        "into": "work_prefs.md",
                        "title": "Preferences",
                        "content": "User uses UTC, prefers concise replies, and likes tables.",
                    }
                ],
                "reason": "overlap",
            }
        ),
        '{"ops":[],"reason":"done"}',
    )
    _patch_memory_consolidation_client(monkeypatch, client)

    result = runner.invoke(app, ["memory", "consolidate", "--workspace", str(workspace)])

    assert result.exit_code == 0
    assert "Memory before:" in result.output
    assert "Applied: consolidate work_prefs.md: merged 2 → 1" in result.output
    assert "freed " in result.output
    assert store.total_chars() < before
    assert store.get("reply_prefs.md") is None


def test_ohmo_evals_embed_command_runs_embedding_index(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    async def fake_write_ohmo_embedding_index(
        *,
        workspace: str | Path | None = None,
        inference_url: str | None = None,
        batch_size: int = 32,
    ):
        calls.append(
            {
                "workspace": workspace,
                "inference_url": inference_url,
                "batch_size": batch_size,
            }
        )
        return SimpleNamespace(
            manifest_path=Path(workspace) / "evals" / "embeddings" / "embedding_manifest.json",
            records_path=Path(workspace) / "evals" / "embeddings" / "embedding_records.jsonl",
            manifest=SimpleNamespace(
                embedding_count=3,
                facet_count=4,
                dimensions=1024,
            ),
        )

    monkeypatch.setattr(
        "ohmo.cli.write_ohmo_embedding_index",
        fake_write_ohmo_embedding_index,
    )

    result = runner.invoke(
        app,
        [
            "evals",
            "embed",
            "--workspace",
            str(workspace),
            "--batch-size",
            "7",
            "--inference-url",
            "https://inference.example",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "workspace": workspace.resolve(),
            "inference_url": "https://inference.example",
            "batch_size": 7,
        }
    ]
    assert "Wrote embedding manifest:" in result.output
    assert "Wrote embedding records:" in result.output
    assert "Indexed 3/4 facets with dimensions=1024" in result.output


def test_ohmo_evals_mine_command_runs_candidate_mining(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_write_ohmo_eval_mine(*, workspace: str | Path | None = None):
        calls.append({"workspace": workspace})
        return SimpleNamespace(
            candidates=SimpleNamespace(
                manifest_path=Path(workspace) / "evals" / "candidates" / "candidate_manifest.json",
                records_path=Path(workspace) / "evals" / "candidates" / "candidates.jsonl",
                manifest=SimpleNamespace(record_count=2),
            ),
            cases=SimpleNamespace(
                manifest_path=Path(workspace) / "evals" / "cases" / "case_manifest.json",
                records_path=Path(workspace) / "evals" / "cases" / "case_drafts.jsonl",
                manifest=SimpleNamespace(record_count=2),
            ),
        )

    monkeypatch.setattr("ohmo.cli.write_ohmo_eval_mine", fake_write_ohmo_eval_mine)

    result = runner.invoke(
        app,
        [
            "evals",
            "mine",
            "--workspace",
            str(workspace),
        ],
    )

    assert result.exit_code == 0
    assert calls == [{"workspace": workspace.resolve()}]
    assert "Wrote candidate manifest:" in result.output
    assert "Wrote candidate records:" in result.output
    assert "Wrote case manifest:" in result.output
    assert "Wrote case records:" in result.output
    assert "Mined 2 candidates and 2 draft cases" in result.output


def test_ohmo_evals_cases_list_command_outputs_metadata_json(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_review_ohmo_eval_case_drafts(
        *,
        workspace: str | Path | None = None,
        case_id: str | None = None,
        limit: int = 20,
    ):
        calls.append({"workspace": workspace, "case_id": case_id, "limit": limit})
        return SimpleNamespace(
            total_count=2,
            shown=[
                SimpleNamespace(
                    case_id="case_001",
                    case_kind="tool_workflow",
                    episode_id="ep-1",
                    review_status="draft",
                    input_facet_count=1,
                    expected_facet_count=2,
                    tool_names=["web_fetch"],
                    raw_prompt_text="SECRET PROMPT",
                )
            ],
        )

    monkeypatch.setattr(
        "ohmo.cli.review_ohmo_eval_case_drafts",
        fake_review_ohmo_eval_case_drafts,
    )

    result = runner.invoke(
        app,
        [
            "evals",
            "cases",
            "list",
            "--workspace",
            str(workspace),
            "--limit",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert calls == [{"workspace": workspace.resolve(), "case_id": None, "limit": 1}]
    payload = json.loads(result.output)
    assert payload == {
        "privacy": "metadata_only",
        "action": "cases_list",
        "total_count": 2,
        "shown_count": 1,
        "limit": 1,
        "cases": [
            {
                "case_id": "case_001",
                "case_kind": "tool_workflow",
                "episode_id": "ep-1",
                "review_status": "draft",
                "input_facet_count": 1,
                "expected_facet_count": 2,
                "tool_names": ["web_fetch"],
                "capability_path": [],
            }
        ],
    }
    assert "SECRET PROMPT" not in result.output


def test_ohmo_evals_cases_show_command_outputs_metadata_json(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_review_ohmo_eval_case_drafts(
        *,
        workspace: str | Path | None = None,
        case_id: str | None = None,
        limit: int = 20,
    ):
        calls.append({"workspace": workspace, "case_id": case_id, "limit": limit})
        return SimpleNamespace(
            total_count=2,
            shown=[
                SimpleNamespace(
                    case_id="case_001",
                    case_kind="tool_workflow",
                    episode_id="ep-1",
                    review_status="draft",
                    input_facet_count=1,
                    expected_facet_count=2,
                    tool_names=["web_fetch"],
                    raw_final_text="SECRET FINAL",
                )
            ],
        )

    monkeypatch.setattr(
        "ohmo.cli.review_ohmo_eval_case_drafts",
        fake_review_ohmo_eval_case_drafts,
    )

    result = runner.invoke(
        app,
        [
            "evals",
            "cases",
            "show",
            "case_001",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert calls == [{"workspace": workspace.resolve(), "case_id": "case_001", "limit": 1}]
    payload = json.loads(result.output)
    assert payload["privacy"] == "metadata_only"
    assert payload["action"] == "cases_show"
    assert payload["case_id"] == "case_001"
    assert payload["shown_count"] == 1
    assert payload["cases"][0]["tool_names"] == ["web_fetch"]
    assert "SECRET FINAL" not in result.output


def test_ohmo_evals_review_command_lists_case_drafts(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []
    manifest_calls: list[dict[str, object]] = []

    def fake_review_ohmo_eval_case_drafts(
        *,
        workspace: str | Path | None = None,
        case_id: str | None = None,
        limit: int = 20,
    ):
        calls.append({"workspace": workspace, "case_id": case_id, "limit": limit})
        return SimpleNamespace(
            total_count=17,
            shown=[
                SimpleNamespace(
                    case_id="case_001",
                    case_kind="tool_workflow",
                    episode_id="ep-1",
                    review_status="draft",
                    input_facet_count=1,
                    expected_facet_count=2,
                    tool_names=["web_fetch"],
                )
            ],
        )

    monkeypatch.setattr(
        "ohmo.cli.review_ohmo_eval_case_drafts",
        fake_review_ohmo_eval_case_drafts,
    )

    def fake_write_ohmo_eval_review_manifest(
        *,
        workspace: str | Path | None = None,
        case_id: str | None = None,
        limit: int = 20,
        filename: str = "review_manifest.json",
    ):
        manifest_calls.append(
            {
                "workspace": workspace,
                "case_id": case_id,
                "limit": limit,
                "filename": filename,
            }
        )
        return SimpleNamespace(
            path=Path(workspace) / "evals" / "cases" / filename,
            total_count=17,
            shown_count=1,
        )

    monkeypatch.setattr(
        "ohmo.cli.write_ohmo_eval_review_manifest",
        fake_write_ohmo_eval_review_manifest,
    )

    result = runner.invoke(
        app,
        [
            "evals",
            "review",
            "--workspace",
            str(workspace),
            "--limit",
            "1",
            "--manifest",
            "review_manifest.json",
        ],
    )

    assert result.exit_code == 0
    assert calls == [{"workspace": workspace.resolve(), "case_id": None, "limit": 1}]
    assert manifest_calls == [
        {
            "workspace": workspace.resolve(),
            "case_id": None,
            "limit": 1,
            "filename": "review_manifest.json",
        }
    ]
    assert "Draft eval cases:" in result.output
    assert "case_001 tool_workflow" in result.output
    assert "facets=1/2" in result.output
    assert "Showing 1/17 draft cases." in result.output
    assert "Wrote review manifest:" in result.output


def test_ohmo_evals_review_command_outputs_json_summary(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"

    def fake_review_ohmo_eval_case_drafts(
        *,
        workspace: str | Path | None = None,
        case_id: str | None = None,
        limit: int = 20,
    ):
        del workspace, case_id, limit
        return SimpleNamespace(
            total_count=17,
            shown=[
                SimpleNamespace(
                    case_id="case_001",
                    case_kind="tool_workflow",
                    episode_id="ep-1",
                    review_status="draft",
                    input_facet_count=1,
                    expected_facet_count=2,
                    tool_names=["web_fetch"],
                    raw_tool_text="SECRET TOOL",
                )
            ],
        )

    monkeypatch.setattr(
        "ohmo.cli.review_ohmo_eval_case_drafts",
        fake_review_ohmo_eval_case_drafts,
    )

    def fake_write_ohmo_eval_review_manifest(
        *,
        workspace: str | Path | None = None,
        case_id: str | None = None,
        limit: int = 20,
        filename: str = "review_manifest.json",
    ):
        del case_id, limit
        return SimpleNamespace(
            path=Path(workspace) / "evals" / "cases" / filename,
            relative_path=f"cases/{filename}",
            total_count=17,
            shown_count=1,
        )

    monkeypatch.setattr(
        "ohmo.cli.write_ohmo_eval_review_manifest",
        fake_write_ohmo_eval_review_manifest,
    )

    result = runner.invoke(
        app,
        [
            "evals",
            "review",
            "--workspace",
            str(workspace),
            "--limit",
            "1",
            "--manifest",
            "review_manifest.json",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["privacy"] == "metadata_only"
    assert payload["action"] == "review"
    assert payload["total_count"] == 17
    assert payload["shown_count"] == 1
    assert payload["manifest"]["relative_path"] == "cases/review_manifest.json"
    assert payload["cases"][0]["case_id"] == "case_001"
    assert "Draft eval cases:" not in result.output
    assert "SECRET TOOL" not in result.output


def test_ohmo_evals_review_command_validates_manifest(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_validate_ohmo_eval_review_manifest(
        *,
        workspace: str | Path | None = None,
        filename: str = "review_manifest.json",
    ):
        calls.append({"workspace": workspace, "filename": filename})
        return SimpleNamespace(
            path=Path(workspace) / "evals" / "cases" / filename,
            total_count=4,
            approved_count=2,
            rejected_count=1,
            pending_count=1,
            approved_case_ids=["case_001", "case_004"],
        )

    monkeypatch.setattr(
        "ohmo.cli.validate_ohmo_eval_review_manifest",
        fake_validate_ohmo_eval_review_manifest,
    )

    result = runner.invoke(
        app,
        [
            "evals",
            "review",
            "--workspace",
            str(workspace),
            "--validate-manifest",
            "review_manifest.json",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "workspace": workspace.resolve(),
            "filename": "review_manifest.json",
        }
    ]
    assert "Review manifest is valid:" in result.output
    assert "total=4 approved=2 rejected=1 pending=1" in result.output
    assert "- case_001" in result.output
    assert "- case_004" in result.output
    assert "Promote approved with: ohmo evals promote --manifest review_manifest.json" in result.output


def test_ohmo_evals_review_command_validate_manifest_outputs_json(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"

    def fake_validate_ohmo_eval_review_manifest(
        *,
        workspace: str | Path | None = None,
        filename: str = "review_manifest.json",
    ):
        return SimpleNamespace(
            path=Path(workspace) / "evals" / "cases" / filename,
            relative_path=f"cases/{filename}",
            total_count=4,
            approved_count=2,
            rejected_count=1,
            pending_count=1,
            missing_case_ids=[],
            approved_case_ids=["case_001", "case_004"],
        )

    monkeypatch.setattr(
        "ohmo.cli.validate_ohmo_eval_review_manifest",
        fake_validate_ohmo_eval_review_manifest,
    )

    result = runner.invoke(
        app,
        [
            "evals",
            "review",
            "--workspace",
            str(workspace),
            "--validate-manifest",
            "review_manifest.json",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == {
        "privacy": "metadata_only",
        "action": "validate_manifest",
        "manifest": {
            "path": str(workspace / "evals" / "cases" / "review_manifest.json"),
            "relative_path": "cases/review_manifest.json",
        },
        "total_count": 4,
        "approved_count": 2,
        "rejected_count": 1,
        "pending_count": 1,
        "missing_case_ids": [],
        "approved_case_ids": ["case_001", "case_004"],
    }
    assert "Promote approved with:" not in result.output


def test_ohmo_evals_review_command_validate_manifest_reports_errors(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"

    def fake_validate_ohmo_eval_review_manifest(
        *,
        workspace: str | Path | None = None,
        filename: str = "review_manifest.json",
    ):
        raise ValueError("review manifest references missing draft cases: missing-case")

    monkeypatch.setattr(
        "ohmo.cli.validate_ohmo_eval_review_manifest",
        fake_validate_ohmo_eval_review_manifest,
    )

    result = runner.invoke(
        app,
        [
            "evals",
            "review",
            "--workspace",
            str(workspace),
            "--validate-manifest",
            "review_manifest.json",
        ],
    )

    assert result.exit_code == 1
    assert "missing draft cases: missing-case" in result.stderr


def test_ohmo_evals_promote_command_promotes_selected_case_drafts(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_promote_ohmo_eval_case_drafts(
        *,
        workspace: str | Path | None = None,
        case_ids: list[str] | None = None,
        promote_all: bool = False,
        manifest_filename: str | None = None,
        dry_run: bool = False,
        reviewer: str = "",
    ):
        calls.append(
            {
                "workspace": workspace,
                "case_ids": case_ids,
                "promote_all": promote_all,
                "manifest_filename": manifest_filename,
                "dry_run": dry_run,
                "reviewer": reviewer,
            }
        )
        return SimpleNamespace(
            promoted_count=2,
            remaining_unpromoted_count=15,
            selected_case_ids=["case_001", "case_002"],
            dry_run=False,
            manifest_path=Path(workspace) / "evals" / "cases" / "gold_manifest.json",
            records_path=Path(workspace) / "evals" / "cases" / "gold_cases.jsonl",
        )

    monkeypatch.setattr(
        "ohmo.cli.promote_ohmo_eval_case_drafts",
        fake_promote_ohmo_eval_case_drafts,
    )

    result = runner.invoke(
        app,
        [
            "evals",
            "promote",
            "--workspace",
            str(workspace),
            "--case-id",
            "case_001",
            "--case-id",
            "case_002",
            "--reviewer",
            "reviewer-1",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "workspace": workspace.resolve(),
            "case_ids": ["case_001", "case_002"],
            "promote_all": False,
            "manifest_filename": None,
            "dry_run": False,
            "reviewer": "reviewer-1",
        }
    ]
    assert "Promoted 2 draft cases." in result.output
    assert "Wrote gold manifest:" in result.output
    assert "Wrote gold records:" in result.output
    assert "Remaining unpromoted draft cases: 15" in result.output


def test_ohmo_evals_promote_command_supports_dry_run(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"

    def fake_promote_ohmo_eval_case_drafts(
        *,
        workspace: str | Path | None = None,
        case_ids: list[str] | None = None,
        promote_all: bool = False,
        manifest_filename: str | None = None,
        dry_run: bool = False,
        reviewer: str = "",
    ):
        return SimpleNamespace(
            promoted_count=1,
            remaining_unpromoted_count=0,
            selected_case_ids=["case_001"],
            dry_run=True,
            manifest_path=Path(workspace) / "evals" / "cases" / "gold_manifest.json",
            records_path=Path(workspace) / "evals" / "cases" / "gold_cases.jsonl",
        )

    monkeypatch.setattr(
        "ohmo.cli.promote_ohmo_eval_case_drafts",
        fake_promote_ohmo_eval_case_drafts,
    )

    result = runner.invoke(
        app,
        [
            "evals",
            "promote",
            "--workspace",
            str(workspace),
            "--case-id",
            "case_001",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Would promote 1 draft cases:" in result.output
    assert "- case_001" in result.output
    assert "No files written." in result.output


def test_ohmo_evals_promote_command_supports_review_manifest(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_promote_ohmo_eval_case_drafts(
        *,
        workspace: str | Path | None = None,
        case_ids: list[str] | None = None,
        promote_all: bool = False,
        manifest_filename: str | None = None,
        dry_run: bool = False,
        reviewer: str = "",
    ):
        calls.append(
            {
                "workspace": workspace,
                "case_ids": case_ids,
                "promote_all": promote_all,
                "manifest_filename": manifest_filename,
                "dry_run": dry_run,
                "reviewer": reviewer,
            }
        )
        return SimpleNamespace(
            promoted_count=1,
            remaining_unpromoted_count=4,
            selected_case_ids=["case_001"],
            dry_run=False,
            manifest_path=Path(workspace) / "evals" / "cases" / "gold_manifest.json",
            records_path=Path(workspace) / "evals" / "cases" / "gold_cases.jsonl",
        )

    monkeypatch.setattr(
        "ohmo.cli.promote_ohmo_eval_case_drafts",
        fake_promote_ohmo_eval_case_drafts,
    )

    result = runner.invoke(
        app,
        [
            "evals",
            "promote",
            "--workspace",
            str(workspace),
            "--manifest",
            "review_manifest.json",
            "--reviewer",
            "reviewer-1",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "workspace": workspace.resolve(),
            "case_ids": None,
            "promote_all": False,
            "manifest_filename": "review_manifest.json",
            "dry_run": False,
            "reviewer": "reviewer-1",
        }
    ]
    assert "Promoted 1 draft cases." in result.output


def test_ohmo_evals_pack_command_builds_runnable_pack(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_build_ohmo_eval_pack(
        *,
        workspace: str | Path | None = None,
        pack_filename: str = "eval_pack.json",
    ):
        calls.append({"workspace": workspace, "pack_filename": pack_filename})
        return SimpleNamespace(
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "packs" / pack_filename,
            ),
            case_count=3,
            gold_case_count=4,
        )

    monkeypatch.setattr("ohmo.cli.build_ohmo_eval_pack", fake_build_ohmo_eval_pack)

    result = runner.invoke(
        app,
        [
            "evals",
            "pack",
            "--workspace",
            str(workspace),
        ],
    )

    assert result.exit_code == 0
    assert calls == [{"workspace": workspace.resolve(), "pack_filename": "eval_pack.json"}]
    assert "Wrote runnable pack:" in result.output
    assert "Built runnable pack with 3 cases from 4 gold cases" in result.output


def test_ohmo_evals_pack_command_passes_output_filename(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_build_ohmo_eval_pack(
        *,
        workspace: str | Path | None = None,
        pack_filename: str = "eval_pack.json",
    ):
        calls.append({"workspace": workspace, "pack_filename": pack_filename})
        return SimpleNamespace(
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "packs" / pack_filename,
            ),
            case_count=3,
            gold_case_count=4,
        )

    monkeypatch.setattr("ohmo.cli.build_ohmo_eval_pack", fake_build_ohmo_eval_pack)

    result = runner.invoke(
        app,
        [
            "evals",
            "pack",
            "--workspace",
            str(workspace),
            "--output",
            "custom_eval_pack.json",
        ],
    )

    assert result.exit_code == 0
    assert calls == [{"workspace": workspace.resolve(), "pack_filename": "custom_eval_pack.json"}]
    assert "custom_eval_pack.json" in result.output


def test_ohmo_evals_smoke_command_runs_report_only_eval(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_run_ohmo_eval_smoke(
        *,
        workspace: str | Path | None = None,
        pack_filename: str = "eval_pack.json",
        report_filename: str = "smoke_report.json",
        limit: int | None = None,
        report_only: bool = False,
    ):
        calls.append(
            {
                "workspace": workspace,
                "pack_filename": pack_filename,
                "report_filename": report_filename,
                "limit": limit,
                "report_only": report_only,
            }
        )
        return SimpleNamespace(
            report_only=report_only,
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "reports" / report_filename,
                report=SimpleNamespace(case_count=2, passed_count=1, failed_count=1),
            ),
        )

    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_smoke", fake_run_ohmo_eval_smoke)

    failed = runner.invoke(
        app,
        [
            "evals",
            "smoke",
            "--workspace",
            str(workspace),
            "--pack",
            "custom_pack.json",
            "--limit",
            "2",
        ],
    )
    report_only = runner.invoke(
        app,
        [
            "evals",
            "smoke",
            "--workspace",
            str(workspace),
            "--pack",
            "custom_pack.json",
            "--limit",
            "2",
            "--report-only",
        ],
    )

    assert failed.exit_code == 1
    assert "Wrote smoke report:" in failed.output
    assert "Smoke evaluated 2 cases: passed=1 failed=1" in failed.output
    assert report_only.exit_code == 0
    assert "Report-only mode: failures did not fail the command" in report_only.output
    assert calls == [
        {
            "workspace": workspace.resolve(),
            "pack_filename": "custom_pack.json",
            "report_filename": "smoke_report.json",
            "limit": 2,
            "report_only": False,
        },
        {
            "workspace": workspace.resolve(),
            "pack_filename": "custom_pack.json",
            "report_filename": "smoke_report.json",
            "limit": 2,
            "report_only": True,
        },
    ]


def test_ohmo_evals_smoke_command_passes_output_filename(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_run_ohmo_eval_smoke(
        *,
        workspace: str | Path | None = None,
        pack_filename: str = "eval_pack.json",
        report_filename: str = "smoke_report.json",
        limit: int | None = None,
        report_only: bool = False,
    ):
        calls.append(
            {
                "workspace": workspace,
                "pack_filename": pack_filename,
                "report_filename": report_filename,
                "limit": limit,
                "report_only": report_only,
            }
        )
        return SimpleNamespace(
            report_only=report_only,
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "reports" / report_filename,
                report=SimpleNamespace(case_count=1, passed_count=1, failed_count=0),
            ),
        )

    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_smoke", fake_run_ohmo_eval_smoke)

    result = runner.invoke(
        app,
        [
            "evals",
            "smoke",
            "--workspace",
            str(workspace),
            "--output",
            "custom_smoke_report.json",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "workspace": workspace.resolve(),
            "pack_filename": "eval_pack.json",
            "report_filename": "custom_smoke_report.json",
            "limit": None,
            "report_only": False,
        }
    ]
    assert "custom_smoke_report.json" in result.output


def test_ohmo_evals_smoke_command_outputs_json_summary(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"

    def fake_run_ohmo_eval_smoke(
        *,
        workspace: str | Path | None = None,
        pack_filename: str = "eval_pack.json",
        report_filename: str = "smoke_report.json",
        limit: int | None = None,
        report_only: bool = False,
    ):
        del pack_filename, limit
        return SimpleNamespace(
            report_only=report_only,
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "reports" / report_filename,
                relative_path=f"reports/{report_filename}",
                report=SimpleNamespace(
                    report_kind="smoke_report",
                    report_id="smoke-1",
                    pack_id="pack-1",
                    case_count=2,
                    passed_count=2,
                    failed_count=0,
                    cases=[SimpleNamespace(raw_prompt_text="SECRET PROMPT")],
                ),
            ),
        )

    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_smoke", fake_run_ohmo_eval_smoke)

    result = runner.invoke(
        app,
        [
            "evals",
            "smoke",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == {
        "privacy": "metadata_only",
        "report_path": str(workspace / "evals" / "reports" / "smoke_report.json"),
        "relative_path": "reports/smoke_report.json",
        "report_kind": "smoke_report",
        "report_id": "smoke-1",
        "pack_id": "pack-1",
        "case_count": 2,
        "passed_count": 2,
        "failed_count": 0,
        "report_only": False,
    }
    assert "Wrote smoke report:" not in result.output
    assert "SECRET PROMPT" not in result.output


def test_ohmo_evals_run_command_runs_eval_report(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_run_ohmo_eval_report(
        *,
        workspace: str | Path | None = None,
        pack_filename: str = "eval_pack.json",
        report_filename: str = "eval_report.json",
        limit: int | None = None,
        samples: int = 1,
        report_only: bool = False,
        executor_name: str = "replay-tools",
        agent_runner_name: str = "scripted",
        model: str | None = None,
        provider_profile: str | None = None,
        system_prompt: str | None = None,
        scorer: str | None = None,
        fixture_match: str = "order",
        max_turns: int = 8,
        judge_votes: int = 3,
        judge_grounding: bool = False,
        **_kwargs,
    ):
        calls.append(
            {
                "workspace": workspace,
                "pack_filename": pack_filename,
                "report_filename": report_filename,
                "limit": limit,
                "samples": samples,
                "report_only": report_only,
                "executor_name": executor_name,
                "agent_runner_name": agent_runner_name,
                "model": model,
                "provider_profile": provider_profile,
                "system_prompt": system_prompt,
                "fixture_match": fixture_match,
            }
        )
        return SimpleNamespace(
            report_only=report_only,
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "reports" / report_filename,
                report=SimpleNamespace(case_count=2, passed_count=1, failed_count=1),
            ),
        )

    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_report", fake_run_ohmo_eval_report)

    failed = runner.invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
            "--pack",
            "custom_pack.json",
            "--limit",
            "2",
            "--samples",
            "3",
            "--executor",
            "replay-tools",
            "--fixture-match",
            "arguments",
        ],
    )
    report_only = runner.invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
            "--pack",
            "custom_pack.json",
            "--limit",
            "2",
            "--report-only",
        ],
    )

    assert failed.exit_code == 1
    assert "Wrote eval report:" in failed.output
    assert "Eval run evaluated 2 cases: passed=1 failed=1" in failed.output
    assert "blocked=0 error=0" in failed.output
    assert report_only.exit_code == 0
    assert "Report-only mode: failures did not fail the command" in report_only.output
    assert calls == [
        {
            "workspace": workspace.resolve(),
            "pack_filename": "custom_pack.json",
            "report_filename": "eval_report.json",
            "limit": 2,
            "samples": 3,
            "report_only": False,
            "executor_name": "replay-tools",
            "agent_runner_name": "scripted",
            "model": None,
            "provider_profile": None,
            "system_prompt": None,
            "fixture_match": "arguments",
        },
        {
            "workspace": workspace.resolve(),
            "pack_filename": "custom_pack.json",
            "report_filename": "eval_report.json",
            "limit": 2,
            "samples": 3,
            "report_only": True,
            "executor_name": "replay-tools",
            "agent_runner_name": "scripted",
            "model": None,
            "provider_profile": None,
            "system_prompt": None,
            "fixture_match": "args_then_order",
        },
    ]


def test_ohmo_evals_run_command_threads_sandbox_agent_runner(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_run_ohmo_eval_report(
        *,
        workspace: str | Path | None = None,
        pack_filename: str = "eval_pack.json",
        report_filename: str = "eval_report.json",
        limit: int | None = None,
        samples: int = 1,
        report_only: bool = False,
        executor_name: str = "replay-tools",
        agent_runner_name: str = "scripted",
        model: str | None = None,
        provider_profile: str | None = None,
        system_prompt: str | None = None,
        scorer: str | None = None,
        fixture_match: str = "order",
        max_turns: int = 8,
        judge_votes: int = 3,
        judge_grounding: bool = False,
        **_kwargs,
    ):
        calls.append(
            {
                "workspace": workspace,
                "pack_filename": pack_filename,
                "report_filename": report_filename,
                "limit": limit,
                "samples": samples,
                "report_only": report_only,
                "executor_name": executor_name,
                "agent_runner_name": agent_runner_name,
                "model": model,
                "provider_profile": provider_profile,
                "system_prompt": system_prompt,
                "scorer": scorer,
                "fixture_match": fixture_match,
            }
        )
        return SimpleNamespace(
            report_only=report_only,
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "reports" / report_filename,
                report=SimpleNamespace(case_count=1, passed_count=1, failed_count=0),
            ),
        )

    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_report", fake_run_ohmo_eval_report)

    result = runner.invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
            "--agent-runner",
            "sandbox",
            "--scorer",
            "state_outcome_oracle_v1",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "workspace": workspace.resolve(),
            "pack_filename": "eval_pack.json",
            "report_filename": "eval_report.json",
            "limit": None,
            "samples": 3,
            "report_only": False,
            "executor_name": "replay-tools",
            "agent_runner_name": "sandbox",
            "model": None,
            "provider_profile": None,
            "system_prompt": None,
            "scorer": "state_outcome_oracle_v1",
            "fixture_match": "args_then_order",
        }
    ]


def test_ohmo_evals_run_command_threads_fs_sandbox_network_options(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_run_ohmo_eval_report(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            report_only=kwargs.get("report_only", False),
            write=SimpleNamespace(
                path=(
                    Path(kwargs["workspace"])
                    / "evals"
                    / "reports"
                    / "eval_report.json"
                ),
                report=SimpleNamespace(case_count=1, passed_count=1, failed_count=0),
            ),
        )

    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_report", fake_run_ohmo_eval_report)

    result = runner.invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
            "--agent-runner",
            "fs-sandbox",
            "--sandbox-net-mode",
            "netns:evalns",
            "--sandbox-proxy-url",
            "http://10.77.0.1:3128",
            "--sandbox-browser-socket",
            "/tmp/browser-cli-ohmo.sock",
            "--sandbox-browser-name",
            "ohmo",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["agent_runner_name"] == "fs-sandbox"
    assert calls[0]["sandbox_net_mode"] == "netns:evalns"
    assert calls[0]["sandbox_proxy_url"] == "http://10.77.0.1:3128"
    assert calls[0]["sandbox_browser_socket"] == "/tmp/browser-cli-ohmo.sock"
    assert calls[0]["sandbox_browser_name"] == "ohmo"


def test_ohmo_evals_run_command_threads_live_read_agent_runner(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_run_ohmo_eval_report(
        *,
        workspace: str | Path | None = None,
        pack_filename: str = "eval_pack.json",
        report_filename: str = "eval_report.json",
        limit: int | None = None,
        samples: int = 1,
        report_only: bool = False,
        executor_name: str = "replay-tools",
        agent_runner_name: str = "scripted",
        model: str | None = None,
        provider_profile: str | None = None,
        system_prompt: str | None = None,
        scorer: str | None = None,
        judge_profile: str | None = None,
        judge_model: str | None = None,
        fixture_match: str = "order",
        max_turns: int = 8,
        judge_votes: int = 3,
        judge_grounding: bool = False,
        **_kwargs,
    ):
        calls.append(
            {
                "workspace": workspace,
                "pack_filename": pack_filename,
                "report_filename": report_filename,
                "limit": limit,
                "samples": samples,
                "report_only": report_only,
                "executor_name": executor_name,
                "agent_runner_name": agent_runner_name,
                "model": model,
                "provider_profile": provider_profile,
                "system_prompt": system_prompt,
                "scorer": scorer,
                "judge_profile": judge_profile,
                "judge_model": judge_model,
                "fixture_match": fixture_match,
            }
        )
        return SimpleNamespace(
            report_only=report_only,
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "reports" / report_filename,
                report=SimpleNamespace(case_count=1, passed_count=1, failed_count=0),
            ),
        )

    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_report", fake_run_ohmo_eval_report)

    result = runner.invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
            "--agent-runner",
            "query-engine-live-read",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "workspace": workspace.resolve(),
            "pack_filename": "eval_pack.json",
            "report_filename": "eval_report.json",
            "limit": None,
            "samples": 3,
            "report_only": False,
            "executor_name": "replay-tools",
            "agent_runner_name": "query-engine-live-read",
            "model": None,
            "provider_profile": None,
            "system_prompt": None,
            "scorer": None,
            "judge_profile": None,
            "judge_model": None,
            "fixture_match": "args_then_order",
        }
    ]


def test_ohmo_evals_run_session_command_runs_session_eval(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_run_ohmo_session_eval(
        *,
        workspace: str | Path | None = None,
        report_filename: str = "session_report.json",
        limit: int | None = None,
        model: str | None = None,
        provider_profile: str | None = None,
        system_prompt: str | None = None,
        samples: int = 1,
        gold_capabilities_by_session=None,
        user_sim_profile: str | None = None,
        user_sim_model: str | None = None,
        user_sim_goal_anchored: bool = True,
        clarification_allowed_by_session=None,
        fixture_match: str = "order",
        max_session_turns: int | None = None,
        max_turns: int = 100,
        segment: bool = True,
        gap_minutes: float = 30.0,
        min_turns: int = 2,
    ):
        calls.append(
            {
                "workspace": workspace,
                "report_filename": report_filename,
                "limit": limit,
                "model": model,
                "provider_profile": provider_profile,
                "system_prompt": system_prompt,
                "samples": samples,
                "gold_capabilities_by_session": gold_capabilities_by_session,
                "user_sim_profile": user_sim_profile,
                "user_sim_model": user_sim_model,
                "user_sim_goal_anchored": user_sim_goal_anchored,
                "clarification_allowed_by_session": clarification_allowed_by_session,
                "fixture_match": fixture_match,
                "max_session_turns": max_session_turns,
                "max_turns": max_turns,
                "segment": segment,
                "gap_minutes": gap_minutes,
                "min_turns": min_turns,
            }
        )
        return SimpleNamespace(
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "reports" / report_filename,
                relative_path=f"reports/{report_filename}",
                report=SimpleNamespace(
                    report_kind="session_report",
                    report_id="session-eval:test",
                    session_count=2,
                    passed_count=2,
                    failed_count=0,
                ),
            ),
        )

    monkeypatch.setattr("ohmo.cli.run_ohmo_session_eval", fake_run_ohmo_session_eval)

    result = runner.invoke(
        app,
        [
            "evals",
            "run-session",
            "--workspace",
            str(workspace),
            "--output",
            "custom_session_report.json",
            "--limit",
            "2",
            "--samples",
            "3",
            "--model",
            "eval-model",
            "--profile",
            "eval-profile",
            "--system-prompt",
            "eval prompt",
            "--user-sim-profile",
            "user-profile",
            "--user-sim-model",
            "user-model",
            "--no-user-sim-goal-anchored",
            "--fixture-match",
            "arguments",
            "--max-session-turns",
            "5",
            "--max-turns",
            "12",
            "--segment",
            "--gap-minutes",
            "45",
            "--min-turns",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote session eval report:" in result.output
    assert "Session eval evaluated 2 sessions: passed=2 failed=0" in result.output
    assert calls == [
        {
            "workspace": workspace.resolve(),
            "report_filename": "custom_session_report.json",
            "limit": 2,
            "model": "eval-model",
            "provider_profile": "eval-profile",
            "system_prompt": "eval prompt",
            "samples": 3,
            "gold_capabilities_by_session": None,
            "user_sim_profile": "user-profile",
            "user_sim_model": "user-model",
            "user_sim_goal_anchored": False,
            "clarification_allowed_by_session": None,
            "fixture_match": "arguments",
            "max_session_turns": 5,
            "max_turns": 12,
            "segment": True,
            "gap_minutes": 45.0,
            "min_turns": 3,
        }
    ]

    calls.clear()
    explicit_true = runner.invoke(
        app,
        [
            "evals",
            "run-session",
            "--workspace",
            str(workspace),
            "--user-sim-goal-anchored",
        ],
    )

    assert explicit_true.exit_code == 0
    assert calls[0]["user_sim_goal_anchored"] is True


def test_ohmo_evals_fixture_match_rejects_invalid_value(tmp_path: Path):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"

    eval_result = runner.invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
            "--fixture-match",
            "wrong",
        ],
    )
    session_result = runner.invoke(
        app,
        [
            "evals",
            "run-session",
            "--workspace",
            str(workspace),
            "--fixture-match",
            "wrong",
        ],
    )

    assert eval_result.exit_code == 1
    assert "unknown fixture match mode: wrong" in eval_result.stderr
    assert session_result.exit_code == 1
    assert "unknown fixture match mode: wrong" in session_result.stderr


def test_ohmo_evals_run_command_passes_output_filename(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_run_ohmo_eval_report(
        *,
        workspace: str | Path | None = None,
        pack_filename: str = "eval_pack.json",
        report_filename: str = "eval_report.json",
        limit: int | None = None,
        samples: int = 1,
        report_only: bool = False,
        executor_name: str = "replay-tools",
        agent_runner_name: str = "scripted",
        model: str | None = None,
        provider_profile: str | None = None,
        system_prompt: str | None = None,
        scorer: str | None = None,
        fixture_match: str = "order",
        max_turns: int = 8,
        judge_votes: int = 3,
        judge_grounding: bool = False,
        **_kwargs,
    ):
        calls.append(
            {
                "workspace": workspace,
                "pack_filename": pack_filename,
                "report_filename": report_filename,
                "limit": limit,
                "report_only": report_only,
                "executor_name": executor_name,
                "agent_runner_name": agent_runner_name,
                "model": model,
                "provider_profile": provider_profile,
                "system_prompt": system_prompt,
                "fixture_match": fixture_match,
            }
        )
        return SimpleNamespace(
            report_only=report_only,
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "reports" / report_filename,
                report=SimpleNamespace(
                    case_count=1,
                    passed_count=1,
                    failed_count=0,
                    blocked_count=0,
                    error_count=0,
                ),
            ),
        )

    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_report", fake_run_ohmo_eval_report)

    result = runner.invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
            "--output",
            "custom_eval_report.json",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "workspace": workspace.resolve(),
            "pack_filename": "eval_pack.json",
            "report_filename": "custom_eval_report.json",
            "limit": None,
            "report_only": False,
            "executor_name": "replay-tools",
            "agent_runner_name": "scripted",
            "model": None,
            "provider_profile": None,
            "system_prompt": None,
            "fixture_match": "args_then_order",
        }
    ]
    assert "custom_eval_report.json" in result.output


def test_ohmo_evals_run_command_outputs_json_summary(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"

    def fake_run_ohmo_eval_report(
        *,
        workspace: str | Path | None = None,
        pack_filename: str = "eval_pack.json",
        report_filename: str = "eval_report.json",
        limit: int | None = None,
        samples: int = 1,
        report_only: bool = False,
        executor_name: str = "replay-tools",
        agent_runner_name: str = "scripted",
        model: str | None = None,
        provider_profile: str | None = None,
        system_prompt: str | None = None,
        scorer: str | None = None,
        fixture_match: str = "order",
        max_turns: int = 8,
        judge_votes: int = 3,
        judge_grounding: bool = False,
        **_kwargs,
    ):
        del pack_filename, limit, executor_name, agent_runner_name
        del model, provider_profile, system_prompt, fixture_match
        return SimpleNamespace(
            report_only=report_only,
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "reports" / report_filename,
                relative_path=f"reports/{report_filename}",
                report=SimpleNamespace(
                    report_kind="execution_report",
                    report_id="exec-1",
                    pack_id="pack-1",
                    case_count=2,
                    passed_count=2,
                    failed_count=0,
                    blocked_count=0,
                    error_count=0,
                    cases=[SimpleNamespace(raw_final_text="SECRET FINAL")],
                ),
            ),
        )

    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_report", fake_run_ohmo_eval_report)

    result = runner.invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
            "--output",
            "custom_eval_report.json",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == {
        "privacy": "metadata_only",
        "report_path": str(workspace / "evals" / "reports" / "custom_eval_report.json"),
        "relative_path": "reports/custom_eval_report.json",
        "report_kind": "execution_report",
        "report_id": "exec-1",
        "pack_id": "pack-1",
        "case_count": 2,
        "passed_count": 2,
        "failed_count": 0,
        "blocked_count": 0,
        "error_count": 0,
        "report_only": False,
    }
    assert "Wrote eval report:" not in result.output
    assert "SECRET FINAL" not in result.output


def test_ohmo_evals_run_command_passes_query_engine_runner_options(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_run_ohmo_eval_report(
        *,
        workspace: str | Path | None = None,
        pack_filename: str = "eval_pack.json",
        report_filename: str = "eval_report.json",
        limit: int | None = None,
        samples: int = 1,
        report_only: bool = False,
        executor_name: str = "replay-tools",
        agent_runner_name: str = "scripted",
        model: str | None = None,
        provider_profile: str | None = None,
        system_prompt: str | None = None,
        scorer: str | None = None,
        fixture_match: str = "order",
        max_turns: int = 8,
        judge_votes: int = 3,
        judge_grounding: bool = False,
        **_kwargs,
    ):
        calls.append(
            {
                "workspace": workspace,
                "pack_filename": pack_filename,
                "report_filename": report_filename,
                "limit": limit,
                "report_only": report_only,
                "executor_name": executor_name,
                "agent_runner_name": agent_runner_name,
                "model": model,
                "provider_profile": provider_profile,
                "system_prompt": system_prompt,
                "fixture_match": fixture_match,
            }
        )
        return SimpleNamespace(
            report_only=report_only,
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "reports" / report_filename,
                report=SimpleNamespace(
                    case_count=1,
                    passed_count=1,
                    failed_count=0,
                    blocked_count=0,
                    error_count=0,
                ),
            ),
        )

    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_report", fake_run_ohmo_eval_report)

    result = runner.invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
            "--agent-runner",
            "query-engine",
            "--model",
            "eval-model",
            "--profile",
            "openai-compatible",
            "--system-prompt",
            "eval system",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "workspace": workspace.resolve(),
            "pack_filename": "eval_pack.json",
            "report_filename": "eval_report.json",
            "limit": None,
            "report_only": False,
            "executor_name": "replay-tools",
            "agent_runner_name": "query-engine",
            "model": "eval-model",
            "provider_profile": "openai-compatible",
            "system_prompt": "eval system",
            "fixture_match": "args_then_order",
        }
    ]


def test_ohmo_evals_run_command_threads_judge_options(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_run_ohmo_eval_report(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            report_only=False,
            write=SimpleNamespace(
                path=Path(kwargs["workspace"]) / "evals" / "reports" / "eval_report.json",
                report=SimpleNamespace(
                    case_count=1,
                    passed_count=1,
                    failed_count=0,
                    blocked_count=0,
                    error_count=0,
                ),
            ),
        )

    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_report", fake_run_ohmo_eval_report)

    result = runner.invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
            "--scorer",
            "freezing_judge",
            "--judge-profile",
            "judge-profile",
            "--judge-model",
            "judge-model",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["scorer"] == "freezing_judge"
    assert calls[0]["judge_profile"] == "judge-profile"
    assert calls[0]["judge_model"] == "judge-model"


def test_ohmo_evals_run_command_threads_synth_options(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_run_ohmo_eval_report(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            report_only=False,
            write=SimpleNamespace(
                path=Path(kwargs["workspace"]) / "evals" / "reports" / "eval_report.json",
                report=SimpleNamespace(
                    case_count=1,
                    passed_count=1,
                    failed_count=0,
                    blocked_count=0,
                    error_count=0,
                ),
            ),
        )

    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_report", fake_run_ohmo_eval_report)

    result = runner.invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
            "--fixture-match",
            "synth",
            "--synth-profile",
            "synth-profile",
            "--synth-model",
            "synth-model",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["fixture_match"] == "synth"
    assert calls[0]["synth_profile"] == "synth-profile"
    assert calls[0]["synth_model"] == "synth-model"


def test_ohmo_evals_run_command_threads_history_options(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_run_ohmo_eval_report(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            report_only=False,
            write=SimpleNamespace(
                path=Path(kwargs["workspace"]) / "evals" / "reports" / "eval_report.json",
                report=SimpleNamespace(
                    case_count=1,
                    passed_count=1,
                    failed_count=0,
                    blocked_count=0,
                    error_count=0,
                ),
            ),
        )

    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_report", fake_run_ohmo_eval_report)

    result = runner.invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
            "--history-profile",
            "history-profile",
            "--history-model",
            "history-model",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["history_profile"] == "history-profile"
    assert calls[0]["history_model"] == "history-model"


def test_ohmo_evals_run_command_threads_max_turns(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_run_ohmo_eval_report(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            report_only=False,
            write=SimpleNamespace(
                path=Path(kwargs["workspace"]) / "evals" / "reports" / "eval_report.json",
                report=SimpleNamespace(
                    case_count=1,
                    passed_count=1,
                    failed_count=0,
                    blocked_count=0,
                    error_count=0,
                ),
            ),
        )

    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_report", fake_run_ohmo_eval_report)

    result = runner.invoke(
        app,
        ["evals", "run", "--workspace", str(workspace), "--max-turns", "50"],
    )
    assert result.exit_code == 0
    assert calls[0]["max_turns"] == 50

    # default when the flag is omitted
    result = runner.invoke(
        app,
        ["evals", "run", "--workspace", str(workspace)],
    )
    assert result.exit_code == 0
    assert calls[1]["max_turns"] == 100


def test_ohmo_evals_run_command_check_config_does_not_run_eval(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    check_calls: list[dict[str, object]] = []

    def fake_check_ohmo_eval_run_config(
        *,
        workspace: str | Path | None = None,
        pack_filename: str = "eval_pack.json",
        limit: int | None = None,
        executor_name: str = "replay-tools",
        agent_runner_name: str = "scripted",
        model: str | None = None,
        provider_profile: str | None = None,
        system_prompt: str | None = None,
        scorer: str | None = None,
        fixture_match: str = "order",
        max_turns: int = 8,
        judge_votes: int = 3,
        judge_grounding: bool = False,
        **_kwargs,
    ):
        check_calls.append(
            {
                "workspace": workspace,
                "pack_filename": pack_filename,
                "limit": limit,
                "executor_name": executor_name,
                "agent_runner_name": agent_runner_name,
                "model": model,
                "provider_profile": provider_profile,
                "system_prompt": system_prompt,
                "fixture_match": fixture_match,
            }
        )
        return SimpleNamespace(
            pack_id="pack-1",
            pack_case_count=5,
            selected_case_count=2,
            executor_name="replay-tools",
            agent_runner_name="query-engine",
            model="eval-model",
            provider_profile="openai-compatible",
            replay_tools_only=True,
        )

    monkeypatch.setattr(
        "ohmo.cli.check_ohmo_eval_run_config",
        fake_check_ohmo_eval_run_config,
    )

    def fail_run_ohmo_eval_report(**kwargs):
        raise AssertionError("eval run should not execute in --check-config mode")

    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_report", fail_run_ohmo_eval_report)

    result = runner.invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
            "--pack",
            "custom_pack.json",
            "--limit",
            "2",
            "--agent-runner",
            "query-engine",
            "--model",
            "eval-model",
            "--profile",
            "openai-compatible",
            "--check-config",
        ],
    )

    assert result.exit_code == 0
    assert check_calls == [
        {
            "workspace": workspace.resolve(),
            "pack_filename": "custom_pack.json",
            "limit": 2,
            "executor_name": "replay-tools",
            "agent_runner_name": "query-engine",
            "model": "eval-model",
            "provider_profile": "openai-compatible",
            "system_prompt": None,
            "fixture_match": "args_then_order",
        }
    ]
    assert "Eval run configuration is valid." in result.output
    assert "selected=2/5" in result.output
    assert "agent_runner=query-engine" in result.output
    assert "profile=openai-compatible model=eval-model" in result.output


def test_ohmo_evals_run_command_check_config_outputs_json(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"

    def fake_check_ohmo_eval_run_config(
        *,
        workspace: str | Path | None = None,
        pack_filename: str = "eval_pack.json",
        limit: int | None = None,
        executor_name: str = "replay-tools",
        agent_runner_name: str = "scripted",
        model: str | None = None,
        provider_profile: str | None = None,
        system_prompt: str | None = None,
        scorer: str | None = None,
        fixture_match: str = "order",
        max_turns: int = 8,
        judge_votes: int = 3,
        judge_grounding: bool = False,
        **_kwargs,
    ):
        del workspace, pack_filename, limit, executor_name, agent_runner_name
        del model, provider_profile, system_prompt, fixture_match
        return SimpleNamespace(
            pack_id="pack-1",
            pack_case_count=5,
            selected_case_count=2,
            executor_name="replay-tools",
            agent_runner_name="query-engine",
            model="eval-model",
            provider_profile="openai-compatible",
            replay_tools_only=True,
        )

    monkeypatch.setattr(
        "ohmo.cli.check_ohmo_eval_run_config",
        fake_check_ohmo_eval_run_config,
    )

    def fail_run_ohmo_eval_report(**kwargs):
        raise AssertionError("eval run should not execute in --check-config mode")

    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_report", fail_run_ohmo_eval_report)

    result = runner.invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
            "--check-config",
            "--system-prompt",
            "SECRET SYSTEM PROMPT",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == {
        "privacy": "metadata_only",
        "pack_id": "pack-1",
        "pack_case_count": 5,
        "selected_case_count": 2,
        "executor_name": "replay-tools",
        "agent_runner_name": "query-engine",
        "model": "eval-model",
        "provider_profile": "openai-compatible",
        "replay_tools_only": True,
    }
    assert "Eval run configuration is valid." not in result.output
    assert "SECRET SYSTEM PROMPT" not in result.output


def test_ohmo_evals_run_command_help_lists_supported_executor_and_runner_ids():
    runner = CliRunner()

    result = runner.invoke(app, ["evals", "run", "--help"])

    assert result.exit_code == 0
    output = " ".join(result.output.split())
    assert "--executor" in output
    assert "Eval executor to use:" in output
    assert "replay-tools" in output
    assert "--agent-runner" in output
    assert "--samples" in output
    assert "Agent runner to use inside the" in output
    assert "inside the executor:" in output
    assert "scripted," in output
    assert "query-engine" in output
    assert "query-engine-live-read" in output


def test_ohmo_evals_run_command_reports_blocked_and_error_counts(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"

    def fake_run_ohmo_eval_report(
        *,
        workspace: str | Path | None = None,
        pack_filename: str = "eval_pack.json",
        report_filename: str = "eval_report.json",
        limit: int | None = None,
        samples: int = 1,
        report_only: bool = False,
        executor_name: str = "replay-tools",
        agent_runner_name: str = "scripted",
        model: str | None = None,
        provider_profile: str | None = None,
        system_prompt: str | None = None,
        scorer: str | None = None,
        fixture_match: str = "order",
        max_turns: int = 8,
        judge_votes: int = 3,
        judge_grounding: bool = False,
        **_kwargs,
    ):
        del pack_filename, report_filename
        del agent_runner_name, model, provider_profile, system_prompt, fixture_match
        return SimpleNamespace(
            report_only=report_only,
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "reports" / "eval_report.json",
                report=SimpleNamespace(
                    case_count=4,
                    passed_count=1,
                    failed_count=1,
                    blocked_count=1,
                    error_count=1,
                ),
            ),
        )

    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_report", fake_run_ohmo_eval_report)

    failed = runner.invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
        ],
    )
    report_only = runner.invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
            "--report-only",
        ],
    )

    assert failed.exit_code != 0
    assert "Eval run evaluated 4 cases: passed=1 failed=1 blocked=1 error=1" in failed.output
    assert report_only.exit_code == 0
    assert "Eval run evaluated 4 cases: passed=1 failed=1 blocked=1 error=1" in report_only.output
    assert "Report-only mode: failures did not fail the command" in report_only.output


def test_ohmo_evals_run_command_surfaces_unknown_executor_errors(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"

    def fake_run_ohmo_eval_report(**kwargs):
        raise ValueError("unknown eval executor: live-agent")

    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_report", fake_run_ohmo_eval_report)

    result = runner.invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
            "--executor",
            "live-agent",
        ],
    )

    assert result.exit_code == 1
    assert "unknown eval executor: live-agent" in result.stderr


def test_ohmo_evals_run_command_surfaces_unknown_runner_errors_from_check_config(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"

    def fake_check_ohmo_eval_run_config(**kwargs):
        raise ValueError("unknown eval agent runner: live-agent")

    def fail_run_ohmo_eval_report(**kwargs):
        raise AssertionError("eval run should not execute in --check-config mode")

    monkeypatch.setattr(
        "ohmo.cli.check_ohmo_eval_run_config",
        fake_check_ohmo_eval_run_config,
    )
    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_report", fail_run_ohmo_eval_report)

    result = runner.invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
            "--agent-runner",
            "live-agent",
            "--check-config",
        ],
    )

    assert result.exit_code == 1
    assert "unknown eval agent runner: live-agent" in result.stderr


def test_ohmo_evals_compare_command_fails_on_regressions_and_supports_report_only(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_compare_ohmo_eval_reports(
        *,
        workspace: str | Path | None = None,
        baseline_report: str | Path,
        candidate_report: str | Path = "eval_report.json",
        report_filename: str = "eval_compare.json",
        score_tolerance: float = 0.0,
        report_only: bool = False,
    ):
        calls.append(
            {
                "workspace": workspace,
                "baseline_report": baseline_report,
                "candidate_report": candidate_report,
                "report_filename": report_filename,
                "score_tolerance": score_tolerance,
                "report_only": report_only,
            }
        )
        return SimpleNamespace(
            report_only=report_only,
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "reports" / report_filename,
                report=SimpleNamespace(
                    case_count=2,
                    compared_count=2,
                    unchanged_count=0,
                    improvement_count=1,
                    regression_count=1,
                    added_count=0,
                    removed_count=0,
                    score_delta=-0.25,
                ),
            ),
        )

    monkeypatch.setattr("ohmo.cli.compare_ohmo_eval_reports", fake_compare_ohmo_eval_reports)

    failed = runner.invoke(
        app,
        [
            "evals",
            "compare",
            "--workspace",
            str(workspace),
            "--baseline",
            "baseline.json",
            "--candidate",
            "candidate.json",
            "--output",
            "compare.json",
            "--score-tolerance",
            "0.1",
        ],
    )
    report_only = runner.invoke(
        app,
        [
            "evals",
            "compare",
            "--workspace",
            str(workspace),
            "--baseline",
            "baseline.json",
            "--report-only",
        ],
    )

    assert failed.exit_code == 1
    assert "Wrote eval comparison report:" in failed.output
    assert "regressed=1" in failed.output
    assert report_only.exit_code == 0
    assert "Report-only mode: regressions did not fail the command" in report_only.output
    assert calls == [
        {
            "workspace": workspace.resolve(),
            "baseline_report": "baseline.json",
            "candidate_report": "candidate.json",
            "report_filename": "compare.json",
            "score_tolerance": 0.1,
            "report_only": False,
        },
        {
            "workspace": workspace.resolve(),
            "baseline_report": "baseline.json",
            "candidate_report": "eval_report.json",
            "report_filename": "eval_compare.json",
            "score_tolerance": 0.0,
            "report_only": True,
        },
    ]


def test_ohmo_evals_compare_command_outputs_json_summary(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"

    def fake_compare_ohmo_eval_reports(
        *,
        workspace: str | Path | None = None,
        baseline_report: str | Path,
        candidate_report: str | Path = "eval_report.json",
        report_filename: str = "eval_compare.json",
        score_tolerance: float = 0.0,
        report_only: bool = False,
    ):
        del baseline_report, candidate_report, score_tolerance
        return SimpleNamespace(
            report_only=report_only,
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "reports" / report_filename,
                relative_path=f"reports/{report_filename}",
                report=SimpleNamespace(
                    report_kind="execution_comparison_report",
                    report_id="compare-1",
                    baseline_report_id="baseline-1",
                    candidate_report_id="candidate-1",
                    baseline_pack_id="pack-old",
                    candidate_pack_id="pack-new",
                    case_count=2,
                    compared_count=2,
                    unchanged_count=1,
                    improvement_count=1,
                    regression_count=0,
                    added_count=0,
                    removed_count=0,
                    score_delta=0.25,
                    cases=[SimpleNamespace(raw_tool_text="SECRET TOOL")],
                ),
            ),
        )

    monkeypatch.setattr("ohmo.cli.compare_ohmo_eval_reports", fake_compare_ohmo_eval_reports)

    result = runner.invoke(
        app,
        [
            "evals",
            "compare",
            "--workspace",
            str(workspace),
            "--baseline",
            "baseline.json",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == {
        "privacy": "metadata_only",
        "report_path": str(workspace / "evals" / "reports" / "eval_compare.json"),
        "relative_path": "reports/eval_compare.json",
        "report_kind": "execution_comparison_report",
        "report_id": "compare-1",
        "baseline_report_id": "baseline-1",
        "candidate_report_id": "candidate-1",
        "baseline_pack_id": "pack-old",
        "candidate_pack_id": "pack-new",
        "case_count": 2,
        "compared_count": 2,
        "unchanged_count": 1,
        "improvement_count": 1,
        "regression_count": 0,
        "added_count": 0,
        "removed_count": 0,
        "score_delta": 0.25,
        "report_only": False,
    }
    assert "Compared eval reports:" not in result.output
    assert "SECRET TOOL" not in result.output


def test_ohmo_evals_baseline_save_command(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_save_ohmo_eval_baseline(
        *,
        workspace: str | Path | None = None,
        source_report: str | Path = "eval_report.json",
        name: str = "main",
        overwrite: bool = False,
    ):
        calls.append(
            {
                "workspace": workspace,
                "source_report": source_report,
                "name": name,
                "overwrite": overwrite,
            }
        )
        return SimpleNamespace(
            name="main",
            path=Path(workspace) / "evals" / "reports" / "baselines" / "main.json",
            relative_path="reports/baselines/main.json",
            report_id="report-1",
            case_count=3,
            passed_count=2,
            non_passed_count=1,
        )

    monkeypatch.setattr("ohmo.cli.save_ohmo_eval_baseline", fake_save_ohmo_eval_baseline)

    result = runner.invoke(
        app,
        [
            "evals",
            "baseline",
            "save",
            "--workspace",
            str(workspace),
            "--from-report",
            "candidate.json",
            "--name",
            "main",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "workspace": workspace.resolve(),
            "source_report": "candidate.json",
            "name": "main",
            "overwrite": True,
        }
    ]
    assert "Saved eval baseline:" in result.output
    assert "Baseline main: cases=3 passed=2 non_passed=1" in result.output
    assert "ohmo evals compare --baseline baselines/main.json" in result.output


def test_ohmo_evals_baseline_list_command(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"

    def fake_list_ohmo_eval_baselines(*, workspace: str | Path | None = None):
        return SimpleNamespace(
            baselines=[
                SimpleNamespace(
                    name="main",
                    case_count=3,
                    passed_count=2,
                    failed_count=1,
                    blocked_count=0,
                    error_count=0,
                    relative_path="reports/baselines/main.json",
                )
            ]
        )

    monkeypatch.setattr("ohmo.cli.list_ohmo_eval_baselines", fake_list_ohmo_eval_baselines)

    result = runner.invoke(
        app,
        [
            "evals",
            "baseline",
            "list",
            "--workspace",
            str(workspace),
        ],
    )

    assert result.exit_code == 0
    assert "Eval baselines:" in result.output
    assert "- main cases=3 passed=2 failed=1 blocked=0 error=0" in result.output
    assert "path=reports/baselines/main.json" in result.output


def test_ohmo_evals_baseline_list_command_outputs_json_summary(
    tmp_path: Path,
    monkeypatch,
):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"

    def fake_list_ohmo_eval_baselines(*, workspace: str | Path | None = None):
        return SimpleNamespace(
            baselines=[
                SimpleNamespace(
                    name="main",
                    path=Path(workspace) / "evals" / "reports" / "baselines" / "main.json",
                    relative_path="reports/baselines/main.json",
                    report_id="exec-1",
                    pack_id="pack-1",
                    case_count=3,
                    passed_count=2,
                    failed_count=1,
                    blocked_count=0,
                    error_count=0,
                    raw_prompt_text="SECRET PROMPT",
                )
            ]
        )

    monkeypatch.setattr("ohmo.cli.list_ohmo_eval_baselines", fake_list_ohmo_eval_baselines)

    result = runner.invoke(
        app,
        [
            "evals",
            "baseline",
            "list",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == {
        "privacy": "metadata_only",
        "baseline_count": 1,
        "baselines": [
            {
                "name": "main",
                "path": str(workspace / "evals" / "reports" / "baselines" / "main.json"),
                "relative_path": "reports/baselines/main.json",
                "report_id": "exec-1",
                "pack_id": "pack-1",
                "case_count": 3,
                "passed_count": 2,
                "failed_count": 1,
                "blocked_count": 0,
                "error_count": 0,
            }
        ],
    }
    assert "Eval baselines:" not in result.output
    assert "SECRET PROMPT" not in result.output


def test_ohmo_evals_baseline_list_command_empty(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"

    monkeypatch.setattr(
        "ohmo.cli.list_ohmo_eval_baselines",
        lambda *, workspace=None: SimpleNamespace(baselines=[]),
    )

    result = runner.invoke(
        app,
        [
            "evals",
            "baseline",
            "list",
            "--workspace",
            str(workspace),
        ],
    )

    assert result.exit_code == 0
    assert "No eval baselines saved." in result.output

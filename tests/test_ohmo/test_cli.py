import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from ohmo.cli import app


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
        dry_run: bool = False,
        reviewer: str = "",
    ):
        calls.append(
            {
                "workspace": workspace,
                "case_ids": case_ids,
                "promote_all": promote_all,
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


def test_ohmo_evals_pack_command_builds_runnable_pack(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_build_ohmo_eval_pack(*, workspace: str | Path | None = None):
        calls.append({"workspace": workspace})
        return SimpleNamespace(
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "packs" / "eval_pack.json",
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
    assert calls == [{"workspace": workspace.resolve()}]
    assert "Wrote runnable pack:" in result.output
    assert "Built runnable pack with 3 cases from 4 gold cases" in result.output


def test_ohmo_evals_smoke_command_runs_report_only_eval(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_run_ohmo_eval_smoke(
        *,
        workspace: str | Path | None = None,
        pack_filename: str = "eval_pack.json",
        limit: int | None = None,
        report_only: bool = False,
    ):
        calls.append(
            {
                "workspace": workspace,
                "pack_filename": pack_filename,
                "limit": limit,
                "report_only": report_only,
            }
        )
        return SimpleNamespace(
            report_only=report_only,
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "reports" / "smoke_report.json",
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
            "limit": 2,
            "report_only": False,
        },
        {
            "workspace": workspace.resolve(),
            "pack_filename": "custom_pack.json",
            "limit": 2,
            "report_only": True,
        },
    ]


def test_ohmo_evals_run_command_runs_eval_report(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"
    calls: list[dict[str, object]] = []

    def fake_run_ohmo_eval_report(
        *,
        workspace: str | Path | None = None,
        pack_filename: str = "eval_pack.json",
        limit: int | None = None,
        report_only: bool = False,
        executor_name: str = "replay-tools",
        agent_runner_name: str = "scripted",
        model: str | None = None,
        provider_profile: str | None = None,
        system_prompt: str = "You are running an Ohmo replay-only eval.",
    ):
        calls.append(
            {
                "workspace": workspace,
                "pack_filename": pack_filename,
                "limit": limit,
                "report_only": report_only,
                "executor_name": executor_name,
                "agent_runner_name": agent_runner_name,
                "model": model,
                "provider_profile": provider_profile,
                "system_prompt": system_prompt,
            }
        )
        return SimpleNamespace(
            report_only=report_only,
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "reports" / "eval_report.json",
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
            "--executor",
            "replay-tools",
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
            "limit": 2,
            "report_only": False,
            "executor_name": "replay-tools",
            "agent_runner_name": "scripted",
            "model": None,
            "provider_profile": None,
            "system_prompt": "You are running an Ohmo replay-only eval.",
        },
        {
            "workspace": workspace.resolve(),
            "pack_filename": "custom_pack.json",
            "limit": 2,
            "report_only": True,
            "executor_name": "replay-tools",
            "agent_runner_name": "scripted",
            "model": None,
            "provider_profile": None,
            "system_prompt": "You are running an Ohmo replay-only eval.",
        },
    ]


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
        limit: int | None = None,
        report_only: bool = False,
        executor_name: str = "replay-tools",
        agent_runner_name: str = "scripted",
        model: str | None = None,
        provider_profile: str | None = None,
        system_prompt: str = "You are running an Ohmo replay-only eval.",
    ):
        calls.append(
            {
                "workspace": workspace,
                "pack_filename": pack_filename,
                "limit": limit,
                "report_only": report_only,
                "executor_name": executor_name,
                "agent_runner_name": agent_runner_name,
                "model": model,
                "provider_profile": provider_profile,
                "system_prompt": system_prompt,
            }
        )
        return SimpleNamespace(
            report_only=report_only,
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "reports" / "eval_report.json",
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
            "limit": None,
            "report_only": False,
            "executor_name": "replay-tools",
            "agent_runner_name": "query-engine",
            "model": "eval-model",
            "provider_profile": "openai-compatible",
            "system_prompt": "eval system",
        }
    ]


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
        limit: int | None = None,
        report_only: bool = False,
        executor_name: str = "replay-tools",
        agent_runner_name: str = "scripted",
        model: str | None = None,
        provider_profile: str | None = None,
        system_prompt: str = "You are running an Ohmo replay-only eval.",
    ):
        del agent_runner_name, model, provider_profile, system_prompt
        return SimpleNamespace(
            report_only=report_only,
            write=SimpleNamespace(
                path=Path(workspace) / "evals" / "reports" / "eval_report.json",
                report=SimpleNamespace(
                    case_count=3,
                    passed_count=1,
                    failed_count=0,
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

    assert failed.exit_code == 1
    assert "Eval run evaluated 3 cases: passed=1 failed=0 blocked=1 error=1" in failed.output
    assert report_only.exit_code == 0
    assert "Report-only mode: failures did not fail the command" in report_only.output


def test_ohmo_evals_run_command_rejects_unknown_executor(tmp_path: Path):
    runner = CliRunner()
    workspace = tmp_path / ".ohmo-home"

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

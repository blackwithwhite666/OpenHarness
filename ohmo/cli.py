"""CLI entry point for the ohmo personal-agent app."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import typer

from openharness.auth.manager import AuthManager
from openharness.config import load_settings

from ohmo.gateway.config import load_gateway_config, save_gateway_config
from ohmo.gateway.models import GatewayConfig
from ohmo.gateway.service import (
    OhmoGatewayService,
    gateway_status,
    start_gateway_process,
    stop_gateway_process,
)
from ohmo.evals import (
    build_ohmo_eval_pack,
    check_ohmo_eval_run_config,
    compare_ohmo_eval_reports,
    list_ohmo_eval_baselines,
    promote_ohmo_eval_case_drafts,
    review_ohmo_eval_case_drafts,
    run_ohmo_eval_report,
    run_ohmo_eval_smoke,
    save_ohmo_eval_baseline,
    validate_ohmo_eval_review_manifest,
    write_ohmo_embedding_index,
    write_ohmo_eval_mine,
    write_ohmo_eval_review_manifest,
)
from ohmo.memory import add_memory_entry, list_memory_files, remove_memory_entry
from ohmo.runtime import launch_ohmo_react_tui, run_ohmo_backend, run_ohmo_print_mode
from ohmo.session_storage import OhmoSessionBackend
from ohmo.workspace import (
    get_gateway_config_path,
    get_workspace_root,
    get_soul_path,
    get_state_path,
    get_user_path,
    initialize_workspace,
    workspace_health,
)


app = typer.Typer(
    name="ohmo",
    help="ohmo: a personal-agent app built on top of OpenHarness.",
    invoke_without_command=True,
    add_completion=False,
)
memory_app = typer.Typer(name="memory", help="Manage .ohmo memory")
soul_app = typer.Typer(name="soul", help="Inspect or edit soul.md")
user_app = typer.Typer(name="user", help="Inspect or edit user.md")
gateway_app = typer.Typer(name="gateway", help="Run the ohmo gateway")
evals_app = typer.Typer(name="evals", help="Build ohmo eval/data-flywheel artifacts")
evals_baseline_app = typer.Typer(name="baseline", help="Manage ohmo eval baselines")

app.add_typer(memory_app)
app.add_typer(soul_app)
app.add_typer(user_app)
app.add_typer(gateway_app)
app.add_typer(evals_app)
evals_app.add_typer(evals_baseline_app)

_INTERACTIVE_CHANNELS = ("telegram", "slack", "discord", "feishu")
_WORKSPACE_HELP = "Path to the ohmo workspace (defaults to ~/.ohmo)"


def _can_use_questionary() -> bool:
    """Return True when a real interactive terminal is available."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    if sys.stdin is not sys.__stdin__ or sys.stdout is not sys.__stdout__:
        return False
    try:
        import questionary  # noqa: F401
    except ImportError:
        return False
    return True


def _select_with_questionary(
    title: str,
    options: list[tuple[str, str]],
    *,
    default_value: str | None = None,
) -> str:
    import questionary

    choices = [
        questionary.Choice(
            title=label,
            value=value,
            checked=(value == default_value),
        )
        for value, label in options
    ]
    result = questionary.select(title, choices=choices, default=default_value).ask()
    if result is None:
        raise typer.Abort()
    return str(result)


def _confirm_prompt(message: str, *, default: bool = False) -> bool:
    """Ask for confirmation, preferring questionary in a real TTY."""
    if _can_use_questionary():
        import questionary

        result = questionary.confirm(message, default=default).ask()
        if result is None:
            raise typer.Abort()
        return bool(result)
    return typer.confirm(message, default=default)


def _text_prompt(message: str, *, default: str = "") -> str:
    """Prompt for text input, preferring questionary in a real TTY."""
    if _can_use_questionary():
        import questionary

        result = questionary.text(message, default=default).ask()
        if result is None:
            raise typer.Abort()
        return str(result)
    return typer.prompt(message, default=default)


def _select_from_menu(
    title: str,
    options: list[tuple[str, str]],
    *,
    default_value: str | None = None,
) -> str:
    """Render a simple numbered picker and return the selected value."""
    if _can_use_questionary():
        return _select_with_questionary(title, options, default_value=default_value)
    print(title)
    default_index = 1
    for index, (value, label) in enumerate(options, 1):
        marker = " (default)" if value == default_value else ""
        if value == default_value:
            default_index = index
        print(f"  {index}. {label}{marker}")
    raw = typer.prompt("Choose", default=str(default_index))
    try:
        selected = options[int(raw) - 1]
    except (ValueError, IndexError):
        raise typer.BadParameter(f"Invalid selection: {raw}") from None
    return selected[0]


def _format_provider_profile_label(info: dict[str, object]) -> str:
    label = str(info["label"])
    if bool(info["configured"]):
        return label
    return f"{label} (missing)"


def _prompt_provider_profile(workspace: str | Path) -> str:
    settings = load_settings()
    statuses = AuthManager(settings).get_profile_statuses()
    default_value = load_gateway_config(workspace).provider_profile
    hints = {
        "claude-api": ("Claude / Kimi / GLM / MiniMax", "fg:#7aa2f7"),
        "openai-compatible": ("OpenAI / OpenRouter", "fg:#9ece6a"),
    }

    if _can_use_questionary():
        import questionary

        choices = []
        for name, info in statuses.items():
            label = str(info["label"])
            missing = "" if bool(info["configured"]) else " (missing)"
            hint = hints.get(name)
            if hint is None:
                title = label if not missing else [("", label), ("fg:#d3869b", missing)]
            else:
                hint_text, hint_style = hint
                title = [
                    ("", f"{label}  "),
                    (hint_style, hint_text),
                ]
                if missing:
                    title.extend([("", "  "), ("fg:#d3869b", missing.strip())])
            choices.append(questionary.Choice(title=title, value=name, checked=(name == default_value)))
        result = questionary.select("Choose provider profile for ohmo:", choices=choices, default=default_value).ask()
        if result is None:
            raise typer.Abort()
        return str(result)

    options = []
    for name, info in statuses.items():
        label = _format_provider_profile_label(info)
        hint = hints.get(name)
        if hint is not None:
            label = f"{label} ({hint[0]})"
        options.append((name, label))
    return _select_from_menu(
        "Choose provider profile for ohmo:",
        options,
        default_value=default_value,
    )


def _prompt_channels(existing: GatewayConfig) -> tuple[list[str], dict[str, dict]]:
    enabled: list[str] = []
    configs: dict[str, dict] = {}
    print("Configure channels for ohmo gateway:")
    for channel in _INTERACTIVE_CHANNELS:
        current = channel in existing.enabled_channels
        prior = dict(existing.channel_configs.get(channel, {}))
        if current:
            enabled.append(channel)
            if not _confirm_prompt(f"Reconfigure {channel}?", default=False):
                configs[channel] = prior
                continue
        elif not _confirm_prompt(f"Enable {channel}?", default=False):
            continue
        else:
            enabled.append(channel)
        allow_from_raw = _text_prompt(
            f"{channel} allow_from (comma separated user/chat IDs; leave blank to deny all; '*' for everyone)",
            default=",".join(prior.get("allow_from", [])),
        )
        allow_from = [item.strip() for item in allow_from_raw.split(",") if item.strip()]
        config: dict[str, object] = {"allow_from": allow_from}
        if channel == "telegram":
            config["token"] = _text_prompt(
                "Telegram bot token",
                default=str(prior.get("token", "")),
            )
            config["reply_to_message"] = _confirm_prompt(
                "Reply to the original Telegram message?",
                default=bool(prior.get("reply_to_message", True)),
            )
        elif channel == "slack":
            config["bot_token"] = _text_prompt(
                "Slack bot token",
                default=str(prior.get("bot_token", "")),
            )
            config["app_token"] = _text_prompt(
                "Slack app token",
                default=str(prior.get("app_token", "")),
            )
            config["mode"] = "socket"
            config["reply_in_thread"] = _confirm_prompt(
                "Reply in thread?",
                default=bool(prior.get("reply_in_thread", True)),
            )
            config["group_policy"] = _select_from_menu(
                "Slack group policy:",
                [
                    ("mention", "Mention only"),
                    ("open", "Always reply in channels"),
                    ("allowlist", "Only allow configured channels"),
                ],
                default_value=str(prior.get("group_policy", "mention")),
            )
        elif channel == "discord":
            config["token"] = _text_prompt(
                "Discord bot token",
                default=str(prior.get("token", "")),
            )
            config["gateway_url"] = _text_prompt(
                "Discord gateway URL",
                default=str(prior.get("gateway_url", "wss://gateway.discord.gg/?v=10&encoding=json")),
            )
            config["intents"] = int(
                _text_prompt(
                    "Discord intents bitmask",
                    default=str(prior.get("intents", 513)),
                )
            )
            config["group_policy"] = _select_from_menu(
                "Discord group policy:",
                [
                    ("mention", "Mention only"),
                    ("open", "Always reply in channels"),
                ],
                default_value=str(prior.get("group_policy", "mention")),
            )
        elif channel == "feishu":
            config["app_id"] = _text_prompt(
                "Feishu app id",
                default=str(prior.get("app_id", "")),
            )
            config["app_secret"] = _text_prompt(
                "Feishu app secret",
                default=str(prior.get("app_secret", "")),
            )
            config["encrypt_key"] = _text_prompt(
                "Feishu encrypt key",
                default=str(prior.get("encrypt_key", "")),
            )
            config["verification_token"] = _text_prompt(
                "Feishu verification token",
                default=str(prior.get("verification_token", "")),
            )
            config["react_emoji"] = _text_prompt(
                "Feishu reaction emoji",
                default=str(prior.get("react_emoji", "OK")),
            )
            config["group_policy"] = _select_from_menu(
                "Feishu group policy:",
                [
                    ("managed_or_mention", "Managed groups open; other groups require @mention"),
                    ("mention", "Always require @mention in groups"),
                    ("open", "Always reply to group messages"),
                ],
                default_value=str(prior.get("group_policy", "managed_or_mention")),
            )
            prior_bot_names = prior.get("bot_names", ["ohmo", "openclaw", "openharness"])
            if isinstance(prior_bot_names, str):
                prior_bot_names_default = prior_bot_names
            else:
                prior_bot_names_default = ",".join(str(item) for item in prior_bot_names)
            bot_names_raw = _text_prompt(
                "Feishu bot mention names (comma separated)",
                default=prior_bot_names_default,
            )
            config["bot_names"] = [item.strip() for item in bot_names_raw.split(",") if item.strip()]
            config["bot_open_id"] = _text_prompt(
                "Feishu bot open_id for exact mention detection (optional)",
                default=str(prior.get("bot_open_id", "")),
            )
        configs[channel] = config
    return enabled, configs


def _run_gateway_config_wizard(workspace: str | Path) -> GatewayConfig:
    """Interactive flow for provider/channel setup."""
    existing = load_gateway_config(workspace)
    provider_profile = _prompt_provider_profile(workspace)
    enabled_channels, channel_configs = _prompt_channels(existing)
    send_progress = _confirm_prompt(
        "Send progress updates to channels?",
        default=existing.send_progress,
    )
    send_tool_hints = _confirm_prompt(
        "Send tool hints to channels?",
        default=existing.send_tool_hints,
    )
    allow_remote_admin_commands = _confirm_prompt(
        "Allow explicitly listed administrative slash commands from remote channels?",
        default=existing.allow_remote_admin_commands,
    )
    default_allowlist = ", ".join(existing.allowed_remote_admin_commands)
    allowed_remote_admin_commands: list[str] = []
    if allow_remote_admin_commands:
        allowlist_raw = _text_prompt(
            "Allowed remote admin commands (comma-separated, e.g. permissions, plan)",
            default=default_allowlist,
        )
        allowed_remote_admin_commands = [
            item.strip().lstrip("/")
            for item in allowlist_raw.split(",")
            if item.strip()
        ]
    config = existing.model_copy(
        update={
            "provider_profile": provider_profile,
            "enabled_channels": enabled_channels,
            "channel_configs": channel_configs,
            "send_progress": send_progress,
            "send_tool_hints": send_tool_hints,
            "allow_remote_admin_commands": allow_remote_admin_commands,
            "allowed_remote_admin_commands": allowed_remote_admin_commands,
        }
    )
    save_gateway_config(config, workspace)
    return config


def _print_gateway_config_summary(config: GatewayConfig) -> None:
    if config.enabled_channels:
        print(
            "Configured channels: "
            + ", ".join(config.enabled_channels)
            + f" | provider_profile={config.provider_profile}"
        )
        deny_all_channels = [
            name for name in config.enabled_channels
            if not list(config.channel_configs.get(name, {}).get("allow_from", []))
        ]
        if deny_all_channels:
            print(
                "Remote access denied until allow_from is configured for: "
                + ", ".join(deny_all_channels)
            )
    else:
        print(f"Configured provider_profile={config.provider_profile}; no channels enabled yet.")
    if config.allow_remote_admin_commands and config.allowed_remote_admin_commands:
        print(
            "Remote admin opt-in enabled for: "
            + ", ".join(f"/{name}" for name in config.allowed_remote_admin_commands)
        )
    else:
        print("Remote admin commands remain local-only.")


def _maybe_restart_gateway(*, cwd: str | Path, workspace: str | Path) -> None:
    state = gateway_status(cwd, workspace)
    if not state.running:
        return
    if not _confirm_prompt("Gateway is running. Restart now to apply changes?", default=True):
        print("Configuration saved. Restart later with `ohmo gateway restart`.")
        return
    stop_gateway_process(cwd, workspace)
    pid = start_gateway_process(cwd, workspace)
    print(f"ohmo gateway restarted (pid={pid})")


def _configure_gateway_logging(workspace: str | Path | None = None) -> None:
    """Configure foreground gateway logging."""
    config = load_gateway_config(workspace)
    level_name = str(config.log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        force=True,
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    print_mode: str | None = typer.Option(None, "--print", "-p", help="Run a single prompt and exit"),
    model: str | None = typer.Option(None, "--model", help="Model override for this session"),
    profile: str | None = typer.Option(None, "--profile", help="Provider profile to use"),
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
    max_turns: int | None = typer.Option(None, "--max-turns", help="Override max turns"),
    cwd: str = typer.Option(str(Path.cwd()), "--cwd", help="Working directory"),
    backend_only: bool = typer.Option(False, "--backend-only", hidden=True),
    resume: str | None = typer.Option(None, "--resume", help="Resume an ohmo session by id"),
    continue_session: bool = typer.Option(False, "--continue", help="Continue the latest ohmo session"),
) -> None:
    """Launch the ohmo app or invoke a subcommand."""
    if ctx.invoked_subcommand is not None:
        return

    cwd_path = str(Path(cwd).resolve())
    workspace_root = initialize_workspace(workspace)
    backend = OhmoSessionBackend(workspace_root)
    restore_messages = None
    restore_tool_metadata = None
    if continue_session:
        latest = backend.load_latest(cwd_path)
        if latest is None:
            print("No previous ohmo session found in this directory.", file=sys.stderr)
            raise typer.Exit(1)
        restore_messages = latest.get("messages")
        restore_tool_metadata = latest.get("tool_metadata")
    elif resume:
        snapshot = backend.load_by_id(cwd_path, resume)
        if snapshot is None:
            print(f"ohmo session not found: {resume}", file=sys.stderr)
            raise typer.Exit(1)
        restore_messages = snapshot.get("messages")
        restore_tool_metadata = snapshot.get("tool_metadata")

    if backend_only:
        raise SystemExit(
            asyncio.run(
                run_ohmo_backend(
                    cwd=cwd_path,
                    workspace=workspace_root,
                    model=model,
                    max_turns=max_turns,
                    provider_profile=profile,
                    restore_messages=restore_messages,
                    restore_tool_metadata=restore_tool_metadata,
                )
            )
        )

    if print_mode is not None:
        raise SystemExit(
            asyncio.run(
                run_ohmo_print_mode(
                    prompt=print_mode,
                    cwd=cwd_path,
                    workspace=workspace_root,
                    model=model,
                    max_turns=max_turns,
                    provider_profile=profile,
                )
            )
        )

    raise SystemExit(
        asyncio.run(
            launch_ohmo_react_tui(
                cwd=cwd_path,
                workspace=workspace_root,
                model=model,
                max_turns=max_turns,
                provider_profile=profile,
            )
        )
    )


@app.command("init")
def init_cmd(
    cwd: str = typer.Option(str(Path.cwd()), "--cwd", help="Project working directory (reserved for future project overrides)"),
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        help="Run the provider/channel setup wizard when attached to a terminal",
    ),
) -> None:
    """Initialize the .ohmo workspace."""
    root_path = get_workspace_root(workspace)
    already_exists = root_path.exists()
    root = initialize_workspace(root_path)
    print(f"Initialized ohmo workspace at {root}")
    if already_exists:
        print("ohmo workspace already exists.")
        if not interactive:
            print("Use `ohmo config` to update provider and channel settings.")
            return
        if not _confirm_prompt("Open configuration now?", default=True):
            print("Use `ohmo config` when you want to change provider or channel settings.")
            return
    if interactive:
        config = _run_gateway_config_wizard(root)
        _print_gateway_config_summary(config)
        print(f"Saved gateway config to {get_gateway_config_path(root)}")


@app.command("config")
def config_cmd(
    cwd: str = typer.Option(str(Path.cwd()), "--cwd", help="Project working directory"),
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
) -> None:
    """Configure provider profile and gateway channels."""
    cwd_path = str(Path(cwd).resolve())
    workspace_root = initialize_workspace(workspace)
    config = _run_gateway_config_wizard(workspace_root)
    _print_gateway_config_summary(config)
    print(f"Saved gateway config to {get_gateway_config_path(workspace_root)}")
    _maybe_restart_gateway(cwd=cwd_path, workspace=workspace_root)


@app.command("doctor")
def doctor_cmd(
    cwd: str = typer.Option(str(Path.cwd()), "--cwd", help="Project working directory"),
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
) -> None:
    """Check .ohmo workspace and provider readiness."""
    cwd_path = str(Path(cwd).resolve())
    workspace_root = initialize_workspace(workspace)
    health = workspace_health(workspace_root)
    settings = load_settings()
    statuses = AuthManager(settings).get_profile_statuses()
    lines = ["ohmo doctor:"]
    for name, ok in health.items():
        lines.append(f"- {name}: {'ok' if ok else 'missing'}")
    lines.append(f"- project_cwd: {cwd_path}")
    lines.append(f"- workspace_root: {workspace_root}")
    lines.append(f"- workspace_state: {get_state_path(workspace_root)}")
    lines.append(f"- gateway_config: {get_gateway_config_path(workspace_root)}")
    lines.append("- available_profiles:")
    for name, info in statuses.items():
        lines.append(
            f"  - {name}: {info['label']} ({'configured' if info['configured'] else 'missing auth'})"
        )
    print("\n".join(lines))


@memory_app.command("list")
def memory_list_cmd(workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP)) -> None:
    for path in list_memory_files(workspace):
        print(path.name)


@memory_app.command("add")
def memory_add_cmd(
    title: str = typer.Argument(...),
    content: str = typer.Argument(...),
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
) -> None:
    try:
        path = add_memory_entry(workspace, title, content)
    except ValueError as exc:  # safety-scan / size refusal
        print(str(exc), file=sys.stderr)
        raise typer.Exit(1)
    print(f"Added memory entry {path.name}")


@memory_app.command("remove")
def memory_remove_cmd(
    name: str = typer.Argument(...),
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
) -> None:
    if remove_memory_entry(workspace, name):
        print(f"Removed memory entry {name}")
        return
    print(f"Memory entry not found: {name}", file=sys.stderr)
    raise typer.Exit(1)


def _show_or_edit(path: Path, set_text: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if set_text is not None:
        path.write_text(set_text.strip() + "\n", encoding="utf-8")
        print(f"Updated {path}")
        return
    if not path.exists():
        print(f"{path} does not exist yet.", file=sys.stderr)
        raise typer.Exit(1)
    print(path.read_text(encoding="utf-8"))


@soul_app.command("show")
def soul_show_cmd(workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP)) -> None:
    _show_or_edit(get_soul_path(workspace), None)


@soul_app.command("edit")
def soul_edit_cmd(
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
    set_text: str | None = typer.Option(None, "--set", help="Replace soul.md with this text"),
) -> None:
    _show_or_edit(get_soul_path(workspace), set_text)


@user_app.command("show")
def user_show_cmd(workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP)) -> None:
    _show_or_edit(get_user_path(workspace), None)


@user_app.command("edit")
def user_edit_cmd(
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
    set_text: str | None = typer.Option(None, "--set", help="Replace user.md with this text"),
) -> None:
    _show_or_edit(get_user_path(workspace), set_text)


@gateway_app.command("run")
def gateway_run_cmd(
    cwd: str = typer.Option(str(Path.cwd()), "--cwd", help="Project working directory"),
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
) -> None:
    """Run the ohmo gateway in the foreground."""
    _configure_gateway_logging(workspace)
    service = OhmoGatewayService(cwd, workspace)
    raise SystemExit(asyncio.run(service.run_foreground()))


@gateway_app.command("start")
def gateway_start_cmd(
    cwd: str = typer.Option(str(Path.cwd()), "--cwd", help="Project working directory"),
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
) -> None:
    pid = start_gateway_process(cwd, workspace)
    print(f"ohmo gateway started (pid={pid})")


@gateway_app.command("stop")
def gateway_stop_cmd(
    cwd: str = typer.Option(str(Path.cwd()), "--cwd", help="Project working directory"),
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
) -> None:
    if stop_gateway_process(cwd, workspace):
        print("ohmo gateway stopped.")
        return
    print("ohmo gateway is not running.")


@gateway_app.command("restart")
def gateway_restart_cmd(
    cwd: str = typer.Option(str(Path.cwd()), "--cwd", help="Project working directory"),
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
) -> None:
    stop_gateway_process(cwd, workspace)
    pid = start_gateway_process(cwd, workspace)
    print(f"ohmo gateway restarted (pid={pid})")


@gateway_app.command("status")
def gateway_status_cmd(
    cwd: str = typer.Option(str(Path.cwd()), "--cwd", help="Project working directory"),
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
) -> None:
    state = gateway_status(cwd, workspace)
    print(state.model_dump_json(indent=2))


@evals_app.command("embed")
def evals_embed_cmd(
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
    batch_size: int = typer.Option(32, "--batch-size", min=1, help="Texts per inference batch"),
    inference_url: str | None = typer.Option(
        None,
        "--inference-url",
        help="Override INFERENCE_URL for the telegent-style inference service",
    ),
) -> None:
    """Build a dense embedding index for captured ohmo eval episodes."""
    workspace_root = initialize_workspace(workspace)
    result = asyncio.run(
        write_ohmo_embedding_index(
            workspace=workspace_root,
            inference_url=inference_url,
            batch_size=batch_size,
        )
    )
    print(f"Wrote embedding manifest: {result.manifest_path}")
    print(f"Wrote embedding records: {result.records_path}")
    print(
        "Indexed "
        f"{result.manifest.embedding_count}/{result.manifest.facet_count} facets "
        f"with dimensions={result.manifest.dimensions}"
    )


@evals_app.command("mine")
def evals_mine_cmd(
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
) -> None:
    """Mine metadata-only eval candidates and draft cases."""
    workspace_root = initialize_workspace(workspace)
    result = write_ohmo_eval_mine(workspace=workspace_root)
    print(f"Wrote candidate manifest: {result.candidates.manifest_path}")
    print(f"Wrote candidate records: {result.candidates.records_path}")
    print(f"Wrote case manifest: {result.cases.manifest_path}")
    print(f"Wrote case records: {result.cases.records_path}")
    print(
        "Mined "
        f"{result.candidates.manifest.record_count} candidates and "
        f"{result.cases.manifest.record_count} draft cases"
    )


@evals_app.command("review")
def evals_review_cmd(
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
    case_id: str | None = typer.Option(None, "--case-id", help="Show one draft case"),
    limit: int = typer.Option(20, "--limit", min=1, help="Maximum draft cases to show"),
    manifest_filename: str | None = typer.Option(
        None,
        "--manifest",
        help="Write a metadata-only review manifest under evals/cases",
    ),
    validate_manifest_filename: str | None = typer.Option(
        None,
        "--validate-manifest",
        help="Validate a review manifest under evals/cases without writing files",
    ),
) -> None:
    """Review metadata-only draft eval cases."""
    workspace_root = initialize_workspace(workspace)
    if validate_manifest_filename:
        try:
            validation = validate_ohmo_eval_review_manifest(
                workspace=workspace_root,
                filename=validate_manifest_filename,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            raise typer.Exit(1)
        print(f"Review manifest is valid: {validation.path}")
        print(
            "Review decisions: "
            f"total={validation.total_count} "
            f"approved={validation.approved_count} "
            f"rejected={validation.rejected_count} "
            f"pending={validation.pending_count}"
        )
        if validation.approved_case_ids:
            print("Approved cases:")
            for approved_case_id in validation.approved_case_ids:
                print(f"- {approved_case_id}")
        print(
            "Promote approved with: "
            f"ohmo evals promote --manifest {validate_manifest_filename}"
        )
        return

    try:
        result = review_ohmo_eval_case_drafts(
            workspace=workspace_root,
            case_id=case_id,
            limit=limit,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise typer.Exit(1)

    manifest_write = None
    if manifest_filename:
        try:
            manifest_write = write_ohmo_eval_review_manifest(
                workspace=workspace_root,
                case_id=case_id,
                limit=limit,
                filename=manifest_filename,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            raise typer.Exit(1)

    if case_id:
        if not result.shown:
            print(f"Draft eval case not found: {case_id}", file=sys.stderr)
            raise typer.Exit(1)
        item = result.shown[0]
        print(f"Draft eval case: {item.case_id}")
        print(f"- kind: {item.case_kind}")
        print(f"- episode: {item.episode_id}")
        print(f"- status: {item.review_status}")
        print(f"- input_facets: {item.input_facet_count}")
        print(f"- expected_facets: {item.expected_facet_count}")
        print(f"- tools: {', '.join(item.tool_names) if item.tool_names else '-'}")
        if manifest_write is not None:
            print(f"Wrote review manifest: {manifest_write.path}")
        print(f"Promote with: ohmo evals promote --case-id {item.case_id}")
        return

    print("Draft eval cases:")
    for item in result.shown:
        tools = ",".join(item.tool_names) if item.tool_names else "-"
        print(
            f"- {item.case_id} {item.case_kind} "
            f"episode={item.episode_id} "
            f"facets={item.input_facet_count}/{item.expected_facet_count} "
            f"tools={tools}"
        )
    print(f"Showing {len(result.shown)}/{result.total_count} draft cases.")
    if manifest_write is not None:
        print(f"Wrote review manifest: {manifest_write.path}")
    print("Promote with: ohmo evals promote --case-id <id>")


@evals_app.command("promote")
def evals_promote_cmd(
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
    case_ids: list[str] = typer.Option(
        [],
        "--case-id",
        help="Draft case id to promote; repeat for multiple cases",
    ),
    promote_all: bool = typer.Option(False, "--all", help="Promote all draft cases"),
    manifest_filename: str | None = typer.Option(
        None,
        "--manifest",
        help="Promote cases marked approved in a review manifest under evals/cases",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show selected cases without writing"),
    reviewer: str = typer.Option("", "--reviewer", help="Reviewer id to store in gold cases"),
) -> None:
    """Promote reviewed draft eval cases into the gold pack."""
    workspace_root = initialize_workspace(workspace)
    try:
        result = promote_ohmo_eval_case_drafts(
            workspace=workspace_root,
            case_ids=case_ids or None,
            promote_all=promote_all,
            manifest_filename=manifest_filename,
            dry_run=dry_run,
            reviewer=reviewer,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise typer.Exit(1)

    if result.dry_run:
        print(f"Would promote {result.promoted_count} draft cases:")
        for selected_case_id in result.selected_case_ids:
            print(f"- {selected_case_id}")
        print("No files written.")
        return

    print(f"Promoted {result.promoted_count} draft cases.")
    print(f"Wrote gold manifest: {result.manifest_path}")
    print(f"Wrote gold records: {result.records_path}")
    print(f"Remaining unpromoted draft cases: {result.remaining_unpromoted_count}")


@evals_app.command("pack")
def evals_pack_cmd(
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
) -> None:
    """Build a runnable eval pack from reviewed gold cases."""
    workspace_root = initialize_workspace(workspace)
    try:
        result = build_ohmo_eval_pack(workspace=workspace_root)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise typer.Exit(1)

    print(f"Wrote runnable pack: {result.write.path}")
    print(
        "Built runnable pack with "
        f"{result.case_count} cases from {result.gold_case_count} gold cases"
    )


@evals_app.command("smoke")
def evals_smoke_cmd(
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
    pack_filename: str = typer.Option(
        "eval_pack.json",
        "--pack",
        help="Runnable pack filename under evals/packs",
    ),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Smoke subset size"),
    report_only: bool = typer.Option(
        False,
        "--report-only",
        help="Exit 0 after writing the report even when smoke checks fail",
    ),
) -> None:
    """Run metadata-only smoke checks over a runnable eval pack."""
    workspace_root = initialize_workspace(workspace)
    try:
        result = run_ohmo_eval_smoke(
            workspace=workspace_root,
            pack_filename=pack_filename,
            limit=limit,
            report_only=report_only,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise typer.Exit(1)

    report = result.write.report
    print(f"Wrote smoke report: {result.write.path}")
    print(
        "Smoke evaluated "
        f"{report.case_count} cases: "
        f"passed={report.passed_count} failed={report.failed_count}"
    )
    if result.report_only:
        print("Report-only mode: failures did not fail the command")
        return
    if report.failed_count:
        raise typer.Exit(1)


@evals_app.command("run")
def evals_run_cmd(
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
    pack_filename: str = typer.Option(
        "eval_pack.json",
        "--pack",
        help="Runnable pack filename under evals/packs",
    ),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Eval subset size"),
    executor_name: str = typer.Option(
        "replay-tools",
        "--executor",
        help="Eval executor to use: replay-tools",
    ),
    agent_runner_name: str = typer.Option(
        "scripted",
        "--agent-runner",
        help="Agent runner to use inside the executor: scripted or query-engine",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Model override for --agent-runner query-engine",
    ),
    provider_profile: str | None = typer.Option(
        None,
        "--profile",
        help="Provider profile override for --agent-runner query-engine",
    ),
    system_prompt: str = typer.Option(
        "You are running an Ohmo replay-only eval.",
        "--system-prompt",
        help="System prompt for --agent-runner query-engine",
    ),
    report_only: bool = typer.Option(
        False,
        "--report-only",
        help="Exit 0 after writing the report even when eval cases fail",
    ),
    check_config: bool = typer.Option(
        False,
        "--check-config",
        help="Validate pack/auth/executor setup without executing eval cases",
    ),
) -> None:
    """Run deterministic replay-tools eval checks over a runnable eval pack."""
    workspace_root = initialize_workspace(workspace)
    try:
        if check_config:
            check = check_ohmo_eval_run_config(
                workspace=workspace_root,
                pack_filename=pack_filename,
                limit=limit,
                executor_name=executor_name,
                agent_runner_name=agent_runner_name,
                model=model,
                provider_profile=provider_profile,
                system_prompt=system_prompt,
            )
            print("Eval run configuration is valid.")
            print(
                "Pack "
                f"{check.pack_id}: "
                f"selected={check.selected_case_count}/{check.pack_case_count} "
                f"executor={check.executor_name} "
                f"agent_runner={check.agent_runner_name} "
                f"replay_tools_only={str(check.replay_tools_only).lower()}"
            )
            if check.model:
                print(
                    "Query engine "
                    f"profile={check.provider_profile or '-'} model={check.model}"
                )
            return
        result = run_ohmo_eval_report(
            workspace=workspace_root,
            pack_filename=pack_filename,
            limit=limit,
            report_only=report_only,
            executor_name=executor_name,
            agent_runner_name=agent_runner_name,
            model=model,
            provider_profile=provider_profile,
            system_prompt=system_prompt,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise typer.Exit(1)

    report = result.write.report
    print(f"Wrote eval report: {result.write.path}")
    print(
        "Eval run evaluated "
        f"{report.case_count} cases: "
        f"passed={report.passed_count} failed={report.failed_count} "
        f"blocked={getattr(report, 'blocked_count', 0)} "
        f"error={getattr(report, 'error_count', 0)}"
    )
    if result.report_only:
        print("Report-only mode: failures did not fail the command")
        return
    if (
        report.failed_count
        + getattr(report, "blocked_count", 0)
        + getattr(report, "error_count", 0)
    ):
        raise typer.Exit(1)


@evals_app.command("compare")
def evals_compare_cmd(
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
    baseline_report: str = typer.Option(
        ...,
        "--baseline",
        help="Baseline execution report filename under evals/reports, or an absolute path",
    ),
    candidate_report: str = typer.Option(
        "eval_report.json",
        "--candidate",
        help="Candidate execution report filename under evals/reports, or an absolute path",
    ),
    report_filename: str = typer.Option(
        "eval_compare.json",
        "--output",
        help="Comparison report filename under evals/reports",
    ),
    score_tolerance: float = typer.Option(
        0.0,
        "--score-tolerance",
        min=0.0,
        help="Allowed per-case score drop before same-status cases regress",
    ),
    report_only: bool = typer.Option(
        False,
        "--report-only",
        help="Exit 0 after writing the comparison even when regressions are found",
    ),
) -> None:
    """Compare two execution reports and fail on regressions."""
    workspace_root = initialize_workspace(workspace)
    try:
        result = compare_ohmo_eval_reports(
            workspace=workspace_root,
            baseline_report=baseline_report,
            candidate_report=candidate_report,
            report_filename=report_filename,
            score_tolerance=score_tolerance,
            report_only=report_only,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise typer.Exit(1)

    report = result.write.report
    print(f"Wrote eval comparison report: {result.write.path}")
    print(
        "Compared eval reports: "
        f"cases={report.case_count} compared={report.compared_count} "
        f"unchanged={report.unchanged_count} improved={report.improvement_count} "
        f"regressed={report.regression_count} added={report.added_count} "
        f"removed={report.removed_count} score_delta={report.score_delta:.3f}"
    )
    if result.report_only:
        print("Report-only mode: regressions did not fail the command")
        return
    if report.regression_count:
        raise typer.Exit(1)


@evals_baseline_app.command("save")
def evals_baseline_save_cmd(
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
    source_report: str = typer.Option(
        "eval_report.json",
        "--from-report",
        help="Execution report filename under evals/reports, or an absolute path",
    ),
    name: str = typer.Option(
        "main",
        "--name",
        help="Baseline name saved under evals/reports/baselines/<name>.json",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace an existing baseline with the same name",
    ),
) -> None:
    """Save an execution report as a named comparison baseline."""
    workspace_root = initialize_workspace(workspace)
    try:
        result = save_ohmo_eval_baseline(
            workspace=workspace_root,
            source_report=source_report,
            name=name,
            overwrite=overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise typer.Exit(1)

    compare_path = result.relative_path.removeprefix("reports/")
    print(f"Saved eval baseline: {result.path}")
    print(
        "Baseline "
        f"{result.name}: cases={result.case_count} "
        f"passed={result.passed_count} non_passed={result.non_passed_count}"
    )
    print(f"Compare with: ohmo evals compare --baseline {compare_path}")


@evals_baseline_app.command("list")
def evals_baseline_list_cmd(
    workspace: str | None = typer.Option(None, "--workspace", help=_WORKSPACE_HELP),
) -> None:
    """List saved eval report baselines."""
    workspace_root = initialize_workspace(workspace)
    try:
        result = list_ohmo_eval_baselines(workspace=workspace_root)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise typer.Exit(1)

    if not result.baselines:
        print("No eval baselines saved.")
        return
    print("Eval baselines:")
    for baseline in result.baselines:
        print(
            f"- {baseline.name} cases={baseline.case_count} "
            f"passed={baseline.passed_count} failed={baseline.failed_count} "
            f"blocked={baseline.blocked_count} error={baseline.error_count} "
            f"path={baseline.relative_path}"
        )

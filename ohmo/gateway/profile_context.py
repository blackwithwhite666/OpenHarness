"""Fail-closed speaker-specific knowledge-base prompt references."""

from __future__ import annotations

from pathlib import Path

from ohmo.gateway.models import GatewayConfig
from ohmo.gateway.turn_context import TurnContext


def render_profile_context(config: GatewayConfig, turn_ctx: TurnContext | None) -> str:
    """Return bounded README references for the authenticated Telegram speaker.

    Resolution is deliberately all-or-nothing: a stale or unsafe configured
    project suppresses the entire speaker fragment rather than exposing a
    partial or redirected project set.
    """
    if turn_ctx is None or turn_ctx.channel.strip().lower() != "telegram":
        return ""
    project_slugs = config.principal_knowledge_projects.get(turn_ctx.principal)
    if not project_slugs or config.knowledge_base_root is None:
        return ""

    try:
        projects_root = (config.knowledge_base_root / "projects").resolve(strict=True)
        readmes = tuple(
            _resolve_readme(projects_root, slug) for slug in project_slugs
        )
    except (OSError, ValueError):
        return ""

    references = "\n".join(f"- {slug}: {readme}" for slug, readme in zip(project_slugs, readmes))
    return (
        "# Speaker knowledge-base project references\n"
        "For this authenticated speaker only, invoke the `knowledge` skill and read "
        "only the smallest relevant slice, starting with the applicable README.md:\n"
        f"{references}"
    )


def _resolve_readme(projects_root: Path, slug: str) -> Path:
    readme = (projects_root / slug / "README.md").resolve(strict=True)
    if not readme.is_file() or readme.parent.parent != projects_root:
        raise ValueError("knowledge project README escapes configured projects root")
    return readme

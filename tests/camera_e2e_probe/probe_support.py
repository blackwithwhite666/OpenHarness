"""Small offline-checkable guards for the optional private Camera probe."""

from __future__ import annotations

import hashlib
import io
import os
import re
import subprocess
import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from ohmo.gateway.camera import CAMERA_AUTHORITY

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


def synthetic_wellness_self_scope(user_id: str):
    """Build explicit in-process self authority for a synthetic owner probe."""
    from telegent.mcp_simple_auth.wellness import WellnessAuthorizationContext
    from telegent.mcp_simple_auth.wellness_delegation import AuthorizedWellnessRead
    from telegent.mcp_simple_auth.wellness_identity import (
        WellnessParticipant,
        WellnessParticipantRegistry,
    )

    registry = WellnessParticipantRegistry(
        default_participant_id=123,
        participants={
            123: WellnessParticipant(123, user_id, "synthetic_owner")
        },
    )
    authorization_context = WellnessAuthorizationContext()

    def authorized_read(body: Mapping[str, object]) -> AuthorizedWellnessRead:
        params = body.get("params")
        if not isinstance(params, Mapping):
            raise ValueError("synthetic wellness probe requires params")
        frozen_body = MappingProxyType(
            {"params": MappingProxyType(dict(params))}
        )
        return AuthorizedWellnessRead(
            reader_participant_id=123,
            target_participant_id=123,
            target_user_id=user_id,
            is_self=True,
            body=frozen_body,
        )

    return registry, authorization_context, authorized_read


async def call_wellness_with_synthetic_self(
    app,
    authorization_context,
    authorized_read,
    arguments: Mapping[str, object],
):
    """Scope one direct helper read and always restore its async-local context."""
    token = authorization_context.set(authorized_read(arguments))
    try:
        return await app.call_tool("get_wellness_data", dict(arguments))
    finally:
        authorization_context.reset(token)


def create_storage_run_dir(root: Path) -> Path:
    """Keep each projection under the ignored worktree root after the run ends."""
    parent = root / "tmp" / "camera-native-docker" / "storage-runs"
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if parent.resolve() != root.resolve() / "tmp" / "camera-native-docker" / "storage-runs":
        raise ValueError("storage root escapes the worktree task directory")
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(parent)],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ignored.returncode != 0:
        raise ValueError("storage root must be Git-ignored")
    os.chmod(parent, 0o700)
    return Path(tempfile.mkdtemp(prefix="finalizer-", dir=parent))


def source_jpeg(path_text: str | None, expected_sha: str | None, root: Path) -> bytes | None:
    if path_text is None and expected_sha is None:
        return None
    if not path_text or not expected_sha or not _SHA256.fullmatch(expected_sha):
        raise ValueError("private source requires a path and lowercase SHA-256")
    try:
        path = Path(path_text).resolve(strict=True)
    except OSError:
        raise ValueError("private source path is unavailable") from None
    if not path.is_file() or not path.is_relative_to(root.resolve()):
        raise ValueError("private source must be a file inside the worktree")
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(path)],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ignored.returncode != 0:
        raise ValueError("private source must be Git-ignored")
    try:
        data = path.read_bytes()
    except OSError:
        raise ValueError("private source cannot be read") from None
    if not 4 <= len(data) <= 10 * 1024 * 1024:
        raise ValueError("private source must be a bounded JPEG")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "JPEG":
                    raise ValueError("private source must be a JPEG")
                image.load()
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("private source must be a decodable bounded JPEG") from exc
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("private source SHA-256 mismatch")
    return data


class NativeClientPreconditionError(ValueError):
    """Explicit lead-run profile/client requirement without auth fallback."""


def require_isolated_native_settings(settings) -> None:
    """Reject configured external runtime surfaces before auth resolution."""
    forbidden = []
    for name in ("hooks", "mcp_servers", "enabled_plugins"):
        if getattr(settings, name, None):
            forbidden.append(name)
    for name in ("allow_project_plugins", "allow_project_skills"):
        if getattr(settings, name, False):
            forbidden.append(name)
    if getattr(settings, "project_skill_dirs", None):
        forbidden.append("project_skill_dirs")
    if forbidden:
        raise NativeClientPreconditionError(
            "native Camera settings enable forbidden runtime surfaces: " + ", ".join(forbidden)
        )


def native_profile_clients(settings, *, resolver=None, codex_client_type=None):
    """Resolve two Codex subscription clients only for explicit native opt-in."""
    from openharness.api.codex_client import CodexApiClient
    from openharness.api.resolver import resolve_api_client_from_settings

    resolve = resolver or resolve_api_client_from_settings
    expected_type = codex_client_type or CodexApiClient
    require_isolated_native_settings(settings)
    profile_name, profile = settings.resolve_profile()
    model = (profile.last_model or "").strip() or profile.default_model
    if not (
        profile_name == "codex"
        and profile.provider == "openai_codex"
        and profile.auth_source == "codex_subscription"
        and model == "gpt-6-luna"
    ):
        raise NativeClientPreconditionError(
            "native Camera requires the Codex subscription profile with model gpt-6-luna"
        )
    bot_client = resolve(settings)
    user_client = resolve(settings)
    if (
        not isinstance(bot_client, expected_type)
        or not isinstance(user_client, expected_type)
        or bot_client is user_client
    ):
        raise NativeClientPreconditionError(
            "native Camera requires two distinct CodexApiClient instances; no provider fallback"
        )
    return bot_client, user_client


def native_preflight_and_clients(
    settings,
    *,
    scenario: str,
    source_path: str | None,
    source_sha256: str | None,
    root: Path,
    resolver=None,
    codex_client_type=None,
):
    """Check all non-auth native inputs before either subscription resolution."""
    if not isinstance(scenario, str) or not scenario.strip():
        raise NativeClientPreconditionError("native Camera requires a non-empty owner scenario")
    require_isolated_native_settings(settings)
    try:
        source_bytes = source_jpeg(source_path, source_sha256, root)
    except ValueError as exc:
        raise NativeClientPreconditionError(f"invalid native Camera source: {exc}") from None
    if source_bytes is None:
        raise NativeClientPreconditionError("native Camera requires a bounded JPEG and SHA-256")
    clients = native_profile_clients(
        settings, resolver=resolver, codex_client_type=codex_client_type
    )
    return clients, source_bytes


def unique_honcho_scope() -> tuple[str, str]:
    run_id = uuid4().hex
    return f"camera-joined-{run_id}", f"camera-joined-session-{run_id}"


def verify_source_worktree(
    path: Path, expected_head: str, *, require_clean: bool
) -> tuple[str, str]:
    """Require a full pinned SHA; acceptance also requires a clean source tree."""
    if not isinstance(expected_head, str) or not _GIT_SHA.fullmatch(expected_head):
        raise ValueError("Camera source revision must be a lowercase full 40-character Git SHA")
    root = path.resolve(strict=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not _GIT_SHA.fullmatch(head) or head != expected_head:
        raise ValueError("Camera source worktree revision changed")
    if require_clean and dirty:
        raise ValueError("Camera source worktree has tracked or non-ignored untracked changes")
    return head, dirty


def require_bound_answer(message, candidate_id: str, native_photo_id: int | str = 77) -> str:
    metadata = message.metadata
    turn_id = metadata.get("_camera_turn_id")
    source_target = (
        metadata.get("native_message_id")
        if metadata.get("callback_query")
        else metadata.get("reply_to_message_id")
    )
    if not (
        metadata.get("_camera_authority") is CAMERA_AUTHORITY
        and metadata.get("_camera_answer") == "yes"
        and metadata.get("_camera_candidate_id") == candidate_id
        and str(source_target) == str(native_photo_id)
        and isinstance(turn_id, str)
        and turn_id
        and len(message.media) == 1
    ):
        raise AssertionError("Camera owner reply was not bound to the native photo")
    return turn_id


def select_finalizer_event(
    messages, candidate_id: str, answer_message_id: str, native_photo_id: int | str
):
    """Reject fixture events and unrelated assistant turns before sync."""
    matches = [
        item
        for item in messages
        if item.metadata.get("role") == "assistant"
        and item.metadata.get("camera_candidate_id") == candidate_id
        and item.metadata.get("camera_answer_bound") == "yes"
        and item.metadata.get("camera_reply_to_native_message_id") == str(native_photo_id)
        and item.metadata.get("source_message_id") == answer_message_id
        and item.metadata.get("nutrition_annotation_status") == "recorded"
    ]
    if len(matches) != 1:
        raise AssertionError("expected exactly one validated finalizer meal event")
    return matches[0]

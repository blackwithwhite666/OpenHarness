"""Honcho workspace bootstrap and eval-only provisioning helpers."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ohmo.memory_service.honcho_client import HonchoClient, Peer, Session, Workspace

_RESOURCE_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")
_DEFAULT_SESSION = "ohmo"


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Resources verified by :func:`bootstrap_workspace`."""

    workspace: Workspace
    peers: tuple[Peer, ...]
    session: Session


async def bootstrap_workspace(
    client: HonchoClient,
    *,
    peers: Sequence[str] = ("ohmo", "ohmo-curated", "owner"),
    observe_others: bool = True,
    session: str = _DEFAULT_SESSION,
) -> BootstrapResult:
    """Idempotently establish the Phase-1 workspace topology.

    Honcho's workspace, peer, and session POST routes are get-or-create
    operations.  Supplying the complete peer mapping with session creation also
    makes re-runs safe: existing memberships are retained and no duplicate
    resources are created.  Only the ``ohmo`` derived-recall peer observes
    others; the curated mirror and owner never do.
    """
    if not isinstance(observe_others, bool):
        raise TypeError("observe_others must be a boolean")
    peer_names = tuple(peers)
    if not peer_names:
        raise ValueError("at least one bootstrap peer is required")
    if len(set(peer_names)) != len(peer_names):
        raise ValueError("bootstrap peer names must be unique")
    _validate_resource_name(session, "session")
    for peer in peer_names:
        _validate_resource_name(peer, "peer")

    workspace_result = await client.get_or_create_workspace()
    peer_results = tuple([await client.get_or_create_peer(peer) for peer in peer_names])
    # Only the assistant peer observes others (forms facts about the owner). The
    # assistant + curated peers are NOT observed-about: honcho attributes each
    # observation to the message SENDER, so leaving observe_me on the assistant
    # makes the deriver mint "ohmo said X" self-observations from the assistant's
    # own verbose turns (pollution). observe_me stays on for human subjects (owner
    # / family), whose turns are the legitimate source of facts about them.
    _NON_SUBJECT_PEERS = frozenset({"ohmo", "ohmo-curated"})
    peer_configuration = {
        peer: {
            "observe_others": observe_others if peer == "ohmo" else False,
            "observe_me": peer not in _NON_SUBJECT_PEERS,
        }
        for peer in peer_names
    }
    session_result = await client.get_or_create_session(
        session,
        peers=peer_configuration,
    )
    return BootstrapResult(workspace_result, peer_results, session_result)


async def provision_eval_workspace(
    *,
    base_url: str,
    admin_jwt: str,
    run: str,
    case: str,
    sample: str | int,
    ttl: dt.timedelta | int | float,
) -> tuple[str, str]:
    """Create an isolated eval workspace and mint its short-lived scoped JWT.

    The admin credential is intentionally a required explicit argument.  It is
    neither read from environment variables nor accepted through gateway
    configuration, so the runtime service cannot accidentally synthesize this
    provisioning authority.
    """
    if not admin_jwt:
        raise ValueError("admin_jwt is required for eval provisioning")
    components = (str(run), str(case), str(sample))
    for component in components:
        _validate_resource_name(component, "eval workspace component")
    workspace_name = "ohmo-eval-" + "-".join(components)
    expiry = dt.datetime.now(dt.timezone.utc) + _coerce_ttl(ttl)

    async with HonchoClient(
        base_url=base_url,
        jwt=admin_jwt,
        workspace=workspace_name,
    ) as client:
        await client.get_or_create_workspace()
        scoped_jwt = await client.create_key(expires_at=expiry)
    return workspace_name, scoped_jwt


def _coerce_ttl(value: dt.timedelta | int | float) -> dt.timedelta:
    if isinstance(value, dt.timedelta):
        result = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result = dt.timedelta(seconds=value)
    else:
        raise TypeError("ttl must be a timedelta or a number of seconds")
    if result <= dt.timedelta(0):
        raise ValueError("ttl must be positive")
    return result


def _validate_resource_name(value: str, description: str) -> None:
    if not value or not _RESOURCE_NAME.fullmatch(value):
        raise ValueError(f"{description} must contain only letters, digits, '_' or '-'")


__all__ = ["BootstrapResult", "bootstrap_workspace", "provision_eval_workspace"]

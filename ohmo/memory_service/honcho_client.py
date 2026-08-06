"""Small async client for the Honcho APIs owned by the memory service.

This module deliberately uses raw HTTP rather than Honcho's SDK.  Keeping the
client here prevents the gateway and tool loop from acquiring Honcho
credentials, while still giving the memory service typed responses for the
routes it will use in later rollout phases.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json as jsonlib
from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias, cast
from urllib.parse import quote

import httpx

from ohmo.memory_audit import memory_audit_event

JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]
QueryValue: TypeAlias = str | int | float | bool | None
ReasoningLevel: TypeAlias = Literal["minimal", "low", "medium", "high", "max"]

_MISSING = object()


class HonchoError(RuntimeError):
    """Raised when Honcho rejects a request or returns a malformed response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        method: str | None = None,
        path: str | None = None,
    ) -> None:
        details = []
        if method and path:
            details.append(f"{method} {path}")
        if status_code is not None:
            details.append(f"HTTP {status_code}")
        prefix = "Honcho request failed"
        if details:
            prefix += f" ({', '.join(details)})"
        super().__init__(f"{prefix}: {message}")
        self.status_code = status_code
        self.method = method
        self.path = path


@dataclasses.dataclass(frozen=True, slots=True)
class Conclusion:
    """External representation of a Honcho conclusion."""

    id: str
    content: str
    observer_id: str
    observed_id: str
    session_id: str | None
    level: str
    created_at: dt.datetime

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "Conclusion":
        return cls(
            id=_required_str(payload, "id"),
            content=_required_str(payload, "content"),
            observer_id=_required_str(payload, "observer_id"),
            observed_id=_required_str(payload, "observed_id"),
            session_id=_optional_str(payload, "session_id"),
            level=_required_str(payload, "level", default="explicit"),
            created_at=_required_datetime(payload, "created_at"),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Message:
    """Message accepted for asynchronous Honcho ingestion."""

    id: str
    content: str
    peer_id: str
    session_id: str
    metadata: Mapping[str, object]
    created_at: dt.datetime
    workspace_id: str
    token_count: int

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "Message":
        return cls(
            id=_required_str(payload, "id"),
            content=_required_str(payload, "content"),
            peer_id=_required_str(payload, "peer_id"),
            session_id=_required_str(payload, "session_id"),
            metadata=_mapping(payload.get("metadata", {}), "message metadata"),
            created_at=_required_datetime(payload, "created_at"),
            workspace_id=_required_str(payload, "workspace_id"),
            token_count=_required_int(payload, "token_count"),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Workspace:
    """Honcho workspace returned by its get-or-create route."""

    id: str
    metadata: Mapping[str, object]
    configuration: Mapping[str, object]
    created_at: dt.datetime

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "Workspace":
        return cls(
            id=_required_str(payload, "id"),
            metadata=_mapping(payload.get("metadata", {}), "workspace metadata"),
            configuration=_mapping(payload.get("configuration", {}), "workspace configuration"),
            created_at=_required_datetime(payload, "created_at"),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Peer:
    """Honcho peer returned by its get-or-create route."""

    id: str
    workspace_id: str
    metadata: Mapping[str, object]
    configuration: Mapping[str, object]
    created_at: dt.datetime

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "Peer":
        return cls(
            id=_required_str(payload, "id"),
            workspace_id=_required_str(payload, "workspace_id"),
            metadata=_mapping(payload.get("metadata", {}), "peer metadata"),
            configuration=_mapping(payload.get("configuration", {}), "peer configuration"),
            created_at=_required_datetime(payload, "created_at"),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Session:
    """Honcho session returned by its get-or-create route."""

    id: str
    workspace_id: str
    is_active: bool
    metadata: Mapping[str, object]
    configuration: Mapping[str, object]
    created_at: dt.datetime

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "Session":
        is_active = payload.get("is_active")
        if not isinstance(is_active, bool):
            raise HonchoError("field 'is_active' must be a boolean")
        return cls(
            id=_required_str(payload, "id"),
            workspace_id=_required_str(payload, "workspace_id"),
            is_active=is_active,
            metadata=_mapping(payload.get("metadata", {}), "session metadata"),
            configuration=_mapping(payload.get("configuration", {}), "session configuration"),
            created_at=_required_datetime(payload, "created_at"),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class PeerContext:
    """A deterministic working representation plus the corresponding peer card."""

    peer_id: str
    target_id: str
    representation: str | None
    peer_card: tuple[str, ...] | None

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "PeerContext":
        raw_card = payload.get("peer_card")
        if raw_card is None:
            peer_card = None
        elif isinstance(raw_card, list) and all(isinstance(item, str) for item in raw_card):
            peer_card = tuple(raw_card)
        else:
            raise HonchoError("field 'peer_card' must be an array of strings or null")
        return cls(
            peer_id=_required_str(payload, "peer_id"),
            target_id=_required_str(payload, "target_id"),
            representation=_optional_str(payload, "representation"),
            peer_card=peer_card,
        )


class HonchoClient:
    """One-workspace asynchronous Honcho v3 client.

    ``jwt`` is intentionally a constructor argument rather than a gateway
    configuration object.  Runtime callers pass a workspace-scoped token; the
    eval provisioning helper passes its separately supplied admin token.
    """

    def __init__(
        self,
        base_url: str,
        jwt: str,
        workspace: str,
        *,
        timeout: float | httpx.Timeout = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("Honcho base_url must not be empty")
        if not jwt:
            raise ValueError("Honcho jwt must not be empty")
        if not workspace:
            raise ValueError("Honcho workspace must not be empty")
        api_url = base_url.rstrip("/")
        if not api_url.endswith("/v3"):
            api_url += "/v3"
        self.workspace = workspace
        self._client = httpx.AsyncClient(
            base_url=api_url + "/",
            headers={"Authorization": f"Bearer {jwt}", "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> "HonchoClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the single underlying HTTP connection pool."""
        await self._client.aclose()

    async def create_conclusions(
        self, conclusions: Sequence[Mapping[str, object] | object]
    ) -> list[Conclusion]:
        """Create a batch of explicit conclusions in this workspace."""
        body = {"conclusions": [_json_object(item) for item in conclusions]}
        payload = await self._request("POST", self._workspace_path("conclusions"), json=body)
        return [Conclusion.from_json(item) for item in _object_list(payload, "conclusions")]

    async def list_conclusions(
        self,
        filters: Mapping[str, object] | None = None,
        reverse: bool = False,
    ) -> list[Conclusion]:
        """List conclusions in the server's recency order."""
        payload = await self._request(
            "POST",
            self._workspace_path("conclusions/list"),
            json={"filters": None if filters is None else dict(filters)},
            params={"reverse": str(reverse).lower()},
        )
        if isinstance(payload, dict):
            payload = payload.get("items")
        return [Conclusion.from_json(item) for item in _object_list(payload, "conclusion page")]

    async def query_conclusions(
        self,
        query: str,
        *,
        observer: str,
        observed: str,
        top_k: int = 10,
        distance: float | None = None,
    ) -> list[Conclusion]:
        """Semantically query one required observer/observed conclusion pair."""
        if not observer or not observed:
            raise ValueError("observer and observed are required")
        body: JSONObject = {
            "query": query,
            "filters": {"observer": observer, "observed": observed},
            "top_k": top_k,
        }
        if distance is not None:
            body["distance"] = distance
        payload = await self._request("POST", self._workspace_path("conclusions/query"), json=body)
        return [Conclusion.from_json(item) for item in _object_list(payload, "conclusions")]

    async def delete_conclusion(self, conclusion_id: str) -> None:
        """Hard-delete one conclusion through Honcho's public route."""
        await self._request(
            "DELETE",
            self._workspace_path(f"conclusions/{_segment(conclusion_id)}"),
        )

    async def create_messages(
        self,
        session: str,
        messages: Sequence[Mapping[str, object] | object],
    ) -> list[Message]:
        """Submit metadata-carrying messages for asynchronous ingestion.

        Phase 1 builds this method but deliberately does not call it from a
        runtime turn path.
        """
        body = {"messages": [_json_object(item) for item in messages]}
        payload = await self._request(
            "POST",
            self._workspace_path(f"sessions/{_segment(session)}/messages"),
            json=body,
        )
        return [Message.from_json(item) for item in _object_list(payload, "messages")]

    async def find_messages_by_client_op_id(
        self,
        session: str,
        client_op_id: str,
        *,
        page_size: int = 100,
        max_pages: int = 100,
    ) -> list[Message]:
        """Find messages carrying one exact operation id.

        Honcho's deployed route is a paginated ``POST`` list endpoint.  The
        metadata filter is intentionally nested: filtering the top-level
        ``metadata`` object with ``client_op_id`` is not equivalent to the
        deployed JSONB containment query.
        """
        if not session or not client_op_id:
            raise ValueError("session and client_op_id are required")
        if not 1 <= page_size <= 100:
            raise ValueError("page_size must be between 1 and 100")
        if not 1 <= max_pages <= 1000:
            raise ValueError("max_pages must be between 1 and 1000")

        found: list[Message] = []
        for page in range(1, max_pages + 1):
            payload = await self._request(
                "POST",
                self._workspace_path(f"sessions/{_segment(session)}/messages/list"),
                json={"filters": {"metadata": {"client_op_id": client_op_id}}},
                params={"page": page, "size": page_size},
            )
            if isinstance(payload, list):
                raw_items = payload
                total = None
            else:
                page_payload = _mapping(payload, "message page")
                raw_items = page_payload.get("items", [])
                total = page_payload.get("total")
                pages = page_payload.get("pages")
                current_page = page_payload.get("page", page)
            items = [Message.from_json(item) for item in _object_list(raw_items, "message page")]
            found.extend(items)

            if not items:
                break
            if isinstance(total, int) and not isinstance(total, bool) and len(found) >= total:
                break
            if (
                isinstance(pages, int)
                and not isinstance(pages, bool)
                and isinstance(current_page, int)
                and current_page >= pages
            ):
                break
            if isinstance(payload, list) or (
                not isinstance(total, int)
                and not isinstance(pages, int)
                and len(items) < page_size
            ):
                break
            # Some test and proxy implementations expose a cursor instead of
            # the standard fastapi-pagination page fields.  Stop only when it
            # explicitly says there is no next page; otherwise page forward.
            if isinstance(payload, dict) and payload.get("next") is None and "next" in payload:
                break
        else:
            raise HonchoError("message lookup exceeded pagination limit")
        return found

    async def dialectic(
        self,
        query: str,
        *,
        observer: str,
        observed: str | None = None,
        session: str | None = None,
        reasoning_level: ReasoningLevel = "low",
    ) -> str | None:
        """Run Honcho's LLM-heavy ``/chat`` recall for a peer pair."""
        body: JSONObject = {
            "query": query,
            "stream": False,
            "reasoning_level": reasoning_level,
        }
        if observed is not None:
            body["target"] = observed
        if session is not None:
            body["session_id"] = session
        payload = _mapping(
            await self._request(
                "POST",
                self._workspace_path(f"peers/{_segment(observer)}/chat"),
                json=body,
            ),
            "dialectic response",
        )
        return _optional_str(payload, "content")

    async def query(
        self,
        query: str,
        *,
        observer: str,
        observed: str | None = None,
        session: str | None = None,
        reasoning_level: ReasoningLevel = "low",
    ) -> str | None:
        """Alias for :meth:`dialectic` used by recall callers."""
        return await self.dialectic(
            query,
            observer=observer,
            observed=observed,
            session=session,
            reasoning_level=reasoning_level,
        )

    async def get_representation(
        self,
        observer: str,
        *,
        observed: str | None = None,
        session: str | None = None,
        search_query: str | None = None,
        search_top_k: int | None = None,
        search_max_distance: float | None = None,
        include_most_frequent: bool | None = None,
        max_conclusions: int | None = None,
    ) -> str:
        """Read the fork's deterministic working representation."""
        body: JSONObject = {}
        optional = {
            "target": observed,
            "session_id": session,
            "search_query": search_query,
            "search_top_k": search_top_k,
            "search_max_distance": search_max_distance,
            "include_most_frequent": include_most_frequent,
            "max_conclusions": max_conclusions,
        }
        body.update({key: value for key, value in optional.items() if value is not None})
        payload = _mapping(
            await self._request(
                "POST",
                self._workspace_path(f"peers/{_segment(observer)}/representation"),
                json=body,
            ),
            "representation response",
        )
        return _required_str(payload, "representation")

    async def get_peer_context(
        self,
        observer: str,
        *,
        observed: str | None = None,
        search_query: str | None = None,
        search_top_k: int | None = None,
        search_max_distance: float | None = None,
        include_most_frequent: bool | None = None,
        max_conclusions: int | None = None,
    ) -> PeerContext:
        """Read the deterministic context/peer-card response."""
        optional: dict[str, QueryValue] = {
            "target": observed,
            "search_query": search_query,
            "search_top_k": search_top_k,
            "search_max_distance": search_max_distance,
            "include_most_frequent": include_most_frequent,
            "max_conclusions": max_conclusions,
        }
        params = {key: value for key, value in optional.items() if value is not None}
        payload = _mapping(
            await self._request(
                "GET",
                self._workspace_path(f"peers/{_segment(observer)}/context"),
                params=params,
            ),
            "peer context response",
        )
        return PeerContext.from_json(payload)

    chat = dialectic
    representation = get_representation
    context = get_peer_context

    async def get_or_create_workspace(self) -> Workspace:
        """Get or create the client's explicit workspace."""
        payload = _mapping(
            await self._request("POST", "workspaces", json={"id": self.workspace}),
            "workspace response",
        )
        return Workspace.from_json(payload)

    async def get_or_create_peer(self, peer: str) -> Peer:
        """Get or create one peer in the client's workspace."""
        payload = _mapping(
            await self._request("POST", self._workspace_path("peers"), json={"id": peer}),
            "peer response",
        )
        return Peer.from_json(payload)

    async def get_or_create_session(
        self,
        session: str,
        *,
        peers: Mapping[str, Mapping[str, object]] | None = None,
    ) -> Session:
        """Get or create a session and ensure its configured peer membership."""
        body: dict[str, object] = {"id": session}
        if peers is not None:
            body["peers"] = {name: dict(configuration) for name, configuration in peers.items()}
        payload = _mapping(
            await self._request("POST", self._workspace_path("sessions"), json=body),
            "session response",
        )
        return Session.from_json(payload)

    async def create_key(self, *, expires_at: dt.datetime) -> str:
        """Mint a workspace-scoped key through the admin-only keys route."""
        if expires_at.tzinfo is None or expires_at.utcoffset() is None:
            raise ValueError("expires_at must be timezone-aware")
        payload = _mapping(
            await self._request(
                "POST",
                "keys",
                params={
                    "workspace_id": self.workspace,
                    "expires_at": expires_at.isoformat(),
                },
            ),
            "key response",
        )
        return _required_str(payload, "key")

    def _workspace_path(self, suffix: str) -> str:
        return f"workspaces/{_segment(self.workspace)}/{suffix}"

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: object = _MISSING,
        params: Mapping[str, QueryValue] | None = None,
    ) -> JSONValue:
        memory_audit_event(
            "honcho_request",
            outcome="attempt",
            operation=method.lower(),
            workspace=self.workspace,
        )
        try:
            if json is _MISSING:
                response = await self._client.request(method, path, params=params)
            else:
                response = await self._client.request(method, path, json=json, params=params)
        except httpx.HTTPError as error:
            raise HonchoError(str(error), method=method, path=path) from error

        if not response.is_success:
            raise HonchoError(
                _error_detail(response),
                status_code=response.status_code,
                method=method,
                path=path,
            )
        if response.status_code == 204 or not response.content:
            return None
        try:
            return cast(JSONValue, response.json())
        except (jsonlib.JSONDecodeError, UnicodeDecodeError) as error:
            raise HonchoError(
                "response was not valid JSON",
                status_code=response.status_code,
                method=method,
                path=path,
            ) from error


def _json_object(value: Mapping[str, object] | object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = dataclasses.asdict(value)
        if isinstance(result, dict):
            return result
    raise TypeError("Honcho batch items must be mappings or dataclass instances")


def _segment(value: str) -> str:
    if not value:
        raise ValueError("Honcho resource identifiers must not be empty")
    return quote(value, safe="")


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise HonchoError(f"{description} must be a JSON object")
    return value


def _object_list(value: object, description: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise HonchoError(f"{description} must be a JSON array")
    return [_mapping(item, f"{description} item") for item in value]


def _required_str(payload: Mapping[str, object], key: str, *, default: str | None = None) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str):
        raise HonchoError(f"field {key!r} must be a string")
    return value


def _optional_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        raise HonchoError(f"field {key!r} must be a string or null")
    return value


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise HonchoError(f"field {key!r} must be an integer")
    return value


def _required_datetime(payload: Mapping[str, object], key: str) -> dt.datetime:
    value = _required_str(payload, key)
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise HonchoError(f"field {key!r} must be an ISO datetime") from error


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except (jsonlib.JSONDecodeError, UnicodeDecodeError):
        return response.text.strip() or response.reason_phrase
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("message")
        if isinstance(detail, str):
            return detail
    return jsonlib.dumps(payload, ensure_ascii=False, separators=(",", ":"))


__all__ = [
    "Conclusion",
    "HonchoClient",
    "HonchoError",
    "Message",
    "Peer",
    "PeerContext",
    "Session",
    "Workspace",
]

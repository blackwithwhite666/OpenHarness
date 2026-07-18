from __future__ import annotations

import base64
import datetime as dt
import inspect
import json
import os
import secrets
from collections import Counter
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from ohmo.memory_service import bootstrap as bootstrap_module
from ohmo.memory_service.bootstrap import bootstrap_workspace, provision_eval_workspace
from ohmo.memory_service.honcho_client import HonchoClient, HonchoError

_NOW = "2026-07-18T09:00:00Z"


class FakeHoncho:
    """In-memory transport matching the current Honcho fork's v3 schemas."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.workspaces: set[str] = set()
        self.peers: set[tuple[str, str]] = set()
        self.sessions: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
        self.conclusions: list[dict[str, object]] = []
        self.side_effects: Counter[str] = Counter()
        self.minted_key = "workspace-scoped-jwt"

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        authorization = request.headers.get("Authorization")
        if authorization not in {"Bearer workspace-jwt", "Bearer admin-jwt"}:
            return httpx.Response(401, json={"detail": "Invalid JWT"})

        path = request.url.path
        parts = path.strip("/").split("/")
        if parts[:2] == ["v3", "workspaces"] and len(parts) == 2:
            return self._workspace(request)
        if parts[:2] == ["v3", "keys"] and len(parts) == 2:
            return self._key(request, authorization)
        if len(parts) < 4 or parts[:2] != ["v3", "workspaces"]:
            return httpx.Response(404, json={"detail": "not found"})

        workspace = parts[2]
        resource = parts[3:]
        if resource == ["peers"]:
            return self._peer(request, workspace)
        if resource == ["sessions"]:
            return self._session(request, workspace)
        if resource[:1] == ["conclusions"]:
            return self._conclusions(request, workspace, resource[1:])
        if len(resource) == 3 and resource[0] == "sessions" and resource[2] == "messages":
            return self._messages(request, workspace, resource[1])
        if len(resource) == 3 and resource[0] == "peers" and resource[2] == "chat":
            return self._chat(request, resource[1])
        if len(resource) == 3 and resource[0] == "peers" and resource[2] == "representation":
            return self._representation(request, resource[1])
        if len(resource) == 3 and resource[0] == "peers" and resource[2] == "context":
            return self._context(request, resource[1])
        return httpx.Response(404, json={"detail": "not found"})

    def _workspace(self, request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        body = _request_json(request)
        assert set(body) == {"id"}
        workspace = str(body["id"])
        created = workspace not in self.workspaces
        if created:
            self.workspaces.add(workspace)
            self.side_effects["workspace"] += 1
        return httpx.Response(
            201 if created else 200,
            json={"id": workspace, "metadata": {}, "configuration": {}, "created_at": _NOW},
        )

    def _peer(self, request: httpx.Request, workspace: str) -> httpx.Response:
        assert request.method == "POST"
        body = _request_json(request)
        assert set(body) == {"id"}
        peer = str(body["id"])
        key = (workspace, peer)
        created = key not in self.peers
        if created:
            self.peers.add(key)
            self.side_effects["peer"] += 1
        return httpx.Response(
            201 if created else 200,
            json={
                "id": peer,
                "workspace_id": workspace,
                "metadata": {},
                "configuration": {},
                "created_at": _NOW,
            },
        )

    def _session(self, request: httpx.Request, workspace: str) -> httpx.Response:
        assert request.method == "POST"
        body = _request_json(request)
        assert set(body).issubset({"id", "peers"})
        session = str(body["id"])
        key = (workspace, session)
        created = key not in self.sessions
        if created:
            self.sessions[key] = body.get("peers", {})  # type: ignore[assignment]
            self.side_effects["session"] += 1
        return httpx.Response(
            201 if created else 200,
            json={
                "id": session,
                "workspace_id": workspace,
                "is_active": True,
                "metadata": {},
                "configuration": {},
                "created_at": _NOW,
            },
        )

    def _conclusions(
        self, request: httpx.Request, workspace: str, suffix: list[str]
    ) -> httpx.Response:
        if request.method == "POST" and not suffix:
            body = _request_json(request)
            assert set(body) == {"conclusions"}
            created = []
            for value in body["conclusions"]:
                item = dict(value)
                result = {
                    "id": f"conclusion-{len(self.conclusions) + 1}",
                    "content": item["content"],
                    "observer_id": item["observer_id"],
                    "observed_id": item["observed_id"],
                    "session_id": item.get("session_id"),
                    "level": "explicit",
                    "created_at": _NOW,
                    "workspace_id": workspace,
                }
                self.conclusions.append(result)
                created.append(result)
            return httpx.Response(201, json=created)
        if request.method == "POST" and suffix == ["list"]:
            body = _request_json(request)
            assert set(body) == {"filters"}
            items = self._filtered(body["filters"])
            if request.url.params.get("reverse") == "true":
                items.reverse()
            return httpx.Response(
                200,
                json={"items": items, "total": len(items), "page": 1, "size": 50, "pages": 1},
            )
        if request.method == "POST" and suffix == ["query"]:
            body = _request_json(request)
            assert {"query", "filters", "top_k"}.issubset(body)
            assert {"observer", "observed"}.issubset(body["filters"])
            return httpx.Response(
                200,
                json=self._filtered(body["filters"])[: int(body["top_k"])],
            )
        if request.method == "DELETE" and len(suffix) == 1:
            before = len(self.conclusions)
            self.conclusions = [item for item in self.conclusions if item["id"] != suffix[0]]
            assert len(self.conclusions) == before - 1
            return httpx.Response(204)
        return httpx.Response(404, json={"detail": "conclusion route not found"})

    def _filtered(self, raw_filters: object) -> list[dict[str, object]]:
        filters = raw_filters or {}
        assert isinstance(filters, dict)
        observer = filters.get("observer", filters.get("observer_id"))
        observed = filters.get("observed", filters.get("observed_id"))
        return [
            item
            for item in self.conclusions
            if (observer is None or item["observer_id"] == observer)
            and (observed is None or item["observed_id"] == observed)
        ]

    def _messages(self, request: httpx.Request, workspace: str, session: str) -> httpx.Response:
        assert request.method == "POST"
        body = _request_json(request)
        assert set(body) == {"messages"}
        messages = []
        for index, item in enumerate(body["messages"], start=1):
            messages.append(
                {
                    "id": f"message-{index}",
                    "content": item["content"],
                    "peer_id": item["peer_id"],
                    "session_id": session,
                    "metadata": item.get("metadata") or {},
                    "created_at": _NOW,
                    "workspace_id": workspace,
                    "token_count": 4,
                }
            )
        return httpx.Response(201, json=messages)

    def _chat(self, request: httpx.Request, observer: str) -> httpx.Response:
        assert request.method == "POST"
        body = _request_json(request)
        assert body == {
            "query": "What does the owner prefer?",
            "stream": False,
            "reasoning_level": "medium",
            "target": "owner",
            "session_id": "ohmo",
        }
        return httpx.Response(200, json={"content": f"{observer} remembers concise replies"})

    def _representation(self, request: httpx.Request, observer: str) -> httpx.Response:
        assert request.method == "POST"
        assert _request_json(request) == {
            "target": "owner",
            "search_query": "format",
            "include_most_frequent": True,
            "max_conclusions": 20,
        }
        return httpx.Response(200, json={"representation": f"{observer}: concise replies"})

    def _context(self, request: httpx.Request, observer: str) -> httpx.Response:
        assert request.method == "GET"
        assert dict(request.url.params) == {
            "target": "owner",
            "search_query": "format",
            "search_top_k": "3",
            "include_most_frequent": "true",
        }
        return httpx.Response(
            200,
            json={
                "peer_id": observer,
                "target_id": "owner",
                "representation": "Prefers concise replies",
                "peer_card": ["owner", "concise"],
            },
        )

    def _key(self, request: httpx.Request, authorization: str) -> httpx.Response:
        assert request.method == "POST"
        if authorization != "Bearer admin-jwt":
            return httpx.Response(401, json={"detail": "Resource requires admin privileges"})
        assert set(request.url.params) == {"workspace_id", "expires_at"}
        expires_at = dt.datetime.fromisoformat(request.url.params["expires_at"])
        assert expires_at.tzinfo is not None
        self.side_effects["key"] += 1
        return httpx.Response(200, json={"key": self.minted_key})


def _request_json(request: httpx.Request) -> dict[str, Any]:
    value = json.loads(request.content)
    assert isinstance(value, dict)
    return value


@pytest.fixture
def fake_honcho() -> FakeHoncho:
    return FakeHoncho()


@pytest.fixture
async def honcho_client(fake_honcho: FakeHoncho):
    async with HonchoClient(
        "https://honcho.test",
        "workspace-jwt",
        "workspace-one",
        transport=fake_honcho.transport(),
    ) as client:
        yield client


async def test_client_matches_conclusion_message_and_recall_contracts(
    honcho_client: HonchoClient, fake_honcho: FakeHoncho
) -> None:
    conclusion_input = {
        "content": "The owner prefers concise replies",
        "observer_id": "ohmo",
        "observed_id": "owner",
        "session_id": "ohmo",
    }
    created = await honcho_client.create_conclusions([conclusion_input])
    assert created[0].observer_id == "ohmo"
    assert created[0].observed_id == "owner"
    assert created[0].created_at.tzinfo is not None

    listed = await honcho_client.list_conclusions(
        {"observer_id": "ohmo", "observed_id": "owner"}, reverse=True
    )
    assert [item.id for item in listed] == [created[0].id]
    list_request = fake_honcho.requests[-1]
    assert list_request.url.path == "/v3/workspaces/workspace-one/conclusions/list"
    assert list_request.url.params["reverse"] == "true"

    queried = await honcho_client.query_conclusions(
        "reply format", observer="ohmo", observed="owner", top_k=4, distance=0.3
    )
    assert queried == created
    query_body = _request_json(fake_honcho.requests[-1])
    assert query_body == {
        "query": "reply format",
        "filters": {"observer": "ohmo", "observed": "owner"},
        "top_k": 4,
        "distance": 0.3,
    }

    messages = await honcho_client.create_messages(
        "ohmo",
        [
            {
                "content": "Please be concise",
                "peer_id": "owner",
                "metadata": {"logical_turn_id": "turn-1", "client_op_id": "op-1"},
            }
        ],
    )
    assert messages[0].metadata["client_op_id"] == "op-1"
    assert fake_honcho.requests[-1].url.path.endswith("/sessions/ohmo/messages")

    dialectic = await honcho_client.dialectic(
        "What does the owner prefer?",
        observer="ohmo",
        observed="owner",
        session="ohmo",
        reasoning_level="medium",
    )
    assert dialectic == "ohmo remembers concise replies"

    representation = await honcho_client.get_representation(
        "ohmo",
        observed="owner",
        search_query="format",
        include_most_frequent=True,
        max_conclusions=20,
    )
    assert representation == "ohmo: concise replies"
    context = await honcho_client.get_peer_context(
        "ohmo",
        observed="owner",
        search_query="format",
        search_top_k=3,
        include_most_frequent=True,
    )
    assert context.representation == "Prefers concise replies"
    assert context.peer_card == ("owner", "concise")

    await honcho_client.delete_conclusion(created[0].id)
    assert not fake_honcho.conclusions
    assert fake_honcho.requests[-1].method == "DELETE"

    assert all(
        request.headers["Authorization"] == "Bearer workspace-jwt"
        for request in fake_honcho.requests
    )


@pytest.mark.parametrize(
    ("status_code", "detail"),
    [(401, "Invalid JWT"), (500, "database unavailable")],
)
async def test_client_raises_clear_error_for_non_success(status_code: int, detail: str) -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(status_code, json={"detail": detail})
    )
    async with HonchoClient(
        "https://honcho.test/v3",
        "workspace-jwt",
        "workspace-one",
        transport=transport,
    ) as client:
        with pytest.raises(HonchoError, match=rf"HTTP {status_code}.*{detail}") as error:
            await client.get_or_create_workspace()
    assert error.value.status_code == status_code
    assert "workspace-jwt" not in str(error.value)


async def test_bootstrap_is_idempotent_and_configures_only_derived_observer(
    fake_honcho: FakeHoncho,
) -> None:
    async with HonchoClient(
        "https://honcho.test",
        "workspace-jwt",
        "workspace-one",
        transport=fake_honcho.transport(),
    ) as client:
        first = await bootstrap_workspace(client)
        second = await bootstrap_workspace(client)

    assert first.workspace.id == second.workspace.id == "workspace-one"
    assert [peer.id for peer in first.peers] == ["ohmo", "ohmo-curated", "owner"]
    assert first.session.id == "ohmo"
    assert fake_honcho.sessions[("workspace-one", "ohmo")] == {
        "ohmo": {"observe_others": True},
        "ohmo-curated": {"observe_others": False},
        "owner": {"observe_others": False},
    }
    assert fake_honcho.side_effects == Counter({"peer": 3, "workspace": 1, "session": 1})
    assert sum(request.url.path == "/v3/workspaces" for request in fake_honcho.requests) == 2


async def test_eval_provisioning_uses_explicit_admin_token_and_keys_route(
    fake_honcho: FakeHoncho,
) -> None:
    constructed: list[dict[str, object]] = []

    def client_factory(**kwargs: object) -> HonchoClient:
        constructed.append(dict(kwargs))
        return HonchoClient(**kwargs, transport=fake_honcho.transport())  # type: ignore[arg-type]

    before = dt.datetime.now(dt.timezone.utc)
    with patch.object(bootstrap_module, "HonchoClient", side_effect=client_factory):
        workspace, scoped_jwt = await provision_eval_workspace(
            base_url="https://honcho.test",
            admin_jwt="admin-jwt",
            run="run7",
            case="case2",
            sample=3,
            ttl=dt.timedelta(minutes=15),
        )

    assert workspace == "ohmo-eval-run7-case2-3"
    assert scoped_jwt == fake_honcho.minted_key
    assert constructed == [
        {
            "base_url": "https://honcho.test",
            "jwt": "admin-jwt",
            "workspace": "ohmo-eval-run7-case2-3",
        }
    ]
    assert [request.url.path for request in fake_honcho.requests] == [
        "/v3/workspaces",
        "/v3/keys",
    ]
    assert all(
        request.headers["Authorization"] == "Bearer admin-jwt" for request in fake_honcho.requests
    )
    expires_at = dt.datetime.fromisoformat(fake_honcho.requests[-1].url.params["expires_at"])
    assert dt.timedelta(minutes=14) < expires_at - before < dt.timedelta(minutes=16)
    assert (
        inspect.signature(provision_eval_workspace).parameters["admin_jwt"].default
        is inspect.Parameter.empty
    )
    assert "GatewayConfig" not in bootstrap_module.__dict__


async def test_eval_provisioning_rejects_missing_admin_credential() -> None:
    with pytest.raises(ValueError, match="admin_jwt is required"):
        await provision_eval_workspace(
            base_url="https://honcho.test",
            admin_jwt="",
            run="run",
            case="case",
            sample=1,
            ttl=60,
        )


@pytest.mark.skipif(os.environ.get("OHMO_HONCHO_LIVE") != "1", reason="live Honcho smoke disabled")
async def test_live_honcho_create_query_smoke() -> None:
    base_url = os.environ.get("OHMO_HONCHO_BASE_URL")
    jwt = os.environ.get("OHMO_HONCHO_JWT")
    if not base_url or not jwt:
        pytest.skip("OHMO_HONCHO_BASE_URL and OHMO_HONCHO_JWT are required")
    workspace = os.environ.get("OHMO_HONCHO_WORKSPACE") or _jwt_workspace(jwt)
    if not workspace:
        pytest.skip("workspace claim is absent; set OHMO_HONCHO_WORKSPACE")

    marker = f"live-smoke-{secrets.token_hex(8)}"
    async with HonchoClient(base_url, jwt, workspace, timeout=30.0) as client:
        await bootstrap_workspace(client)
        created = await client.create_conclusions(
            [
                {
                    "content": f"OpenHarness Honcho {marker}",
                    "observer_id": "ohmo-curated",
                    "observed_id": "owner",
                }
            ]
        )
        try:
            queried = await client.query_conclusions(
                marker,
                observer="ohmo-curated",
                observed="owner",
                top_k=10,
            )
            assert created[0].id in {item.id for item in queried}
        finally:
            await client.delete_conclusion(created[0].id)


def _jwt_workspace(token: str) -> str | None:
    try:
        encoded = token.split(".")[1]
        payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    except (IndexError, ValueError, json.JSONDecodeError):
        return None
    workspace = payload.get("w") if isinstance(payload, dict) else None
    return workspace if isinstance(workspace, str) and workspace else None

"""Private per-call signing for delegated wellness reads."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

META_KEY = "io.telegent/wellness-delegation/v1"
_KID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@dataclass(frozen=True)
class WellnessDelegationConfig:
    key: bytes = field(repr=False)
    kid: str
    issuer: str
    audience: str
    client_id: str

    @classmethod
    def from_env(cls) -> "WellnessDelegationConfig":
        names = (
            "WELLNESS_DELEGATION_SIGNING_KEY",
            "WELLNESS_DELEGATION_KID",
            "WELLNESS_DELEGATION_ISSUER",
            "WELLNESS_DELEGATION_AUDIENCE",
            "WELLNESS_DELEGATION_CLIENT_ID",
        )
        values = [os.environ.get(name) for name in names]
        if any(value is None or value == "" for value in values):
            raise ValueError("wellness delegation configuration is incomplete")
        secret, kid, issuer, audience, client_id = values
        assert secret is not None and kid is not None and issuer is not None
        assert audience is not None and client_id is not None
        key = secret.encode("utf-8")
        configured = (kid, issuer, audience, client_id)
        if (
            len(key) < 32
            or any(len(value) > 512 for value in values if value is not None)
            or any(ord(char) < 0x20 for value in values for char in value or "")
            or not _KID.fullmatch(kid)
            or any(
                value != value.strip() for value in configured[1:]
            )
        ):
            raise ValueError("wellness delegation configuration is invalid")
        return cls(key, kid, issuer, audience, client_id)


@dataclass(frozen=True)
class TrustedWellnessActor:
    """An actor captured directly from this inbound turn's admission."""

    telegram_id: str


def _validate_json(value: object) -> None:
    if value is None or type(value) in (str, bool, int, float):
        return
    if type(value) is list:
        for item in value:
            _validate_json(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("wellness request body has a non-string key")
            _validate_json(key)
            _validate_json(item)
        return
    raise ValueError("wellness request body contains a non-JSON type")


def canonical_body(body: dict[str, object]) -> bytes:
    """Canonical UTF-8 JSON for the final decoded SDK argument dictionary."""
    if type(body) is not dict:
        raise ValueError("wellness request body must be a JSON object")
    _validate_json(body)
    try:
        encoded = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("wellness request body is not canonical JSON") from exc
    if len(encoded) > 65536:
        raise ValueError("wellness request body is too large")
    return encoded


def sign_wellness_call(
    config: WellnessDelegationConfig,
    actor: TrustedWellnessActor,
    body: dict[str, object],
    *,
    now: int | None = None,
) -> dict[str, str]:
    subject = actor.telegram_id
    if (
        type(subject) is not str
        or not subject.isascii()
        or not subject.isdigit()
        or subject.startswith("0")
    ):
        raise ValueError("trusted wellness Telegram principal is invalid")
    encoded_body = canonical_body(body)
    issued = int(datetime.now(timezone.utc).timestamp()) if now is None else now
    if isinstance(issued, bool) or not isinstance(issued, int):
        raise ValueError("wellness token time is invalid")
    header = {"alg": "HS256", "typ": "JWT", "kid": config.kid}
    claims = {
        "schema_version": 1,
        "iss": config.issuer,
        "aud": config.audience,
        "azp": config.client_id,
        "sub": f"telegram:{subject}",
        "iat": issued,
        "exp": issued + 60,
        "tool": "get_wellness_data",
        "body_sha256": hashlib.sha256(encoded_body).hexdigest(),
    }

    def part(value: object) -> str:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8", errors="strict")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    signing_input = f"{part(header)}.{part(claims)}".encode("ascii")
    signature = hmac.new(config.key, signing_input, hashlib.sha256).digest()
    token = (
        f"{signing_input.decode('ascii')}."
        f"{base64.urlsafe_b64encode(signature).rstrip(b'=').decode('ascii')}"
    )
    if len(token.encode("ascii")) > 4096:
        raise ValueError("wellness delegation token is too large")
    return {META_KEY: token}

from __future__ import annotations

from openharness.channels.bus.events import OutboundDeliveryReceipt
from openharness.channels.impl.base import BaseChannel


class _Cfg:
    def __init__(self, allow: list[str]) -> None:
        self.allow_from = allow


class _FakeChannel(BaseChannel):
    name = "telegram"

    def __init__(self, allow: list[str]) -> None:  # skip BaseChannel.__init__ (no bus needed)
        self.config = _Cfg(allow)

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def send(self, *args, **kwargs) -> OutboundDeliveryReceipt | None: ...


def test_allow_from_at_prefixed_username_matches():
    ch = _FakeChannel(["@blackwithwhite"])
    assert ch.is_allowed("116870365|blackwithwhite")
    assert ch.is_allowed("999|blackwithwhite")


def test_allow_from_bare_username_matches():
    assert _FakeChannel(["blackwithwhite"]).is_allowed("116870365|blackwithwhite")


def test_allow_from_numeric_id_matches():
    assert _FakeChannel(["116870365"]).is_allowed("116870365|blackwithwhite")


def test_allow_from_denies_unknown():
    assert not _FakeChannel(["@alice"]).is_allowed("1|bob")


def test_allow_from_wildcard_allows_all():
    assert _FakeChannel(["*"]).is_allowed("1|bob")


def test_allow_from_empty_denies_all():
    assert not _FakeChannel([]).is_allowed("1|bob")

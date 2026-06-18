"""Tests for the ohmo contact registry."""

from __future__ import annotations

from pathlib import Path

from ohmo.contact_registry import ContactStore


def test_record_inbound_creates_and_updates_contact(tmp_path: Path):
    store = ContactStore(tmp_path)

    store.record_inbound(
        channel="telegram",
        chat_id="123",
        user_id="456",
        username="@alice",
        first_name="Alice",
        display_name="Alice Doe",
    )
    first = store.get("telegram", "123")

    assert first is not None
    assert first.first_seen == first.last_seen
    assert first.message_count == 1
    assert first.username == "alice"

    store.record_inbound(channel="telegram", chat_id="123", username="@alice2")
    second = store.get("telegram", "123")

    assert second is not None
    assert second.message_count == 2
    assert second.last_seen > first.last_seen
    assert second.username == "alice2"
    assert second.first_name == "Alice"


def test_record_inbound_does_not_overwrite_known_values_with_empty(tmp_path: Path):
    store = ContactStore(tmp_path)
    store.record_inbound(
        channel="telegram",
        chat_id="123",
        user_id="456",
        username="alice",
        first_name="Alice",
        display_name="Alice Doe",
    )

    store.record_inbound(channel="telegram", chat_id="123")
    contact = store.get("telegram", "123")

    assert contact is not None
    assert contact.user_id == "456"
    assert contact.username == "alice"
    assert contact.first_name == "Alice"
    assert contact.display_name == "Alice Doe"


def test_resolve_uses_expected_match_tiers(tmp_path: Path):
    store = ContactStore(tmp_path)
    store.record_inbound(
        channel="telegram",
        chat_id="100",
        user_id="500",
        username="@AliceDev",
        first_name="Alice",
        display_name="Alice Doe",
    )
    store.record_inbound(
        channel="telegram",
        chat_id="200",
        user_id="600",
        username="bob",
        first_name="Bob",
        display_name="Robert Builder",
    )
    store.record_inbound(
        channel="telegram",
        chat_id="-300",
        user_id="700",
        username="builder",
        first_name="Rob",
        display_name="Rob Builder",
    )

    assert [c.chat_id for c in store.resolve("@AliceDev")] == ["100"]
    assert [c.chat_id for c in store.resolve("AliceDev")] == ["100"]
    assert [c.chat_id for c in store.resolve("100")] == ["100"]
    assert [c.chat_id for c in store.resolve("500")] == ["100"]
    assert [c.chat_id for c in store.resolve("alice")] == ["100"]
    assert store.resolve("dev") == []
    assert [c.chat_id for c in store.resolve("dev", fuzzy=True)] == ["100"]
    assert store.resolve("unknown") == []
    assert store.resolve("build") == []
    assert {c.chat_id for c in store.resolve("build", fuzzy=True)} == {"200", "-300"}


def test_list_orders_contacts_by_recency(tmp_path: Path):
    store = ContactStore(tmp_path)
    store.record_inbound(channel="telegram", chat_id="100", username="first")
    store.record_inbound(channel="telegram", chat_id="200", username="second")

    assert [contact.chat_id for contact in store.list()] == ["200", "100"]


def test_resolve_empty_and_other_channel_do_not_match(tmp_path: Path):
    store = ContactStore(tmp_path)
    store.record_inbound(channel="feishu", chat_id="100", username="alice")

    assert store.resolve("") == []
    assert store.resolve("@alice") == []
    assert [c.chat_id for c in store.resolve("@alice", channel="feishu")] == ["100"]


def test_get_returns_none_for_unknown(tmp_path: Path):
    assert ContactStore(tmp_path).get("telegram", "missing") is None


def test_fresh_store_reads_existing_file(tmp_path: Path):
    ContactStore(tmp_path).record_inbound(
        channel="telegram",
        chat_id="123",
        username="alice",
    )

    assert ContactStore(tmp_path).get("telegram", "123") is not None

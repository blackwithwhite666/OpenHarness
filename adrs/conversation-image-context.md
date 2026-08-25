# ADR: Durable conversation image context

- **Status:** Accepted
- **Date:** 2026-08-25
- **Owner:** Ohmo / OpenHarness

## Context

Nutrition photos and ordinary channel photos are genuine events in Marina's
shared conversation. Persisting their inline base64 in conversation history made
snapshots grow and caused providers to resend historical pixels on unrelated
turns. The architecture must preserve conversational continuity while separating
durable history from transient provider transport, and retries must not duplicate
the same inbound event.

## Decision

- Keep image events in the existing Ohmo conversation and runtime.
- Represent durable images with `AttachmentRefBlock`: a SHA-256 attachment ID,
  image MIME type, byte size, and bounded label, with no inline bytes. Store the
  object once in Ohmo's global content-addressed attachment store using atomic,
  verified reads and writes. Global deduplication does not grant access.
- Retain `ImageBlock` as a transport-only compatibility type. Pixels may appear
  in a provider request for the active inbound turn or immediately after an
  explicit `load_conversation_image` call, but are externalized before durable
  history or any snapshot is written, including error paths.
- Project historical references to providers as bounded textual placeholders.
  Provider adapters never resolve an `AttachmentRefBlock` automatically.
- Authorize `load_conversation_image` conversation-locally. Each bundle injects
  an allowlist callable derived from attachment references currently present in
  that bundle's `engine.messages`. Unauthorized and nonexistent IDs return the
  same unavailable result; trusted tool output alone may supply transient pixels
  to the following provider call in the current run.
- Assign a stable `event_id` from trusted channel identifiers, or from the trusted
  nutrition candidate ID. Retrying the same event reprocesses it without adding
  another durable user turn, while retaining its attachment reference.
- Lazily migrate legacy inline-image snapshots when loaded. Equal payloads map to
  one stored SHA-256 object. Every newly written Ohmo snapshot must contain no
  inline `ImageBlock` data.

## Rejected alternatives

- **Isolated nutrition session or estimation pipeline:** this would split real
  conversation events and lose shared conversational context.
- **URLs alone:** URLs are not a trusted, durable, content-addressed ownership
  boundary and introduce external availability and provider-fetch behavior.
- **Destructive image deletion:** this removes the logical event and prevents an
  authorized model from reopening it on demand.
- **Provider `previous_response_id` as the foundation:** provider-owned opaque
  state cannot be the durable source of truth for local persistence, migration,
  authorization, or cross-provider behavior. It may be an optimization later.

## Consequences and known limitations

- Unrelated text turns carry placeholders but no historical `input_image`
  payloads; generic request-local `ImageBlock` callers remain compatible.
- Current-turn images and explicitly reopened images still incur provider vision
  cost.
- Content-addressed object garbage collection is future work.
- Legacy data is converted in memory on load and written in the new form by a
  later snapshot save; existing snapshot files are not eagerly rewritten.

## Implementation and tests

Key implementation files:

- `src/openharness/engine/messages.py`
- `src/openharness/engine/query.py`
- `src/openharness/engine/query_engine.py`
- `src/openharness/api/codex_client.py`
- `src/openharness/api/openai_client.py`
- `ohmo/attachment_store.py`
- `ohmo/conversation_image_tool.py`
- `ohmo/gateway/runtime.py`
- `ohmo/session_storage.py`

Focused coverage is in:

- `tests/test_ohmo/test_conversation_attachments.py`
- `tests/test_ohmo/test_gateway.py`
- `tests/test_engine/test_query_engine.py`
- `tests/test_api/test_codex_client.py`
- `tests/test_api/test_openai_client.py`
- `tests/test_services/test_compact.py`

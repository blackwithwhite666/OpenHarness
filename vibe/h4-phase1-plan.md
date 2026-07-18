# Plan: h4-phase1 — honcho client + catalog + shadow mode (H4 rollout steps 2–3)

**Status: DESIGN ONLY — not executed.** Successor to `vibe/h4-phase0-plan.md` (merged as
`h4/phase0-file-seam`, PR #55). Source ADR: `/Users/dldmitry/agents-playgroud/adrs/honcho-openharness-backend.md`
(rev 6). Phase 0 landed the *seam* (async `MemoryBackend`, prompt seam, factory/config, command guard,
`TurnContext` plumbing) with **no honcho in the hot path**. Phase 1 builds the honcho-facing pieces up to
**shadow mode** — honcho is queried **off-path** and never shown to the agent; the catalog is the
authority; nothing user-visible changes.

Deploy target for live checks: `https://memory.worfalomey.top/` (honcho, JWT-authed) + embeds via
`https://inference.worfalomey.top/` — both already live (see `adrs/honcho-memory-service.md`).

Repo `/Users/dldmitry/tmp/OpenHarness`; branch off the merged Phase-0 head. Tests: `.venv/bin/python -m pytest`.
Defaults: model=gpt-5.6-sol reasoning=xhigh workdir=/Users/dldmitry/tmp/OpenHarness timeout=45m

## What Phase 1 delivers (and what it does NOT)
- ✅ transactional SQLite **catalog** (the future single authority) + FTS5, a `CatalogMemoryBackend`,
  the `.md`→catalog **migration**, the **out-of-process memory service** (separate UID) holding the
  catalog + honcho JWT, the async **honcho client** (conclusions/context/messages) + eval-only
  provisioning credential + at-least-once **outbox**, and **shadow mode** (off-path honcho recall + a
  completeness/latency/ranking comparison log).
- ⛔ **Not** in Phase 1 (→ Phase 2): user-visible honcho recall (opt-in), the confidentiality-gate
  **enforcement**, the **OS-attested tool sandbox** (`tools_confined`'s hard boundary), curated writes as
  the source of truth, conversation-learning (messages→deriver) + dreamer. Phase 1 keeps the catalog as
  authority and honcho strictly off the correctness path.

## Hard dependency to resolve before any user-visible recall (flag, not built here)
`tools_confined` (ADR §4.4) requires an **OS-attested tool sandbox** running tools under a confined
principal distinct from the gateway/memory-service UIDs. Phase 1's out-of-process service + gateway-only
RPC socket auth is *necessary but not sufficient* — the OS sandbox attestation is a separate infra lift
that gates Phase 2. Phase 1 shadow mode is safe without it only because honcho recall is never surfaced
and the honcho collection holds only the owner's own mirror.

## Global constraints (restate per step)
- Do **not** commit/push/deploy from agent contracts — orchestrator gates git + any live check.
- The honcho JWT and catalog must **never** be constructed in the gateway process once the memory service
  exists (steps ≥4): they live in the separate-UID service, reachable only over the gateway-authorized RPC.
- Keep the catalog authoritative; honcho stays off the correctness path in every Phase-1 step.
- Match Phase-0 style; new files < 600 LOC; async surfaces; `asyncio_mode=auto`.

---

## Step: p1-catalog — transactional SQLite catalog + FTS5
**What:** `ohmo/memory_catalog.py` — a SQLite catalog: table with `name/slug, title, content, size, usage,
pinned, generation, source∈{curated,derived}, archive_status∈{active,archived}, honcho_conclusion_ids
(json), outbox_state`; `BEGIN IMMEDIATE` write-serialization + `busy_timeout` + schema-migration locking;
CRUD + list/get + soft-archive; **FTS5** virtual table over active+archived `content`; the re-homed write
guarantees from ohmo (exact-dedup, 4000/24000 budgets, threat-scan) enforced **inside the transaction**;
`record_use` usage ranking. Blended `search` = catalog FTS hits (+ later honcho derived hits, labelled).
**Files:** `ohmo/memory_catalog.py`, `tests/test_ohmo/test_memory_catalog.py`.
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_memory_catalog.py -q`
(concurrency test: two writers, budget/dedup hold under `BEGIN IMMEDIATE`; FTS returns active+archived).

## Step: p1-catalog-backend — CatalogMemoryBackend (Protocol impl)
**What:** `CatalogMemoryBackend(MemoryBackend)` (async) over the catalog, a drop-in beside
`FileMemoryBackend`. `render_prompt` renders the curated block from the catalog under the composite budget
(ADR §6). `append_turn` no-op for now (derived learning is Phase 2). Extend `make_memory_backend` to
return it under a new internal kind (still **not** selectable in prod; `memory_backend` default stays
`file`).
**Files:** `ohmo/memory_backend.py` (add class + factory branch), `tests/test_ohmo/test_catalog_backend.py`.
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_catalog_backend.py tests/test_ohmo/test_memory_backend_file.py -q`
(the model surface — list/get/search/add/update/remove/render_prompt — matches `FileMemoryBackend` for the
same inputs; the characterization suite still green).

## Step: p1-migration — .md → catalog migration + reconciliation
**What:** `ohmo/tools/migrate_memory_to_catalog.py` — dry-run inventory of active + archived `.md`,
threat-scan, deterministic idempotency (re-runnable), **archived-title fallback** (filename-derived title +
a reconciliation warning — archived files lost their index title, ADR §7), a reconciliation report, and a
catalog↔file export/rollback path. **No honcho.**
**Files:** the tool + `tests/test_ohmo/test_migrate_memory.py` (fixtures: active+archived; idempotent
re-run; title fallback; rollback).
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_migrate_memory.py -q`.

## Step: p1-memory-service — out-of-process memory service (separate UID) + gateway client
**What:** a standalone service process (own module, e.g. `ohmo/memory_service/`) exposing the async
backend ops over a **Unix domain socket**; it owns the catalog (and later the honcho JWT). Socket is
**authorized to the gateway principal only** (socket perms + a per-turn capability token minted by the
gateway; `prepare_turn` never exposed to the tool loop). A thin async `MemoryServiceClient(MemoryBackend)`
in the gateway speaks to it. systemd-user unit runs the service under a **distinct UID**. (ADR §4.4 primary
boundary.) The OS-attested tool sandbox is a *documented Phase-2 prerequisite*, not built here.
**Files:** `ohmo/memory_service/{server,client,protocol}.py`, a systemd unit template, tests.
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_memory_service.py -q`
(client↔server round-trip over UDS; a connection without the capability token is refused; catalog ops match
the in-process `CatalogMemoryBackend`).

## Step: p1-honcho-client — async honcho client + bootstrap + eval provisioning credential
**What:** inside the memory service: an async honcho client (conclusions create/list/query, context/
representation, messages) reusing the deployed stack; **workspace/peer/session bootstrap** (peers `ohmo`,
`ohmo-curated`, owner; `observe_others` config); a **workspace-scoped runtime JWT** held only by the
service; and a **separate eval-only admin/provisioning credential** (never in the gateway) that creates
`ohmo-eval-<run>-<case>-<sample>` workspaces + mints short-lived tokens (ADR §4.3, §7). Contract tests use
a **fake honcho HTTP** server; one opt-in live smoke against `memory.worfalomey.top`.
**Files:** `ohmo/memory_service/honcho_client.py`, `ohmo/memory_service/bootstrap.py`, tests (mock HTTP).
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_honcho_client.py -q`
(mocked conclusions/context/messages round-trip; bootstrap idempotent; the provisioning credential path is
admin-only per `honcho/src/routers/workspaces.py`).

## Step: p1-outbox — at-least-once curated mirror (outbox), off the correctness path
**What:** the durable **outbox** (in the catalog txn, ADR §6): curated `add/update/remove` write the
catalog **and** enqueue a mirror op to `ohmo-curated→owner` (drained under atomic leases, at-least-once,
records only **acknowledged** `honcho_conclusion_ids`; reconciliation finds timeout orphans). **Nothing
user-visible depends on mirror exactness.** No dreaming on the curated collection (direct conclusions don't
schedule dreams).
**Files:** `ohmo/memory_service/outbox.py`, tests (timeout-after-commit ⇒ at-least-once, no data loss;
reconciliation prunes orphans; mirror failure never blocks the catalog write).
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_outbox.py -q`.

## Step: p1-shadow — shadow mode (off-path honcho recall + comparison log)
**What:** wire the memory service to, on each owner turn, **also** query honcho derived `search`/`context`
**off-path** and **log** completeness/latency/ranking vs the catalog — **without** showing honcho output to
the agent. Conversation ingestion + dreamer stay **off**. Gate behind a `memory_backend=shadow` config that
is **owner-only**; prod default stays `file`. Add a small report so the comparison can be reviewed before
Phase 2's opt-in recall.
**Files:** the shadow hook in the memory service + a comparison-log writer + tests.
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_shadow_mode.py -q`
(the agent-visible prompt is byte-identical to `file`/catalog with shadow on — honcho output never enters
the prompt; the comparison log is written).

---

## Orchestrator (out-of-contract) steps
- After p1-honcho-client + p1-shadow: a **live shadow dry-run** against `memory.worfalomey.top` (real
  bootstrap on a throwaway workspace, mirror a few curated entries, compare recall) — orchestrator-run,
  not in an agent contract.
- Full-suite acceptance + PR per the Phase-0 pattern (one PR for Phase 1, or split catalog vs service).

## Phase 2 (later, separate plan)
Opt-in user-visible recall + the fail-closed confidentiality-gate **enforcement** + the **OS-attested tool
sandbox** + positive recursive allowlist; then curated-writes-as-authority, conversation-learning + dreamer,
multi-sender, and (only after reconciliation + DR export) any default switch.

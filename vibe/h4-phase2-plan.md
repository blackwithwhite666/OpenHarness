# Plan: h4-phase2 — user-visible recall + the hard confidentiality boundary

**Status: DESIGN ONLY — not executed.** Gated on **Phase 4 PASS**. Source ADR:
`/Users/dldmitry/agents-playgroud/adrs/honcho-openharness-backend.md` (rev 6; §4.4 tool confinement,
§5-D5 gate). This is the **first phase where honcho output reaches the owner** — so it lands the
**OS-attested tool sandbox + fail-closed gate BEFORE** surfacing any recall.

Repo `/Users/dldmitry/tmp/OpenHarness`. Host for live checks: the 93.77 OpenHarness server (systemd-user).
Tests: `.venv/bin/python -m pytest`.
Defaults: model=gpt-5.6-sol reasoning=xhigh workdir=/Users/dldmitry/tmp/OpenHarness timeout=45m

## What Phase 2 delivers (and what it does NOT)
- ✅ the **OS-attested tool sandbox** (`tools_confined`'s hard boundary) + **positive recursive allowlist**;
  the **fail-closed confidentiality-gate ENFORCEMENT**; **opt-in user-visible honcho derived recall**; the
  default curated-authority switch **file→catalog**; **conversation-learning** (messages→deriver) +
  **dreamer** (owner-scoped, curated excluded); **multi-sender**.
- ⛔ **Not** (→ Phase 3): honcho **owning identity** (D6-A), the honcho-backed **default**, and file-backend
  **removal** — all deferred behind reconciliation-at-scale + DR.

## Global constraints (restate per step)
- The **OS-attested tool sandbox is the hard boundary**: **NO** user-visible recall lands before
  `p2-tool-sandbox` + `p2-gate-enforcement` are green. The gate is **fail-closed** — any missing conjunct ⇒
  no memory surfaced.
- The honcho JWT + catalog remain ONLY in the separate-UID memory service; **never** constructed in the gateway.
- **Every step re-passes the Phase-4 benchmark** (regression engine) before merge.
- Agents code + tests only; the **orchestrator** does the host hardening (systemd/landlock), git, deploy, canary.

---

## Step: p2-tool-sandbox — OS-attested tool sandbox + positive allowlist
**What to read first:** how ohmo currently spawns the tool-execution loop; the memory-service socket auth
from `p1-memory-service`. ADR §4.4. Host = the 93.77 OpenHarness server (systemd-user).
**What:** run the tool loop under a **confined principal distinct from BOTH the gateway and memory-service
UIDs**, with **OS-attested confinement** (Linux: landlock + seccomp + systemd hardening —
`ReadOnlyPaths`/`InaccessiblePaths` covering the catalog + JWT, `NoNewPrivileges`, `PrivateTmp`) so a
compromised tool cannot read the catalog/JWT or reach the memory-service socket except through the
sanctioned RPC. A **positive recursive allowlist** (default-deny) of paths/endpoints the confined principal
may touch. Expose an attestation signal `tools_confined: bool` the gate reads.
**Files:** the sandbox launcher + allowlist config + a systemd hardening unit template + tests.
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_tool_sandbox.py -q` **+ orchestrator real
confinement smoke on 93.77** (a probe tool reading the catalog path / opening the JWT is **DENIED**; an
allowlisted op succeeds). `tools_confined` defaults **FALSE** until attested.
**NOTE:** genuinely hard, host-specific; may need orchestrator infra iteration.

## Step: p2-gate-enforcement — fail-closed confidentiality gate
**What:** turn Phase-0's *computed* identity into an *acting* gate. Recall is surfaced ONLY when
`is_canonical_owner ∧ is_trusted_private_chat ∧ principal_isolated_session ∧ tools_confined` are **all**
true; any false ⇒ **fail-closed** (no recall, structured log). `principal_isolated_session` = session key
provably bound to the canonical principal; `tools_confined` = the attested signal from `p2-tool-sandbox`.
**Files:** the gate module + wiring in the memory service + `tests/test_ohmo/test_gate_enforcement.py`.
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_gate_enforcement.py -q`
(4-conjunct truth table: only all-true surfaces memory; each single false ⇒ nothing surfaced).

## Step: p2-visible-recall — opt-in user-visible honcho derived recall
**What:** when the gate is green **AND** the owner opted in (config), blend honcho **derived** hits into the
memory block shown to the agent — clearly **LABELLED** curated vs derived — under the composite budget.
Reuses the Phase-1 shadow query path, now **surfaced**. First time honcho output reaches the model.
**Files:** the blended render in the memory service + `tests/test_ohmo/test_visible_recall.py`.
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_visible_recall.py -q` **+ re-pass the Phase-4
benchmark** (gate green + opt-in ⇒ labelled derived hits present & budgeted; gate red ⇒ absent).

## Step: p2-default-catalog — switch curated authority file→catalog
**What:** flip `GatewayConfig.memory_backend` default `file→catalog` (gated on Phase-1 migration + a clean
reconciliation check). Re-enable `/memory` + `/dream` on the **catalog** kind (they operate on the catalog
now, reversing the `p1`/Phase-0 honcho-disable for this kind). One-time file→catalog migration on first boot.
**Files:** config default + command re-enablement + `tests/test_ohmo/test_default_catalog.py`.
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_default_catalog.py tests/test_commands -q`
(default is catalog; `/memory` ops hit the catalog; migration idempotent).

## Step: p2-conversation-learning — messages→deriver + dreamer (owner-scoped)
**What:** on owner turns, **ingest** the conversation into honcho `messages` (async, off-path) so the
deriver learns; enable the **dreamer** on the owner's **DERIVED** collection only — the **curated mirror is
NEVER dreamed**. Ingestion never blocks the turn.
**Files:** the ingestion hook + config + `tests/test_ohmo/test_conversation_learning.py`.
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_conversation_learning.py -q` **+ re-pass the
benchmark** (ingestion async & non-blocking; the curated collection is excluded from dreaming; recall
improves across turns).

## Step: p2-multi-sender — beyond the single owner
**What:** generalize `owner_principals` + per-principal isolation so each trusted principal gets an isolated
session + isolated honcho tenant + independent gate evaluation. No cross-principal leak.
**Files:** multi-principal routing + `tests/test_ohmo/test_multi_sender.py`.
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_multi_sender.py -q`
(two principals ⇒ isolated memory; the gate is evaluated per principal).

---

## Exit state
The owner sees **gate-gated, labelled** honcho recall; the **catalog is the default curated authority**;
conversation-learning is live. Identity/tenancy remain **local** (the honcho-owns-identity switch is Phase 3).

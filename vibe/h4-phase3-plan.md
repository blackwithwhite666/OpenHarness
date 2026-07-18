# Plan: h4-phase3 — terminal switch (D6-A: honcho owns identity)

**Status: DESIGN ONLY — not executed.** The **deferred terminal fork**. Lands ONLY after Phase 2 is stable,
the Phase-4 benchmark holds, **reconciliation-at-scale** is zero-orphan, and a **DR export** is proven.
Source ADR: `/Users/dldmitry/agents-playgroud/adrs/honcho-openharness-backend.md` (rev 6; D6-A).

Repo `/Users/dldmitry/tmp/OpenHarness`. Tests: `.venv/bin/python -m pytest`.
Defaults: model=gpt-5.6-sol reasoning=xhigh workdir=/Users/dldmitry/tmp/OpenHarness timeout=45m

## What Phase 3 delivers (and what it does NOT)
- ✅ **reconciliation-at-scale** over the full owner store; a **DR export/restore** path (honcho is NOT a
  backup); the **D6-A switch** (honcho owns identity, honcho-backed default); the **file-backend decommission**.
- ⛔ **Not**: anything that lands before reconciliation + DR are green.

## Global constraints (restate per step)
- honcho becomes **load-bearing for identity** here → a **cold-restore of the authority MUST be proven
  first** (`p3-dr-export`) and reconciliation MUST be **zero-orphan** (`p3-reconciliation-at-scale`).
- Agents code + tests only; the **orchestrator** runs the live reconciliation/DR/canary + git/deploy.
- Every step re-passes the Phase-4 benchmark.

---

## Step: p3-reconciliation-at-scale — full-store catalog↔honcho reconciliation
**What:** a reconciliation job over the **ENTIRE** owner memory verifying every curated catalog entry has an
**acknowledged** honcho mirror, and flagging orphans in **both** directions, with metrics. Must run
**zero-orphan** before any identity switch.
**Files:** the reconciliation job + report + `tests/test_ohmo/test_reconciliation_scale.py`.
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_reconciliation_scale.py -q` **+ orchestrator
real run** (seeded orphans detected; a clean store ⇒ zero unreconciled on the live owner store).

## Step: p3-dr-export — durable DR export + restore (authority cold-restore)
**What:** a periodic **export** of the catalog (the authority) to a durable **off-box** location + a
**restore** path. Because Phase 3 makes honcho load-bearing, a cold-restore of the authority must be proven.
**Files:** the export/restore tool + `tests/test_ohmo/test_dr_export.py`.
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_dr_export.py -q` **+ orchestrator schedules the
periodic export** (export→wipe→restore round-trip is byte-identical).

## Step: p3-honcho-identity — D6-A: honcho owns identity, default honcho-backed
**What:** the deferred fork — honcho becomes the **identity/tenancy authority**; the default backend
switches to **honcho-backed**. Lands ONLY after `p3-reconciliation-at-scale` + `p3-dr-export` are green and
the benchmark holds.
**Files:** config + wiring + `tests/test_ohmo/test_honcho_identity.py`.
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_honcho_identity.py -q` **+ orchestrator canary
on 93.77**.

## Step: p3-decommission-file — remove the file backend
**What:** after the honcho-backed default is stable, remove `FileMemoryBackend` + the `.md` path (retain the
migration tool for historical import).
**Files:** the deletion + test updates.
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo -q` (file backend gone; suite green).

---

## Exit state
honcho owns identity; the **catalog remains the local transactional authority**, mirrored to honcho; the
file backend is retired; the DR export is running. **End of the H4 arc.**

# Ohmo Eval Data Flywheel Completion TODO

This is the active completion checklist for the remaining Ohmo eval/data
flywheel work. Mark items done only after code/docs are updated and the
relevant targeted tests pass.

## P0

- [x] Rehydrate `EvalStore` SQLite lookup indexes from JSON/JSONL artifacts.
  - `evals.sqlite` is a rebuildable lookup index, not source of truth.
  - If the DB is missing, existing episode/event/embedding artifacts remain usable.

- [x] Detect stale review manifests before promotion.
  - Validate approved manifest rows against current draft cases.
  - Fail validation/promotion when case kind, episode id, facet counts, or tool names drift.

- [x] Freeze execution report schema contract.
  - Document statuses, check names, `observed_trace`, and privacy invariants.
  - Add tests that fail on accidental report-shape drift.

## P1

- [x] Add explicit output filenames for runnable eval artifacts.
  - Support A/B and CI runs without overwriting default pack/report files.

- [x] Add metadata-only eval artifact inspection commands.
  - Provide CLI review helpers for listing/showing cases without raw text.

- [x] Improve embedding/index operations for repeated local runs.
  - Expose rebuild/skip summary and content-hash reuse behavior.

- [x] Use embeddings and usage-graph signals in candidate mining.
  - Candidate score should explicitly reflect embedding availability and graph motifs.

- [x] Introduce an extensible scorer/judge contract for eval execution.
  - Keep deterministic exact-match scoring as the default scorer.

- [x] Cap resource snapshot directory aggregation.
  - Avoid unbounded recursive scans over large workspaces.

- [x] Add compact baseline promotion runbook examples.
  - Document an end-to-end capture-to-compare sequence.

## P2

- [x] Add optional machine-readable CLI summaries.
  - Prefer `--json` output for run, smoke, compare, baseline list, and review.

- [x] Revisit broader SQLite projections.
  - Explicitly keep them deferred unless documented trigger conditions become true.

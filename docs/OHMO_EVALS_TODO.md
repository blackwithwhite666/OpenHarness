# Ohmo eval data flywheel TODO

This checklist tracks the remaining stabilization work after the initial Ohmo
eval/data-flywheel skeleton, review/promote flow, replay-tools executor,
report comparison, baselines, and golden-path tests.

## Done

- [x] Workspace-local eval store for episodes, events, resources, and artifacts.
- [x] Usage graph, text facets, dense embedding materialization, and candidate mining.
- [x] Review manifests, manifest validation, and draft-to-gold promotion.
- [x] Runnable packs, smoke reports, execution reports, comparisons, and baselines.
- [x] Replay-tools executor with scripted and query-engine agent runners.
- [x] Metadata-only reports and golden-path privacy regression coverage.
- [x] Replay executor sync API works from async callers.
- [x] Public `ohmo evals run` CLI contract is covered for supported ids,
  blocked/error counts, report-only exits, and unknown executor/runner errors.

## Completed

- [x] P0: Rehydrate the SQLite lookup index from JSON/JSONL source of truth.
  - `EvalStore` documents JSON/JSONL as canonical, but lookups depend on
    `evals.sqlite`; deleting the lookup DB should not make a workspace blind.
  - Rebuild episode/event/embedding lookup rows from local artifacts without
    touching raw data outside the workspace.
  - Target files: `src/openharness/evals/store.py`,
    `tests/test_evals/test_store.py`.

- [x] P0: Detect stale review manifests before promotion.
  - Validate `episode_id`, `case_kind`, facet counts, and tool names against
    current draft cases, or add a draft fingerprint to review items.
  - Fail promotion if an approved item no longer matches the current draft.
  - Target files: `ohmo/evals/review.py`,
    `tests/test_ohmo/test_eval_review.py`.

- [x] P0: Freeze the execution report schema contract.
  - Document status names, check names, `observed_trace`, and privacy invariants.
  - Add schema-oriented tests that fail on accidental field/status drift.
  - Target files: `docs/OHMO_EVALS.md`, `tests/test_evals/test_runner.py`.

- [x] P1: Add explicit output filenames for runnable eval artifacts.
  - Support CI/A-B runs without overwriting `eval_pack.json`,
    `smoke_report.json`, or `eval_report.json`.
  - Target files: `ohmo/cli.py`, `ohmo/evals/pack.py`,
    `ohmo/evals/runner.py`, `tests/test_ohmo/test_cli.py`.

- [x] P1: Add eval artifact inspection commands.
  - Provide metadata-only `ohmo evals cases list/show` or equivalent review helpers.
  - Avoid raw prompt/tool/final text in all output.
  - Target files: `ohmo/cli.py`, `ohmo/evals/review.py`, `tests/test_ohmo/test_cli.py`.

- [x] P1: Improve embedding/index operations for repeated local runs.
  - Add explicit rebuild/skip summary and content-hash reuse notes.
  - Keep SQLite as a small lookup/index only until traffic or workspace size justifies more.
  - Target files: `src/openharness/evals/embeddings.py`,
    `src/openharness/evals/store.py`, `tests/test_evals/test_embeddings.py`.

- [x] P1: Use embeddings and usage-graph signals in candidate mining.
  - `ohmo evals embed` currently builds an index, while candidate scoring is
    mostly event/tool heuristics.
  - Add explicit selection/scoring signals without persisting raw text.
  - Target files: `src/openharness/evals/candidates.py`,
    `src/openharness/evals/embeddings.py`,
    `tests/test_evals/test_candidates.py`.

- [x] P1: Introduce an extensible scorer/judge contract for eval execution.
  - Exact final-text matching is good for replay smoke but too strict for
    query-engine model runs.
  - Keep default deterministic scoring while making future semantic judges explicit.
  - Target files: `src/openharness/evals/execution.py`,
    `tests/test_evals/test_runner.py`.

- [x] P1: Cap resource snapshot directory aggregation.
  - Avoid unbounded `rglob` over large workspaces; expose truncated metadata
    rather than raw paths or contents.
  - Target files: `ohmo/evals/resources.py`,
    `tests/test_ohmo/test_eval_resources.py`.

- [x] P1: Add compact runbook examples for baseline promotion.
  - Show a complete command sequence from capture to compare.
  - Include expected artifact paths and non-zero exit behavior.
  - Target file: `docs/OHMO_EVALS.md`.

- [x] P2: Add optional machine-readable CLI summaries.
  - Consider `--json` for run, smoke, compare, baseline list, and review.
  - Keep default human output stable.
  - Target files: `ohmo/cli.py`, `tests/test_ohmo/test_cli.py`.

- [x] P2: Revisit broader SQLite projections only if the trigger conditions in
  `docs/OHMO_EVALS.md` become true.
  - Candidate triggers: large workspaces, slow review filters, concurrent writers,
    or repeated embedding lookup cost.
  - Target files: `src/openharness/evals/store.py`, `docs/OHMO_EVALS.md`.

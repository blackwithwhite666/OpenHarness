# GAIA eval harness (`tests/eval/gaia/`)

A thin **measurement rig** that runs OpenHarness's `deep-research` sub-agent on a
fixed [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) validation
subset and scores the answer with the GAIA quasi-exact-match scorer.

Design contract: [`adrs/deep-research-openharness.md`](../../../../agents-playgroud/adrs/deep-research-openharness.md)
(§2½ architecture, §4 eval harness, §8 first PR).

## Three layers — and which one this is (ADR §2½)

"Harness" is overloaded. There are three distinct things:

1. **OpenHarness** — the framework/runtime (`src/openharness/{engine,tools,…}`).
2. **The deep-research capability** — *not a separate system*; it is an
   `AgentDefinition(name="deep-research", …)` **inside** OpenHarness, invoked in
   prod by the gateway. The product.
3. **This GAIA eval harness** — a **test driver** in `tests/eval/gaia/`. It
   *runs* the agent on GAIA tasks and *scores* the answer. A measurement rig,
   not a runtime; it does not exist inside the running bot.

**Direction of dependency:** this harness imports/drives OpenHarness, **never**
the reverse. The single intersection point is OpenHarness's existing
headless-agent subprocess boundary (`agent_tool.py` →
`registry.get_executor("subprocess").spawn(config)` with `cwd` +
`subagent_type`) — the runner spawns the `deep-research` agent through the *same*
boundary the main agent uses for sub-agents.

```
EVAL path (this dir):
  pytest / runner
    └ loader: GAIA task → per-task cwd (+ copy attachment)
       └ runner: spawn(subagent_type="deep-research", cwd=<task dir>, prompt=<q>)
            └ … deep-research agent … → final answer (string)
       └ scorer (quasi-exact-match) → pass/fail + telemetry → REPORT.md
```

## How to run

Offline scorer tests (default pytest gate, **zero network, zero model**):

```bash
.venv/bin/python -m pytest tests/eval/gaia/test_scorer.py -q
```

The live-web runner is gated behind `@pytest.mark.eval` (see *Conventions*) and
is **excluded** from the default gate — it needs an `HF_TOKEN`, the gated GAIA
dataset, and live web/Serper access, none of which run in CI.

## What this PR (PR1 / M0, scorer-only) ships

- **`scorer.py`** — `question_scorer(model_answer, ground_truth, strict=…)`
  (number / list / string dispatch) + tolerant `extract_answer(text)`.
  Faithful port of the canonical HF leaderboard scorer; `strict=True` (default)
  is bit-for-bit official, `strict=False` is our EU-comma / units / safe-list
  extension (each divergence behind the flag, separately tested).
- **`test_scorer.py`** — adversarial cases (empty/None, reasoning-prefix, the
  EU-comma trio, comma-inside-a-list-element, trailing units, sentinel-absent
  extraction, commentary-wrapped, case/whitespace), the canonical reference
  pairs, OWL-derived fixtures, and the offline runner helpers
  (`wilson_ci`, JSONL/REPORT writers). Runs in the default gate.

## What is stubbed (pending HF access + the agent)

| Stub | Where | Why |
|---|---|---|
| GAIA snapshot download | `loader.py` `download_gaia_snapshot` / `load_validation_tasks` | GAIA is HF-gated (`HF_TOKEN` + accepted terms; un-gated call 401s — ADR §7). Not configured in PR1. |
| The deep-research agent spawn | `run_subset.py` `_spawn_deep_research` / `run_subset` | The `deep-research` `AgentDefinition` does not exist yet — it ships in PR3 / M1. The spawn call is a clearly-marked TODO at the exact subprocess boundary. |
| Real `task_id`s in the manifests | `dev.yaml` / `gate.yaml` | Populated from `loader.load_validation_tasks()` once HF access is configured (PR2). Current rows are SCHEMA placeholders fixing the column contract + intended level/capability spread. |

**Real and tested now** (offline): the scorer, the answer extractor, and the
runner's `wilson_ci` / `aggregate` / `write_jsonl` / `write_report` / `score_run`
helpers.

## The `@pytest.mark.eval` convention

- **Scorer tests** (`test_scorer.py`) are **unmarked** → they run in the default
  `pytest` gate. They are pure (no network/model), so they belong in CI.
- **The live-web runner** (`run_subset.py`, once implemented) is
  **`@pytest.mark.eval`** → excluded from the default gate; run on demand at
  milestone boundaries (`pytest -m eval`, with `HF_TOKEN` + web access). The
  `eval` marker is registered in `pyproject.toml` `[tool.pytest.ini_options]`
  `markers` so an unmarked run never accidentally hits the network.

## Splits (ADR §1, §4)

Two **disjoint**, fixed-`task_id` manifests, 30 rows each
(L1=10 / L2=15 / L3=5):

- **`dev.yaml`** — seen during prompt tuning; iterate here.
- **`gate.yaml`** — held out; run only at milestone boundaries (primary metric:
  gate-split accuracy, per level + overall, **Wilson 95% CIs**; gate passes only
  when the *lower* CI bound clears the bar). **Gate on L1+L2; L3 tracked, never
  gated** (the L3 wall). Per-split capability coverage: ≥2 web-browsing,
  ≥2 multi-hop, ≥2 file-attachment, ≥1 multimodal, ≥1 pure-reasoning.

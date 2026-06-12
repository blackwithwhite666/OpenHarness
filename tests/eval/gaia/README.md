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

## What ships (PR1 scorer + PR2 loader/runner/manifests)

- **`scorer.py`** (PR1) — `question_scorer(model_answer, ground_truth, strict=…)`
  (number / list / string dispatch) + tolerant `extract_answer(text)`.
  Faithful port of the canonical HF leaderboard scorer; `strict=True` (default)
  is bit-for-bit official, `strict=False` is our EU-comma / units / safe-list
  extension (each divergence behind the flag, separately tested).
- **`loader.py`** (PR2) — `download_gaia_snapshot(token)` (the one HF-gated
  network call; `huggingface_hub` imported lazily) + offline
  `load_validation_tasks(snapshot_root) -> list[Task]` reading
  `2023/validation/metadata.jsonl`, skipping `0-0-0-0-0`, absolutizing
  `file_name`. `Task = {task_id, question, answer, level, file_path|None}`.
- **`run_subset.py`** (PR2) — runnable
  `python -m tests.eval.gaia.run_subset --split {dev,gate} --agent <subagent_type> --k 3`.
  Per task: per-task working dir + attachment copy (ADR §2(c)), spawn via the
  REAL subprocess boundary (`get_backend_registry().get_executor("subprocess").spawn(TeammateSpawnConfig(...))`
  with a PINNED `model`, never `inherit` — ADR §7), poll to terminal,
  `read_task_output`, K=3, score, write per-sha JSONL + `REPORT.md` (per-level
  acc + Wilson CIs + median tokens/latency + extraction-/infra-failure rates,
  tagged distinctly). For M0 `--agent` is the current general worker
  (`general-purpose`); `deep-research` ships in PR3.
- **`build_manifests.py`** (PR2) — stratified disjoint dev+gate selection
  (30 each, L1=10/L2=15/L3=5) by a FIXED seed →
  `python -m tests.eval.gaia.build_manifests --snapshot <root>`.
- **`test_scorer.py` / `test_loader.py` / `test_build_manifests.py` /
  `test_runner.py`** — all offline (zero network/model/subprocess). The loader
  is tested against a FAKE snapshot dir; the selector against a SYNTHETIC task
  list; the runner end-to-end by monkeypatching the spawn to return canned
  transcripts (good / sentinel-less / spawn-failure) and asserting the REPORT
  math + failure tagging. Run in the default gate.

## What still needs HF_TOKEN / a live model (deferred to a real run)

| Deferred | Where | Why |
|---|---|---|
| GAIA snapshot download | `loader.download_gaia_snapshot` | HF-gated (`HF_TOKEN` + accepted terms; un-gated call 401s — ADR §7). Gated behind the token; never invoked by the unit suite. |
| Real `task_id`s frozen into the manifests | `build_manifests.main` → `dev.yaml` / `gate.yaml` | Needs the dataset to enumerate tasks. Current manifest rows are SCHEMA placeholders (`task_id: TODO`) until `build_manifests` is run with a token. |
| The actual agent run (spawn → live model) | `run_subset._spawn_agent` / `run_subset.main` | Spawning the worker invokes a live model (`ANTHROPIC_API_KEY`) and live web/Serper. Offline tests inject `spawn_fn`; the M0 baseline `REPORT.md` is produced by a real `run_subset` run. |
| `deep-research` AgentDefinition | `src/openharness/coordinator/agent_definitions.py` | Ships in PR3 / M1. M0 runs `--agent general-purpose`. |

**Real and tested now** (offline): the scorer, the answer extractor, the loader
parsing/skip/absolutize, the manifest selector, and the runner's K=3
orchestration + `wilson_ci` / `aggregate` / `write_jsonl` / `write_report` /
`score_run` / `prepare_task_dir` / `build_prompt` helpers.

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

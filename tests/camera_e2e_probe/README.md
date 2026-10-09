# Camera E2E probe

This probe joins the Telegent Camera fixture/client, the OpenHarness Camera listener and runtime, an isolated official Honcho API, and Telegent's nutrition projection. The offline mode uses fixture clients and synthetic data. Native mode is a separately authorized acceptance run using the configured Codex subscription and a separate virtual user client. Neither mode is a food-estimate quality claim.

## Current setup

Use two clean linked worktrees: one for this OpenHarness checkout and one for Telegent. Set each expected SHA to the exact 40-character commit checked out in that worktree. Use the existing pinned official Honcho checkout; the Compose file builds it and verifies the source SHA. The examples below use placeholders for local paths, selected SHAs, private source inputs, and the dynamically mapped API port.

```sh
export CAMERA_OPENHARNESS_WORKTREE=/path/to/clean/openharness-worktree
export CAMERA_TELEGENT_WORKTREE=/path/to/clean/telegent-worktree
export CAMERA_HONCHO_SOURCE=/path/to/official-honcho-at-pinned-sha
export CAMERA_OPENHARNESS_SHA=REPLACE_WITH_OPENHARNESS_40_HEX_COMMIT
export CAMERA_TELEGENT_SHA=REPLACE_WITH_TELEGENT_40_HEX_COMMIT
export CAMERA_ACCEPTANCE=1
export CAMERA_HONCHO_SOURCE_SHA=0d85a34917418485b9eb387eacda608bcf2de426
export CAMERA_SCRATCH="$CAMERA_OPENHARNESS_WORKTREE/tmp/camera-normal-chat-docker"
export CAMERA_PROJECT=camera-e2e-unique-run-name

test "$(git -C "$CAMERA_OPENHARNESS_WORKTREE" rev-parse HEAD)" = "$CAMERA_OPENHARNESS_SHA"
test -z "$(git -C "$CAMERA_OPENHARNESS_WORKTREE" status --porcelain)"
test "$(git -C "$CAMERA_TELEGENT_WORKTREE" rev-parse HEAD)" = "$CAMERA_TELEGENT_SHA"
test -z "$(git -C "$CAMERA_TELEGENT_WORKTREE" status --porcelain)"
test "$(git -C "$CAMERA_HONCHO_SOURCE" rev-parse HEAD)" = "$CAMERA_HONCHO_SOURCE_SHA"

docker compose -p "$CAMERA_PROJECT" -f "$CAMERA_OPENHARNESS_WORKTREE/tests/camera_e2e_probe/compose.yaml" up -d --build --wait --wait-timeout 900
docker compose -p "$CAMERA_PROJECT" -f "$CAMERA_OPENHARNESS_WORKTREE/tests/camera_e2e_probe/compose.yaml" port api 8000
# Copy the numeric port printed above into CAMERA_API_PORT.
export CAMERA_API_PORT=REPLACE_WITH_MAPPED_API_PORT
export CAMERA_HONCHO_URL="http://127.0.0.1:$CAMERA_API_PORT"
curl -fsS "$CAMERA_HONCHO_URL/health"
```

Compose starts the official Honcho API, PostgreSQL/pgvector, and Redis. PostgreSQL and Redis use project-scoped volumes; the API is bound to a dynamic loopback port. Honcho embeddings and its deriver are disabled. Keep the project and its volumes after a run; do not run `down -v` or delete probe data.

Run the offline functional probe and storage checks from the OpenHarness worktree's existing locked test environment. Substitute the mapped API port reported by Compose.

```sh
cd "$CAMERA_OPENHARNESS_WORKTREE"
export CAMERA_RUN_MODE=offline
CAMERA_TELEGENT_WORKTREE="$CAMERA_TELEGENT_WORKTREE" \
  CAMERA_OPENHARNESS_SHA="$CAMERA_OPENHARNESS_SHA" \
  CAMERA_TELEGENT_SHA="$CAMERA_TELEGENT_SHA" \
  CAMERA_ACCEPTANCE=1 CAMERA_RUN_MODE=offline CAMERA_HONCHO_URL="$CAMERA_HONCHO_URL" \
  .venv/bin/python tests/camera_e2e_probe/run_joined.py
CAMERA_HONCHO_URL="$CAMERA_HONCHO_URL" .venv/bin/python tests/camera_e2e_probe/run_honcho.py
CAMERA_TELEGENT_WORKTREE="$CAMERA_TELEGENT_WORKTREE" \
  CAMERA_HONCHO_URL="$CAMERA_HONCHO_URL" .venv/bin/python tests/camera_e2e_probe/run_storage_chain.py
```

The offline run checks fixture upload through the production Telegent Camera submission client, native photo and prompt receipts, an ordinary owner text turn, selected image loading, trace finalization, one durable Honcho meal, and Telegent's current meal and wellness intake. A repeated owner message is checked for zero new events and photos; the current runtime returns a stale-event gateway error for that repeated message. Set `CAMERA_CORRECTION_JOIN=context-items-date` to exercise sparse item and date corrections with runtime reconstruction, or `CAMERA_CORRECTION_JOIN=portion-denial` to exercise an immutable denial. Set `CAMERA_RESTART_JOIN=two-photo` to check that a late old-source replay leaves the newer photo's attention intact. Each run creates unique synthetic Honcho workspace and session IDs. It does not call a model or Dropbox/Telegram service. `run_storage_chain.py` writes synthetic observation/correction fixtures to the isolated Honcho API and checks Telegent sync and wellness reads. `run_honcho.py` checks a synthetic nonmeal Honcho message.

The direct wellness-helper reads in `run_storage_chain.py` and in `run_joined.py`'s before-answer and finalizer checks use an explicit synthetic self scope: participant `123` maps to the probe's synthetic owner. Each read installs that authorization only for the call and resets it in `finally`. These are in-process storage/helper checks; they do not establish HTTP or MCP wire authorization. Signed wire authorization has separate focused client and verifier tests.

## Native subscription acceptance

For native acceptance, configure the existing Codex subscription profile for `gpt-6-luna` with medium reasoning and bind its settings directory read-only. Use a separate native subscription client for the virtual user. `run_joined.py` rejects provider fallback and checks the subscription profile, model, medium effort, and two distinct clients. Also use a lead-selected clean OpenHarness/Telegent pair. Supply the owner scenario and an approved bounded JPEG with its matching SHA-256. The source JPEG must be Git-ignored and inside the OpenHarness worktree.

```sh
export CAMERA_RUN_MODE=native
export CAMERA_NATIVE_CONFIG_DIR=/read-only/path/to/existing/native-settings
export CAMERA_USER_SCENARIO='Describe the synthetic or approved meal and confirm the offered portion.'
export CAMERA_SOURCE_JPEG="$CAMERA_OPENHARNESS_WORKTREE/tmp/<approved-private-image>.jpg"
export CAMERA_SOURCE_SHA256=REPLACE_WITH_LOWERCASE_SHA256

CAMERA_TELEGENT_WORKTREE="$CAMERA_TELEGENT_WORKTREE" \
  CAMERA_OPENHARNESS_SHA="$CAMERA_OPENHARNESS_SHA" \
  CAMERA_TELEGENT_SHA="$CAMERA_TELEGENT_SHA" \
  CAMERA_ACCEPTANCE=1 CAMERA_RUN_MODE=native \
  CAMERA_NATIVE_CONFIG_DIR="$CAMERA_NATIVE_CONFIG_DIR" \
  CAMERA_USER_SCENARIO="$CAMERA_USER_SCENARIO" \
  CAMERA_SOURCE_JPEG="$CAMERA_SOURCE_JPEG" \
  CAMERA_SOURCE_SHA256="$CAMERA_SOURCE_SHA256" \
  CAMERA_HONCHO_URL="$CAMERA_HONCHO_URL" \
  .venv/bin/python tests/camera_e2e_probe/run_joined.py
```

The native run uses the already configured Codex subscription and a distinct `gpt-6-luna` virtual user client to produce an ordinary owner text turn from the supplied scenario and visible prompt. Do not add OpenRouter, API-key fallback, or other provider credentials. The runner validates the exact source pair before and after the run, captures server-returned event identity, checks trusted Camera source/capture bindings, and verifies the current meal and wellness projection. Probe outputs and local projections stay under the OpenHarness worktree's ignored `tmp/camera-normal-chat-docker/` tree. Preserve those outputs and existing service volumes for review.

## Historical evidence

Results from September 2026 are historical prototype evidence, not current acceptance. Earlier offline checks exercised the synthetic Telegent-to-Ohmo upload and Honcho-to-Telegent storage chain. A bounded native Luna run also exercised a Camera owner answer and exact event projection. Separate earlier runs exposed missing capture-time propagation and missing finalizer annotations; those failures led to later repairs. None of these historical runs establishes current source-pair acceptance, a clean database, or nutrition estimate accuracy. Consult the retained private run artifacts through the lead; do not copy private event identifiers, source paths, images, or dialogue into this public README.

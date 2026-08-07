# Nutrition ingest runbook

This runbook operates the OpenHarness consumer of Telegent's synchronized
`nutrition-assets` artifacts. The consumer is a Marina-only confirmation
adapter. It verifies artifacts and records the answer; it never reclassifies
images. Producer CLIP remains shadow until the separate recall/shadow gate is
accepted, and the consumer never runs Qwen or CLIP.

## Configuration validation

1. Copy the non-secret shape from `deploy/server/gateway.example.json` into
   the service workspace. Replace every `${...}` placeholder through the
   deployment secret/config mechanism. Do not put tokens, private identifiers,
   usernames, replies, or paths in logs or documentation.
2. Keep `nutrition_ingest.enabled=false` until all values are replaced and the
   synchronized root exists, is owner-only, and contains only the expected
   protocol tree. The principal, private chat id, and `telegram:<principal>`
   session key must be the same positive numeric Marina binding. The tenant is
   exactly `marina`; usernames are not authorization. The numeric
   `family_principals` key in the example is a synthetic shape-only placeholder,
   not an authorized account; replace it together with the `${...}` binding
   values before enabling the feature.
   The Marina `tenant_honcho` binding also owns the Honcho reader/writer
   session (default `ohmo`) and observed peer. Do not derive the Honcho session
   from the Telegram session key; reader and writer must use the same binding.
3. Validate the exact runtime model before restart:

   ```bash
   .venv/bin/python -c \
     'from ohmo.gateway.config import load_gateway_config; load_gateway_config()'
   systemctl --user is-active ohmo-gateway.service
   systemctl --user show ohmo-gateway.service -p ActiveState -p SubState
   ```

   If the local CLI exposes a different config-validate spelling, use that
   existing command, not a new configuration key. A failed validation is a
   hard stop.

## Deploy and Dropbox preflight

Use the existing deploy-branch workflow and approved restart path. Review the
dirty worktree before any update; preserve unrelated implementation changes.
Before an operator deploys, inspect the dirty state without resetting or
rewriting concurrent work:

```bash
git status --short
git diff --check
git log -1 --oneline
```

Only the reviewed deploy-branch change may be installed.
Before starting or restarting the gateway, require the Dropbox daemon/service
that creates synchronized files on the consumer host to run with a systemd
drop-in setting `UMask=0077`. As a one-time prerequisite, harden the existing
`nutrition-assets` tree to owner-only permissions and verify that no group or
other permission bits remain, for example:

```bash
chmod -R go-rwx "$NUTRITION_ASSETS_ROOT"
find "$NUTRITION_ASSETS_ROOT" -perm /go-rwx -print -quit
```

Setting the root mode once is insufficient: future synchronized children
inherit the Dropbox daemon's umask.

After deployment, verify the Dropbox selective sync for the synchronized
`nutrition-assets` root, owner-only permissions, free space, and service
readiness. Do not recursively print the tree. The scanner ignores temporary
objects and accepts only a manifest whose named image has the exact declared
size and SHA-256.

Inspect aggregate status with the privacy-safe command:

```bash
.venv/bin/python -m ohmo nutrition-ingest status --workspace "$OHMO_WORKSPACE"
```

The output may contain bounded states and counts only. Select an operator-known
candidate locally for replay; never copy its id into logs, chat, or metrics.

Before each confirmation attempt, the consumer fail-closed checks the inclusive
last-seven-day window of the configured Marina Honcho session. Only
gateway-owned Marina user metadata participates. Exact SHA-256 matches and
same-algorithm DCT pHash matches (`dct-phash-16x16-v1`, Hamming distance <= 4)
become terminal `seen` results. SHA-256 covers both ordinary Marina Telegram
user messages and trusted `dropbox_camera` observations. DCT pHash is limited
to ordinary Telegram messages and requires the candidate's authoritative EXIF
capture time to be within two hours of the Honcho message creation time; a
Dropbox observation is exact-SHA-only. SHA-256 is checked first, and the old
aHash algorithm is never compared with the DCT hash. A failed or partial
Honcho read leaves the candidate unsent and retryable.

The consumer writes owner-only, atomic suppression tombstones below `_seen`.
Once authoritative normalized EXIF is strictly older than seven days, it writes
the tombstone before deleting a verified direct-child candidate directory. The
exact seven-day boundary is retained. This expiry is unconditional, including
pending or ambiguous Telegram delivery and incomplete consumed estimation;
the compact tombstone preserves the final local state summary for audit.

At the beginning of each enabled poll, under the coordinator lock, the consumer
also removes only direct-child directories matching the legacy producer staging
name `.dropbox-camera-v1-<64 lowercase hex>-<8 lowercase alphanumeric or
underscore>` when their filesystem mtime is strictly older than one hour. It
uses the same direct-child, real-directory, containment, and root-fsync checks
as candidate deletion. These raw staging directories are deleted without
writing `_seen` tombstones. Any inspection or deletion failure stops the poll
before an outbound prompt; files, symlinks, malformed names, reserved
directories, valid candidate directories, and staging at the one-hour boundary
are preserved.

## Rollout order

1. **Disabled:** producer discovery and consumer notifications are disabled.
   Validate config, sync, schema parity, service health, and no outbound
   messages.
2. **Shadow:** run the producer with both `--dropbox-camera-mode=shadow` and
   `--dropbox-camera-clip-mode=shadow`. Qwen remains authoritative; CLIP is
   diagnostic and the consumer does not classify. Compare aggregate shadow
   counts, privacy checks, and idempotency evidence.
3. **Publish canary:** after shadow acceptance, enable the consumer for the
   fixed Marina binding only. One ready positive artifact produces one native
   Telegram photo with the two-line caption `Вы это съели?` followed by
   `Дата: DD.MM.YYYY HH:MM (по EXIF фото)` and exactly these buttons, in order:
   `Да, я это съела`, `Нет, не ела`, `Это не еда`.
4. **Normal publish:** expand only after the canary proves exact routing,
   confirmation-before-estimation, authoritative EXIF meal timestamps after
   `Да`, durable observation reconciliation, and
   restart idempotency. Keep CLIP shadow until its recorded recall/shadow gate
   is accepted; never make the consumer a second classifier.

## Confirmation state machine

- A verified candidate is queued in deterministic discovery order. There is at
  most one pending Marina confirmation.
- `Нет, не ела` records `not_consumed`, emits only the existing acknowledgement,
  and advances the queue. `Да, я это съела` records `consumed` and only then invokes
  the existing nutrition estimator, producing exactly one consumed observation under its
  stable operation id. After the observation is durably committed, Marina
  receives one concise reply such as `КБЖУ: 550 ккал · Б 30 г · Ж 20 г · У 45 г`
  followed by a short note that the photo-based portion is uncertain. The
  numbers come from the validated schema-v2 annotation; model prose is not
  parsed. Missing, negative, non-finite, boolean, or otherwise invalid calorie
  or macro values fail closed and are retried without sending a misleading
  result.
- `Это не еда` is an audited terminal `non_food` sidecar outcome. It retains
  the Marina recipient, prompt, and reply binding, creates no `meal_observation`,
  produces no KBJU result, and never enters estimation. A native confirmation
  callback is accepted only when its exact callback data and message id bind to
  the current prompt. Plain text and ordinary Marina chat pass through the
  normal gateway path. A bound `nutrition:` callback with a wrong/old candidate
  prefix, wrong/missing option, or no pending candidate is swallowed without
  mutation. For compatibility, a legacy `ask:` callback is swallowed only when
  its native message id exactly matches the current nutrition prompt; unrelated
  `ask:` callbacks pass through to normal Ohmo.
- While an ordinary Marina chat turn is in flight, the coordinator pauses
  publication of the next confirmation. The gateway sends an explicit start
  lifecycle event and releases it on completion, exception, or cancellation.
  The queue remains sequential and has no hourly throttle.
- After `Да, я это съела`, the trusted prompt requires inspection of visible
  pixels before estimation. Packaging, labels, menus, advertisements without
  edible contents, residue/smears or an empty dish, a face/person without food,
  plain water or a non-caloric drink alone, and context-only/off-camera food
  must return exactly `{"code":"no_visible_consumable_portion"}`. The trusted
  runtime validates that exact response, skips the Honcho append, records a
  terminal `non_food` sidecar outcome with no observation or KBJU, and sends a
  concise acknowledgement that nothing was recorded. A malformed response or
  missing annotation remains a retryable fail-closed error.
- An ambiguous prompt is quarantined as `delivery_unknown`. It is not resent
  automatically and does not block later eligible candidates, while the
  coordinator still permits only one actual `pending_confirmation` prompt.
- Pending latency, confirmation outcome, retry/dead-letter, delivery-unknown,
  duplicate suppression, and end-to-end latency are aggregate metrics with
  bounded stage/error labels only.

## Delivery-unknown and recovery

Exactly-once processing here means stable candidate and Honcho operation ids,
durable state transitions, and suppression of already-seen media. It does not
make Telegram transport exactly-once. Telegram can accept a prompt while its
receipt is lost, so prompt delivery remains inherently ambiguous.

The КБЖУ reply is marked completed before its outbound Telegram send, with a
stable per-candidate `:summary:v1` operation id and the confirmation photo's
native message id when the channel supports reply binding. This preserves
at-most-once completion ordering but intentionally does not invent exactly-once
Telegram transport: a crash or send failure after local completion can leave
the user-facing summary undelivered, while a channel-side acceptance followed
by a lost receipt can leave delivery ambiguous. Reconcile that residual
ambiguity from local sidecar/channel evidence; do not automatically resend a
completed summary.

If a prompt send has no unambiguous single-message receipt, the result sidecar
enters `delivery_unknown`. Quarantine it without automatic resend so later
eligible candidates can proceed; the native callback guard prevents a callback
for the quarantined prompt from confirming another candidate. Reconcile the
channel receipt and sidecar locally, then either acknowledge the existing
prompt or use the operator replay command. A retryable error may use capped
exponential backoff; exhausted prompt or estimation attempts become a dead
letter.

```bash
.venv/bin/python -m ohmo nutrition-ingest replay --workspace "$OHMO_WORKSPACE" \
  "$OPERATOR_SELECTED_CANDIDATE"
```

The command must be invoked with an operator-selected value and prints only a
bounded JSON result (`{"replayed": true|false}`). Never include that value in
captured evidence.

On restart, a persisted `prompt_sending` is reconciled as
`delivery_unknown`, not resent. A persisted consumed result is reconciled by
stable Honcho/observation operation id, so a crash after remote commit cannot
create a second observation. A partial or mismatched image remains invisible
to the consumer; repair sync or dead-letter it after bounded retries without
mutating the manifest.

## Evidence and rollback

Retain aggregate counts for verified candidates, pending/confirmation
outcomes, retries, dead letters, delivery-unknown states, duplicate
suppression, and latency. Evidence must not contain candidate ids, paths,
filenames, EXIF, image or reply content, usernames, owner labels, or
principal/chat/tenant/session identifiers. Reconcile detailed sidecars only on
the owner host and store them under the existing audit boundary.

To roll back, disable consumer notifications and producer discovery, stop the
service only through the approved systemd/deploy workflow, and restore direct
Qwen (`--dropbox-camera-clip-mode=disabled`) if producer classification must
continue. Do not delete synchronized images, manifests, result/error sidecars,
or audit events. Leave pending work replayable and require a fresh shadow and
Marina-only canary acceptance before re-enabling publish.

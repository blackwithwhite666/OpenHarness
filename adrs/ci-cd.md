# ADR: CI/CD for the ohmo gateway (self-hosted, idle-guarded)

- Status: accepted
- Date: 2026-07-01
- Related:
  - [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml) — the workflow
  - [`ci/wait_for_idle.sh`](../ci/wait_for_idle.sh) — the idle guard
  - `.github/workflows/ci.yml` — pre-existing upstream CI (GitHub-hosted matrix on `main`/PRs); left untouched

## Context

The ohmo Telegram gateway runs as a user-systemd unit (`ohmo-gateway.service`)
on the agent host `93.77.160.211`, installed via **pipx** from this fork's
`deploy` branch. Rolling out a change was manual: merge into `deploy`, then on
the box `pipx install --force git+…@deploy` + `systemctl --user restart`.

Two constraints shape the automation:

- **The gateway is the deploy target and the only viable runner host.** The box
  egresses to OpenAI/Telegram through a mandatory OpenVPN tunnel, so the deploy
  (and the restart) must run *on the box*. The runner lives there.
- **A restart mid-turn loses the user's request.** `systemctl restart` SIGTERMs
  the active agent session; Telegram has already ACKed the update so it does not
  redeliver — the request is silently dropped and the user must resend. So a
  deploy must wait for the gateway to be idle before restarting.

## Decision

### Self-hosted runner on the box
Under `~/ci/actions-runner` on `93.77.160.211` (user `blackwithwhite`), a system
systemd service (`actions.runner.blackwithwhite666-OpenHarness.ohmo-agent-93-77`),
labels `self-hosted, Linux, X64`. Host already has python3.12, pipx, node18.

### Workflow (`deploy.yml`, branch `deploy`)
- Triggers: `push` and `pull_request` on `deploy`.
- `test` job: `python -m venv` → `pip install -e '.[dev]'` →
  `pytest -m "not eval" --ignore=tests/eval`. The `eval` marker tags the live-web
  GAIA runner; `tests/eval/` also needs `pyarrow` (an optional dep not installed
  in the gate). Everything else (~1998 tests) must pass. The suite already
  self-guards live paths via `skipif` (e.g. the upstream `/home/tangjiabin/…`
  path), so no extra skip list is needed.
- `deploy` job: `needs: test`, `if push && ref == refs/heads/deploy`:
  1. `pipx install --force git+…@deploy` — reinstalls from the pushed HEAD
     (the local checkout is only used for CI helper scripts).
  2. `ci/wait_for_idle.sh` — hold until no in-flight turn.
  3. `ci/restart_gateway.sh` stops the user unit, safely stops and waits for a
     matching detached workspace gateway, then starts `ohmo-gateway.service`
     and asserts `is-active`.
- `concurrency: deploy-${{ github.ref }}` + the single runner serialize
  `test`→`deploy`.

### Idle guard (`ci/wait_for_idle.sh`)
In-flight is detected from the user journal markers `ohmo runtime processing
start` vs `…complete`: a turn is open when, in a ~10-min window, `start` lines
outnumber `complete` lines. It polls every 10s up to `OHMO_IDLE_MAX_WAIT` (300s),
then **restarts anyway** — the gateway persists in-flight/buffered requests on
stop and notifies the user on startup (deploy `c3fd5f9`), so a forced restart
degrades gracefully rather than blocking the pipeline forever.

### `systemctl --user` from the runner
The runner is a *system* service, so `systemctl --user` / `journalctl --user`
need the user manager's socket: the deploy step exports
`XDG_RUNTIME_DIR=/run/user/$(id -u)` and
`DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus`. Lingering is enabled,
so `/run/user/1000` persists.

### Gate: auto-deploy on green `deploy`
No manual approval (GitHub Environment approval is unavailable on the private
free-plan repo, same as the sibling `agents-playgroud` setup). Deploy runs
automatically once `test` is green on a `deploy` push.

## Consequences

- A push/merge to `deploy` that passes tests redeploys the live gateway with no
  human step — treat `deploy` as releasable.
- No GitHub secrets: pipx pulls public source from GitHub; the runner's trust
  comes from running as `blackwithwhite` on the box.
- The idle guard narrows but doesn't eliminate the drop window (bounded wait +
  backstop). Anyone who can merge to `deploy` can restart prod.
- Rollback: `pipx install --force git+…@<previous-sha>` + restart (idle-guarded).

## Operations

- Runner: repo Settings → Actions → Runners, or `sudo systemctl status
  actions.runner.*` on the box; logs `journalctl -u actions.runner.*`.
- Re-register: fresh token from `gh api -X POST
  repos/blackwithwhite666/OpenHarness/actions/runners/registration-token`, then
  `~/ci/actions-runner/config.sh … --replace`.
- Manual deploy still works: `pipx install --force git+…@deploy` +
  `bash ci/wait_for_idle.sh` + `bash ci/restart_gateway.sh`.

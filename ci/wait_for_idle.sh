#!/usr/bin/env bash
# Wait until the ohmo Telegram gateway has no in-flight user turn before a
# restart. Restarting mid-turn sends SIGTERM, which cancels the active agent
# session; Telegram has already ACKed the update so it does NOT redeliver, and
# the user's request is silently lost. We therefore hold the deploy until the
# gateway is idle.
#
# In-flight detection from the user journal (markers verified live):
#   ohmo runtime processing start    ... session_id=X   (turn opened)
#   ohmo runtime processing complete ... session_id=X   (turn finished)
# A turn is open when, in a recent window, there are more "start" than
# "complete" lines. Bounded wait: after OHMO_IDLE_MAX_WAIT we restart anyway —
# the gateway persists in-flight/buffered requests on stop and notifies the user
# on startup ("⚠️ Я перезапустился…"), so a forced restart degrades gracefully.
set -euo pipefail

UNIT="ohmo-gateway.service"
MAX_WAIT="${OHMO_IDLE_MAX_WAIT:-300}" # seconds
INTERVAL="${OHMO_IDLE_INTERVAL:-10}"  # seconds
WINDOW="${OHMO_IDLE_WINDOW:-10 min}"  # journal look-back
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

_count() { # $1 = grep pattern -> number of matching journal lines in the window
  journalctl --user -u "$UNIT" --since "-${WINDOW}" --no-pager 2>/dev/null \
    | grep -cE "$1" || true
}

is_busy() {
  local starts completes
  starts=$(_count "ohmo runtime processing start")
  completes=$(_count "ohmo runtime processing complete")
  [ "${starts:-0}" -gt "${completes:-0}" ]
}

waited=0
while is_busy; do
  if [ "$waited" -ge "$MAX_WAIT" ]; then
    echo "idle-guard: still busy after ${MAX_WAIT}s — restarting anyway (backstop persists in-flight)."
    exit 0
  fi
  echo "idle-guard: a user turn is in-flight; waiting ${INTERVAL}s (elapsed ${waited}s)…"
  sleep "$INTERVAL"
  waited=$((waited + INTERVAL))
done
echo "idle-guard: gateway idle — safe to restart."

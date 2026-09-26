#!/usr/bin/env bash
# Stop both supported gateway ownership modes before starting the user unit.
# A detached `ohmo gateway run` may survive a systemd stop, while a stale PID
# file must never cause `ohmo gateway stop` to signal an unrelated process.
set -euo pipefail

UNIT="${OHMO_GATEWAY_UNIT:-ohmo-gateway.service}"
WORKSPACE="${OHMO_GATEWAY_WORKSPACE:-$HOME/.ohmo}"
PID_FILE="${OHMO_GATEWAY_PID_FILE:-$WORKSPACE/gateway.pid}"
MAX_WAIT="${OHMO_GATEWAY_STOP_MAX_WAIT:-60}"
INTERVAL="${OHMO_GATEWAY_STOP_INTERVAL:-1}"
START_DELAY="${OHMO_GATEWAY_START_DELAY:-3}"

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"
export PATH="$HOME/.local/bin:$PATH"

die() {
  echo "gateway-restart: $*" >&2
  exit 1
}

is_workspace_gateway_args() {
  local args="$1"
  [[ " $args " == *" -m ohmo gateway run "* ]] \
    && [[ " $args " == *" --workspace $WORKSPACE "* ]]
}

validate_pid_file() {
  [[ -e "$PID_FILE" ]] || return 0

  local pid args
  pid="$(<"$PID_FILE")"
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || die "refusing malformed PID file: $PID_FILE"

  # Never pass a PID-file value to the public stop command: its legacy lookup
  # could otherwise signal a PID that was reused after this check. A dead PID
  # can be discarded; a live PID must prove it is this workspace's gateway.
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    return 0
  fi
  args="$(ps -p "$pid" -o args= 2>/dev/null)" \
    || die "cannot inspect live PID $pid from $PID_FILE"
  is_workspace_gateway_args "$args" \
    || die "refusing unverified live PID $pid from $PID_FILE"
  rm -f "$PID_FILE"
}

workspace_gateway_pids() {
  local output pid args
  output="$(ps -eo pid=,args=)" || return 1
  while read -r pid args; do
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || continue
    if is_workspace_gateway_args "$args"; then
      printf '%s\n' "$pid"
    fi
  done <<< "$output"
  return 0
}

wait_for_workspace_gateway_exit() {
  local waited=0
  local pids
  while true; do
    pids="$(workspace_gateway_pids)" \
      || die "cannot inspect workspace gateway processes"
    [[ -z "$pids" ]] && return
    if ((waited >= MAX_WAIT)); then
      die "workspace gateway still running after ${MAX_WAIT}s (PIDs: ${pids//$'\n'/ })"
    fi
    echo "gateway-restart: waiting for detached workspace gateway (PIDs: ${pids//$'\n'/ })"
    sleep "$INTERVAL"
    waited=$((waited + INTERVAL))
  done
}

systemctl --user stop "$UNIT"
validate_pid_file
ohmo gateway stop --workspace "$WORKSPACE"
wait_for_workspace_gateway_exit
systemctl --user start "$UNIT"
sleep "$START_DELAY"
systemctl --user is-active "$UNIT"
journalctl --user -u "$UNIT" -n 15 --no-pager

#!/usr/bin/env bash
# Self-heal the headful Chrome/CDP backend that all browser-backed ohmo skills
# (afisha, maps --provider browser, auto, travel-browser, balancir, ...) depend on.
#
# kasmvnc-chrome.service is Type=forking and tracks Xvnc as its MainPID; Chrome is
# a grandchild launched by ~/.vnc/xstartup, so when Chrome dies but Xvnc stays up
# systemd considers the unit healthy and never restarts it. This watchdog instead
# checks CDP reachability directly and heals BOTH layers: Chrome (via the service)
# and the long-lived browser-cli daemon (which stays bound to a dead CDP client).
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
CDP="http://127.0.0.1:9222/json/version"

cdp_up() { curl -sf --max-time 5 "$CDP" >/dev/null 2>&1; }
daemon_attached() { browser-cli status 2>/dev/null | grep -q '"target_id": "'; }

if cdp_up; then
  # Chrome is reachable; make sure the persistent daemon is actually attached
  # (it can stay bound to a previously-dead CDP client after a Chrome restart).
  if ! daemon_attached; then
    logger -t cdp-healthcheck "CDP up but browser-cli detached — restarting daemon"
    browser-cli restart >/dev/null 2>&1 || true
  fi
  exit 0
fi

logger -t cdp-healthcheck "CDP :9222 unreachable — restarting kasmvnc-chrome.service"
systemctl --user restart kasmvnc-chrome.service || true
for _ in $(seq 1 24); do cdp_up && break; sleep 5; done
browser-cli restart >/dev/null 2>&1 || true
if cdp_up; then
  logger -t cdp-healthcheck "CDP recovered"
else
  logger -t cdp-healthcheck "CDP STILL down after restart"
fi

#!/usr/bin/env bash
# Stage 1: install display + browser stack on 93.77 (Ubuntu 24.04 noble).
# Idempotent. Logs to ~/provision_browser.log when run via setsid.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "apt update"
sudo apt-get update -y

log "base packages (fluxbox, dbus, ssl-cert, xvfb fallback)"
sudo apt-get install -y wget curl ca-certificates fluxbox dbus-x11 ssl-cert xvfb python3 jq

log "google-chrome-stable"
if ! command -v google-chrome-stable >/dev/null 2>&1; then
  wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb -O /tmp/chrome.deb
  sudo apt-get install -y /tmp/chrome.deb
fi
google-chrome-stable --version 2>&1 | head -1 || true

log "resolve latest KasmVNC noble amd64 .deb"
KURL=$(python3 - <<'PY'
import json,urllib.request
try:
    d=json.load(urllib.request.urlopen('https://api.github.com/repos/kasmtech/KasmVNC/releases/latest',timeout=30))
    c=[a['browser_download_url'] for a in d.get('assets',[]) if 'noble' in a['name'] and a['name'].endswith('amd64.deb')]
    print(c[0] if c else '')
except Exception as e:
    print('')
PY
)
log "kasmvnc url: ${KURL:-NONE}"
if [ -n "${KURL}" ]; then
  if ! command -v vncserver >/dev/null 2>&1; then
    wget -q "$KURL" -O /tmp/kasmvnc.deb
    sudo apt-get install -y /tmp/kasmvnc.deb
  fi
else
  log "WARN: could not resolve KasmVNC noble asset; will fall back to Xvfb"
fi
command -v vncserver >/dev/null 2>&1 && (vncserver -version 2>&1 | head -1) || log "vncserver not installed"

log "add $USER to ssl-cert group"
sudo adduser "$USER" ssl-cert >/dev/null 2>&1 || true

log "DONE stage1"

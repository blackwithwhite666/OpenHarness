#!/usr/bin/env bash
# Stage 2a: KasmVNC + Chrome configs, manual start + verify (no systemd yet).
set -uo pipefail
log(){ echo "[$(date +%H:%M:%S)] $*"; }

mkdir -p ~/.vnc ~/chrome-profile

log "write ~/.vnc/kasmvnc.yaml"
cat > ~/.vnc/kasmvnc.yaml <<'YAML'
network:
  protocol: http
  interface: 127.0.0.1
  websocket_port: 6080
  use_ipv4: true
  use_ipv6: false
  ssl:
    require_ssl: false
desktop:
  resolution:
    width: 1920
    height: 1080
  allow_resize: true
logging:
  log_writer_name: all
  log_dest: logfile
  level: 30
YAML

log "write ~/.vnc/xstartup"
cat > ~/.vnc/xstartup <<'XS'
#!/bin/bash
export DISPLAY=:1
unset SESSION_MANAGER
[ -x /usr/bin/dbus-launch ] && eval "$(dbus-launch --sh-syntax --exit-with-session)"
fluxbox >/tmp/fluxbox.log 2>&1 &
exec google-chrome-stable \
  --user-data-dir="$HOME/chrome-profile" \
  --remote-debugging-port=9222 \
  --remote-debugging-address=127.0.0.1 \
  --no-first-run --no-default-browser-check \
  --disable-dev-shm-usage --disable-gpu \
  --password-store=basic \
  --disable-features=Translate \
  --window-size=1920,1080 --start-maximized \
  about:blank
XS
chmod +x ~/.vnc/xstartup

log "create kasm web user (basic auth, rw)"
WEB_PW=$(openssl rand -base64 18 | tr -dc 'A-Za-z0-9' | head -c 18)
echo -e "${WEB_PW}\n${WEB_PW}\n" | kasmvncpasswd -u blackwithwhite -w -r ~/.kasmpasswd >/dev/null 2>&1
chmod 600 ~/.kasmpasswd
echo "WEB_PASSWORD=${WEB_PW}"

log "kill stale :1 + locks"
vncserver -kill :1 >/dev/null 2>&1 || true
rm -f /tmp/.X1-lock /tmp/.X11-unix/X1 2>/dev/null || true

log "start vncserver :1 (daemonized)"
vncserver :1 -geometry 1920x1080 </dev/null 2>&1 | tail -20

log "wait for chrome CDP"
for i in $(seq 1 20); do
  if curl -fs http://127.0.0.1:9222/json/version >/dev/null 2>&1; then break; fi
  sleep 1
done

log "=== verify ==="
echo "--- vncserver -list ---"; vncserver -list 2>/dev/null
echo "--- listening sockets (6080/9222) ---"; ss -ltnp 2>/dev/null | grep -E ":6080|:9222" || echo "none yet"
echo "--- chrome procs ---"; pgrep -af "google-chrome" | head -3 || echo "no chrome"
echo "--- CDP /json/version ---"; curl -fs http://127.0.0.1:9222/json/version 2>/dev/null | head -c 400; echo
log "DONE stage2a"

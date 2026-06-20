#!/usr/bin/env bash
# Fix: user-owned TLS cert so KasmVNC under systemd-user doesn't depend on the
# root-owned snakeoil key (ssl-cert group not present in the boot-time user manager).
set -uo pipefail
log(){ echo "[$(date +%H:%M:%S)] $*"; }

if [ ! -f "$HOME/.vnc/self.pem" ]; then
  log "generate user-owned self-signed cert"
  openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$HOME/.vnc/self.key" -out "$HOME/.vnc/self.pem" \
    -days 3650 -subj "/CN=localhost" 2>/dev/null
fi
chmod 600 "$HOME/.vnc/self.key"

log "rewrite ~/.vnc/kasmvnc.yaml with user-owned cert"
cat > "$HOME/.vnc/kasmvnc.yaml" <<YAML
network:
  protocol: http
  interface: 127.0.0.1
  websocket_port: 6080
  use_ipv4: true
  use_ipv6: false
  ssl:
    require_ssl: false
    pem_certificate: $HOME/.vnc/self.pem
    pem_key: $HOME/.vnc/self.key
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

log "clean any stale :1 session/locks"
vncserver -kill :1 >/dev/null 2>&1 || true
pkill -f "Xkasmvnc.*:1" 2>/dev/null || true
rm -f /tmp/.X1-lock /tmp/.X11-unix/X1 "$HOME"/.vnc/*:1.pid 2>/dev/null || true
sleep 1

log "restart service"
systemctl --user reset-failed kasmvnc-chrome.service 2>/dev/null || true
systemctl --user restart kasmvnc-chrome.service
sleep 12

log "=== status ==="
systemctl --user --no-pager status kasmvnc-chrome.service 2>&1 | head -12
log "=== CDP ==="
for i in $(seq 1 25); do curl -fs http://127.0.0.1:9222/json/version >/dev/null 2>&1 && break; sleep 1; done
curl -fs http://127.0.0.1:9222/json/version 2>/dev/null | head -2
log "=== sockets ==="
ss -ltn 2>/dev/null | grep -E ":6080|:9222" || echo none
log DONE

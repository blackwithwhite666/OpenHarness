#!/bin/bash
export HOME=/home/blackwithwhite
export XDG_RUNTIME_DIR=/run/user/$(id -u)
DB="python3 $HOME/bin/dropbox"
EXC="$HOME/dropbox_excludes.txt"
LOG="$HOME/dropbox_exclude_apply.log"
echo "[$(date)] watcher start" >> "$LOG"
# 1) wait for link (info.json created on link), up to ~2h
for i in $(seq 1 3600); do
  [ -f "$HOME/.dropbox/info.json" ] && break
  sleep 2
done
if [ ! -f "$HOME/.dropbox/info.json" ]; then echo "[$(date)] never linked, exit" >> "$LOG"; exit 0; fi
echo "[$(date)] link detected" >> "$LOG"
# 2) apply excludes in rounds until all present or disk low
for round in $(seq 1 100000); do
  current="$($DB exclude list 2>/dev/null)"
  missing=0
  while IFS= read -r rel; do
    [ -z "$rel" ] && continue
    if printf "%s\n" "$current" | grep -qxF "$rel"; then continue; fi
    name="${rel#Dropbox/}"
    $DB exclude add "$HOME/Dropbox/$name" >/dev/null 2>&1
    missing=$((missing+1))
  done < "$EXC"
  free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  echo "[$(date)] round=$round still_missing=$missing free=${free}G" >> "$LOG"
  if [ "${free:-99}" -lt 8 ]; then
    echo "[$(date)] LOW DISK <8G — stopping dropbox to prevent fill" >> "$LOG"
    systemctl --user stop dropbox 2>/dev/null
    pkill -f ".dropbox-dist/dropbox" 2>/dev/null
    break
  fi
  [ "$missing" -eq 0 ] && { echo "[$(date)] ALL 74 excludes applied" >> "$LOG"; break; }
  sleep 4
done
echo "[$(date)] final exclude list:" >> "$LOG"
$DB exclude list >> "$LOG" 2>&1
echo "[$(date)] du ~/Dropbox: $(du -sh $HOME/Dropbox 2>/dev/null | cut -f1)" >> "$LOG"
echo "[$(date)] watcher done" >> "$LOG"

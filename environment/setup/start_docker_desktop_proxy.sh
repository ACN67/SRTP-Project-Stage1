#!/usr/bin/env bash
set -euo pipefail
if ! pgrep -f "docker-desktop-user-distro proxy" >/dev/null 2>&1; then
  mkdir -p "$HOME/.local/state"
  nohup /mnt/wsl/docker-desktop/docker-desktop-user-distro proxy --distro-name Ubuntu > "$HOME/.local/state/docker-desktop-proxy.log" 2>&1 &
  sleep 3
fi
exec "$HOME/.local/bin/docker" ""

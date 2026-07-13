#!/usr/bin/env bash
set -euo pipefail
if ! pgrep -f "docker-desktop-user-distro proxy" >/dev/null 2>&1; then
  mkdir -p "/home/keshu/.local/state"
  nohup /mnt/wsl/docker-desktop/docker-desktop-user-distro proxy --distro-name Ubuntu > "/home/keshu/.local/state/docker-desktop-proxy.log" 2>&1 &
  sleep 3
fi
exec "/home/keshu/.local/bin/docker" ""

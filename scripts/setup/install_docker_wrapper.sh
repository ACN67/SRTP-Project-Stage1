#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$HOME/.local/bin"
rm -f "/home/keshu/.local/bin/docker"

cat > "$HOME/.local/bin/docker" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [ -S /var/run/docker.sock ] && [ -x /mnt/wsl/docker-desktop/cli-tools/usr/bin/docker ]; then
  exec /mnt/wsl/docker-desktop/cli-tools/usr/bin/docker "$@"
fi

if [ -x "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe" ]; then
  exec "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe" "$@"
fi

echo "docker CLI is not available. Start Docker Desktop or enable WSL integration." >&2
exit 127
EOF

chmod +x "$HOME/.local/bin/docker"
echo "Installed $HOME/.local/bin/docker"

#!/bin/sh
set -eu

# Prepare SSH dir
mkdir -p /home/node/.ssh

# Copy Windows-mounted key files into container-owned paths, then fix perms
if [ -f /tmp/host_id_ed25519 ]; then
  cp /tmp/host_id_ed25519 /home/node/.ssh/id_ed25519
  chmod 600 /home/node/.ssh/id_ed25519
fi

if [ -f /tmp/host_known_hosts ]; then
  cp /tmp/host_known_hosts /home/node/.ssh/known_hosts
  chmod 644 /home/node/.ssh/known_hosts
fi

# Tighten n8n config (stops the “permissions too wide” warning)
[ -f /home/node/.n8n/config ] && chmod 600 /home/node/.n8n/config || true

# Hand over to n8n
exec n8n start

#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$WORKSPACE_ROOT"

# Install opencode globally via npm
npm install -g opencode-ai

# Verify opencode
opencode --version

# Install uv if needed
if ! command -v uv &> /dev/null; then
    pip install uv
fi

# Sync workspace (dev deps included by default in sync)
uv sync

echo "Dev environment ready."
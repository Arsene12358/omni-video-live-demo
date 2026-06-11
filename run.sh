#!/usr/bin/env bash
# Start the live-demo backend. Assumes an omni server is already serving on $OMNI_PORT
# and writing its stdout to $OMNI_LOG (for KV/epoch metrics); put clips in $CLIP_DIR.
set -euo pipefail

export OMNI_HOST="${OMNI_HOST:-localhost}"
export OMNI_PORT="${OMNI_PORT:-8901}"
export OMNI_LOG="${OMNI_LOG:-server.log}"
export CLIP_DIR="${CLIP_DIR:-clips}"
export GPU_INDEX="${GPU_INDEX:-0}"
PORT="${PORT:-8800}"

cd "$(dirname "$0")/backend"
exec uvicorn demo_backend:app --host 0.0.0.0 --port "$PORT"

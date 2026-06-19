#!/usr/bin/env bash
# ChatClinic demo launcher — starts backend (FastAPI) + frontend (Next.js).
# Backend MUST use the `med` conda env (has torchxrayvision, open_clip, transformers, torch+CUDA).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MED_PY="/home/jinotter3/miniconda3/envs/med/bin/python"
export TORCHXRAYVISION_CACHE_DIR="$ROOT/checkpoints/torchxrayvision"  # use local CXR weights, no re-download

echo "[1/3] Starting backend on http://127.0.0.1:8001 (med env) ..."
"$MED_PY" -m uvicorn app.main:app --host 127.0.0.1 --port 8001 > "$ROOT/backend.log" 2>&1 &
BACKEND_PID=$!
trap 'echo; echo "Stopping..."; kill $BACKEND_PID 2>/dev/null || true; exit 0' INT TERM

# wait for backend health
for i in $(seq 1 40); do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8001/health 2>/dev/null | grep -q 200; then
    echo "      backend ready."
    break
  fi
  sleep 1
done

echo "[2/3] Starting frontend on http://localhost:3000 ..."
( cd webapp && npm run dev )   # runs in foreground; Ctrl+C stops everything

echo "[3/3] (frontend exited) stopping backend ..."
kill $BACKEND_PID 2>/dev/null || true

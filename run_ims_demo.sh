#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_ims_demo.sh
#
# One-command local launcher for the FULL IMS Platform stack:
#   - backend/server.py  (FastAPI, port 8001) -- mounts the legacy Flask
#     explorer.html app at "/", plus /api/analysis/*, /api/mrc_designer/*,
#     /api/four_state/*, /api/network_mrc/* (the new auto-MRC routes), etc.
#   - frontend/          (Vite + React, port 3000) -- proxies /api/*,
#     /explorer.html and /assets to the backend, and serves the native React
#     routes (e.g. /mrc, used by the "Stabilising MRC" analysis surface).
#
# The two pieces are split on purpose (see frontend/vite.config.ts's proxy
# block and the repo README): explorer.html's "Analysis Hub" and
# "Stabilising MRC" nav items only work when BOTH are running and you view
# the app through the Vite dev server (port 3000), not the backend alone.
# Hitting the backend directly on :8001 will 404 on /mrc and will not have
# the Vite proxy in front of /api/* -- that mismatch is what produces
# "Failed to load analysis registry" / "Unexpected token '<'" and a 404 on
# Stabilising MRC.
#
# Usage:
#   ./run_ims_demo.sh              # backend on :8001, frontend on :3000
#   BACKEND_PORT=8002 FRONTEND_PORT=3001 ./run_ims_demo.sh
#
# Then open:  http://127.0.0.1:${FRONTEND_PORT:-3000}/explorer.html
# (In Codespaces, open the forwarded URL for the frontend port with
#  /explorer.html appended -- Codespaces will prompt to make it public/private
#  the first time a port is forwarded.)
#
# Stop with Ctrl+C (stops both processes).
# ---------------------------------------------------------------------------
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
VENV_DIR="$BACKEND_DIR/.venv_demo"
BACKEND_PORT="${BACKEND_PORT:-8001}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

if [ ! -d "$BACKEND_DIR/ims_platform" ]; then
  echo "error: expected $BACKEND_DIR/ims_platform to exist. Run this script from the repo root." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: python3 not found on PATH." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Backend: venv + deps
# ---------------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating backend virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Installing/verifying backend dependencies (first run only takes a minute)"
pip install --quiet --upgrade pip
if [ -f "$BACKEND_DIR/requirements.txt" ]; then
  pip install --quiet -r "$BACKEND_DIR/requirements.txt" || true
fi
# Belt-and-braces: these are imported at module load time by the IMS Platform
# app but are not always listed in requirements.txt.
pip install --quiet \
  "flask>=3.0" "numpy>=1.26" "scipy>=1.9" "sympy>=1.12" "matplotlib>=3.7" \
  "pyyaml>=6.0" "reportlab>=4.0" "pdfkit>=1.0" \
  "fastapi>=0.110" "uvicorn>=0.27" "motor>=3.4" "pymongo>=4.6" \
  "python-dotenv>=1.0" "pydantic>=2.6" "email-validator>=2.1" "pyjwt>=2.8" \
  "passlib>=1.7" "bcrypt==4.0.1" "python-jose>=3.3" "python-multipart>=0.0.9"

# backend/.env is gitignored and required at import time (MONGO_URL / DB_NAME).
# If you already have a real MongoDB configured, drop your own backend/.env
# before running this script and it will be left untouched. Otherwise this
# generates a harmless placeholder: Motor connects lazily, so the app boots
# and every route exercised by the Hero / Analysis Hub / Stabilising MRC /
# Network Builder + MRC workflow (none of which touch Mongo) works fine.
if [ ! -f "$BACKEND_DIR/.env" ]; then
  echo "==> No backend/.env found -- writing a placeholder Mongo config (safe: nothing in this workflow touches the DB)"
  cat > "$BACKEND_DIR/.env" <<EOF
MONGO_URL=mongodb://localhost:27017
DB_NAME=ims_demo
CORS_ORIGINS=*
EOF
fi

# ---------------------------------------------------------------------------
# Frontend: npm deps
# ---------------------------------------------------------------------------
if [ -d "$FRONTEND_DIR" ]; then
  if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo "==> Installing frontend dependencies (first run only, can take a couple of minutes)"
    # --legacy-peer-deps: this dependency graph trips a known npm/arborist
    # crash ("Cannot read properties of null (reading 'edgesOut')") under
    # strict peer-dep resolution on newer npm; legacy resolution avoids it.
    ( cd "$FRONTEND_DIR" && npm install --legacy-peer-deps )
  fi
else
  echo "warning: $FRONTEND_DIR not found -- Analysis Hub / Stabilising MRC (/mrc) will 404 without it." >&2
fi

# ---------------------------------------------------------------------------
# Launch both, tear down together on Ctrl+C / exit
# ---------------------------------------------------------------------------
PIDS=()
cleanup() {
  echo ""
  echo "==> Stopping..."
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT INT TERM

cd "$BACKEND_DIR"
echo "==> Starting backend (FastAPI + legacy Flask app mounted at /) on :$BACKEND_PORT"
"$PYTHON_BIN" -m uvicorn server:app --host 0.0.0.0 --port "$BACKEND_PORT" > "$SCRIPT_DIR/backend.demo.log" 2>&1 &
PIDS+=($!)

if [ -d "$FRONTEND_DIR" ]; then
  cd "$FRONTEND_DIR"
  echo "==> Starting frontend (Vite dev server, proxies /api + /explorer.html + /assets to :$BACKEND_PORT) on :$FRONTEND_PORT"
  VITE_PORT="$FRONTEND_PORT" npx vite --port "$FRONTEND_PORT" --host > "$SCRIPT_DIR/frontend.demo.log" 2>&1 &
  PIDS+=($!)
fi

sleep 3

cat <<EOF

==================================================================
 IMS Platform Explorer -- local demo stack
 Open this in your browser:   http://127.0.0.1:${FRONTEND_PORT}/explorer.html
 (backend alone, no Analysis Hub / Stabilising MRC: http://127.0.0.1:${BACKEND_PORT}/)
 Logs:  $SCRIPT_DIR/backend.demo.log , $SCRIPT_DIR/frontend.demo.log
 Stop with:                   Ctrl+C
==================================================================

EOF

wait

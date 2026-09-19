#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_ims_demo.sh
#
# One-command local launcher for the IMS Platform Explorer (hero page +
# Network Builder + MRC workflow), independent of the Mongo-backed FastAPI
# shell in backend/server.py. It runs the underlying Flask app
# (ims_platform.server.app:create_app) directly, which serves:
#   - explorer.html (the hero page you just reviewed)
#   - /api/network_mrc/inspect, /auto_apply, /auto_closed_loop
#   - every other existing IMS Platform route
#
# Usage:
#   ./run_ims_demo.sh            # start on http://127.0.0.1:5050
#   PORT=8080 ./run_ims_demo.sh  # start on a different port
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
VENV_DIR="$BACKEND_DIR/.venv_demo"
PORT="${PORT:-5050}"

if [ ! -d "$BACKEND_DIR/ims_platform" ]; then
  echo "error: expected $BACKEND_DIR/ims_platform to exist. Run this script from a checkout of the repo." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: python3 not found on PATH." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Installing/verifying dependencies (first run only takes a minute)"
pip install --quiet --upgrade pip
# Core deps the Flask app needs at import time.
pip install --quiet \
  "flask>=3.0" "numpy>=1.26" "scipy>=1.9" "sympy>=1.12" "matplotlib>=3.7" \
  "pyyaml>=6.0" "reportlab>=4.0" "pdfkit>=1.0" "pytest>=8.0"

cd "$BACKEND_DIR"

cat <<EOF

==================================================================
 IMS Platform Explorer — local demo server
 Open this in your browser:   http://127.0.0.1:${PORT}/
 Stop the server with:        Ctrl+C
==================================================================

EOF

exec "$PYTHON_BIN" -c "
import sys
sys.path.insert(0, '.')
from ims_platform.server.app import create_app
app = create_app()
app.run(host='127.0.0.1', port=${PORT}, debug=False)
"

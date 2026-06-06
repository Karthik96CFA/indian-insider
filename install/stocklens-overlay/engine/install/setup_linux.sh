#!/usr/bin/env bash
# StockLens engine — in-repo install + cron automation (monorepo).
set -euo pipefail

ENGINE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${ENGINE_ROOT}/.." && pwd)"
PY="$(command -v python3)" || { echo "python3 not found." >&2; exit 1; }

echo "==> StockLens engine setup"
echo "    Repo:   ${REPO_ROOT}"
echo "    Engine: ${ENGINE_ROOT}"

mkdir -p "${ENGINE_ROOT}/.state/logs"

if [[ ! -f "${ENGINE_ROOT}/.env" ]]; then
  cp "${ENGINE_ROOT}/config/.env.example" "${ENGINE_ROOT}/.env"
  echo "==> Created ${ENGINE_ROOT}/.env — add your API keys."
else
  echo "==> Keeping existing ${ENGINE_ROOT}/.env"
fi

echo "==> Installing Python dependencies"
"${PY}" -m pip install --user -r "${ENGINE_ROOT}/requirements.txt"

echo "==> Initializing database schema"
"${PY}" -c "import sys; sys.path.insert(0, '${ENGINE_ROOT}/agents'); from common import initialize_db; initialize_db(); print('DB ready')"

echo "==> Registering cron jobs"
bash "${ENGINE_ROOT}/install/schedule_linux.sh"

cat <<EOF

Engine setup complete.

  Config:  ${ENGINE_ROOT}/.env
  Logs:    ${ENGINE_ROOT}/.state/logs
  Test:    python ${ENGINE_ROOT}/agents/stocklens_bridge.py --test
  Tick:    python ${ENGINE_ROOT}/agents/orchestrator.py --phase tick

Automation schedule (UTC — IST = UTC+5:30):
  Every 30 min     tick
  Mon–Fri 02:00    morning
  Mon–Fri 02:30    briefing
  Mon–Fri 10:30    portfolio
  Mon–Fri 12:30    eod
  Monday  03:30    weekly
  Sunday  00:30    research
EOF

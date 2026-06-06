#!/usr/bin/env bash
# StockLens engine — in-repo install + launchd automation (monorepo).
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

"${PY}" -m pip install --user -r "${ENGINE_ROOT}/requirements.txt"
"${PY}" -c "import sys; sys.path.insert(0, '${ENGINE_ROOT}/agents'); from common import initialize_db; initialize_db(); print('DB ready')"
bash "${ENGINE_ROOT}/install/schedule_mac.sh"

cat <<EOF

Engine setup complete.

  Config:  ${ENGINE_ROOT}/.env
  Logs:    ${ENGINE_ROOT}/.state/logs
  Test:    python ${ENGINE_ROOT}/agents/stocklens_bridge.py --test
EOF

#!/usr/bin/env bash
# Install Indian Insider to ~/indian-insider and register launchd automation.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HOME/indian-insider"
PY="$(command -v python3)" || { echo "python3 not found." >&2; exit 1; }

echo "==> Installing Indian Insider to ${DEST}"
mkdir -p "${DEST}/agents" "${DEST}/config" "${DEST}/.state/logs"
rsync -a --delete "${REPO_ROOT}/agents/" "${DEST}/agents/"
rsync -a "${REPO_ROOT}/config/" "${DEST}/config/"
rsync -a "${REPO_ROOT}/install/" "${DEST}/install/"
cp -f "${REPO_ROOT}/requirements.txt" "${DEST}/requirements.txt"

if [[ ! -f "${DEST}/.env" ]]; then
  cp "${DEST}/config/.env.example" "${DEST}/.env"
  echo "==> Created ${DEST}/.env — add your API keys."
fi

"${PY}" -m pip install --user -r "${DEST}/requirements.txt"
"${PY}" -c "import sys; sys.path.insert(0, '${DEST}/agents'); from common import initialize_db; initialize_db(); print('DB ready')"
bash "${DEST}/install/schedule_mac.sh"
echo "Setup complete. Edit ~/indian-insider/.env then check logs in ~/.state/logs/"

#!/usr/bin/env bash
# Install Indian Insider to ~/indian-insider and register full automation cron.
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
  echo "==> Created ${DEST}/.env — edit it with your API keys before the pipeline runs."
else
  echo "==> Keeping existing ${DEST}/.env"
fi

echo "==> Installing Python dependencies"
"${PY}" -m pip install --user -r "${DEST}/requirements.txt"

echo "==> Initializing database schema"
"${PY}" "${DEST}/agents/orchestrator.py" --phase tick --dry-run >/dev/null 2>&1 || true
"${PY}" -c "import sys; sys.path.insert(0, '${DEST}/agents'); from common import initialize_db; initialize_db(); print('DB ready')"

echo "==> Registering cron jobs"
bash "${DEST}/install/schedule_linux.sh"

cat <<'EOF'

Setup complete.

Next steps:
  1. Edit ~/indian-insider/.env (GEMINI_API_KEY, GMAIL_*, optional TELEGRAM_*)
  2. Logs: ~/indian-insider/.state/logs/
  3. Smoke test: python3 ~/indian-insider/agents/orchestrator.py --phase tick --dry-run

Automation schedule (UTC — IST = UTC+5:30):
  Every 30 min     tick      collector → consensus → alerts
  Mon–Fri 02:00    morning   Doraemon + Suneo scouts
  Mon–Fri 02:30    briefing  Telegram morning digest
  Mon–Fri 10:30    portfolio Dekisugi drift check
  Mon–Fri 12:30    eod       Shinchan FII/DII
  Monday  03:30    weekly    Nobita RBI macro
  Sunday  00:30    research  full factor refresh + score snapshots
EOF

#!/usr/bin/env bash
# Indian Insider — full automation cron (all times UTC; IST = UTC+5:30).
set -euo pipefail
ROOT="$HOME/indian-insider"
AGENTS="$ROOT/agents"
LOGS="$ROOT/.state/logs"
PY="$(command -v python3)"
[[ -z "$PY" ]] && { echo "python3 not found." >&2; exit 1; }
mkdir -p "$LOGS"
MARK_START="# >>> indian-insider (managed) >>>"
MARK_END="# <<< indian-insider (managed) <<<"
current="$(crontab -l 2>/dev/null || true)"
stripped="$(printf '%s\n' "$current" | awk -v s="$MARK_START" -v e="$MARK_END" '$0==s{skip=1;next}$0==e{skip=0;next}!skip{print}')"
run() { echo "${1} ${PY} ${AGENTS}/orchestrator.py --phase ${2} --continue-on-error >> ${LOGS}/${3}.cron.log 2>&1"; }
block="$(cat <<EOF
${MARK_START}
# Real-time pipeline — every 30 minutes
$(run "*/30 * * * *" "tick" "tick")
# Mon–Fri 07:30 IST — pre-market scouts
$(run "0 2 * * 1-5" "morning" "morning")
# Mon–Fri 08:00 IST — morning Telegram briefing
$(run "30 2 * * 1-5" "briefing" "briefing")
# Mon–Fri 16:00 IST — portfolio drift (Dekisugi)
$(run "30 10 * * 1-5" "portfolio" "portfolio")
# Mon–Fri 18:00 IST — FII/DII close analysis
$(run "30 12 * * 1-5" "eod" "eod")
# Monday 09:00 IST — RBI macro scout
$(run "30 3 * * 1" "weekly" "weekly")
# Sunday 06:00 IST — weekly factor refresh + score snapshots
$(run "30 0 * * 0" "research" "research")
${MARK_END}
EOF
)"
printf '%s\n\n%s\n' "$stripped" "$block" | crontab -
echo "Indian Insider automation registered (7 cron phases). Logs → $LOGS"

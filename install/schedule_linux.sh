#!/usr/bin/env bash
# All cron times UTC. IST = UTC+5:30
set -euo pipefail
ROOT="$HOME/indian-insider"; AGENTS="$ROOT/agents"; LOGS="$ROOT/.state/logs"
PY="$(command -v python3)"; [[ -z "$PY" ]] && { echo "python3 not found." >&2; exit 1; }
mkdir -p "$LOGS"
MARK_START="# >>> indian-insider (managed) >>>"
MARK_END="# <<< indian-insider (managed) <<<"
current="$(crontab -l 2>/dev/null || true)"
stripped="$(printf '%s\n' "$current" | awk -v s="$MARK_START" -v e="$MARK_END" '$0==s{skip=1;next}$0==e{skip=0;next}!skip{print}')"
run() { echo "${1} ${PY} ${AGENTS}/${2} >> ${LOGS}/${3}.cron.log 2>&1"; }
block="$(cat <<EOF
${MARK_START}
$(run "0,30 * * * *" "nse_collector.py" "collector")
$(run "2,32 * * * *" "event_detector.py" "detector")
$(run "4,34 * * * *" "scoring_engine.py" "scorer")
$(run "6,36 * * * *" "gian.py"           "gian")
${MARK_END}
EOF
)"
printf '%s\n\n%s\n' "$stripped" "$block" | crontab -
echo "All 7 agents registered. Logs → $LOGS"

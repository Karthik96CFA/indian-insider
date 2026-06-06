#!/usr/bin/env bash
# All times UTC. IST = UTC+5:30
# IST 07:30=UTC 02:00 | IST 08:00=UTC 02:30 | IST Mon 09:00=UTC Mon 03:30
# IST 16:00=UTC 10:30 | IST 18:00=UTC 12:30
set -euo pipefail
ROOT="$HOME/indian-insider"; AGENTS="$ROOT/agents"; LOGS="$ROOT/.state/logs"
LA_DIR="$HOME/Library/LaunchAgents"; PY="$(command -v python3)"
[[ -z "$PY" ]] && { echo "python3 not found." >&2; exit 1; }
mkdir -p "$LA_DIR" "$LOGS"

write_plist() {
  local name="$1" script="$2"; shift 2; local sched="$*"
  local label="in.market.insider.${name}"; local plist="$LA_DIR/${label}.plist"
  cat >"$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>${label}</string>
  <key>ProgramArguments</key><array><string>${PY}</string><string>${AGENTS}/${script}</string></array>
  <key>WorkingDirectory</key><string>${ROOT}</string>
  <key>StandardOutPath</key><string>${LOGS}/${name}.out.log</string>
  <key>StandardErrorPath</key><string>${LOGS}/${name}.err.log</string>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string></dict>
  ${sched}
</dict></plist>
EOF
  launchctl unload "$plist" 2>/dev/null || true
  launchctl load "$plist" && echo "  ✓ ${label}"
}

at_hm()      { echo "<key>StartCalendarInterval</key><dict><key>Hour</key><integer>$1</integer><key>Minute</key><integer>$2</integer></dict>"; }
weekly_at()  { echo "<key>StartCalendarInterval</key><dict><key>Weekday</key><integer>$1</integer><key>Hour</key><integer>$2</integer><key>Minute</key><integer>$3</integer></dict>"; }
every_sec()  { echo "<key>StartInterval</key><integer>$1</integer>"; }

echo "Registering Indian Insider agents with launchd…"
write_plist "collector" "nse_collector.py" "$(every_sec 1800)"
write_plist "detector"  "event_detector.py"  "$(every_sec 1800)"
write_plist "scorer"    "scoring_engine.py"  "$(every_sec 1800)"
write_plist "gian"      "gian.py"            "$(every_sec 1800)"
echo; echo "All pipeline agents registered. Logs → $LOGS"

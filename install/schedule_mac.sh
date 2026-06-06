#!/usr/bin/env bash
# Indian Insider — full automation via launchd (times UTC; IST = UTC+5:30).
# Weekday gates are enforced inside orchestrator.py (IST).
set -euo pipefail
ROOT="$HOME/indian-insider"
AGENTS="$ROOT/agents"
LOGS="$ROOT/.state/logs"
LA_DIR="$HOME/Library/LaunchAgents"
PY="$(command -v python3)"
[[ -z "$PY" ]] && { echo "python3 not found." >&2; exit 1; }
mkdir -p "$LA_DIR" "$LOGS"

write_plist() {
  local name="$1" phase="$2"; shift 2
  local sched="$*"
  local label="in.market.insider.${name}"
  local plist="$LA_DIR/${label}.plist"
  cat >"$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>${label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${PY}</string>
    <string>${AGENTS}/orchestrator.py</string>
    <string>--phase</string>
    <string>${phase}</string>
    <string>--continue-on-error</string>
  </array>
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
weekday_at() { echo "<key>StartCalendarInterval</key><dict><key>Weekday</key><integer>$1</integer><key>Hour</key><integer>$2</integer><key>Minute</key><integer>$3</integer></dict>"; }
every_sec()  { echo "<key>StartInterval</key><integer>$1</integer>"; }

echo "Registering Indian Insider automation with launchd…"
write_plist "tick"      "tick"      "$(every_sec 1800)"
write_plist "morning"   "morning"   "$(at_hm 2 0)"
write_plist "briefing"  "briefing"  "$(at_hm 2 30)"
write_plist "portfolio" "portfolio" "$(at_hm 10 30)"
write_plist "eod"       "eod"       "$(at_hm 12 30)"
write_plist "weekly"    "weekly"    "$(at_hm 3 30)"
write_plist "research"  "research"  "$(weekday_at 0 0 30)"
echo
echo "All automation phases registered. Logs → $LOGS"

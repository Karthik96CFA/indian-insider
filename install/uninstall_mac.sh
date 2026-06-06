#!/usr/bin/env bash
set -euo pipefail
LA_DIR="$HOME/Library/LaunchAgents"
for name in doraemon shinchan nobita dekisugi suneo doraemi collector detector scorer gian; do
  plist="$LA_DIR/in.market.insider.${name}.plist"
  [[ -f "$plist" ]] || continue
  launchctl unload "$plist" 2>/dev/null || true
  rm -f "$plist" && echo "  - removed in.market.insider.${name}"
done
echo "All agents unregistered. Scripts + state remain at ~/indian-insider/."

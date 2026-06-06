#!/usr/bin/env bash
set -euo pipefail
LA_DIR="$HOME/Library/LaunchAgents"
for name in tick morning briefing portfolio eod weekly research; do
  plist="$LA_DIR/in.market.stocklens.engine.${name}.plist"
  [[ -f "$plist" ]] || continue
  launchctl unload "$plist" 2>/dev/null || true
  rm -f "$plist" && echo "  - removed in.market.stocklens.engine.${name}"
done
echo "Engine launchd jobs removed. Engine files remain in your stocklens clone."

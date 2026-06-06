#!/usr/bin/env bash
set -euo pipefail
MARK_START="# >>> stocklens-engine (managed) >>>"
MARK_END="# <<< stocklens-engine (managed) <<<"
current="$(crontab -l 2>/dev/null || true)"
printf '%s\n' "$current" | awk -v s="$MARK_START" -v e="$MARK_END" '$0==s{skip=1;next}$0==e{skip=0;next}!skip{print}' | crontab -
echo "StockLens engine block removed from crontab."

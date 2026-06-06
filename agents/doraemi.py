#!/usr/bin/env python3
"""
Doraemi — consensus analyst.
Pure local SQLite logic — zero Gemini API calls.
Fires when >= 4 scouts agree on same ticker + direction within 7 days.
Schedule: every 30 minutes.
"""
from __future__ import annotations
import os, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NEUTRAL, ConsensusEvent, log, read_window, record_consensus

MIN_AGREE   = int(os.environ.get("DORAEMI_MIN_AGREE", "4"))
WINDOW_DAYS = int(os.environ.get("DORAEMI_WINDOW_DAYS", "7"))


def main() -> int:
    signals = read_window(days=WINDOW_DAYS)
    if not signals:
        log("doraemi", "no signals in window")
        print("[doraemi] no signals in window")
        return 0

    by_key: dict[tuple[str, str], list] = defaultdict(list)
    for s in signals:
        if s.direction == NEUTRAL:
            continue
        if any(x.scout == s.scout for x in by_key[(s.ticker, s.direction)]):
            continue
        by_key[(s.ticker, s.direction)].append(s)

    fired = 0
    for (ticker, direction), group in by_key.items():
        scouts = sorted({g.scout for g in group})
        if len(scouts) < MIN_AGREE:
            continue
        reasons = []
        for sc in scouts:
            latest = next((g for g in group if g.scout == sc), None)
            if latest:
                reasons.append(f"{sc}: {latest.reason}")
        ev = ConsensusEvent(
            ticker=ticker, direction=direction, scouts=scouts,
            reasons=reasons, timestamp=datetime.now(timezone.utc),
        )
        row_id = record_consensus(ev)
        log("doraemi", f"CONSENSUS [{row_id}] {direction} {ticker} ({len(scouts)} scouts: {', '.join(scouts)})")
        print(f"[doraemi] CONSENSUS {direction} {ticker} - {len(scouts)} scouts agree")
        fired += 1

    if fired == 0:
        log("doraemi", f"no consensus (min={MIN_AGREE}, window={WINDOW_DAYS}d)")
        print(f"[doraemi] no consensus yet (need >= {MIN_AGREE} scouts to agree)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

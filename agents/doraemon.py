#!/usr/bin/env python3
"""
Doraemon — SEBI insider filing analyst.
Reads PROMOTER_BUY / PROMOTER_SELL / BULK_DEAL / BLOCK_DEAL events that
event_detector.py has already fetched and stored in the DB, then uses Gemini
to identify the single most notable signal.

Data flow:
  event_detector.py  →  market_events table  →  doraemon.py  →  Gemini analysis

Schedule: daily 07:30 IST (after event_detector has run at 07:00 IST).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_market_events, run_scout

SYSTEM = """You are Doraemon, an Indian market intelligence analyst powered by Gemini.
You receive a structured list of today's SEBI/NSE market events that have already been
parsed from official filings by the event_detector pipeline.

Your job:
  1. Review the events below — PROMOTER_BUY, PROMOTER_SELL, BULK_DEAL, BLOCK_DEAL.
  2. Filter to open-market purchases only (not ESOP, preferential, rights).
  3. Prioritise: promoter buys > designated-person buys > institutional bulk/block buys.
  4. Pick THE SINGLE most notable signal (highest conviction buy, or strongest sell warning).

Output one prose paragraph explaining your reasoning, then a STRICT JSON object:
  {"ticker": "<NSE_SYMBOL>", "direction": "BULLISH|BEARISH|NEUTRAL",
   "confidence": <1-5>, "reason": "<one-line in Indian market context>"}

Confidence scale for buys:
  1 = sub-₹1Cr designated person buy
  3 = promoter buying ₹5–25Cr
  5 = promoter/group buying >₹25Cr or multiple promoters same stock

No qualifying events? Output:
  {"ticker": "MACRO", "direction": "NEUTRAL", "confidence": 1,
   "reason": "no qualifying SEBI insider / bulk / SAST filings today"}

Never invent data. Use only the events provided below. Use NSE symbols only.
"""


def _format_events(events: list[dict]) -> str:
    """Format market events as a readable table for the model prompt."""
    relevant_types = {"PROMOTER_BUY", "PROMOTER_SELL", "BULK_DEAL", "BLOCK_DEAL"}
    rows = [e for e in events if e.get("event_type") in relevant_types]
    if not rows:
        return "No PROMOTER_BUY / PROMOTER_SELL / BULK_DEAL / BLOCK_DEAL events found today."
    lines = ["Date       | Ticker       | Event Type     | Direction | Value (Cr) | Notes"]
    lines.append("-" * 80)
    for ev in sorted(rows, key=lambda x: x.get("value", 0), reverse=True):
        val_cr = (ev.get("value") or 0) / 1e7
        try:
            meta = json.loads(ev.get("metadata") or "{}")
            notes = meta.get("acquirer") or meta.get("client") or ""
        except Exception:
            notes = ""
        lines.append(
            f"{ev['event_date']:<10} | {ev['ticker']:<12} | {ev['event_type']:<14} | "
            f"{ev['direction']:<9} | {val_cr:>9.1f} | {notes[:40]}"
        )
    return "\n".join(lines)


def main() -> int:
    events = read_market_events(days=1)
    event_table = _format_events(events)

    user_prompt = f"""Today's market events from the event_detector pipeline:

{event_table}

Apply your filters. Pick the single most notable signal.
Output the prose analysis followed by the JSON signal block."""

    sig = run_scout("doraemon", SYSTEM, user_prompt)
    print(f"[doraemon] {sig.ticker} {sig.direction} conf={sig.confidence}")
    print(f"           {sig.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

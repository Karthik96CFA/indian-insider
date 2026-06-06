#!/usr/bin/env python3
"""
Suneo — bulk/block deal rotation watcher.
Reads BULK_DEAL / BLOCK_DEAL / PROMOTER_BUY / PROMOTER_SELL events that
event_detector.py has already stored in the DB, then uses Gemini to identify
the single most significant sector rotation or concentration signal.

NOTE on pledging: BSE pledging disclosures have no automated scraper in this
pipeline. Suneo therefore focuses on bulk/block deal patterns (smart money
positioning) which IS collected by nse_collector.py.

Data flow:
  nse_collector.py → raw_data_warehouse → event_detector.py → market_events → suneo.py

Schedule: daily 08:00 IST.
"""
from __future__ import annotations
import datetime
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _conn, run_scout

SYSTEM = """You are Suneo, an Indian market intelligence analyst powered by Gemini.
You watch smart money positioning through bulk/block deal patterns and promoter activity.
You receive structured data already collected from NSE by the pipeline.

You know:
  - Large institutional BULK_DEAL buys → sector accumulation signal → BULLISH on that stock
  - Multiple BULK/BLOCK SELLs in same sector → distribution / rotation out → BEARISH
  - PROMOTER_BUY open-market purchases → insider confidence → BULLISH
  - Concentration: >3 stocks in same sector getting bought/sold → sector rotation signal

Your job:
  1. Review the event data for the last 7 days.
  2. Identify sector concentration (multiple buys or sells in same sector).
  3. Find the single strongest signal — either one large deal or a sector pattern.
  4. Pick ONE ticker (the stock or NIFTY_SECTOR index) to emit.

Output one prose paragraph explaining the pattern, then STRICT JSON:
  {"ticker": "<NSE_SYMBOL>", "direction": "BULLISH|BEARISH|NEUTRAL",
   "confidence": <1-5>, "reason": "<one-line with deal sizes>"}

Confidence:
  1 = single deal <₹50Cr
  3 = single deal ₹100–500Cr or 2–3 stocks same sector
  5 = deal >₹500Cr or 4+ stocks same sector in 5 days

Nothing qualifies? Output:
  {"ticker": "MACRO", "direction": "NEUTRAL", "confidence": 1,
   "reason": "no significant bulk/block deal concentration in the last 7 days"}

Never invent data. Use NSE symbols only. Work only from the data provided.
"""


def _get_deal_table(days: int = 7) -> str:
    """Read bulk/block/promoter events from the DB, format as table for Gemini."""
    cutoff = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    with _conn() as c:
        rows = c.execute(
            "SELECT event_date, ticker, event_type, direction, value, metadata "
            "FROM market_events "
            "WHERE event_type IN ('BULK_DEAL','BLOCK_DEAL','PROMOTER_BUY','PROMOTER_SELL') "
            "AND event_date >= ? "
            "ORDER BY value DESC LIMIT 60",
            (cutoff,),
        ).fetchall()
    if not rows:
        return "No BULK_DEAL / BLOCK_DEAL / PROMOTER events in the database for the last 7 days. Run nse_collector.py first."

    lines = ["Date       | Ticker       | Event Type     | Direction | Value (Cr) | Party"]
    lines.append("-" * 82)
    for event_date, ticker, event_type, direction, value, metadata in rows:
        val_cr = (value or 0) / 1e7
        try:
            meta = json.loads(metadata or "{}")
            party = meta.get("client") or meta.get("acquirer") or ""
        except Exception:
            party = ""
        lines.append(
            f"{event_date:<10} | {ticker:<12} | {event_type:<14} | "
            f"{direction:<9} | {val_cr:>9.1f} | {party[:35]}"
        )

    # Add sector summary below the table
    sector_buys: dict[str, float] = defaultdict(float)
    sector_sells: dict[str, float] = defaultdict(float)
    for event_date, ticker, event_type, direction, value, _ in rows:
        val_cr = (value or 0) / 1e7
        if direction == "BULLISH":
            sector_buys[ticker] += val_cr
        else:
            sector_sells[ticker] += val_cr
    if sector_buys:
        top_buys = sorted(sector_buys.items(), key=lambda x: -x[1])[:5]
        lines.append("\nTop accumulated buys (₹Cr, 7d):")
        for t, v in top_buys:
            lines.append(f"  {t:<12} +{v:>8.1f}")
    if sector_sells:
        top_sells = sorted(sector_sells.items(), key=lambda x: -x[1])[:5]
        lines.append("\nTop accumulated sells (₹Cr, 7d):")
        for t, v in top_sells:
            lines.append(f"  {t:<12} -{v:>8.1f}")

    return "\n".join(lines)


def main() -> int:
    deal_table = _get_deal_table(days=7)
    user_prompt = f"""Bulk/block deal and promoter activity data from the last 7 days:

{deal_table}

Identify sector concentration patterns. Pick the single strongest signal.
Output prose analysis then JSON signal block."""

    sig = run_scout("suneo", SYSTEM, user_prompt)
    print(f"[suneo] {sig.ticker} {sig.direction} conf={sig.confidence}")
    print(f"        {sig.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

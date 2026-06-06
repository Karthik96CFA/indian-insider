#!/usr/bin/env python3
"""
Shinchan — FII/DII daily flow analyst.
Reads FII_NET_FLOW / DII_NET_FLOW events from the DB (populated by
nse_collector.py → event_detector.py), then uses Gemini to interpret
the 5-day trend and generate a NIFTY direction signal.

Data flow:
  nse_collector.py → raw_data_warehouse → event_detector.py → market_events → shinchan.py

Schedule: daily 18:00 IST (after market close, after nse_collector + event_detector have run).
"""
from __future__ import annotations
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _conn, run_scout

SYSTEM = """You are Shinchan, an Indian market FII/DII flow analyst powered by Gemini.
You receive structured FII/DII flow data already fetched from NSE by the collector pipeline.

Your job:
  1. Review the flow data below for the last 5 trading days.
  2. Compute FII 5-day cumulative net flow and DII 5-day cumulative net flow.
  3. Assess convergence/divergence.

Direction rules for NIFTY:
  - FII 5-day cumulative > ₹2,000 Cr net buy → BULLISH
  - FII 5-day cumulative < −₹2,000 Cr net sell → BEARISH
  - FII + DII both net buying → strong BULLISH (raise confidence by 1)
  - FII + DII both net selling → strong BEARISH (raise confidence by 1)
  - Diverging → NEUTRAL

Output one prose paragraph with the actual ₹Cr figures, then STRICT JSON:
  {"ticker": "NIFTY", "direction": "BULLISH|BEARISH|NEUTRAL",
   "confidence": <1-5>, "reason": "<one-line with ₹Cr cumulative figures>"}

Confidence:
  1 = single-day data or very small flows (<₹500 Cr)
  3 = 3-day trend, ₹2,000–5,000 Cr cumulative
  5 = 5-day aligned FII+DII, >₹5,000 Cr cumulative

No data? Output:
  {"ticker": "NIFTY", "direction": "NEUTRAL", "confidence": 1,
   "reason": "FII/DII data unavailable — market holiday or collector not run"}

Use only the data provided. Do not invent figures.
"""


def _get_fiidii_table(days: int = 7) -> str:
    """Read FII/DII flow events from the DB and format as a table for Gemini."""
    cutoff = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    with _conn() as c:
        rows = c.execute(
            "SELECT event_date, event_type, direction, value FROM market_events "
            "WHERE ticker='NIFTY' AND event_type IN ('FII_NET_FLOW','DII_NET_FLOW') "
            "AND event_date >= ? ORDER BY event_date DESC, event_type ASC",
            (cutoff,),
        ).fetchall()
    if not rows:
        return "No FII/DII data in the database for the last 7 days. Run nse_collector.py first."
    lines = ["Date       | Type         | Direction | Net Flow (₹ Cr)"]
    lines.append("-" * 55)
    for event_date, event_type, direction, value in rows:
        sign = "+" if direction == "BULLISH" else "-"
        lines.append(f"{event_date:<10} | {event_type:<12} | {direction:<9} | {sign}{value / 1e7:>10.1f}")
    return "\n".join(lines)


def main() -> int:
    flow_table = _get_fiidii_table(days=7)
    user_prompt = f"""FII/DII flow data from the collector pipeline (last 7 days):

{flow_table}

Compute cumulative flows, assess convergence, apply direction rules.
Output the prose analysis followed by the JSON signal block."""

    sig = run_scout("shinchan", SYSTEM, user_prompt)
    print(f"[shinchan] {sig.ticker} {sig.direction} conf={sig.confidence}")
    print(f"           {sig.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

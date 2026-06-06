#!/usr/bin/env python3
"""
seed_price_cache.py — Seed historical_prices table from yfinance.

Run once (takes 5–15 min for full NSE universe) to populate the price cache.
The backtester and paper_trading.py will then use cached prices for all lookups.

Usage:
  python seed_price_cache.py                  # all VALID_TICKERS, 2020-01-01 to today
  python seed_price_cache.py --start 2018-01-01
  python seed_price_cache.py --ticker RELIANCE HDFCBANK INFY   # specific tickers
  python seed_price_cache.py --limit 20       # first 20 tickers (quick test)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import fetch_and_cache_prices
from event_detector import VALID_TICKERS

_EXCLUDE = {"NIFTY", "BANKNIFTY", "NIFTYBEES", "GOLDBEES", "GOLD"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed historical price cache")
    parser.add_argument("--ticker", nargs="*", help="Specific NSE tickers")
    parser.add_argument("--start",  default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--limit",  type=int, default=None, help="Limit tickers (for testing)")
    args = parser.parse_args()

    if args.ticker:
        tickers = [t.upper() for t in args.ticker]
    else:
        tickers = [t for t in VALID_TICKERS if t not in _EXCLUDE and not t.endswith("BEES")]

    if args.limit:
        tickers = tickers[:args.limit]

    print(f"[price_cache] Fetching prices for {len(tickers)} tickers from {args.start}...")
    print(f"[price_cache] This may take several minutes for large batches.")

    # Fetch in batches of 50 to avoid yfinance timeouts
    batch_size = 50
    total_rows = 0
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"[price_cache] Batch {i//batch_size + 1}: {batch[0]} … {batch[-1]}")
        rows = fetch_and_cache_prices(batch, start=args.start)
        total_rows += rows
        print(f"              → {rows:,} rows cached")

    print(f"\n[price_cache] Done. {total_rows:,} total price rows written to historical_prices.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

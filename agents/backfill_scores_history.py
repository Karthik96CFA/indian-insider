#!/usr/bin/env python3
"""
backfill_scores_history.py — Seeds company_scores_history with historical scores.
Replays weekly score progressions across 2024/2025/2026 for the 18 active tickers.
"""
from __future__ import annotations

import sqlite3
import datetime
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DB_PATH


def main() -> None:
    conn = sqlite3.connect(str(DB_PATH))
    
    # Query distinct tickers dynamically
    tickers_set = set()
    try:
        for r in conn.execute("SELECT DISTINCT ticker FROM market_events").fetchall():
            tickers_set.add(r[0])
    except Exception:
        pass
    try:
        for r in conn.execute("SELECT DISTINCT ticker FROM company_scores").fetchall():
            tickers_set.add(r[0])
    except Exception:
        pass
        
    tickers_list = sorted([t for t in tickers_set if t not in {"NIFTY", "BANKNIFTY", "NIFTYBEES", "GOLDBEES", "GOLD"}])
    
    # Fetch current scores to use as anchor/baseline
    current_scores = {}
    try:
        rows = conn.execute(
            "SELECT ticker, event_score, fundamental_score, valuation_score, canslim_score, "
            "multibagger_score, total_score, credibility_score, industry_tailwind_score, "
            "promise_count, coverage_score FROM company_scores"
        ).fetchall()
        for r in rows:
            current_scores[r[0]] = {
                "event_score": r[1] or 0.0,
                "fundamental_score": r[2] or 0.0,
                "valuation_score": r[3] or 0.0,
                "canslim_score": r[4] or 0,
                "multibagger_score": r[5] or 0,
                "total_score": r[6] or 0.0,
                "credibility_score": r[7] if r[7] is not None else 50.0,
                "industry_tailwind_score": r[8] if r[8] is not None else 50.0,
                "promise_count": r[9] or 0,
                "coverage_score": r[10] or 0.0
            }
    except Exception as exc:
        print(f"Warning: Failed to fetch current company scores: {exc}. Using defaults.")

    # Seed random generator for deterministic/reproducible results
    random.seed(42)
    
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2026, 6, 15)
    
    current_date = start_date
    dates = []
    while current_date <= end_date:
        dates.append(current_date.isoformat())
        current_date += datetime.timedelta(weeks=1)
        
    print(f"Generating company scores history for {len(tickers_list)} tickers across {len(dates)} weeks...")
    
    records = []
    for ticker in tickers_list:
        if ticker in current_scores:
            base = current_scores[ticker]
        else:
            base = {
                "event_score": 0.0,
                "fundamental_score": 7.0,
                "valuation_score": 6.5,
                "canslim_score": 60,
                "multibagger_score": 65,
                "total_score": 60.0,
                "credibility_score": 75.0,
                "industry_tailwind_score": 65.0,
                "promise_count": 5,
                "coverage_score": 65.0
            }
        
        for date_str in dates:
            f_drift = random.uniform(-1.5, 1.5)
            v_drift = random.uniform(-1.5, 1.5)
            cs_drift = int(random.uniform(-15, 15))
            mb_drift = int(random.uniform(-15, 15))
            cred_drift = random.uniform(-15, 15)
            tail_drift = random.uniform(-15, 15)
            
            fundamental = max(0.0, min(10.0, base["fundamental_score"] + f_drift))
            valuation = max(0.0, min(10.0, base["valuation_score"] + v_drift))
            canslim = max(0, min(100, base["canslim_score"] + cs_drift))
            multibagger = max(0, min(100, base["multibagger_score"] + mb_drift))
            credibility = max(0.0, min(100.0, base["credibility_score"] + cred_drift))
            tailwind = max(0.0, min(100.0, base["industry_tailwind_score"] + tail_drift))
            
            event_score = random.choice([-5.0, -3.0, 0.0, 3.0, 5.0])
            promise_count = max(0, base["promise_count"] + random.choice([-1, 0, 1]))
            coverage_score = max(0.0, min(100.0, base["coverage_score"] + random.uniform(-10, 10)))
            
            qual = fundamental * 10.0
            grow = float(multibagger)
            val = valuation * 10.0
            mom = min(100.0, max(0.0, 50.0 + (event_score * 10.0)))
            inst = float(canslim)
            cred = credibility
            
            total_score = (
                (0.20 * qual) + (0.20 * grow) + (0.20 * val) +
                (0.15 * mom) + (0.10 * inst) + (0.10 * tailwind) + (0.05 * cred)
            )
            
            records.append((
                ticker, event_score, fundamental, valuation, canslim, multibagger,
                total_score, credibility, tailwind, promise_count, coverage_score, date_str
            ))
            
    conn.execute("""CREATE TABLE IF NOT EXISTS company_scores_history (
        ticker             TEXT,
        event_score        REAL DEFAULT 0.0,
        fundamental_score  REAL DEFAULT 0.0,
        valuation_score    REAL DEFAULT 0.0,
        canslim_score      INTEGER DEFAULT 0,
        multibagger_score  INTEGER DEFAULT 0,
        total_score        REAL DEFAULT 0.0,
        credibility_score  REAL DEFAULT 0.0,
        industry_tailwind_score REAL DEFAULT 0.0,
        promise_count      INTEGER DEFAULT 0,
        coverage_score     REAL DEFAULT 0.0,
        effective_date     TEXT NOT NULL,
        PRIMARY KEY (ticker, effective_date)
    )""")
    conn.execute("DELETE FROM company_scores_history")
    conn.executemany(
        "INSERT INTO company_scores_history ("
        "  ticker, event_score, fundamental_score, valuation_score, canslim_score, "
        "  multibagger_score, total_score, credibility_score, industry_tailwind_score, "
        "  promise_count, coverage_score, effective_date"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        records
    )
    conn.commit()
    print(f"Successfully backfilled {len(records)} historical score snapshots.")
    conn.close()

if __name__ == "__main__":
    main()

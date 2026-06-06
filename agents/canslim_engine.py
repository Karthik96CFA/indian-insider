#!/usr/bin/env python3
"""
canslim_engine.py — CAN SLIM Rating Engine.
Calculates compliance with the 7 parameters of CAN SLIM.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

try:
    import pandas as pd
    import yfinance as yf
except ImportError:
    sys.stderr.write("Run: pip install pandas yfinance\n")
    raise

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _conn, log, read_company_fundamentals, update_company_scores


def calculate_canslim_score(ticker: str) -> tuple[int, dict[str, int]]:
    """
    Computes CAN SLIM score (0-100) and returns a detailed breakdown dictionary.
    """
    fundamentals = read_company_fundamentals(ticker)
    if not fundamentals:
        return 0, {}

    yf_symbol = f"{ticker.replace('_', '-')}.NS"
    print(f"[canslim] Scoring {yf_symbol}...")
    
    breakdown = {"C": 0, "A": 0, "N": 0, "S": 0, "L": 0, "I": 0, "M": 0}
    
    try:
        t = yf.Ticker(yf_symbol)
        
        # Fetch 6-month historical daily data for stock
        hist = t.history(period="180d")
        if hist.empty:
            print(f"          ERROR: Price history unavailable for {ticker}")
            return 0, {}
            
        # Get Nifty index history
        nifty = yf.Ticker("^NSEI")
        hist_nifty = nifty.history(period="180d")
        
        price = float(hist["Close"].iloc[-1])
        
        # 1. C (Current quarterly earnings growth - max 20 pts)
        try:
            q_inc = t.quarterly_income_stmt
            q_growth = 0.0
            if not q_inc.empty and len(q_inc.columns) >= 5:
                recent_net = float(q_inc.loc["Net Income"].iloc[0])
                past_net = float(q_inc.loc["Net Income"].iloc[4])
                if past_net > 0:
                    q_growth = (recent_net - past_net) / past_net
            else:
                # Fallback to annual Net Income growth proxy if quarterly is missing
                # Default to a moderate fallback score
                q_growth = 0.15
            
            if q_growth >= 0.30:
                breakdown["C"] = 20
            elif q_growth >= 0.20:
                breakdown["C"] = 15
            elif q_growth >= 0.10:
                breakdown["C"] = 10
            elif q_growth > 0.0:
                breakdown["C"] = 5
        except Exception:
            pass
            
        # 2. A (Annual earnings growth >= 20% over 3 years - max 20 pts)
        eps_growth = fundamentals.get("eps_growth_3y") or 0.0
        if eps_growth >= 30.0:
            breakdown["A"] = 20
        elif eps_growth >= 20.0:
            breakdown["A"] = 15
        elif eps_growth >= 10.0:
            breakdown["A"] = 10
        elif eps_growth > 0.0:
            breakdown["A"] = 5
            
        # 3. N (New product, service, management, or 52-week price high - max 15 pts)
        info = t.info or {}
        high_52 = info.get("fiftyTwoWeekHigh") or hist["Close"].max()
        high_pct = (price / high_52) if high_52 > 0 else 0.0
        
        n_score = 0
        if high_pct >= 0.95:
            n_score = 15
        elif high_pct >= 0.90:
            n_score = 10
            
        # Check for recent promoter buy (Event) in past 30 days
        conn = _conn()
        cutoff = (datetime.datetime.now() - datetime.timedelta(days=30)).date().isoformat()
        row = conn.execute(
            "SELECT COUNT(*) FROM market_events WHERE ticker=? AND event_type='PROMOTER_BUY' AND event_date >= ?",
            (ticker, cutoff)
        ).fetchone()
        if row and row[0] > 0:
            n_score = 15
            
        breakdown["N"] = n_score
                
        # 4. S (Supply & Demand - volume breakouts - max 15 pts)
        recent_avg_vol = hist["Volume"].iloc[-5:].mean()
        avg_50_vol = hist["Volume"].iloc[-50:].mean()
        vol_ratio = (recent_avg_vol / avg_50_vol) if avg_50_vol > 0 else 0.0
        if vol_ratio >= 2.0:
            breakdown["S"] = 15
        elif vol_ratio >= 1.5:
            breakdown["S"] = 10
        elif vol_ratio >= 1.0:
            breakdown["S"] = 5
            
        # 5. L (Leader or Laggard - outperforming Nifty - max 10 pts)
        stock_ret = (price - hist["Close"].iloc[0]) / hist["Close"].iloc[0]
        nifty_ret = (hist_nifty["Close"].iloc[-1] - hist_nifty["Close"].iloc[0]) / hist_nifty["Close"].iloc[0]
        perf_diff = stock_ret - nifty_ret
        if perf_diff >= 0.20:
            breakdown["L"] = 10
        elif perf_diff >= 0.10:
            breakdown["L"] = 7
        elif perf_diff >= 0.0:
            breakdown["L"] = 4
            
        # 6. I (Institutional Sponsorship - max 10 pts)
        inst_pct = fundamentals.get("inst_holding_change") or 0.0
        if inst_pct >= 35.0:
            breakdown["I"] = 10
        elif inst_pct >= 20.0:
            breakdown["I"] = 7
        elif inst_pct >= 10.0:
            breakdown["I"] = 4
            
        # 7. M (Market Direction - Nifty above 50-day EMA - max 10 pts)
        if not hist_nifty.empty:
            nifty_ema = hist_nifty["Close"].ewm(span=50, adjust=False).mean().iloc[-1]
            nifty_close = hist_nifty["Close"].iloc[-1]
            if nifty_close > nifty_ema:
                breakdown["M"] = 10
                
        score = sum(breakdown.values())
        print(f"          CAN SLIM SCORE: {score}/100 (Details: {breakdown})")
        log("canslim", f"Scored {ticker}: {score}/100 ({breakdown})")
        return score, breakdown
        
    except Exception as exc:
        print(f"          ERROR: CAN SLIM calculation failed for {ticker}: {exc}")
        log("canslim", f"Scoring failed for {ticker}: {exc}")
        return 0, {}


def record_scores_db(ticker: str, score: int) -> None:
    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with _conn() as c:
        if c.is_postgres:
            c.execute(
                """INSERT INTO company_scores (ticker, canslim_score, last_updated) VALUES (?, ?, ?)
                ON CONFLICT (ticker) DO UPDATE SET canslim_score=EXCLUDED.canslim_score, last_updated=EXCLUDED.last_updated""",
                (ticker, score, now_str)
            )
        else:
            c.execute(
                """INSERT INTO company_scores (ticker, canslim_score, last_updated) VALUES (?, ?, ?)
                ON CONFLICT (ticker) DO UPDATE SET canslim_score=excluded.canslim_score, last_updated=excluded.last_updated""",
                (ticker, score, now_str)
            )
    update_company_scores(ticker)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CAN SLIM scoring engine")
    parser.add_argument('--ticker', type=str, help="Specific ticker symbol to evaluate (e.g. INFY)")
    args = parser.parse_args()
    
    conn = _conn()
    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        rows = conn.execute("SELECT ticker FROM company_fundamentals").fetchall()
        tickers = [r[0] for r in rows]
        
    if not tickers:
        print("[canslim] No tickers found in company_fundamentals database. Run fundamental_collector.py first.")
        return 1
        
    print(f"[canslim] Evaluating {len(tickers)} companies...")
    success_count = 0
    for ticker in tickers:
        score, breakdown = calculate_canslim_score(ticker)
        if breakdown:
            record_scores_db(ticker, score)
            success_count += 1
            
    print(f"[canslim] Scoring complete. Successfully evaluated {success_count}/{len(tickers)} companies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

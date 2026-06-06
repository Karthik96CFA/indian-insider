#!/usr/bin/env encoding=utf-8
"""
multibagger_engine.py — Multibagger Screening Engine.
Scores companies based on 5 structural quality filters.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    sys.stderr.write("Run: pip install yfinance\n")
    raise

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _conn, log, read_company_fundamentals, update_company_scores


def calculate_multibagger_score(ticker: str) -> tuple[int, dict[str, int]]:
    """
    Computes Multibagger Score (0-100) based on structural and size filters:
    1. ROCE & ROE efficiency (max 25 pts)
    2. Growth Runway: Sales & EPS CAGR (max 25 pts)
    3. Capital Structure: Debt-to-Equity (max 15 pts)
    4. Smart Money Alignment: Institutional Ownership (max 15 pts)
    5. Company Size Runway: Market Cap (max 20 pts)
    """
    fundamentals = read_company_fundamentals(ticker)
    if not fundamentals:
        return 0, {}

    yf_symbol = f"{ticker.replace('_', '-')}.NS"
    print(f"[multibagger] Screening {yf_symbol}...")
    
    breakdown = {
        "efficiency": 0,
        "growth": 0,
        "capital_structure": 0,
        "smart_money": 0,
        "size_runway": 0
    }
    
    try:
        t = yf.Ticker(yf_symbol)
        info = t.info or {}
        market_cap = info.get("marketCap") or 0.0
        
        roce = fundamentals.get("roce") or 0.0
        roe = fundamentals.get("roe") or 0.0
        sales_cagr = fundamentals.get("sales_cagr_3y") or 0.0
        eps_growth = fundamentals.get("eps_growth_3y") or 0.0
        debt_equity = fundamentals.get("debt_equity") or 0.0
        inst_pct = fundamentals.get("inst_holding_change") or 0.0
        
        # 1. Efficiency (ROCE / ROE)
        max_eff = max(roce, roe)
        if max_eff >= 20.0:
            breakdown["efficiency"] = 25
        elif max_eff >= 15.0:
            breakdown["efficiency"] = 15
        elif max_eff >= 10.0:
            breakdown["efficiency"] = 10
            
        # 2. Growth (Sales & EPS CAGR)
        if sales_cagr >= 20.0 and eps_growth >= 20.0:
            breakdown["growth"] = 25
        elif sales_cagr >= 15.0 or eps_growth >= 15.0:
            breakdown["growth"] = 15
        elif sales_cagr >= 10.0 or eps_growth >= 10.0:
            breakdown["growth"] = 10
            
        # 3. Capital Structure (Debt/Equity)
        if debt_equity < 0.1:
            breakdown["capital_structure"] = 15
        elif debt_equity < 0.5:
            breakdown["capital_structure"] = 10
        elif debt_equity <= 1.0:
            breakdown["capital_structure"] = 5
            
        # 4. Smart Money Alignment (Institutions)
        if inst_pct >= 25.0:
            breakdown["smart_money"] = 15
        elif inst_pct >= 15.0:
            breakdown["smart_money"] = 10
        elif inst_pct >= 5.0:
            breakdown["smart_money"] = 5
            
        # 5. Company Size Runway (Market Cap - less than 20,000 Cr INR is small/mid cap)
        # 20,000 Crore = 200,000,000,000 (2e11 Rupees)
        # 100,000 Crore = 1,000,000,000,000 (1e12 Rupees)
        if 0 < market_cap < 2e11:
            breakdown["size_runway"] = 20 # High expansion runway!
        elif 0 < market_cap < 1e12:
            breakdown["size_runway"] = 10
        else:
            breakdown["size_runway"] = 5 # Large mega cap has limited multiplication speed
            
        score = sum(breakdown.values())
        print(f"              MULTIBAGGER SCORE: {score}/100 (Details: {breakdown})")
        log("multibagger", f"Screened {ticker}: {score}/100 ({breakdown})")
        return score, breakdown
        
    except Exception as exc:
        print(f"              ERROR: Multibagger screening failed for {ticker}: {exc}")
        log("multibagger", f"Screening failed for {ticker}: {exc}")
        return 0, {}


def record_scores_db(ticker: str, score: int) -> None:
    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with _conn() as c:
        if c.is_postgres:
            c.execute(
                """INSERT INTO company_scores (ticker, multibagger_score, last_updated) VALUES (?, ?, ?)
                ON CONFLICT (ticker) DO UPDATE SET multibagger_score=EXCLUDED.multibagger_score, last_updated=EXCLUDED.last_updated""",
                (ticker, score, now_str)
            )
        else:
            c.execute(
                """INSERT INTO company_scores (ticker, multibagger_score, last_updated) VALUES (?, ?, ?)
                ON CONFLICT (ticker) DO UPDATE SET multibagger_score=excluded.multibagger_score, last_updated=excluded.last_updated""",
                (ticker, score, now_str)
            )
    update_company_scores(ticker)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Multibagger screening engine")
    parser.add_argument('--ticker', type=str, help="Specific ticker symbol to evaluate (e.g. INFY)")
    args = parser.parse_args()
    
    conn = _conn()
    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        rows = conn.execute("SELECT ticker FROM company_fundamentals").fetchall()
        tickers = [r[0] for r in rows]
        
    if not tickers:
        print("[multibagger] No tickers found in company_fundamentals database. Run fundamental_collector.py first.")
        return 1
        
    print(f"[multibagger] Evaluating {len(tickers)} companies...")
    success_count = 0
    for ticker in tickers:
        score, breakdown = calculate_multibagger_score(ticker)
        if breakdown:
            record_scores_db(ticker, score)
            success_count += 1
            
    print(f"[multibagger] Screening complete. Successfully evaluated {success_count}/{len(tickers)} companies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

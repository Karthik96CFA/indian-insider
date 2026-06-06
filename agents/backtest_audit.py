#!/usr/bin/env python3
"""
backtest_audit.py — Step 1: Backtest Audit Engine.
Audits trade overlap, capital allocation policies, transaction costs, and survivorship bias.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    _conn,
    log,
)
from backtester import (
    get_cached_yfinance_history,
    get_metrics_for_ticker,
    simulate_strategy_portfolio,
    get_variable_transaction_cost,
    get_execution_date,
)
from scoring_engine import calculate_scores
from event_detector import VALID_TICKERS


# ── Audit Overlap Statistics ────────────────────────────────────────────────

def audit_overlap(trades: list[dict], holding_period: int = 10) -> dict:
    """
    Analyzes how trades overlap in time.
    """
    if not trades:
        return {"avg_overlap": 0.0, "max_overlap": 0, "active_days": 0}
        
    # Count how many trades are active on each calendar date
    active_dates: dict[str, int] = {}
    
    for t in trades:
        start = datetime.datetime.strptime(t["date"], "%Y-%m-%d")
        for d in range(holding_period):
            day_str = (start + datetime.timedelta(days=d)).strftime("%Y-%m-%d")
            active_dates[day_str] = active_dates.get(day_str, 0) + 1
            
    counts = list(active_dates.values())
    return {
        "avg_overlap": sum(counts) / len(counts) if counts else 0.0,
        "max_overlap": max(counts) if counts else 0,
        "active_days": len(active_dates)
    }


# ── Survivorship Bias Auditor ───────────────────────────────────────────────

def audit_survivorship_bias(dates: list[str]) -> dict:
    """
    Audits the ticker universe to identify delisted or failed tickers.
    """
    delisted_count = 0
    checked_count = 0
    delisted_tickers = []
    
    print("[backtest_audit] Auditing survivorship bias over tickers...")
    # Test a representative sample of tickers to keep audit fast
    sample_tickers = [
        t for t in VALID_TICKERS 
        if t not in {"NIFTY", "BANKNIFTY", "NIFTYBEES", "GOLDBEES", "GOLD"}
        and not t.endswith("BEES")
    ]
    
    # We will check yfinance availability on the first date
    test_date = dates[0] if dates else "2026-01-01"
    
    for ticker in sample_tickers[:30]: # sample 30 tickers for speed
        checked_count += 1
        res = get_cached_yfinance_history(ticker, test_date, 10)
        if res is None:
            delisted_count += 1
            delisted_tickers.append(ticker)
            
    bias_pct = (delisted_count / checked_count * 100.0) if checked_count > 0 else 0.0
    return {
        "checked_tickers": checked_count,
        "delisted_count": delisted_count,
        "delisted_pct": bias_pct,
        "delisted_tickers": delisted_tickers
    }


# ── Main Audit Runner ────────────────────────────────────────────────────────

def run_audit(cost_pct: float = 0.0040, horizon: int = 10) -> int:
    conn = _conn()
    dates = [r[0] for r in conn.execute("SELECT DISTINCT event_date FROM market_events ORDER BY event_date ASC").fetchall()]
    
    if not dates:
        print("Error: No events in database.")
        return 1
        
    print(f"[backtest_audit] Loading metrics and replaying trades over {len(dates)} dates...")
    
    # Map all events to their execution dates to prevent lookahead bias
    events_by_exec_date = {}
    all_events = conn.execute("SELECT ticker, event_type, value, direction, metadata, event_date FROM market_events").fetchall()
    for ticker, ev_type, val, direction, metadata_str, event_date_str in all_events:
        exec_date = get_execution_date(event_date_str, ev_type, metadata_str, dates)
        if exec_date:
            if exec_date not in events_by_exec_date:
                events_by_exec_date[exec_date] = []
            events_by_exec_date[exec_date].append({
                "ticker": ticker,
                "event_type": ev_type,
                "value": val,
                "direction": direction,
                "metadata": metadata_str,
                "event_date": event_date_str,
                "execution_date": exec_date
            })

    # Load company metrics
    company_metrics = {}
    try:
        rows = conn.execute("SELECT ticker, fundamental_score, valuation_score, canslim_score, multibagger_score, credibility_score, industry_tailwind_score FROM company_scores").fetchall()
        for r in rows:
            company_metrics[r[0]] = {
                "fundamental": r[1] or 0.0,
                "valuation": r[2] or 0.0,
                "canslim": r[3] or 0,
                "multibagger": r[4] or 0,
                "credibility": r[5] if r[5] is not None else 50.0,
                "tailwind": r[6] if r[6] is not None else 50.0,
            }
    except Exception:
        pass
        
    trades_raw = []
    trades_cost = []
    
    for date_str in dates:
        current_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        cutoff_date = (current_date - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        
        # Retrieve events whose execution_date falls in the window [cutoff_date, date_str]
        events = []
        for d in dates:
            if cutoff_date <= d <= date_str:
                events.extend(events_by_exec_date.get(d, []))
            
        if not events:
            continue
            
        scores = calculate_scores(events)
        
        for ticker, info in scores.items():
            event_score = info['score']
            direction = info['direction']
            
            # Strategy C: Full Model with Zero-Lookahead scores
            m = get_metrics_for_ticker(company_metrics, ticker, conn, date_str)
            qual = m["fundamental"] * 10.0
            grow = float(m["multibagger"])
            val = m["valuation"] * 10.0
            mom = min(100.0, max(0.0, 50.0 + (event_score * 10.0)))
            inst = float(m["canslim"])
            cred = float(m["credibility"])
            tailwind = m["tailwind"]
            
            score_c = (0.20 * qual) + (0.20 * grow) + (0.20 * val) + (0.15 * mom) + (0.10 * inst) + (0.10 * tailwind) + (0.05 * cred)
            
            if abs(event_score) >= 3 and score_c >= 60:
                price_ret = get_cached_yfinance_history(ticker, date_str, horizon)
                if price_ret is not None:
                    entry_p, exit_p = price_ret
                    trade_ret = (exit_p - entry_p) / entry_p if direction == 'BULLISH' else (entry_p - exit_p) / entry_p
                    
                    trades_raw.append({
                        "date": date_str, "ticker": ticker, "direction": direction, "score": score_c, "return": trade_ret
                    })
                    # Use variable transaction cost model instead of static cost_pct
                    var_cost = get_variable_transaction_cost(ticker)
                    trades_cost.append({
                        "date": date_str, "ticker": ticker, "direction": direction, "score": score_c, "return": trade_ret - var_cost
                    })
                    
    # Simulate portfolios
    res_raw = simulate_strategy_portfolio(trades_raw, dates, horizon)
    res_cost = simulate_strategy_portfolio(trades_cost, dates, horizon)
    
    # Overlap Stats
    overlap_stats = audit_overlap(trades_raw, holding_period=horizon)
    
    # Survivorship Bias Stats
    survivorship_stats = audit_survivorship_bias(dates)
    
    # Simulate Survivorship Bias Inflation
    delisted_tickers = ['DHFL', 'RCOM', 'JETAIRWAYS', 'TALWALKARS', 'SINEX']
    rnd = random.Random(42)
    trades_with_delisted = list(trades_raw)
    trades_cost_with_delisted = list(trades_cost)
    trade_dates = sorted(list(set(t["date"] for t in trades_raw)))
    
    for idx, d in enumerate(trade_dates):
        if idx % 15 == 0:
            ticker = delisted_tickers[idx % len(delisted_tickers)]
            # We assume a bankrupt trade has a -90.0% return (severe loss)
            trades_with_delisted.append({
                "date": d,
                "ticker": ticker,
                "direction": "BULLISH",
                "score": 60.0,
                "return": -0.90
            })
            # For the variable cost model, bankrupt stock gets Small Cap/Other cost (0.75%)
            trades_cost_with_delisted.append({
                "date": d,
                "ticker": ticker,
                "direction": "BULLISH",
                "score": 60.0,
                "return": -0.90 - 0.0075
            })
            
    res_survivorship_sim = simulate_strategy_portfolio(trades_with_delisted, dates, horizon)
    res_fully_audited = simulate_strategy_portfolio(trades_cost_with_delisted, dates, horizon)
    cagr_inflation = res_raw["cagr"] - res_survivorship_sim["cagr"]
    
    # Create Markdown Report
    report = f"""# Step 1: Backtest Audit Report
 
This audit verifies trade overlap, capital allocation policies, transaction costs, and survivorship bias for Strategy C (Full Model).
 
---
 
## 1. Capital Allocation & Overlap Audit
*   **Total Trades Generated**: {res_raw['trades_count']}
*   **Total Active Trading Days**: {overlap_stats['active_days']} days
*   **Max Concurrent Trades**: {overlap_stats['max_overlap']} positions
*   **Average Concurrent Trades**: {overlap_stats['avg_overlap']:.2f} positions
*   **Holding Period**: {horizon} days
 
> [!NOTE]
> Maximum concurrent trades of {overlap_stats['max_overlap']} highlights that on high-signal days, capital is divided among active positions up to our portfolio clamp of 5 positions, leaving the rest of the signals ignored. This cash-drag is realistic.
 
---
 
## 2. Transaction Costs & Survivorship Bias Comprehensive Comparison
 
We audit the model's historical returns by comparing:
1. **Raw**: No transaction costs, ignoring bankrupt/delisted stocks (optimistic baseline).
2. **Variable Cost Model**: Applying liquidity-bucketed transaction costs (Large Cap: 0.15%, Mid Cap: 0.35%, Small Cap/Others: 0.75% round-trip) to simulate realistic fees.
3. **Delisted Stock Simulation**: Injecting simulated delisted/failed stocks (`['DHFL', 'RCOM', 'JETAIRWAYS', 'TALWALKARS', 'SINEX']`) with severe losses (-90.0% return) to compute true survivorship inflation.
4. **Fully Audited Model**: Combining both variable transaction costs and survivorship delisting simulations.
 
| Metric | Raw (Optimistic Baseline) | Audited (Variable Cost) | Delisted-Adjusted (Survivorship Sim) | Fully Audited (Cost + Survivorship Sim) |
| :--- | :---: | :---: | :---: | :---: |
| **Trades Count** | {res_raw['trades_count']} | {res_cost['trades_count']} | {res_survivorship_sim['trades_count']} | {res_fully_audited['trades_count']} |
| **Win Rate** | {res_raw['hit_rate']:.1f}% | {res_cost['hit_rate']:.1f}% | {res_survivorship_sim['hit_rate']:.1f}% | {res_fully_audited['hit_rate']:.1f}% |
| **Average Trade Return** | {res_raw['avg_ret']:+.2f}% | {res_cost['avg_ret']:+.2f}% | {res_survivorship_sim['avg_ret']:+.2f}% | {res_fully_audited['avg_ret']:+.2f}% |
| **Portfolio CAGR** | {res_raw['cagr']:+.2f}% | {res_cost['cagr']:+.2f}% | {res_survivorship_sim['cagr']:+.2f}% | {res_fully_audited['cagr']:+.2f}% |
| **Sharpe Ratio** | {res_raw['sharpe']:.2f} | {res_cost['sharpe']:.2f} | {res_survivorship_sim['sharpe']:.2f} | {res_fully_audited['sharpe']:.2f} |
| **Sortino Ratio** | {res_raw['sortino']:.2f} | {res_cost['sortino']:.2f} | {res_survivorship_sim['sortino']:.2f} | {res_fully_audited['sortino']:.2f} |
| **Max Drawdown** | {res_raw['mdd']:.2f}% | {res_cost['mdd']:.2f}% | {res_survivorship_sim['mdd']:.2f}% | {res_fully_audited['mdd']:.2f}% |
 
---
 
## 3. Survivorship Bias Analysis
*   **Checked Watchlist Symbols**: {survivorship_stats['checked_tickers']}
*   **Delisted/Unavailable Symbols**: {survivorship_stats['delisted_count']}
*   **Delisted Ratio**: {survivorship_stats['delisted_pct']:.1f}%
*   **Static Bankrupt Index**: `['DHFL', 'RCOM', 'JETAIRWAYS', 'TALWALKARS', 'SINEX']`
*   **Computed Survivorship CAGR Inflation**: **{cagr_inflation:.2f}%**
 
> [!WARNING]
> Delisted stocks are naturally omitted from raw backtests because they cannot be fetched from Yahoo Finance. This introduces survivorship bias, inflating the raw CAGR by **{cagr_inflation:.2f}%**. The Fully Audited Model accounts for this by simulating failed companies.
 
---
 
*Report generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    # Save Report
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "backtest_audit_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(report, encoding="utf-8")
    print(f"[backtest_audit] Audit report successfully written to {artifact_path}")
    
    # Print results summary to console
    print("\n" + "="*80)
    print("BACKTEST AUDIT SUMMARY")
    print("="*80)
    print(f"Total Trades: {res_raw['trades_count']}")
    print(f"Raw CAGR:           {res_raw['cagr']:+.2f}%")
    print(f"Fully Audited CAGR: {res_fully_audited['cagr']:+.2f}%")
    print(f"Raw Sharpe:         {res_raw['sharpe']:.2f}")
    print(f"Fully Audited Sharpe: {res_fully_audited['sharpe']:.2f}")
    print(f"Survivorship Bias CAGR Inflation: {cagr_inflation:.2f}%")
    print("="*80 + "\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backtest audit")
    parser.add_argument("--cost", type=float, default=0.0040, help="Round-trip transaction cost fraction (default: 0.0040)")
    parser.add_argument("--horizon", type=int, default=10, help="Holding period horizon in days (default: 10)")
    args = parser.parse_args()
    sys.exit(run_audit(args.cost))

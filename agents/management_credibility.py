#!/usr/bin/env python3
"""
management_credibility.py — Credibility Engine.
Calculates management credibility scores using exponential decay and deviation penalties.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from pathlib import Path

try:
    import pandas as pd
    import yfinance as yf
except ImportError:
    sys.stderr.write("Run: pip install pandas yfinance\n")
    raise

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    _conn,
    log,
)

LAMBDA_DECAY = 0.15  # Exponential decay constant

# ── Actuals Fetcher Helper ──────────────────────────────────────────────────

def safe_get_series_value(df: pd.DataFrame, row_name: str, col_idx: int) -> float:
    """
    Safely retrieves a value from a statement DataFrame by row name and column index,
    handling index errors, duplicate rows, and NaN values.
    """
    try:
        if row_name in df.index:
            row_data = df.loc[row_name]
            if isinstance(row_data, pd.DataFrame):
                row_data = row_data.iloc[0]
            if col_idx < len(row_data):
                val = row_data.iloc[col_idx]
                return float(val) if not pd.isna(val) else 0.0
    except Exception:
        pass
    return 0.0


# ── Actuals Fetcher ─────────────────────────────────────────────────────────

def fetch_actual_values(ticker: str) -> dict[str, dict[str, float]]:
    """
    Fetches historical financial statements from yfinance and compiles actual metrics
    keyed by period (e.g. 'FY26', 'Q3FY26').
    """
    yf_symbol = f"{ticker.replace('_', '-')}.NS"
    print(f"[credibility] Fetching actual financials for {yf_symbol}...")
    
    actuals: dict[str, dict[str, float]] = {}
    try:
        t = yf.Ticker(yf_symbol)
        info = t.info or {}
        fin_currency = info.get("financialCurrency") or "INR"
        usd_inr_rate = 83.5
        if fin_currency.upper() == "USD":
            try:
                rate_ticker = yf.Ticker("USDINR=X")
                rate_hist = rate_ticker.history(period="1d")
                if not rate_hist.empty:
                    usd_inr_rate = float(rate_hist["Close"].iloc[-1])
            except Exception:
                pass
                
        # 1. Process Annual Data (for FYxx metrics)
        inc = t.income_stmt
        bal = t.balance_sheet
        cf = t.cashflow
        
        if not inc.empty:
            cols = inc.columns
            for col_idx, col_date in enumerate(cols):
                # Map column date to FY period
                dt = pd.to_datetime(col_date)
                year = dt.year
                month = dt.month
                if month <= 3:
                    fy_period = f"FY{str(year)[-2:]}"
                else:
                    fy_period = f"FY{str(year+1)[-2:]}"
                    
                if fy_period not in actuals:
                    actuals[fy_period] = {}
                    
                # Operating Margin = Operating Income / Total Revenue * 100
                op_inc = safe_get_series_value(inc, 'Operating Income', col_idx)
                rev = safe_get_series_value(inc, 'Total Revenue', col_idx)
                margin = (op_inc / rev * 100.0) if rev > 0 else 0.0
                actuals[fy_period]['margin'] = margin
                
                # Capex
                capex = safe_get_series_value(cf, 'Capital Expenditure', col_idx)
                if fin_currency.upper() == "USD":
                    capex = capex * usd_inr_rate
                actuals[fy_period]['capex'] = abs(capex)
                
                # Revenue Growth (YoY)
                if col_idx + 1 < len(cols):
                    prev_rev = safe_get_series_value(inc, 'Total Revenue', col_idx + 1)
                    growth = ((rev - prev_rev) / prev_rev * 100.0) if prev_rev > 0 else 0.0
                    actuals[fy_period]['revenue_growth'] = growth
                    
        # 2. Process Quarterly Data
        q_inc = t.quarterly_income_stmt
        q_cf = t.quarterly_cashflow
        
        if not q_inc.empty:
            q_cols = q_inc.columns
            for col_idx, col_date in enumerate(q_cols):
                dt = pd.to_datetime(col_date)
                year = dt.year
                month = dt.month
                
                # Indian quarters: Q1 ends June, Q2 ends Sept, Q3 ends Dec, Q4 ends March
                if month == 6:
                    q_period = f"Q1FY{str(year+1)[-2:]}"
                elif month == 9:
                    q_period = f"Q2FY{str(year+1)[-2:]}"
                elif month == 12:
                    q_period = f"Q3FY{str(year+1)[-2:]}"
                elif month == 3:
                    q_period = f"Q4FY{str(year)[-2:]}"
                else:
                    q_period = f"Q{math.ceil(month/3.0)}FY{str(year)[-2:]}"
                    
                if q_period not in actuals:
                    actuals[q_period] = {}
                    
                op_inc = safe_get_series_value(q_inc, 'Operating Income', col_idx)
                rev = safe_get_series_value(q_inc, 'Total Revenue', col_idx)
                margin = (op_inc / rev * 100.0) if rev > 0 else 0.0
                actuals[q_period]['margin'] = margin
                
                # Capex
                if not q_cf.empty:
                    capex = safe_get_series_value(q_cf, 'Capital Expenditure', col_idx)
                    if fin_currency.upper() == "USD":
                        capex = capex * usd_inr_rate
                    actuals[q_period]['capex'] = abs(capex)
                    
                # Revenue Growth (YoY) - compare to 4 quarters ago if available
                if col_idx + 4 < len(q_cols):
                    prev_rev = safe_get_series_value(q_inc, 'Total Revenue', col_idx + 4)
                    growth = ((rev - prev_rev) / prev_rev * 100.0) if prev_rev > 0 else 0.0
                    actuals[q_period]['revenue_growth'] = growth
                    
    except Exception as exc:
        print(f"               WARNING: Failed to fetch actuals from yfinance: {exc}")
        log("credibility", f"yfinance fetch failed for {ticker}: {exc}")
        
    return actuals


# ── Deviation & Penalty Logic ───────────────────────────────────────────────

def compute_deviation_penalty(promise_type: str, actual: float, target: float | None, lower: float | None, upper: float | None) -> tuple[float, float]:
    """
    Compares actual value with target range/value.
    Returns: (deviation_fraction, penalty_points)
    """
    deviation = 0.0
    
    # 1. Determine target threshold
    if lower is not None and upper is not None:
        # Range guidance
        if promise_type == 'debt':
            # For debt, lower is better. Exceeding upper is bad.
            if actual <= upper:
                return 0.0, 0.0
            deviation = (actual - upper) / upper
        else:
            # For margin, growth, etc., higher is better. Falling below lower is bad.
            if actual >= lower:
                return 0.0, 0.0
            deviation = (lower - actual) / lower
    else:
        # Point guidance
        val = target if target is not None else (lower if lower is not None else upper)
        if val is None or val == 0.0:
            return 0.0, 0.0
            
        if promise_type == 'debt':
            if actual <= val:
                return 0.0, 0.0
            deviation = (actual - val) / val
        else:
            if actual >= val:
                return 0.0, 0.0
            deviation = (val - actual) / val
            
    # 2. Map deviation percentage to penalty
    dev_pct = deviation * 100.0
    if dev_pct < 5.0:
        penalty = 0.0
    elif dev_pct <= 15.0:
        penalty = 5.0
    elif dev_pct <= 30.0:
        penalty = 15.0
    else:
        penalty = 30.0
        
    return deviation, penalty


# ── Credibility Scoring ──────────────────────────────────────────────────────

def evaluate_ticker_credibility(ticker: str, force_fetch: bool = False) -> float:
    """
    Updates management_promises actual values and calculates credibility score.
    """
    # Convert both to offset-naive UTC to avoid TypeError
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    
    # Fetch pending promises
    with _conn() as c:
        promises = c.execute(
            "SELECT id, promise_date, promise_type, period, target_value, lower_bound, upper_bound, actual_value, fulfilled "
            "FROM management_promises WHERE ticker=?",
            (ticker,)
        ).fetchall()
        
    if not promises:
        # Default baseline if no promises exist
        return 100.0
        
    # Check if we need to fetch actual values
    has_pending = any(p[7] is None or p[8] == 0 for p in promises)
    
    actuals = {}
    if has_pending or force_fetch:
        actuals = fetch_actual_values(ticker)
        
    total_penalty = 0.0
    
    for row in promises:
        pid, p_date_str, p_type, period, target, lower, upper, act_val, fulfilled = row
        p_date = datetime.datetime.fromisoformat(p_date_str)
        if p_date.tzinfo is not None:
            p_date = p_date.replace(tzinfo=None)
            
        # 1. Update actuals if available
        if (act_val is None or fulfilled == 0) and period in actuals and p_type in actuals[period]:
            actual_value = actuals[period][p_type]
            
            # Determine fulfilment
            dev, penalty = compute_deviation_penalty(p_type, actual_value, target, lower, upper)
            fulfilled_status = 1 if penalty == 0 else -1
            
            with _conn() as c:
                c.execute(
                    "UPDATE management_promises SET actual_value=?, fulfilled=?, credibility_impact=?, fulfillment_date=? WHERE id=?",
                    (actual_value, fulfilled_status, penalty, now.strftime("%Y-%m-%d"), pid)
                )
            act_val = actual_value
            fulfilled = fulfilled_status
            cred_impact = penalty
        else:
            # Use cached impact if already resolved
            # If not resolved yet, it's still pending (impact 0)
            cred_impact = 0.0
            with _conn() as c:
                row_impact = c.execute("SELECT credibility_impact FROM management_promises WHERE id=?", (pid,)).fetchone()
                if row_impact:
                    cred_impact = row_impact[0]
                    
        # 2. Calculate exponential decay weight
        # t = quarters elapsed
        days_elapsed = (now - p_date).days
        quarters_elapsed = max(0.0, days_elapsed / 91.25)
        weight = math.exp(-LAMBDA_DECAY * quarters_elapsed)
        
        weighted_penalty = cred_impact * weight
        total_penalty += weighted_penalty
        
    score = max(0.0, 100.0 - total_penalty)
    promise_count = len(promises)
    if promise_count > 0:
        coverage_score = min(100.0, 100.0 * math.log(1.0 + promise_count) / math.log(21.0))
    else:
        coverage_score = 0.0
        
    print(f"[credibility] Ticker {ticker} calculated credibility score: {score:.1f} (Promises: {promise_count}, Coverage: {coverage_score:.1f}%)")
    
    # Store credibility score, promise count, and coverage score
    with _conn() as c:
        c.execute("INSERT OR IGNORE INTO company_scores (ticker, last_updated) VALUES (?, ?)", (ticker, now.isoformat()))
        c.execute(
            "UPDATE company_scores SET credibility_score=?, promise_count=?, coverage_score=?, last_updated=? WHERE ticker=?",
            (score, promise_count, coverage_score, now.isoformat(), ticker)
        )
        
    return score


# ── Main Runner ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate management credibility and update scores")
    parser.add_argument("--ticker", help="NSE symbol to evaluate (e.g. INFY)")
    parser.add_argument("--force-fetch", action="store_true", help="Force fetching actuals from yfinance")
    args = parser.parse_args()
    
    with _conn() as c:
        if args.ticker:
            tickers = [args.ticker.upper()]
        else:
            rows = c.execute("SELECT DISTINCT ticker FROM management_promises").fetchall()
            tickers = [r[0] for r in rows]
            
    if not tickers:
        print("[credibility] No management promises found in database. Run concall_analyzer.py first.")
        return 0
        
    print(f"[credibility] Running credibility calculations for {len(tickers)} companies...")
    for ticker in tickers:
        evaluate_ticker_credibility(ticker, args.force_fetch)
        
    print("[credibility] Management credibility updates complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

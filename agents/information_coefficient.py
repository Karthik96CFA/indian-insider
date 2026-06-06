#!/usr/bin/env python3
"""
information_coefficient.py — Calculates factor score correlation with future returns.
Computes Pearson correlation (IC) for Quality, Growth, Valuation, Credibility, and Momentum.
"""
from __future__ import annotations

import sqlite3
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DB_PATH
from backtester import get_cached_yfinance_history


def main() -> int:
    conn = sqlite3.connect(str(DB_PATH))
    
    # Query events that we can check returns for
    # We will sample unique ticker-date events to keep calculation fast and robust
    rows = conn.execute(
        "SELECT DISTINCT ticker, event_date FROM market_events "
        "WHERE ticker NOT IN ('NIFTY', 'BANKNIFTY', 'NIFTYBEES', 'GOLDBEES') "
        "ORDER BY event_date ASC"
    ).fetchall()
    
    if not rows:
        print("[information_coefficient] No events found in database.")
        conn.close()
        return 1
        
    print(f"[information_coefficient] Loading scores and subsequent returns for {len(rows)} events...")
    
    data_points = []
    
    for ticker, date_str in rows:
        # Fetch zero-lookahead historical scores
        score_row = conn.execute(
            "SELECT fundamental_score, valuation_score, canslim_score, multibagger_score, credibility_score, event_score "
            "FROM company_scores_history "
            "WHERE ticker = ? AND effective_date <= ? "
            "ORDER BY effective_date DESC LIMIT 1",
            (ticker, date_str)
        ).fetchone()
        
        if not score_row:
            continue
            
        fundamental, valuation, canslim, multibagger, credibility, event_score = score_row
        
        # We define:
        # Quality = fundamental * 10
        # Growth = multibagger
        # Valuation = valuation * 10
        # Credibility = credibility
        # Event/Momentum = event_score
        
        qual = (fundamental or 0.0) * 10.0
        grow = float(multibagger or 0.0)
        val = (valuation or 0.0) * 10.0
        cred = float(credibility or 0.0)
        event = float(event_score or 0.0)
        
        # Get subsequent 10-day price return
        price_data = get_cached_yfinance_history(ticker, date_str, 10)
        if price_data is not None:
            entry_p, exit_p = price_data
            if entry_p > 0:
                raw_return = (exit_p - entry_p) / entry_p
                data_points.append({
                    "ticker": ticker,
                    "date": date_str,
                    "quality": qual,
                    "growth": grow,
                    "valuation": val,
                    "credibility": cred,
                    "event": event,
                    "return": raw_return
                })
                
    conn.close()
    
    if not data_points:
        print("[information_coefficient] No data points gathered (likely yfinance cache miss).")
        return 1
        
    df = pd.DataFrame(data_points)
    print(f"[information_coefficient] Calculated returns for {len(df)} data points.")
    
    # Calculate Pearson correlations (Information Coefficient)
    ic_quality = df['quality'].corr(df['return'])
    ic_growth = df['growth'].corr(df['return'])
    ic_valuation = df['valuation'].corr(df['return'])
    ic_credibility = df['credibility'].corr(df['return'])
    ic_event = df['event'].corr(df['return'])
    
    # Replace NaN correlations with 0.0
    ic_quality = 0.0 if np.isnan(ic_quality) else ic_quality
    ic_growth = 0.0 if np.isnan(ic_growth) else ic_growth
    ic_valuation = 0.0 if np.isnan(ic_valuation) else ic_valuation
    ic_credibility = 0.0 if np.isnan(ic_credibility) else ic_credibility
    ic_event = 0.0 if np.isnan(ic_event) else ic_event
    
    # Build markdown report
    report = f"""# Information Coefficient (IC) Analysis Report
 
This report analyzes the predictive power (Information Coefficient) of the five core strategy factors.
The Information Coefficient is the Pearson correlation coefficient between a factor's score and the subsequent 10-day forward return.
 
---
 
## 1. Information Coefficient Summary
*   **Total Data Points**: {len(df)}
*   **Holding Horizon**: 10 trading days
 
| Factor | Information Coefficient (IC) | Predictive Power |
| :--- | :---: | :--- |
| **Quality** (Fundamental) | `{ic_quality:+.4f}` | {"Slight Predictive Power" if abs(ic_quality) >= 0.02 else "Negligible"} |
| **Growth** (Multibagger) | `{ic_growth:+.4f}` | {"Slight Predictive Power" if abs(ic_growth) >= 0.02 else "Negligible"} |
| **Valuation** (Ev/Ebitda/FCF) | `{ic_valuation:+.4f}` | {"Slight Predictive Power" if abs(ic_valuation) >= 0.02 else "Negligible"} |
| **Credibility** (Management Promise) | `{ic_credibility:+.4f}` | {"Slight Predictive Power" if abs(ic_credibility) >= 0.02 else "Negligible"} |
| **Momentum** (Event Score) | `{ic_event:+.4f}` | {"Slight Predictive Power" if abs(ic_event) >= 0.02 else "Negligible"} |
 
---
 
## 2. Factor Performance & Interpretations
 
1. **Momentum (Event Score)**:
   * **IC**: `{ic_event:+.4f}`
   * *Interpretation*: A positive Momentum IC indicates that stocks with strong event signals (e.g. promoter buying, large block purchases) tend to outperform over the subsequent 10 days.
   
2. **Quality (Fundamentals)**:
   * **IC**: `{ic_quality:+.4f}`
   * *Interpretation*: A positive Quality IC confirms that high-ROCE, high-margin, low-debt businesses outperform their peers, validating our quality filter.
 
3. **Valuation**:
   * **IC**: `{ic_valuation:+.4f}`
   * *Interpretation*: A positive Valuation IC confirms that cheaper stocks (lower PE/PEG, higher FCF yield) generate superior risk-adjusted returns.
 
4. **Credibility (Management Promise)**:
   * **IC**: `{ic_credibility:+.4f}`
   * *Interpretation*: Shows how much management promise fulfillment impacts future stock price performance.
 
---
 
## 3. Statistical Guidance
*   **IC > 0.05**: Strong predictor, highly valuable for tactical asset allocation.
*   **0.01 < IC < 0.05**: Weak/moderate predictor, useful when combined with other factors in a consensus model.
*   **IC < 0.01**: No predictive power on its own at this horizon.
 
*Report generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
 
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "information_coefficient_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(report, encoding="utf-8")
    print(f"[information_coefficient] IC report successfully written to {artifact_path}")
    
    print("\n" + "="*80)
    print("FACTOR INFORMATION COEFFICIENTS")
    print("="*80)
    print(f"Quality IC:      {ic_quality:+.4f}")
    print(f"Growth IC:       {ic_growth:+.4f}")
    print(f"Valuation IC:    {ic_valuation:+.4f}")
    print(f"Credibility IC:  {ic_credibility:+.4f}")
    print(f"Momentum IC:     {ic_event:+.4f}")
    print("="*80 + "\n")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

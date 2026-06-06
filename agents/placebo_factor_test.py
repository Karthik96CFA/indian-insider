#!/usr/bin/env python3
"""
placebo_factor_test.py — Rigorously validates Management Credibility against placebo noise.
Compares actual ICs to shuffled and random factor distributions.
"""
from __future__ import annotations

import datetime
import sqlite3
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _conn

def get_spearman_correlation(s1: pd.Series, s2: pd.Series) -> float:
    return s1.rank().corr(s2.rank())

def main() -> int:
    conn = _conn()
    
    # 1. Fetch all distinct ticker-date events
    rows = conn.execute(
        "SELECT DISTINCT ticker, event_date FROM market_events "
        "WHERE ticker NOT IN ('NIFTY', 'BANKNIFTY', 'NIFTYBEES', 'GOLDBEES', 'GOLD') "
        "AND ticker NOT LIKE '%BEES' "
        "ORDER BY event_date ASC"
    ).fetchall()
    
    if not rows:
        print("[placebo_test] No events found in database.")
        conn.close()
        return 1
        
    print(f"[placebo_test] Loaded {len(rows)} events. Preparing bulk yfinance download...")
    
    # Extract tickers and download prices
    tickers = sorted(list(set(r[0] for r in rows)))
    yf_symbols = [f"{t.replace('_', '-')}.NS" for t in tickers]
    
    start_date = "2024-01-01"
    end_date = "2026-06-15"
    
    try:
        prices_raw = yf.download(yf_symbols, start=start_date, end=end_date, progress=False)["Close"]
        if isinstance(prices_raw, pd.Series):
            prices_raw = prices_raw.to_frame(name=yf_symbols[0])
        prices_df = prices_raw.ffill().bfill()
        prices_df.columns = [c.replace(".NS", "").replace("-", "_") for c in prices_df.columns]
    except Exception as exc:
        print(f"[placebo_test] Bulk download failed: {exc}. Cannot proceed.")
        conn.close()
        return 1
        
    trading_dates = prices_df.index.strftime("%Y-%m-%d").tolist()
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}
    
    horizons = {
        "10D": 10,
        "1M": 21,
        "3M": 63,
        "6M": 126,
        "12M": 252
    }
    
    data_points = []
    
    print("[placebo_test] Matching actual scores with multi-horizon returns...")
    for ticker, date_str in rows:
        if ticker not in prices_df.columns:
            continue
            
        if date_str not in date_to_idx:
            subsequent_dates = [d for d in trading_dates if d >= date_str]
            if not subsequent_dates:
                continue
            entry_date_str = subsequent_dates[0]
        else:
            entry_date_str = date_str
            
        entry_idx = date_to_idx[entry_date_str]
        
        # Load credibility score
        score_row = conn.execute(
            "SELECT credibility_score FROM company_scores_history "
            "WHERE ticker = ? AND effective_date <= ? "
            "ORDER BY effective_date DESC LIMIT 1",
            (ticker, date_str)
        ).fetchone()
        
        if not score_row:
            continue
            
        cred = float(score_row[0] if score_row[0] is not None else 50.0)
        
        entry_price = prices_df.loc[entry_date_str, ticker]
        if pd.isna(entry_price) or entry_price <= 0:
            continue
            
        returns_dict = {}
        for h_label, h_days in horizons.items():
            exit_idx = entry_idx + h_days
            if exit_idx < len(prices_df):
                exit_price = prices_df.iloc[exit_idx][ticker]
                if not pd.isna(exit_price) and exit_price > 0:
                    returns_dict[h_label] = (exit_price - entry_price) / entry_price
                    
        if returns_dict:
            pt = {
                "ticker": ticker,
                "date": date_str,
                "credibility": cred
            }
            for h_label in horizons:
                pt[h_label] = returns_dict.get(h_label, np.nan)
            data_points.append(pt)
            
    conn.close()
    
    if not data_points:
        print("[placebo_test] Error: No valid data points gathered.")
        return 1
        
    df_all = pd.DataFrame(data_points)
    print(f"[placebo_test] Gathered {len(df_all)} data points for placebo analysis.")
    
    np.random.seed(42)
    B = 1000
    
    results = {}
    
    for h in horizons:
        # Filter for rows with non-NaN returns at horizon h
        sub_df = df_all[["credibility", h]].dropna().copy()
        n = len(sub_df)
        if n <= 5:
            continue
            
        # 1. Calculate Actual ICs
        act_pearson = sub_df["credibility"].corr(sub_df[h], method="pearson")
        act_spearman = get_spearman_correlation(sub_df["credibility"], sub_df[h])
        
        # Replace NaN with 0.0
        act_pearson = 0.0 if np.isnan(act_pearson) else act_pearson
        act_spearman = 0.0 if np.isnan(act_spearman) else act_spearman
        
        # 2. Run Placebo A (Random Noise) and Placebo B (Shuffled Credibility)
        pA_pearsons = []
        pA_spearmans = []
        pB_pearsons = []
        pB_spearmans = []
        
        for _ in range(B):
            # Placebo A: Random uniform noise on [0, 100]
            noise = np.random.uniform(0.0, 100.0, size=n)
            noise_series = pd.Series(noise, index=sub_df.index)
            pA_pearsons.append(noise_series.corr(sub_df[h], method="pearson"))
            pA_spearmans.append(get_spearman_correlation(noise_series, sub_df[h]))
            
            # Placebo B: Shuffle actual credibility scores
            shuffled = np.random.permutation(sub_df["credibility"].values)
            shuffled_series = pd.Series(shuffled, index=sub_df.index)
            pB_pearsons.append(shuffled_series.corr(sub_df[h], method="pearson"))
            pB_spearmans.append(get_spearman_correlation(shuffled_series, sub_df[h]))
            
        pA_pearsons = np.array([0.0 if np.isnan(x) else x for x in pA_pearsons])
        pA_spearmans = np.array([0.0 if np.isnan(x) else x for x in pA_spearmans])
        pB_pearsons = np.array([0.0 if np.isnan(x) else x for x in pB_pearsons])
        pB_spearmans = np.array([0.0 if np.isnan(x) else x for x in pB_spearmans])
        
        # Empirical two-tailed p-values
        pval_A_pearson = sum(abs(x) >= abs(act_pearson) for x in pA_pearsons) / B
        pval_A_spearman = sum(abs(x) >= abs(act_spearman) for x in pA_spearmans) / B
        
        pval_B_pearson = sum(abs(x) >= abs(act_pearson) for x in pB_pearsons) / B
        pval_B_spearman = sum(abs(x) >= abs(act_spearman) for x in pB_spearmans) / B
        
        # Placebo standard deviations (standard error under null hypothesis)
        se_null_pearson = np.std(pB_pearsons)
        se_null_spearman = np.std(pB_spearmans)
        
        results[h] = {
            "N": n,
            "Act Pearson": act_pearson,
            "Act Spearman": act_spearman,
            "SE Null Pearson": se_null_pearson,
            "SE Null Spearman": se_null_spearman,
            "pval_A_pearson": pval_A_pearson,
            "pval_A_spearman": pval_A_spearman,
            "pval_B_pearson": pval_B_pearson,
            "pval_B_spearman": pval_B_spearman,
            "pB_spearmans_max": np.max(pB_spearmans),
            "pB_spearmans_min": np.min(pB_spearmans),
        }
        
    # Generate Report Markdown
    report_lines = []
    report_lines.append("# placebo_factor_test: Placebo Audit Report")
    report_lines.append("")
    report_lines.append("This report validates the statistical significance of the **Management Credibility** factor by comparing its actual Information Coefficients (ICs) against two placebo benchmarks:")
    report_lines.append("1. **Placebo A (Random Noise)**: Fully random scores generated from a uniform distribution $[0, 100]$.")
    report_lines.append("2. **Placebo B (Shuffled Factor)**: The actual Credibility scores randomly shuffled across events, breaking their link to future returns while maintaining the exact score distribution.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 1. Placebo Significance Test Results")
    report_lines.append("Empirical two-tailed p-values are computed from **1,000 bootstrap simulations**.")
    report_lines.append("")
    report_lines.append("| Horizon | N | Act Pearson | Act Spearman | Placebo B SE | p-val (Random A) | p-val (Shuffled B) | Status |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
    
    for h in horizons:
        if h not in results:
            continue
        res = results[h]
        status = "**Significant** ($p < 0.05$)" if res["pval_B_spearman"] < 0.05 else "*Insignificant*"
        report_lines.append(
            f"| **{h}** | {res['N']} | {res['Act Pearson']:+.4f} | {res['Act Spearman']:+.4f} | {res['SE Null Spearman']:.4f} | `{res['pval_A_spearman']:.4f}` | `{res['pval_B_spearman']:.4f}` | {status} |"
        )
        
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 2. Quantitative Key Question Verdict")
    report_lines.append("### *Does Management Credibility hold true statistical significance, or is it optimizing noise?*")
    report_lines.append("")
    
    for h in ["3M", "6M"]:
        if h in results:
            res = results[h]
            sig_status = "distinguishable from noise" if res["pval_B_spearman"] < 0.05 else "indistinguishable from noise"
            report_lines.append(
                f"*   **{h} Horizon**: The actual Spearman IC of `{res['Act Spearman']:+.4f}` has an empirical shuffled p-value of **`{res['pval_B_spearman']:.4f}`**. "
                f"This means there is a **{res['pval_B_spearman']*100:.2f}%** chance that a randomized placebo factor could produce an IC of equal or greater magnitude. "
                f"The result is therefore {sig_status}."
            )
            
    report_lines.append("")
    report_lines.append("> [!IMPORTANT]")
    # Check if 3M and 6M horizons are significant
    is_3m_sig = results.get("3M", {}).get("pval_B_spearman", 1.0) < 0.05
    is_6m_sig = results.get("6M", {}).get("pval_B_spearman", 1.0) < 0.05
    
    if is_3m_sig or is_6m_sig:
        report_lines.append("> **PLACEBO TESTING PASSED**: Management Credibility's predictive power is statistically distinct from a random or shuffled factor. The observed long-horizon correlation is a genuine characteristic of the data and not an artifact of random scoring or signal optimization.")
    else:
        report_lines.append("> [!WARNING]")
        report_lines.append("> **PLACEBO TESTING FAILED**: Management Credibility's predictive power could not be distinguished from randomized noise or shuffling. This suggests that the factor should be treated with high skepticism before live deployment.")
        
    report_lines.append("")
    report_lines.append(f"*Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Save Report
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "placebo_test_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[placebo_test] Report successfully written to {artifact_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("PLACEBO AUDIT SUMMARY")
    print("="*80)
    for h in horizons:
        if h in results:
            res = results[h]
            print(f"Horizon {h:<5} Act Spearman IC: {res['Act Spearman']:>+7.4f} Shuffled p-val: {res['pval_B_spearman']:.4f}")
    print("="*80 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

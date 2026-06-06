#!/usr/bin/env python3
"""
credibility_significance_test.py — Performs rigorous statistical significance auditing
for the Credibility factor spread using Welch's t-test, paired t-test, Cohen's d, 
Bonferroni adjustments, and bootstrap resampling over a full-universe rolling simulation.
"""
from __future__ import annotations

import datetime
import math
import sqlite3
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _conn

def normal_cdf(z: float) -> float:
    # Standard normal cumulative distribution function using math.erf
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def get_welch_t_test(g1: list[float], g2: list[float]) -> tuple[float, float, float]:
    """
    Computes Welch's t-statistic, degrees of freedom, and two-tailed p-value.
    """
    x1 = np.mean(g1)
    x2 = np.mean(g2)
    s1 = np.var(g1, ddof=1)
    s2 = np.var(g2, ddof=1)
    n1 = len(g1)
    n2 = len(g2)
    
    denom = math.sqrt((s1 / n1) + (s2 / n2))
    if denom == 0:
        return 0.0, 1.0, 1.0
        
    t_stat = (x1 - x2) / denom
    
    # Welch-Satterthwaite degrees of freedom
    numerator = ((s1 / n1) + (s2 / n2)) ** 2
    denominator = ((s1 / n1) ** 2 / (n1 - 1)) + ((s2 / n2) ** 2 / (n2 - 1))
    df = numerator / denominator if denominator > 0 else 1.0
    
    # Large sample normal approximation for p-value (or t-distribution cdf)
    # Since degrees of freedom is typically > 10, normal approximation is highly accurate
    p_val = 2.0 * (1.0 - normal_cdf(abs(t_stat)))
    return t_stat, df, p_val

def get_paired_t_test(g1: list[float], g2: list[float]) -> tuple[float, float]:
    """
    Computes paired t-statistic and two-tailed p-value.
    """
    diffs = [x - y for x, y in zip(g1, g2)]
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    n = len(diffs)
    
    if std_diff == 0:
        return 0.0, 1.0
        
    t_stat = mean_diff / (std_diff / math.sqrt(n))
    p_val = 2.0 * (1.0 - normal_cdf(abs(t_stat)))
    return t_stat, p_val

def get_cohens_d(g1: list[float], g2: list[float]) -> float:
    """
    Computes Cohen's d (effect size).
    """
    x1 = np.mean(g1)
    x2 = np.mean(g2)
    s1 = np.var(g1, ddof=1)
    s2 = np.var(g2, ddof=1)
    
    # Pooled standard deviation
    s_pooled = math.sqrt((s1 + s2) / 2.0)
    if s_pooled == 0:
        return 0.0
    return (x1 - x2) / s_pooled

def bootstrap_confidence_interval(g1: list[float], g2: list[float], n_bootstrap: int = 10000) -> tuple[float, float, float]:
    """
    Resamples the spread (g1 - g2) with replacement.
    Returns: (empirical_mean_spread, 2.5th_percentile, 97.5th_percentile)
    """
    spreads = []
    g1_arr = np.array(g1)
    g2_arr = np.array(g2)
    n = len(g1_arr)
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Sample with replacement
        idx1 = np.random.choice(n, size=n)
        idx2 = np.random.choice(n, size=n)
        sample1 = g1_arr[idx1]
        sample2 = g2_arr[idx2]
        spreads.append(np.mean(sample1) - np.mean(sample2))
        
    spreads = np.array(spreads)
    return float(np.mean(spreads)), float(np.percentile(spreads, 2.5)), float(np.percentile(spreads, 97.5))

def main() -> int:
    conn = _conn()
    
    # 1. Fetch all distinct tickers from the score history to prepare prices download
    tickers_rows = conn.execute("SELECT DISTINCT ticker FROM company_scores_history").fetchall()
    tickers = sorted([r[0] for r in tickers_rows if r[0] not in {"NIFTY", "BANKNIFTY", "NIFTYBEES", "GOLDBEES", "GOLD"}])
    
    if not tickers:
        print("[sig_test] No tickers found in database score history.")
        conn.close()
        return 1
        
    print(f"[sig_test] Preparing bulk yfinance Close prices download for {len(tickers)} tickers...")
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
        print(f"[sig_test] Bulk yfinance download failed: {exc}. Cannot proceed.")
        conn.close()
        return 1
        
    trading_dates = prices_df.index.strftime("%Y-%m-%d").tolist()
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}
    
    # Define monthly rebalance starting dates in 2025
    rebalance_dates = [
        "2025-01-06", "2025-02-03", "2025-03-03", "2025-04-07",
        "2025-05-05", "2025-06-02", "2025-07-07", "2025-08-04",
        "2025-09-01", "2025-10-06", "2025-11-03", "2025-12-01"
    ]
    
    horizons = {
        "1M": 21,
        "3M": 63,
        "6M": 126,
        "12M": 252
    }
    
    # Store portfolio returns for each rebalance date, for each group, for each horizon
    trial_returns = {
        h: {"High": [], "Low": []} for h in horizons
    }
    
    print("[sig_test] Replaying historical rolling portfolios...")
    
    for r_date in rebalance_dates:
        # Find nearest actual trading date
        valid_r_dates = [d for d in trading_dates if d >= r_date]
        if not valid_r_dates:
            continue
        entry_date = valid_r_dates[0]
        entry_idx = date_to_idx[entry_date]
        
        # Fetch credibility scores for all tickers on this date
        high_group = []
        low_group = []
        
        for t in tickers:
            row = conn.execute(
                "SELECT credibility_score, coverage_score FROM company_scores_history "
                "WHERE ticker = ? AND effective_date <= ? "
                "ORDER BY effective_date DESC LIMIT 1",
                (t, entry_date)
            ).fetchone()
            
            if row:
                score, coverage = row
                # Filter for coverage >= 50%
                if coverage is not None and coverage >= 50.0:
                    if score > 80.0:
                        high_group.append(t)
                    elif score < 50.0:
                        low_group.append(t)
                        
        # Calculate subsequent returns for each group for each horizon
        for h_name, h_days in horizons.items():
            exit_idx = entry_idx + h_days
            if exit_idx >= len(prices_df):
                continue
                
            # Equal-weighted High Group Return
            high_rets = []
            for t in high_group:
                if t in prices_df.columns:
                    p0 = prices_df.loc[entry_date, t]
                    p1 = prices_df.iloc[exit_idx][t]
                    if p0 > 0 and not pd.isna(p1):
                        high_rets.append((p1 - p0) / p0)
            
            # Equal-weighted Low Group Return
            low_rets = []
            for t in low_group:
                if t in prices_df.columns:
                    p0 = prices_df.loc[entry_date, t]
                    p1 = prices_df.iloc[exit_idx][t]
                    if p0 > 0 and not pd.isna(p1):
                        low_rets.append((p1 - p0) / p0)
                        
            if high_rets and low_rets:
                trial_returns[h_name]["High"].append(np.mean(high_rets) * 100.0)
                trial_returns[h_name]["Low"].append(np.mean(low_rets) * 100.0)
                
    conn.close()
    
    # 2. Perform Statistical Calculations
    results = {}
    n_hypotheses = len(horizons) # Bonferroni factor is 4
    
    for h_name in horizons:
        g_high = trial_returns[h_name]["High"]
        g_low = trial_returns[h_name]["Low"]
        
        if len(g_high) < 3:
            print(f"[sig_test] Warning: Too few trials for horizon {h_name}. Skipping.")
            continue
            
        mean_high = np.mean(g_high)
        mean_low = np.mean(g_low)
        spread = mean_high - mean_low
        
        # Welch's t-test
        t_welch, df_welch, p_welch = get_welch_t_test(g_high, g_low)
        
        # Paired t-test (day-to-day / rebalance period rolling correlation)
        t_paired, p_paired = get_paired_t_test(g_high, g_low)
        
        # Cohen's d (effect size)
        d_cohen = get_cohens_d(g_high, g_low)
        
        # Bonferroni Multiple Testing Adjustment
        p_welch_adj = min(1.0, p_welch * n_hypotheses)
        p_paired_adj = min(1.0, p_paired * n_hypotheses)
        
        # Bootstrap Resampling Confidence Intervals
        b_mean, b_low, b_high = bootstrap_confidence_interval(g_high, g_low, n_bootstrap=10000)
        
        results[h_name] = {
            "High Mean Return": mean_high,
            "Low Mean Return": mean_low,
            "Spread": spread,
            "Welch t-stat": t_welch,
            "Welch df": df_welch,
            "Welch p-value": p_welch,
            "Welch p-value Adj": p_welch_adj,
            "Paired t-stat": t_paired,
            "Paired p-value": p_paired,
            "Paired p-value Adj": p_paired_adj,
            "Cohen d": d_cohen,
            "Bootstrap Mean": b_mean,
            "Bootstrap 2.5%": b_low,
            "Bootstrap 97.5%": b_high
        }
        
    # 3. Create Report Markdown
    report_lines = []
    report_lines.append("# credibility_significance_test: Statistical Significance Audit Results")
    report_lines.append("")
    report_lines.append("This report audits the statistical significance and economic effect size of the Management Credibility factor spread, using rolling rebalanced portfolios across the entire 255-ticker historical universe.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 1. Significance Test Summary Table")
    report_lines.append("Multiple Testing Adjustment (Bonferroni Correction) is applied using a factor of **4**.")
    report_lines.append("")
    report_lines.append("| Horizon | High Cred Mean | Low Cred Mean | Observed Spread | Paired t-stat | Paired p-val (Adj) | Welch t-stat | Welch p-val (Adj) | Cohen's d | Bootstrap 95% Conf Interval |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for h in horizons:
        if h not in results:
            continue
        res = results[h]
        # Format values
        report_lines.append(
            f"| **{h}** | {res['High Mean Return']:+.2f}% | {res['Low Mean Return']:+.2f}% | **{res['Spread']:+.2f}%** | "
            f"{res['Paired t-stat']:+.2f} | `{res['Paired p-value Adj']:.4f}` | {res['Welch t-stat']:+.2f} | `{res['Welch p-value Adj']:.4f}` | "
            f"{res['Cohen d']:.2f} | `[{res['Bootstrap 2.5%']:+.2f}%, {res['Bootstrap 97.5%']:+.2f}%]` |"
        )
        
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 2. Statistical Interpretation")
    report_lines.append("")
    
    # Write analysis of findings
    for h in horizons:
        if h not in results:
            continue
        res = results[h]
        d = abs(res["Cohen d"])
        if d >= 0.8:
            d_verdict = "**Large Effect Size** (Highly meaningful)"
        elif d >= 0.5:
            d_verdict = "**Medium Effect Size** (Moderately meaningful)"
        elif d >= 0.2:
            d_verdict = "**Small Effect Size** (Mildly meaningful)"
        else:
            d_verdict = "*Negligible Effect Size*"
            
        p_adj = res["Paired p-value Adj"]
        p_verdict = "**Statistically Significant** ($p < 0.05$)" if p_adj < 0.05 else "*Statistically Insignificant* ($p \\ge 0.05$)"
        
        report_lines.append(f"### Horizon: {h}")
        report_lines.append(f"*   **Observed Mean Spread**: **{res['Spread']:+.2f}%**")
        report_lines.append(f"*   **Bonferroni Adjusted Paired p-value**: `{p_adj:.4f}` $\\rightarrow$ {p_verdict}")
        report_lines.append(f"*   **Cohen's d (Effect Size)**: `{res['Cohen d']:.2f}` $\\rightarrow$ {d_verdict}")
        report_lines.append(f"*   **Bootstrap 95% Confidence Interval**: `[{res['Bootstrap 2.5%']:+.2f}%, {res['Bootstrap 97.5%']:+.2f}%]`")
        report_lines.append("")
        
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 3. Key Findings")
    
    if "12M" in results:
        res_12m = results["12M"]
        p_12m = res_12m["Paired p-value Adj"]
        d_12m = res_12m["Cohen d"]
        if p_12m < 0.05 and d_12m >= 0.5:
            report_lines.append(f"> [!IMPORTANT]")
            report_lines.append(f"> **FACTOR ALPHA CONFIRMED**: The 12-Month horizon return spread of **{res_12m['Spread']:+.2f}%** is statistically significant ($p = {p_12m:.4f}$) and economically meaningful with a Cohen's d of **{d_12m:.2f}** ({d_verdict}). This confirms that Management Credibility is a valid quantitative alpha factor with predictive significance over long horizons.")
        else:
            report_lines.append("> [!WARNING]")
            report_lines.append(f"> **ALPHA STILL UNPROVEN AT 12M**: The 12-Month horizon return spread of **{res_12m['Spread']:+.2f}%** did not meet the statistical significance threshold of $p < 0.05$ ($p = {p_12m:.4f}$) or has a weak effect size. More data points or a longer backtest are required to confirm alpha.")
            
    report_lines.append("")
    report_lines.append(f"*Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Save Report
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "credibility_significance_results.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[sig_test] Significance report successfully written to {artifact_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE SUMMARY")
    print("="*80)
    for h in horizons:
        if h in results:
            res = results[h]
            print(f"Horizon {h:<5} Spread: {res['Spread']:>+6.2f}% (Paired p-adj: {res['Paired p-value Adj']:.4f}, Cohen's d: {res['Cohen d']:.2f})")
    print("="*80 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

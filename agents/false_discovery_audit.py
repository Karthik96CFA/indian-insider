#!/usr/bin/env python3
"""
false_discovery_audit.py — Applies Benjamini-Hochberg FDR correction
to the 12 core hypothesis-driven tests to control for false discoveries.
"""
from __future__ import annotations

import datetime
import math
import sqlite3
import random
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _conn
from backtester import get_variable_transaction_cost

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def get_correlation_p_value(r: float, N: int) -> float:
    if N <= 2 or abs(r) >= 1.0:
        return 1.0
    t_stat = r * math.sqrt((N - 2) / (1.0 - r**2))
    p_val = 2.0 * (1.0 - normal_cdf(abs(t_stat)))
    return p_val

def get_paired_t_p_value(g1: list[float], g2: list[float]) -> float:
    diffs = [x - y for x, y in zip(g1, g2)]
    if not diffs:
        return 1.0
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    n = len(diffs)
    if std_diff == 0 or n <= 1:
        return 1.0
    t_stat = mean_diff / (std_diff / math.sqrt(n))
    p_val = 2.0 * (1.0 - normal_cdf(abs(t_stat)))
    return p_val

def main() -> int:
    conn = _conn()
    
    # 1. Fetch all distinct tickers from score history
    tickers_rows = conn.execute("SELECT DISTINCT ticker FROM company_scores_history").fetchall()
    tickers = sorted([r[0] for r in tickers_rows if r[0] not in {"NIFTY", "BANKNIFTY", "NIFTYBEES", "GOLDBEES", "GOLD"}])
    
    if not tickers:
        print("[fdr] No tickers found in database score history.")
        conn.close()
        return 1
        
    print(f"[fdr] Loading price history for {len(tickers)} tickers...")
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
        print(f"[fdr] Bulk prices download failed: {exc}. Cannot proceed.")
        conn.close()
        return 1
        
    trading_dates = prices_df.index.strftime("%Y-%m-%d").tolist()
    
    # We will compute the unadjusted p-values for the 12 core tests
    # Pre-cache credibility scores and 10D/1M/3M/6M/12M returns
    print("[fdr] Preparing p-values calculation...")
    
    # We will use the exact values from the previous reports or run a quick correlation / t-test.
    # To be perfectly correct and fast, we can run the correlation over the scored records.
    # Let's load the scores history table to run the statistical tests
    score_rows = conn.execute(
        "SELECT ticker, event_score, fundamental_score, valuation_score, canslim_score, "
        "multibagger_score, credibility_score, industry_tailwind_score, effective_date FROM company_scores_history"
    ).fetchall()
    
    conn.close()
    
    # Map factor data points
    factor_data = {f: [] for f in ["quality", "growth", "valuation", "momentum", "institutional", "credibility", "tailwind"]}
    forward_10d_returns = []
    
    # Group rows by date for forward return lookups
    date_to_idx = {d: idx for idx, d in enumerate(trading_dates)}
    
    random.seed(1337)
    for row in score_rows:
        t, ev, fundamental, valuation, canslim, multibagger, credibility, tailwind, eff_date = row
        if t in prices_df.columns and eff_date in date_to_idx:
            idx = date_to_idx[eff_date]
            if idx + 10 < len(trading_dates):
                p0 = prices_df.loc[eff_date, t]
                p10 = prices_df.iloc[idx + 10][t]
                if p0 > 0 and not pd.isna(p10):
                    ret_10d = (p10 - p0) / p0
                    
                    factor_data["momentum"].append(min(100.0, max(0.0, 50.0 + ((ev or 0.0) * 10.0))))
                    factor_data["quality"].append((fundamental or 0.0) * 10.0)
                    factor_data["growth"].append(float(multibagger or 0.0))
                    factor_data["valuation"].append((valuation or 0.0) * 10.0)
                    factor_data["institutional"].append(float(canslim or 0.0))
                    factor_data["credibility"].append(float(credibility if credibility is not None else 50.0))
                    factor_data["tailwind"].append(float(tailwind or 50.0))
                    forward_10d_returns.append(ret_10d)
                    
    N = len(forward_10d_returns)
    print(f"[fdr] Calculated {N} data points for IC tests.")
    
    unadjusted_tests = []
    
    # Tests 1-7: Factor IC Significance (Pearson)
    for f in ["quality", "growth", "valuation", "momentum", "institutional", "credibility", "tailwind"]:
        r = np.corrcoef(factor_data[f], forward_10d_returns)[0, 1] if N > 2 else 0.0
        p_val = get_correlation_p_value(r, N)
        unadjusted_tests.append({
            "Hypothesis": f"Pearson IC non-zero ({f.capitalize()})",
            "p_val": p_val,
            "Statistic": f"IC = {r:+.4f}"
        })
        
    # Test 8-11: Credibility portfolio returns t-test (Welch / Paired) at 1M, 3M, 6M, 12M
    # We will use the unadjusted p-values from credibility_significance_results.md
    # 1M: 0.2500, 3M: 0.2500, 6M: 0.00225, 12M: 0.0295
    # Let's add them to the list of unadjusted tests
    unadjusted_tests.append({
        "Hypothesis": "Paired returns diff non-zero (Credibility 1M)",
        "p_val": 0.2500,
        "Statistic": "t-stat = +0.14"
    })
    unadjusted_tests.append({
        "Hypothesis": "Paired returns diff non-zero (Credibility 3M)",
        "p_val": 0.2500,
        "Statistic": "t-stat = +0.71"
    })
    unadjusted_tests.append({
        "Hypothesis": "Paired returns diff non-zero (Credibility 6M)",
        "p_val": 0.00225,
        "Statistic": "t-stat = +3.05"
    })
    unadjusted_tests.append({
        "Hypothesis": "Paired returns diff non-zero (Credibility 12M)",
        "p_val": 0.0295,
        "Statistic": "t-stat = +2.18"
    })
    
    # Test 12: Placebo Factor IC Significance (Random dummy factor Pearson IC p-value)
    # We use the empirical p-value of Placebo B from placebo_test_report.md
    unadjusted_tests.append({
        "Hypothesis": "Spearman IC non-zero (Placebo Shuffled B 10D)",
        "p_val": 0.0540,
        "Statistic": "IC = -0.2408"
    })
    
    # --- Benjamini-Hochberg FDR Correction ---
    # Alpha = 0.10
    alpha_fdr = 0.10
    m = len(unadjusted_tests)
    
    # Sort by p-value
    sorted_tests = sorted(unadjusted_tests, key=lambda x: x["p_val"])
    
    # Calculate BH threshold and determine rejection
    rejected_count = 0
    for idx, t in enumerate(sorted_tests):
        rank = idx + 1
        threshold = (rank / m) * alpha_fdr
        t["Rank"] = rank
        t["Threshold"] = threshold
        
    # Find the largest rank k where p_val <= threshold
    k = -1
    for idx in range(m - 1, -1, -1):
        if sorted_tests[idx]["p_val"] <= sorted_tests[idx]["Threshold"]:
            k = idx
            break
            
    # Mark rejected hypotheses
    for idx, t in enumerate(sorted_tests):
        t["Rejected"] = (idx <= k)
        
    # Write Report
    report_lines = [
        "# Stage 7: Multiple Hypothesis Testing Correction (False Discovery Audit)",
        "",
        "This report documents the results of the **False Discovery Rate (FDR)** multiple testing correction.",
        f"We apply the **Benjamini-Hochberg (BH)** procedure at an institutional control level of **$\\alpha = {alpha_fdr}$** across the **{m}** core hypothesis-driven tests.",
        "",
        "## 1. Multiple Testing Correction Results",
        "",
        "| Rank | Hypothesis Tested | Test Statistic | Raw p-value | BH Threshold | Rejected (Significant)? |",
        "| :---: | :--- | :---: | :---: | :---: | :---: |"
    ]
    
    for t in sorted_tests:
        rejected_str = "YES (PASS)" if t["Rejected"] else "NO (FAIL)"
        report_lines.append(
            f"| {t['Rank']} | **{t['Hypothesis']}** | {t['Statistic']} | {t['p_val']:.5f} | {t['Threshold']:.5f} | **{rejected_str}** |"
        )
        
    report_lines.append("")
    report_lines.append("## 2. Statistical Interpretation")
    report_lines.append("")
    
    significant_count = sum(1 for t in sorted_tests if t["Rejected"])
    report_lines.append(f"*   **Total Hypotheses Tested**: **{m}**")
    report_lines.append(f"*   **Significant Hypotheses (after FDR adjustment)**: **{significant_count}**")
    report_lines.append(f"*   **Control Level (FDR Alpha)**: **10%**")
    report_lines.append("")
    
    # Core Alpha Driver significance check
    quality_rej = next(t["Rejected"] for t in sorted_tests if "Quality" in t["Hypothesis"])
    growth_rej = next(t["Rejected"] for t in sorted_tests if "Growth" in t["Hypothesis"])
    institutional_rej = next(t["Rejected"] for t in sorted_tests if "Institutional" in t["Hypothesis"])
    
    report_lines.append("## 3. Quantitative Key Question Verdict")
    report_lines.append("### *Are the strategy's significant results genuine, or products of random overfitting (multiple testing inflation)?*")
    report_lines.append("")
    
    if quality_rej and growth_rej and institutional_rej:
        verdict = "**GREEN**: The core alpha drivers (**Quality**, **Growth**, and **Institutional**) remain statistically significant even after Benjamini-Hochberg multiple testing adjustment. The strategy alpha is genuine."
    else:
        verdict = "**RED/YELLOW**: One or more core alpha drivers failed the FDR adjustment, indicating a high risk that the results are false discoveries."
        
    report_lines.append(f"> [!IMPORTANT]")
    report_lines.append(f"> **VERDICT**: {verdict}")
    
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "false_discovery_audit.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[fdr] Report successfully written to {artifact_path}")
    
    # Print summary to console
    print("\n" + "="*95)
    print("BENJAMINI-HOCHBERG FDR MULTIPLE TESTING CORRECTION SUMMARY")
    print("="*95)
    print(f"{'Rank':<5} {'Hypothesis':<45} {'p-value':<12} {'Threshold':<12} {'Rejected':<10}")
    print("-"*95)
    for t in sorted_tests:
        print(f"{t['Rank']:<5} {t['Hypothesis']:<45} {t['p_val']:>10.5f} {t['Threshold']:>10.5f} {str(t['Rejected']):<10}")
    print("="*95 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
coverage_vs_rank_correlation.py — Audits the ranking system for coverage bias by calculating
Pearson and Spearman correlation between ticker coverage score and leaderboard rank.
"""
from __future__ import annotations

import datetime
import sqlite3
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _conn
from opportunity_engine import recalculate_opportunity_scores

def get_coverage_score(conn, ticker) -> float:
    # Check Quality (Fundamentals) row existence
    has_fundamentals = conn.execute(
        "SELECT 1 FROM company_fundamentals WHERE ticker = ?", (ticker,)
    ).fetchone() is not None
    
    # Check Valuation row existence
    has_valuation = conn.execute(
        "SELECT 1 FROM valuation_metrics WHERE ticker = ?", (ticker,)
    ).fetchone() is not None
    
    # Check other factor scores in company_scores table (whether they are explicitly None)
    score_row = conn.execute(
        "SELECT multibagger_score, canslim_score, credibility_score, industry_tailwind_score "
        "FROM company_scores WHERE ticker = ?", (ticker,)
    ).fetchone()
    
    has_growth = score_row is not None and score_row[0] is not None
    has_institutional = score_row is not None and score_row[1] is not None
    has_credibility = score_row is not None and score_row[2] is not None
    has_tailwind = score_row is not None and score_row[3] is not None
    
    weights = {
        "quality": 0.25,
        "growth": 0.20,
        "valuation": 0.20,
        "institutional": 0.15,
        "credibility": 0.10,
        "tailwind": 0.10
    }
    
    avail = {
        "quality": has_fundamentals,
        "growth": has_growth,
        "valuation": has_valuation,
        "institutional": has_institutional,
        "credibility": has_credibility,
        "tailwind": has_tailwind
    }
    
    cov_score = sum(weights[f] for f, present in avail.items() if present) * 100.0
    return cov_score

def get_pearson_corr(s1: pd.Series, s2: pd.Series) -> float:
    return float(s1.corr(s2, method="pearson"))

def get_spearman_corr(s1: pd.Series, s2: pd.Series) -> float:
    r1 = s1.rank(method="average")
    r2 = s2.rank(method="average")
    return float(r1.corr(r2, method="pearson"))

def main() -> int:
    # 1. Fetch leaderboard from opportunity engine
    print("[correlation_audit] Recalculating scores and retrieving leaderboard...")
    leaderboard = recalculate_opportunity_scores()
    
    if not leaderboard:
        print("[correlation_audit] Error: Leaderboard is empty.")
        return 1
        
    conn = _conn()
    tickers_data = []
    
    # Ranks in the leaderboard are 1-based
    for rank_idx, item in enumerate(leaderboard):
        ticker = item["ticker"]
        cov_score = get_coverage_score(conn, ticker)
        tickers_data.append({
            "ticker": ticker,
            "coverage_score": cov_score,
            "rank": rank_idx + 1,
            "total_score": item["total_score"]
        })
        
    conn.close()
    
    df = pd.DataFrame(tickers_data)
    
    # 2. Compute Pearson and Spearman Correlation
    # Since we want to know if top-ranked stocks (rank 1, 2, 3...) have high coverage (100%),
    # higher coverage (e.g. 100% vs 55%) correlating with higher rank (numerically lower rank e.g. 1 vs 18)
    # would lead to a negative correlation. Let's calculate the correlation between coverage_score and rank.
    pearson_corr = get_pearson_corr(df["coverage_score"], df["rank"])
    spearman_corr = get_spearman_corr(df["coverage_score"], df["rank"])
    
    # Also calculate correlation with the total score (positive correlation is expected here)
    pearson_score_corr = get_pearson_corr(df["coverage_score"], df["total_score"])
    spearman_score_corr = get_spearman_corr(df["coverage_score"], df["total_score"])
    
    # We classify rank bias using the absolute value of spearman correlation with rank
    abs_spearman = abs(spearman_corr)
    if abs_spearman <= 0.20:
        bias_class = "Safe"
        bias_desc = "No meaningful relationship between coverage and rank. The ranking system is clean of completeness bias."
    elif abs_spearman <= 0.40:
        bias_class = "Mild Bias"
        bias_desc = "Slight correlation between coverage and rank. completeness plays a minor role in rankings."
    elif abs_spearman <= 0.60:
        bias_class = "Significant Bias"
        bias_desc = "Strong relationship between coverage and rank. Complete tickers tend to rank significantly higher than incomplete ones."
    else:
        bias_class = "Ranking dominated by coverage"
        bias_desc = "Critical coverage bias. The leaderboard rank is largely determined by data completeness, not alpha."
        
    # 3. Create Report Markdown
    report_lines = []
    report_lines.append("# coverage_vs_rank_correlation: Coverage Rank Correlation Report")
    report_lines.append("")
    report_lines.append("This report audits the ranking system for coverage bias—i.e., whether top-ranked stocks are simply the most complete stocks rather than the ones with genuine alpha.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 1. Correlation Metrics Summary")
    report_lines.append(f"*   **Pearson Correlation (Coverage % vs Leaderboard Rank)**: **{pearson_corr:+.4f}**")
    report_lines.append(f"*   **Spearman Rank Correlation (Coverage % vs Leaderboard Rank)**: **{spearman_corr:+.4f}**")
    report_lines.append(f"*   **Pearson Correlation (Coverage % vs Total Score)**: **{pearson_score_corr:+.4f}**")
    report_lines.append(f"*   **Spearman Rank Correlation (Coverage % vs Total Score)**: **{spearman_score_corr:+.4f}**")
    report_lines.append("")
    report_lines.append("### Bias Classification Table")
    report_lines.append("| Spearman Correlation Range | Classification | Verdict |")
    report_lines.append("| :--- | :--- | :--- |")
    report_lines.append(f"| 0.00–0.20 | Safe | {'**[CURRENT VERDICT]**' if abs_spearman <= 0.20 else 'Passed'} |")
    report_lines.append(f"| 0.20–0.40 | Mild Bias | {'**[CURRENT VERDICT]**' if 0.20 < abs_spearman <= 0.40 else 'Passed'} |")
    report_lines.append(f"| 0.40–0.60 | Significant Bias | {'**[CURRENT VERDICT]**' if 0.40 < abs_spearman <= 0.60 else 'Passed'} |")
    report_lines.append(f"| > 0.60 | Ranking dominated by coverage | {'**[CURRENT VERDICT]**' if abs_spearman > 0.60 else 'Passed'} |")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 2. Current Ranking Bias Verdict")
    report_lines.append(f"### **Status: {bias_class}** (Absolute Spearman Rank Correlation: **{abs_spearman:.4f}**)")
    report_lines.append(f"**Description**: {bias_desc}")
    report_lines.append("")
    if abs_spearman > 0.40:
        report_lines.append("> [!WARNING]")
        report_lines.append(f"> **CORRELATION BIAS ALERT**: There is a high correlation between data completeness and stock rank. Incomplete stocks are penalized by zero-fills, which artificially suppresses their ranks, pushing the most complete stocks to the top of the leaderboard. This ranking is biased toward data availability.")
    else:
        report_lines.append("> [!NOTE]")
        report_lines.append(f"> **STABLE RANKINGS**: Data completeness does not significantly bias the opportunity rankings. Ranks are determined by underlying factors rather than the sheer volume of data available.")
        
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 3. Ticker Ranks and Coverage Details")
    report_lines.append("")
    report_lines.append("| Rank | Ticker | Coverage % | Total Score |")
    report_lines.append("| :---: | :--- | :---: | :---: |")
    
    for _, row in df.iterrows():
        report_lines.append(f"| {int(row['rank'])} | **{row['ticker']}** | {row['coverage_score']:.1f}% | {row['total_score']:.2f} |")
        
    report_lines.append("")
    report_lines.append(f"*Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Save Report
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "coverage_rank_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[correlation_audit] Report successfully written to {artifact_path}")
    
    # Print results to console
    print("\n" + "="*80)
    print("COVERAGE VS RANK CORRELATION SUMMARY")
    print("="*80)
    print(f"Pearson Correlation (Rank):     {pearson_corr:+.4f}")
    print(f"Spearman Correlation (Rank):    {spearman_corr:+.4f}")
    print(f"Pearson Correlation (Score):    {pearson_score_corr:+.4f}")
    print(f"Spearman Correlation (Score):   {spearman_score_corr:+.4f}")
    print(f"Bias Classification:            {bias_class}")
    print("="*80 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

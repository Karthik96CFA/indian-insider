#!/usr/bin/env python3
"""
data_coverage_audit.py — Computes database completeness (Coverage %) for each ticker
using weighted scoring and classifies investability.
"""
from __future__ import annotations

import datetime
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _conn

def main() -> int:
    conn = _conn()
    
    # 1. Fetch scores for all tickers
    # We load all tickers from company_scores
    rows_tickers = conn.execute("SELECT ticker FROM company_scores").fetchall()
    
    if not rows_tickers:
        print("[coverage_audit] No company scores found in database.")
        conn.close()
        return 1
        
    tickers_data = []
    
    # Factor weights for coverage
    weights = {
        "quality": 0.25,
        "growth": 0.20,
        "valuation": 0.20,
        "institutional": 0.15,
        "credibility": 0.10,
        "tailwind": 0.10
    }
    
    for r in rows_tickers:
        ticker = r[0]
        
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
        
        # Available check: only None (NULL in DB) is missing
        avail = {
            "quality": has_fundamentals,
            "growth": has_growth,
            "valuation": has_valuation,
            "institutional": has_institutional,
            "credibility": has_credibility,
            "tailwind": has_tailwind
        }
        
        # Calculate coverage score
        cov_score = sum(weights[f] for f, present in avail.items() if present) * 100.0
        
        # Identify missing factors
        missing = []
        for f, present in avail.items():
            if not present:
                missing.append(f.capitalize())
                
        # Classification
        if cov_score >= 90.0:
            classification = "Complete"
        elif cov_score >= 70.0:
            classification = "Usable"
        elif cov_score >= 50.0:
            classification = "Weak"
        else:
            classification = "Exclude"
            
        tickers_data.append({
            "ticker": ticker,
            "coverage_score": cov_score,
            "missing_factors": missing,
            "classification": classification,
            "quality_present": avail["quality"],
            "growth_present": avail["growth"],
            "valuation_present": avail["valuation"],
            "institutional_present": avail["institutional"],
            "credibility_present": avail["credibility"],
            "tailwind_present": avail["tailwind"]
        })
        
    conn.close()
    
    df = pd.DataFrame(tickers_data)
    
    # 2. Universe Statistics
    avg_cov = df["coverage_score"].mean()
    med_cov = df["coverage_score"].median()
    
    dist = df["classification"].value_counts().to_dict()
    buckets = ["Complete", "Usable", "Weak", "Exclude"]
    for b in buckets:
        if b not in dist:
            dist[b] = 0
            
    # Calculate percentage of universe that is investable (Usable or Complete -> >= 70%)
    investable_count = sum(1 for item in tickers_data if item["coverage_score"] >= 70.0)
    investable_pct = (investable_count / len(tickers_data)) * 100.0
    
    # 3. Create Report Markdown
    report_lines = []
    report_lines.append("# data_coverage_audit: Data Coverage Report")
    report_lines.append("")
    report_lines.append("This report audits the data completeness of all tickers in the database based on the weighted coverage structure:")
    report_lines.append("*   Quality (Fundamentals): **25%**")
    report_lines.append("*   Growth (Multibagger): **20%**")
    report_lines.append("*   Valuation: **20%**")
    report_lines.append("*   Institutional (Canslim): **15%**")
    report_lines.append("*   Credibility: **10%**")
    report_lines.append("*   Tailwind: **10%**")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 1. Universe Summary Statistics")
    report_lines.append(f"*   **Total Tickers Audited**: {len(df)}")
    report_lines.append(f"*   **Average Coverage Score**: {avg_cov:.1f}%")
    report_lines.append(f"*   **Median Coverage Score**: {med_cov:.1f}%")
    report_lines.append(f"*   **Investable Universe (Score $\\ge 70\\%$)**: **{investable_pct:.1f}%** ({investable_count} out of {len(df)} tickers)")
    report_lines.append("")
    report_lines.append("### Coverage Classification Distribution")
    report_lines.append("| Classification | Coverage Range | Ticker Count | Percentage |")
    report_lines.append("| :--- | :---: | :---: | :---: |")
    for b in buckets:
        cnt = dist[b]
        pct = (cnt / len(df)) * 100.0
        report_lines.append(f"| **{b}** | {'>= 90%' if b == 'Complete' else '70-90%' if b == 'Usable' else '50-70%' if b == 'Weak' else 'Below 50%'} | {cnt} | {pct:.1f}% |")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 2. Ticker Coverage Details")
    report_lines.append("")
    report_lines.append("| Ticker | Coverage % | Classification | Missing Factors |")
    report_lines.append("| :--- | :---: | :---: | :--- |")
    
    # Sort tickers by coverage score descending
    df_sorted = df.sort_values(by="coverage_score", ascending=False)
    for _, row in df_sorted.iterrows():
        missing_str = ", ".join(row["missing_factors"]) if row["missing_factors"] else "*None*"
        report_lines.append(f"| **{row['ticker']}** | {row['coverage_score']:.1f}% | {row['classification']} | {missing_str} |")
        
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 3. Quantitative Key Question Verdict")
    report_lines.append("### *How much of the universe is actually investable?*")
    report_lines.append("")
    if investable_pct >= 70.0:
        report_lines.append(f"> [!IMPORTANT]")
        report_lines.append(f"> **SUFFICIENT COVERAGE**: **{investable_pct:.1f}%** of the universe is investable (Usable or Complete). The database is sufficiently populated to support quantitative ranking and factor validation.")
    else:
        report_lines.append("> [!WARNING]")
        report_lines.append(f"> **SEVERE COVERAGE GAP**: Only **{investable_pct:.1f}%** of the universe ({investable_count} out of {len(df)} tickers) is actually investable (Coverage Score $\\ge 70\\%$). The database has a critical weakness in factor coverage that will cause rank skews and concentration bias. Data collection must be prioritized.")
        
    report_lines.append("")
    report_lines.append(f"*Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Save Report
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "data_coverage_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[coverage_audit] Report successfully written to {artifact_path}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("DATABASE COVERAGE AUDIT SUMMARY")
    print("="*80)
    print(f"Total Tickers:                 {len(df)}")
    print(f"Average Coverage:              {avg_cov:.1f}%")
    print(f"Median Coverage:               {med_cov:.1f}%")
    print(f"Investable Universe (>=70%):   {investable_count} ({investable_pct:.1f}%)")
    print("="*80 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

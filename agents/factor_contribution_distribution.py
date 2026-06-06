#!/usr/bin/env python3
"""
factor_contribution_distribution.py — Audits factor contributions, effective weights,
and dominance statistics for all investable tickers (coverage >= 70%).
"""
from __future__ import annotations

import datetime
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opportunity_engine import recalculate_opportunity_scores

def make_ascii_histogram(values: list[float], bins: int = 5, min_val: float = 0.0, max_val: float = 100.0) -> str:
    """
    Generates a simple text-based ASCII histogram for a set of values.
    """
    if not values:
        return "No data"
    
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    counts, _ = np.histogram(values, bins=bin_edges)
    
    lines = []
    max_count = max(counts) if max(counts) > 0 else 1
    max_width = 15
    
    for i in range(bins):
        bin_label = f"[{bin_edges[i]:4.1f} - {bin_edges[i+1]:4.1f})"
        bar = "#" * int((counts[i] / max_count) * max_width)
        lines.append(f"{bin_label}: {bar:<15} ({counts[i]})")
        
    return "\n".join(lines)

def main() -> int:
    print("[contribution_audit] Recalculating scores and retrieving leaderboard...")
    leaderboard = recalculate_opportunity_scores()
    
    if not leaderboard:
        print("[contribution_audit] Error: Leaderboard is empty.")
        return 1
        
    # Filter for investable tickers (coverage >= 70%)
    investable = [item for item in leaderboard if item.get("coverage_score", 0.0) >= 70.0]
    
    if not investable:
        print("[contribution_audit] Error: No investable tickers found (coverage >= 70%).")
        return 1
        
    print(f"[contribution_audit] Running factor contribution audit on {len(investable)} investable tickers...")
    
    factors = ["quality", "growth", "valuation", "momentum", "institutional", "tailwind", "credibility"]
    
    # Accumulate metrics
    records = []
    for item in investable:
        ticker = item["ticker"]
        total_score = item["total_score"]
        
        row_record = {
            "ticker": ticker,
            "total_score": total_score,
            "coverage_score": item["coverage_score"]
        }
        
        # Add contributions and effective weights
        for f in factors:
            row_record[f"{f}_contrib"] = item["contributions"].get(f, 0.0)
            row_record[f"{f}_weight"] = item["norm_weights"].get(f, 0.0)
            row_record[f"{f}_raw"] = item["raw_scores"].get(f, 0.0)
            row_record[f"{f}_pct"] = item["percentiles"].get(f, 0.0)
            
        # Determine dominant factor (which factor contributed the most)
        contribs = {f: item["contributions"].get(f, 0.0) for f in factors}
        dominant_factor = max(contribs, key=contribs.get)
        row_record["dominant_factor"] = dominant_factor
        row_record["dominant_contrib"] = contribs[dominant_factor]
        
        records.append(row_record)
        
    df = pd.DataFrame(records)
    
    # Calculate statistics
    avg_contrib = {}
    max_contrib = {}
    avg_weight = {}
    
    for f in factors:
        avg_contrib[f] = df[f"{f}_contrib"].mean()
        max_contrib[f] = df[f"{f}_contrib"].max()
        avg_weight[f] = df[f"{f}_weight"].mean()
        
    dominance_counts = df["dominant_factor"].value_counts().to_dict()
    for f in factors:
        if f not in dominance_counts:
            dominance_counts[f] = 0
            
    # Write Report
    report_lines = []
    report_lines.append("# factor_contribution_distribution: Factor Contribution Audit Report")
    report_lines.append("")
    report_lines.append(f"This report audits the factor score contributions, effective normalized weights, and dominance distribution across the **{len(investable)}** investable tickers (Coverage Score $\\ge 70\\%$) in the active universe.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 1. Summary Statistics By Factor")
    report_lines.append("")
    report_lines.append("| Factor | Avg Effective Weight | Avg Contribution | Max Contribution | Dominance Count |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for f in factors:
        pct_weight = avg_weight[f] * 100.0
        report_lines.append(
            f"| **{f.capitalize()}** | {pct_weight:.1f}% | {avg_contrib[f]:.2f} | {max_contrib[f]:.2f} | {dominance_counts[f]} stocks |"
        )
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 2. Quantitative Key Question Verdicts")
    report_lines.append("")
    
    # Identify highest average contributor and dominant factor
    highest_avg_factor = max(avg_contrib, key=avg_contrib.get)
    dominance_winner = max(dominance_counts, key=dominance_counts.get)
    
    report_lines.append(f"### *1. How much of the final score comes from each factor?*")
    report_lines.append("Across all investable stocks, the average score breakdown is as follows:")
    for f in factors:
        contrib_pct = (avg_contrib[f] / df["total_score"].mean()) * 100.0 if df["total_score"].mean() > 0 else 0.0
        report_lines.append(f"*   **{f.capitalize()}**: **{avg_contrib[f]:.2f} points** ({contrib_pct:.1f}% of total score)")
    report_lines.append("")
    
    report_lines.append(f"### *2. Is Management Credibility actually driving rankings?*")
    report_lines.append("")
    if dominance_winner == "credibility" or highest_avg_factor == "credibility":
        report_lines.append("> [!IMPORTANT]")
        report_lines.append(f"> **YES, CREDIBILITY DRIVES RANKINGS**: Credibility is the dominant factor, winning dominance in **{dominance_counts['credibility']}** out of {len(investable)} investable tickers, and has an average contribution of **{avg_contrib['credibility']:.2f} points**. This is because corporate events have decayed to zero weight, leaving structural credibility as the largest driver.")
    else:
        report_lines.append("> [!NOTE]")
        report_lines.append(f"> **NO, ANOTHER FACTOR DOMINATES**: **{highest_avg_factor.capitalize()}** is the largest driver of final rankings, with an average contribution of **{avg_contrib[highest_avg_factor]:.2f} points** (dominant in {dominance_counts[highest_avg_factor]} stocks). Credibility remains a supporting factor with an average contribution of **{avg_contrib['credibility']:.2f} points**.")
        
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 3. Ticker Contribution Details")
    report_lines.append("")
    
    for _, row in df.iterrows():
        ticker = row["ticker"]
        report_lines.append(f"### **{ticker}** (Total Score: **{row['total_score']:.2f}**)")
        report_lines.append(f"*   **Dominant Factor**: **{row['dominant_factor'].capitalize()}** (Contributed **{row['dominant_contrib']:.2f}** points)")
        report_lines.append("")
        report_lines.append("| Factor | Raw Value | Percentile Rank | Effective Weight | Contribution | % of Score |")
        report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for f in factors:
            raw_val = row[f"{f}_raw"]
            pct_rank = row[f"{f}_pct"]
            eff_wt = row[f"{f}_weight"]
            contrib = row[f"{f}_contrib"]
            contrib_pct = (contrib / row["total_score"]) * 100.0 if row["total_score"] > 0 else 0.0
            report_lines.append(
                f"| {f.capitalize()} | {raw_val:.1f} | {pct_rank:.1f} | {eff_wt:.4f} | {contrib:.2f} | {contrib_pct:.1f}% |"
            )
        report_lines.append("")
        
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 4. Factor Contribution Text Histograms")
    report_lines.append("")
    
    for f in factors:
        report_lines.append(f"### **{f.capitalize()}** Contribution Histogram")
        report_lines.append("```text")
        hist_text = make_ascii_histogram(df[f"{f}_contrib"].tolist(), bins=5, min_val=0.0, max_val=25.0)
        report_lines.append(hist_text)
        report_lines.append("```")
        report_lines.append("")
        
    report_lines.append(f"*Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Save Report
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "factor_contribution_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[contribution_audit] Report successfully written to {artifact_path}")
    
    # Print statistics summary to console
    print("\n" + "="*80)
    print("FACTOR CONTRIBUTION AUDIT SUMMARY")
    print("="*80)
    for f in factors:
        print(f"{f.capitalize():<15} Avg Weight: {avg_weight[f]*100:>5.1f}%  Avg Contrib: {avg_contrib[f]:>5.2f}  Max Contrib: {max_contrib[f]:>5.2f}  Dominance: {dominance_counts[f]} stocks")
    print("="*80 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

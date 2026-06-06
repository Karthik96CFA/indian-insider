#!/usr/bin/env python3
"""
factor_attribution.py — Step 8: Factor Attribution Reporter.
Explains stock opportunity rankings using positive and negative factor attribution values.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    _conn,
    log,
)
from opportunity_engine import recalculate_opportunity_scores


def generate_attribution_report() -> int:
    print("[attribution] Calculating factor exposures and attributions...")
    
    # 1. Fetch updated leaderboard rankings
    leaderboard = recalculate_opportunity_scores()
    
    if not leaderboard:
        print("[attribution] Error: Leaderboard is empty. Verify company_scores table.")
        return 1
        
    # 2. Compute means of each factor across the universe to define the baseline
    n = len(leaderboard)
    keys = ["quality", "growth", "valuation", "momentum", "institutional", "tailwind", "credibility"]
    weights = {
        "quality": 0.20,
        "growth": 0.20,
        "valuation": 0.20,
        "momentum": 0.15,
        "institutional": 0.10,
        "tailwind": 0.10,
        "credibility": 0.05
    }
    
    means = {k: sum(item[k] for item in leaderboard) / n for k in keys}
    
    # 3. Print attribution leaderboard
    print("\n" + "="*95)
    print(f"{'Rank':<5} {'Ticker':<10} {'Total':<8} {'Positive Attributions (+)':<34} {'Negative Attributions (-)':<34}")
    print("="*95)
    
    for idx, item in enumerate(leaderboard[:15]): # print top 15 for brevity
        pos_attr = []
        neg_attr = []
        
        for k in keys:
            ticker_val = item[k]
            mean_val = means[k]
            weight = weights[k]
            
            # Weighted contribution difference
            diff = (ticker_val - mean_val) * weight
            
            if diff > 1.5: # threshold for significant positive driver
                pos_attr.append(f"{k.capitalize()} (+{diff:.1f})")
            elif diff < -1.5: # threshold for significant negative drag
                neg_attr.append(f"{k.capitalize()} ({diff:.1f})")
                
        pos_str = ", ".join(pos_attr) if pos_attr else "None"
        neg_str = ", ".join(neg_attr) if neg_attr else "None"
        
        print(f"{idx+1:<5} {item['ticker']:<10} {item['total_score']:<8.2f} {pos_str:<34} {neg_str:<34}")
        
    print("="*95 + "\n")
    
    # Write a detailed markdown report for documentation
    report_content = []
    report_content.append("# Step 8: Multi-Factor Attribution Report")
    report_content.append("")
    report_content.append("This report breaks down the stock opportunity leaderboard rankings into positive and negative factor attribution values (excess contributions over the universe mean).")
    report_content.append("")
    report_content.append("## 1. Factor Weights & Universe Averages")
    report_content.append("")
    report_content.append("| Factor | Weight (%) | Universe Average Score (0-100) |")
    report_content.append("| :--- | :---: | :---: |")
    for k in keys:
        report_content.append(f"| **{k.capitalize()}** | {weights[k]*100.0:.1f}% | {means[k]:.1f} |")
    report_content.append("")
    report_content.append("## 2. Leaderboard Attribution Breakdown (Top 10)")
    report_content.append("")
    report_content.append("| Rank | Ticker | Total Score | Positive Drivers (Excess Contribution > +1.0) | Negative Drags (Excess Contribution < -1.0) |")
    report_content.append("| :--- | :--- | :---: | :--- | :--- |")
    
    for idx, item in enumerate(leaderboard[:10]):
        pos_attr = []
        neg_attr = []
        for k in keys:
            diff = (item[k] - means[k]) * weights[k]
            if diff > 1.0:
                pos_attr.append(f"**{k.capitalize()}** (+{diff:.1f})")
            elif diff < -1.0:
                neg_attr.append(f"*{k.capitalize()}* ({diff:.1f})")
        pos_str = ", ".join(pos_attr) if pos_attr else "Neutral"
        neg_str = ", ".join(neg_attr) if neg_attr else "Neutral"
        report_content.append(f"| {idx+1} | **{item['ticker']}** | {item['total_score']:.2f} | {pos_str} | {neg_str} |")
        
    report_content.append("")
    report_content.append("## 3. Explanatory Rationale")
    report_content.append("")
    report_content.append("> [!TIP]")
    report_content.append("> *   **Positive Drivers** identify the specific quantitative factors pulling a stock up relative to the universe. For instance, a stock with a high **Credibility** or **Momentum** score will have a strong positive attribution from these factors.")
    report_content.append("> *   **Negative Drags** expose the factor weaknesses of ranked companies. A company may be ranked highly overall due to Quality and Growth, but suffer a negative drag from **Valuation** (expensive multiples) or **Tailwind** (stagnant sector). This lets the Chairperson make balanced, informed decisions.")
    
    report_content.append("")
    report_content.append(f"*Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "factor_attribution_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(report_content), encoding="utf-8")
    print(f"[attribution] Factor attribution report successfully written to {artifact_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(generate_attribution_report())

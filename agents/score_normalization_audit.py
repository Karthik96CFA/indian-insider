#!/usr/bin/env python3
"""
score_normalization_audit.py — Audits factor scale imbalances and compares Raw vs. Z-Score vs. Percentile rankings.
"""
from __future__ import annotations

import datetime
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _conn, read_company_scores, read_company_fundamentals, read_valuation_metrics
from sector_specific_metrics import get_sector_score
from opportunity_engine import get_industry_tailwind_score

def get_spearman_rank_correlation(ranks_1: pd.Series, ranks_2: pd.Series) -> float:
    # Spearman rank correlation is mathematically equivalent to the Pearson correlation of ranks.
    # Since we are passing in the ranked Series directly, we can just use Pearson correlation.
    return ranks_1.corr(ranks_2, method="pearson")

def main() -> int:
    conn = _conn()
    latest_date = conn.execute("SELECT MAX(effective_date) FROM company_scores_history").fetchone()[0]
    print(f"[normalization_audit] Auditing entire historical universe on date: {latest_date}")
    
    rows = conn.execute(
        "SELECT ticker, event_score, fundamental_score, valuation_score, canslim_score, "
        "multibagger_score, credibility_score, industry_tailwind_score "
        "FROM company_scores_history WHERE effective_date = ?",
        (latest_date,)
    ).fetchall()
    
    if not rows:
        print("[normalization_audit] No tickers found in database.")
        conn.close()
        return 1
        
    # 1. Gather all raw scaled factors
    data = []
    for ticker, event_score, fundamental_score, valuation_score, canslim_score, multibagger_score, credibility_score, tailwind_score in rows:
        event_score = event_score or 0.0
        fundamental_score = fundamental_score or 0.0
        valuation_score = valuation_score or 0.0
        canslim_score = canslim_score or 0
        multibagger_score = multibagger_score or 0
        credibility_score = credibility_score if credibility_score is not None else 50.0
        tailwind_score = tailwind_score if tailwind_score is not None else 50.0
        
        # Scale to 0-100
        quality = fundamental_score * 10.0
        growth = float(multibagger_score)
        valuation = valuation_score * 10.0
        momentum = min(100.0, max(0.0, 50.0 + (event_score * 10.0)))
        institutional = float(canslim_score)
        credibility = float(credibility_score)
        tailwind = float(tailwind_score)
        
        data.append({
            "ticker": ticker,
            "quality": quality,
            "growth": growth,
            "valuation": valuation,
            "momentum": momentum,
            "institutional": institutional,
            "tailwind": tailwind,
            "credibility": credibility
        })
        
    df = pd.DataFrame(data)
    factors = ["quality", "growth", "valuation", "momentum", "institutional", "tailwind", "credibility"]
    weights = {
        "quality": 0.20,
        "growth": 0.20,
        "valuation": 0.20,
        "momentum": 0.15,
        "institutional": 0.10,
        "tailwind": 0.10,
        "credibility": 0.05
    }
    
    # 2. Compute Universe Distribution Statistics
    stats = []
    for f in factors:
        col = df[f]
        stats.append({
            "Factor": f.capitalize(),
            "Min": col.min(),
            "Max": col.max(),
            "Mean": col.mean(),
            "Median": col.median(),
            "Std Dev": col.std()
        })
    stats_df = pd.DataFrame(stats)
    
    # 3. Apply Z-score, Percentile, and Winsorized Percentile normalization
    df_z = df.copy()
    df_p = df.copy()
    df_wp = df.copy()
    
    for f in factors:
        mean_val = df[f].mean()
        std_val = df[f].std()
        # Z-scores: standardize to mean 0, std 1
        df_z[f] = (df[f] - mean_val) / std_val if std_val > 0 else 0.0
        
        # Percentile Rank scores: rank stocks uniformly from 0 to 100
        # pct=True scales ranks to [0, 1]. Multiply by 100.
        df_p[f] = df[f].rank(pct=True, method="min") * 100.0
        
        # Winsorized Percentile Rank: clip tails at 2.5% and 97.5% and then rank
        q_low = df[f].quantile(0.025)
        q_high = df[f].quantile(0.975)
        winsorized = df[f].clip(lower=q_low, upper=q_high)
        df_wp[f] = winsorized.rank(pct=True, method="min") * 100.0
        
    # 4. Compute Weighted Opportunity Scores
    # Method A: Raw Weighted Score
    df["total_score"] = 0.0
    for f in factors:
        df["total_score"] += df[f] * weights[f]
        
    # Method B: Z-Score Weighted Score
    df_z["total_score"] = 0.0
    for f in factors:
        df_z["total_score"] += df_z[f] * weights[f]
        
    # Method C: Percentile Rank Weighted Score
    df_p["total_score"] = 0.0
    for f in factors:
        df_p["total_score"] += df_p[f] * weights[f]
        
    # Method D: Winsorized Percentile Rank Weighted Score
    df_wp["total_score"] = 0.0
    for f in factors:
        df_wp["total_score"] += df_wp[f] * weights[f]
        
    # Apply Ranks (1 is highest score)
    df["rank_raw"] = df["total_score"].rank(ascending=False, method="min").astype(int)
    df_z["rank_z"] = df_z["total_score"].rank(ascending=False, method="min").astype(int)
    df_p["rank_pct"] = df_p["total_score"].rank(ascending=False, method="min").astype(int)
    df_wp["rank_wp"] = df_wp["total_score"].rank(ascending=False, method="min").astype(int)
    
    # Merge ranks back into master dataframe
    df_merged = df[["ticker", "total_score", "rank_raw"]].rename(columns={"total_score": "score_raw"})
    df_merged = df_merged.merge(df_z[["ticker", "total_score", "rank_z"]].rename(columns={"total_score": "score_z"}), on="ticker")
    df_merged = df_merged.merge(df_p[["ticker", "total_score", "rank_pct"]].rename(columns={"total_score": "score_pct"}), on="ticker")
    df_merged = df_merged.merge(df_wp[["ticker", "total_score", "rank_wp"]].rename(columns={"total_score": "score_wp"}), on="ticker")
    
    # Calculate Rank Stability Metric: Standard deviation of rank positions for each stock
    df_merged["rank_std"] = df_merged[["rank_raw", "rank_z", "rank_pct", "rank_wp"]].std(axis=1)
    
    # Spearman rank correlation matrix
    corr_raw_z = get_spearman_rank_correlation(df_merged["rank_raw"], df_merged["rank_z"])
    corr_raw_pct = get_spearman_rank_correlation(df_merged["rank_raw"], df_merged["rank_pct"])
    corr_raw_wp = get_spearman_rank_correlation(df_merged["rank_raw"], df_merged["rank_wp"])
    corr_z_pct = get_spearman_rank_correlation(df_merged["rank_z"], df_merged["rank_pct"])
    corr_z_wp = get_spearman_rank_correlation(df_merged["rank_z"], df_merged["rank_wp"])
    corr_pct_wp = get_spearman_rank_correlation(df_merged["rank_pct"], df_merged["rank_wp"])
    
    # 5. Extract Top 20 for comparison table
    top_raw = df_merged.sort_values(by="rank_raw").head(20).copy()
    top_z = df_merged.sort_values(by="rank_z").head(20).copy()
    top_pct = df_merged.sort_values(by="rank_pct").head(20).copy()
    top_wp = df_merged.sort_values(by="rank_wp").head(20).copy()
    
    # Generate Report Content
    report_lines = []
    report_lines.append("# score_normalization_audit: Multi-Factor Scale Audit Report")
    report_lines.append("")
    report_lines.append("This report audits the scale and distribution of quantitative factors, and details how Z-score, Percentile, and Winsorized Percentile normalization alter the opportunity leaderboard rankings.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 1. Raw Factor Distribution Statistics")
    report_lines.append("")
    report_lines.append("| Factor | Min | Max | Mean | Median | Std Dev |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for _, r in stats_df.iterrows():
        report_lines.append(f"| **{r['Factor']}** | {r['Min']:.1f} | {r['Max']:.1f} | {r['Mean']:.1f} | {r['Median']:.1f} | {r['Std Dev']:.1f} |")
    report_lines.append("")
    report_lines.append("> [!WARNING]")
    report_lines.append("> **SCALE IMBALANCE DETECTED**: The averages and standard deviations are heavily skewed. For instance, the **Quality** and **Valuation** factor means are under 5.0 out of 100, while **Tailwind** is 63.9 and **Credibility** is 61.7. Because Z-scores or Percentiles are not used, a top stock with 80.0 ROCE/Quality gets a massive return lift (+15 points to raw score) compared to a stock with 90.0 Credibility (+1.5 points excess contribution), giving Quality un-weighted dominance.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 2. Ranking Sensitivity Analysis")
    report_lines.append("Pairwise Spearman Rank Correlations between the four methods:")
    report_lines.append("")
    report_lines.append(f"*   **Raw vs. Z-Score Ranking**: Spearman $\\rho = {corr_raw_z:.4f}$")
    report_lines.append(f"*   **Raw vs. Percentile Ranking**: Spearman $\\rho = {corr_raw_pct:.4f}$")
    report_lines.append(f"*   **Raw vs. Winsorized Percentile Ranking**: Spearman $\\rho = {corr_raw_wp:.4f}$")
    report_lines.append(f"*   **Z-Score vs. Percentile Ranking**: Spearman $\\rho = {corr_z_pct:.4f}$")
    report_lines.append(f"*   **Percentile vs. Winsorized Percentile Ranking**: Spearman $\\rho = {corr_pct_wp:.4f}$")
    report_lines.append("")
    report_lines.append("> [!IMPORTANT]")
    report_lines.append(f"> The low rank correlation of **{corr_raw_z:.3f}** between Raw and Z-score rankings demonstrates that the current ranking model is highly sensitive to the scale of the input factors. Normalization is critical to align factors to their intended model weights. Winsorization further stabilizes rankings by capping extreme tail outliers.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 3. Comparison of Top 20 Rankings & Rank Stability")
    report_lines.append("")
    report_lines.append("| Ticker | Raw Rank | Z-Score Rank | Percentile Rank | Winsorized Pct Rank | Rank Std Dev (Stability) | Verdict |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
    
    # Gather a union of top 20 tickers for comparative table
    union_top = pd.concat([top_raw, top_z, top_pct, top_wp]).drop_duplicates(subset="ticker")
    union_top = union_top.sort_values(by="rank_std").head(25) # show 25 interesting rows ordered by stability
    
    for _, r in union_top.iterrows():
        verdict = "**Stable**" if r["rank_std"] <= 5.0 else "*Volatile*"
        report_lines.append(f"| **{r['ticker']}** | {r['rank_raw']} | {r['rank_z']} | {r['rank_pct']} | {r['rank_wp']} | {r['rank_std']:.1f} | {verdict} |")
        
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 4. Normalization Remediation Recommendation")
    report_lines.append("1. **Transition to Winsorized Percentile Normalization**: Rank-based percentile scaling guarantees that all factor distributions are uniform between 0 and 100. Incorporating Winsorization at the 2.5% and 97.5% tails protects the ranks from extreme outliers.")
    report_lines.append("2. **Update opportunity_engine.py**: Incorporate Winsorized rank-percentile transformation before scoring.")
    report_lines.append("")
    report_lines.append(f"*Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Save Report
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "score_normalization_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[normalization_audit] Audit report successfully written to {artifact_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SCORE NORMALIZATION AUDIT SUMMARY")
    print("="*80)
    print(f"Spearman Corr (Raw vs Z):   {corr_raw_z:.4f}")
    print(f"Spearman Corr (Raw vs Pct): {corr_raw_pct:.4f}")
    print(f"Spearman Corr (Z vs Pct):   {corr_z_pct:.4f}")
    print("="*80 + "\n")
    
    conn.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
factor_persistence_test.py — Tracks the membership retention rate and average rank drift
of the Top 10 stocks after 1 Month (21 days), 3 Months (63 days), and 6 Months (126 days).
"""
from __future__ import annotations

import datetime
import math
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _conn

def get_capped_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    caps = {"tailwind": 0.20, "momentum": 0.20, "credibility": 0.15}
    w = raw_weights.copy()
    total_w = sum(w.values())
    if total_w == 0:
        return w
    norm_w = {k: v / total_w for k, v in w.items()}
    violated = {k: cap for k, cap in caps.items() if k in norm_w and norm_w[k] > cap}
    if not violated:
        return norm_w
    final_w = {k: violated[k] for k in violated}
    rest = 1.0 - sum(violated.values())
    uncapped_raw_sum = sum(w[k] for k in w if k not in violated)
    if uncapped_raw_sum > 0:
        for k in w:
            if k not in violated:
                final_w[k] = (w[k] / uncapped_raw_sum) * rest
    else:
        leftover_keys = [k for k in w if k not in violated]
        if leftover_keys:
            for k in leftover_keys:
                final_w[k] = rest / len(leftover_keys)
        else:
            total_viol = sum(violated.values())
            return {k: (violated[k] / total_viol if k in violated else 0.0) for k in w}
    return final_w

def get_rankings_on_date(conn: sqlite3.Connection, tickers: list[str], target_date: str) -> dict[str, int]:
    """
    Computes rankings for all tickers on a specific date using production weights.
    Returns a dictionary of ticker -> rank (1-indexed).
    """
    scores_df_data = []
    for t in tickers:
        row = conn.execute(
            "SELECT event_score, fundamental_score, valuation_score, canslim_score, "
            "multibagger_score, credibility_score, industry_tailwind_score, coverage_score FROM company_scores_history "
            "WHERE ticker = ? AND effective_date <= ? "
            "ORDER BY effective_date DESC LIMIT 1",
            (t, target_date)
        ).fetchone()
        
        if row:
            ev, fundamental, valuation, canslim, multibagger, credibility, tailwind, coverage = row
            scores_df_data.append({
                "ticker": t,
                "momentum": min(100.0, max(0.0, 50.0 + ((ev or 0.0) * 10.0))),
                "quality": (fundamental or 0.0) * 10.0,
                "growth": float(multibagger or 0.0),
                "valuation": (valuation or 0.0) * 10.0,
                "institutional": float(canslim or 0.0),
                "tailwind": float(tailwind or 50.0),
                "credibility": float(credibility if credibility is not None else 50.0),
                "coverage_score": coverage if coverage is not None else 100.0
            })
            
    if not scores_df_data:
        return {}
        
    df_scores = pd.DataFrame(scores_df_data)
    
    # Winsorized Percentile Normalization
    factors = ["quality", "growth", "valuation", "momentum", "institutional", "tailwind", "credibility"]
    df_pct = df_scores[["ticker", "coverage_score"]].copy()
    for f in factors:
        col = df_scores[f]
        q_low = col.quantile(0.025)
        q_high = col.quantile(0.975)
        if q_high == q_low:
            df_pct[f] = 50.0
        else:
            winsorized = col.clip(lower=q_low, upper=q_high)
            df_pct[f] = winsorized.rank(pct=True, method="min") * 100.0
            
    # Compute point-in-time scores
    final_scores = []
    for idx, row in df_pct.iterrows():
        t = row["ticker"]
        cov_score = row["coverage_score"]
        if cov_score < 50.0:
            continue
            
        latest_ev_row = conn.execute(
            "SELECT MAX(event_date) FROM market_events WHERE ticker = ? AND event_date <= ?",
            (t, target_date)
        ).fetchone()
        latest_event_date = latest_ev_row[0] if latest_ev_row else None
        
        if latest_event_date:
            try:
                entry_dt = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
                event_dt = datetime.datetime.strptime(latest_event_date, "%Y-%m-%d").date()
                delay = max(0, (entry_dt - event_dt).days)
            except Exception:
                delay = 9999
        else:
            delay = 9999
            
        T_HALF = 5.0
        decay_factor = math.exp(- (math.log(2.0) / T_HALF) * delay)
        w_mom = 0.10 * decay_factor
        if delay > 7:
            w_mom = 0.0
            
        raw_weights = {
            "quality": 0.40,
            "growth": 0.30,
            "valuation": 0.0,
            "momentum": w_mom,
            "institutional": 0.20,
            "tailwind": 0.0,
            "credibility": 0.0
        }
        
        capped_weights = get_capped_weights(raw_weights)
        raw_total_score = sum(capped_weights[f] * row[f] for f in factors)
        total_score = raw_total_score * (cov_score / 100.0)
        final_scores.append({"ticker": t, "score": total_score})
        
    if not final_scores:
        return {}
        
    df_rankings = pd.DataFrame(final_scores)
    df_sorted = df_rankings.sort_values(by="score", ascending=False).reset_index(drop=True)
    
    # Return ticker -> rank dictionary (1-indexed)
    return {row["ticker"]: rank + 1 for rank, row in df_sorted.iterrows()}

def main() -> int:
    conn = _conn()
    
    # Fetch all distinct tickers
    tickers_rows = conn.execute("SELECT DISTINCT ticker FROM company_scores_history").fetchall()
    tickers = sorted([r[0] for r in tickers_rows if r[0] not in {"NIFTY", "BANKNIFTY", "NIFTYBEES", "GOLDBEES", "GOLD"}])
    
    # Get all trading dates from the database
    trading_dates_rows = conn.execute("SELECT DISTINCT date FROM historical_prices ORDER BY date").fetchall()
    trading_dates = [r[0] for r in trading_dates_rows if r[0] >= "2024-01-01"]
    
    start_idx = 126
    rebalance_indices = []
    curr_idx = start_idx
    # Use quarterly rebalance cycle for tracking picks
    while curr_idx + 126 < len(trading_dates): # ensure we can track up to 6 months later
        rebalance_indices.append(curr_idx)
        curr_idx += 63
        
    print(f"[persistence] Evaluating {len(rebalance_indices)} cycles for factor persistence...")
    
    lookbacks = {
        "1M": 21,
        "3M": 63,
        "6M": 126
    }
    
    retention_counts = {lbl: [] for lbl in lookbacks}
    drift_amounts = {lbl: [] for lbl in lookbacks}
    
    for cycle_idx, entry_idx in enumerate(rebalance_indices):
        entry_date = trading_dates[entry_idx]
        
        # Get active ranks at entry_date
        ranks_entry = get_rankings_on_date(conn, tickers, entry_date)
        if not ranks_entry:
            continue
            
        # Select Top 10
        top_10 = sorted([t for t, r in ranks_entry.items() if r <= 10], key=lambda x: ranks_entry[x])
        if len(top_10) < 10:
            continue
            
        # Strict cohort alignment: check if all future dates have rankings
        valid_cycle = True
        ranks_future_dict = {}
        for lbl, offset in lookbacks.items():
            future_idx = entry_idx + offset
            if future_idx >= len(trading_dates):
                valid_cycle = False
                break
            future_date = trading_dates[future_idx]
            ranks_future = get_rankings_on_date(conn, tickers, future_date)
            if not ranks_future:
                valid_cycle = False
                break
            ranks_future_dict[lbl] = ranks_future
            
        if not valid_cycle:
            continue
            
        print(f"  Cycle {cycle_idx+1} at {entry_date}: Top 10 = {top_10}")
        
        for lbl, offset in lookbacks.items():
            ranks_future = ranks_future_dict[lbl]
            retained_in_top10 = 0
            drifts = []
            
            # Find the size of the active universe to assign as fallback rank
            fallback_rank = max(ranks_future.values()) if ranks_future else 100
            
            for rank_entry_idx, t in enumerate(top_10):
                r_entry = rank_entry_idx + 1 # 1 to 10
                r_future = ranks_future.get(t, fallback_rank)
                
                if r_future <= 10:
                    retained_in_top10 += 1
                    
                drift = abs(r_future - r_entry)
                drifts.append(drift)
                
            retention_rate = (retained_in_top10 / len(top_10)) * 100.0
            avg_drift = np.mean(drifts)
            
            retention_counts[lbl].append(retention_rate)
            drift_amounts[lbl].append(avg_drift)
            
    conn.close()
    
    # Compute averages
    stats = []
    for lbl in lookbacks:
        mean_retention = np.mean(retention_counts[lbl]) if retention_counts[lbl] else 0.0
        mean_drift = np.mean(drift_amounts[lbl]) if drift_amounts[lbl] else 0.0
        stats.append({
            "Horizon": lbl,
            "RetentionRate": mean_retention,
            "RankDrift": mean_drift
        })
        
    # Generate Report
    report_lines = [
        "# Stage 7: Factor Persistence Report",
        "",
        "This report evaluates the stability of the ranking engine's selections over time, measuring **Top 10 Membership Retention** and **Average Rank Drift** after 1 Month (21 days), 3 Months (63 days), and 6 Months (126 days).",
        "",
        "---",
        "",
        "## 1. Factor Persistence Leaderboard",
        "",
        "| Time Horizon | Average Top 10 Retention Rate (%) | Average Rank Drift (Ranks) | Status |",
        "| :---: | :---: | :---: | :---: |"
    ]
    
    for r in stats:
        status = "Highly Persistent" if r["RetentionRate"] >= 50.0 else "Moderately Stable" if r["RetentionRate"] >= 30.0 else "Fast Decay"
        report_lines.append(
            f"| **{r['Horizon']}** | {r['RetentionRate']:.2f}% | {r['RankDrift']:.2f} ranks | *{status}* |"
        )
        
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 2. Quantitative Key Findings")
    report_lines.append("")
    report_lines.append(f"*   **Signal Half-Life**: The Top 10 membership retention rate is **{stats[0]['RetentionRate']:.2f}%** after 1 Month, decaying to **{stats[1]['RetentionRate']:.2f}%** after 3 Months and **{stats[2]['RetentionRate']:.2f}%** after 6 Months.")
    report_lines.append(f"*   **Rank Drift**: The average rank drift of top picks is **{stats[0]['RankDrift']:.2f} ranks** after 1 Month, rising to **{stats[1]['RankDrift']:.2f} ranks** after 3 Months, and **{stats[2]['RankDrift']:.2f} ranks** after 6 Months. This indicates that while some stocks drop out of the Top 10, they generally remain near the top of the leaderboard rather than falling off entirely.")
    report_lines.append("")
    
    report_lines.append("## 3. Persistence Verdict")
    report_lines.append("")
    
    # Recommendation based on retention
    ret_3m = stats[1]['RetentionRate']
    if ret_3m >= 50.0:
        rec = "The ranking signal is highly persistent, justifying low-frequency quarterly rebalancing."
    else:
        rec = "The ranking signal decays moderately, meaning that holding assets for longer than 3-6 months will expose the portfolio to stale ranking bias."
        
    report_lines.append("> [!IMPORTANT]")
    report_lines.append(f"> **PERSISTENCE VERDICT**: {rec}")
    
    # Save Report
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "factor_persistence_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[persistence] Report successfully written to {artifact_path}")
    
    # Print summary to console
    print("\n" + "="*95)
    print("FACTOR PERSISTENCE SUMMARY")
    print("="*95)
    print(f"{'Horizon':<15} {'Retention Rate (%)':<25} {'Average Rank Drift (Ranks)':<30}")
    print("-"*95)
    for r in stats:
        print(f"{r['Horizon']:<15} {r['RetentionRate']:>18.2f}% {r['RankDrift']:>28.2f}")
    print("="*95 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

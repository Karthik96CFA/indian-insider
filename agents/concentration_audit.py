#!/usr/bin/env python3
"""
concentration_audit.py — Analyzes portfolio concentration metrics (HHI, Top Position,
Top 5, and Sector allocation) to audit concentration risk under the new production weights.
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
from common import _conn, read_company_fundamentals
from backtester import get_variable_transaction_cost
from opportunity_engine import get_industry_tailwind_score
from sector_specific_metrics import get_sector_score

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

def main() -> int:
    conn = _conn()
    
    # 1. Fetch all distinct tickers
    tickers_rows = conn.execute("SELECT DISTINCT ticker FROM company_scores_history").fetchall()
    tickers = sorted([r[0] for r in tickers_rows if r[0] not in {"NIFTY", "BANKNIFTY", "NIFTYBEES", "GOLDBEES", "GOLD"}])
    
    # Load daily prices to compute lookback returns for Volatility Target weighting
    print("[concentration] Loading price history...")
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
        print(f"[concentration] Bulk prices download failed: {exc}. Cannot proceed.")
        conn.close()
        return 1
        
    trading_dates = prices_df.index.strftime("%Y-%m-%d").tolist()
    
    start_idx = 126
    rebalance_indices = []
    curr_idx = start_idx
    while curr_idx + 63 < len(trading_dates):
        rebalance_indices.append(curr_idx)
        curr_idx += 63
        
    print(f"[concentration] Auditing {len(rebalance_indices)} rebalance cycles...")
    
    # Ticker to Sector map
    ticker_sectors = {}
    for t in tickers:
        fundamentals = read_company_fundamentals(t)
        sector = "General"
        if fundamentals:
            try:
                _, sector, _ = get_sector_score(t, fundamentals)
            except Exception:
                sector = fundamentals.get("sector") or "General"
        ticker_sectors[t] = sector
        
    # Analysis records
    hhi_values = []
    top_pos_weights = []
    top_5_weights = []
    sector_weights_all = []
    
    # Focus analysis: Quality vs IT, Institutional vs Financials
    quality_scores_all = []
    inst_scores_all = []
    it_flags = []
    fin_flags = []
    
    for cycle_idx, entry_idx in enumerate(rebalance_indices):
        entry_date = trading_dates[entry_idx]
        
        # Fetch scores
        scores_df_data = []
        for t in tickers:
            row = conn.execute(
                "SELECT event_score, fundamental_score, valuation_score, canslim_score, "
                "multibagger_score, credibility_score, industry_tailwind_score, coverage_score FROM company_scores_history "
                "WHERE ticker = ? AND effective_date <= ? "
                "ORDER BY effective_date DESC LIMIT 1",
                (t, entry_date)
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
                
        # Point-In-Time momentum decay and final score
        final_scores = []
        for idx, row in df_pct.iterrows():
            t = row["ticker"]
            cov_score = row["coverage_score"]
            if cov_score < 50.0:
                continue
                
            latest_ev_row = conn.execute(
                "SELECT MAX(event_date) FROM market_events WHERE ticker = ? AND event_date <= ?",
                (t, entry_date)
            ).fetchone()
            latest_event_date = latest_ev_row[0] if latest_ev_row else None
            
            if latest_event_date:
                try:
                    entry_dt = datetime.datetime.strptime(entry_date, "%Y-%m-%d").date()
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
            
            final_scores.append({
                "ticker": t,
                "score": total_score,
                "quality": row["quality"],
                "institutional": row["institutional"]
            })
            
        df_rankings = pd.DataFrame(final_scores)
        
        # Collect sector data for correlation
        for idx, row in df_rankings.iterrows():
            sec = ticker_sectors[row["ticker"]]
            quality_scores_all.append(row["quality"])
            inst_scores_all.append(row["institutional"])
            it_flags.append(1.0 if sec == "IT Services" else 0.0)
            fin_flags.append(1.0 if sec == "Banking" else 0.0)
            
        # Filter valid entry prices
        valid_tickers = [t for t in df_rankings["ticker"].tolist() if t in prices_df.columns and not pd.isna(prices_df.loc[entry_date, t]) and prices_df.loc[entry_date, t] > 0]
        df_rankings_filtered = df_rankings[df_rankings["ticker"].isin(valid_tickers)].copy()
        
        # Select Top 10
        top_10 = df_rankings_filtered.sort_values(by="score", ascending=False).head(10)["ticker"].tolist()
        if len(top_10) == 0:
            continue
            
        # Volatility Target weighting for the Top 10
        lookback_df = prices_df.iloc[entry_idx - 126 : entry_idx][top_10].ffill().bfill()
        lookback_returns = lookback_df.pct_change().fillna(0.0)
        vols = lookback_returns.std().to_numpy()
        vols = np.where(vols <= 0.0, 1e-4, vols)
        inv_vols = 1.0 / vols
        weights = inv_vols / np.sum(inv_vols)
        
        # Compute concentration metrics
        hhi = np.sum((weights * 100.0) ** 2)
        top_pos = np.max(weights) * 100.0
        top_5 = np.sum(sorted(weights, reverse=True)[:5]) * 100.0
        
        hhi_values.append(hhi)
        top_pos_weights.append(top_pos)
        top_5_weights.append(top_5)
        
        # Sector concentration
        sector_alloc = {}
        for i, t in enumerate(top_10):
            sec = ticker_sectors[t]
            sector_alloc[sec] = sector_alloc.get(sec, 0.0) + weights[i] * 100.0
        sector_weights_all.append(sector_alloc)
        
    conn.close()
    
    # Averages
    mean_hhi = np.mean(hhi_values)
    mean_top_pos = np.mean(top_pos_weights)
    mean_top_5 = np.mean(top_5_weights)
    
    # Average sector weights
    avg_sector_weights = {}
    for alloc in sector_weights_all:
        for sec, w in alloc.items():
            avg_sector_weights[sec] = avg_sector_weights.get(sec, 0.0) + w / len(sector_weights_all)
            
    # Correlation analysis: Quality vs IT, Institutional vs Banking
    quality_scores_all = np.array(quality_scores_all)
    inst_scores_all = np.array(inst_scores_all)
    it_flags = np.array(it_flags)
    fin_flags = np.array(fin_flags)
    
    # Point-biserial correlation
    corr_quality_it = np.corrcoef(quality_scores_all, it_flags)[0, 1] if len(it_flags) > 1 and np.std(it_flags) > 0 else 0.0
    corr_inst_fin = np.corrcoef(inst_scores_all, fin_flags)[0, 1] if len(fin_flags) > 1 and np.std(fin_flags) > 0 else 0.0
    
    # Generate Report
    report_lines = [
        "# Stage 7: Portfolio Concentration Audit Report",
        "",
        "This report audits portfolio concentration risk across assets, sectors, and factors under the new production multi-factor weights.",
        "",
        "---",
        "",
        "## 1. Portfolio Concentration Metrics",
        "",
        f"*   **Average Herfindahl-Hirschman Index (HHI)**: **{mean_hhi:.2f}** (A score of 1000 is perfectly equal for 10 assets; < 1500 indicates low concentration)",
        f"*   **Average Top Position Weight**: **{mean_top_pos:.2f}%**",
        f"*   **Average Top 5 Positions Weight**: **{mean_top_5:.2f}%**",
        "",
        "### Sector Concentration Breakdown",
        "",
        "| Sector | Average Allocation (%) |",
        "| :--- | :---: |"
    ]
    
    for sec, w in sorted(avg_sector_weights.items(), key=lambda x: x[1], reverse=True):
        report_lines.append(f"| **{sec}** | {w:.2f}% |")
        
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 2. Factor Concentration Diagnostics")
    report_lines.append("Audits whether specific factors create structural sector biases in the active scoring universe.")
    report_lines.append("")
    report_lines.append(f"*   **Quality Score vs. IT Services Sector Correlation**: **{corr_quality_it:+.4f}**")
    report_lines.append(f"    *   *Interpretation*: A positive correlation indicates that higher Quality scores tend to select IT Services companies. (Correlation: **{corr_quality_it:+.4f}**)")
    report_lines.append(f"*   **Institutional Score vs. Banking Sector Correlation**: **{corr_inst_fin:+.4f}**")
    report_lines.append(f"    *   *Interpretation*: A positive correlation indicates that higher Institutional scores tend to select Banking/Financial companies. (Correlation: **{corr_inst_fin:+.4f}**)")
    report_lines.append("")
    report_lines.append("### Key Concentration Questions answered:")
    report_lines.append("")
    
    # Quality forces IT Services?
    if corr_quality_it > 0.30:
        ans_it = "Yes, the Quality factor exhibits a strong positive correlation with IT Services, meaning it structurally forces IT sector concentration."
    else:
        ans_it = "No, the Quality factor does not force IT Services concentration (correlation is weak/moderate), showing that high-quality firms are selected across various industries."
        
    # Institutional forces Banking?
    if corr_inst_fin > 0.30:
        ans_fin = "Yes, the Institutional factor (Canslim/Mutual Fund ownership) exhibits a strong positive correlation with Banking/Financials, forcing financial sector concentration."
    else:
        ans_fin = "No, the Institutional factor does not force Banking concentration, showing that institutions accumulate shares across multiple sectors."
        
    report_lines.append(f"1. **Does the Quality factor force IT concentration?**")
    report_lines.append(f"   *   *Verdict*: {ans_it}")
    report_lines.append("")
    report_lines.append(f"2. **Does the Institutional factor force Financial concentration?**")
    report_lines.append(f"   *   *Verdict*: {ans_fin}")
    report_lines.append("")
    
    report_lines.append("## 3. Concentration Verdict")
    report_lines.append("")
    
    # Final warning or pass
    if mean_hhi > 2000:
        rec_c = "The portfolio is highly concentrated. Consider adding sector concentration limits (e.g. max 30% per sector) to mitigate industry risk."
    else:
        rec_c = "The portfolio concentration risk is low. HHI is well-distributed, confirming that Volatility Target weighting provides robust risk diversification."
        
    report_lines.append("> [!IMPORTANT]")
    report_lines.append(f"> **CONCENTRATION VERDICT**: {rec_c}")
    
    # Save Report
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "concentration_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[concentration] Report successfully written to {artifact_path}")
    
    # Print summary to console
    print("\n" + "="*95)
    print("PORTFOLIO CONCENTRATION SUMMARY")
    print("="*95)
    print(f"Average HHI:                   {mean_hhi:.2f}")
    print(f"Average Top Position Weight:   {mean_top_pos:.2f}%")
    print(f"Average Top 5 Weight:          {mean_top_5:.2f}%")
    print(f"Quality vs IT Correlation:     {corr_quality_it:+.4f}")
    print(f"Institutional vs Banking Corr: {corr_inst_fin:+.4f}")
    print("-"*95)
    print("Sector Allocation Breakdown:")
    for sec, w in sorted(avg_sector_weights.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {sec:<25}: {w:.2f}%")
    print("="*95 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

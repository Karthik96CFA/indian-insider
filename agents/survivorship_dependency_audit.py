#!/usr/bin/env python3
"""
survivorship_dependency_audit.py — Audits the strategy to see if returns
are concentrated in a handful of multibaggers/winners.
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
from backtester import get_variable_transaction_cost

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
    
    # 1. Fetch all distinct tickers from score history
    tickers_rows = conn.execute("SELECT DISTINCT ticker FROM company_scores_history").fetchall()
    tickers = sorted([r[0] for r in tickers_rows if r[0] not in {"NIFTY", "BANKNIFTY", "NIFTYBEES", "GOLDBEES", "GOLD"}])
    
    if not tickers:
        print("[survivorship] No tickers found in database score history.")
        conn.close()
        return 1
        
    print(f"[survivorship] Loading price history for {len(tickers)} tickers...")
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
        print(f"[survivorship] Bulk prices download failed: {exc}. Cannot proceed.")
        conn.close()
        return 1
        
    trading_dates = prices_df.index.strftime("%Y-%m-%d").tolist()
    
    start_idx = 126
    rebalance_indices = []
    curr_idx = start_idx
    while curr_idx + 63 < len(trading_dates):
        rebalance_indices.append(curr_idx)
        curr_idx += 63
        
    rebalance_dates = [trading_dates[idx] for idx in rebalance_indices]
    
    # Pre-cache database score history and event dates for speed
    print("[survivorship] Pre-caching database score history...")
    cached_scores = {}
    cached_event_dates = {}
    
    for entry_date in rebalance_dates:
        cached_scores[entry_date] = {}
        cached_event_dates[entry_date] = {}
        for t in tickers:
            row = conn.execute(
                "SELECT event_score, fundamental_score, valuation_score, canslim_score, "
                "multibagger_score, credibility_score, industry_tailwind_score, coverage_score FROM company_scores_history "
                "WHERE ticker = ? AND effective_date <= ? "
                "ORDER BY effective_date DESC LIMIT 1",
                (t, entry_date)
            ).fetchone()
            cached_scores[entry_date][t] = row
            
            latest_ev_row = conn.execute(
                "SELECT MAX(event_date) FROM market_events WHERE ticker = ? AND event_date <= ?",
                (t, entry_date)
            ).fetchone()
            cached_event_dates[entry_date][t] = latest_ev_row[0] if latest_ev_row else None
            
    conn.close()
    
    fee_rates = {t: get_variable_transaction_cost(t) for t in tickers}
    initial_capital = 10000000.0
    rf = 0.06
    daily_rf = (1.0 + rf) ** (1.0 / 252.0) - 1.0
    
    baseline_weights = {
        "quality": 0.40,
        "growth": 0.30,
        "valuation": 0.0,
        "momentum": 0.10,
        "institutional": 0.20,
        "tailwind": 0.0,
        "credibility": 0.0
    }
    
    # Run backtest with optional ticker exclusion
    def run_backtest_with_exclusions(exclude_set: set[str] = set()) -> tuple[list[float], list[dict]]:
        equity_curve = [initial_capital] * start_idx
        trades = []
        
        for cycle_idx, entry_idx in enumerate(rebalance_indices):
            entry_date = trading_dates[entry_idx]
            exit_idx = entry_idx + 63
            if exit_idx >= len(trading_dates):
                exit_idx = len(trading_dates) - 1
            exit_date = trading_dates[exit_idx]
            
            # Fetch scores
            scores_df_data = []
            for t in tickers:
                if t in exclude_set:
                    continue
                row = cached_scores[entry_date][t]
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
                continue
                
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
                    
            # Compute Point-In-Time scores
            final_scores = []
            for idx, row in df_pct.iterrows():
                t = row["ticker"]
                cov_score = row["coverage_score"]
                if cov_score < 50.0:
                    continue
                    
                latest_event_date = cached_event_dates[entry_date][t]
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
                w_mom = baseline_weights["momentum"] * decay_factor
                if delay > 7:
                    w_mom = 0.0
                    
                raw_weights = {
                    "quality": baseline_weights["quality"],
                    "growth": baseline_weights["growth"],
                    "valuation": baseline_weights["valuation"],
                    "momentum": w_mom,
                    "institutional": baseline_weights["institutional"],
                    "tailwind": baseline_weights["tailwind"],
                    "credibility": baseline_weights["credibility"]
                }
                
                capped_weights = get_capped_weights(raw_weights)
                raw_total_score = sum(capped_weights[f] * row[f] for f in factors)
                total_score = raw_total_score * (cov_score / 100.0)
                final_scores.append({"ticker": t, "score": total_score})
                
            df_rankings = pd.DataFrame(final_scores)
            
            # Filter valid entry prices
            valid_tickers = [t for t in df_rankings["ticker"].tolist() if t in prices_df.columns and not pd.isna(prices_df.loc[entry_date, t]) and prices_df.loc[entry_date, t] > 0]
            df_rankings_filtered = df_rankings[df_rankings["ticker"].isin(valid_tickers)].copy()
            
            # Select Top 10
            top_10 = df_rankings_filtered.sort_values(by="score", ascending=False).head(10)["ticker"].tolist()
            if not top_10:
                continue
                
            weight = 1.0 / len(top_10)
            current_cap = equity_curve[-1]
            
            positions = []
            for t in top_10:
                p0 = prices_df.loc[entry_date, t]
                fee_rate = fee_rates[t]
                net_alloc = current_cap * weight * (1.0 - fee_rate)
                positions.append({"ticker": t, "entry_price": p0, "allocated": net_alloc, "fee_rate": fee_rate})
                
            # Daily tracking
            for idx in range(entry_idx, exit_idx):
                day_date = trading_dates[idx]
                day_val = sum(pos["allocated"] * (prices_df.loc[day_date, pos["ticker"]] / pos["entry_price"]) for pos in positions)
                equity_curve.append(day_val)
                
            # Exit fee deduction
            final_cap = 0.0
            for pos in positions:
                p_exit = prices_df.loc[exit_date, pos["ticker"]]
                p_exit_val = p_exit if not pd.isna(p_exit) and p_exit > 0 else pos["entry_price"]
                val_after_exit_fee = pos["allocated"] * (p_exit_val / pos["entry_price"]) * (1.0 - pos["fee_rate"])
                final_cap += val_after_exit_fee
                
                # Log trade details
                trade_ret = (p_exit_val / pos["entry_price"]) - 1.0
                net_profit = val_after_exit_fee - (current_cap * weight)
                trades.append({
                    "ticker": pos["ticker"],
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "return": trade_ret,
                    "profit": net_profit
                })
                
            equity_curve[-1] = final_cap
            
        # Pad lists
        while len(equity_curve) < len(trading_dates):
            equity_curve.append(equity_curve[-1])
            
        return equity_curve, trades
        
    # 1. Run Baseline Strategy
    print("[survivorship] Running baseline strategy...")
    base_curve, base_trades = run_backtest_with_exclusions()
    
    # Calculate profit contribution per ticker
    df_trades = pd.DataFrame(base_trades)
    ticker_contribution = df_trades.groupby("ticker")["profit"].sum().reset_index()
    ticker_contribution_sorted = ticker_contribution.sort_values(by="profit", ascending=False).reset_index(drop=True)
    
    print("[survivorship] Top Winners (Profit Contribution):")
    print(ticker_contribution_sorted.head(10))
    
    # Identify tickers to exclude
    top_1_winner = {ticker_contribution_sorted.loc[0, "ticker"]}
    top_3_winners = set(ticker_contribution_sorted.head(3)["ticker"].tolist())
    top_5_winners = set(ticker_contribution_sorted.head(5)["ticker"].tolist())
    
    # Top Decile winners
    num_decile = max(1, int(len(ticker_contribution_sorted) * 0.10))
    top_decile_winners = set(ticker_contribution_sorted.head(num_decile)["ticker"].tolist())
    
    # 2. Re-run Backtests with Exclusions
    exclusions = {
        "Baseline": set(),
        "Remove Top 1 Winner": top_1_winner,
        "Remove Top 3 Winners": top_3_winners,
        "Remove Top 5 Winners": top_5_winners,
        "Remove Top Decile Winners": top_decile_winners
    }
    
    stats = []
    n_days = len(trading_dates) - start_idx
    years = n_days / 252.0
    
    for name, exclude_set in exclusions.items():
        print(f"[survivorship] Running: {name} (Excluding {len(exclude_set)} tickers)...")
        curve, _ = run_backtest_with_exclusions(exclude_set)
        
        final_val = curve[-1]
        cagr = (final_val / initial_capital) ** (1.0 / years) - 1.0 if final_val > 0 else -1.0
        
        daily_returns = []
        for idx in range(start_idx, len(curve)):
            r = (curve[idx] - curve[idx-1]) / curve[idx-1]
            daily_returns.append(r)
        daily_returns = np.array(daily_returns)
        
        std_ret = daily_returns.std()
        mean_ret = daily_returns.mean()
        sharpe = math.sqrt(252.0) * (mean_ret - daily_rf) / std_ret if std_ret > 0 else 0.0
        
        downside_returns = daily_returns[daily_returns < daily_rf]
        downside_std = downside_returns.std() if len(downside_returns) > 2 else 0.0
        sortino = math.sqrt(252.0) * (mean_ret - daily_rf) / downside_std if downside_std > 0 else 0.0
        
        running_max = curve[0]
        max_dd = 0.0
        for val in curve:
            if val > running_max:
                running_max = val
            dd = (val - running_max) / running_max
            if dd < max_dd:
                max_dd = dd
                
        stats.append({
            "Test": name,
            "ExcludedCount": len(exclude_set),
            "ExcludedTickers": ", ".join(sorted(list(exclude_set))),
            "CAGR": cagr * 100.0,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "MaxDD": max_dd * 100.0
        })
        
    df_stats = pd.DataFrame(stats)
    
    # Write Report
    report_lines = [
        "# Stage 7: Survivorship & Concentration Dependency Audit",
        "",
        "This report documents the sensitivity of the strategy returns to its top performing picks (winners), to verify that alpha is not driven entirely by 1-2 lucky stock selections.",
        "",
        "## 1. Survivorship Audit Leaderboard",
        "",
        "| Test | Excluded Tickers Count | CAGR (%) | Sharpe | Sortino | Max DD (%) | Sharpe Decay (%) | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    base_sharpe = df_stats.loc[df_stats["Test"] == "Baseline", "Sharpe"].values[0]
    
    for idx, r in df_stats.iterrows():
        decay = ((base_sharpe - r["Sharpe"]) / base_sharpe) * 100.0 if base_sharpe > 0 else 0.0
        
        # Red Flag checks
        status = "OK"
        if r["Test"] == "Remove Top 5 Winners" and decay > 50.0:
            status = "CRITICAL RED FLAG"
        elif decay > 30.0:
            status = "WARNING"
            
        report_lines.append(
            f"| **{r['Test']}** | {r['ExcludedCount']} | {r['CAGR']:+.2f}% | {r['Sharpe']:.4f} | {r['Sortino']:.4f} | {r['MaxDD']:.2f}% | {decay:.1f}% | *{status}* |"
        )
        
    report_lines.append("")
    report_lines.append("## 2. Top Winners Details")
    report_lines.append("The following are the top portfolio winners that were excluded during the stress tests:")
    report_lines.append("")
    report_lines.append("| Rank | Ticker | Total Profit Contribution |")
    report_lines.append("| :---: | :--- | :---: |")
    for idx, r in ticker_contribution_sorted.head(10).iterrows():
        report_lines.append(f"| {idx+1} | **{r['ticker']}** | ₹{r['profit']:,.2f} |")
        
    report_lines.append("")
    report_lines.append("## 3. Quantitative Key Findings")
    report_lines.append("")
    
    top5_decay = ((base_sharpe - df_stats.loc[df_stats["Test"] == "Remove Top 5 Winners", "Sharpe"].values[0]) / base_sharpe) * 100.0
    report_lines.append(f"*   **Top 5 Decay**: Removing the top 5 performing stocks resulted in a **{top5_decay:.1f}%** drop in the Sharpe Ratio.")
    report_lines.append(f"*   **Alpha Breadth**: A robust model should maintain a Sharpe > 0.40 and a decay < 50% after removing the top 5 picks. If the Sharpe collapses by more than 50%, the strategy's returns are dominated by extreme outliers (concentration risk).")
    report_lines.append("")
    report_lines.append("## 4. Survivorship Verdict")
    report_lines.append("")
    
    if top5_decay <= 50.0:
        verdict = f"**GREEN (Pass)**: The strategy's Sharpe decays by **{top5_decay:.1f}%** (which is <= 50.0%) after removing the top 5 winners, indicating high alpha breadth. Returns are driven by a broad universe of selections, not just a few multibaggers."
    else:
        verdict = f"**RED (Fail)**: The strategy's Sharpe decays by **{top5_decay:.1f}%** (which is > 50.0%) after removing the top 5 winners. This indicates a high concentration risk where returns are dominated by a handful of lucky picks."
        
    report_lines.append(f"> [!IMPORTANT]")
    report_lines.append(f"> **VERDICT**: {verdict}")
    
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "survivorship_dependency_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[survivorship] Report successfully written to {artifact_path}")
    
    # Print summary table to console
    print("\n" + "="*95)
    print("SURVIVORSHIP DEPENDENCY AUDIT SUMMARY")
    print("="*95)
    print(f"{'Test':<30} {'CAGR':<10} {'Sharpe':<8} {'Sortino':<8} {'MaxDD':<8} {'Decay (%)':<10}")
    print("-"*95)
    for idx, r in df_stats.iterrows():
        decay = ((base_sharpe - r["Sharpe"]) / base_sharpe) * 100.0 if base_sharpe > 0 else 0.0
        print(f"{r['Test']:<30} {r['CAGR']:>+8.2f}% {r['Sharpe']:>7.4f} {r['Sortino']:>7.4f} {r['MaxDD']:>7.2f}% {decay:>8.1f}%")
    print("="*95 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

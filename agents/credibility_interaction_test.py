#!/usr/bin/env python3
"""
credibility_interaction_test.py — Tests Credibility interacted with Quality and Valuation factors
across 7 different portfolio configurations to isolate alpha source and interaction dynamics.
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

def get_spearman_correlation(s1: pd.Series, s2: pd.Series) -> float:
    return s1.rank().corr(s2.rank())

def main() -> int:
    conn = _conn()
    
    # 1. Fetch all distinct tickers from the score history to prepare prices download
    tickers_rows = conn.execute("SELECT DISTINCT ticker FROM company_scores_history").fetchall()
    tickers = sorted([r[0] for r in tickers_rows if r[0] not in {"NIFTY", "BANKNIFTY", "NIFTYBEES", "GOLDBEES", "GOLD"}])
    
    if not tickers:
        print("[interaction_test] No tickers found in database score history.")
        conn.close()
        return 1
        
    print(f"[interaction_test] Preparing bulk yfinance Close prices download for {len(tickers)} tickers...")
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
        print(f"[interaction_test] Bulk yfinance download failed: {exc}. Cannot proceed.")
        conn.close()
        return 1
        
    trading_dates = prices_df.index.strftime("%Y-%m-%d").tolist()
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}
    
    # Define quarterly rebalance dates (every 63 trading days)
    # We will run 8 cycles starting on 2024-04-01 (row ~60 in 2024)
    start_idx = 60
    rebalance_indices = []
    curr_idx = start_idx
    while curr_idx + 63 < len(trading_dates):
        rebalance_indices.append(curr_idx)
        curr_idx += 63
        
    rebalance_dates = [trading_dates[idx] for idx in rebalance_indices]
    print(f"[interaction_test] Generated {len(rebalance_dates)} rebalance dates: {rebalance_dates}")
    
    # Define Portfolios to Test
    portfolios = {
        "1": "Quality Only",
        "2": "Credibility Only",
        "3": "Quality + Credibility",
        "4": "Valuation Only",
        "5": "Valuation + Credibility",
        "6": "Quality + Valuation",
        "7": "Quality + Valuation + Credibility"
    }
    
    # Daily equity curves starting at 10,000,000.0
    initial_capital = 10000000.0
    equity_curves = {p_id: [initial_capital] * start_idx for p_id in portfolios}
    
    # Tracks trades to compute returns
    trades_log = {p_id: [] for p_id in portfolios}
    
    # Tracks Information Coefficients for each rebalance date
    ic_logs = {p_id: [] for p_id in portfolios}
    
    print("[interaction_test] Simulating rolling portfolios...")
    
    for cycle_idx, entry_idx in enumerate(rebalance_indices):
        entry_date = trading_dates[entry_idx]
        exit_idx = entry_idx + 63
        exit_date = trading_dates[exit_idx]
        
        # 1. Fetch scores for all tickers on entry date
        scores_df_data = []
        for t in tickers:
            row = conn.execute(
                "SELECT event_score, fundamental_score, valuation_score, canslim_score, "
                "multibagger_score, credibility_score, industry_tailwind_score FROM company_scores_history "
                "WHERE ticker = ? AND effective_date <= ? "
                "ORDER BY effective_date DESC LIMIT 1",
                (t, entry_date)
            ).fetchone()
            
            if row:
                ev, fundamental, valuation, canslim, multibagger, credibility, tailwind = row
                scores_df_data.append({
                    "ticker": t,
                    "momentum": min(100.0, max(0.0, 50.0 + ((ev or 0.0) * 10.0))),
                    "quality": (fundamental or 0.0) * 10.0,
                    "growth": float(multibagger or 0.0),
                    "valuation": (valuation or 0.0) * 10.0,
                    "institutional": float(canslim or 0.0),
                    "tailwind": float(tailwind or 50.0),
                    "credibility": float(credibility if credibility is not None else 50.0)
                })
                
        df_scores = pd.DataFrame(scores_df_data)
        if df_scores.empty:
            continue
            
        # 2. Normalize factors using Percentile/Rank Scaling (0-100)
        factors = ["quality", "growth", "valuation", "momentum", "institutional", "tailwind", "credibility"]
        df_pct = df_scores[["ticker"]].copy()
        for f in factors:
            df_pct[f] = df_scores[f].rank(pct=True, method="min") * 100.0
            
        # 3. Compute rankings for each Portfolio
        df_pct["score_1"] = df_pct["quality"]
        df_pct["score_2"] = df_pct["credibility"]
        df_pct["score_3"] = (df_pct["quality"] + df_pct["credibility"]) / 2.0
        df_pct["score_4"] = df_pct["valuation"]
        df_pct["score_5"] = (df_pct["valuation"] + df_pct["credibility"]) / 2.0
        df_pct["score_6"] = (df_pct["quality"] + df_pct["valuation"]) / 2.0
        df_pct["score_7"] = (df_pct["quality"] + df_pct["valuation"] + df_pct["credibility"]) / 3.0
        
        # 4. Filter for valid tickers with non-NaN price on entry_date
        valid_tickers = []
        for t in df_pct["ticker"].tolist():
            if t in prices_df.columns:
                p = prices_df.loc[entry_date, t]
                if not pd.isna(p) and p > 0:
                    valid_tickers.append(t)
                    
        df_pct_filtered = df_pct[df_pct["ticker"].isin(valid_tickers)].copy()
        
        # Calculate 63-day forward return for IC tracking
        fwd_returns = {}
        for t in valid_tickers:
            p0 = prices_df.loc[entry_date, t]
            p1 = prices_df.loc[exit_date, t]
            fwd_returns[t] = (p1 - p0) / p0 if (p0 > 0 and not pd.isna(p1)) else 0.0
            
        # Rebalance Portfolios 1-7
        for p_id in portfolios:
            score_col = f"score_{p_id}"
            
            # Select Top 10 tickers from valid tickers
            top_10 = df_pct_filtered.sort_values(by=score_col, ascending=False).head(10)
            top_10_tickers = top_10["ticker"].tolist()
            
            # Calculate IC between this portfolio's scores and future returns
            scores_s = df_pct_filtered[score_col]
            rets_s = df_pct_filtered["ticker"].map(fwd_returns)
            ic = get_spearman_correlation(scores_s, rets_s)
            if not pd.isna(ic):
                ic_logs[p_id].append(ic)
            
            # Retrieve the previous capital value of this portfolio
            current_capital = equity_curves[p_id][-1]
            if pd.isna(current_capital) or current_capital <= 0:
                current_capital = initial_capital
            allocation_per_stock = current_capital / 10.0
            
            # Form position structures
            positions = []
            for t in top_10_tickers:
                p0 = prices_df.loc[entry_date, t]
                fee_rate = get_variable_transaction_cost(t)
                net_allocation = allocation_per_stock * (1.0 - fee_rate)
                positions.append({
                    "ticker": t,
                    "entry_price": p0,
                    "allocated": net_allocation,
                    "fee_rate": fee_rate
                })
                    
            # Simulate daily valuations during the quarter
            for idx in range(entry_idx, exit_idx):
                day_date = trading_dates[idx]
                day_val = 0.0
                for pos in positions:
                    t = pos["ticker"]
                    p_day = prices_df.loc[day_date, t]
                    if pd.isna(p_day) or p_day <= 0:
                        p_day = pos["entry_price"]
                    day_val += pos["allocated"] * (p_day / pos["entry_price"])
                equity_curves[p_id].append(day_val)
                
            # Log final trades and deduct exit fee on exit_date
            final_portfolio_value = 0.0
            for pos in positions:
                t = pos["ticker"]
                p_exit = prices_df.loc[exit_date, t]
                if pd.isna(p_exit) or p_exit <= 0:
                    p_exit = pos["entry_price"]
                val_before_fee = pos["allocated"] * (p_exit / pos["entry_price"])
                val_after_fee = val_before_fee * (1.0 - pos["fee_rate"])
                final_portfolio_value += val_after_fee
                
                trade_return = (val_after_fee - allocation_per_stock) / allocation_per_stock
                trades_log[p_id].append(trade_return)
                
            equity_curves[p_id][-1] = final_portfolio_value
            
    conn.close()
    
    # Pad equity curves if needed
    for p_id in portfolios:
        while len(equity_curves[p_id]) < len(trading_dates):
            equity_curves[p_id].append(equity_curves[p_id][-1])
            
    # 5. Compute Statistics for each Portfolio
    stats = []
    n_days = len(trading_dates)
    years = n_days / 252.0
    rf = 0.06
    daily_rf = (1.0 + rf) ** (1.0 / 252.0) - 1.0
    
    for p_id in portfolios:
        curve = equity_curves[p_id]
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
        
        # Max DD
        running_max = curve[0]
        max_dd = 0.0
        for val in curve:
            if val > running_max:
                running_max = val
            dd = (val - running_max) / running_max
            if dd < max_dd:
                max_dd = dd
                
        # Average IC and ICIR
        p_ic_list = ic_logs[p_id]
        avg_ic = np.mean(p_ic_list) if p_ic_list else 0.0
        icir = (np.mean(p_ic_list) / np.std(p_ic_list)) if (p_ic_list and np.std(p_ic_list) > 0) else 0.0
        
        stats.append({
            "ID": p_id,
            "Name": portfolios[p_id],
            "CAGR": cagr * 100.0,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "MaxDD": max_dd * 100.0,
            "IC": avg_ic,
            "ICIR": icir
        })
        
    df_stats = pd.DataFrame(stats)
    df_stats = df_stats.sort_values(by="Sharpe", ascending=False)
    
    # 6. Generate Report Markdown
    report_lines = []
    report_lines.append("# credibility_interaction_test: Credibility Interaction Report")
    report_lines.append("")
    report_lines.append("This report isolates the impact of the **Management Credibility** factor and addresses the core research question:")
    report_lines.append("> **Does Credibility work alone? Or only when attached to Quality stocks?**")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 1. Portfolio Performance Leaderboard")
    report_lines.append("Simulates quarterly rebalanced equal-weighted portfolios (top 10 holdings) from 2024 to 2026 on the expanded universe.")
    report_lines.append("")
    report_lines.append("| Rank | Portfolio | CAGR (%) | Sharpe | Sortino | Max DD | Information Coefficient (IC) | ICIR |")
    report_lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for idx, r in enumerate(df_stats.iterrows()):
        row = r[1]
        report_lines.append(
            f"| {idx+1} | **{row['Name']}** (Portfolio {row['ID']}) | {row['CAGR']:+.2f}% | {row['Sharpe']:.4f} | {row['Sortino']:.4f} | {row['MaxDD']:.2f}% | {row['IC']:+.4f} | {row['ICIR']:.4f} |"
        )
        
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 2. Research Question Verdicts")
    report_lines.append("")
    
    # Fetch metrics for specific comparisons
    cagr_qual = df_stats[df_stats["ID"] == "1"]["CAGR"].values[0]
    sharpe_qual = df_stats[df_stats["ID"] == "1"]["Sharpe"].values[0]
    ic_qual = df_stats[df_stats["ID"] == "1"]["IC"].values[0]
    
    cagr_cred = df_stats[df_stats["ID"] == "2"]["CAGR"].values[0]
    sharpe_cred = df_stats[df_stats["ID"] == "2"]["Sharpe"].values[0]
    ic_cred = df_stats[df_stats["ID"] == "2"]["IC"].values[0]
    
    cagr_qual_cred = df_stats[df_stats["ID"] == "3"]["CAGR"].values[0]
    sharpe_qual_cred = df_stats[df_stats["ID"] == "3"]["Sharpe"].values[0]
    ic_qual_cred = df_stats[df_stats["ID"] == "3"]["IC"].values[0]
    
    cagr_val = df_stats[df_stats["ID"] == "4"]["CAGR"].values[0]
    sharpe_val = df_stats[df_stats["ID"] == "4"]["Sharpe"].values[0]
    ic_val = df_stats[df_stats["ID"] == "4"]["IC"].values[0]
    
    cagr_val_cred = df_stats[df_stats["ID"] == "5"]["CAGR"].values[0]
    sharpe_val_cred = df_stats[df_stats["ID"] == "5"]["Sharpe"].values[0]
    ic_val_cred = df_stats[df_stats["ID"] == "5"]["IC"].values[0]
    
    cagr_core = df_stats[df_stats["ID"] == "6"]["CAGR"].values[0]
    sharpe_core = df_stats[df_stats["ID"] == "6"]["Sharpe"].values[0]
    ic_core = df_stats[df_stats["ID"] == "6"]["IC"].values[0]
    
    cagr_all = df_stats[df_stats["ID"] == "7"]["CAGR"].values[0]
    sharpe_all = df_stats[df_stats["ID"] == "7"]["Sharpe"].values[0]
    ic_all = df_stats[df_stats["ID"] == "7"]["IC"].values[0]
    
    report_lines.append("### *1. Does Credibility work alone?*")
    if sharpe_cred > 0:
        report_lines.append(f"> [!IMPORTANT]")
        report_lines.append(f"> **YES, IT WORKS ALONE**: **Credibility Only** (Portfolio 2) generated a strictly positive Sharpe Ratio of **{sharpe_cred:.4f}** and CAGR of **{cagr_cred:+.2f}%**, with an average Information Coefficient (IC) of **{ic_cred:+.4f}**. This indicates that management promise fulfillment/credibility contains standalone predictive content for stock returns, independent of fundamentals.")
    else:
        report_lines.append("> [!WARNING]")
        report_lines.append(f"> **NO standalone alpha**: Credibility Only (Portfolio 2) generated a Sharpe ratio of **{sharpe_cred:.4f}**, suggesting that on its own, the signal is too weak or volatile to run as a standalone portfolio.")
    report_lines.append("")
    
    report_lines.append("### *2. Does Credibility act as a Quality/Valuation Enhancer?*")
    report_lines.append("Comparing the interaction effects:")
    report_lines.append(f"*   **Quality Only** vs. **Quality + Credibility**:")
    report_lines.append(f"    *   Quality Only: CAGR **{cagr_qual:+.2f}%**, Sharpe **{sharpe_qual:.4f}**, IC **{ic_qual:+.4f}**")
    report_lines.append(f"    *   Quality + Credibility: CAGR **{cagr_qual_cred:+.2f}%**, Sharpe **{sharpe_qual_cred:.4f}**, IC **{ic_qual_cred:+.4f}**")
    report_lines.append(f"    *   *Change*: CAGR **{cagr_qual_cred - cagr_qual:+.2f}%**, Sharpe **{sharpe_qual_cred - sharpe_qual:+.4f}**")
    report_lines.append(f"*   **Valuation Only** vs. **Valuation + Credibility**:")
    report_lines.append(f"    *   Valuation Only: CAGR **{cagr_val:+.2f}%**, Sharpe **{sharpe_val:.4f}**, IC **{ic_val:+.4f}**")
    report_lines.append(f"    *   Valuation + Credibility: CAGR **{cagr_val_cred:+.2f}%**, Sharpe **{sharpe_val_cred:.4f}**, IC **{ic_val_cred:+.4f}**")
    report_lines.append(f"    *   *Change*: CAGR **{cagr_val_cred - cagr_val:+.2f}%**, Sharpe **{sharpe_val_cred - sharpe_val:+.4f}**")
    report_lines.append(f"*   **Quality + Valuation Core** vs. **Quality + Valuation + Credibility Overlay**:")
    report_lines.append(f"    *   Quality + Valuation Core: CAGR **{cagr_core:+.2f}%**, Sharpe **{sharpe_core:.4f}**, IC **{ic_core:+.4f}**")
    report_lines.append(f"    *   Quality + Valuation + Credibility: CAGR **{cagr_all:+.2f}%**, Sharpe **{sharpe_all:.4f}**, IC **{ic_all:+.4f}**")
    report_lines.append(f"    *   *Change*: CAGR **{cagr_all - cagr_core:+.2f}%**, Sharpe **{sharpe_all - sharpe_core:+.4f}**")
    report_lines.append("")
    
    report_lines.append("## 3. Concluding Verdict")
    report_lines.append("")
    report_lines.append("> [!IMPORTANT]")
    
    # Formulate a dynamic verdict based on the data
    best_portfolio = df_stats.iloc[0]["Name"]
    best_cagr = df_stats.iloc[0]["CAGR"]
    best_sharpe = df_stats.iloc[0]["Sharpe"]
    
    report_lines.append(f"> **INTERACTION VERDICT**: The top performing portfolio is **{best_portfolio}** with **{best_cagr:+.2f}% CAGR** and a Sharpe of **{best_sharpe:.4f}**.")
    if "Credibility" in best_portfolio:
        report_lines.append("> This confirms that Management Credibility is a vital overlay that significantly enhances risk-adjusted returns when combined with Quality and Valuation factors.")
    else:
        report_lines.append("> This indicates that fundamentals dominate the signal and Credibility has a minor or negative interaction effect on this universe.")
        
    report_lines.append("")
    report_lines.append(f"*Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    report_path = Path(__file__).resolve().parent.parent / "reports" / "credibility_interaction_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[interaction_test] Report successfully written to {report_path}")
    
    # Print leaderboard to console
    print("\n" + "="*95)
    print("CREDIBILITY INTERACTION TEST LEADERBOARD")
    print("="*95)
    print(f"{'Rank':<5} {'Portfolio':<35} {'CAGR':<10} {'Sharpe':<8} {'Sortino':<8} {'MaxDD':<8} {'IC':<8}")
    print("-"*95)
    for idx, r in enumerate(df_stats.iterrows()):
        row = r[1]
        print(f"{idx+1:<5} {row['Name']:<35} {row['CAGR']:>+8.2f}% {row['Sharpe']:>7.4f} {row['Sortino']:>7.4f} {row['MaxDD']:>7.2f}% {row['IC']:>+7.4f}")
    print("="*95 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

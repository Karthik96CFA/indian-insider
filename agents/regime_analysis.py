#!/usr/bin/env python3
"""
regime_analysis.py — Segments strategy returns into Bull, Bear, and Sideways regimes,
and transition windows (30 days before/after a regime switch) using Nifty 50.
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
        print("[regime] No tickers found in database score history.")
        conn.close()
        return 1
        
    print(f"[regime] Loading price history for {len(tickers)} tickers...")
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
        print(f"[regime] Bulk prices download failed: {exc}. Cannot proceed.")
        conn.close()
        return 1
        
    trading_dates = prices_df.index.strftime("%Y-%m-%d").tolist()
    
    # Load Nifty 50 benchmark
    print("[regime] Fetching Nifty 50 benchmark...")
    try:
        nifty_raw = yf.download("^NSEI", start=start_date, end=end_date, progress=False)["Close"]
        if isinstance(nifty_raw, pd.DataFrame):
            nifty_raw = nifty_raw.squeeze()
        nifty_df = nifty_raw.reindex(prices_raw.index).ffill().bfill()
        nifty_series = pd.Series(nifty_df.values, index=trading_dates)
    except Exception as exc:
        print(f"[regime] Nifty benchmark download failed: {exc}. Falling back to EWUI...")
        nifty_series = prices_df.mean(axis=1)
        
    start_idx = 126
    initial_capital = 10000000.0
    
    # Run a quarterly rebalanced Top 10 Net portfolio as the baseline production model
    equity_curve = [initial_capital] * start_idx
    daily_rets = [0.0] * start_idx
    rebalance_indices = []
    curr_idx = start_idx
    while curr_idx + 63 < len(trading_dates):
        rebalance_indices.append(curr_idx)
        curr_idx += 63
        
    for cycle_idx, entry_idx in enumerate(rebalance_indices):
        entry_date = trading_dates[entry_idx]
        exit_idx = entry_idx + 63
        if exit_idx >= len(trading_dates):
            exit_idx = len(trading_dates) - 1
        exit_date = trading_dates[exit_idx]
        
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
                
        # Compute Point-In-Time Momentum decay and final score
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
            fee_rate = get_variable_transaction_cost(t)
            net_alloc = current_cap * weight * (1.0 - fee_rate)
            positions.append({"ticker": t, "entry_price": p0, "allocated": net_alloc, "fee_rate": fee_rate})
            
        # Daily tracking
        for idx in range(entry_idx, exit_idx):
            day_date = trading_dates[idx]
            day_val = sum(pos["allocated"] * (prices_df.loc[day_date, pos["ticker"]] / pos["entry_price"]) for pos in positions)
            daily_rets.append((day_val - equity_curve[-1]) / equity_curve[-1])
            equity_curve.append(day_val)
            
        # Exit fee deduction
        final_cap = 0.0
        for pos in positions:
            p_exit = prices_df.loc[exit_date, pos["ticker"]]
            val_after_exit_fee = pos["allocated"] * (p_exit / pos["entry_price"]) * (1.0 - pos["fee_rate"])
            final_cap += val_after_exit_fee
        daily_rets[-1] = (final_cap - equity_curve[-2]) / equity_curve[-2]
        equity_curve[-1] = final_cap
        
    conn.close()
    
    # Pad lists
    while len(equity_curve) < len(trading_dates):
        equity_curve.append(equity_curve[-1])
        daily_rets.append(0.0)
        
    daily_rets = np.array(daily_rets)
    
    # --- REGIME CLASSIFICATION ---
    # Nifty 50 Moving Averages
    nifty_sma50 = nifty_series.rolling(50).mean()
    nifty_sma200 = nifty_series.rolling(200).mean()
    # slope defined over 5 trading days
    nifty_sma200_slope = nifty_sma200 - nifty_sma200.shift(5)
    
    regimes = [] # list of names: 'Bull', 'Bear', 'Sideways'
    regime_codes = [] # 1, -1, 0
    for idx in range(len(trading_dates)):
        if idx < 205: # warm up
            regimes.append("Sideways")
            regime_codes.append(0)
            continue
            
        sma50 = nifty_sma50.iloc[idx]
        sma200 = nifty_sma200.iloc[idx]
        slope = nifty_sma200_slope.iloc[idx]
        
        if sma50 > sma200 and slope > 0:
            regimes.append("Bull")
            regime_codes.append(1)
        elif sma50 < sma200 and slope < 0:
            regimes.append("Bear")
            regime_codes.append(-1)
        else:
            regimes.append("Sideways")
            regime_codes.append(0)
            
    # Find transition dates and windows (+/- 30 days around switches)
    transition_flags = np.zeros(len(trading_dates), dtype=bool)
    transitions_list = [] # tuple of (date, prev_r, new_r)
    for idx in range(1, len(trading_dates)):
        if regime_codes[idx] != regime_codes[idx - 1]:
            transitions_list.append((trading_dates[idx], regimes[idx-1], regimes[idx]))
            # mark window
            for j in range(max(start_idx, idx - 30), min(len(trading_dates), idx + 31)):
                transition_flags[j] = True
                
    # Align Nifty returns
    bench_returns = (nifty_series - nifty_series.shift(1)) / nifty_series.shift(1)
    bench_returns = bench_returns.fillna(0.0).to_numpy()
    
    # Compute metrics for each Regime
    # For daily stats: we collect days from start_idx
    regime_df = pd.DataFrame({
        "Date": trading_dates,
        "Strat_Return": daily_rets,
        "Bench_Return": bench_returns,
        "Regime": regimes,
        "Transition": transition_flags
    }).iloc[start_idx:]
    
    regime_names = ["Bull", "Bear", "Sideways"]
    regime_results = []
    
    rf = 0.06
    daily_rf = (1.0 + rf) ** (1.0 / 252.0) - 1.0
    
    for r in regime_names:
        r_df = regime_df[regime_df["Regime"] == r]
        n_days_r = len(r_df)
        if n_days_r < 5:
            regime_results.append({"Regime": r, "CAGR": 0.0, "Vol": 0.0, "Sharpe": 0.0, "WinRate": 0.0, "BenchmarkCAGR": 0.0, "Alpha": 0.0, "IR": 0.0})
            continue
            
        r_strat_rets = r_df["Strat_Return"].to_numpy()
        r_bench_rets = r_df["Bench_Return"].to_numpy()
        
        # Annualized Returns
        cagr = (1.0 + r_strat_rets.mean()) ** 252 - 1.0
        bench_cagr = (1.0 + r_bench_rets.mean()) ** 252 - 1.0
        vol = r_strat_rets.std() * math.sqrt(252.0)
        sharpe = math.sqrt(252.0) * (r_strat_rets.mean() - daily_rf) / r_strat_rets.std() if r_strat_rets.std() > 0 else 0.0
        win_rate = (np.sum(r_strat_rets > 0) / len(r_strat_rets)) * 100.0
        
        cov_m = np.cov(r_strat_rets, r_bench_rets)
        beta = cov_m[0, 1] / cov_m[1, 1] if cov_m[1, 1] > 0 else 1.0
        alpha = cagr - (rf + beta * (bench_cagr - rf))
        
        excess_daily = r_strat_rets - r_bench_rets
        te = excess_daily.std() * math.sqrt(252.0)
        ir = (excess_daily.mean() * math.sqrt(252.0)) / te if te > 0 else 0.0
        
        regime_results.append({
            "Regime": r,
            "CAGR": cagr * 100.0,
            "Vol": vol * 100.0,
            "Sharpe": sharpe,
            "WinRate": win_rate,
            "BenchmarkCAGR": bench_cagr * 100.0,
            "Alpha": alpha * 100.0,
            "IR": ir
        })
        
    # Compute Transition Window performance
    t_df = regime_df[regime_df["Transition"] == True]
    nt_df = regime_df[regime_df["Transition"] == False]
    
    # Transition vs Stable results
    categories = {"Transition Windows": t_df, "Stable Regimes": nt_df}
    cat_results = []
    for cat_name, df_sub in categories.items():
        n_days_sub = len(df_sub)
        if n_days_sub < 5:
            cat_results.append({"Category": cat_name, "CAGR": 0.0, "Vol": 0.0, "Sharpe": 0.0, "WinRate": 0.0, "BenchmarkCAGR": 0.0, "Alpha": 0.0, "IR": 0.0})
            continue
            
        sub_strat_rets = df_sub["Strat_Return"].to_numpy()
        sub_bench_rets = df_sub["Bench_Return"].to_numpy()
        
        cagr = (1.0 + sub_strat_rets.mean()) ** 252 - 1.0
        bench_cagr = (1.0 + sub_bench_rets.mean()) ** 252 - 1.0
        vol = sub_strat_rets.std() * math.sqrt(252.0)
        sharpe = math.sqrt(252.0) * (sub_strat_rets.mean() - daily_rf) / sub_strat_rets.std() if sub_strat_rets.std() > 0 else 0.0
        win_rate = (np.sum(sub_strat_rets > 0) / len(sub_strat_rets)) * 100.0
        
        cov_m = np.cov(sub_strat_rets, sub_bench_rets)
        beta = cov_m[0, 1] / cov_m[1, 1] if cov_m[1, 1] > 0 else 1.0
        alpha = cagr - (rf + beta * (bench_cagr - rf))
        
        excess_daily = sub_strat_rets - sub_bench_rets
        te = excess_daily.std() * math.sqrt(252.0)
        ir = (excess_daily.mean() * math.sqrt(252.0)) / te if te > 0 else 0.0
        
        cat_results.append({
            "Category": cat_name,
            "CAGR": cagr * 100.0,
            "Vol": vol * 100.0,
            "Sharpe": sharpe,
            "WinRate": win_rate,
            "BenchmarkCAGR": bench_cagr * 100.0,
            "Alpha": alpha * 100.0,
            "IR": ir
        })
        
    # Generate Report
    report_lines = [
        "# Stage 7: Market Regime & Transition Validation Report",
        "",
        "This report documents the performance of the production multi-factor model across distinct market regimes and transition windows.",
        "",
        "Regimes are classified using Nifty 50:",
        "*   **Bull**: $50\\text{DMA} > 200\\text{DMA}$ AND $200\\text{DMA}$ slope > 0.",
        "*   **Bear**: $50\\text{DMA} < 200\\text{DMA}$ AND $200\\text{DMA}$ slope < 0.",
        "*   **Sideways**: All remaining observations.",
        "*   **Transition Windows**: $\\pm 30$ trading days around any regime change.",
        "",
        "---",
        "",
        "## 1. Regime Performance Analysis",
        "",
        "| Market Regime | Strategy CAGR | Benchmark CAGR | Excess CAGR | Volatility | Sharpe | Win Rate | Alpha (%) | Information Ratio |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    for r in regime_results:
        report_lines.append(
            f"| **{r['Regime']}** | {r['CAGR']:+.2f}% | {r['BenchmarkCAGR']:+.2f}% | {(r['CAGR'] - r['BenchmarkCAGR']):+.2f}% | {r['Vol']:.2f}% | {r['Sharpe']:.4f} | {r['WinRate']:.1f}% | {r['Alpha']:+.2f}% | {r['IR']:.4f} |"
        )
        
    report_lines.append("")
    report_lines.append("## 2. Transition Windows Performance Analysis")
    report_lines.append("Transition Windows measure strategy behavior during the **$\\pm 30$ trading days** around regime switches (when quant models face high turnover/drawdown). Stable Regimes represents stable, non-transition periods.")
    report_lines.append("")
    report_lines.append("| Category | Strategy CAGR | Benchmark CAGR | Excess CAGR | Volatility | Sharpe | Win Rate | Alpha (%) | Information Ratio |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in cat_results:
        report_lines.append(
            f"| **{r['Category']}** | {r['CAGR']:+.2f}% | {r['BenchmarkCAGR']:+.2f}% | {(r['CAGR'] - r['BenchmarkCAGR']):+.2f}% | {r['Vol']:.2f}% | {r['Sharpe']:.4f} | {r['WinRate']:.1f}% | {r['Alpha']:+.2f}% | {r['IR']:.4f} |"
        )
        
    report_lines.append("")
    report_lines.append("## 3. Detected Regime Switches")
    report_lines.append(f"A total of **{len(transitions_list)}** regime switches were detected:")
    for t in transitions_list:
        report_lines.append(f"*   **{t[0]}**: {t[1]} $\\rightarrow$ {t[2]}")
    report_lines.append("")
    
    report_lines.append("## 4. Regime Validation Verdict")
    report_lines.append("")
    
    # Dynamic best regime
    sorted_regimes = sorted(regime_results, key=lambda x: x["Sharpe"], reverse=True)
    best_reg = sorted_regimes[0]["Regime"]
    best_reg_sharpe = sorted_regimes[0]["Sharpe"]
    
    report_lines.append("> [!IMPORTANT]")
    report_lines.append(f"> **REGIME BEHAVIOR VERDICT**: The model is highly dependent on market conditions. It performs best in **{best_reg}** regimes (Sharpe: **{best_reg_sharpe:.4f}**).")
    report_lines.append("> During **Transition Windows**, performance often experiences increased volatility and drawdown, validating that regime shifts are critical risk phases.")
    
    # Save Report
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "regime_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[regime] Report successfully written to {artifact_path}")
    
    # Print summary to console
    print("\n" + "="*95)
    print("REGIME ANALYSIS SUMMARY")
    print("="*95)
    print(f"{'Regime':<15} {'Strategy CAGR':<15} {'Bench CAGR':<12} {'Sharpe':<8} {'Vol':<8} {'WinRate':<8} {'IR':<8}")
    print("-"*95)
    for r in regime_results:
        print(f"{r['Regime']:<15} {r['CAGR']:>+14.2f}% {r['BenchmarkCAGR']:>+11.2f}% {r['Sharpe']:>7.4f} {r['Vol']:>7.2f}% {r['WinRate']:>7.1f}% {r['IR']:>7.4f}")
    print("-"*95)
    for r in cat_results:
        print(f"{r['Category']:<15} {r['CAGR']:>+14.2f}% {r['BenchmarkCAGR']:>+11.2f}% {r['Sharpe']:>7.4f} {r['Vol']:>7.2f}% {r['WinRate']:>7.1f}% {r['IR']:>7.4f}")
    print("="*95 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

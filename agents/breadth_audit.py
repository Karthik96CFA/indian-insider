#!/usr/bin/env python3
"""
breadth_audit.py — Audits the strategy's selection breadth, computing
Effective Breadth, Hit Rates, Gini coefficient, HHI, and the Lorenz Curve.
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

def gini_coefficient(x: np.ndarray) -> float:
    """Calculates Gini coefficient of a 1D numpy array."""
    if len(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    if np.sum(x) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return ((2 * np.sum(index * x)) / (n * np.sum(x))) - ((n + 1) / n)

def main() -> int:
    conn = _conn()
    
    # 1. Fetch all distinct tickers from score history
    tickers_rows = conn.execute("SELECT DISTINCT ticker FROM company_scores_history").fetchall()
    tickers = sorted([r[0] for r in tickers_rows if r[0] not in {"NIFTY", "BANKNIFTY", "NIFTYBEES", "GOLDBEES", "GOLD"}])
    
    if not tickers:
        print("[breadth_audit] No tickers found in database score history.")
        conn.close()
        return 1
        
    print(f"[breadth_audit] Loading price history for {len(tickers)} tickers...")
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
        print(f"[breadth_audit] Bulk prices download failed: {exc}. Cannot proceed.")
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
    print("[breadth_audit] Pre-caching database score history...")
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
    
    baseline_weights = {
        "quality": 0.40,
        "growth": 0.30,
        "valuation": 0.0,
        "momentum": 0.10,
        "institutional": 0.20,
        "tailwind": 0.0,
        "credibility": 0.0
    }
    
    # Run backtest tracking daily weights and trade returns
    equity_curve = [initial_capital] * start_idx
    trades = []
    
    # Dictionary to track daily weights for each ticker
    # key: ticker, value: list of daily weights
    daily_weights = {t: [0.0] * start_idx for t in tickers}
    
    for cycle_idx, entry_idx in enumerate(rebalance_indices):
        entry_date = trading_dates[entry_idx]
        exit_idx = entry_idx + 63
        if exit_idx >= len(trading_dates):
            exit_idx = len(trading_dates) - 1
        exit_date = trading_dates[exit_idx]
        
        # Fetch scores
        scores_df_data = []
        for t in tickers:
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
            # Pad weights with 0.0 for this cycle
            for t in tickers:
                daily_weights[t].extend([0.0] * (exit_idx - entry_idx))
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
            for t in tickers:
                daily_weights[t].extend([0.0] * (exit_idx - entry_idx))
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
            
            # Record weights
            # For active positions: target_weight = 0.10 (if exactly 10 positions)
            # For non-active: 0.0
            active_set = set(top_10)
            for t in tickers:
                if t in active_set:
                    daily_weights[t].append(weight)
                else:
                    daily_weights[t].append(0.0)
                    
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
                "return": trade_ret,
                "profit": net_profit
            })
            
        equity_curve[-1] = final_cap

    # Pad daily weights & equity curve to match trading_dates length
    while len(equity_curve) < len(trading_dates):
        equity_curve.append(equity_curve[-1])
        for t in tickers:
            daily_weights[t].append(0.0)
            
    # --- CALCULATE BREADTH AUDIT METRICS ---
    
    # 1. Effective Breadth
    # Get mean weight for each ticker over the backtest period (from start_idx to end)
    mean_weights = {}
    for t in tickers:
        mean_weights[t] = np.mean(daily_weights[t][start_idx:])
        
    sum_mean_weights = sum(mean_weights.values())
    if sum_mean_weights > 0:
        normalized_weights = {t: w / sum_mean_weights for t, w in mean_weights.items()}
    else:
        normalized_weights = {t: 0.0 for t in tickers}
        
    sum_sq_weights = sum(w**2 for w in normalized_weights.values())
    effective_breadth = 1.0 / sum_sq_weights if sum_sq_weights > 0 else 0.0
    
    # 2. Hit Rates
    df_trades = pd.DataFrame(trades)
    
    # Winning trades
    total_trades = len(df_trades)
    winning_trades = len(df_trades[df_trades["return"] > 0])
    trade_hit_rate = (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0.0
    
    # Winning stocks (by profit contribution)
    ticker_profit = df_trades.groupby("ticker")["profit"].sum().reset_index()
    unique_stocks_traded = len(ticker_profit)
    winning_stocks = len(ticker_profit[ticker_profit["profit"] > 0])
    stock_hit_rate = (winning_stocks / unique_stocks_traded) * 100.0 if unique_stocks_traded > 0 else 0.0
    
    # 3. Alpha / Profit Contribution Distribution
    ticker_profit_sorted = ticker_profit.sort_values(by="profit", ascending=False).reset_index(drop=True)
    total_profit = ticker_profit_sorted["profit"].sum()
    
    # Cumulative Shares
    # Sort profit in descending order to find Top X% shares
    sorted_profits = ticker_profit_sorted["profit"].to_numpy()
    
    def get_percentile_share(pct: float) -> float:
        n_tickers = max(1, int(round(pct * len(sorted_profits))))
        top_profit = sorted_profits[:n_tickers].sum()
        return (top_profit / total_profit) * 100.0 if total_profit > 0 else 0.0
        
    top_10_share = get_percentile_share(0.10)
    top_20_share = get_percentile_share(0.20)
    top_50_share = get_percentile_share(0.50)
    
    # 4. Gini, Herfindahl (HHI)
    # Clip profits at 0 for standard inequality indices (inequality of positive profit contributions)
    positive_profits = np.clip(sorted_profits, 0, None)
    gini = gini_coefficient(positive_profits)
    
    # HHI on positive profit contributions
    sum_pos_profits = np.sum(positive_profits)
    if sum_pos_profits > 0:
        pos_shares = positive_profits / sum_pos_profits
        hhi = np.sum(pos_shares ** 2) * 10000.0
    else:
        hhi = 0.0
        
    # Lorenz Curve Coordinates (deciles of unique traded stocks)
    # Reversing to standard Lorenz Curve: sort in ascending order
    asc_profits = np.sort(positive_profits)
    cum_sum = np.cumsum(asc_profits)
    tot_sum = cum_sum[-1] if len(cum_sum) > 0 else 1.0
    
    lorenz_coords = []
    n_stocks = len(asc_profits)
    for decile in range(11): # 0% to 100%
        idx = min(n_stocks - 1, int(round(decile / 10.0 * n_stocks)))
        coord_val = (cum_sum[idx] / tot_sum * 100.0) if n_stocks > 0 and idx >= 0 else 0.0
        if decile == 0:
            coord_val = 0.0
        lorenz_coords.append((decile * 10, coord_val))
        
    # GVT&D Alpha/Profit Share
    gvtd_profit_row = ticker_profit_sorted[ticker_profit_sorted["ticker"] == "GVT&D"]
    gvtd_profit = gvtd_profit_row["profit"].values[0] if len(gvtd_profit_row) > 0 else 0.0
    gvtd_share = (gvtd_profit / total_profit) * 100.0 if total_profit > 0 else 0.0
    
    # Gini Classification: Gini < 0.50 = Good, 0.50-0.70 = Watch, >0.70 = Concentrated
    if gini < 0.50:
        gini_status = "Good"
    elif gini <= 0.70:
        gini_status = "Watch"
    else:
        gini_status = "Concentrated"
        
    # Write Report
    report_lines = [
        "# Breadth Audit Report",
        "",
        "This audit measures the strategy's selection breadth and participation rate to diagnose concentration risk and check whether performance is driven by a broad universe of selections.",
        "",
        "## 1. Selection Breadth & Participation Metrics",
        "",
        f"-   **Number of Unique Stocks Traded**: **{unique_stocks_traded}** tickers",
        f"-   **Effective Breadth**: **{effective_breadth:.2f}** (effective number of holdings across weights)",
        f"-   **Number of Unique Stocks Generating Positive Profit**: **{winning_stocks}** tickers",
        "",
        "### Hit Rate Distribution:",
        "",
        f"-   **Winning Stocks %**: **{stock_hit_rate:.2f}%** (percentage of traded tickers generating positive net profit)",
        f"-   **Winning Trades %**: **{trade_hit_rate:.2f}%** (percentage of individual trades closed in profit)",
        "",
        "---",
        "",
        "## 2. Alpha Contribution Distribution",
        "",
        "Shows the share of total profit generated by the top percentiles of stocks:",
        "",
        "| Percentile Group | Cumulative Profit Contribution | Share of Total Profit (%) |",
        "| :--- | :---: | :---: |",
        f"| **Top 10% of Stocks** | ₹{sorted_profits[:max(1, int(round(0.10*len(sorted_profits))))].sum():,.2f} | {top_10_share:.2f}% |",
        f"| **Top 20% of Stocks** | ₹{sorted_profits[:max(1, int(round(0.20*len(sorted_profits))))].sum():,.2f} | {top_20_share:.2f}% |",
        f"| **Top 50% of Stocks** | ₹{sorted_profits[:max(1, int(round(0.50*len(sorted_profits))))].sum():,.2f} | {top_50_share:.2f}% |",
        "",
        "---",
        "",
        "## 3. Inequality & Concentration Metrics",
        "",
        f"-   **Gini Coefficient (Positive Profits)**: **{gini:.4f}** (Status: **{gini_status.upper()}**)",
        f"-   **Herfindahl-Hirschman Index (HHI)**: **{hhi:.1f}**",
        "",
        "### Institutional Thresholds:",
        "",
        "| Gini Coefficient | Status | Current Status |",
        "| :--- | :---: | :---: |",
        "| <0.50 | Good | |",
        "| 0.50-0.70 | Watch | |",
        f"| >0.70 | Concentrated | **{gini_status.upper()} ({gini:.4f})** |",
        "",
        "### Lorenz Curve Cumulative Chart Data:",
        "Represents the cumulative profit generated by ascending percentiles of positive-profit stocks.",
        "",
        "| Traded Tickers Percentile (%) | Cumulative Share of Positive Profit (%) |",
        "| :---: | :---: |"
    ]
    
    for dec, share in lorenz_coords:
        report_lines.append(f"| {dec}% | {share:.2f}% |")
        
    report_lines.append("")
    report_lines.append("## 4. Breadth Audit Verdict")
    report_lines.append("")
    
    # Verdict logic
    if gini >= 0.70 or gvtd_share > 30.0:
        verdict = f"**RED (Concentrated Selection)**: The strategy's selection breadth is weak. The Gini coefficient is **{gini:.4f}** (Concentrated) and the top winner `GVT&D` explains **{gvtd_share:.1f}%** of all profit. This indicates that alpha is not distributed, failing institutional requirements for portfolio breadth."
    else:
        verdict = f"**GREEN (Pass)**: The selection breadth is healthy and returns are diversified across a broad range of tickers."
        
    report_lines.append(f"> [!IMPORTANT]")
    report_lines.append(f"> **VERDICT**: {verdict}")
    
    # Write to report
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "breadth_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[breadth_audit] Report successfully written to {artifact_path}")
    
    # Print summary to console
    print("\n" + "="*95)
    print("BREADTH AUDIT RESULTS")
    print("="*95)
    print(f"Unique Tickers Traded: {unique_stocks_traded} | Winning Tickers: {winning_stocks}")
    print(f"Effective Breadth:     {effective_breadth:.2f}")
    print(f"Winning Stocks Hit Rate: {stock_hit_rate:.2f}% | Winning Trades Hit Rate: {trade_hit_rate:.2f}%")
    print(f"Gini Coefficient:      {gini:.4f} (Status: {gini_status}) | HHI: {hhi:.2f}")
    print(f"Top 10% Share:         {top_10_share:.2f}% | Top 50% Share: {top_50_share:.2f}%")
    print("="*95 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

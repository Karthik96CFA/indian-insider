#!/usr/bin/env python3
"""
gvtd_dependency_audit.py — Audits the dependency of the strategy on GVT&D,
calculating CAGR, Sharpe, CAPM Alpha, and Profit Contribution metrics.
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
        print("[gvtd_audit] No tickers found in database score history.")
        conn.close()
        return 1
        
    print(f"[gvtd_audit] Loading price history for {len(tickers)} tickers...")
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
        print(f"[gvtd_audit] Bulk prices download failed: {exc}. Cannot proceed.")
        conn.close()
        return 1
        
    trading_dates = prices_df.index.strftime("%Y-%m-%d").tolist()
    
    # Fetch Nifty 50 benchmark
    print("[gvtd_audit] Fetching Nifty 50 benchmark...")
    try:
        nifty_raw = yf.download("^NSEI", start=start_date, end=end_date, progress=False)["Close"]
        if isinstance(nifty_raw, pd.DataFrame):
            nifty_raw = nifty_raw.squeeze()
        nifty_df = nifty_raw.reindex(prices_raw.index).ffill().bfill()
        nifty_series = pd.Series(nifty_df.values, index=trading_dates)
    except Exception as exc:
        print(f"[gvtd_audit] Nifty benchmark download failed: {exc}. Falling back to EWUI...")
        nifty_series = prices_df.mean(axis=1)
        
    bench_daily_rets = nifty_series.pct_change().fillna(0.0).to_numpy()
    
    start_idx = 126
    rebalance_indices = []
    curr_idx = start_idx
    while curr_idx + 63 < len(trading_dates):
        rebalance_indices.append(curr_idx)
        curr_idx += 63
        
    rebalance_dates = [trading_dates[idx] for idx in rebalance_indices]
    
    # Pre-cache database score history and event dates for speed
    print("[gvtd_audit] Pre-caching database score history...")
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
    def run_backtest_with_exclusions(exclude_set: set[str] = set()) -> tuple[list[float], list[dict], np.ndarray]:
        equity_curve = [initial_capital] * start_idx
        trades = []
        daily_returns = [0.0] * start_idx
        
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
                daily_returns.append((day_val - equity_curve[-1]) / equity_curve[-1])
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
                
            daily_returns[-1] = (final_cap - equity_curve[-2]) / equity_curve[-2]
            equity_curve[-1] = final_cap
            
        # Pad lists
        while len(equity_curve) < len(trading_dates):
            equity_curve.append(equity_curve[-1])
            daily_returns.append(0.0)
            
        return equity_curve, trades, np.array(daily_returns)
        
    # 1. Run Baseline (with GVT&D)
    print("[gvtd_audit] Running baseline strategy...")
    base_curve, base_trades, base_daily_rets = run_backtest_with_exclusions()
    
    # 2. Run Exclusion (without GVT&D)
    print("[gvtd_audit] Running strategy without GVT&D...")
    ex_curve, ex_trades, ex_daily_rets = run_backtest_with_exclusions(exclude_set={"GVT&D"})
    
    # Calculate performance metrics
    n_days = len(trading_dates) - start_idx
    years = n_days / 252.0
    
    def get_performance_stats(curve: list[float], daily_rets_full: np.ndarray) -> dict:
        final_val = curve[-1]
        cagr = (final_val / initial_capital) ** (1.0 / years) - 1.0 if final_val > 0 else -1.0
        
        daily_returns_slice = daily_rets_full[start_idx:]
        std_ret = daily_returns_slice.std()
        mean_ret = daily_returns_slice.mean()
        sharpe = math.sqrt(252.0) * (mean_ret - daily_rf) / std_ret if std_ret > 0 else 0.0
        
        # CAPM Alpha against Nifty 50
        bench_slice = bench_daily_rets[start_idx:]
        cov_matrix = np.cov(daily_returns_slice, bench_slice)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 1.0
        alpha_daily = daily_returns_slice.mean() - beta * bench_slice.mean()
        alpha_ann = alpha_daily * 252.0
        
        # Max DD
        running_max = curve[0]
        max_dd = 0.0
        for val in curve:
            if val > running_max:
                running_max = val
            dd = (val - running_max) / running_max
            if dd < max_dd:
                max_dd = dd
                
        return {
            "CAGR": cagr * 100.0,
            "Sharpe": sharpe,
            "Alpha": alpha_ann * 100.0,
            "Beta": beta,
            "MaxDD": max_dd * 100.0
        }
        
    base_stats = get_performance_stats(base_curve, base_daily_rets)
    ex_stats = get_performance_stats(ex_curve, ex_daily_rets)
    
    # Calculate Profit Contribution
    df_trades = pd.DataFrame(base_trades)
    total_portfolio_profit = df_trades["profit"].sum()
    
    # Profit by ticker
    ticker_profit = df_trades.groupby("ticker")["profit"].sum().reset_index()
    ticker_profit = ticker_profit.sort_values(by="profit", ascending=False).reset_index(drop=True)
    
    # GVT&D Profit
    gvtd_profit_row = ticker_profit[ticker_profit["ticker"] == "GVT&D"]
    gvtd_profit = gvtd_profit_row["profit"].values[0] if len(gvtd_profit_row) > 0 else 0.0
    gvtd_contribution_pct = (gvtd_profit / total_portfolio_profit) * 100.0 if total_portfolio_profit > 0 else 0.0
    
    # Classification of GVT&D
    # <10% Healthy, 10-20% Watch, 20-30% Concern, >30% Critical
    if gvtd_contribution_pct < 10.0:
        gvtd_status = "Healthy"
    elif gvtd_contribution_pct <= 20.0:
        gvtd_status = "Watch"
    elif gvtd_contribution_pct <= 30.0:
        gvtd_status = "Concern"
    else:
        gvtd_status = "Critical"
        
    # Cumulative contributions for Top 1, Top 3, Top 5, Top 10
    top_1_profit = ticker_profit.head(1)["profit"].sum()
    top_3_profit = ticker_profit.head(3)["profit"].sum()
    top_5_profit = ticker_profit.head(5)["profit"].sum()
    top_10_profit = ticker_profit.head(10)["profit"].sum()
    
    top_1_contrib = (top_1_profit / total_portfolio_profit) * 100.0 if total_portfolio_profit > 0 else 0.0
    top_3_contrib = (top_3_profit / total_portfolio_profit) * 100.0 if total_portfolio_profit > 0 else 0.0
    top_5_contrib = (top_5_profit / total_portfolio_profit) * 100.0 if total_portfolio_profit > 0 else 0.0
    top_10_contrib = (top_10_profit / total_portfolio_profit) * 100.0 if total_portfolio_profit > 0 else 0.0
    
    # Write Report
    report_lines = [
        "# GVT&D Dependency Audit Report",
        "",
        "This audit isolates the exact return, risk, and alpha contributions of the dominant ticker **GVT&D** to the overall portfolio strategy.",
        "",
        "## 1. GVT&D Contribution Analysis",
        "",
        f"-   **Total Portfolio Profit**: **₹{total_portfolio_profit:,.2f}**",
        f"-   **GVT&D Profit Contribution**: **₹{gvtd_profit:,.2f}**",
        f"-   **GVT&D Return Attribution %**: **{gvtd_contribution_pct:.2f}%**",
        f"-   **GVT&D Concentration Status**: **{gvtd_status.upper()}**",
        "",
        "### Contribution Thresholds & Status:",
        "",
        "| Contribution | Status | Current Status |",
        "| :--- | :---: | :---: |",
        "| <10% | Healthy | |",
        "| 10-20% | Watch | |",
        "| 20-30% | Concern | |",
        f"| >30% | Critical | **{gvtd_status.upper()} ({gvtd_contribution_pct:.1f}%)** |",
        "",
        "---",
        "",
        "## 2. Cumulative Concentration Leaderboard",
        "",
        "| Holding Group | Cumulative Profit Contribution | Contribution % |",
        "| :--- | :---: | :---: |",
        f"| **Top 1 Holding ({ticker_profit.loc[0, 'ticker']})** | ₹{top_1_profit:,.2f} | {top_1_contrib:.2f}% |",
        f"| **Top 3 Holdings** | ₹{top_3_profit:,.2f} | {top_3_contrib:.2f}% |",
        f"| **Top 5 Holdings** | ₹{top_5_profit:,.2f} | {top_5_contrib:.2f}% |",
        f"| **Top 10 Holdings** | ₹{top_10_profit:,.2f} | {top_10_contrib:.2f}% |",
        "",
        "---",
        "",
        "## 3. Performance Impact Comparison",
        "",
        "Isolates the portfolio metrics when `GVT&D` is included vs. excluded from the ranking universe:",
        "",
        "| Metric | Baseline (With GVT&D) | Ex-GVT&D (Without GVT&D) | Net Contribution |",
        "| :--- | :---: | :---: | :---: |",
        f"| **CAGR (%)** | {base_stats['CAGR']:+.2f}% | {ex_stats['CAGR']:+.2f}% | {base_stats['CAGR'] - ex_stats['CAGR']:+.2f}% |",
        f"| **Sharpe Ratio** | {base_stats['Sharpe']:.4f} | {ex_stats['Sharpe']:.4f} | {base_stats['Sharpe'] - ex_stats['Sharpe']:+.4f} |",
        f"| **CAPM Alpha (%)** | {base_stats['Alpha']:+.2f}% | {ex_stats['Alpha']:+.2f}% | {base_stats['Alpha'] - ex_stats['Alpha']:+.2f}% |",
        f"| **CAPM Beta** | {base_stats['Beta']:.4f} | {ex_stats['Beta']:.4f} | {base_stats['Beta'] - ex_stats['Beta']:+.4f} |",
        f"| **Max Drawdown (%)** | {base_stats['MaxDD']:.2f}% | {ex_stats['MaxDD']:.2f}% | {base_stats['MaxDD'] - ex_stats['MaxDD']:+.2f}% |",
        "",
        "---",
        "",
        "## 4. GVT&D Audit Verdict",
        "",
        "> [!IMPORTANT]"
    ]
    
    alpha_contrib_pct = (1.0 - (ex_stats["Alpha"] / base_stats["Alpha"])) * 100.0 if base_stats["Alpha"] != 0 else 0.0
    
    if gvtd_contribution_pct > 30.0 or alpha_contrib_pct > 25.0:
        verdict = f"**RED (Critical Concentration)**: GVT&D represents **{gvtd_contribution_pct:.1f}%** of total portfolio profit and contributes **{alpha_contrib_pct:.1f}%** of the strategy's CAPM Alpha. This exceeds institutional concentration thresholds, making the strategy highly dependent on a single stock."
    else:
        verdict = f"**GREEN (Pass)**: GVT&D is within acceptable limits for return attribution and alpha contribution."
        
    report_lines.append(f"> **VERDICT**: {verdict}")
    
    # Write to report
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "gvtd_dependency_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[gvtd_audit] Report successfully written to {artifact_path}")
    
    # Print summary to console
    print("\n" + "="*95)
    print("GVT&D DEPENDENCY AUDIT RESULTS")
    print("="*95)
    print(f"GVT&D Contribution: {gvtd_contribution_pct:.2f}% (Status: {gvtd_status})")
    print(f"Baseline CAGR: {base_stats['CAGR']:+.2f}% | Sharpe: {base_stats['Sharpe']:.4f} | Alpha: {base_stats['Alpha']:+.2f}%")
    print(f"Ex-GVT&D CAGR: {ex_stats['CAGR']:+.2f}% | Sharpe: {ex_stats['Sharpe']:.4f} | Alpha: {ex_stats['Alpha']:+.2f}%")
    print("="*95 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

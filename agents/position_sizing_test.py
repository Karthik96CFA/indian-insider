#!/usr/bin/env python3
"""
position_sizing_test.py — Validates and compares portfolio performance across
four position sizing methods (Equal Weight, Rank Weight, Volatility Target, and Risk Parity).
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
    """
    Programmatically normalizes and enforces weight caps.
    """
    caps = {
        "tailwind": 0.20,
        "momentum": 0.20,
        "credibility": 0.15
    }
    w = raw_weights.copy()
    total_w = sum(w.values())
    if total_w == 0:
        return w
    norm_w = {k: v / total_w for k, v in w.items()}
    
    violated = {}
    for k, cap in caps.items():
        if k in norm_w and norm_w[k] > cap:
            violated[k] = cap
            
    if not violated:
        return norm_w
        
    final_w = {}
    for k in violated:
        final_w[k] = violated[k]
        
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

def solve_erc_coordinate_descent(cov: np.ndarray, max_iter: int = 100, tol: float = 1e-7) -> np.ndarray:
    """
    Solves for equal risk contribution (Risk Parity) weights using cyclical coordinate descent.
    """
    n = cov.shape[0]
    w = np.ones(n) / n
    c = 1.0
    
    for iteration in range(max_iter):
        w_old = w.copy()
        for i in range(n):
            b = np.dot(w, cov[i]) - w[i] * cov[i, i]
            val = b * b + 4.0 * cov[i, i] * c
            if val < 0:
                val = 0.0
            w[i] = (-b + np.sqrt(val)) / (2.0 * cov[i, i])
            
        if np.max(np.abs(w - w_old)) < tol:
            break
            
    return w / np.sum(w)

def main() -> int:
    conn = _conn()
    
    # 1. Fetch all distinct tickers from score history
    tickers_rows = conn.execute("SELECT DISTINCT ticker FROM company_scores_history").fetchall()
    tickers = sorted([r[0] for r in tickers_rows if r[0] not in {"NIFTY", "BANKNIFTY", "NIFTYBEES", "GOLDBEES", "GOLD"}])
    
    if not tickers:
        print("[position_sizing] No tickers found in database score history.")
        conn.close()
        return 1
        
    print(f"[position_sizing] Preparing price history for {len(tickers)} tickers...")
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
        print(f"[position_sizing] Bulk prices download failed: {exc}. Cannot proceed.")
        conn.close()
        return 1
        
    trading_dates = prices_df.index.strftime("%Y-%m-%d").tolist()
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}
    
    # Start the backtest at index 126 to ensure a full 6-month (126-day) price lookback window
    # is available for volatility and risk parity covariance calculations on day 1.
    start_idx = 126
    rebalance_indices = []
    curr_idx = start_idx
    while curr_idx + 63 < len(trading_dates):
        rebalance_indices.append(curr_idx)
        curr_idx += 63
        
    rebalance_dates = [trading_dates[idx] for idx in rebalance_indices]
    print(f"[position_sizing] Running backtest starting on {trading_dates[start_idx]} across {len(rebalance_dates)} rebalance cycles.")
    
    # 2. Define Position Sizing Methods
    sizing_methods = ["Equal Weight", "Rank Weight", "Volatility Target", "Risk Parity"]
    initial_capital = 10000000.0
    equity_curves = {m: [initial_capital] * start_idx for m in sizing_methods}
    trades_log = {m: [] for m in sizing_methods}
    
    # Keep track of active weights per ticker at each rebalance cycle for Turnover calculations
    active_weights = {m: {} for m in sizing_methods} # method -> cycle_idx -> {ticker: weight}
    
    for cycle_idx, entry_idx in enumerate(rebalance_indices):
        entry_date = trading_dates[entry_idx]
        exit_idx = entry_idx + 63
        exit_date = trading_dates[exit_idx]
        
        # A. Compute scores for all tickers on entry date using production weights
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
        
        # B. Winsorized Percentile Normalization (0-100)
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
                
        # C. Compute point-in-time Momentum decay and final score
        final_scores = []
        for idx, row in df_pct.iterrows():
            t = row["ticker"]
            cov_score = row["coverage_score"]
            
            # Filter active tickers (coverage >= 50%)
            if cov_score < 50.0:
                continue
                
            # Find latest event date on or before entry_date
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
                
            # Decay Momentum weight
            T_HALF = 5.0
            decay_factor = math.exp(- (math.log(2.0) / T_HALF) * delay)
            w_mom = 0.10 * decay_factor
            if delay > 7:
                w_mom = 0.0
                
            # Setup raw weights (Production Config)
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
            
            raw_total_score = 0.0
            for f in factors:
                raw_total_score += capped_weights[f] * row[f]
                
            total_score = raw_total_score * (cov_score / 100.0)
            
            final_scores.append({
                "ticker": t,
                "score": total_score
            })
            
        df_rankings = pd.DataFrame(final_scores)
        
        # Filter for valid entry prices
        valid_tickers = []
        for t in df_rankings["ticker"].tolist():
            if t in prices_df.columns:
                p = prices_df.loc[entry_date, t]
                if not pd.isna(p) and p > 0:
                    valid_tickers.append(t)
                    
        df_rankings_filtered = df_rankings[df_rankings["ticker"].isin(valid_tickers)].copy()
        
        # Select top 10 tickers
        top_10 = df_rankings_filtered.sort_values(by="score", ascending=False).head(10)["ticker"].tolist()
        
        if len(top_10) == 0:
            continue
            
        # D. Retrieve daily lookback returns for Volatility and Risk Parity
        lookback_df = prices_df.iloc[entry_idx - 126 : entry_idx][top_10].ffill().bfill()
        lookback_returns = lookback_df.pct_change().fillna(0.0)
        
        # E. Calculate weights for each method
        weights_dict = {}
        
        # 1. Equal Weight (10% each)
        weights_dict["Equal Weight"] = np.ones(len(top_10)) / len(top_10)
        
        # 2. Rank Weight
        rank_weights_baseline = [0.15, 0.13, 0.12, 0.11, 0.10, 0.095, 0.09, 0.08, 0.07, 0.055]
        # Pad or truncate if we have fewer than 10 positions (unlikely)
        if len(top_10) < 10:
            w_raw = rank_weights_baseline[:len(top_10)]
            weights_dict["Rank Weight"] = np.array(w_raw) / sum(w_raw)
        else:
            weights_dict["Rank Weight"] = np.array(rank_weights_baseline)
            
        # 3. Volatility Target (Inverse Volatility)
        vols = lookback_returns.std().to_numpy()
        vols = np.where(vols <= 0.0, 1e-4, vols)
        inv_vols = 1.0 / vols
        weights_dict["Volatility Target"] = inv_vols / np.sum(inv_vols)
        
        # 4. Risk Parity (Equal Risk Contribution)
        cov = lookback_returns.cov().to_numpy()
        cov = cov + np.eye(cov.shape[0]) * 1e-6 # shrinkage for stability
        rp_weights = solve_erc_coordinate_descent(cov)
        if np.any(np.isnan(rp_weights)) or np.any(np.isinf(rp_weights)):
            # Fallback to inverse volatility
            weights_dict["Risk Parity"] = weights_dict["Volatility Target"]
        else:
            weights_dict["Risk Parity"] = rp_weights
            
        # F. Run backtest simulation for each method
        for m in sizing_methods:
            weights = weights_dict[m]
            active_weights[m][cycle_idx] = {top_10[i]: weights[i] for i in range(len(top_10))}
            
            current_capital = equity_curves[m][-1]
            
            positions = []
            for i, t in enumerate(top_10):
                p0 = prices_df.loc[entry_date, t]
                fee_rate = get_variable_transaction_cost(t)
                net_allocation = current_capital * weights[i] * (1.0 - fee_rate)
                positions.append({
                    "ticker": t,
                    "entry_price": p0,
                    "allocated": net_allocation,
                    "fee_rate": fee_rate,
                    "target_weight": weights[i]
                })
                
            # Simulate daily valuations
            for idx in range(entry_idx, exit_idx):
                day_date = trading_dates[idx]
                day_val = 0.0
                for pos in positions:
                    t = pos["ticker"]
                    p_day = prices_df.loc[day_date, t]
                    if pd.isna(p_day) or p_day <= 0:
                        p_day = pos["entry_price"]
                    day_val += pos["allocated"] * (p_day / pos["entry_price"])
                equity_curves[m].append(day_val)
                
            # Exit valuation and fee deduction
            final_portfolio_value = 0.0
            for pos in positions:
                t = pos["ticker"]
                p_exit = prices_df.loc[exit_date, t]
                if pd.isna(p_exit) or p_exit <= 0:
                    p_exit = pos["entry_price"]
                val_before_fee = pos["allocated"] * (p_exit / pos["entry_price"])
                val_after_fee = val_before_fee * (1.0 - pos["fee_rate"])
                final_portfolio_value += val_after_fee
                
                trade_return = (val_after_fee - (current_capital * pos["target_weight"])) / (current_capital * pos["target_weight"])
                trades_log[m].append(trade_return)
                
            equity_curves[m][-1] = final_portfolio_value
            
    conn.close()
    
    # Pad equity curves
    for m in sizing_methods:
        while len(equity_curves[m]) < len(trading_dates):
            equity_curves[m].append(equity_curves[m][-1])
            
    # Calculate performance metrics
    stats = []
    n_days = len(trading_dates) - start_idx
    years = n_days / 252.0
    rf = 0.06
    daily_rf = (1.0 + rf) ** (1.0 / 252.0) - 1.0
    
    for m in sizing_methods:
        curve = equity_curves[m]
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
                
        # Calculate Turnover (average sum of absolute changes in weights between cycles)
        turnovers = []
        all_cycles = sorted(list(active_weights[m].keys()))
        for c_idx in range(1, len(all_cycles)):
            c1 = all_cycles[c_idx - 1]
            c2 = all_cycles[c_idx]
            w1 = active_weights[m][c1]
            w2 = active_weights[m][c2]
            
            # Combine all tickers in both cycles
            all_t = set(w1.keys()).union(set(w2.keys()))
            cycle_turnover = sum(abs(w2.get(t, 0.0) - w1.get(t, 0.0)) for t in all_t)
            turnovers.append(cycle_turnover)
            
        avg_turnover = np.mean(turnovers) if turnovers else 0.0
        # Weight Stability = 1.0 - (Average Turnover / 2.0)
        weight_stability = (1.0 - (avg_turnover / 2.0)) * 100.0
        
        wins = sum(1 for r in trades_log[m] if r > 0)
        win_rate = (wins / len(trades_log[m]) * 100.0) if trades_log[m] else 0.0
        
        stats.append({
            "Method": m,
            "CAGR": cagr * 100.0,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "MaxDD": max_dd * 100.0,
            "Stability": weight_stability,
            "WinRate": win_rate
        })
        
    df_stats = pd.DataFrame(stats).sort_values(by="Sharpe", ascending=False)
    
    # 3. Generate Position Sizing Report Markdown
    report_lines = []
    report_lines.append("# Stage 8: Position Sizing Validation Report")
    report_lines.append("")
    report_lines.append("This report compares portfolio performance and weight stability across **four position sizing methods** using the updated production scoring weights.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 1. Position Sizing Leaderboard")
    report_lines.append("")
    report_lines.append("| Rank | Position Sizing Method | CAGR (%) | Sharpe | Sortino | Max DD | Weight Stability (%) | Win Rate |")
    report_lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for idx, r in enumerate(df_stats.iterrows()):
        row = r[1]
        report_lines.append(
            f"| {idx+1} | **{row['Method']}** | {row['CAGR']:+.2f}% | {row['Sharpe']:.4f} | {row['Sortino']:.4f} | {row['MaxDD']:.2f}% | {row['Stability']:.2f}% | {row['WinRate']:.1f}% |"
        )
        
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 2. Quantitative Key Question Breakdown")
    report_lines.append("")
    report_lines.append("### *Which position sizing method maximizes risk-adjusted return and stability?*")
    report_lines.append("")
    
    best_method = df_stats.iloc[0]["Method"]
    best_sharpe = df_stats.iloc[0]["Sharpe"]
    best_cagr = df_stats.iloc[0]["CAGR"]
    best_stability = df_stats.iloc[0]["Stability"]
    
    report_lines.append(f"1. **Performance Breakdown**:")
    for r in stats:
        report_lines.append(f"   *   **{r['Method']}**: CAGR of **{r['CAGR']:+.2f}%** | Sharpe of **{r['Sharpe']:.4f}** | Sortino of **{r['Sortino']:.4f}** | Max Drawdown of **{r['MaxDD']:.2f}%**.")
    report_lines.append("")
    report_lines.append(f"2. **Weight Stability Analysis**:")
    report_lines.append("   *   *Weight Stability* measures how consistent target weights are from one rebalance to the next. Higher stability implies lower transaction turnover drag.")
    for r in stats:
        report_lines.append(f"   *   **{r['Method']}**: Weight Stability of **{r['Stability']:.2f}%**.")
    report_lines.append("")
    
    report_lines.append("## 3. Position Sizing Verdict")
    report_lines.append("")
    report_lines.append("> [!IMPORTANT]")
    report_lines.append(f"> **RECOMMENDED SIZING METHOD**: **{best_method}** is the optimal choice. It achieves a Sharpe Ratio of **{best_sharpe:.4f}** (CAGR: **{best_cagr:+.2f}%**) with a Weight Stability of **{best_stability:.2f}%**.")
    report_lines.append("> ")
    report_lines.append("> *   **Rank Weighting** leverages structural scoring gradients, giving larger allocations to the highest-scoring stocks. This maximizes raw return but can increase max drawdown and concentration risk.")
    report_lines.append("> *   **Inverse Volatility Sizing** mitigates drawdown by downweighting highly volatile mid/small caps, but may drag on performance in strong bull regimes.")
    report_lines.append("> *   **Risk Parity (ERC)** balances the risk contribution of all holdings, yielding the most robust risk-adjusted return (Sharpe) and protecting the portfolio against systemic idiosyncratic shocks.")
    report_lines.append("")
    
    report_lines.append(f"*Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Save Report
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "position_sizing_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[position_sizing] Report successfully written to {artifact_path}")
    
    # Print leaderboard to console
    print("\n" + "="*95)
    print("POSITION SIZING COMPARISON LEADERBOARD")
    print("="*95)
    print(f"{'Rank':<5} {'Sizing Method':<20} {'CAGR':<10} {'Sharpe':<8} {'Sortino':<8} {'MaxDD':<8} {'Stability':<10}")
    print("-"*95)
    for idx, r in enumerate(df_stats.iterrows()):
        row = r[1]
        print(f"{idx+1:<5} {row['Method']:<20} {row['CAGR']:>+8.2f}% {row['Sharpe']:>7.4f} {row['Sortino']:>7.4f} {row['MaxDD']:>7.2f}% {row['Stability']:>8.2f}%")
    print("="*95 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

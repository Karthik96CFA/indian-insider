#!/usr/bin/env python3
"""
reality_check_suite.py — Runs the 10 stress tests, sector concentration diagnostics,
and the Random Factor Test.
"""
from __future__ import annotations

import datetime
import math
import sqlite3
import random
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
    
    # Fetch sectors mapping
    sector_rows = conn.execute("SELECT ticker, sector FROM company_fundamentals").fetchall()
    ticker_sector = {r[0]: r[1] for r in sector_rows}
    
    if not tickers:
        print("[reality_check] No tickers found in database score history.")
        conn.close()
        return 1
        
    print(f"[reality_check] Loading price history for {len(tickers)} tickers...")
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
        print(f"[reality_check] Bulk prices download failed: {exc}. Cannot proceed.")
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
    print("[reality_check] Pre-caching database score history...")
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
    
    # Fast exact backtester
    def run_backtest(
        weights_config: dict[str, float],
        exclude_tickers: set[str] = set(),
        exclude_sectors: set[str] = set(),
        jitter_rebalance: bool = False,
        double_fees: bool = False,
        delay_execution: bool = False
    ) -> tuple[list[float], list[dict], list[dict]]:
        
        # Jitter rebalance indices if requested
        sim_rebalance_indices = []
        if jitter_rebalance:
            for idx in rebalance_indices:
                jitter = random.randint(-10, 10)
                jit_idx = min(len(trading_dates) - 64, max(start_idx, idx + jitter))
                sim_rebalance_indices.append(jit_idx)
            sim_rebalance_indices = sorted(list(set(sim_rebalance_indices)))
        else:
            sim_rebalance_indices = rebalance_indices
            
        equity_curve = [initial_capital] * start_idx
        trades = []
        sector_concentrations = []
        
        factors = ["quality", "growth", "valuation", "momentum", "institutional", "tailwind", "credibility"]
        
        for cycle_idx, entry_idx in enumerate(sim_rebalance_indices):
            entry_date = trading_dates[entry_idx]
            exit_idx = entry_idx + 63
            if exit_idx >= len(trading_dates):
                exit_idx = len(trading_dates) - 1
            exit_date = trading_dates[exit_idx]
            
            # Fetch scores
            scores_df_data = []
            for t in tickers:
                if t in exclude_tickers:
                    continue
                sect = ticker_sector.get(t)
                if sect in exclude_sectors:
                    continue
                    
                row = cached_scores.get(entry_date, {}).get(t)
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
                    
                latest_event_date = cached_event_dates.get(entry_date, {}).get(t)
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
                w_mom = weights_config["momentum"] * decay_factor
                if delay > 7:
                    w_mom = 0.0
                    
                raw_weights = {
                    "quality": weights_config["quality"],
                    "growth": weights_config["growth"],
                    "valuation": weights_config["valuation"],
                    "momentum": w_mom,
                    "institutional": weights_config["institutional"],
                    "tailwind": weights_config["tailwind"],
                    "credibility": weights_config["credibility"]
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
                
            # Sector Diagnostics
            sectors_in_portfolio = [ticker_sector.get(t, "Other") for t in top_10]
            sector_counts = pd.Series(sectors_in_portfolio).value_counts(normalize=True)
            top_sect_w = sector_counts.iloc[0]
            top_2_sect_w = sector_counts.iloc[:2].sum()
            top_3_sect_w = sector_counts.iloc[:3].sum()
            sector_concentrations.append({
                "top_1": top_sect_w,
                "top_2": top_2_sect_w,
                "top_3": top_3_sect_w
            })
            
            # Sizing and Trading
            weight = 1.0 / len(top_10)
            current_cap = equity_curve[-1]
            
            # Delayed execution offset
            trade_entry_idx = entry_idx
            trade_exit_idx = exit_idx
            if delay_execution:
                trade_entry_idx = min(len(trading_dates) - 1, entry_idx + 1)
                trade_exit_idx = min(len(trading_dates) - 1, exit_idx + 1)
                
            trade_entry_date = trading_dates[trade_entry_idx]
            trade_exit_date = trading_dates[trade_exit_idx]
            
            positions = []
            for t in top_10:
                p0 = prices_df.loc[trade_entry_date, t]
                p0_val = p0 if not pd.isna(p0) and p0 > 0 else prices_df.loc[entry_date, t]
                fee_rate = fee_rates[t]
                if double_fees:
                    fee_rate *= 2.0
                net_alloc = current_cap * weight * (1.0 - fee_rate)
                positions.append({"ticker": t, "entry_price": p0_val, "allocated": net_alloc, "fee_rate": fee_rate})
                
            # Daily tracking
            for idx in range(entry_idx, exit_idx):
                day_date = trading_dates[idx]
                day_val = sum(pos["allocated"] * (prices_df.loc[day_date, pos["ticker"]] / pos["entry_price"]) for pos in positions)
                equity_curve.append(day_val)
                
            # Exit fee deduction
            final_cap = 0.0
            for pos in positions:
                p_exit = prices_df.loc[trade_exit_date, pos["ticker"]]
                p_exit_val = p_exit if not pd.isna(p_exit) and p_exit > 0 else pos["entry_price"]
                val_after_exit_fee = pos["allocated"] * (p_exit_val / pos["entry_price"]) * (1.0 - pos["fee_rate"])
                final_cap += val_after_exit_fee
                
                trade_ret = (p_exit_val / pos["entry_price"]) - 1.0
                trades.append({"ticker": pos["ticker"], "return": trade_ret, "profit": val_after_exit_fee - (current_cap * weight)})
                
            equity_curve[-1] = final_cap
            
        # Pad lists
        while len(equity_curve) < len(trading_dates):
            equity_curve.append(equity_curve[-1])
            
        return equity_curve, trades, sector_concentrations
        
    # Run Baseline
    print("[reality_check] Running baseline strategy...")
    base_curve, base_trades, base_sec = run_backtest(baseline_weights)
    
    # Calculate profit contribution to identify top winners
    df_trades = pd.DataFrame(base_trades)
    ticker_contribution = df_trades.groupby("ticker")["profit"].sum().reset_index()
    ticker_contribution_sorted = ticker_contribution.sort_values(by="profit", ascending=False).reset_index(drop=True)
    
    top_1_winner = {ticker_contribution_sorted.loc[0, "ticker"]}
    top_2_winners = set(ticker_contribution_sorted.head(2)["ticker"].tolist())
    top_5_winners = set(ticker_contribution_sorted.head(5)["ticker"].tolist())
    
    # --- Execute the 10 Stress Tests ---
    print("[reality_check] Running the 10 stress tests...")
    
    tests = {
        "Baseline": {},
        "1. Top-1 Removed": {"exclude_tickers": top_1_winner},
        "2. Top-2 Removed": {"exclude_tickers": top_2_winners},
        "3. Top-5 Removed": {"exclude_tickers": top_5_winners},
        "4. Remove IT Sector": {"exclude_sectors": {"Technology"}},
        "5. Remove Banking Sector": {"exclude_sectors": {"Financial Services"}},
        "6. Remove Quality Factor": {"weights_config": {
            "quality": 0.0, "growth": 0.30, "valuation": 0.0, "momentum": 0.10, "institutional": 0.20, "tailwind": 0.0, "credibility": 0.0
        }},
        "7. Remove Growth Factor": {"weights_config": {
            "quality": 0.40, "growth": 0.0, "valuation": 0.0, "momentum": 0.10, "institutional": 0.20, "tailwind": 0.0, "credibility": 0.0
        }},
        "8. Jitter Rebalance Date ±10d": {"jitter_rebalance": True},
        "9. Double Transaction Costs": {"double_fees": True},
        "10. Delayed Execution by 1 Day": {"delay_execution": True}
    }
    
    stats = []
    n_days = len(trading_dates) - start_idx
    years = n_days / 252.0
    
    # For Jitter Rebalance, we run it 50 times and take the average
    for test_name, params in tests.items():
        if test_name == "8. Jitter Rebalance Date ±10d":
            print("  Running Jitter Rebalance (50 simulations)...")
            cagrs_j = []
            sharpes_j = []
            sortinos_j = []
            max_dds_j = []
            for sim in range(50):
                curve, _, _ = run_backtest(baseline_weights, jitter_rebalance=True)
                # Compute returns
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
                        
                cagrs_j.append(cagr)
                sharpes_j.append(sharpe)
                sortinos_j.append(sortino)
                max_dds_j.append(max_dd)
                
            stats.append({
                "Test": test_name,
                "CAGR": np.mean(cagrs_j) * 100.0,
                "Sharpe": np.mean(sharpes_j),
                "Sortino": np.mean(sortinos_j),
                "MaxDD": np.mean(max_dds_j) * 100.0
            })
        else:
            w_config = params.get("weights_config", baseline_weights)
            exclude_t = params.get("exclude_tickers", set())
            exclude_s = params.get("exclude_sectors", set())
            double_f = params.get("double_fees", False)
            delay_e = params.get("delay_execution", False)
            
            curve, _, _ = run_backtest(
                w_config, exclude_tickers=exclude_t, exclude_sectors=exclude_s,
                double_fees=double_f, delay_execution=delay_e
            )
            
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
                "Test": test_name,
                "CAGR": cagr * 100.0,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "MaxDD": max_dd * 100.0
            })
            
    df_stats = pd.DataFrame(stats)
    
    # --- Sector Concentration Diagnostics ---
    df_sec = pd.DataFrame(base_sec)
    sec_stats = {
        "top_1": {
            "mean": df_sec["top_1"].mean() * 100.0,
            "p95": df_sec["top_1"].quantile(0.95) * 100.0,
            "max": df_sec["top_1"].max() * 100.0
        },
        "top_2": {
            "mean": df_sec["top_2"].mean() * 100.0,
            "p95": df_sec["top_2"].quantile(0.95) * 100.0,
            "max": df_sec["top_2"].max() * 100.0
        },
        "top_3": {
            "mean": df_sec["top_3"].mean() * 100.0,
            "p95": df_sec["top_3"].quantile(0.95) * 100.0,
            "max": df_sec["top_3"].max() * 100.0
        }
    }
    
    # --- Random Factor Test ---
    print("[reality_check] Running 100 Random Factor simulations...")
    random.seed(1337)
    np.random.seed(1337)
    
    random_sharpes = []
    random_cagrs = []
    
    for sim in range(100):
        # Generate random rebalance selections
        equity_curve_r = [initial_capital] * start_idx
        for cycle_idx, entry_idx in enumerate(rebalance_indices):
            entry_date = trading_dates[entry_idx]
            exit_idx = entry_idx + 63
            if exit_idx >= len(trading_dates):
                exit_idx = len(trading_dates) - 1
            exit_date = trading_dates[exit_idx]
            
            # Filter valid entry tickers
            valid_tickers = [t for t in tickers if t in prices_df.columns and not pd.isna(prices_df.loc[entry_date, t]) and prices_df.loc[entry_date, t] > 0]
            if not valid_tickers:
                continue
                
            # Random select 10 tickers
            top_10 = random.sample(valid_tickers, min(10, len(valid_tickers)))
            
            weight = 1.0 / len(top_10)
            current_cap = equity_curve_r[-1]
            
            positions = []
            for t in top_10:
                p0 = prices_df.loc[entry_date, t]
                fee_rate = fee_rates[t]
                net_alloc = current_cap * weight * (1.0 - fee_rate)
                positions.append({"ticker": t, "entry_price": p0, "allocated": net_alloc, "fee_rate": fee_rate})
                
            for idx in range(entry_idx, exit_idx):
                day_date = trading_dates[idx]
                day_val = sum(pos["allocated"] * (prices_df.loc[day_date, pos["ticker"]] / pos["entry_price"]) for pos in positions)
                equity_curve_r.append(day_val)
                
            final_cap = 0.0
            for pos in positions:
                p_exit = prices_df.loc[exit_date, pos["ticker"]]
                p_exit_val = p_exit if not pd.isna(p_exit) and p_exit > 0 else pos["entry_price"]
                val_after_exit_fee = pos["allocated"] * (p_exit_val / pos["entry_price"]) * (1.0 - pos["fee_rate"])
                final_cap += val_after_exit_fee
            equity_curve_r[-1] = final_cap
            
        while len(equity_curve_r) < len(trading_dates):
            equity_curve_r.append(equity_curve_r[-1])
            
        # Stats
        final_val = equity_curve_r[-1]
        cagr = (final_val / initial_capital) ** (1.0 / years) - 1.0 if final_val > 0 else -1.0
        
        daily_returns = []
        for idx in range(start_idx, len(equity_curve_r)):
            r = (equity_curve_r[idx] - equity_curve_r[idx-1]) / equity_curve_r[idx-1]
            daily_returns.append(r)
        daily_returns = np.array(daily_returns)
        
        std_ret = daily_returns.std()
        mean_ret = daily_returns.mean()
        sharpe = math.sqrt(252.0) * (mean_ret - daily_rf) / std_ret if std_ret > 0 else 0.0
        
        random_sharpes.append(sharpe)
        random_cagrs.append(cagr * 100.0)
        
    random_sharpes = sorted(random_sharpes)
    random_cagrs = sorted(random_cagrs)
    
    baseline_sharpe = df_stats.loc[df_stats["Test"] == "Baseline", "Sharpe"].values[0]
    baseline_cagr = df_stats.loc[df_stats["Test"] == "Baseline", "CAGR"].values[0]
    
    # Percentile placement
    sharpe_percentile = (sum(1 for s in random_sharpes if s < baseline_sharpe) / len(random_sharpes)) * 100.0
    cagr_percentile = (sum(1 for c in random_cagrs if c < baseline_cagr) / len(random_cagrs)) * 100.0
    
    # Write Report
    report_lines = [
        "# Stage 7: Reality Check Stress Tests Report",
        "",
        "This report documents the performance of the production model under 10 specific stress tests, evaluates sector concentration diagnostics, and runs the Random Factor Test.",
        "",
        "## 1. Reality Check Stress Tests Leaderboard",
        "",
        "| Stress Test | CAGR (%) | Sharpe | Sortino | Max DD (%) | Sharpe Decay (%) | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    for idx, r in df_stats.iterrows():
        decay = ((baseline_sharpe - r["Sharpe"]) / baseline_sharpe) * 100.0 if baseline_sharpe > 0 else 0.0
        status = "PASS"
        if r["Test"] == "3. Top-5 Removed" and r["Sharpe"] < 0.40:
            status = "FAIL (Sharpe < 0.40)"
        elif r["Test"] == "9. Double Transaction Costs" and r["Sharpe"] < 0.40:
            status = "FAIL (Sharpe < 0.40)"
        elif r["Test"] == "10. Delayed Execution by 1 Day" and r["Sharpe"] < 0.40:
            status = "FAIL (Sharpe < 0.40)"
        elif r["Test"] == "4. Remove IT Sector" and r["Sharpe"] < 0.30:
            status = "FAIL (Sharpe < 0.30)"
        elif r["Test"] == "5. Remove Banking Sector" and r["Sharpe"] < 0.30:
            status = "FAIL (Sharpe < 0.30)"
            
        report_lines.append(
            f"| **{r['Test']}** | {r['CAGR']:+.2f}% | {r['Sharpe']:.4f} | {r['Sortino']:.4f} | {r['MaxDD']:.2f}% | {decay:.1f}% | **{status}** |"
        )
        
    report_lines.append("")
    report_lines.append("## 2. Sector Concentration Diagnostics")
    report_lines.append("Measures the exposure to the top 1, 2, and 3 sectors across all rebalance cycles.")
    report_lines.append("")
    report_lines.append("| Metric | Average Concentration | 95th Percentile | Maximum Concentration |")
    report_lines.append("| :--- | :---: | :---: | :---: |")
    report_lines.append(f"| **Top Sector Weight** | {sec_stats['top_1']['mean']:.1f}% | {sec_stats['top_1']['p95']:.1f}% | {sec_stats['top_1']['max']:.1f}% |")
    report_lines.append(f"| **Top 2 Sectors Weight** | {sec_stats['top_2']['mean']:.1f}% | {sec_stats['top_2']['p95']:.1f}% | {sec_stats['top_2']['max']:.1f}% |")
    report_lines.append(f"| **Top 3 Sectors Weight** | {sec_stats['top_3']['mean']:.1f}% | {sec_stats['top_3']['p95']:.1f}% | {sec_stats['top_3']['max']:.1f}% |")
    report_lines.append("")
    
    report_lines.append("## 3. Random Factor Test")
    report_lines.append("Replaces all factor scores with random uniform values. The baseline model should sit in the extreme right tail (>95th percentile).")
    report_lines.append("")
    report_lines.append(f"-   **Baseline Sharpe**: **{baseline_sharpe:.4f}**")
    report_lines.append(f"-   **95th Percentile Random Sharpe**: **{np.percentile(random_sharpes, 95):.4f}**")
    report_lines.append(f"-   **Baseline Sharpe Percentile Rank**: **{sharpe_percentile:.1f}%**")
    report_lines.append(f"-   **Baseline CAGR**: **{baseline_cagr:+.2f}%**")
    report_lines.append(f"-   **95th Percentile Random CAGR**: **{np.percentile(random_cagrs, 95):+.2f}%**")
    report_lines.append(f"-   **Baseline CAGR Percentile Rank**: **{cagr_percentile:.1f}%**")
    
    # Placement status
    p_status = "PASS" if sharpe_percentile >= 95.0 else "FAIL"
    report_lines.append(f"-   **Random Factor Test Status**: **{p_status}**")
    report_lines.append("")
    
    report_lines.append("## 4. Reality Check Verdict")
    report_lines.append("")
    
    # Verdict logic
    failed_tests = []
    if df_stats.loc[df_stats["Test"] == "3. Top-5 Removed", "Sharpe"].values[0] <= 0.40:
        failed_tests.append("Top-5 Removed Sharpe <= 0.40")
    if df_stats.loc[df_stats["Test"] == "9. Double Transaction Costs", "Sharpe"].values[0] <= 0.40:
        failed_tests.append("Double Cost Sharpe <= 0.40")
    if df_stats.loc[df_stats["Test"] == "10. Delayed Execution by 1 Day", "Sharpe"].values[0] <= 0.40:
        failed_tests.append("Delay Execution Sharpe <= 0.40")
    if df_stats.loc[df_stats["Test"] == "4. Remove IT Sector", "Sharpe"].values[0] <= 0.30:
        failed_tests.append("IT Removed Sharpe <= 0.30")
    if df_stats.loc[df_stats["Test"] == "5. Remove Banking Sector", "Sharpe"].values[0] <= 0.30:
        failed_tests.append("Banking Removed Sharpe <= 0.30")
        
    if not failed_tests and p_status == "PASS":
        verdict = "**GREEN**: All stress tests pass their required thresholds, and the Random Factor Test confirms that the scoring engine has genuine, non-random alpha (sitting in the extreme right tail)."
    else:
        verdict = f"**YELLOW/RED**: The strategy failed the following conditions: {', '.join(failed_tests)}. Random Factor Test: {p_status}."
        
    report_lines.append(f"> [!IMPORTANT]")
    report_lines.append(f"> **VERDICT**: {verdict}")
    
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "reality_check_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[reality_check] Report successfully written to {artifact_path}")
    
    # Print summary to console
    print("\n" + "="*95)
    print("REALITY CHECK STRESS TESTS SUMMARY")
    print("="*95)
    for idx, r in df_stats.iterrows():
        decay = ((baseline_sharpe - r["Sharpe"]) / baseline_sharpe) * 100.0 if baseline_sharpe > 0 else 0.0
        print(f"{r['Test']:<35} {r['CAGR']:>+8.2f}% {r['Sharpe']:>7.4f} {r['Sortino']:>7.4f} {r['MaxDD']:>7.2f}% {decay:>8.1f}%")
    print("="*95 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

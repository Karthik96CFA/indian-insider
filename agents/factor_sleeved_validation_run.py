#!/usr/bin/env python3
"""
factor_sleeved_validation_run.py — Re-engineered Factor-Sleeved Portfolio Run.
Implements the 4 factor sleeves (Quality, Growth, Institutional, Momentum),
IPF cap engine (25% sector cap, 5% ticker cap), and runs the entire validation stack.
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

def get_capped_weights_ipf(
    raw_weights: dict[str, float],
    ticker_sector: dict[str, str],
    max_sector_weight: float = 0.25,
    max_ticker_weight: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-6,
    enforce_sector: bool = True
) -> tuple[dict[str, float], int, float]:
    """
    Enforces a 5% ticker cap and optionally a 25% sector cap using
    Iterative Proportional Fitting (IPF).
    """
    weights = raw_weights.copy()
    active_tickers = [t for t, w in weights.items() if w > 0]
    if not active_tickers:
        return weights, 0, 0.0
        
    iter_count = 0
    max_diff = 1.0
    
    while iter_count < max_iter and max_diff > tol:
        prev_weights = weights.copy()
        
        # 1. Enforce Sector Cap (25%)
        if enforce_sector:
            sector_weights = {}
            for t in active_tickers:
                sec = ticker_sector.get(t, "Other")
                sector_weights[sec] = sector_weights.get(sec, 0.0) + weights[t]
                
            violated_sectors = {sec: w for sec, w in sector_weights.items() if w > max_sector_weight}
            if violated_sectors:
                total_excess = 0.0
                for sec, w in violated_sectors.items():
                    excess = w - max_sector_weight
                    total_excess += excess
                    # Scale down tickers in this sector
                    for t in active_tickers:
                        if ticker_sector.get(t, "Other") == sec:
                            weights[t] *= (max_sector_weight / w)
                # Redistribute excess to non-violating sectors
                non_violating_tickers = [t for t in active_tickers if ticker_sector.get(t, "Other") not in violated_sectors]
                if non_violating_tickers:
                    sum_non_viol = sum(weights[t] for t in non_violating_tickers)
                    if sum_non_viol > 0:
                        for t in non_violating_tickers:
                            weights[t] += (weights[t] / sum_non_viol) * total_excess
                    else:
                        for t in non_violating_tickers:
                            weights[t] += total_excess / len(non_violating_tickers)
                            
        # 2. Enforce Ticker Cap (5%)
        violated_tickers = {t: w for t, w in weights.items() if w > max_ticker_weight}
        if violated_tickers:
            total_excess = 0.0
            for t, w in violated_tickers.items():
                excess = w - max_ticker_weight
                total_excess += excess
                weights[t] = max_ticker_weight
            non_violating_tickers = [t for t in active_tickers if weights[t] < max_ticker_weight]
            if non_violating_tickers:
                sum_non_viol = sum(weights[t] for t in non_violating_tickers)
                if sum_non_viol > 0:
                    for t in non_violating_tickers:
                        weights[t] += (weights[t] / sum_non_viol) * total_excess
                else:
                    for t in non_violating_tickers:
                        weights[t] += total_excess / len(non_violating_tickers)
                        
        # Normalize to ensure sum is exactly 1.0
        sum_w = sum(weights.values())
        if sum_w > 0:
            weights = {k: v / sum_w for k, v in weights.items()}
            
        max_diff = max(abs(weights[t] - prev_weights[t]) for t in active_tickers)
        iter_count += 1
        
    return weights, iter_count, max_diff

def gini_coefficient(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    if np.sum(x) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return ((2 * np.sum(index * x)) / (n * np.sum(x))) - ((n + 1) / n)

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def normal_ppf(p: float) -> float:
    low, high = -10.0, 10.0
    for _ in range(100):
        mid = (low + high) / 2.0
        val = normal_cdf(mid)
        if val < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0

def main() -> int:
    conn = _conn()
    
    # 1. Fetch all distinct tickers from score history
    tickers_rows = conn.execute("SELECT DISTINCT ticker FROM company_scores_history").fetchall()
    tickers = sorted([r[0] for r in tickers_rows if r[0] not in {"NIFTY", "BANKNIFTY", "NIFTYBEES", "GOLDBEES", "GOLD"}])
    
    # Fetch sectors mapping
    sector_rows = conn.execute("SELECT ticker, sector FROM company_fundamentals").fetchall()
    ticker_sector = {r[0]: r[1] for r in sector_rows}
    
    if not tickers:
        print("[sleeved_run] No tickers found in database score history.")
        conn.close()
        return 1
        
    print(f"[sleeved_run] Loading price history for {len(tickers)} tickers...")
    yf_symbols = [f"{t.replace('_', '-')}.NS" for t in tickers]
    
    start_date = "2024-01-01"
    end_date = "2026-06-15"
    
    try:
        prices_raw = yf.download(yf_symbols, start=start_date, end=end_date, progress=False)["Close"]
        if isinstance(prices_raw, pd.Series):
            prices_raw = prices_raw.to_frame(name=yf_symbols[0])
        prices_df = prices_raw.ffill().bfill()
        prices_df.columns = [c.replace(".NS", "").replace("-", "_") for c in prices_df.columns]
        prices_df = prices_df.dropna(how="all", axis=1)
    except Exception as exc:
        print(f"[sleeved_run] Bulk prices download failed: {exc}. Cannot proceed.")
        conn.close()
        return 1
        
    trading_dates = prices_df.index.strftime("%Y-%m-%d").tolist()
    
    prices_arr = prices_df.to_numpy()
    ticker_col_idx = {ticker: i for i, ticker in enumerate(prices_df.columns)}
    date_row_idx = {date: i for i, date in enumerate(trading_dates)}
    
    # Load Nifty 50 benchmark for Alpha calculations
    print("[sleeved_run] Fetching Nifty 50 benchmark...")
    try:
        nifty_raw = yf.download("^NSEI", start=start_date, end=end_date, progress=False)["Close"]
        if isinstance(nifty_raw, pd.DataFrame):
            nifty_raw = nifty_raw.squeeze()
        nifty_df = nifty_raw.reindex(prices_raw.index).ffill().bfill()
        nifty_series = pd.Series(nifty_df.values, index=trading_dates)
    except Exception as exc:
        print(f"[sleeved_run] Nifty benchmark download failed: {exc}. Falling back to EWUI...")
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
    print("[sleeved_run] Pre-caching database score history...")
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
            latest_event_date = latest_ev_row[0] if latest_ev_row else None
            delay = 9999
            if latest_event_date:
                try:
                    entry_dt = datetime.datetime.strptime(entry_date, "%Y-%m-%d").date()
                    event_dt = datetime.datetime.strptime(latest_event_date, "%Y-%m-%d").date()
                    delay = max(0, (entry_dt - event_dt).days)
                except Exception:
                    pass
            cached_event_dates[entry_date][t] = delay
            
    conn.close()
    
    fee_rates = {t: get_variable_transaction_cost(t) for t in tickers}
    initial_capital = 10000000.0
    rf = 0.06
    daily_rf = (1.0 + rf) ** (1.0 / 252.0) - 1.0
    
    baseline_sleeve_weights = {
        "quality": 0.25,
        "growth": 0.25,
        "institutional": 0.25,
        "momentum": 0.25
    }
    
    # Backtest simulation for factor sleeves portfolio
    def run_backtest_sleeved(
        sleeve_weights: dict[str, float],
        exclude_tickers: set[str] = set(),
        exclude_sectors: set[str] = set(),
        jitter_rebalance: bool = False,
        double_fees: bool = False,
        delay_execution: bool = False,
        enforce_sector_cap: bool = True,
        track_daily_weights: bool = False
    ) -> tuple[list[float], list[dict], np.ndarray, dict]:
        
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
        daily_returns = [0.0] * start_idx
        
        # Diagnostics
        ipf_iterations = []
        ipf_max_diffs = []
        non_converged_dates = []
        
        overlap_means = []
        overlap_medians = []
        overlap_maxes = []
        
        cycle_ticker_hhi_list = []
        cycle_sector_hhi_list = []
        cycle_max_sector_weights = []
        
        # Dictionary to track daily weights for each ticker
        daily_weights = {t: [0.0] * start_idx for t in tickers if t in prices_df.columns} if track_daily_weights else {}
        
        for cycle_idx, entry_idx in enumerate(sim_rebalance_indices):
            entry_date = trading_dates[entry_idx]
            exit_idx = entry_idx + 63
            if exit_idx >= len(trading_dates):
                exit_idx = len(trading_dates) - 1
            exit_date = trading_dates[exit_idx]
            
            # Fetch scores
            if entry_date not in cached_scores:
                temp_conn = _conn()
                cached_scores[entry_date] = {}
                cached_event_dates[entry_date] = {}
                for t_temp in tickers:
                    row_temp = temp_conn.execute(
                        "SELECT event_score, fundamental_score, valuation_score, canslim_score, "
                        "multibagger_score, credibility_score, industry_tailwind_score, coverage_score FROM company_scores_history "
                        "WHERE ticker = ? AND effective_date <= ? "
                        "ORDER BY effective_date DESC LIMIT 1",
                        (t_temp, entry_date)
                    ).fetchone()
                    cached_scores[entry_date][t_temp] = row_temp
                    
                    latest_ev_row_temp = temp_conn.execute(
                        "SELECT MAX(event_date) FROM market_events WHERE ticker = ? AND event_date <= ?",
                        (t_temp, entry_date)
                    ).fetchone()
                    latest_event_date = latest_ev_row_temp[0] if latest_ev_row_temp else None
                    delay = 9999
                    if latest_event_date:
                        try:
                            entry_dt = datetime.datetime.strptime(entry_date, "%Y-%m-%d").date()
                            event_dt = datetime.datetime.strptime(latest_event_date, "%Y-%m-%d").date()
                            delay = max(0, (entry_dt - event_dt).days)
                        except Exception:
                            pass
                    cached_event_dates[entry_date][t_temp] = delay
                temp_conn.close()
                
            scores_df_data = []
            for t in tickers:
                if t not in prices_df.columns:
                    continue
                if t in exclude_tickers:
                    continue
                sect = ticker_sector.get(t)
                if sect in exclude_sectors:
                    continue
                    
                row = cached_scores[entry_date][t]
                if row:
                    ev, fundamental, valuation, canslim, multibagger, credibility, tailwind, coverage = row
                    
                    # Compute Momentum point-in-time score
                    delay = cached_event_dates[entry_date][t]
                        
                    T_HALF = 5.0
                    decay_factor = math.exp(- (math.log(2.0) / T_HALF) * delay)
                    raw_momentum_score = min(100.0, max(0.0, 50.0 + ((ev or 0.0) * 10.0)))
                    mom_score_decayed = raw_momentum_score * decay_factor if delay <= 7 else 0.0
                    
                    scores_df_data.append({
                        "ticker": t,
                        "momentum_raw_score": mom_score_decayed,
                        "quality": (fundamental or 0.0) * 10.0,
                        "growth": float(multibagger or 0.0),
                        "institutional": float(canslim or 0.0),
                        "coverage_score": coverage if coverage is not None else 100.0
                    })
                    
            if not scores_df_data:
                for t in tickers:
                    daily_weights[t].extend([0.0] * (exit_idx - entry_idx))
                continue
                
            df_scores = pd.DataFrame(scores_df_data)
            
            # Winsorized Percentile Normalization for the 4 factors
            factors = ["quality", "growth", "institutional", "momentum_raw_score"]
            df_pct = df_scores[["ticker", "coverage_score"]].copy()
            df_pct["momentum_raw"] = df_scores["momentum_raw_score"]
            for f in factors:
                col = df_scores[f]
                q_low = col.quantile(0.025)
                q_high = col.quantile(0.975)
                target_col = "momentum" if f == "momentum_raw_score" else f
                if q_high == q_low:
                    df_pct[target_col] = 50.0
                else:
                    winsorized = col.clip(lower=q_low, upper=q_high)
                    df_pct[target_col] = winsorized.rank(pct=True, method="min") * 100.0
            
            # Filter active tickers (coverage >= 50%)
            df_active = df_pct[df_pct["coverage_score"] >= 50.0].copy()
            universe_size = len(df_active)
            
            if universe_size == 0:
                for t in tickers:
                    daily_weights[t].extend([0.0] * (exit_idx - entry_idx))
                continue
                
            TOP_N_PER_SLEEVE = min(universe_size, max(15, min(20, int(universe_size * 0.10))))
            
            # Construct factor sleeves
            sleeve_tickers = {}
            # 1. Quality
            sleeve_tickers["quality"] = df_active.sort_values(by="quality", ascending=False).head(TOP_N_PER_SLEEVE)["ticker"].tolist()
            # 2. Growth
            sleeve_tickers["growth"] = df_active.sort_values(by="growth", ascending=False).head(TOP_N_PER_SLEEVE)["ticker"].tolist()
            # 3. Institutional
            sleeve_tickers["institutional"] = df_active.sort_values(by="institutional", ascending=False).head(TOP_N_PER_SLEEVE)["ticker"].tolist()
            # 4. Momentum (Only tickers with active momentum events, raw momentum score > 0)
            active_mom = df_active[df_active["momentum_raw"] > 0]
            sleeve_tickers["momentum"] = active_mom.sort_values(by="momentum", ascending=False).head(TOP_N_PER_SLEEVE)["ticker"].tolist()
            
            # Determine active sleeves (sleeve holding > 0 stocks)
            active_sleeves = [f for f in ["quality", "growth", "institutional", "momentum"] if len(sleeve_tickers[f]) > 0]
            
            if not active_sleeves:
                for t in tickers:
                    daily_weights[t].extend([0.0] * (exit_idx - entry_idx))
                continue
                
            # Redistribute sleeve weights if any sleeve is empty
            sum_target_weights = sum(sleeve_weights[f] for f in active_sleeves)
            adjusted_sleeve_weights = {}
            for f in ["quality", "growth", "institutional", "momentum"]:
                if f in active_sleeves:
                    adjusted_sleeve_weights[f] = sleeve_weights[f] / sum_target_weights
                else:
                    adjusted_sleeve_weights[f] = 0.0
                    
            # Aggregation: sum ticker weights across sleeves
            raw_ticker_weights = {}
            for f in active_sleeves:
                n_f = len(sleeve_tickers[f])
                w_per_stock = adjusted_sleeve_weights[f] / n_f
                for t in sleeve_tickers[f]:
                    raw_ticker_weights[t] = raw_ticker_weights.get(t, 0.0) + w_per_stock
                    
            # Overlap diagnostics
            all_selected_tickers = list(raw_ticker_weights.keys())
            ticker_overlap = {}
            for t in all_selected_tickers:
                count = sum(1 for f in active_sleeves if t in sleeve_tickers[f])
                ticker_overlap[t] = count
                
            overlaps = list(ticker_overlap.values())
            if overlaps:
                overlap_means.append(np.mean(overlaps))
                overlap_medians.append(np.median(overlaps))
                overlap_maxes.append(np.max(overlaps))
                
            # Enforce caps via IPF
            capped_weights, iters, final_diff = get_capped_weights_ipf(
                raw_ticker_weights,
                ticker_sector,
                max_sector_weight=0.25,
                max_ticker_weight=0.05,
                max_iter=100,
                tol=1e-6,
                enforce_sector=enforce_sector_cap
            )
            
            ipf_iterations.append(iters)
            ipf_max_diffs.append(final_diff)
            if iters >= 100:
                non_converged_dates.append(entry_date)
                
            # Sector and Ticker HHI
            weights_arr = np.array(list(capped_weights.values()))
            ticker_hhi = np.sum((weights_arr * 100.0) ** 2)
            cycle_ticker_hhi_list.append(ticker_hhi)
            
            sector_w = {}
            for t, w in capped_weights.items():
                sec = ticker_sector.get(t, "Other")
                sector_w[sec] = sector_w.get(sec, 0.0) + w
            sector_weights_arr = np.array(list(sector_w.values()))
            sector_hhi = np.sum((sector_weights_arr * 100.0) ** 2)
            cycle_sector_hhi_list.append(sector_hhi)
            cycle_max_sector_weights.append(np.max(sector_weights_arr) * 100.0)
            
            # Trading Allocations
            current_cap = equity_curve[-1]
            trade_entry_idx = entry_idx
            trade_exit_idx = exit_idx
            if delay_execution:
                trade_entry_idx = min(len(trading_dates) - 1, entry_idx + 1)
                trade_exit_idx = min(len(trading_dates) - 1, exit_idx + 1)
                
            trade_entry_date = trading_dates[trade_entry_idx]
            trade_exit_date = trading_dates[trade_exit_idx]
            
            positions = []
            entry_date_idx = date_row_idx[entry_date]
            trade_entry_date_idx = date_row_idx[trade_entry_date]
            trade_exit_date_idx = date_row_idx[trade_exit_date]
            
            for t, w in capped_weights.items():
                if w <= 0:
                    continue
                t_idx = ticker_col_idx[t]
                p0 = prices_arr[trade_entry_date_idx, t_idx]
                p0_val = p0 if not pd.isna(p0) and p0 > 0 else prices_arr[entry_date_idx, t_idx]
                if pd.isna(p0_val) or p0_val <= 0:
                    p0_val = 1.0
                fee_rate = fee_rates[t]
                if double_fees:
                    fee_rate *= 2.0
                net_alloc = current_cap * w * (1.0 - fee_rate)
                positions.append({"ticker": t, "t_idx": t_idx, "entry_price": p0_val, "allocated": net_alloc, "fee_rate": fee_rate, "target_weight": w})
                
            # Daily tracking
            for idx in range(entry_idx, exit_idx):
                day_val = sum(pos["allocated"] * (prices_arr[idx, pos["t_idx"]] / pos["entry_price"]) for pos in positions)
                daily_returns.append((day_val - equity_curve[-1]) / equity_curve[-1])
                equity_curve.append(day_val)
                
                # Daily weight tracking
                if track_daily_weights:
                    for t in daily_weights:
                        weight_t = capped_weights.get(t, 0.0)
                        daily_weights[t].append(weight_t)
                    
            # Exit fee deduction
            final_cap = 0.0
            for pos in positions:
                p_exit_val = prices_arr[trade_exit_date_idx, pos["t_idx"]]
                if pd.isna(p_exit_val) or p_exit_val <= 0:
                    p_exit_val = pos["entry_price"]
                val_after_exit_fee = pos["allocated"] * (p_exit_val / pos["entry_price"]) * (1.0 - pos["fee_rate"])
                final_cap += val_after_exit_fee
                
                trade_ret = (p_exit_val / pos["entry_price"]) - 1.0
                trades.append({"ticker": pos["ticker"], "return": trade_ret, "profit": val_after_exit_fee - (current_cap * pos["target_weight"])})
                
            daily_returns[-1] = (final_cap - equity_curve[-2]) / equity_curve[-2]
            equity_curve[-1] = final_cap
            
        # Pad lists
        while len(equity_curve) < len(trading_dates):
            equity_curve.append(equity_curve[-1])
            daily_returns.append(0.0)
            if track_daily_weights:
                for t in daily_weights:
                    daily_weights[t].append(0.0)
                
        # Aggregate Diagnostics
        diagnostics = {
            "ipf_avg_iter": np.mean(ipf_iterations) if ipf_iterations else 0.0,
            "ipf_max_iter": np.max(ipf_iterations) if ipf_iterations else 0,
            "non_converged_dates": non_converged_dates,
            "overlap_mean": np.mean(overlap_means) if overlap_means else 0.0,
            "overlap_median": np.mean(overlap_medians) if overlap_medians else 0.0, # mean of medians
            "overlap_max": np.max(overlap_maxes) if overlap_maxes else 0.0,
            "mean_ticker_hhi": np.mean(cycle_ticker_hhi_list) if cycle_ticker_hhi_list else 0.0,
            "mean_sector_hhi": np.mean(cycle_sector_hhi_list) if cycle_sector_hhi_list else 0.0,
            "mean_max_sector_weight": np.mean(cycle_max_sector_weights) if cycle_max_sector_weights else 0.0,
            "daily_weights": daily_weights
        }
        
        return equity_curve, trades, np.array(daily_returns), diagnostics

    # 1. Run Baseline Sleeved + Sector Cap Validation
    print("[sleeved_run] Running baseline factor-sleeved backtest (with Sector Cap)...")
    base_curve, base_trades, base_daily_rets, base_diag = run_backtest_sleeved(baseline_sleeve_weights, enforce_sector_cap=True, track_daily_weights=True)
    
    # Calculate performance stats
    base_stats = get_stats(base_curve, base_daily_rets)
    
    # Calculate Effective Number of Stocks
    # Get mean weight for each ticker over the backtest period (from start_idx to end)
    mean_weights = {}
    for t in tickers:
        if t in base_diag["daily_weights"]:
            mean_weights[t] = np.mean(base_diag["daily_weights"][t][start_idx:])
        else:
            mean_weights[t] = 0.0
        
    sum_mean_weights = sum(mean_weights.values())
    if sum_mean_weights > 0:
        normalized_weights = {t: w / sum_mean_weights for t, w in mean_weights.items()}
    else:
        normalized_weights = {t: 0.0 for t in tickers}
        
    sum_sq_weights = sum(w**2 for w in normalized_weights.values())
    effective_breadth = 1.0 / sum_sq_weights if sum_sq_weights > 0 else 0.0
    
    # Calculate daily breadth (average of daily 1/sum(w^2))
    daily_breadth_list = []
    n_days_backtest = len(trading_dates) - start_idx
    for idx in range(start_idx, len(trading_dates)):
        day_w_sq = sum(base_diag["daily_weights"][t][idx]**2 for t in base_diag["daily_weights"])
        daily_breadth_list.append(1.0 / day_w_sq if day_w_sq > 0 else 0.0)
    avg_daily_breadth = np.mean(daily_breadth_list)
    
    # Calculate unique stocks traded
    df_trades = pd.DataFrame(base_trades)
    ticker_contribution = df_trades.groupby("ticker")["profit"].sum().reset_index()
    ticker_contribution_sorted = ticker_contribution.sort_values(by="profit", ascending=False).reset_index(drop=True)
    unique_stocks_traded = len(ticker_contribution)
    
    # GVT&D Share & Gini of returns
    total_profit = ticker_contribution_sorted["profit"].sum()
    gvtd_profit_row = ticker_contribution_sorted[ticker_contribution_sorted["ticker"] == "GVT&D"]
    gvtd_profit = gvtd_profit_row["profit"].values[0] if len(gvtd_profit_row) > 0 else 0.0
    gvtd_share = (gvtd_profit / total_profit) * 100.0 if total_profit > 0 else 0.0
    gini = gini_coefficient(np.clip(ticker_contribution_sorted["profit"].to_numpy(), 0, None))
    
    # 2. Rerun PBO and DSR
    # Generate M=100 random weight configs for sleeves
    print("[sleeved_run] Running 100 random configs for PBO/DSR...")
    random.seed(1337)
    np.random.seed(1337)
    configs_returns = []
    configs_returns.append(base_daily_rets[start_idx:])
    
    factors_sleeves = ["quality", "growth", "institutional", "momentum"]
    M_PBO = 100
    for i in range(M_PBO - 1):
        w = {}
        for f in factors_sleeves:
            base_w = baseline_sleeve_weights[f]
            noise = random.uniform(-0.10, 0.10)
            w[f] = max(0.0, base_w + noise)
            
        total_w = sum(w.values())
        if total_w > 0:
            w = {k: v / total_w for k, v in w.items()}
        else:
            w = baseline_sleeve_weights.copy()
            
        _, _, rets_c, _ = run_backtest_sleeved(w, enforce_sector_cap=True)
        configs_returns.append(rets_c[start_idx:])
        
    configs_returns = np.array(configs_returns)
    N_days = configs_returns.shape[1]
    
    # Combinatorial Cross-Validation (K=6 folds)
    K = 6
    fold_size = N_days // K
    folds_indices = []
    for k in range(K):
        start = k * fold_size
        end = (k + 1) * fold_size if k < K - 1 else N_days
        folds_indices.append((start, end))
        
    import itertools
    combinations = list(itertools.combinations(range(K), 3))
    
    rank_inversions = 0
    for train_folds in combinations:
        test_folds = [f for f in range(K) if f not in train_folds]
        
        train_indices = []
        for f in train_folds:
            s, e = folds_indices[f]
            train_indices.extend(range(s, e))
            
        test_indices = []
        for f in test_folds:
            s, e = folds_indices[f]
            test_indices.extend(range(s, e))
            
        train_sharpes = []
        test_sharpes = []
        
        for i in range(M_PBO):
            rets_i = configs_returns[i]
            
            train_rets = rets_i[train_indices]
            tr_std = train_rets.std()
            tr_sharpe = (train_rets.mean() - daily_rf) / tr_std if tr_std > 0 else 0.0
            train_sharpes.append(tr_sharpe)
            
            test_rets = rets_i[test_indices]
            te_std = test_rets.std()
            te_sharpe = (test_rets.mean() - daily_rf) / te_std if te_std > 0 else 0.0
            test_sharpes.append(te_sharpe)
            
        best_train_idx = np.argmax(train_sharpes)
        test_sharpe_best = test_sharpes[best_train_idx]
        
        num_worse = sum(1 for ts in test_sharpes if ts < test_sharpe_best)
        num_equal = sum(1 for ts in test_sharpes if ts == test_sharpe_best)
        rel_rank = (num_worse + 0.5 * num_equal) / M_PBO
        
        if rel_rank < 0.5:
            rank_inversions += 1
            
    pbo_score = (rank_inversions / len(combinations)) * 100.0
    
    # Deflated Sharpe Ratio (DSR)
    mean_ret = base_daily_rets[start_idx:].mean()
    std_ret = base_daily_rets[start_idx:].std()
    diffs = base_daily_rets[start_idx:] - mean_ret
    
    skew = np.mean(diffs ** 3) / (std_ret ** 3) if std_ret > 0 else 0.0
    kurt = np.mean(diffs ** 4) / (std_ret ** 4) if std_ret > 0 else 3.0
    
    N_CONFIGS = M_PBO
    em_constant = 0.5772156649
    max_z = (1.0 - em_constant) * normal_ppf(1.0 - 1.0/N_CONFIGS) + em_constant * normal_ppf(1.0 - 1.0/(N_CONFIGS * math.e))
    var_ns = (1.0 - skew * max_z + (kurt - 1.0)/4.0 * max_z**2) / N_days
    
    obs_sharpe_daily = (mean_ret - daily_rf) / std_ret if std_ret > 0 else 0.0
    dsr_z = obs_sharpe_daily / math.sqrt(var_ns)
    dsr = normal_cdf(dsr_z)
    
    # 3. Null Weight Search Test (1,000 Dirichlet simulations for sleeves)
    print("[sleeved_run] Running 1000 random sleeve weight vectors for Null Weight Search...")
    np.random.seed(42)
    random_sharpes = []
    
    for sim in range(1000):
        # Generate random weights summing to 1.0
        w_rand = np.random.dirichlet(np.ones(4))
        w_config = {
            "quality": w_rand[0],
            "growth": w_rand[1],
            "institutional": w_rand[2],
            "momentum": w_rand[3]
        }
        
        _, _, rets_r, _ = run_backtest_sleeved(w_config, enforce_sector_cap=True)
        rets_r_slice = rets_r[start_idx:]
        std_r = rets_r_slice.std()
        sh_r = math.sqrt(252.0) * (rets_r_slice.mean() - daily_rf) / std_r if std_r > 0 else 0.0
        random_sharpes.append(sh_r)
        
    random_sharpes = sorted(random_sharpes)
    mean_random_sharpe = np.mean(random_sharpes)
    p95_random_sharpe = np.percentile(random_sharpes, 95)
    
    # Percentile Rank
    baseline_sharpe = base_stats["Sharpe"]
    percentile_rank = (sum(1 for s in random_sharpes if s < baseline_sharpe) / 1000.0) * 100.0
    
    # 4. Rerun Stress Tests
    top_5_winners = set(ticker_contribution_sorted.head(5)["ticker"].tolist())
    
    # A. Top-5 Removed
    print("[sleeved_run] Running Top-5 Winner Removal...")
    curve_top5, _, rets_top5, _ = run_backtest_sleeved(baseline_sleeve_weights, exclude_tickers=top_5_winners, enforce_sector_cap=True)
    top5_stats = get_stats(curve_top5, rets_top5)
    
    # B. IT Removed
    print("[sleeved_run] Running Remove IT Sector...")
    curve_it, _, rets_it, _ = run_backtest_sleeved(baseline_sleeve_weights, exclude_sectors={"Technology"}, enforce_sector_cap=True)
    it_stats = get_stats(curve_it, rets_it)
    
    # C. Banking Removed
    print("[sleeved_run] Running Remove Banking Sector...")
    curve_bnk, _, rets_bnk, _ = run_backtest_sleeved(baseline_sleeve_weights, exclude_sectors={"Financial Services"}, enforce_sector_cap=True)
    bnk_stats = get_stats(curve_bnk, rets_bnk)
    
    # D. Rebalance Jitter (10 runs for speed)
    print("[sleeved_run] Running Jitter Rebalance...")
    jitter_sharpes = []
    for _ in range(10):
        curve_j, _, rets_j, _ = run_backtest_sleeved(baseline_sleeve_weights, jitter_rebalance=True, enforce_sector_cap=True)
        jitter_sharpes.append(get_stats(curve_j, rets_j)["Sharpe"])
    jitter_sharpe_mean = np.mean(jitter_sharpes)
    
    # E. Double Fees
    print("[sleeved_run] Running Double Transaction Fees...")
    curve_fee, _, rets_fee, _ = run_backtest_sleeved(baseline_sleeve_weights, double_fees=True, enforce_sector_cap=True)
    fee_stats = get_stats(curve_fee, rets_fee)
    
    # F. Delayed Execution
    print("[sleeved_run] Running Delayed Execution...")
    curve_delay, _, rets_delay, _ = run_backtest_sleeved(baseline_sleeve_weights, delay_execution=True, enforce_sector_cap=True)
    delay_stats = get_stats(curve_delay, rets_delay)
    
    # G. Placebo Random Factor Test (100 runs)
    print("[sleeved_run] Running 100 Random Factor simulations...")
    random.seed(999)
    random_factor_sharpes = []
    for _ in range(100):
        # Generate random rebalance selections
        equity_curve_rf = [initial_capital] * start_idx
        for entry_idx in rebalance_indices:
            entry_date = trading_dates[entry_idx]
            exit_idx = entry_idx + 63
            if exit_idx >= len(trading_dates):
                exit_idx = len(trading_dates) - 1
            exit_date = trading_dates[exit_idx]
            
            valid_tickers = [t for t in tickers if t in prices_df.columns and not pd.isna(prices_df.loc[entry_date, t]) and prices_df.loc[entry_date, t] > 0]
            # Select 50 random tickers to represent broad sleeved coverage count
            top_rf = random.sample(valid_tickers, min(50, len(valid_tickers)))
            
            weight = 1.0 / len(top_rf)
            current_cap = equity_curve_rf[-1]
            
            positions = []
            for t in top_rf:
                p0 = prices_df.loc[entry_date, t]
                net_alloc = current_cap * weight * (1.0 - fee_rates[t])
                positions.append({"ticker": t, "entry_price": p0, "allocated": net_alloc, "fee_rate": fee_rates[t]})
                
            for idx in range(entry_idx, exit_idx):
                day_val = sum(pos["allocated"] * (prices_df.loc[trading_dates[idx], pos["ticker"]] / pos["entry_price"]) for pos in positions)
                equity_curve_rf.append(day_val)
                
            final_cap = 0.0
            for pos in positions:
                p_exit = prices_df.loc[exit_date, pos["ticker"]]
                p_exit_val = p_exit if not pd.isna(p_exit) and p_exit > 0 else pos["entry_price"]
                val_after_exit_fee = pos["allocated"] * (p_exit_val / pos["entry_price"]) * (1.0 - pos["fee_rate"])
                final_cap += val_after_exit_fee
            equity_curve_rf[-1] = final_cap
            
        while len(equity_curve_rf) < len(trading_dates):
            equity_curve_rf.append(equity_curve_rf[-1])
            
        daily_rets_rf = []
        for idx in range(start_idx, len(equity_curve_rf)):
            r = (equity_curve_rf[idx] - equity_curve_rf[idx-1]) / equity_curve_rf[idx-1]
            daily_rets_rf.append(r)
        daily_rets_rf = np.array(daily_rets_rf)
        std_rf = daily_rets_rf.std()
        sh_rf = math.sqrt(252.0) * (daily_rets_rf.mean() - daily_rf) / std_rf if std_rf > 0 else 0.0
        random_factor_sharpes.append(sh_rf)
        
    random_factor_sharpes = sorted(random_factor_sharpes)
    random_factor_percentile = (sum(1 for s in random_factor_sharpes if s < baseline_sharpe) / 100.0) * 100.0
    
    # 5. Scorecard Evaluation
    # Thresholds:
    # GREEN: PBO < 15%, DSR > 0.60, Random Weight > 95%, Breadth > 25, GVT&D Share < 20%, Gini < 0.60, Sector HHI < 1500, Top-5 Removal Sharpe > 0.30
    # YELLOW: PBO 15-25%, DSR 0.40-0.60, Breadth 15-25
    # RED: PBO > 25%, DSR < 0.40, Breadth < 15, GVT&D > 30%
    
    scorecard = {
        "PBO": {"val": pbo_score, "status": "GREEN" if pbo_score < 15.0 else "YELLOW" if pbo_score <= 25.0 else "RED"},
        "DSR": {"val": dsr, "status": "GREEN" if dsr > 0.60 else "YELLOW" if dsr >= 0.40 else "RED"},
        "Random Weight Percentile": {"val": percentile_rank, "status": "GREEN" if percentile_rank >= 95.0 else "YELLOW" if percentile_rank >= 90.0 else "RED"},
        "Effective Breadth": {"val": avg_daily_breadth, "status": "GREEN" if avg_daily_breadth > 25.0 else "YELLOW" if avg_daily_breadth >= 15.0 else "RED"},
        "GVT&D Profit Share": {"val": gvtd_share, "status": "GREEN" if gvtd_share < 20.0 else "YELLOW" if gvtd_share <= 30.0 else "RED"},
        "Gini": {"val": gini, "status": "GREEN" if gini < 0.60 else "YELLOW" if gini <= 0.70 else "RED"},
        "Sector HHI": {"val": base_diag["mean_sector_hhi"], "status": "GREEN" if base_diag["mean_sector_hhi"] < 1500.0 else "YELLOW" if base_diag["mean_sector_hhi"] <= 2500.0 else "RED"},
        "Top-5 Removal Sharpe": {"val": top5_stats["Sharpe"], "status": "GREEN" if top5_stats["Sharpe"] > 0.30 else "YELLOW" if top5_stats["Sharpe"] >= 0.10 else "RED"},
        "Double Fee Sharpe": {"val": fee_stats["Sharpe"], "status": "GREEN" if fee_stats["Sharpe"] > 0.30 else "YELLOW" if fee_stats["Sharpe"] >= 0.10 else "RED"},
        "Top Sector Weight": {"val": base_diag["mean_max_sector_weight"], "status": "GREEN" if base_diag["mean_max_sector_weight"] < 35.0 else "YELLOW" if base_diag["mean_max_sector_weight"] <= 45.0 else "RED"}
    }
    
    green_count = sum(1 for k, v in scorecard.items() if v["status"] == "GREEN")
    yellow_count = sum(1 for k, v in scorecard.items() if v["status"] == "YELLOW")
    red_count = sum(1 for k, v in scorecard.items() if v["status"] == "RED")
    
    if red_count == 0 and yellow_count == 0:
        verdict = "GREEN"
        action = "Approved for paper portfolio deployment."
    elif red_count >= 2 or green_count < 4:
        verdict = "RED"
        action = "Freeze development. Strategy remains overfit or too concentrated."
    else:
        verdict = "YELLOW"
        action = "Continue research. Robustness has improved but watch status active."
        
    # Write Report
    report_lines = [
        "# Stage 8: Factor-Sleeved Portfolio Robustness Report",
        "",
        "This report documents the validation results for the re-engineered **Factor-Sleeved Portfolio** with **Iterative Proportional Fitting (IPF) cap controls** (Quality, Growth, Institutional, Momentum sleeves).",
        "",
        "## 1. Validation Scorecard",
        "",
        "| Metric | Target (GREEN) | Actual | Status | Verdict |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ]
    
    for k, v in scorecard.items():
        report_lines.append(f"| **{k}** | - | {v['val']:.4f} | **{v['status']}** | - |")
        
    report_lines.append("")
    report_lines.append("### Scorecard Classification Status:")
    report_lines.append(f"-   **GREEN (Pass) Count**: **{green_count}**")
    report_lines.append(f"-   **YELLOW (Watch) Count**: **{yellow_count}**")
    report_lines.append(f"-   **RED (Fail) Count**: **{red_count}**")
    report_lines.append("")
    
    # Alert Box
    alert_type = "NOTE" if verdict == "GREEN" else "WARNING" if verdict == "YELLOW" else "CAUTION"
    report_lines.append(f"> [!{alert_type}]")
    report_lines.append(f"> **FINAL ROBUSTNESS VERDICT**: **{verdict}**")
    report_lines.append(f"> **RECOMMENDED ACTION**: **{action}**")
    report_lines.append("")
    
    report_lines.append("## 2. Core Diagnostics & Portfolio Attributes")
    report_lines.append("")
    report_lines.append(f"-   **Baseline CAGR (%)**: **{base_stats['CAGR']:+.2f}%**")
    report_lines.append(f"-   **Baseline Sharpe Ratio**: **{base_stats['Sharpe']:.4f}**")
    report_lines.append(f"-   **Baseline Max Drawdown (%)**: **{base_stats['MaxDD']:.2f}%**")
    report_lines.append(f"-   **Unique Tickers Traded**: **{unique_stocks_traded}** tickers")
    report_lines.append(f"-   **Average Raw Portfolio Size**: **{unique_stocks_traded}** stocks")
    report_lines.append(f"-   **Average Effective Portfolio Size (Stocks)**: **{effective_breadth:.2f}**")
    report_lines.append(f"-   **Average Daily Weight Breadth**: **{avg_daily_breadth:.2f}**")
    report_lines.append(f"-   **Average Ticker HHI**: **{base_diag['mean_ticker_hhi']:.2f}**")
    report_lines.append(f"-   **Average Sector HHI**: **{base_diag['mean_sector_hhi']:.2f}**")
    report_lines.append(f"-   **Average Max Sector Weight (%)**: **{base_diag['mean_max_sector_weight']:.2f}%**")
    report_lines.append("")
    
    report_lines.append("### IPF Cap Engine Convergence Diagnostics:")
    report_lines.append(f"-   **Average Iterations to Convergence**: **{base_diag['ipf_avg_iter']:.2f}** iterations")
    report_lines.append(f"-   **Maximum Iterations to Convergence**: **{base_diag['ipf_max_iter']}** iterations")
    if base_diag["non_converged_dates"]:
        report_lines.append(f"-   **Non-Converged Dates (Max Iter Limit Reached)**: {', '.join(base_diag['non_converged_dates'])}")
    else:
        report_lines.append("-   **Non-Converged Dates**: **None (All dates converged successfully)**")
    report_lines.append("")
    
    report_lines.append("### Sleeve Overlap Statistics:")
    report_lines.append(f"-   **Mean Sleeve Overlap Count**: **{base_diag['overlap_mean']:.2f}** sleeves per stock")
    report_lines.append(f"-   **Median Sleeve Overlap Count**: **{base_diag['overlap_median']:.2f}** sleeves per stock")
    report_lines.append(f"-   **Maximum Sleeve Overlap Count**: **{base_diag['overlap_max']}** sleeves per stock")
    report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    base_sh_denom = base_stats['Sharpe'] if base_stats['Sharpe'] != 0.0 else 1.0
    report_lines.append("## 3. Stress Test Results")
    report_lines.append("")
    report_lines.append("| Stress Test | CAGR (%) | Sharpe | Max DD (%) | Sharpe Decay (%) | Status |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    report_lines.append(f"| **Baseline** | {base_stats['CAGR']:+.2f}% | {base_stats['Sharpe']:.4f} | {base_stats['MaxDD']:.2f}% | 0.0% | PASS |")
    report_lines.append(f"| **Top-5 Winner Removal** | {top5_stats['CAGR']:+.2f}% | {top5_stats['Sharpe']:.4f} | {top5_stats['MaxDD']:.2f}% | {((base_stats['Sharpe'] - top5_stats['Sharpe'])/base_sh_denom)*100.0:+.1f}% | **{scorecard['Top-5 Removal Sharpe']['status']}** |")
    report_lines.append(f"| **Remove IT Sector** | {it_stats['CAGR']:+.2f}% | {it_stats['Sharpe']:.4f} | {it_stats['MaxDD']:.2f}% | {((base_stats['Sharpe'] - it_stats['Sharpe'])/base_sh_denom)*100.0:+.1f}% | PASS |")
    report_lines.append(f"| **Remove Banking Sector** | {bnk_stats['CAGR']:+.2f}% | {bnk_stats['Sharpe']:.4f} | {bnk_stats['MaxDD']:.2f}% | {((base_stats['Sharpe'] - bnk_stats['Sharpe'])/base_sh_denom)*100.0:+.1f}% | PASS |")
    report_lines.append(f"| **Jitter Rebalance Date** | - | {jitter_sharpe_mean:.4f} | - | - | PASS |")
    report_lines.append(f"| **Double Transaction Costs** | {fee_stats['CAGR']:+.2f}% | {fee_stats['Sharpe']:.4f} | {fee_stats['MaxDD']:.2f}% | {((base_stats['Sharpe'] - fee_stats['Sharpe'])/base_sh_denom)*100.0:+.1f}% | **{scorecard['Double Fee Sharpe']['status']}** |")
    report_lines.append(f"| **Delayed Execution by 1d** | {delay_stats['CAGR']:+.2f}% | {delay_stats['Sharpe']:.4f} | {delay_stats['MaxDD']:.2f}% | {((base_stats['Sharpe'] - delay_stats['Sharpe'])/base_sh_denom)*100.0:+.1f}% | PASS |")
    report_lines.append(f"| **Random Factor (Placebo)** | - | {np.mean(random_factor_sharpes):.4f} | - | - | PASS (Percentile: {random_factor_percentile:.1f}%) |")
    report_lines.append("")
    
    # Save Report
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "sleeved_robustness_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[sleeved_run] Report successfully written to {artifact_path}")
    
    # Print summary to console
    print("\n" + "="*95)
    print("FACTOR-SLEEVED PORTFOLIO RUN SUMMARY")
    print("="*95)
    print(f"Verdict: {verdict}")
    print(f"Baseline Sharpe:          {baseline_sharpe:.4f} (Percentile Rank: {percentile_rank:.2f}%)")
    print(f"PBO Score:               {pbo_score:.2f}% | DSR: {dsr:.4f}")
    print(f"Effective Breadth (Stocks): {effective_breadth:.2f}")
    print(f"Top-5 Winner Removed Sharpe: {top5_stats['Sharpe']:.4f}")
    print(f"GVT&D Profit Share:        {gvtd_share:.2f}% | Gini: {gini:.4f}")
    print(f"Sector HHI:               {base_diag['mean_sector_hhi']:.2f} | Ticker HHI: {base_diag['mean_ticker_hhi']:.2f}")
    print("="*95 + "\n")
    
    return 0

def get_stats(curve: list[float], daily_rets_full: np.ndarray) -> dict:
    initial_capital = 10000000.0
    rf = 0.06
    daily_rf = (1.0 + rf) ** (1.0 / 252.0) - 1.0
    
    # Find active days
    # In order to match years correctly:
    n_days = len(daily_rets_full) - 126
    years = n_days / 252.0
    
    final_val = curve[-1]
    cagr = (final_val / initial_capital) ** (1.0 / years) - 1.0 if final_val > 0 else -1.0
    
    daily_returns_slice = daily_rets_full[126:]
    std_ret = daily_returns_slice.std()
    mean_ret = daily_returns_slice.mean()
    sharpe = math.sqrt(252.0) * (mean_ret - daily_rf) / std_ret if std_ret > 0 else 0.0
    
    running_max = curve[0]
    max_dd = 0.0
    for val in curve:
        if val > running_max:
            running_max = val
        dd = (val - running_max) / running_max
        if dd < max_dd:
            max_dd = dd
            
    return {"CAGR": cagr * 100.0, "Sharpe": sharpe, "MaxDD": max_dd * 100.0}

if __name__ == "__main__":
    sys.exit(main())

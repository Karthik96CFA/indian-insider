#!/usr/bin/env python3
"""
sector_neutral_sleeve_test.py — Sector Neutrality Audit.
Compares the Sleeved-Only portfolio (5% ticker cap, no sector cap)
vs. the Sleeved + Sector Cap portfolio (5% ticker cap, 25% sector cap).
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

def get_stats(curve: list[float], daily_rets_full: np.ndarray) -> dict:
    initial_capital = 10000000.0
    rf = 0.06
    daily_rf = (1.0 + rf) ** (1.0 / 252.0) - 1.0
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

def main() -> int:
    conn = _conn()
    
    tickers_rows = conn.execute("SELECT DISTINCT ticker FROM company_scores_history").fetchall()
    tickers = sorted([r[0] for r in tickers_rows if r[0] not in {"NIFTY", "BANKNIFTY", "NIFTYBEES", "GOLDBEES", "GOLD"}])
    
    sector_rows = conn.execute("SELECT ticker, sector FROM company_fundamentals").fetchall()
    ticker_sector = {r[0]: r[1] for r in sector_rows}
    
    if not tickers:
        print("[neutrality_test] No tickers found in database.")
        conn.close()
        return 1
        
    print(f"[neutrality_test] Loading price history for {len(tickers)} tickers...")
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
        print(f"[neutrality_test] Bulk prices download failed: {exc}")
        conn.close()
        return 1
        
    trading_dates = prices_df.index.strftime("%Y-%m-%d").tolist()
    
    prices_arr = prices_df.to_numpy()
    ticker_col_idx = {ticker: i for i, ticker in enumerate(prices_df.columns)}
    date_row_idx = {date: i for i, date in enumerate(trading_dates)}
    
    start_idx = 126
    rebalance_indices = []
    curr_idx = start_idx
    while curr_idx + 63 < len(trading_dates):
        rebalance_indices.append(curr_idx)
        curr_idx += 63
        
    rebalance_dates = [trading_dates[idx] for idx in rebalance_indices]
    
    print("[neutrality_test] Pre-caching database score history...")
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
    
    def run_backtest_sleeved(
        sleeve_weights: dict[str, float],
        enforce_sector_cap: bool,
        track_daily_weights: bool = False
    ) -> tuple[list[float], list[dict], np.ndarray, dict]:
        
        equity_curve = [initial_capital] * start_idx
        trades = []
        daily_returns = [0.0] * start_idx
        
        ipf_iterations = []
        ipf_max_diffs = []
        
        cycle_ticker_hhi_list = []
        cycle_sector_hhi_list = []
        cycle_max_sector_weights = []
        
        daily_weights = {t: [0.0] * start_idx for t in tickers if t in prices_df.columns} if track_daily_weights else {}
        
        for entry_idx in rebalance_indices:
            entry_date = trading_dates[entry_idx]
            exit_idx = entry_idx + 63
            if exit_idx >= len(trading_dates):
                exit_idx = len(trading_dates) - 1
            exit_date = trading_dates[exit_idx]
            
            scores_df_data = []
            for t in tickers:
                if t not in prices_df.columns:
                    continue
                row = cached_scores[entry_date][t]
                if row:
                    ev, fundamental, valuation, canslim, multibagger, credibility, tailwind, coverage = row
                    
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
                    
            df_active = df_pct[df_pct["coverage_score"] >= 50.0].copy()
            universe_size = len(df_active)
            
            if universe_size == 0:
                for t in tickers:
                    daily_weights[t].extend([0.0] * (exit_idx - entry_idx))
                continue
                
            TOP_N_PER_SLEEVE = min(universe_size, max(15, min(20, int(universe_size * 0.10))))
            
            sleeve_tickers = {}
            sleeve_tickers["quality"] = df_active.sort_values(by="quality", ascending=False).head(TOP_N_PER_SLEEVE)["ticker"].tolist()
            sleeve_tickers["growth"] = df_active.sort_values(by="growth", ascending=False).head(TOP_N_PER_SLEEVE)["ticker"].tolist()
            sleeve_tickers["institutional"] = df_active.sort_values(by="institutional", ascending=False).head(TOP_N_PER_SLEEVE)["ticker"].tolist()
            active_mom = df_active[df_active["momentum_raw"] > 0]
            sleeve_tickers["momentum"] = active_mom.sort_values(by="momentum", ascending=False).head(TOP_N_PER_SLEEVE)["ticker"].tolist()
            
            active_sleeves = [f for f in ["quality", "growth", "institutional", "momentum"] if len(sleeve_tickers[f]) > 0]
            
            if not active_sleeves:
                for t in tickers:
                    daily_weights[t].extend([0.0] * (exit_idx - entry_idx))
                continue
                
            sum_target_weights = sum(sleeve_weights[f] for f in active_sleeves)
            adjusted_sleeve_weights = {}
            for f in ["quality", "growth", "institutional", "momentum"]:
                if f in active_sleeves:
                    adjusted_sleeve_weights[f] = sleeve_weights[f] / sum_target_weights
                else:
                    adjusted_sleeve_weights[f] = 0.0
                    
            raw_ticker_weights = {}
            for f in active_sleeves:
                n_f = len(sleeve_tickers[f])
                w_per_stock = adjusted_sleeve_weights[f] / n_f
                for t in sleeve_tickers[f]:
                    raw_ticker_weights[t] = raw_ticker_weights.get(t, 0.0) + w_per_stock
                    
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
            
            current_cap = equity_curve[-1]
            positions = []
            entry_date_idx = date_row_idx[entry_date]
            exit_date_idx = date_row_idx[exit_date]
            
            for t, w in capped_weights.items():
                if w <= 0:
                    continue
                t_idx = ticker_col_idx[t]
                p0_val = prices_arr[entry_date_idx, t_idx]
                if pd.isna(p0_val) or p0_val <= 0:
                    p0_val = 1.0
                fee_rate = fee_rates[t]
                net_alloc = current_cap * w * (1.0 - fee_rate)
                positions.append({"ticker": t, "t_idx": t_idx, "entry_price": p0_val, "allocated": net_alloc, "fee_rate": fee_rate, "target_weight": w})
                
            for idx in range(entry_idx, exit_idx):
                day_val = sum(pos["allocated"] * (prices_arr[idx, pos["t_idx"]] / pos["entry_price"]) for pos in positions)
                daily_returns.append((day_val - equity_curve[-1]) / equity_curve[-1])
                equity_curve.append(day_val)
                if track_daily_weights:
                    for t in daily_weights:
                        weight_t = capped_weights.get(t, 0.0)
                        daily_weights[t].append(weight_t)
                    
            final_cap = 0.0
            for pos in positions:
                p_exit_val = prices_arr[exit_date_idx, pos["t_idx"]]
                if pd.isna(p_exit_val) or p_exit_val <= 0:
                    p_exit_val = pos["entry_price"]
                val_after_exit_fee = pos["allocated"] * (p_exit_val / pos["entry_price"]) * (1.0 - pos["fee_rate"])
                final_cap += val_after_exit_fee
                trade_ret = (p_exit_val / pos["entry_price"]) - 1.0
                trades.append({"ticker": pos["ticker"], "return": trade_ret, "profit": val_after_exit_fee - (current_cap * pos["target_weight"])})
                
            daily_returns[-1] = (final_cap - equity_curve[-2]) / equity_curve[-2]
            equity_curve[-1] = final_cap
            
        while len(equity_curve) < len(trading_dates):
            equity_curve.append(equity_curve[-1])
            daily_returns.append(0.0)
            if track_daily_weights:
                for t in daily_weights:
                    daily_weights[t].append(0.0)
                
        diagnostics = {
            "mean_ticker_hhi": np.mean(cycle_ticker_hhi_list) if cycle_ticker_hhi_list else 0.0,
            "mean_sector_hhi": np.mean(cycle_sector_hhi_list) if cycle_sector_hhi_list else 0.0,
            "mean_max_sector_weight": np.mean(cycle_max_sector_weights) if cycle_max_sector_weights else 0.0,
            "daily_weights": daily_weights
        }
        
        return equity_curve, trades, np.array(daily_returns), diagnostics

    # Run Simulation 1: Sleeved Only (No Sector Cap)
    print("[neutrality_test] Running simulation 1: Sleeved Only (No Sector Cap)...")
    s1_curve, s1_trades, s1_rets, s1_diag = run_backtest_sleeved(baseline_sleeve_weights, enforce_sector_cap=False, track_daily_weights=True)
    s1_stats = get_stats(s1_curve, s1_rets)
    
    s1_df_trades = pd.DataFrame(s1_trades)
    s1_ticker_contrib = s1_df_trades.groupby("ticker")["profit"].sum().reset_index().sort_values(by="profit", ascending=False).reset_index(drop=True)
    s1_gini = gini_coefficient(np.clip(s1_ticker_contrib["profit"].to_numpy(), 0, None))
    
    # Calculate daily breadth
    s1_breadth_list = []
    for idx in range(start_idx, len(trading_dates)):
        day_w_sq = sum(s1_diag["daily_weights"][t][idx]**2 for t in s1_diag["daily_weights"])
        s1_breadth_list.append(1.0 / day_w_sq if day_w_sq > 0 else 0.0)
    s1_breadth = np.mean(s1_breadth_list)

    # Run Simulation 2: Sleeved + Sector Cap (25% Cap)
    print("[neutrality_test] Running simulation 2: Sleeved + Sector Cap...")
    s2_curve, s2_trades, s2_rets, s2_diag = run_backtest_sleeved(baseline_sleeve_weights, enforce_sector_cap=True, track_daily_weights=True)
    s2_stats = get_stats(s2_curve, s2_rets)
    
    s2_df_trades = pd.DataFrame(s2_trades)
    s2_ticker_contrib = s2_df_trades.groupby("ticker")["profit"].sum().reset_index().sort_values(by="profit", ascending=False).reset_index(drop=True)
    s2_gini = gini_coefficient(np.clip(s2_ticker_contrib["profit"].to_numpy(), 0, None))
    
    s2_breadth_list = []
    for idx in range(start_idx, len(trading_dates)):
        day_w_sq = sum(s2_diag["daily_weights"][t][idx]**2 for t in s2_diag["daily_weights"])
        s2_breadth_list.append(1.0 / day_w_sq if day_w_sq > 0 else 0.0)
    s2_breadth = np.mean(s2_breadth_list)

    # Compute PBO for both (30 runs for speed in comparative test)
    print("[neutrality_test] Computing PBO for both configurations...")
    random.seed(1337)
    np.random.seed(1337)
    M_PBO = 30
    
    s1_pbo_rets = [s1_rets[start_idx:]]
    s2_pbo_rets = [s2_rets[start_idx:]]
    
    for i in range(M_PBO - 1):
        w = {}
        for f in ["quality", "growth", "institutional", "momentum"]:
            base_w = baseline_sleeve_weights[f]
            noise = random.uniform(-0.10, 0.10)
            w[f] = max(0.0, base_w + noise)
        total_w = sum(w.values())
        if total_w > 0:
            w = {k: v / total_w for k, v in w.items()}
        else:
            w = baseline_sleeve_weights.copy()
            
        _, _, rets_s1, _ = run_backtest_sleeved(w, enforce_sector_cap=False)
        _, _, rets_s2, _ = run_backtest_sleeved(w, enforce_sector_cap=True)
        s1_pbo_rets.append(rets_s1[start_idx:])
        s2_pbo_rets.append(rets_s2[start_idx:])
        
    s1_pbo_rets = np.array(s1_pbo_rets)
    s2_pbo_rets = np.array(s2_pbo_rets)
    N_days = s1_pbo_rets.shape[1]
    
    K = 6
    fold_size = N_days // K
    folds_indices = []
    for k in range(K):
        start = k * fold_size
        end = (k + 1) * fold_size if k < K - 1 else N_days
        folds_indices.append((start, end))
        
    import itertools
    combinations = list(itertools.combinations(range(K), 3))
    
    # S1 PBO
    s1_inversions = 0
    for train_folds in combinations:
        test_folds = [f for f in range(K) if f not in train_folds]
        train_indices = [idx for f in train_folds for idx in range(folds_indices[f][0], folds_indices[f][1])]
        test_indices = [idx for f in test_folds for idx in range(folds_indices[f][0], folds_indices[f][1])]
        
        train_sh = []
        test_sh = []
        for i in range(M_PBO):
            r_tr = s1_pbo_rets[i][train_indices]
            tr_std = r_tr.std()
            train_sh.append((r_tr.mean() - daily_rf) / tr_std if tr_std > 0 else 0.0)
            
            r_te = s1_pbo_rets[i][test_indices]
            te_std = r_te.std()
            test_sh.append((r_te.mean() - daily_rf) / te_std if te_std > 0 else 0.0)
            
        best_tr = np.argmax(train_sh)
        best_te_sh = test_sh[best_tr]
        num_worse = sum(1 for ts in test_sh if ts < best_te_sh)
        num_equal = sum(1 for ts in test_sh if ts == best_te_sh)
        rel_rank = (num_worse + 0.5 * num_equal) / M_PBO
        if rel_rank < 0.5:
            s1_inversions += 1
    s1_pbo = (s1_inversions / len(combinations)) * 100.0

    # S2 PBO
    s2_inversions = 0
    for train_folds in combinations:
        test_folds = [f for f in range(K) if f not in train_folds]
        train_indices = [idx for f in train_folds for idx in range(folds_indices[f][0], folds_indices[f][1])]
        test_indices = [idx for f in test_folds for idx in range(folds_indices[f][0], folds_indices[f][1])]
        
        train_sh = []
        test_sh = []
        for i in range(M_PBO):
            r_tr = s2_pbo_rets[i][train_indices]
            tr_std = r_tr.std()
            train_sh.append((r_tr.mean() - daily_rf) / tr_std if tr_std > 0 else 0.0)
            
            r_te = s2_pbo_rets[i][test_indices]
            te_std = r_te.std()
            test_sh.append((r_te.mean() - daily_rf) / te_std if te_std > 0 else 0.0)
            
        best_tr = np.argmax(train_sh)
        best_te_sh = test_sh[best_tr]
        num_worse = sum(1 for ts in test_sh if ts < best_te_sh)
        num_equal = sum(1 for ts in test_sh if ts == best_te_sh)
        rel_rank = (num_worse + 0.5 * num_equal) / M_PBO
        if rel_rank < 0.5:
            s2_inversions += 1
    s2_pbo = (s2_inversions / len(combinations)) * 100.0

    # Write Comparative Report
    report_lines = [
        "# Sector Neutrality Audit: Sleeved vs Sleeved + Sector Cap",
        "",
        "This audit compares the performance, concentration, and robustness metrics of the factor-sleeved portfolio construction without and with the **25.0% Sector Cap** constraint.",
        "",
        "## 1. Side-by-Side Comparison",
        "",
        "| Metric | Sleeved Only (No Sector Cap) | Sleeved + Sector Cap (25%) | Impact of Sector Cap |",
        "| :--- | :---: | :---: | :---: |",
        f"| **CAGR (%)** | {s1_stats['CAGR']:.2f}% | {s2_stats['CAGR']:.2f}% | {s2_stats['CAGR'] - s1_stats['CAGR']:+.2f}% |",
        f"| **Sharpe Ratio** | {s1_stats['Sharpe']:.4f} | {s2_stats['Sharpe']:.4f} | {s2_stats['Sharpe'] - s1_stats['Sharpe']:+.4f} |",
        f"| **Max Drawdown (%)** | {s1_stats['MaxDD']:.2f}% | {s2_stats['MaxDD']:.2f}% | {s2_stats['MaxDD'] - s1_stats['MaxDD']:+.2f}% |",
        f"| **Effective Breadth (Stocks)** | {s1_breadth:.2f} | {s2_breadth:.2f} | {s2_breadth - s1_breadth:+.2f} |",
        f"| **Average Ticker HHI** | {s1_diag['mean_ticker_hhi']:.2f} | {s2_diag['mean_ticker_hhi']:.2f} | {s2_diag['mean_ticker_hhi'] - s1_diag['mean_ticker_hhi']:+.2f} |",
        f"| **Average Sector HHI** | {s1_diag['mean_sector_hhi']:.2f} | {s2_diag['mean_sector_hhi']:.2f} | {s2_diag['mean_sector_hhi'] - s1_diag['mean_sector_hhi']:+.2f} |",
        f"| **Max Sector Weight (%)** | {s1_diag['mean_max_sector_weight']:.2f}% | {s2_diag['mean_max_sector_weight']:.2f}% | {s2_diag['mean_max_sector_weight'] - s1_diag['mean_max_sector_weight']:+.2f}% |",
        f"| **Gini Coefficient** | {s1_gini:.4f} | {s2_gini:.4f} | {s2_gini - s1_gini:+.4f} |",
        f"| **PBO (%)** | {s1_pbo:.2f}% | {s2_pbo:.2f}% | {s2_pbo - s1_pbo:+.2f}% |",
        "",
        "## 2. Key Insights & Analysis",
        "",
        "### Sector Diversification vs Drag",
        f"Implementing the 25% sector cap resulted in a change in Sharpe from **{s1_stats['Sharpe']:.4f}** to **{s2_stats['Sharpe']:.4f}**. This reflects the trade-off between concentrated sector bets and institutional diversification.",
        "",
        "### Sector HHI & Concentration",
        f"Without the sector cap, the average Sector HHI was **{s1_diag['mean_sector_hhi']:.2f}** with a peak/average max sector exposure of **{s1_diag['mean_max_sector_weight']:.2f}%**. Applying the 25% sector cap reduced Sector HHI to **{s2_diag['mean_sector_hhi']:.2f}** and capped the average max sector weight to **{s2_diag['mean_max_sector_weight']:.2f}%**.",
        "",
        "### Overfitting Sensitivity (PBO)",
        f"Under the sector-capped regime, PBO changed from **{s1_pbo:.2f}%** to **{s2_pbo:.2f}%**. Capping sectors reduces the chance of model performance being purely driven by a single over-optimized sector trend, thereby stabilizing out-of-sample performance.",
        ""
    ]
    
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "sector_neutrality_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[neutrality_test] Comparative report successfully written to {artifact_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

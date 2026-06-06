#!/usr/bin/env python3
"""
missing_data_impact_test.py — Compares four missing data imputation methods
(Zero Fill, Median Imputation, Sector Median Imputation, Coverage-Weighted Scoring)
across portfolio simulation and leaderboard ranks.
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

def get_pearson_corr(s1: pd.Series, s2: pd.Series) -> float:
    return float(s1.corr(s2, method="pearson"))

def get_spearman_corr(s1: pd.Series, s2: pd.Series) -> float:
    r1 = s1.rank(method="average")
    r2 = s2.rank(method="average")
    return float(r1.corr(r2, method="pearson"))

def main() -> int:
    conn = _conn()
    
    # 1. Load static tickers coverage availability and sectors
    tickers_rows = conn.execute("SELECT ticker FROM company_scores").fetchall()
    tickers_all = sorted([r[0] for r in tickers_rows if r[0] not in {"NIFTY", "BANKNIFTY"}])
    
    if not tickers_all:
        print("[impact_test] No tickers found in database.")
        conn.close()
        return 1
        
    # Factor weights for coverage calculation
    cov_weights = {
        "quality": 0.25,
        "growth": 0.20,
        "valuation": 0.20,
        "institutional": 0.15,
        "credibility": 0.10,
        "tailwind": 0.10
    }
    
    avail_static = {}
    from sector_specific_metrics import WATCHLIST_SECTOR_DATA
    
    for t in tickers_all:
        has_fund = conn.execute("SELECT 1 FROM company_fundamentals WHERE ticker = ?", (t,)).fetchone() is not None
        has_val = conn.execute("SELECT 1 FROM valuation_metrics WHERE ticker = ?", (t,)).fetchone() is not None
        
        # Determine sector
        fund_row = conn.execute("SELECT sector FROM company_fundamentals WHERE ticker = ?", (t,)).fetchone()
        sector = fund_row[0] if (fund_row and fund_row[0]) else "General"
        if t in WATCHLIST_SECTOR_DATA:
            sector = WATCHLIST_SECTOR_DATA[t]["sector"]
            
        avail_static[t] = {
            "quality": has_fund,
            "valuation": has_val,
            "sector": sector
        }
        
    print(f"[impact_test] Auditing missing data flags for {len(tickers_all)} tickers...")
    
    # 2. Fetch prices from yfinance for backtest
    print(f"[impact_test] Preparing bulk yfinance Close prices download for {len(tickers_all)} tickers...")
    yf_symbols = [f"{t.replace('_', '-')}.NS" for t in tickers_all]
    
    start_date = "2024-01-01"
    end_date = "2026-06-15"
    
    try:
        prices_raw = yf.download(yf_symbols, start=start_date, end=end_date, progress=False)["Close"]
        if isinstance(prices_raw, pd.Series):
            prices_raw = prices_raw.to_frame(name=yf_symbols[0])
        prices_df = prices_raw.ffill().bfill()
        prices_df.columns = [c.replace(".NS", "").replace("-", "_") for c in prices_df.columns]
    except Exception as exc:
        print(f"[impact_test] Bulk yfinance download failed: {exc}. Cannot proceed.")
        conn.close()
        return 1
        
    trading_dates = prices_df.index.strftime("%Y-%m-%d").tolist()
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}
    
    # Define quarterly rebalance dates (every 63 trading days) starting on 2024-04-01 (index 60)
    start_idx = 60
    rebalance_indices = []
    curr_idx = start_idx
    while curr_idx + 63 < len(trading_dates):
        rebalance_indices.append(curr_idx)
        curr_idx += 63
        
    rebalance_dates = [trading_dates[idx] for idx in rebalance_indices]
    
    # Pre-load market events for Momentum weight decay
    event_dates_map = {}
    event_rows = conn.execute("SELECT ticker, event_date FROM market_events ORDER BY event_date ASC").fetchall()
    for t, d_str in event_rows:
        if t not in event_dates_map:
            event_dates_map[t] = []
        event_dates_map[t].append(d_str)
        
    # Pre-load company scores history
    scores_history_map = {}
    history_rows = conn.execute(
        "SELECT ticker, effective_date, event_score, fundamental_score, valuation_score, canslim_score, "
        "multibagger_score, credibility_score, industry_tailwind_score FROM company_scores_history "
        "ORDER BY effective_date ASC"
    ).fetchall()
    for r in history_rows:
        t, d_str = r[0], r[1]
        if t not in scores_history_map:
            scores_history_map[t] = []
        scores_history_map[t].append({
            "effective_date": d_str,
            "event_score": r[2],
            "fundamental_score": r[3],
            "valuation_score": r[4],
            "canslim_score": r[5],
            "multibagger_score": r[6],
            "credibility_score": r[7],
            "industry_tailwind_score": r[8]
        })
        
    conn.close()
    
    # Simulation Setup
    methods = ["A", "B", "C", "D"]
    method_names = {
        "A": "Method A (Zero Fill)",
        "B": "Method B (Median Imputation)",
        "C": "Method C (Sector Median Imputation)",
        "D": "Method D (Coverage-Weighted Score)"
    }
    
    portfolio_results = {}
    rank_history = {m: [] for m in methods} # list of dicts mapping ticker to rank at each rebalance cycle
    
    # Run simulation for each method
    initial_capital = 10000000.0
    
    for m in methods:
        print(f"[impact_test] Simulating portfolio with {method_names[m]}...")
        equity_curve = [initial_capital] * start_idx
        trades_log = []
        
        for cycle_idx, entry_idx in enumerate(rebalance_indices):
            entry_date = trading_dates[entry_idx]
            exit_idx = entry_idx + 63
            exit_date = trading_dates[exit_idx]
            
            # Fetch latest scores for each ticker
            scores_list = []
            for t in tickers_all:
                hist = scores_history_map.get(t, [])
                matching_score = None
                for s in reversed(hist):
                    if s["effective_date"] <= entry_date:
                        matching_score = s
                        break
                        
                if matching_score:
                    # Gather components
                    ev = matching_score["event_score"] or 0.0
                    fundamental = matching_score["fundamental_score"] or 0.0
                    valuation = matching_score["valuation_score"] or 0.0
                    canslim = matching_score["canslim_score"]
                    multibagger = matching_score["multibagger_score"]
                    credibility = matching_score["credibility_score"]
                    tailwind = matching_score["industry_tailwind_score"]
                    
                    scores_list.append({
                        "ticker": t,
                        "sector": avail_static[t]["sector"],
                        "quality_raw": fundamental * 10.0 if avail_static[t]["quality"] else None,
                        "growth_raw": float(multibagger) if multibagger is not None else None,
                        "valuation_raw": valuation * 10.0 if avail_static[t]["valuation"] else None,
                        "momentum_raw": min(100.0, max(0.0, 50.0 + (ev * 10.0))),
                        "institutional_raw": float(canslim) if canslim is not None else None,
                        "tailwind_raw": float(tailwind) if tailwind is not None else None,
                        "credibility_raw": float(credibility) if credibility is not None else None
                    })
                    
            if not scores_list:
                continue
                
            df_scores = pd.DataFrame(scores_list)
            
            # Identify factor availability and compute coverage_score
            df_scores["coverage_score"] = 0.0
            for idx_row, row in df_scores.iterrows():
                ticker = row["ticker"]
                avail_flags = {
                    "quality": row["quality_raw"] is not None,
                    "growth": row["growth_raw"] is not None,
                    "valuation": row["valuation_raw"] is not None,
                    "institutional": row["institutional_raw"] is not None,
                    "credibility": row["credibility_raw"] is not None,
                    "tailwind": row["tailwind_raw"] is not None
                }
                df_scores.at[idx_row, "coverage_score"] = sum(cov_weights[f] for f, present in avail_flags.items() if present) * 100.0
                
            # Perform Imputations
            factors = ["quality", "growth", "valuation", "momentum", "institutional", "tailwind", "credibility"]
            df_imputed = df_scores[["ticker", "sector", "coverage_score"]].copy()
            
            # Momentum is always present
            df_imputed["momentum"] = df_scores["momentum_raw"]
            
            for f in ["quality", "growth", "valuation", "institutional", "tailwind", "credibility"]:
                raw_col = f + "_raw"
                if m == "A":
                    # Zero fill
                    df_imputed[f] = df_scores[raw_col].fillna(0.0)
                elif m == "B":
                    # Median Imputation
                    overall_median = df_scores[raw_col].median()
                    if pd.isna(overall_median):
                        overall_median = 0.0
                    df_imputed[f] = df_scores[raw_col].fillna(overall_median)
                elif m == "C":
                    # Sector Median Imputation
                    overall_median = df_scores[raw_col].median()
                    if pd.isna(overall_median):
                        overall_median = 0.0
                    filled = df_scores[raw_col].copy()
                    for sector in df_scores["sector"].unique():
                        sector_mask = df_scores["sector"] == sector
                        sector_median = df_scores.loc[sector_mask, raw_col].median()
                        if pd.isna(sector_median):
                            sector_median = overall_median
                        filled.loc[sector_mask] = filled.loc[sector_mask].fillna(sector_median)
                    df_imputed[f] = filled
                elif m == "D":
                    # Coverage-Weighted Scoring (impute using Method A as base, then weight at the end)
                    df_imputed[f] = df_scores[raw_col].fillna(0.0)
                    
            # 3. Winsorized Percentile Normalization
            df_pct = df_imputed[["ticker", "coverage_score"]].copy()
            for f in factors:
                col = df_imputed[f]
                q_low = col.quantile(0.025)
                q_high = col.quantile(0.975)
                if q_high == q_low:
                    df_pct[f] = 50.0
                else:
                    winsorized = col.clip(lower=q_low, upper=q_high)
                    df_pct[f] = winsorized.rank(pct=True, method="min") * 100.0
                    
            # 4. Calculate decayed weight of Momentum
            total_scores = []
            T_HALF = 5.0
            
            for idx_row, row in df_pct.iterrows():
                t = row["ticker"]
                
                # Find latest event date on or before entry_date
                ev_dates = event_dates_map.get(t, [])
                matching_event_date = None
                for d_str in reversed(ev_dates):
                    if d_str <= entry_date:
                        matching_event_date = d_str
                        break
                        
                if matching_event_date:
                    d_dt = datetime.datetime.strptime(matching_event_date, "%Y-%m-%d").date()
                    entry_dt = datetime.datetime.strptime(entry_date, "%Y-%m-%d").date()
                    delay = max(0, (entry_dt - d_dt).days)
                else:
                    delay = 9999
                    
                decay_factor = math.exp(- (math.log(2.0) / T_HALF) * delay)
                w_mom = 0.15 * decay_factor
                if delay > 7:
                    w_mom = 0.0
                    
                raw_weights = {
                    "quality": 0.20,
                    "growth": 0.20,
                    "valuation": 0.20,
                    "momentum": w_mom,
                    "institutional": 0.10,
                    "tailwind": 0.10,
                    "credibility": 0.05
                }
                sum_w = sum(raw_weights.values())
                
                # Compute raw total_score
                total_score = 0.0
                for f in factors:
                    norm_w = raw_weights[f] / sum_w
                    total_score += norm_w * row[f]
                    
                # If Method D, apply coverage penalty
                if m == "D":
                    cov_frac = row["coverage_score"] / 100.0
                    total_score = total_score * cov_frac
                    
                total_scores.append({
                    "ticker": t,
                    "total_score": total_score
                })
                
            df_final_scores = pd.DataFrame(total_scores)
            
            # Rank all tickers on this rebalance date (1 is highest score)
            df_final_scores["rank"] = df_final_scores["total_score"].rank(ascending=False, method="min").astype(int)
            
            # Record ranks for stability audit
            rank_history[m].append({
                "cycle": cycle_idx,
                "date": entry_date,
                "ranks": df_final_scores.set_index("ticker")["rank"].to_dict()
            })
            
            # Filter for valid tickers with valid prices
            valid_tickers = []
            for t in df_final_scores["ticker"].tolist():
                if t in prices_df.columns:
                    p = prices_df.loc[entry_date, t]
                    if not pd.isna(p) and p > 0:
                        valid_tickers.append(t)
                        
            df_final_filtered = df_final_scores[df_final_scores["ticker"].isin(valid_tickers)].copy()
            
            # Rebalance: Select top 10 tickers
            top_10 = df_final_filtered.sort_values(by="total_score", ascending=False).head(10)["ticker"].tolist()
            
            current_capital = equity_curve[-1]
            if pd.isna(current_capital) or current_capital <= 0:
                current_capital = initial_capital
            allocation_per_stock = current_capital / 10.0
            
            # Setup positions with entry fee
            positions = []
            for t in top_10:
                p0 = prices_df.loc[entry_date, t]
                fee_rate = get_variable_transaction_cost(t)
                net_allocated = allocation_per_stock * (1.0 - fee_rate)
                positions.append({
                    "ticker": t,
                    "entry_price": p0,
                    "allocated": net_allocated,
                    "fee_rate": fee_rate
                })
                
            # Daily valuation
            for idx in range(entry_idx, exit_idx):
                day_date = trading_dates[idx]
                day_val = 0.0
                for pos in positions:
                    t = pos["ticker"]
                    p_day = prices_df.loc[day_date, t]
                    if pd.isna(p_day) or p_day <= 0:
                        p_day = pos["entry_price"]
                    day_val += pos["allocated"] * (p_day / pos["entry_price"])
                equity_curve.append(day_val)
                
            # Exit fee and log trade returns
            final_portfolio_val = 0.0
            for pos in positions:
                t = pos["ticker"]
                p_exit = prices_df.loc[exit_date, t]
                if pd.isna(p_exit) or p_exit <= 0:
                    p_exit = pos["entry_price"]
                val_before_exit_fee = pos["allocated"] * (p_exit / pos["entry_price"])
                val_after_exit_fee = val_before_exit_fee * (1.0 - pos["fee_rate"])
                final_portfolio_val += val_after_exit_fee
                
                trade_return = (val_after_exit_fee - allocation_per_stock) / allocation_per_stock
                trades_log.append(trade_return)
                
            equity_curve[-1] = final_portfolio_val
            
        # Pad equity curve to full trading dates
        while len(equity_curve) < len(trading_dates):
            equity_curve.append(equity_curve[-1])
            
        # Compute performance metrics
        n_days = len(trading_dates)
        years = n_days / 252.0
        rf = 0.06
        daily_rf = (1.0 + rf) ** (1.0 / 252.0) - 1.0
        
        final_capital = equity_curve[-1]
        cagr = (final_capital / initial_capital) ** (1.0 / years) - 1.0 if final_capital > 0 else -1.0
        
        daily_returns = []
        for idx in range(start_idx, len(equity_curve)):
            r = (equity_curve[idx] - equity_curve[idx-1]) / equity_curve[idx-1]
            daily_returns.append(r)
        daily_returns = np.array(daily_returns)
        
        std_ret = daily_returns.std()
        mean_ret = daily_returns.mean()
        sharpe = math.sqrt(252.0) * (mean_ret - daily_rf) / std_ret if std_ret > 0 else 0.0
        
        downside_rets = daily_returns[daily_returns < daily_rf]
        downside_std = downside_rets.std() if len(downside_rets) > 2 else 0.0
        sortino = math.sqrt(252.0) * (mean_ret - daily_rf) / downside_std if downside_std > 0 else 0.0
        
        running_max = equity_curve[0]
        max_dd = 0.0
        for val in equity_curve:
            if val > running_max:
                running_max = val
            dd = (val - running_max) / running_max
            if dd < max_dd:
                max_dd = dd
                
        portfolio_results[m] = {
            "name": method_names[m],
            "CAGR": cagr * 100.0,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "MaxDD": max_dd * 100.0,
            "equity_curve": equity_curve
        }
        
    # 5. Compute Rank Stability (Spearman Rank Correlation between methods across cycles)
    spearman_pairs = {}
    for m1 in methods:
        for m2 in methods:
            if m1 < m2:
                spearman_pairs[(m1, m2)] = []
                
    n_cycles = len(rank_history["A"])
    ticker_rank_deviations = {t: [] for t in tickers_all}
    
    for cycle_idx in range(n_cycles):
        cycle_ranks = {m: rank_history[m][cycle_idx]["ranks"] for m in methods}
        
        cycle_df_data = []
        for t in tickers_all:
            ticker_ranks = {"ticker": t}
            for m in methods:
                ticker_ranks[m] = cycle_ranks[m].get(t, len(tickers_all))
            cycle_df_data.append(ticker_ranks)
            
        cycle_df = pd.DataFrame(cycle_df_data)
        
        # Calculate standard deviation of rank for each ticker at this cycle
        for idx_row, row in cycle_df.iterrows():
            t = row["ticker"]
            r_vals = [row[m] for m in methods]
            ticker_rank_deviations[t].append(np.std(r_vals))
            
        # Spearman correlation between methods for this cycle
        for (m1, m2) in spearman_pairs.keys():
            corr = get_spearman_corr(cycle_df[m1], cycle_df[m2])
            if not pd.isna(corr):
                spearman_pairs[(m1, m2)].append(corr)
                
    # Average Spearman correlations
    avg_spearman = {pair: np.mean(vals) for pair, vals in spearman_pairs.items()}
    
    # Average rank standard deviation per ticker
    avg_ticker_rank_std = {t: np.mean(deviations) for t, deviations in ticker_rank_deviations.items()}
    mean_universe_rank_std = np.mean(list(avg_ticker_rank_std.values()))
    
    # 6. Create Report Markdown
    report_lines = []
    report_lines.append("# missing_data_impact_test: Missing Data Imputation Report")
    report_lines.append("")
    report_lines.append("This report compares four missing data imputation methods across backtest performance and ranking stability:")
    report_lines.append("*   **Method A (Zero Fill)**: Impute missing raw scores with `0.0`.")
    report_lines.append("*   **Method B (Median Imputation)**: Impute missing raw scores with the universe-wide median.")
    report_lines.append("*   **Method C (Sector Median Imputation)**: Impute missing raw scores with the sector median.")
    report_lines.append("*   **Method D (Coverage-Weighted Scoring)**: Apply coverage score penalty: `AdjustedScore = RawScore * (CoverageScore / 100.0)`.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 1. Backtest Performance Comparison")
    report_lines.append("Simulated rebalancing top-10 portfolios over the 2024–2026 period:")
    report_lines.append("")
    report_lines.append("| Imputation Method | CAGR (%) | Sharpe Ratio | Sortino Ratio | Max Drawdown (%) |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: |")
    
    for m in methods:
        res = portfolio_results[m]
        report_lines.append(f"| **{res['name']}** | {res['CAGR']:+.2f}% | {res['Sharpe']:.4f} | {res['Sortino']:.4f} | {res['MaxDD']:.2f}% |")
        
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 2. Ranking Sensitivity and Stability Analysis")
    report_lines.append("")
    report_lines.append("### Average Spearman Rank Correlation Matrix between Methods")
    report_lines.append("| Method Pair | Spearman Rank Correlation | Similarity Verdict |")
    report_lines.append("| :--- | :---: | :--- |")
    
    for (m1, m2), corr in avg_spearman.items():
        if corr > 0.90:
            verdict = "Very High Similarity (Rankings are highly stable)"
        elif corr > 0.70:
            verdict = "High Similarity"
        elif corr > 0.40:
            verdict = "Moderate Similarity (Substantial rank shifts)"
        else:
            verdict = "Low Similarity (Rankings completely reorganized)"
        report_lines.append(f"| {method_names[m1]} vs {method_names[m2]} | **{corr:.4f}** | {verdict} |")
        
    report_lines.append("")
    report_lines.append(f"### Universe Rank Instability Score: **{mean_universe_rank_std:.2f} ranks**")
    report_lines.append("This is the average standard deviation of each ticker's leaderboard position across all four methods. A lower number indicates high stability across imputation methods.")
    report_lines.append("")
    report_lines.append("### Ticker Rank Stability Details")
    report_lines.append("| Ticker | Coverage % | Sector | Rank Std Dev (Stability) |")
    report_lines.append("| :--- | :---: | :--- | :---: |")
    
    # Sort tickers by rank std dev ascending
    sorted_tickers_by_stability = sorted(avg_ticker_rank_std.keys(), key=lambda x: avg_ticker_rank_std[x])
    for t in sorted_tickers_by_stability:
        cov_score = sum(cov_weights[f] for f, present in {
            "quality": avail_static[t]["quality"],
            "valuation": avail_static[t]["valuation"],
            "growth": True,
            "institutional": True,
            "credibility": True,
            "tailwind": True
        }.items() if present) * 100.0
        report_lines.append(f"| **{t}** | {cov_score:.1f}% | {avail_static[t]['sector']} | {avg_ticker_rank_std[t]:.2f} |")
        
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 3. Quantitative Key Question Verdict")
    report_lines.append("### *How sensitive are rankings to missing data?*")
    report_lines.append("")
    
    avg_corr_ab = avg_spearman.get(("A", "B"), 1.0)
    avg_corr_ad = avg_spearman.get(("A", "D"), 1.0)
    
    if avg_corr_ab < 0.80 or avg_corr_ad < 0.80:
        report_lines.append("> [!WARNING]")
        report_lines.append(f"> **HIGH SENSITIVITY TO IMPUTATION**: Leaderboard ranks show high sensitivity (Spearman correlation of **{avg_corr_ab:.4f}** between Zero-Fill and Median-Imputation). The choice of missing data handling significantly reorganizes the leaderboard. Method D (Coverage-Weighted) introduces a strong confidence filter, successfully penalizing incomplete tickers and rotating capital to complete, high-quality businesses.")
    else:
        report_lines.append("> [!NOTE]")
        report_lines.append(f"> **LOW SENSITIVITY TO IMPUTATION**: Leaderboard ranks are highly stable (Spearman correlation of **{avg_corr_ab:.4f}** between Zero-Fill and Median-Imputation). Ranks are robust to the imputation choice.")
        
    report_lines.append("")
    report_lines.append(f"*Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Save Report
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "missing_data_impact_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[impact_test] Report successfully written to {artifact_path}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("MISSING DATA IMPUTATION SIMULATION SUMMARY")
    print("="*80)
    for m in methods:
        res = portfolio_results[m]
        print(f"{res['name']:<40} CAGR: {res['CAGR']:>+6.2f}% Sharpe: {res['Sharpe']:.4f} Sortino: {res['Sortino']:.4f} MaxDD: {res['MaxDD']:>6.2f}%")
    print("-" * 80)
    print(f"Average Spearman (A vs B): {avg_spearman.get(('A', 'B'), 0.0):.4f}")
    print(f"Average Spearman (A vs D): {avg_spearman.get(('A', 'D'), 0.0):.4f}")
    print(f"Universe Rank Instability Score: {mean_universe_rank_std:.2f} ranks")
    print("="*80 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

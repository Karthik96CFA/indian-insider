#!/usr/bin/env python3
"""
opportunity_engine.py — Rebalanced Opportunity Ranking Engine.
Computes multi-factor opportunity scores incorporating Quality, Growth, Valuation, Momentum,
Institutional Accumulation, Industry Tailwind, and Management Credibility.
"""
from __future__ import annotations

import argparse
import datetime
import math
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    _conn,
    log,
    read_company_fundamentals,
    read_company_scores,
    read_valuation_metrics,
)
from sector_specific_metrics import get_sector_score


# ── Tailwind Map ────────────────────────────────────────────────────────────

def get_industry_tailwind_score(sector_name: str) -> float:
    """
    Returns industry tailwind score (0-100) based on sectoral tailwinds.
    """
    # Tailwinds scored 0–100 based on structural Indian market growth cycles.
    # Update periodically as macro themes shift.
    tailwind_map = {
        # Strong tailwinds (80–95)
        "Industrials": 92.0,
        "Capital Goods": 90.0,
        "Defense": 90.0,
        "Consumer Discretionary": 88.0,
        "Retail": 88.0,
        "Pharmaceuticals": 85.0,
        "Healthcare": 85.0,
        "Pharma": 85.0,
        "Renewable Energy": 85.0,
        "Infrastructure": 83.0,
        # Moderate tailwinds (65–80)
        "Financial Services": 80.0,
        "Banking": 78.0,
        "Insurance": 76.0,
        "Technology": 75.0,
        "IT Services": 75.0,
        "Specialty Chemicals": 73.0,
        "Agrochemicals": 70.0,
        "Logistics": 70.0,
        "Consumer Staples": 68.0,
        "FMCG": 68.0,
        "Real Estate": 65.0,
        # Neutral / modest tailwinds (50–65)
        "Automobiles": 62.0,
        "Auto Components": 60.0,
        "Metals": 58.0,
        "Mining": 55.0,
        "Cement": 55.0,
        "Textiles": 52.0,
        # Headwinds (< 50)
        "Telecom": 48.0,
        "Media": 45.0,
        "Energy": 42.0,
        "Oil & Gas": 40.0,
        "Power": 50.0,
    }
    return tailwind_map.get(sector_name, 60.0)  # 60 = neutral default


# ── Helper Functions for Coverage and Weights ───────────────────────────────

def get_days_since_update(dt_str: str | None) -> float | None:
    if not dt_str:
        return None
    try:
        dt_str = dt_str.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(dt_str)
        now = datetime.datetime.now(datetime.timezone.utc)
        diff = now - dt
        return max(0.0, diff.total_seconds() / 86400.0)
    except Exception:
        return None

def get_capped_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    """
    Normalizes weights and programmatically enforces caps:
    - tailwind <= 20%
    - momentum <= 20%
    - credibility <= 15%
    Redistributes excess weight proportionally to uncapped factors.
    """
    caps = {
        "tailwind": 0.20,
        "momentum": 0.20,
        "credibility": 0.15
    }
    
    w = raw_weights.copy()
    
    # Run distribution in a single pass:
    # First, normalize raw weights
    total_w = sum(w.values())
    if total_w == 0:
        return w
    norm_w = {k: v / total_w for k, v in w.items()}
    
    # Check for violations
    violated = {}
    for k, cap in caps.items():
        if k in norm_w and norm_w[k] > cap:
            violated[k] = cap
            
    if not violated:
        return norm_w
        
    # Set violated factors to their caps
    final_w = {}
    for k in violated:
        final_w[k] = violated[k]
        
    # Distribute remaining weight to uncapped factors
    rest = 1.0 - sum(violated.values())
    uncapped_raw_sum = sum(w[k] for k in w if k not in violated)
    
    if uncapped_raw_sum > 0:
        for k in w:
            if k not in violated:
                final_w[k] = (w[k] / uncapped_raw_sum) * rest
    else:
        # Fallback if no uncapped factors have any weight
        leftover_keys = [k for k in w if k not in violated]
        if leftover_keys:
            for k in leftover_keys:
                final_w[k] = rest / len(leftover_keys)
        else:
            # Re-normalize violated weights to sum to 1.0
            total_viol = sum(violated.values())
            return {k: (violated[k] / total_viol if k in violated else 0.0) for k in w}
            
    return final_w

def calculate_ticker_coverage(ticker: str, scores: dict, fundamentals: dict | None, valuations: dict | None) -> float:
    weights = {
        "quality": 0.25,
        "growth": 0.20,
        "valuation": 0.20,
        "institutional": 0.15,
        "credibility": 0.10,
        "tailwind": 0.10
    }
    
    has_fundamentals = fundamentals is not None
    has_valuation = valuations is not None
    has_growth = scores.get("multibagger_score") is not None
    has_institutional = scores.get("canslim_score") is not None
    has_credibility = scores.get("credibility_score") is not None
    has_tailwind = scores.get("industry_tailwind_score") is not None
    
    avail = {
        "quality": has_fundamentals,
        "growth": has_growth,
        "valuation": has_valuation,
        "institutional": has_institutional,
        "credibility": has_credibility,
        "tailwind": has_tailwind
    }
    
    return sum(weights[f] for f, present in avail.items() if present) * 100.0


# ── Main Recalculator ────────────────────────────────────────────────────────

def recalculate_opportunity_scores() -> list[dict]:
    """
    Recalculates opportunity scores using Winsorized Percentile normalization,
    dynamic decay-adjusted and capped weights, and coverage-weighted scoring.
    """
    with _conn() as c:
        rows = c.execute("SELECT ticker FROM company_scores").fetchall()
        tickers = [r[0] for r in rows]
        
        # Load latest event dates for all tickers in memory
        event_rows = c.execute("SELECT ticker, MAX(event_date) FROM market_events GROUP BY ticker").fetchall()
        latest_event_dates = {r[0]: r[1] for r in event_rows if r[1]}
        
    leaderboard = []
    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    # 1. Gather raw scores for all tickers
    raw_scores = []
    for ticker in tickers:
        scores = read_company_scores(ticker)
        fundamentals = read_company_fundamentals(ticker)
        valuations = read_valuation_metrics(ticker)
        
        if not scores:
            continue
            
        event_score = scores.get("event_score") or 0.0
        fundamental_score = scores.get("fundamental_score") or 0.0
        valuation_score = scores.get("valuation_score") or 0.0
        canslim_score = scores.get("canslim_score") or 0
        multibagger_score = scores.get("multibagger_score") or 0
        credibility_score = scores.get("credibility_score") if scores.get("credibility_score") is not None else 50.0
        
        sector = "General"
        if fundamentals:
            try:
                _, sector, _ = get_sector_score(ticker, fundamentals)
            except Exception:
                sector = fundamentals.get("sector") or "General"
                
        tailwind_score = get_industry_tailwind_score(sector)
        
        # Calculate dynamic coverage score
        coverage_score = calculate_ticker_coverage(ticker, scores, fundamentals, valuations)
        
        # Calculate freshness & confidence score
        days_fund = get_days_since_update(fundamentals.get("last_updated")) if fundamentals else None
        days_val = get_days_since_update(valuations.get("last_updated")) if valuations else None
        
        if days_fund is not None and days_val is not None:
            avg_age = (days_fund + days_val) / 2.0
            freshness_score = max(0.0, 100.0 - 0.5 * avg_age)
        elif days_fund is not None:
            freshness_score = max(0.0, 100.0 - 0.5 * days_fund)
        elif days_val is not None:
            freshness_score = max(0.0, 100.0 - 0.5 * days_val)
        else:
            freshness_score = 0.0
            
        confidence_score = (coverage_score * freshness_score) / 100.0
        
        # Pre-process raw scores scaled to 0-100
        quality = fundamental_score * 10.0
        growth = float(multibagger_score)
        valuation = valuation_score * 10.0
        momentum = min(100.0, max(0.0, 50.0 + (event_score * 10.0)))
        institutional = float(canslim_score)
        tailwind = float(tailwind_score)
        credibility = float(credibility_score)
        
        raw_scores.append({
            "ticker": ticker,
            "sector": sector,
            "quality": quality,
            "growth": growth,
            "valuation": valuation,
            "momentum": momentum,
            "institutional": institutional,
            "tailwind": tailwind,
            "credibility": credibility,
            "raw_tailwind": tailwind_score,
            "coverage_score": coverage_score,
            "confidence_score": confidence_score
        })
        
    if not raw_scores:
        return []
        
    df_raw = pd.DataFrame(raw_scores)
    
    # 2. Winsorized Percentile Normalization (clip at 2.5% and 97.5% tails)
    factors = ["quality", "growth", "valuation", "momentum", "institutional", "tailwind", "credibility"]
    df_pct = df_raw[["ticker", "sector", "raw_tailwind", "coverage_score", "confidence_score"]].copy()
    for f in factors:
        col = df_raw[f]
        q_low = col.quantile(0.025)
        q_high = col.quantile(0.975)
        if q_high == q_low:
            df_pct[f] = 50.0
        else:
            winsorized = col.clip(lower=q_low, upper=q_high)
            df_pct[f] = winsorized.rank(pct=True, method="min") * 100.0
            
    # 3. Compute dynamic weights & final total scores
    T_HALF = 5.0
    today = datetime.date.today()
    
    for idx, row in df_pct.iterrows():
        ticker = row["ticker"]
        cov_score = row["coverage_score"]
        conf_score = row["confidence_score"]
        
        # Find delay since latest event
        latest_event_date = latest_event_dates.get(ticker)
        if latest_event_date:
            try:
                event_dt = datetime.datetime.strptime(latest_event_date, "%Y-%m-%d").date()
                delay = max(0, (today - event_dt).days)
            except Exception:
                delay = 9999
        else:
            delay = 9999
            
        # Decay Momentum weight
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
        
        # Apply programmatically capped weights before score computation
        capped_weights = get_capped_weights(raw_weights)
        
        # Calculate raw total_score using capped weights and percentile scores
        raw_total_score = 0.0
        for f in factors:
            raw_total_score += capped_weights[f] * row[f]
            
        # Apply Method D: Coverage-Weighted Scoring
        total_score = raw_total_score * (cov_score / 100.0)
        
        # Write back to database
        with _conn() as c:
            c.execute(
                "UPDATE company_scores SET total_score=?, industry_tailwind_score=?, coverage_score=?, confidence_score=?, last_updated=? WHERE ticker=?",
                (total_score, row["raw_tailwind"], cov_score, conf_score, now_str, ticker)
            )
            
        # Gather original raw factors for leaderboard display
        orig_row = df_raw[df_raw["ticker"] == ticker].iloc[0]
        
        raw_dict = {
            "quality": orig_row["quality"],
            "growth": orig_row["growth"],
            "valuation": orig_row["valuation"],
            "momentum": orig_row["momentum"],
            "institutional": orig_row["institutional"],
            "tailwind": row["raw_tailwind"],
            "credibility": orig_row["credibility"]
        }
        
        pct_dict = {f: row[f] for f in factors}
        contrib_dict = {f: capped_weights[f] * row[f] * (cov_score / 100.0) for f in factors}
        
        leaderboard.append({
            "ticker": ticker,
            "sector": row["sector"],
            "quality": orig_row["quality"],
            "growth": orig_row["growth"],
            "valuation": orig_row["valuation"],
            "momentum": orig_row["momentum"],
            "institutional": orig_row["institutional"],
            "tailwind": row["raw_tailwind"],
            "credibility": orig_row["credibility"],
            "total_score": total_score,
            "coverage_score": cov_score,
            "confidence_score": conf_score,
            "raw_scores": raw_dict,
            "percentiles": pct_dict,
            "norm_weights": capped_weights,
            "contributions": contrib_dict
        })
        
    # Separate into active (coverage >= 50%) and excluded (coverage < 50%)
    active = [x for x in leaderboard if x["coverage_score"] >= 50.0]
    excluded = [x for x in leaderboard if x["coverage_score"] < 50.0]
    
    # Sort both descending by total_score
    active.sort(key=lambda x: x["total_score"], reverse=True)
    excluded.sort(key=lambda x: x["total_score"], reverse=True)
    
    # Assign ranks and excluded flags
    final_leaderboard = []
    for rank_idx, item in enumerate(active):
        item["rank"] = rank_idx + 1
        item["excluded"] = False
        final_leaderboard.append(item)
        
    for item in excluded:
        item["rank"] = None
        item["excluded"] = True
        final_leaderboard.append(item)
        
    return final_leaderboard


# ── Formatted Leaderboard Print ──────────────────────────────────────────────

def print_leaderboard(leaderboard: list[dict]) -> None:
    """
    Renders the stock opportunity leaderboard in a clear text table.
    """
    header = f"{'Rank':<5} {'Ticker':<10} {'Sector':<15} {'Qual':<5} {'Grow':<5} {'Val':<5} {'Mom':<5} {'Inst':<5} {'Tail':<5} {'Cred':<5} {'Total Score':<12} {'Cov%':<6} {'Conf':<6}"
    rule = "=" * len(header)
    print("\n" + rule)
    print("STOCK OPPORTUNITY LEADERBOARD")
    print(rule)
    print(header)
    print(rule)
    
    active_printed = False
    for item in leaderboard:
        if not item.get("excluded", False):
            active_printed = True
            rank_str = str(item['rank']) if item.get('rank') is not None else "N/A"
            print(
                f"{rank_str:<5} {item['ticker']:<10} {item['sector']:<15} "
                f"{item['quality']:<5.1f} {item['growth']:<5.1f} {item['valuation']:<5.1f} "
                f"{item['momentum']:<5.1f} {item['institutional']:<5.1f} {item['tailwind']:<5.1f} "
                f"{item['credibility']:<5.1f} {item['total_score']:<12.2f} "
                f"{item['coverage_score']:<6.1f} {item['confidence_score']:<6.1f}"
            )
            
    if not active_printed:
        print("No active tickers (all tickers excluded due to coverage < 50%).")
        
    # Excluded tickers section
    excluded_tickers = [x for x in leaderboard if x.get("excluded", False)]
    if excluded_tickers:
        print("\n" + rule)
        print("EXCLUDED TICKERS (Coverage < 50.0%)")
        print(rule)
        print(f"{'Ticker':<10} {'Sector':<15} {'Total Score':<12} {'Cov%':<6} {'Conf':<6} {'Reason':<15}")
        print(rule)
        for item in excluded_tickers:
            print(
                f"{item['ticker']:<10} {item['sector']:<15} "
                f"{item['total_score']:<12.2f} {item['coverage_score']:<6.1f} {item['confidence_score']:<6.1f} "
                f"{'Coverage < 50%':<15}"
            )
    print(rule + "\n")


# ── CLI Entrypoint ──────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Recalculate opportunity rankings and print leaderboard")
    args = parser.parse_args()
    
    print("[opportunity_engine] Recalculating multi-factor rankings...")
    leaderboard = recalculate_opportunity_scores()
    print_leaderboard(leaderboard)
    log("opportunity_engine", f"Recalculated rankings for {len(leaderboard)} companies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

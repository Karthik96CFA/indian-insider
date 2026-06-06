#!/usr/bin/env python3
"""
universe_expansion.py — Expand universe to Nifty 500 stocks, run parallel data collection and ranking.
"""
from __future__ import annotations

import csv
import datetime
import io
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
from common import _conn, log, update_company_scores
from event_detector import VALID_TICKERS
from fundamental_collector import fetch_fundamentals
from valuation_engine import evaluate_valuation
from canslim_engine import calculate_canslim_score, record_scores_db as record_canslim_db
from multibagger_engine import calculate_multibagger_score, record_scores_db as record_multibagger_db
from opportunity_engine import recalculate_opportunity_scores

# Setup Thread Safe DB Adapter
db_lock = threading.Lock()
original_conn = common._conn

class ThreadSafeDBAdapter:
    def __init__(self, adapter):
        self.adapter = adapter
    def __getattr__(self, name):
        return getattr(self.adapter, name)
    def execute(self, sql, params=()):
        return self.adapter.execute(sql, params)
    def commit(self):
        self.adapter.commit()
    def rollback(self):
        self.adapter.rollback()
    def close(self):
        self.adapter.close()
    def __enter__(self):
        db_lock.acquire()
        self.adapter.__enter__()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.adapter.__exit__(exc_type, exc_val, exc_tb)
        finally:
            db_lock.release()

def locked_conn():
    return ThreadSafeDBAdapter(original_conn())

# Monkey patch _conn to make it thread-safe
common._conn = locked_conn

def _fetch_index_csv(url: str, label: str) -> list[str]:
    """Helper: download an niftyindices.com index CSV and return cleaned symbols."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    print(f"[expansion] Fetching {label} from {url}...")
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            f = io.StringIO(r.text)
            reader = csv.DictReader(f)
            tickers = []
            for row in reader:
                symbol = row.get('Symbol')
                if symbol:
                    cleaned = symbol.strip().upper().replace("-", "_")
                    tickers.append(cleaned)
            print(f"[expansion]   → {len(tickers)} tickers from {label}")
            return tickers
        else:
            print(f"[expansion] ERROR: {label} returned HTTP {r.status_code}")
    except Exception as exc:
        print(f"[expansion] ERROR: {label} fetch failed: {exc}")
    return []


def download_nifty500_tickers() -> list[str]:
    """Downloads Nifty 500 + Nifty Midcap 150 + Nifty Smallcap 250 constituents.
    Combined universe: ~800 liquid NSE stocks covering large/mid/small-cap space.
    """
    index_urls = [
        ('https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv',        'Nifty 500'),
        ('https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv',   'Nifty Midcap 150'),
        ('https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv', 'Nifty Smallcap 250'),
    ]
    all_tickers: set[str] = set()
    for url, label in index_urls:
        tickers = _fetch_index_csv(url, label)
        all_tickers.update(tickers)

    if not all_tickers:
        print("[expansion] All index downloads failed — falling back to VALID_TICKERS.")
        return []

    result = sorted(all_tickers)
    print(f"[expansion] Combined universe: {len(result)} unique tickers (Nifty 500 + Midcap 150 + Smallcap 250)")
    return result

def update_event_detector_tickers(all_tickers: set[str]) -> bool:
    """Updates event_detector.py VALID_TICKERS set on disk."""
    file_path = Path(__file__).resolve().parent / "event_detector.py"
    if not file_path.exists():
        print(f"[expansion] ERROR: event_detector.py not found at {file_path}")
        return False
        
    content = file_path.read_text(encoding="utf-8")
    
    start_str = "VALID_TICKERS = {"
    idx = content.find(start_str)
    if idx == -1:
        print("[expansion] ERROR: Could not find VALID_TICKERS definition in event_detector.py")
        return False
        
    end_idx = content.find("}", idx)
    if end_idx == -1:
        print("[expansion] ERROR: Could not find closing brace for VALID_TICKERS")
        return False
        
    formatted_tickers = ",\n    ".join(f'"{t}"' for t in sorted(all_tickers))
    new_block = f"VALID_TICKERS = {{\n    {formatted_tickers}\n}}"
    
    new_content = content[:idx] + new_block + content[end_idx+1:]
    file_path.write_text(new_content, encoding="utf-8")
    print(f"[expansion] Successfully updated event_detector.py with {len(all_tickers)} valid tickers.")
    return True

def collect_for_ticker(ticker: str) -> tuple[str, str, str | None]:
    """Runs data collection and scoring sequence for a single ticker with pacing."""
    import time
    import random
    try:
        # Pacing to avoid yfinance rate limiting
        time.sleep(random.uniform(0.5, 2.0))
        
        # 1. Fetch Fundamentals
        if not fetch_fundamentals(ticker):
            return ticker, "Failed", "Fundamentals collection failed"
            
        # 2. Evaluate Valuation
        if not evaluate_valuation(ticker):
            return ticker, "Failed", "Valuation evaluation failed"
            
        # 3. Canslim Score
        score_canslim, bd_canslim = calculate_canslim_score(ticker)
        if bd_canslim:
            record_canslim_db(ticker, score_canslim)
        else:
            return ticker, "Failed", "Canslim scoring failed"
            
        # 4. Multibagger Score
        score_multi, bd_multi = calculate_multibagger_score(ticker)
        if bd_multi:
            record_multibagger_db(ticker, score_multi)
        else:
            return ticker, "Failed", "Multibagger scoring failed"
            
        return ticker, "Success", None
    except Exception as e:
        return ticker, "Failed", f"Exception: {e}"

MIN_MARKET_CAP_CR = 1000          # ₹1,000 Crore minimum
MIN_MARKET_CAP_INR = MIN_MARKET_CAP_CR * 1e7   # 10,000,000,000 INR


def get_market_cap_inr(ticker: str) -> float | None:
    """
    Return market cap in INR for a ticker.
    1. Check company_fundamentals cache first (fast, no network).
    2. Fall back to yfinance fast_info if not cached.
    Returns None if unavailable.
    """
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT market_cap FROM company_fundamentals WHERE ticker=?", (ticker,)
            ).fetchone()
            if row and row[0] and row[0] > 0:
                return float(row[0])
    except Exception:
        pass

    # Not in cache — quick yfinance lookup
    import yfinance as yf
    try:
        fi = yf.Ticker(f"{ticker.replace('_', '-')}.NS").fast_info
        mc = getattr(fi, "market_cap", None)
        if mc and mc > 0:
            return float(mc)
    except Exception:
        pass
    return None


def passes_market_cap_filter(ticker: str) -> bool:
    """Returns True if the ticker's market cap is ≥ MIN_MARKET_CAP_INR (₹1,000 Cr)."""
    mc = get_market_cap_inr(ticker)
    if mc is None:
        # Unknown — include it so we don't miss new/recent listings
        return True
    return mc >= MIN_MARKET_CAP_INR


def is_ticker_complete(ticker: str) -> bool:
    """Checks if a ticker already has fundamentals and valuation metrics in DB."""
    try:
        with _conn() as conn:
            has_fund = conn.execute("SELECT 1 FROM company_fundamentals WHERE ticker = ?", (ticker,)).fetchone() is not None
            has_val = conn.execute("SELECT 1 FROM valuation_metrics WHERE ticker = ?", (ticker,)).fetchone() is not None
            return has_fund and has_val
    except Exception:
        return False

def count_complete_tickers() -> int:
    """Counts how many tickers currently have complete fundamentals and valuation."""
    try:
        with _conn() as conn:
            rows = conn.execute("SELECT ticker FROM company_scores").fetchall()
            tickers = [r[0] for r in rows]
            count = 0
            for t in tickers:
                has_fund = conn.execute("SELECT 1 FROM company_fundamentals WHERE ticker = ?", (t,)).fetchone() is not None
                has_val = conn.execute("SELECT 1 FROM valuation_metrics WHERE ticker = ?", (t,)).fetchone() is not None
                if has_fund and has_val:
                    count += 1
            return count
    except Exception:
        return 0

def main() -> int:
    start_time = datetime.datetime.now()
    
    # 1. Load Nifty 500 tickers
    nifty500_tickers = download_nifty500_tickers()
    if not nifty500_tickers:
        print("[expansion] Fallback to existing VALID_TICKERS due to empty fetch.")
        nifty500_tickers = list(VALID_TICKERS)
        
    # 2. Merge with existing VALID_TICKERS to keep indices/ETFs
    combined_tickers = set(nifty500_tickers) | VALID_TICKERS
    update_event_detector_tickers(combined_tickers)
    
    # Filter out indices, ETFs, or known non-stocks from collection
    ignored_patterns = {"NIFTY", "BANKNIFTY", "NIFTYBEES", "GOLDBEES", "GOLD", "LIQUIDBEES", "JUNIORBEES"}
    tickers_to_collect = [
        t for t in sorted(nifty500_tickers)
        if t not in ignored_patterns and not t.endswith("BEES")
    ]
    
    # Check current complete count
    initial_complete = count_complete_tickers()
    print(f"[expansion] Database currently has {initial_complete} complete tickers.")

    # Apply market cap filter — skip tickers below ₹1,000 Cr
    print(f"[expansion] Applying ≥ ₹{MIN_MARKET_CAP_CR} Cr market cap filter …")
    tickers_to_collect = [t for t in tickers_to_collect if passes_market_cap_filter(t)]
    print(f"[expansion] {len(tickers_to_collect)} tickers pass the market cap filter.")

    # Filter out already-complete tickers — only collect new/stale ones
    tickers_to_collect = [t for t in tickers_to_collect if not is_ticker_complete(t)]
    to_attempt = tickers_to_collect  # no artificial ceiling — collect full universe

    if not to_attempt:
        print("[expansion] All qualifying tickers already complete. Recalculating scores.")
        successes = []
        failures = []
    else:
        print(f"[expansion] {len(to_attempt)} tickers to collect/refresh (≥ ₹{MIN_MARKET_CAP_CR} Cr, not yet complete).")
        
        successes = []
        failures = []
        
        # 3. Parallel Execution via Thread Pool (max_workers=3 for pacing)
        max_workers = 3
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(collect_for_ticker, t): t for t in to_attempt}
            
            count = 0
            for future in as_completed(futures):
                ticker, status, reason = future.result()
                count += 1
                if status == "Success":
                    successes.append(ticker)
                    print(f"[{count}/{len(to_attempt)}] SUCCESS: {ticker}")
                else:
                    failures.append((ticker, reason))
                    print(f"[{count}/{len(to_attempt)}] FAILED: {ticker} - {reason}")
                    
    # 4. Recalculate Opportunity Scores for the entire universe together
    print("[expansion] Recalculating overall opportunity scores and ranks...")
    recalculate_opportunity_scores()
    
    # 5. Calculate Metrics for Report
    conn = _conn()
    
    # Total active universe in company_scores (coverage >= 50%)
    active_rows = conn.execute(
        "SELECT ticker, coverage_score, confidence_score, fundamental_score, valuation_score, canslim_score "
        "FROM company_scores"
    ).fetchall()
    
    universe_size = len(active_rows)
    
    # Calculate coverage percentages
    # Quality present if fundamental_score exists
    # Valuation present if valuation_score exists
    # Institutional present if canslim_score exists
    quality_count = sum(1 for r in active_rows if r[3] is not None and r[3] > 0.0)
    valuation_count = sum(1 for r in active_rows if r[4] is not None and r[4] > 0.0)
    inst_count = sum(1 for r in active_rows if r[5] is not None and r[5] > 0)
    
    quality_coverage = (quality_count / universe_size * 100.0) if universe_size > 0 else 0.0
    valuation_coverage = (valuation_count / universe_size * 100.0) if universe_size > 0 else 0.0
    inst_coverage = (inst_count / universe_size * 100.0) if universe_size > 0 else 0.0
    
    # Freshness
    freshness_ages = []
    # Fetch fundamentals last updated dates
    fund_updates = conn.execute("SELECT last_updated FROM company_fundamentals").fetchall()
    now = datetime.datetime.now(datetime.timezone.utc)
    for row in fund_updates:
        if row[0]:
            try:
                dt_str = row[0].replace("Z", "+00:00")
                dt = datetime.datetime.fromisoformat(dt_str)
                age = (now - dt).total_seconds() / 86400.0
                freshness_ages.append(max(0.0, age))
            except Exception:
                pass
                
    avg_age_days = (sum(freshness_ages) / len(freshness_ages)) if freshness_ages else 0.0
    avg_freshness_score = max(0.0, 100.0 - 0.5 * avg_age_days)
    
    conn.close()
    
    # 6. Generate Report Markdown
    report_lines = []
    report_lines.append("# Stage 5: Universe Expansion Report")
    report_lines.append("")
    report_lines.append("This report documents the universe expansion of the Indian Insider database from 18 tickers to a large-scale liquid NSE universe.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 1. Expansion Execution Summary")
    report_lines.append("")
    report_lines.append(f"*   **Start Time**: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"*   **End Time**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"*   **Total Duration**: {(datetime.datetime.now() - start_time).total_seconds():.1f} seconds")
    report_lines.append(f"*   **Attempted Tickers**: {len(tickers_to_collect)}")
    report_lines.append(f"*   **Successful Collections**: {len(successes)}")
    report_lines.append(f"*   **Failed Collections**: {len(failures)}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 2. Universe Coverage Metrics")
    report_lines.append("")
    report_lines.append("| Metric | Target Goal | Achieved Value | Status |")
    report_lines.append("| :--- | :---: | :---: | :---: |")
    
    status_qual = "**PASSED**" if quality_coverage >= 90.0 else "*FAILED*"
    status_val = "**PASSED**" if valuation_coverage >= 90.0 else "*FAILED*"
    status_inst = "**PASSED**" if inst_coverage >= 80.0 else "*FAILED*"
    status_univ = "**PASSED**" if universe_size >= 250 else "*FAILED*"
    report_lines.append(f"| **Active Universe Size** | $\ge 250$ stocks | **{universe_size}** | {status_univ} |")
    report_lines.append(f"| **Quality Coverage** | $> 90\\%$ | **{quality_coverage:.1f}%** | {status_qual} |")
    report_lines.append(f"| **Valuation Coverage** | $> 90\\%$ | **{valuation_coverage:.1f}%** | {status_val} |")
    report_lines.append(f"| **Institutional Coverage** | $> 80\\%$ | **{inst_coverage:.1f}%** | {status_inst} |")
    report_lines.append(f"| **Average Data Age** | N/A | **{avg_age_days:.2f} days** | — |")
    report_lines.append(f"| **Average Freshness Score** | N/A | **{avg_freshness_score:.1f}/100** | — |")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 3. Factor Missingness Analysis")
    report_lines.append("")
    report_lines.append(f"*   **Quality (Fundamentals) Missing %**: {100.0 - quality_coverage:.1f}%")
    report_lines.append(f"*   **Valuation Missing %**: {100.0 - valuation_coverage:.1f}%")
    report_lines.append(f"*   **Institutional (Canslim) Missing %**: {100.0 - inst_coverage:.1f}%")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 4. Collection Failures & Reasons")
    if failures:
        report_lines.append("")
        report_lines.append("| Ticker | Failure Reason |")
        report_lines.append("| :--- | :--- |")
        for f_tick, f_reason in failures[:30]:
            report_lines.append(f"| {f_tick} | {f_reason} |")
        if len(failures) > 30:
            report_lines.append(f"| *... and {len(failures) - 30} more* | |")
    else:
        report_lines.append("No collection failures occurred during execution.")

    report_lines.append("")
    report_lines.append(f"*Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    report_path = Path(__file__).resolve().parent.parent / "reports" / "coverage_expansion_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[expansion] Report saved to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

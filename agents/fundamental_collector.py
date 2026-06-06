#!/usr/bin/env python3
"""
fundamental_collector.py — Ingests company financial statements from Yahoo Finance
and computes core fundamental metrics.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

try:
    import pandas as pd
    import yfinance as yf
except ImportError:
    sys.stderr.write("Run: pip install pandas yfinance\n")
    raise

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import log, record_company_fundamentals
from event_detector import VALID_TICKERS


def get_val(df: pd.DataFrame, row_name: str, col_idx: int = 0) -> float:
    """
    Safely retrieves a value from a statement DataFrame by row name and column index.
    """
    try:
        if row_name in df.index:
            val = df.loc[row_name].iloc[col_idx]
            if pd.isna(val):
                return 0.0
            return float(val)
    except Exception:
        pass
    return 0.0


def calculate_cagr(recent: float, past: float, periods: int) -> float:
    """
    Calculates CAGR.
    - Standard geometric CAGR when both values are positive.
    - Returns 0.0 when past is zero or when the sign change makes CAGR
      economically undefined (e.g. negative-to-positive EPS turnaround).
      Callers should treat 0.0 as 'not meaningful' rather than 'zero growth'.
    """
    if periods <= 0:
        return 0.0
    if past > 0 and recent > 0:
        return (recent / past) ** (1.0 / periods) - 1.0
    # Sign-change or zero-base cases: CAGR is undefined — return 0.0 as sentinel
    return 0.0


def fetch_fundamentals(ticker: str) -> bool:
    """
    Fetches statements from yfinance, calculates metrics, and records them in the DB.
    """
    yf_symbol = f"{ticker.replace('_', '-')}.NS"
    print(f"[fundamentals] Fetching {yf_symbol}...")
    
    try:
        t = yf.Ticker(yf_symbol)
        
        # Financial statements
        inc = t.income_stmt
        bal = t.balance_sheet
        cf = t.cashflow
        
        if inc.empty or bal.empty or cf.empty:
            print(f"               WARNING: Empty financial statements for {ticker}")
            return False
            
        # Get column length for historical CAGR calculations
        cols = inc.columns
        periods = 3
        if len(cols) < 4:
            periods = len(cols) - 1
            
        # 1. ROCE = EBIT / (Total Assets - Current Liabilities)
        ebit = get_val(inc, 'EBIT', 0)
        assets = get_val(bal, 'Total Assets', 0)
        current_liab = get_val(bal, 'Current Liabilities', 0)
        cap_employed = assets - current_liab
        roce = (ebit / cap_employed * 100.0) if cap_employed > 0 else 0.0
        
        # 2. ROE = Net Income / Stockholders Equity
        net_inc = get_val(inc, 'Net Income', 0)
        equity = get_val(bal, 'Stockholders Equity', 0) or get_val(bal, 'Common Stock Equity', 0)
        roe = (net_inc / equity * 100.0) if equity > 0 else 0.0
        
        # 3. Debt to Equity = Total Debt / Stockholders Equity
        debt = get_val(bal, 'Total Debt', 0)
        debt_equity = (debt / equity) if equity > 0 else 0.0
        
        # 4. Operating Margin = Operating Income / Total Revenue
        op_inc = get_val(inc, 'Operating Income', 0)
        rev = get_val(inc, 'Total Revenue', 0)
        operating_margin = (op_inc / rev * 100.0) if rev > 0 else 0.0
        
        # 5. Free Cash Flow (FCF)
        fcf = get_val(cf, 'Free Cash Flow', 0)
        if fcf == 0.0:
            fcf = get_val(cf, 'Operating Cash Flow', 0) - abs(get_val(cf, 'Capital Expenditure', 0))
            
        # Check and convert currency if financial statements report in USD
        info = t.info or {}
        fin_currency = info.get("financialCurrency") or "INR"
        if fin_currency.upper() == "USD":
            usd_inr_rate = 83.5
            try:
                rate_ticker = yf.Ticker("USDINR=X")
                rate_hist = rate_ticker.history(period="1d")
                if not rate_hist.empty:
                    usd_inr_rate = float(rate_hist["Close"].iloc[-1])
                    print(f"               [currency] USDINR exchange rate fetched: {usd_inr_rate:.2f}")
                else:
                    print(f"               [currency] WARNING: USDINR price history empty. Using fallback {usd_inr_rate}")
            except Exception as exc:
                print(f"               [currency] WARNING: Failed to fetch USDINR rate: {exc}. Using fallback {usd_inr_rate}")
            
            fcf_inr = fcf * usd_inr_rate
            print(f"               [currency] Scaled FCF from {fcf/1e9:.3f}B USD to {fcf_inr/1e7:.1f} Cr INR")
            fcf = fcf_inr

        # 6. Sales CAGR (3-Year)
        rev_recent = get_val(inc, 'Total Revenue', 0)
        rev_past = get_val(inc, 'Total Revenue', min(periods, len(cols) - 1))
        sales_cagr = calculate_cagr(rev_recent, rev_past, periods) * 100.0
        
        # 7. EPS Growth (3-Year)
        eps_recent = get_val(inc, 'Diluted EPS', 0)
        eps_past = get_val(inc, 'Diluted EPS', min(periods, len(cols) - 1))
        eps_growth = calculate_cagr(eps_recent, eps_past, periods) * 100.0
        
        # 8. Institutional ownership change (Held % Institutions)
        inst_pct = (info.get('heldPercentInstitutions') or info.get('institutionsPercentHeld') or 0.0) * 100.0
        
        # Get Sector and Market Cap
        sector = info.get("sector")
        # marketCap is in INR for .NS tickers; store as-is (in rupees)
        market_cap_raw = info.get("marketCap") or info.get("market_cap")
        market_cap = float(market_cap_raw) if market_cap_raw else None

        # Record to Database
        record_company_fundamentals(
            ticker=ticker,
            roce=roce,
            roe=roe,
            debt_equity=debt_equity,
            operating_margin=operating_margin,
            fcf=fcf,
            sales_cagr_3y=sales_cagr,
            eps_growth_3y=eps_growth,
            inst_holding_change=inst_pct,
            sector=sector,
            market_cap=market_cap,
        )
        
        print(f"               SUCCESS: ROCE={roce:.1f}%, ROE={roe:.1f}%, D/E={debt_equity:.2f}, FCF={fcf/1e7:.1f} Cr")
        log("fundamentals", f"Updated fundamentals for {ticker} (ROCE={roce:.1f}%, ROE={roe:.1f}%)")
        return True
        
    except Exception as exc:
        print(f"               ERROR: Failed to fetch fundamentals for {ticker}: {exc}")
        log("fundamentals", f"Error fetching fundamentals for {ticker}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect company fundamental metrics from Yahoo Finance")
    parser.add_argument('--ticker', type=str, help="Specific ticker symbol to fetch (e.g. RELIANCE)")
    parser.add_argument('--limit', type=int, help="Limit number of tickers to process (for quick testing)")
    args = parser.parse_args()
    
    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        # Filter out indices or ETFs from the validation list
        tickers = [
            t for t in VALID_TICKERS 
            if t not in {"NIFTY", "BANKNIFTY", "NIFTYBEES", "GOLDBEES", "GOLD"}
            and not t.endswith("BEES")
        ]
        
    if args.limit:
        tickers = tickers[:args.limit]
        
    print(f"[fundamentals] Starting collection for {len(tickers)} companies...")
    
    success_count = 0
    for ticker in tickers:
        if fetch_fundamentals(ticker):
            success_count += 1
            
    print(f"[fundamentals] Collection complete. Successfully updated {success_count}/{len(tickers)} companies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

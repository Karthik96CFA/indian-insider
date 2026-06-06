#!/usr/bin/env python3
"""
valuation_engine.py — Computes relative valuation (PE, PEG, FCF Yield)
and implied growth rates using a Reverse DCF bisection solver.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    sys.stderr.write("Run: pip install yfinance\n")
    raise

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    _conn,
    log,
    read_company_fundamentals,
    record_valuation_metrics,
    update_company_scores,
)


def calculate_implied_dcf_growth(
    market_cap: float,
    fcf: float,
    discount_rate: float = 0.12,
    terminal_growth: float = 0.04,
) -> float:
    """
    Reverse DCF Bisection Solver:
    Solves for the implied growth rate g that equates the present value
    of FCFs (years 1-10) and terminal value to the current market cap.
    """
    if fcf <= 0 or market_cap <= 0:
        return 0.0  # DCF model is undefined for negative FCF or market cap

    def pv_calc(g: float) -> float:
        pv_flows = 0.0
        cf = fcf
        for t in range(1, 11):
            cf = cf * (1.0 + g)
            pv_flows += cf / ((1.0 + discount_rate) ** t)
        cf_10 = cf
        tv = (cf_10 * (1.0 + terminal_growth)) / (discount_rate - terminal_growth)
        pv_terminal = tv / ((1.0 + discount_rate) ** 10)
        return pv_flows + pv_terminal

    # Bisection search range
    low = -0.50  # -50% growth
    high = 2.00  # +200% growth
    target = market_cap

    # If bounds don't bracket the target, return boundary values
    if pv_calc(low) > target:
        return low * 100.0
    if pv_calc(high) < target:
        return high * 100.0

    for _ in range(50):  # 50 iterations provides extreme floating-point precision
        mid = (low + high) / 2.0
        pv = pv_calc(mid)
        if pv < target:
            low = mid
        else:
            high = mid
            
    return mid * 100.0


def evaluate_valuation(ticker: str, discount_rate: float = 0.12, terminal_growth: float = 0.04) -> bool:
    """
    Calculates valuation metrics for a given ticker and writes to DB.
    """
    fundamentals = read_company_fundamentals(ticker)
    if not fundamentals:
        print(f"[valuation] ERROR: Fundamentals not found for {ticker}. Run fundamental_collector.py first.")
        return False

    yf_symbol = f"{ticker.replace('_', '-')}.NS"
    print(f"[valuation] Valuing {yf_symbol}...")
    
    try:
        t = yf.Ticker(yf_symbol)
        info = t.info or {}
        
        # Get current price
        price = info.get("currentPrice")
        if not price:
            # Fallback to history close
            hist = t.history(period="5d")
            if hist.empty:
                print(f"            ERROR: Price history unavailable for {ticker}")
                return False
            price = float(hist["Close"].iloc[-1])
            
        # Get Market Cap
        market_cap = info.get("marketCap")
        if not market_cap:
            # Fallback calculation using shares outstanding
            shares = info.get("sharesOutstanding")
            if not shares:
                # Fallback to balance sheet Ordinary Shares Number
                shares = get_shares_from_balance_sheet(t)
            if not shares or shares <= 0:
                print(f"            ERROR: Shares outstanding unavailable for {ticker}")
                return False
            market_cap = price * shares
            
        # Get PE
        pe = info.get("trailingPE")
        if not pe or pe <= 0:
            # Calculate from fundamentals
            eps_3y = fundamentals.get("eps_growth_3y") or 0.0
            # If no PE in info, try to calculate from net income
            # Or default to 0.0
            pe = 0.0
            
        # Calculate PEG with safety checks (growth must be >= 5% to compute PEG)
        eps_growth_3y = fundamentals.get("eps_growth_3y") or 0.0
        if pe > 0 and eps_growth_3y >= 5.0:
            peg = pe / eps_growth_3y
        else:
            peg = None

        # Get EV/EBITDA
        ev_ebitda = info.get("enterpriseToEbitda") or 0.0
        
        # Calculate FCF Yield
        fcf = fundamentals.get("fcf") or 0.0
        fcf_yield = (fcf / market_cap * 100.0) if market_cap > 0 else 0.0
        
        # Calculate Implied DCF Growth Rate
        implied_growth = calculate_implied_dcf_growth(market_cap, fcf, discount_rate, terminal_growth)
        
        # Record to DB
        record_valuation_metrics(
            ticker=ticker,
            pe=pe,
            peg=peg,
            ev_ebitda=ev_ebitda,
            fcf_yield=fcf_yield,
            implied_dcf_growth=implied_growth
        )
        update_company_scores(ticker)
        
        peg_str = f"{peg:.2f}" if peg is not None else "N/A"
        print(f"            VALUATION: PE={pe:.1f}, PEG={peg_str}, FCF Yield={fcf_yield:.2f}%, Implied DCF Growth={implied_growth:.1f}%")
        log("valuation", f"Valued {ticker} (PE={pe:.1f}, PEG={peg_str}, Implied DCF Growth={implied_growth:.1f}%)")
        return True
        
    except Exception as exc:
        print(f"            ERROR: Valuation failed for {ticker}: {exc}")
        log("valuation", f"Valuation failed for {ticker}: {exc}")
        return False


def get_shares_from_balance_sheet(t: yf.Ticker) -> float:
    try:
        bs = t.balance_sheet
        if "Ordinary Shares Number" in bs.index:
            return float(bs.loc["Ordinary Shares Number"].iloc[0])
    except Exception:
        pass
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate company valuations and growth rates")
    parser.add_argument('--ticker', type=str, help="Specific ticker symbol to fetch (e.g. INFY)")
    parser.add_argument('--discount-rate', type=float, default=0.12, help="Discount rate for DCF (default: 0.12)")
    parser.add_argument('--terminal-growth', type=float, default=0.04, help="Terminal growth rate for DCF (default: 0.04)")
    args = parser.parse_args()
    
    conn = _conn()
    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        # Load all tickers from company_fundamentals
        rows = conn.execute("SELECT ticker FROM company_fundamentals").fetchall()
        tickers = [r[0] for r in rows]
        
    if not tickers:
        print("[valuation] No tickers found in company_fundamentals database. Run fundamental_collector.py first.")
        return 1
        
    print(f"[valuation] Evaluating {len(tickers)} companies...")
    success_count = 0
    for ticker in tickers:
        if evaluate_valuation(ticker, args.discount_rate, args.terminal_growth):
            success_count += 1
            
    print(f"[valuation] Valuation complete. Successfully valued {success_count}/{len(tickers)} companies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

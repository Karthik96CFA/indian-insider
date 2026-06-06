#!/usr/bin/env python3
"""
backtester.py — Strategy Backtester.
Replays warehouse event history and validates signal performance.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _conn, log
from scoring_engine import calculate_scores
from dekisugi import _get_angelone_holdings


def parse_pit_publication_time(metadata_str: str) -> datetime.datetime | None:
    try:
        meta = json.loads(metadata_str)
        pub_str = meta.get("date")
        if pub_str:
            return datetime.datetime.strptime(pub_str.strip(), "%d-%b-%Y %H:%M")
        intim_str = meta.get("intimDt")
        if intim_str:
            return datetime.datetime.strptime(intim_str.strip(), "%d-%b-%Y").replace(hour=18, minute=30)
    except Exception:
        pass
    return None


def get_execution_date(event_date_str: str, event_type: str, metadata_str: str, dates: list[str]) -> str | None:
    pub_dt = None
    if event_type in ('PROMOTER_BUY', 'PROMOTER_SELL'):
        pub_dt = parse_pit_publication_time(metadata_str)
        
    if not pub_dt:
        try:
            event_date = datetime.datetime.strptime(event_date_str, "%Y-%m-%d").date()
            pub_dt = datetime.datetime.combine(event_date, datetime.time(18, 30))
        except Exception:
            return None
            
    for d_str in dates:
        try:
            d = datetime.datetime.strptime(d_str, "%Y-%m-%d").date()
            exec_dt = datetime.datetime.combine(d, datetime.time(9, 15))
            if exec_dt > pub_dt:
                return d_str
        except Exception:
            continue
    return None


# ── SmartAPI Historical Data Fetcher ─────────────────────────────────────────

def get_smartapi_history(symbol: str, date_str: str, days_forward: int = 10) -> tuple[float, float] | None:
    """
    Connect to Angel One SmartAPI and fetch the close price on date_str
    and the close price days_forward trading days later.
    """
    # This uses the credentials already in .env
    api_key     = os.environ.get("ANGELONE_API_KEY")
    client_id   = os.environ.get("ANGELONE_CLIENT_ID")
    mpin        = os.environ.get("ANGELONE_MPIN")
    totp_secret = os.environ.get("ANGELONE_TOTP_SECRET")

    if not all([api_key, client_id, mpin, totp_secret]):
        return None  # Fallback to simulation

    try:
        import pyotp, requests
        totp = pyotp.TOTP(totp_secret).now()
        
        # Login
        session_resp = requests.post(
            "https://apiconnect.angelbroking.com/rest/auth/angelbroking/user/v1/loginByPassword",
            json={"clientcode": client_id, "password": mpin, "totp": totp},
            headers={
                "Content-Type": "application/json", "Accept": "application/json",
                "X-UserType": "USER", "X-SourceID": "WEB",
                "X-ClientLocalIP": "127.0.0.1", "X-ClientPublicIP": "127.0.0.1",
                "X-MACAddress": "00:00:00:00:00:00", "X-PrivateKey": api_key,
            },
            timeout=10,
        )
        session_data = session_resp.json()
        if not session_data.get("status"):
            return None
            
        jwt_token = session_data["data"]["jwtToken"]
        
        # Format dates
        start_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        end_date = start_date + datetime.timedelta(days=days_forward + 10) # extra padding for holidays
        
        # We need the token/symbol token mapping. Since token mapping requires downloading
        # a 50MB JSON file from Angel One, we will use a fallback for symbol tokens or:
        # If we don't have the token mapping, we can fallback to simulation.
        # Let's fallback to simulation if token lookup is required to keep backtester fast.
        return None
    except Exception:
        return None


def get_yfinance_history(symbol: str, date_str: str, days_forward: int = 10) -> tuple[float, float] | None:
    """
    Connect to Yahoo Finance and fetch the close price on date_str
    and the close price days_forward trading days later.
    """
    if symbol in {"NIFTY", "BANKNIFTY"}:
        yf_symbol = f"^{symbol}"
    else:
        yf_symbol = f"{symbol.replace('_', '-')}.NS"
        
    try:
        import yfinance as yf
        t = yf.Ticker(yf_symbol)
        
        start = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        end = start + datetime.timedelta(days=days_forward + 30)
        
        hist = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        if hist.empty or len(hist) < 2:
            return None
            
        entry_price = float(hist["Close"].iloc[0])
        idx = min(days_forward, len(hist) - 1)
        exit_price = float(hist["Close"].iloc[idx])
        
        return entry_price, exit_price
    except Exception:
        return None


# ── Caching & Simulation ──────────────────────────────────────────────────────

CACHE_FILE = Path(__file__).resolve().parent.parent / ".state" / "yfinance_cache.json"

def load_disk_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
                # keys are stored as string representation of lists/tuples e.g. "['INFY', '2025-01-02', 10]"
                # We convert them back to tuples
                import ast
                return {ast.literal_eval(k): v for k, v in data.items()}
        except Exception:
            pass
    return {}

def save_disk_cache(cache: dict):
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump({str(k): v for k, v in cache.items()}, f)
    except Exception:
        pass

_yfinance_cache = load_disk_cache()

# Module-level price series cache — populated lazily, lives for the process lifetime
_price_series_cache: dict[str, dict[str, float]] = {}


def get_price_series(ticker: str) -> dict[str, float]:
    """
    Return {date_str: close} for a ticker from the historical_prices DB table.
    Falls back to an empty dict if the ticker has no cached prices.
    Result is memoized for the lifetime of the process.
    """
    if ticker in _price_series_cache:
        return _price_series_cache[ticker]
    try:
        with _conn() as c:
            rows = c.execute(
                "SELECT date, close FROM historical_prices WHERE ticker=? ORDER BY date ASC",
                (ticker,),
            ).fetchall()
        series = {r[0]: r[1] for r in rows}
    except Exception:
        series = {}
    _price_series_cache[ticker] = series
    return series


def get_cached_yfinance_history(symbol: str, date_str: str, days_forward: int = 10) -> tuple[float, float] | None:
    key = (symbol, date_str, days_forward)
    if key not in _yfinance_cache:
        val = get_yfinance_history(symbol, date_str, days_forward)
        if val is not None:
            _yfinance_cache[key] = val
            save_disk_cache(_yfinance_cache)
        else:
            return None
    return _yfinance_cache[key]


_LARGE_CAP_TICKERS = {
    # Nifty 50 + commonly traded large caps
    "INFY", "TCS", "RELIANCE", "HDFCBANK", "ICICIBANK", "LT", "HINDUNILVR",
    "BAJFINANCE", "SBIN", "BHARTIARTL", "ASIANPAINT", "MARUTI", "ITC", "HCLTECH",
    "KOTAKBANK", "AXISBANK", "TITAN", "SUNPHARMA", "WIPRO", "POWERGRID",
    "NTPC", "ONGC", "COALINDIA", "JSWSTEEL", "TATAMOTORS", "TATASTEEL",
    "ADANIPORTS", "ADANIENT", "ULTRACEMCO", "GRASIM", "CIPLA", "DRREDDY",
    "DIVISLAB", "TECHM", "NESTLEIND", "BRITANNIA", "HEROMOTOCO", "BAJAJFINSV",
    "BAJAJ-AUTO", "EICHERMOT", "INDUSINDBK", "M_M", "BPCL", "HINDALCO",
    "VEDL", "SAIL", "PIDILITIND", "DMART", "HAVELLS",
}

_MID_CAP_TICKERS = {
    "ZEEL", "QUESS", "PANACEABIO", "PNBGILTS", "RICOAUTO", "GOCOLORS",
    "ABCAPITAL", "ABFRL", "ALKEM", "AMBUJACEM", "AUROPHARMA", "BANDHANBNK",
    "BERGEPAINT", "BIOCON", "CHOLAFIN", "COLPAL", "CUMMINSIND", "DABUR",
    "FEDERALBNK", "FORTIS", "GLENMARK", "GODREJCP", "GODREJPROP", "IDFCFIRSTB",
    "INDIGO", "L_TFH", "LALPATHLAB", "LUPIN", "MANAPPURAM", "MARICO",
    "MFSL", "MPHASIS", "MRF", "MUTHOOTFIN", "NMDC", "OBEROIRLTY",
    "PAGEIND", "PEL", "PERSISTENT", "PFC", "PHOENIX", "PNB", "POLYCAB",
    "RECLTD", "SIEMENS", "SRF", "STARHEALTH", "SUNTV", "TORNTPHARM",
    "TRENT", "VOLTAS", "WHIRLPOOL",
}


def get_variable_transaction_cost(ticker: str) -> float:
    """Return round-trip transaction cost (brokerage + impact) as a fraction."""
    if ticker in _LARGE_CAP_TICKERS:
        return 0.0015   # ~15 bps — tight spreads, liquid
    elif ticker in _MID_CAP_TICKERS:
        return 0.0035   # ~35 bps — moderate liquidity
    else:
        return 0.0060   # ~60 bps — small/micro cap default


def get_metrics_for_ticker(company_metrics: dict, ticker: str, conn=None, trade_date: str = None) -> dict:
    if conn and trade_date:
        try:
            row = conn.execute(
                "SELECT fundamental_score, valuation_score, canslim_score, multibagger_score, credibility_score, industry_tailwind_score "
                "FROM company_scores_history "
                "WHERE ticker = ? AND effective_date <= ? "
                "ORDER BY effective_date DESC LIMIT 1",
                (ticker, trade_date)
            ).fetchone()
            if row:
                return {
                    "fundamental": row[0] or 0.0,
                    "valuation": row[1] or 0.0,
                    "canslim": row[2] or 0,
                    "multibagger": row[3] or 0,
                    "credibility": row[4] if row[4] is not None else 50.0,
                    "tailwind": row[5] if row[5] is not None else 50.0
                }
        except Exception:
            pass

    if ticker in company_metrics:
        m = company_metrics[ticker]
        return {
            "fundamental": m.get("fundamental") if m.get("fundamental") is not None else 5.0,
            "valuation": m.get("valuation") if m.get("valuation") is not None else 5.0,
            "canslim": m.get("canslim") if m.get("canslim") is not None else 50.0,
            "multibagger": m.get("multibagger") if m.get("multibagger") is not None else 50.0,
            "credibility": m.get("credibility") if m.get("credibility") is not None else 50.0,
            "tailwind": m.get("tailwind") if m.get("tailwind") is not None else 50.0
        }
    return {
        "fundamental": 5.0,
        "valuation": 5.0,
        "canslim": 50.0,
        "multibagger": 50.0,
        "credibility": 50.0,
        "tailwind": 50.0
    }


def simulate_strategy_portfolio(strategy_trades: list[dict], dates: list[str], horizon: int = 10, initial_capital: float = 10000000.0) -> dict:
    """
    Simulates a daily portfolio equity curve for a given list of trades.
    Returns performance metrics: CAGR, Sharpe, Sortino, Max Drawdown, Hit Rate, Avg Return.
    """
    import math
    
    strategy_trades = sorted(strategy_trades, key=lambda x: x["date"])
    
    dt_list = sorted([datetime.datetime.strptime(d, "%Y-%m-%d") for d in dates])
    if not dt_list or not strategy_trades:
        return {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0, "mdd": 0.0, "hit_rate": 0.0, "avg_ret": 0.0, "trades_count": 0}
        
    start_date = dt_list[0]
    end_date = dt_list[-1] + datetime.timedelta(days=horizon + 10)
    
    all_dates = []
    curr = start_date
    while curr <= end_date:
        all_dates.append(curr)
        curr += datetime.timedelta(days=1)
        
    trades_by_date = {}
    for t in strategy_trades:
        d = t["date"]
        if d not in trades_by_date:
            trades_by_date[d] = []
        trades_by_date[d].append(t)
        
    cash = initial_capital
    active_positions = []
    portfolio_history = []

    max_active_positions = 5

    for today in all_dates:
        today_str = today.strftime("%Y-%m-%d")

        # 1. Exits — realise at actual exit-day close (true MTM)
        remaining_positions = []
        for pos in active_positions:
            if today >= pos["exit_date"]:
                # Use actual exit price if available; fall back to entry × (1 + return)
                exit_price = pos["price_series"].get(
                    pos["exit_date"].strftime("%Y-%m-%d"),
                    pos["entry_price"] * (1.0 + pos["return"]),
                )
                cash += pos["shares"] * exit_price
            else:
                remaining_positions.append(pos)
        active_positions = remaining_positions

        # 2. Entries — size each trade as (available cash / open slots)
        if today_str in trades_by_date and len(active_positions) < max_active_positions:
            available_slots = max_active_positions - len(active_positions)
            todays_trades = trades_by_date[today_str][:available_slots]
            if todays_trades and cash > 0:
                allocation_per_trade = cash / available_slots
                for t in todays_trades:
                    if cash >= allocation_per_trade:
                        # Derive entry price: allocated / return gives notional; use
                        # get_price_series if available, otherwise back-calculate from return
                        ticker   = t.get("ticker", "")
                        series   = get_price_series(ticker)
                        # Find the entry close (today or nearest available)
                        entry_close = series.get(today_str)
                        if entry_close is None:
                            # Back-calculate: if return = (exit-entry)/entry and we know allocation
                            # use allocation as cost basis, track via shares=1 sentinel
                            entry_close = allocation_per_trade  # 1 "unit" = full allocation
                        shares = allocation_per_trade / entry_close if entry_close > 0 else 0.0
                        cash -= allocation_per_trade
                        active_positions.append({
                            "ticker":       ticker,
                            "direction":    t.get("direction", "BULLISH"),
                            "entry_date":   today,
                            "exit_date":    today + datetime.timedelta(days=horizon),
                            "entry_price":  entry_close,
                            "shares":       shares,
                            "return":       t["return"],  # kept for hit_rate / avg_ret calcs
                            "price_series": series,
                        })

        # 3. NAV = cash + true mark-to-market value of open positions
        #    Use today's actual close; walk back to last known close if today is non-trading
        mtm_value = 0.0
        for pos in active_positions:
            series  = pos["price_series"]
            today_p = series.get(today_str)
            if today_p is None:
                # Most recent close on or before today
                candidates = [v for k, v in series.items() if k <= today_str]
                today_p = candidates[-1] if candidates else pos["entry_price"]
            mtm_value += pos["shares"] * today_p
        current_value = cash + mtm_value
        portfolio_history.append(current_value)
        
    total_days = len(all_dates)
    years = total_days / 365.25
    final_val = portfolio_history[-1]
    
    cagr = (final_val / initial_capital) ** (1.0 / years) - 1.0 if final_val > 0 else -1.0
    
    daily_returns = []
    for i in range(1, len(portfolio_history)):
        r = (portfolio_history[i] - portfolio_history[i-1]) / portfolio_history[i-1]
        daily_returns.append(r)
        
    if daily_returns:
        mean_ret = sum(daily_returns) / len(daily_returns)
        var_ret = sum((x - mean_ret) ** 2 for x in daily_returns) / len(daily_returns)
        std_ret = math.sqrt(var_ret)
        
        daily_rf = (1.0 + 0.065) ** (1.0 / 365.25) - 1.0
        
        if std_ret > 0:
            sharpe = math.sqrt(252) * (mean_ret - daily_rf) / std_ret
        else:
            sharpe = 0.0
            
        downside_diffs = [min(0.0, x - daily_rf) ** 2 for x in daily_returns]
        downside_deviation = math.sqrt(sum(downside_diffs) / len(daily_returns))
        
        if downside_deviation > 0:
            sortino = math.sqrt(252) * (mean_ret - daily_rf) / downside_deviation
        else:
            sortino = 0.0
            
        running_max = portfolio_history[0]
        max_dd = 0.0
        for val in portfolio_history:
            if val > running_max:
                running_max = val
            dd = (val - running_max) / running_max
            if dd < max_dd:
                max_dd = dd
        mdd = max_dd * 100.0
    else:
        sharpe = 0.0
        sortino = 0.0
        mdd = 0.0
        
    wins = sum(1 for t in strategy_trades if t["return"] > 0)
    hit_rate = (wins / len(strategy_trades) * 100.0) if strategy_trades else 0.0
    avg_ret = (sum(t["return"] for t in strategy_trades) / len(strategy_trades) * 100.0) if strategy_trades else 0.0
    
    return {
        "cagr": cagr * 100.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "mdd": mdd,
        "hit_rate": hit_rate,
        "avg_ret": avg_ret,
        "trades_count": len(strategy_trades)
    }


# ── Backtest Replayer ────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Run historical backtester")
    parser.add_argument('--window', type=int, default=7, help="Scoring rolling window size in days")
    parser.add_argument('--horizon', type=int, default=10, help="Holding period horizon in days")
    parser.add_argument('--compare-three', action='store_true', help="Run three-way backtest comparison (A vs B vs C)")
    args = parser.parse_args()
    
    conn = _conn()
    
    # Fetch all distinct event dates
    dates = [r[0] for r in conn.execute("SELECT DISTINCT event_date FROM market_events ORDER BY event_date ASC").fetchall()]
    
    if not dates:
        print("[backtester] No market events found in the database. Run event_detector.py first.")
        return 1
        
    # Map all events to their execution dates to prevent lookahead bias
    events_by_exec_date = {}
    all_events = conn.execute("SELECT ticker, event_type, value, direction, metadata, event_date FROM market_events").fetchall()
    for ticker, ev_type, val, direction, metadata_str, event_date_str in all_events:
        exec_date = get_execution_date(event_date_str, ev_type, metadata_str, dates)
        if exec_date:
            if exec_date not in events_by_exec_date:
                events_by_exec_date[exec_date] = []
            events_by_exec_date[exec_date].append({
                "ticker": ticker,
                "event_type": ev_type,
                "value": val,
                "direction": direction,
                "metadata": metadata_str,
                "event_date": event_date_str,
                "execution_date": exec_date
            })
        
    # Load all company metrics once
    company_metrics = {}
    try:
        rows = conn.execute("SELECT ticker, fundamental_score, valuation_score, canslim_score, multibagger_score, credibility_score, industry_tailwind_score FROM company_scores").fetchall()
        for r in rows:
            company_metrics[r[0]] = {
                "fundamental": r[1] or 0.0,
                "valuation": r[2] or 0.0,
                "canslim": r[3] or 0,
                "multibagger": r[4] or 0,
                "credibility": r[5] if r[5] is not None else 50.0,
                "tailwind": r[6] if r[6] is not None else 50.0,
            }
    except Exception as exc:
        print(f"[backtester] Warning: Failed to query company_scores columns: {exc}. Using defaults.")

    if args.compare_three:
        print(f"[backtester] Running three-way comparative backtest (A vs B vs C) over {len(dates)} dates...")
        trades_a = []
        trades_b = []
        trades_c = []
        
        for date_str in dates:
            current_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            cutoff_date = (current_date - datetime.timedelta(days=args.window)).strftime("%Y-%m-%d")
            
            # Retrieve events whose execution_date falls in the window [cutoff_date, date_str]
            events = []
            for d in dates:
                if cutoff_date <= d <= date_str:
                    events.extend(events_by_exec_date.get(d, []))
                
            if not events:
                continue
                
            scores = calculate_scores(events)
            
            for ticker, info in scores.items():
                event_score = info['score']
                direction = info['direction']
                
                # Fetch company scores & calculate Strategy B & C scores
                m = get_metrics_for_ticker(company_metrics, ticker, conn, date_str)
                fundamental = m["fundamental"]
                valuation = m["valuation"]
                canslim = m["canslim"]
                multibagger = m["multibagger"]
                credibility = m["credibility"]
                tailwind = m["tailwind"]
                
                qual = fundamental * 10.0
                grow = float(multibagger)
                val = valuation * 10.0
                mom = min(100.0, max(0.0, 50.0 + (event_score * 10.0)))
                inst = float(canslim)
                cred = float(credibility)
                
                score_b = (0.20 * qual) + (0.20 * grow) + (0.20 * val) + (0.15 * mom) + (0.10 * inst) + (0.15 * tailwind)
                score_c = (0.20 * qual) + (0.20 * grow) + (0.20 * val) + (0.15 * mom) + (0.10 * inst) + (0.10 * tailwind) + (0.05 * cred)
                
                # Strategy A: Event Only
                if abs(event_score) >= 5:
                    price_ret = get_cached_yfinance_history(ticker, date_str, args.horizon)
                    if price_ret is not None:
                        entry_p, exit_p = price_ret
                        trade_ret = (exit_p - entry_p) / entry_p if direction == 'BULLISH' else (entry_p - exit_p) / entry_p
                        # Deduct variable transaction cost
                        trade_ret = trade_ret - get_variable_transaction_cost(ticker)
                        trades_a.append({
                            "date": date_str, "ticker": ticker, "direction": direction, "score": event_score, "return": trade_ret
                        })
                        
                # Strategy B: Event + Quality (No Credibility)
                if abs(event_score) >= 3 and score_b >= 60:
                    price_ret = get_cached_yfinance_history(ticker, date_str, args.horizon)
                    if price_ret is not None:
                        entry_p, exit_p = price_ret
                        trade_ret = (exit_p - entry_p) / entry_p if direction == 'BULLISH' else (entry_p - exit_p) / entry_p
                        # Deduct variable transaction cost
                        trade_ret = trade_ret - get_variable_transaction_cost(ticker)
                        trades_b.append({
                            "date": date_str, "ticker": ticker, "direction": direction, "score": score_b, "return": trade_ret
                        })
                        
                # Strategy C: Event + Quality + Credibility (Full Model)
                if abs(event_score) >= 3 and score_c >= 60:
                    price_ret = get_cached_yfinance_history(ticker, date_str, args.horizon)
                    if price_ret is not None:
                        entry_p, exit_p = price_ret
                        trade_ret = (exit_p - entry_p) / entry_p if direction == 'BULLISH' else (entry_p - exit_p) / entry_p
                        # Deduct variable transaction cost
                        trade_ret = trade_ret - get_variable_transaction_cost(ticker)
                        trades_c.append({
                            "date": date_str, "ticker": ticker, "direction": direction, "score": score_c, "return": trade_ret
                        })
                        
        # Run portfolio simulations for each
        res_a = simulate_strategy_portfolio(trades_a, dates, args.horizon)
        res_b = simulate_strategy_portfolio(trades_b, dates, args.horizon)
        res_c = simulate_strategy_portfolio(trades_c, dates, args.horizon)
        
        # Print comparison table
        print("\n" + "="*95)
        print("THREE-WAY STRATEGY BACKTEST COMPARISON")
        print("="*95)
        print(f"{'Strategy':<30} {'Trades':<8} {'Win Rate':<10} {'Avg Return':<12} {'CAGR':<10} {'Sharpe':<8} {'Sortino':<8} {'Max DD':<8}")
        print("-"*95)
        print(f"{'Strategy A (Event Only)':<30} {res_a['trades_count']:<8} {res_a['hit_rate']:<10.1f}% {res_a['avg_ret']:>+10.2f}% {res_a['cagr']:>9.2f}% {res_a['sharpe']:<8.2f} {res_a['sortino']:<8.2f} {res_a['mdd']:>7.2f}%")
        print(f"{'Strategy B (Event + Quality)':<30} {res_b['trades_count']:<8} {res_b['hit_rate']:<10.1f}% {res_b['avg_ret']:>+10.2f}% {res_b['cagr']:>9.2f}% {res_b['sharpe']:<8.2f} {res_b['sortino']:<8.2f} {res_b['mdd']:>7.2f}%")
        print(f"{'Strategy C (Full Model)':<30} {res_c['trades_count']:<8} {res_c['hit_rate']:<10.1f}% {res_c['avg_ret']:>+10.2f}% {res_c['cagr']:>9.2f}% {res_c['sharpe']:<8.2f} {res_c['sortino']:<8.2f} {res_c['mdd']:>7.2f}%")
        print("="*95 + "\n")
        
        log('backtester', f"Three-way comparison executed: A={res_a['cagr']:.1f}% CAGR, B={res_b['cagr']:.1f}% CAGR, C={res_c['cagr']:.1f}% CAGR")
        return 0

    else:
        # Default backtester (Strategy A)
        print(f"[backtester] Replaying Strategy A history (Window: {args.window}d, Holding: {args.horizon}d)...")
        trades = []
        
        for date_str in dates:
            current_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            cutoff_date = (current_date - datetime.timedelta(days=args.window)).strftime("%Y-%m-%d")
            
            # Retrieve events whose execution_date falls in the window [cutoff_date, date_str]
            events = []
            for d in dates:
                if cutoff_date <= d <= date_str:
                    events.extend(events_by_exec_date.get(d, []))
                
            if not events:
                continue
                
            scores = calculate_scores(events)
            
            for ticker, info in scores.items():
                score = info['score']
                direction = info['direction']
                
                if abs(score) >= 5:
                    price_ret = get_cached_yfinance_history(ticker, date_str, args.horizon)
                    if price_ret is not None:
                        entry_p, exit_p = price_ret
                        trade_ret = (exit_p - entry_p) / entry_p if direction == 'BULLISH' else (entry_p - exit_p) / entry_p
                        trades.append({
                            "date": date_str, "ticker": ticker, "direction": direction, "score": score, "return": trade_ret, "source": "YahooFinance"
                        })
                        
        if not trades:
            print("[backtester] No signals generated during the backtest window.")
            return 0
            
        print(f"\n[backtester] Backtest Results Summary ({len(trades)} trades generated):")
        print("=" * 80)
        print(f"{'Date':<12} {'Ticker':<12} {'Direction':<10} {'Score':<6} {'Return (%)':<12} {'Source':<12}")
        print("-" * 80)
        
        total_ret = 0.0
        wins = 0
        for t in trades[:30]:
            ret_pct = t['return'] * 100.0
            print(f"{t['date']:<12} {t['ticker']:<12} {t['direction']:<10} {t['score']:<6} {ret_pct:>+10.2f}%   {t['source']:<12}")
            total_ret += t['return']
            if t['return'] > 0:
                wins += 1
                
        if len(trades) > 30:
            print(f"  ... and {len(trades) - 30} more trades")

        win_rate = wins / len(trades) * 100.0 if trades else 0.0
        avg_ret  = total_ret / len(trades) * 100.0 if trades else 0.0
        print("=" * 80)
        print(f"Total trades: {len(trades)}  |  Win rate: {win_rate:.1f}%  |  Avg return: {avg_ret:+.2f}%")

    # ── Portfolio simulation ──────────────────────────────────────────────────
    print("\n[backtester] Running portfolio simulation …")
    port_result = simulate_strategy_portfolio(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        max_positions=args.max_pos,
        horizon_days=args.horizon,
    )

    if port_result:
        m = port_result.get("metrics", {})
        print(f"  CAGR:      {m.get('cagr', 0)*100:+.2f}%")
        print(f"  Sharpe:    {m.get('sharpe', 0):.3f}")
        print(f"  Sortino:   {m.get('sortino', 0):.3f}")
        print(f"  Max DD:    {m.get('mdd', 0)*100:.2f}%")
        print(f"  Trades:    {port_result.get('n_trades', 0)}")
    else:
        print("  (no portfolio data — check DB has market_events and historical_prices)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

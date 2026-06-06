#!/usr/bin/env python3
"""
nav_accuracy_audit.py — Compares linear-interpolation MTM vs true daily MTM.

Pulls real backtester trades from market_events + historical_prices, runs the
same portfolio simulation two ways, then computes the delta on every risk metric.

Decision rule (printed at the end):
  Sharpe delta < 0.10 AND Max DD delta < 2%  → keep interpolation (acceptable)
  Otherwise                                   → recommend switching to true MTM

Usage:
  python nav_accuracy_audit.py                 # 10-day horizon, 5 max positions
  python nav_accuracy_audit.py --horizon 20    # longer hold
  python nav_accuracy_audit.py --start 2023-01-01
"""
from __future__ import annotations

import argparse
import datetime
import math
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DB_PATH

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


# ── DB helpers ────────────────────────────────────────────────────────────────

def _ro_conn() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB_PATH}?immutable=1", uri=True)


def load_events(start: str, end: str) -> list[dict]:
    """Load PROMOTER_BUY / BULK_DEAL / BLOCK_DEAL events with price-history coverage."""
    conn = _ro_conn()
    rows = conn.execute(
        "SELECT me.ticker, me.event_date, me.direction "
        "FROM market_events me "
        "WHERE me.event_type IN ('PROMOTER_BUY','BULK_DEAL','BLOCK_DEAL') "
        "AND me.direction = 'BULLISH' "
        "AND me.event_date BETWEEN ? AND ? "
        "AND EXISTS (SELECT 1 FROM historical_prices hp WHERE hp.ticker = me.ticker) "
        "ORDER BY me.event_date ASC",
        (start, end),
    ).fetchall()
    conn.close()
    return [{"ticker": r[0], "date": r[1], "direction": r[2]} for r in rows]


def load_prices(ticker: str) -> dict[str, float]:
    """Return {date_str: close} for a ticker."""
    conn = _ro_conn()
    rows = conn.execute(
        "SELECT date, close FROM historical_prices WHERE ticker=? ORDER BY date ASC",
        (ticker,),
    ).fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


# ── Trade execution logic (mirrors backtester) ────────────────────────────────

def get_entry_exit(
    prices: dict[str, float],
    signal_date: str,
    horizon: int,
) -> tuple[float, float, str, str] | None:
    """
    Return (entry_price, exit_price, entry_date, exit_date).
    Entry = next available close AFTER signal_date (T+1).
    Exit  = close horizon trading days later.
    """
    sorted_dates = sorted(prices.keys())
    after = [d for d in sorted_dates if d > signal_date]
    if len(after) < horizon + 1:
        return None
    entry_date = after[0]
    exit_date  = after[horizon]
    entry_price = prices[entry_date]
    exit_price  = prices[exit_date]
    if entry_price <= 0:
        return None
    return entry_price, exit_price, entry_date, exit_date


def get_daily_prices_for_hold(
    prices: dict[str, float],
    entry_date: str,
    exit_date: str,
) -> list[float]:
    """Return ordered closes from entry_date to exit_date (inclusive)."""
    return [prices[d] for d in sorted(prices.keys())
            if entry_date <= d <= exit_date and d in prices]


# ── Portfolio simulation: two flavours ───────────────────────────────────────

def _metrics(equity_curve: list[float], initial: float) -> dict:
    """Compute CAGR, Sharpe, Sortino, Max Drawdown from a daily NAV series."""
    if len(equity_curve) < 2:
        return {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0, "mdd": 0.0}

    years = len(equity_curve) / 252
    final = equity_curve[-1]
    cagr  = (final / initial) ** (1.0 / years) - 1.0 if final > 0 and years > 0 else -1.0

    rets = [(equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            for i in range(1, len(equity_curve))]

    if not rets:
        return {"cagr": cagr, "sharpe": 0.0, "sortino": 0.0, "mdd": 0.0}

    mean_r  = sum(rets) / len(rets)
    var_r   = sum((r - mean_r) ** 2 for r in rets) / len(rets)
    std_r   = math.sqrt(var_r) or 1e-9
    rf_daily = (1.065 ** (1 / 252)) - 1

    sharpe  = math.sqrt(252) * (mean_r - rf_daily) / std_r

    dd_sq   = [min(0.0, r - rf_daily) ** 2 for r in rets]
    dd_dev  = math.sqrt(sum(dd_sq) / len(rets)) or 1e-9
    sortino = math.sqrt(252) * (mean_r - rf_daily) / dd_dev

    peak, mdd = initial, 0.0
    running = initial
    for r in rets:
        running *= (1 + r)
        peak = max(peak, running)
        mdd  = max(mdd, (peak - running) / peak)

    return {"cagr": cagr, "sharpe": sharpe, "sortino": sortino, "mdd": mdd}


def simulate_linear(
    trades: list[dict],
    initial_capital: float,
    max_pos: int,
) -> tuple[list[float], dict]:
    """Linear interpolation MTM — current backtester approach."""
    # Build a continuous date axis
    all_dates = sorted({d for t in trades for d in [t["entry_date"], t["exit_date"]]})
    if not all_dates:
        return [], {}

    start = datetime.date.fromisoformat(all_dates[0])
    end   = datetime.date.fromisoformat(all_dates[-1])
    axis: list[str] = []
    cur = start
    while cur <= end:
        axis.append(cur.isoformat())
        cur += datetime.timedelta(days=1)

    by_entry: dict[str, list[dict]] = {}
    for t in trades:
        by_entry.setdefault(t["entry_date"], []).append(t)

    cash     = initial_capital
    active: list[dict] = []
    curve:  list[float] = []

    for today_str in axis:
        today = datetime.date.fromisoformat(today_str)

        # Exits
        remaining = []
        for pos in active:
            if today_str >= pos["exit_date"]:
                cash += pos["allocated"] * (1.0 + pos["total_return"])
            else:
                remaining.append(pos)
        active = remaining

        # Entries
        if today_str in by_entry and len(active) < max_pos:
            slots  = max_pos - len(active)
            new_t  = by_entry[today_str][:slots]
            if new_t and cash > 0:
                alloc = cash / slots
                for t in new_t:
                    if cash >= alloc:
                        cash -= alloc
                        active.append({
                            "entry_date":   today_str,
                            "exit_date":    t["exit_date"],
                            "allocated":    alloc,
                            "total_return": t["total_return"],
                        })

        # Linear MTM
        mtm = 0.0
        for pos in active:
            hold = max((datetime.date.fromisoformat(pos["exit_date"])
                        - datetime.date.fromisoformat(pos["entry_date"])).days, 1)
            elapsed  = (today - datetime.date.fromisoformat(pos["entry_date"])).days
            fraction = min(elapsed / hold, 1.0)
            mtm += pos["allocated"] * (1.0 + pos["total_return"] * fraction)

        curve.append(cash + mtm)

    return curve, _metrics(curve, initial_capital)


def simulate_true_mtm(
    trades: list[dict],
    initial_capital: float,
    max_pos: int,
) -> tuple[list[float], dict]:
    """True MTM using actual daily closes from historical_prices."""
    all_dates = sorted({d for t in trades for d in [t["entry_date"], t["exit_date"]]})
    if not all_dates:
        return [], {}

    start = datetime.date.fromisoformat(all_dates[0])
    end   = datetime.date.fromisoformat(all_dates[-1])
    axis: list[str] = []
    cur = start
    while cur <= end:
        axis.append(cur.isoformat())
        cur += datetime.timedelta(days=1)

    by_entry: dict[str, list[dict]] = {}
    for t in trades:
        by_entry.setdefault(t["entry_date"], []).append(t)

    cash     = initial_capital
    active: list[dict] = []
    curve:  list[float] = []

    for today_str in axis:
        today = datetime.date.fromisoformat(today_str)

        # Exits
        remaining = []
        for pos in active:
            if today_str >= pos["exit_date"]:
                # Realise at actual exit price
                exit_p = pos["prices"].get(pos["exit_date"], pos["entry_price"])
                realized = pos["shares"] * exit_p
                cash += realized
            else:
                remaining.append(pos)
        active = remaining

        # Entries
        if today_str in by_entry and len(active) < max_pos:
            slots  = max_pos - len(active)
            new_t  = by_entry[today_str][:slots]
            if new_t and cash > 0:
                alloc = cash / slots
                for t in new_t:
                    if cash >= alloc and t["entry_price"] > 0:
                        cash -= alloc
                        shares = alloc / t["entry_price"]
                        active.append({
                            "entry_date":  today_str,
                            "exit_date":   t["exit_date"],
                            "entry_price": t["entry_price"],
                            "shares":      shares,
                            "prices":      t["prices"],
                        })

        # True MTM: shares × today's close (or last known close)
        mtm = 0.0
        for pos in active:
            # Walk back from today to find the most recent available price
            p = pos["prices"].get(today_str)
            if p is None:
                # Use most recent price up to today
                candidates = [v for k, v in pos["prices"].items() if k <= today_str]
                p = candidates[-1] if candidates else pos["entry_price"]
            mtm += pos["shares"] * p

        curve.append(cash + mtm)

    return curve, _metrics(curve, initial_capital)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="NAV accuracy audit: linear vs true MTM")
    parser.add_argument("--horizon", type=int, default=10, help="Hold period in trading days")
    parser.add_argument("--start",   default="2022-01-01", help="Backtest start date")
    parser.add_argument("--end",     default="2025-12-31", help="Backtest end date")
    parser.add_argument("--max-pos", type=int, default=5,  help="Max concurrent positions")
    parser.add_argument("--capital", type=float, default=10_000_000, help="Initial capital")
    args = parser.parse_args()

    print(f"[nav_audit] Loading events {args.start} → {args.end}, horizon={args.horizon}d …")
    raw_events = load_events(args.start, args.end)
    print(f"[nav_audit] {len(raw_events)} qualifying events found")

    if not raw_events:
        print("[nav_audit] No events — check DB has market_events and historical_prices populated")
        return 1

    # Enrich events with price data
    print("[nav_audit] Loading price histories …")
    price_cache: dict[str, dict[str, float]] = {}
    trades: list[dict] = []

    for ev in raw_events:
        ticker = ev["ticker"]
        if ticker not in price_cache:
            price_cache[ticker] = load_prices(ticker)
        prices = price_cache[ticker]
        if not prices:
            continue

        result = get_entry_exit(prices, ev["date"], args.horizon)
        if result is None:
            continue

        entry_price, exit_price, entry_date, exit_date = result
        total_return = (exit_price - entry_price) / entry_price

        # Get the daily price series for the hold period (for true MTM)
        hold_prices = {d: p for d, p in prices.items() if entry_date <= d <= exit_date}

        trades.append({
            "ticker":       ticker,
            "signal_date":  ev["date"],
            "entry_date":   entry_date,
            "exit_date":    exit_date,
            "entry_price":  entry_price,
            "exit_price":   exit_price,
            "total_return": total_return,
            "prices":       hold_prices,
        })

    print(f"[nav_audit] {len(trades)} trades with full price history")

    if not trades:
        print("[nav_audit] No trades with price data — exiting")
        return 1

    # Run both simulations
    print("[nav_audit] Running linear-interpolation simulation …")
    lin_curve, lin_m = simulate_linear(trades, args.capital, args.max_pos)

    print("[nav_audit] Running true mark-to-market simulation …")
    mtm_curve, mtm_m = simulate_true_mtm(trades, args.capital, args.max_pos)

    # Print comparison
    header = f"\n{'Metric':<20} {'Linear MTM':>14} {'True MTM':>14} {'Delta':>12} {'Decision':>12}"
    sep    = "-" * 74
    print(header)
    print(sep)

    metrics_def = [
        ("CAGR (%)",   "cagr",    100, 2, lambda d: "ok" if abs(d) < 1.0 else "⚠ review"),
        ("Sharpe",     "sharpe",    1, 3, lambda d: "ok" if abs(d) < 0.10 else "⚠ review"),
        ("Sortino",    "sortino",   1, 3, lambda d: "ok" if abs(d) < 0.15 else "⚠ review"),
        ("Max DD (%)", "mdd",     100, 2, lambda d: "ok" if abs(d) < 2.0  else "⚠ review"),
    ]

    all_ok = True
    rows_out = []
    for label, key, scale, dec, verdict_fn in metrics_def:
        lv    = lin_m.get(key, 0.0) * scale
        tv    = mtm_m.get(key, 0.0) * scale
        delta = tv - lv
        verdict = verdict_fn(delta)
        if "review" in verdict:
            all_ok = False
        fmt = f".{dec}f"
        row = f"{label:<20} {lv:>{14}.{dec}f} {tv:>{14}.{dec}f} {delta:>{12}.{dec}f} {verdict:>12}"
        print(row)
        rows_out.append((label, lv, tv, delta, verdict))

    print(sep)
    print(f"\n{'Trades simulated:':<30} {len(trades)}")
    print(f"{'Equity curve length (days):':<30} {len(lin_curve)}")

    if all_ok:
        print("\n✅ VERDICT: Linear interpolation is acceptable.")
        print("   Sharpe delta < 0.10 and Max DD delta < 2% — keep current approach.")
        verdict_str = "KEEP INTERPOLATION"
    else:
        print("\n⚠️  VERDICT: Differences exceed thresholds.")
        print("   Consider switching simulate_strategy_portfolio() to true daily MTM.")
        verdict_str = "CONSIDER TRUE MTM"

    # Write report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "nav_accuracy_audit.md"
    lines = [
        "# NAV Accuracy Audit: Linear Interpolation vs True MTM",
        "",
        f"**Backtest period**: {args.start} → {args.end}  ",
        f"**Hold horizon**: {args.horizon} trading days  ",
        f"**Max positions**: {args.max_pos}  ",
        f"**Trades simulated**: {len(trades)}  ",
        f"**Equity curve length**: {len(lin_curve)} days  ",
        "",
        "## Results",
        "",
        f"| Metric | Linear MTM | True MTM | Delta | Decision |",
        f"|:-------|----------:|--------:|------:|:---------|",
    ]
    for label, lv, tv, delta, verdict in rows_out:
        lines.append(f"| {label} | {lv:.3f} | {tv:.3f} | {delta:+.3f} | {verdict} |")
    lines += [
        "",
        f"## Verdict: **{verdict_str}**",
        "",
        ("Linear interpolation is acceptable for this dataset." if all_ok else
         "Metric differences exceed thresholds — review `simulate_strategy_portfolio()`."),
        "",
        f"*Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[nav_audit] Report written → {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

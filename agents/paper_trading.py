#!/usr/bin/env python3
"""
paper_trading.py — DB-backed paper portfolio tracker.

CLI usage:
  python paper_trading.py add  RELIANCE BUY  10 2450.50   [--reason "insider buy signal"]
  python paper_trading.py add  HDFCBANK SELL  5 1620.00   [--reason "bulk deal sell"]
  python paper_trading.py close <trade_id> [--price 2510.00]   # uses latest DB price if omitted
  python paper_trading.py view                                  # show open trades + P&L
  python paper_trading.py history [--days 30]                  # closed trades
  python paper_trading.py summary                              # total P&L summary

Telegram command integration (called from gian.py or a bot webhook):
  /paper add RELIANCE BUY 10 2450.50
  /paper close 3
  /paper view
  /paper summary

The schema (paper_trades table) is created by common.py initialize_db().
Current price is looked up from historical_prices (populated by yfinance cache).
If no cached price exists, entry_price is used and a warning is shown.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _conn, log

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

_BOT_TOKEN_KEY = "TELEGRAM_BOT_TOKEN"
_CHAT_ID_KEY   = "TELEGRAM_CHAT_ID"


# ── DB helpers ────────────────────────────────────────────────────────────────

class Trade(NamedTuple):
    trade_id:    int
    ticker:      str
    side:        str
    qty:         int
    entry_price: float
    entry_date:  str
    exit_price:  float | None
    exit_date:   str | None
    status:      str
    reason:      str | None


def _latest_price(ticker: str) -> float | None:
    """Fetch the most recent close from historical_prices cache."""
    with _conn() as c:
        row = c.execute(
            "SELECT close FROM historical_prices "
            "WHERE ticker=? ORDER BY date DESC LIMIT 1",
            (ticker,),
        ).fetchone()
    return float(row[0]) if row else None


def _open_trades() -> list[Trade]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id,ticker,side,qty,entry_price,entry_date,"
            "exit_price,exit_date,status,reason "
            "FROM paper_trades WHERE status='OPEN' ORDER BY entry_date ASC"
        ).fetchall()
    return [Trade(*r) for r in rows]


def _all_trades(days: int | None = None) -> list[Trade]:
    with _conn() as c:
        if days is not None:
            cutoff = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
            rows = c.execute(
                "SELECT id,ticker,side,qty,entry_price,entry_date,"
                "exit_price,exit_date,status,reason "
                "FROM paper_trades WHERE entry_date >= ? ORDER BY entry_date DESC",
                (cutoff,),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT id,ticker,side,qty,entry_price,entry_date,"
                "exit_price,exit_date,status,reason "
                "FROM paper_trades ORDER BY entry_date DESC LIMIT 50"
            ).fetchall()
    return [Trade(*r) for r in rows]


def _add_trade(ticker: str, side: str, qty: int, price: float, reason: str | None) -> int:
    today = datetime.date.today().isoformat()
    ts    = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with _conn() as c:
        if c.is_postgres:
            cur = c.execute(
                "INSERT INTO paper_trades (ticker,side,qty,entry_price,entry_date,status,reason,ts) "
                "VALUES (?,?,?,?,?,?,?,?) RETURNING id",
                (ticker, side, qty, price, today, "OPEN", reason, ts),
            )
            return int(cur.fetchone()[0])
        else:
            c.execute(
                "INSERT INTO paper_trades (ticker,side,qty,entry_price,entry_date,status,reason,ts) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (ticker, side, qty, price, today, "OPEN", reason, ts),
            )
            cur2 = c.execute("SELECT last_insert_rowid()")
            return int(cur2.fetchone()[0])


def _close_trade(trade_id: int, exit_price: float) -> bool:
    today = datetime.date.today().isoformat()
    with _conn() as c:
        c.execute(
            "UPDATE paper_trades SET status='CLOSED',exit_price=?,exit_date=? "
            "WHERE id=? AND status='OPEN'",
            (exit_price, today, trade_id),
        )
        # Check rows affected
        row = c.execute(
            "SELECT id FROM paper_trades WHERE id=? AND status='CLOSED'", (trade_id,)
        ).fetchone()
    return row is not None


# ── P&L calculation ───────────────────────────────────────────────────────────

def _pnl(t: Trade, current: float) -> float:
    if t.side == "BUY":
        return (current - t.entry_price) * t.qty
    else:  # SELL / short
        return (t.entry_price - current) * t.qty


def _pct(t: Trade, current: float) -> float:
    cost = t.entry_price * t.qty
    return (_pnl(t, current) / cost * 100) if cost != 0 else 0.0


# ── Command handlers ──────────────────────────────────────────────────────────

def cmd_add(ticker: str, side: str, qty: int, price: float, reason: str | None) -> str:
    ticker = ticker.upper()
    side   = side.upper()
    if side not in ("BUY", "SELL"):
        return f"❌ Side must be BUY or SELL, got: {side}"
    if qty <= 0:
        return "❌ Quantity must be > 0"
    if price <= 0:
        return "❌ Price must be > 0"

    trade_id = _add_trade(ticker, side, qty, price, reason)
    cost = qty * price
    log("paper_trading", f"Added trade #{trade_id}: {side} {qty} {ticker} @ ₹{price:.2f}")
    return (
        f"✅ Paper trade #{trade_id} added\n"
        f"  {side} {qty} × {ticker} @ ₹{price:,.2f}\n"
        f"  Cost: ₹{cost:,.0f}"
        + (f"\n  Note: {reason}" if reason else "")
    )


def cmd_close(trade_id: int, exit_price: float | None) -> str:
    # Get the trade first
    with _conn() as c:
        row = c.execute(
            "SELECT id,ticker,side,qty,entry_price,entry_date,"
            "exit_price,exit_date,status,reason "
            "FROM paper_trades WHERE id=? AND status='OPEN'",
            (trade_id,),
        ).fetchone()
    if not row:
        return f"❌ No open trade with id #{trade_id}"

    t = Trade(*row)

    if exit_price is None:
        exit_price = _latest_price(t.ticker)
        if exit_price is None:
            return (
                f"⚠️ No cached price for {t.ticker}.\n"
                f"Run: python fundamental_collector.py --ticker {t.ticker}\n"
                f"Or provide price manually: close {trade_id} --price <price>"
            )

    ok = _close_trade(trade_id, exit_price)
    if not ok:
        return f"❌ Failed to close trade #{trade_id}"

    pnl = _pnl(t, exit_price)
    pct = _pct(t, exit_price)
    sign = "▲" if pnl >= 0 else "▼"
    log("paper_trading", f"Closed trade #{trade_id}: {t.ticker} exit ₹{exit_price:.2f} PnL ₹{pnl:.0f}")
    return (
        f"✅ Trade #{trade_id} closed\n"
        f"  {t.side} {t.qty} × {t.ticker}\n"
        f"  Entry ₹{t.entry_price:,.2f} → Exit ₹{exit_price:,.2f}\n"
        f"  P&L: {sign} ₹{abs(pnl):,.0f} ({pct:+.1f}%)"
    )


def cmd_view() -> str:
    trades = _open_trades()
    if not trades:
        return "📭 No open paper trades.\nAdd one: /paper add <TICKER> <BUY|SELL> <QTY> <PRICE>"

    lines = ["<b>📈 Open Paper Trades</b>"]
    total_pnl   = 0.0
    total_cost  = 0.0
    price_warn  = []

    for t in trades:
        current = _latest_price(t.ticker) or t.entry_price
        if _latest_price(t.ticker) is None:
            price_warn.append(t.ticker)
        pnl  = _pnl(t, current)
        pct  = _pct(t, current)
        cost = t.entry_price * t.qty
        total_pnl  += pnl
        total_cost += cost
        sign = "▲" if pnl >= 0 else "▼"
        lines.append(
            f"#{t.trade_id} {t.side} {t.qty}×<b>{t.ticker}</b> "
            f"@ ₹{t.entry_price:,.1f} | now ₹{current:,.1f} "
            f"| {sign} ₹{abs(pnl):,.0f} ({pct:+.1f}%) [{t.entry_date}]"
        )

    total_pct = (total_pnl / total_cost * 100) if total_cost else 0.0
    sign = "▲" if total_pnl >= 0 else "▼"
    lines.append(f"\n<b>Total P&L: {sign} ₹{abs(total_pnl):,.0f} ({total_pct:+.1f}%)</b>")
    if price_warn:
        lines.append(f"⚠️ Using entry price for (no cache): {', '.join(price_warn)}")
    return "\n".join(lines)


def cmd_history(days: int = 30) -> str:
    trades = [t for t in _all_trades(days=days) if t.status == "CLOSED"]
    if not trades:
        return f"No closed trades in the last {days} days."

    lines = [f"<b>📋 Closed Trades (last {days}d)</b>"]
    total_pnl = 0.0
    wins = 0

    for t in trades:
        ep = t.exit_price or t.entry_price
        pnl  = _pnl(t, ep)
        pct  = _pct(t, ep)
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        sign = "▲" if pnl >= 0 else "▼"
        lines.append(
            f"#{t.trade_id} {t.side} {t.qty}×{t.ticker} "
            f"₹{t.entry_price:,.1f}→₹{ep:,.1f} "
            f"{sign} ₹{abs(pnl):,.0f} ({pct:+.1f}%)"
        )

    win_rate = wins / len(trades) * 100 if trades else 0
    sign = "▲" if total_pnl >= 0 else "▼"
    lines.append(
        f"\n<b>Total: {sign} ₹{abs(total_pnl):,.0f} | "
        f"Win rate: {win_rate:.0f}% ({wins}/{len(trades)})</b>"
    )
    return "\n".join(lines)


def cmd_summary() -> str:
    all_closed = [t for t in _all_trades() if t.status == "CLOSED"]
    open_trades = _open_trades()

    lines = ["<b>📊 Paper Portfolio Summary</b>"]

    # Closed trades stats
    if all_closed:
        realised = sum(_pnl(t, t.exit_price or t.entry_price) for t in all_closed)
        wins = sum(1 for t in all_closed if _pnl(t, t.exit_price or t.entry_price) > 0)
        win_rate = wins / len(all_closed) * 100
        sign = "▲" if realised >= 0 else "▼"
        lines.append(f"Realised P&L:  {sign} ₹{abs(realised):,.0f} ({win_rate:.0f}% win rate over {len(all_closed)} trades)")
    else:
        lines.append("Realised P&L:  – (no closed trades yet)")

    # Open trades stats
    if open_trades:
        unrealised = sum(_pnl(t, _latest_price(t.ticker) or t.entry_price) for t in open_trades)
        invested   = sum(t.entry_price * t.qty for t in open_trades)
        sign = "▲" if unrealised >= 0 else "▼"
        lines.append(f"Unrealised P&L: {sign} ₹{abs(unrealised):,.0f} on ₹{invested:,.0f} invested ({len(open_trades)} open)")
    else:
        lines.append("Unrealised P&L: – (no open trades)")

    lines.append(f"\nTotal trades: {len(all_closed) + len(open_trades)} ({len(open_trades)} open, {len(all_closed)} closed)")
    return "\n".join(lines)


# ── Telegram send helper ──────────────────────────────────────────────────────

def _send_telegram(text: str) -> None:
    if not _HAS_REQUESTS:
        print(text)
        return
    token   = os.environ.get(_BOT_TOKEN_KEY)
    chat_id = os.environ.get(_CHAT_ID_KEY)
    if not token or not chat_id:
        print(text)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    _requests.post(
        url,
        json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
        timeout=15,
    ).raise_for_status()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paper trading tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  add TICKER SIDE QTY PRICE [--reason TEXT]   Add a paper trade
  close ID [--price PRICE]                    Close a trade (uses cached price if omitted)
  view                                         Show open trades with live P&L
  history [--days N]                           Show closed trades (default 30 days)
  summary                                      Overall P&L summary
        """,
    )
    parser.add_argument("command", choices=["add", "close", "view", "history", "summary"])
    parser.add_argument("args", nargs="*", help="Command arguments")
    parser.add_argument("--reason", default=None, help="Optional note for the trade")
    parser.add_argument("--price",  type=float, default=None, help="Exit price for close command")
    parser.add_argument("--days",   type=int,   default=30,   help="History window in days")
    parser.add_argument("--telegram", action="store_true", help="Send output to Telegram as well")
    opts = parser.parse_args()

    msg: str

    if opts.command == "add":
        if len(opts.args) < 4:
            parser.error("add requires: TICKER SIDE QTY PRICE")
        try:
            ticker = opts.args[0]
            side   = opts.args[1]
            qty    = int(opts.args[2])
            price  = float(opts.args[3])
        except (ValueError, IndexError) as exc:
            parser.error(f"Invalid arguments: {exc}")
        msg = cmd_add(ticker, side, qty, price, opts.reason)

    elif opts.command == "close":
        if not opts.args:
            parser.error("close requires a trade ID")
        try:
            trade_id = int(opts.args[0])
        except ValueError:
            parser.error("Trade ID must be an integer")
        msg = cmd_close(trade_id, opts.price)

    elif opts.command == "view":
        msg = cmd_view()

    elif opts.command == "history":
        msg = cmd_history(days=opts.days)

    elif opts.command == "summary":
        msg = cmd_summary()

    else:
        parser.error(f"Unknown command: {opts.command}")

    # Strip HTML tags for terminal output
    import re
    plain = re.sub(r"<[^>]+>", "", msg)
    print(plain)

    if opts.telegram:
        try:
            _send_telegram(msg)
            print("[paper] Sent to Telegram ✓")
        except Exception as exc:
            sys.stderr.write(f"[paper] Telegram send failed: {exc}\n")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

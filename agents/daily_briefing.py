#!/usr/bin/env python3
"""
daily_briefing.py — Morning market digest via Telegram.

Runs at 08:00 IST every trading day (after nse_collector + event_detector
have already populated the DB). Sends a single, structured Telegram message
covering:

  1. Today's top-5 opportunities (by total_score from company_scores)
  2. Signals from the last 24 hours (from signals table)
  3. Active consensus alerts (from consensus table, not yet dispatched)
  4. Paper portfolio P&L snapshot (from paper_trades table, if any)
  5. Data freshness indicator

Usage:
  python daily_briefing.py             # send morning brief
  python daily_briefing.py --dry-run   # print to stdout without sending
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

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# Telegram env var names (must match what gian.py uses)
_BOT_TOKEN_KEY  = "TELEGRAM_BOT_TOKEN"
_CHAT_ID_KEY    = "TELEGRAM_CHAT_ID"


def _send_telegram(text: str) -> None:
    """Send a message via Telegram Bot API. Raises on failure."""
    if not _HAS_REQUESTS:
        raise RuntimeError("requests not installed. Run: pip install requests")
    token   = os.environ.get(_BOT_TOKEN_KEY)
    chat_id = os.environ.get(_CHAT_ID_KEY)
    if not token or not chat_id:
        raise RuntimeError(
            f"Set {_BOT_TOKEN_KEY} and {_CHAT_ID_KEY} in ~/indian-insider/.env"
        )
    url  = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = _requests.post(
        url,
        json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
        timeout=15,
    )
    resp.raise_for_status()


# ── Data fetchers ─────────────────────────────────────────────────────────────

def _top_opportunities(n: int = 5) -> list[dict]:
    """Top N tickers by total_score."""
    with _conn() as c:
        rows = c.execute(
            "SELECT cs.ticker, cs.total_score, cs.event_score, cs.fundamental_score, "
            "cs.credibility_score, cf.sector "
            "FROM company_scores cs "
            "LEFT JOIN company_fundamentals cf ON cs.ticker = cf.ticker "
            "WHERE cs.total_score > 0 "
            "ORDER BY cs.total_score DESC LIMIT ?",
            (n,),
        ).fetchall()
    return [
        {
            "ticker":      r[0],
            "total":       r[1],
            "event":       r[2],
            "fundamental": r[3],
            "credibility": r[4],
            "sector":      r[5] or "–",
        }
        for r in rows
    ]


def _recent_signals(hours: int = 24) -> list[dict]:
    """Signals emitted in the last N hours."""
    cutoff = (datetime.datetime.now(datetime.timezone.utc)
              - datetime.timedelta(hours=hours)).isoformat()
    with _conn() as c:
        rows = c.execute(
            "SELECT scout, ticker, direction, confidence, reason, ts "
            "FROM signals WHERE ts >= ? ORDER BY confidence DESC, ts DESC LIMIT 20",
            (cutoff,),
        ).fetchall()
    return [
        {"scout": r[0], "ticker": r[1], "direction": r[2],
         "confidence": r[3], "reason": r[4], "ts": r[5]}
        for r in rows
    ]


def _pending_consensus() -> list[dict]:
    """Consensus events not yet dispatched."""
    with _conn() as c:
        rows = c.execute(
            "SELECT ticker, direction, scouts, reasons, ts "
            "FROM consensus WHERE dispatched=0 ORDER BY ts DESC LIMIT 10",
        ).fetchall()
    result = []
    for ticker, direction, scouts_json, reasons_json, ts in rows:
        try:
            scouts  = json.loads(scouts_json)
            reasons = json.loads(reasons_json)
        except Exception:
            scouts  = [scouts_json]
            reasons = [reasons_json]
        result.append({"ticker": ticker, "direction": direction,
                        "scouts": scouts, "reasons": reasons, "ts": ts})
    return result


def _paper_pnl() -> list[dict]:
    """Open paper trades with unrealised P&L from latest historical price."""
    with _conn() as c:
        # Check if paper_trades table exists
        exists = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='paper_trades'"
        ).fetchone()
        if not exists:
            return []
        rows = c.execute(
            "SELECT pt.id, pt.ticker, pt.side, pt.qty, pt.entry_price, pt.entry_date, "
            "hp.close AS current_price "
            "FROM paper_trades pt "
            "LEFT JOIN historical_prices hp ON pt.ticker = hp.ticker "
            "  AND hp.date = (SELECT MAX(date) FROM historical_prices WHERE ticker = pt.ticker) "
            "WHERE pt.status = 'OPEN' "
            "ORDER BY pt.entry_date ASC",
        ).fetchall()
    result = []
    for trade_id, ticker, side, qty, entry, entry_date, current in rows:
        current = current or entry
        pnl = (current - entry) * qty if side == "BUY" else (entry - current) * qty
        pct = (pnl / (entry * qty) * 100) if entry * qty != 0 else 0.0
        result.append({
            "id": trade_id, "ticker": ticker, "side": side,
            "qty": qty, "entry": entry, "current": current,
            "pnl": pnl, "pct": pct, "entry_date": entry_date,
        })
    return result


def _data_freshness() -> str:
    """Return the latest event_date in market_events to show data age."""
    with _conn() as c:
        row = c.execute(
            "SELECT MAX(event_date) FROM market_events"
        ).fetchone()
    latest = row[0] if row and row[0] else "no data"
    today  = datetime.date.today().isoformat()
    if latest == today:
        return f"✅ fresh (as of {latest})"
    elif latest == "no data":
        return "⚠️ no market events — run nse_collector.py + event_detector.py"
    else:
        return f"⚠️ stale (last: {latest}, today: {today})"


# ── Message builder ───────────────────────────────────────────────────────────

_DIR_EMOJI = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}
_CONF_STAR  = {1: "★☆☆☆☆", 2: "★★☆☆☆", 3: "★★★☆☆", 4: "★★★★☆", 5: "★★★★★"}


def build_message(
    opportunities: list[dict],
    signals:        list[dict],
    consensus:      list[dict],
    paper_trades:   list[dict],
    freshness:      str,
) -> str:
    now_ist = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=5, minutes=30)
    lines: list[str] = []

    lines.append(f"<b>🌅 Indian Insider — Morning Brief</b>")
    lines.append(f"<i>{now_ist.strftime('%A, %d %b %Y %H:%M IST')}</i>")
    lines.append(f"Data: {freshness}")
    lines.append("")

    # ── Section 1: Top Opportunities ────────────────────────────────────────
    lines.append("<b>📊 Top Opportunities</b>")
    if opportunities:
        for i, op in enumerate(opportunities, 1):
            lines.append(
                f"{i}. <b>{op['ticker']}</b> — score {op['total']:.1f} "
                f"| ev {op['event']:.1f} | fund {op['fundamental']:.1f} "
                f"| cred {op['credibility']:.1f} | {op['sector']}"
            )
    else:
        lines.append("  No scored opportunities yet — run opportunity_engine.py")
    lines.append("")

    # ── Section 2: Today's Signals ──────────────────────────────────────────
    lines.append("<b>📡 Signals (last 24h)</b>")
    if signals:
        for s in signals:
            emoji = _DIR_EMOJI.get(s["direction"], "⬜")
            stars = _CONF_STAR.get(s["confidence"], "?")
            lines.append(
                f"{emoji} <b>{s['ticker']}</b> [{s['scout']}] {stars}"
            )
            lines.append(f"   <i>{s['reason']}</i>")
    else:
        lines.append("  No signals in last 24h — scouts haven't run yet today")
    lines.append("")

    # ── Section 3: Consensus Alerts ─────────────────────────────────────────
    lines.append("<b>🔔 Consensus Alerts</b>")
    if consensus:
        for c in consensus:
            emoji = _DIR_EMOJI.get(c["direction"], "⬜")
            scouts_str = ", ".join(c["scouts"])
            lines.append(
                f"{emoji} <b>{c['ticker']}</b> — {c['direction']} "
                f"({len(c['scouts'])} scouts: {scouts_str})"
            )
            if c["reasons"]:
                lines.append(f"   <i>{c['reasons'][0]}</i>")
    else:
        lines.append("  No pending consensus alerts")
    lines.append("")

    # ── Section 4: Paper Portfolio ───────────────────────────────────────────
    lines.append("<b>📈 Paper Portfolio</b>")
    if paper_trades:
        total_pnl = sum(t["pnl"] for t in paper_trades)
        lines.append(f"  Total unrealised P&L: {'▲' if total_pnl >= 0 else '▼'} ₹{abs(total_pnl):,.0f}")
        for t in paper_trades:
            sign = "▲" if t["pnl"] >= 0 else "▼"
            lines.append(
                f"  {t['ticker']} {t['side']} ×{t['qty']} | entry ₹{t['entry']:,.1f} "
                f"→ ₹{t['current']:,.1f} | {sign} ₹{abs(t['pnl']):,.0f} ({t['pct']:+.1f}%)"
            )
    else:
        lines.append("  No open paper trades — use paper_trading.py to add")
    lines.append("")

    lines.append("<i>Reply /help for commands • Indian Insider v1.0</i>")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Send morning briefing via Telegram")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print message to stdout instead of sending")
    args = parser.parse_args()

    try:
        ops        = _top_opportunities()
        sigs       = _recent_signals()
        cons       = _pending_consensus()
        paper      = _paper_pnl()
        freshness  = _data_freshness()
    except Exception as exc:
        sys.stderr.write(f"[briefing] DB read failed: {exc}\n")
        log("briefing", f"DB read error: {exc}")
        return 1

    message = build_message(ops, sigs, cons, paper, freshness)

    if args.dry_run:
        print(message)
        return 0

    try:
        _send_telegram(message)
        print("[briefing] Morning brief sent ✓")
        log("briefing", f"Sent: {len(ops)} opps, {len(sigs)} signals, {len(cons)} consensus, {len(paper)} trades")
    except Exception as exc:
        sys.stderr.write(f"[briefing] Telegram send failed: {exc}\n")
        log("briefing", f"Telegram error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Gian — dispatcher.
Pure local logic — zero Gemini API calls.
Sends Gmail (always) + Telegram (optional) for pending consensus events.
NEVER places trades.
Schedule: every 30 minutes.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import log, mark_dispatched, pending_consensus, render_consensus, send_email, send_telegram
from stocklens_bridge import push_consensus


def main() -> int:
    pending = pending_consensus()
    if not pending:
        log("gian", "no pending consensus events")
        print("[gian] nothing to dispatch")
        return 0

    delivered = 0
    for row_id, ev in pending:
        body    = render_consensus(ev)
        subject = f"[INDIA INSIDER] CONSENSUS {ev.direction} on {ev.ticker}"
        
        email_ok = False
        telegram_ok = False
        stocklens_ok = False

        # StockLens signal queue (primary product UI)
        try:
            stocklens_ok = push_consensus(ev, row_id)
            if stocklens_ok:
                log("gian", f"stocklens pushed for [{row_id}] {ev.ticker}")
        except Exception as exc:
            log("gian", f"stocklens FAILED for [{row_id}]: {exc}")
            print(f"[gian] stocklens FAILED for {ev.ticker}: {exc}")

        # Try Email
        try:
            send_email(subject, body)
            log("gian", f"email sent for [{row_id}] {ev.ticker}")
            email_ok = True
        except Exception as exc:
            log("gian", f"email FAILED for [{row_id}]: {exc}")
            print(f"[gian] email FAILED for {ev.ticker}: {exc}")
            
        # Try Telegram
        try:
            if send_telegram(f"*{subject}*\n\n```\n{body}\n```"):
                log("gian", f"telegram sent for [{row_id}]")
                telegram_ok = True
        except Exception as exc:
            log("gian", f"telegram FAILED for [{row_id}]: {exc}")
            print(f"[gian] telegram FAILED for {ev.ticker}: {exc}")
            
        if email_ok or telegram_ok or stocklens_ok:
            mark_dispatched(row_id)
            delivered += 1
            channels = []
            if stocklens_ok:
                channels.append("stocklens")
            if email_ok:
                channels.append("email")
            if telegram_ok:
                channels.append("telegram")
            print(f"[gian] dispatched {ev.direction} {ev.ticker} via {', '.join(channels)}")

    log("gian", f"delivered {delivered}/{len(pending)} pending events")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
stocklens_bridge.py — Push Indian Insider intel into StockLens signal_events.

StockLens ingests via POST /api/integrations/signals (see stocklens repo).

Env (engine/.env):
  STOCKLENS_URL=https://your-app.up.railway.app
  STOCKLENS_INTEGRATION_SECRET=optional-shared-secret
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    sys.stderr.write("Run: pip install requests\n")
    raise

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ConsensusEvent, Signal, _conn, log


def _base_url() -> str | None:
    url = (os.environ.get("STOCKLENS_URL") or "").strip().rstrip("/")
    return url or None


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    secret = os.environ.get("STOCKLENS_INTEGRATION_SECRET", "").strip()
    if secret:
        h["X-Integration-Secret"] = secret
    return h


def _event_hash(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def push_signal_payload(payload: dict) -> bool:
    """POST one signal to StockLens. Returns True on success."""
    base = _base_url()
    if not base:
        return False

    url = f"{base}/api/integrations/signals"
    try:
        resp = requests.post(url, json=payload, headers=_headers(), timeout=20)
        if resp.status_code == 200:
            data = resp.json() if resp.content else {}
            msg = data.get("message", "ok")
            log("stocklens", f"POST {payload.get('symbol')} {payload.get('event_type')}: {msg}")
            return True
        log("stocklens", f"POST failed {resp.status_code}: {resp.text[:200]}")
        print(f"[stocklens] POST failed {resp.status_code}: {resp.text[:200]}")
        return False
    except Exception as exc:
        log("stocklens", f"POST error: {exc}")
        print(f"[stocklens] POST error: {exc}")
        return False


def consensus_to_payload(ev: ConsensusEvent, row_id: int = 0) -> dict | None:
    ticker = (ev.ticker or "").upper().strip()
    if not ticker or ticker in {"MACRO", "GOLD"}:
        return None

    direction = ev.direction.upper()
    is_macro = ticker in {"NIFTY", "BANKNIFTY"}
    day = ev.timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d")

    if is_macro:
        event_type = f"macro_{direction.lower()}"
        category = "macro"
    else:
        event_type = "consensus_bullish" if direction == "BULLISH" else "consensus_bearish"
        category = "positioning"

    scout_count = len(ev.scouts or [])
    if scout_count >= 4:
        confidence = "HIGH"
        severity = "critical" if direction == "BEARISH" else "warning"
        strength = min(100, scout_count * 20 + 20)
    elif scout_count >= 2:
        confidence = "MEDIUM"
        severity = "warning"
        strength = min(100, scout_count * 15 + 30)
    else:
        confidence = "MEDIUM"
        severity = "warning"
        strength = min(100, 50 + scout_count * 10)

    if not is_macro and "promoter" in " ".join(ev.reasons).lower():
        strength = min(100, strength + 15)
        category = "governance"

    return {
        "symbol": ticker if not is_macro else "NIFTY",
        "event_type": event_type,
        "category": category,
        "severity": severity,
        "signal_strength": strength,
        "confidence": confidence,
        "source_engine": "indian_insider",
        "payload_json": {
            "direction": direction,
            "scouts": ev.scouts,
            "reasons": ev.reasons,
            "consensus_row_id": row_id,
            "engine": "indian_insider",
        },
        "event_hash": _event_hash("consensus", ticker, direction, day, str(row_id)),
        "cooldown_minutes": 1440,
        "ttl_days": 10,
    }


def push_consensus(ev: ConsensusEvent, row_id: int = 0) -> bool:
    if not _base_url():
        return False
    payload = consensus_to_payload(ev, row_id)
    if not payload:
        return False
    ok = push_signal_payload(payload)
    if ok:
        print(f"[stocklens] pushed consensus {ev.direction} {ev.ticker}")
    return ok


def scout_to_payload(sig: Signal) -> dict | None:
    ticker = (sig.ticker or "").upper().strip()
    if not ticker or ticker == "MACRO":
        return None
    if sig.direction == "NEUTRAL" and sig.confidence <= 2:
        return None

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    direction = sig.direction.upper()
    event_type = f"scout_{sig.scout}_{direction.lower()}"
    category = "macro" if ticker in {"NIFTY", "BANKNIFTY"} else "catalyst"

    conf_map = {1: "LOW", 2: "LOW", 3: "MEDIUM", 4: "HIGH", 5: "HIGH"}
    confidence = conf_map.get(sig.confidence, "MEDIUM")
    severity = "warning" if direction != "NEUTRAL" else "info"
    strength = min(100, sig.confidence * 18)

    return {
        "symbol": ticker if ticker not in {"NIFTY", "BANKNIFTY"} else "NIFTY",
        "event_type": event_type,
        "category": category,
        "severity": severity,
        "signal_strength": strength,
        "confidence": confidence,
        "source_engine": f"indian_insider_{sig.scout}",
        "payload_json": {
            "scout": sig.scout,
            "direction": direction,
            "confidence": sig.confidence,
            "reason": sig.reason,
        },
        "event_hash": _event_hash("scout", sig.scout, ticker, direction, day, sig.reason[:80]),
        "cooldown_minutes": 720,
        "ttl_days": 7,
    }


def push_scout_signal(sig: Signal) -> bool:
    if not _base_url():
        return False
    payload = scout_to_payload(sig)
    if not payload:
        return False
    return push_signal_payload(payload)


def sync_opportunities(limit: int = 15) -> int:
    """Push top opportunity scores from company_scores to StockLens."""
    if not _base_url():
        print("[stocklens] STOCKLENS_URL not set — skip opportunity sync")
        return 0

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _conn() as c:
        rows = c.execute(
            """SELECT ticker, total_score, event_score, fundamental_score, credibility_score
               FROM company_scores
               WHERE total_score > 0
               ORDER BY total_score DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()

    pushed = 0
    for rank, row in enumerate(rows, start=1):
        ticker, total, event_s, fund_s, cred_s = row
        if not ticker or ticker in {"NIFTY", "BANKNIFTY", "NIFTYBEES", "GOLDBEES"}:
            continue
        strength = int(min(100, max(0, float(total or 0))))
        confidence = "HIGH" if strength >= 75 else "MEDIUM" if strength >= 55 else "LOW"
        payload = {
            "symbol": ticker,
            "event_type": "opportunity_rank",
            "category": "catalyst",
            "severity": "info" if rank > 5 else "warning",
            "signal_strength": strength,
            "confidence": confidence,
            "source_engine": "indian_insider_opportunity",
            "payload_json": {
                "rank": rank,
                "total_score": float(total or 0),
                "event_score": float(event_s or 0),
                "fundamental_score": float(fund_s or 0),
                "credibility_score": float(cred_s or 0),
                "as_of": day,
            },
            "event_hash": _event_hash("opportunity", ticker, day, str(rank)),
            "cooldown_minutes": 1440,
            "ttl_days": 3,
        }
        if push_signal_payload(payload):
            pushed += 1

    print(f"[stocklens] synced {pushed}/{len(rows)} opportunity rankings")
    log("stocklens", f"synced {pushed} opportunity rankings")
    return pushed


def push_pending_consensus() -> int:
    """Push all undispatched consensus rows (fallback if Gian skipped)."""
    from common import pending_consensus

    if not _base_url():
        return 0
    n = 0
    for row_id, ev in pending_consensus():
        if push_consensus(ev, row_id):
            n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description="Indian Insider → StockLens bridge")
    parser.add_argument("--sync-opportunities", action="store_true", help="Push top ranked stocks")
    parser.add_argument("--push-pending", action="store_true", help="Push pending consensus events")
    parser.add_argument("--test", action="store_true", help="Send a test signal")
    parser.add_argument("--limit", type=int, default=15, help="Max opportunities to sync")
    args = parser.parse_args()

    if not _base_url():
        print("[stocklens] Set STOCKLENS_URL in engine/.env")
        return 1

    if args.test:
        ok = push_signal_payload({
            "symbol": "RELIANCE",
            "event_type": "integration_test",
            "category": "catalyst",
            "severity": "info",
            "signal_strength": 50,
            "confidence": "LOW",
            "source_engine": "indian_insider",
            "payload_json": {"message": "Indian Insider bridge test"},
            "event_hash": _event_hash("test", datetime.now(timezone.utc).isoformat()),
            "cooldown_minutes": 60,
            "ttl_days": 1,
        })
        if ok:
            print("[stocklens] test signal sent OK — check StockLens → Alerts → Signal queue")
        else:
            print("[stocklens] test signal FAILED — check STOCKLENS_URL and Railway app")
        return 0 if ok else 1

    if args.push_pending:
        n = push_pending_consensus()
        print(f"[stocklens] pushed {n} pending consensus event(s)")
        return 0

    if args.sync_opportunities or len(sys.argv) == 1:
        sync_opportunities(limit=args.limit)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

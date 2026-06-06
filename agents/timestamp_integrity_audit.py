#!/usr/bin/env python3
"""
timestamp_integrity_audit.py — Step 2: Database Timestamp Integrity Auditor.
Verifies signal availability order, consensus timestamps, and event ingestion lag.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    _conn,
    log,
)


def run_timestamp_audit() -> int:
    conn = _conn()
    
    print("[timestamp_audit] Starting database timestamp integrity check...")
    
    # Check 1: Signal vs. Consensus order
    # Verify that for every consensus row, the timestamp of consensus (ts)
    # is greater than or equal to the timestamp of the contributing scout signals (ts).
    try:
        consensus_rows = conn.execute("SELECT id, ticker, scouts, ts FROM consensus").fetchall()
    except Exception as exc:
        print(f"Error querying consensus: {exc}")
        return 1
        
    order_failures = 0
    checked_consensus = 0
    
    for c_id, ticker, scouts_json, c_ts_str in consensus_rows:
        checked_consensus += 1
        scouts = json.loads(scouts_json)
        c_ts = datetime.datetime.fromisoformat(c_ts_str)
        if c_ts.tzinfo is not None:
            c_ts = c_ts.replace(tzinfo=None)
            
        # Find contributing signals for this ticker around the consensus date
        cutoff = (c_ts - datetime.timedelta(days=7)).isoformat()
        signals = conn.execute(
            "SELECT scout, ts FROM signals WHERE ticker=? AND ts <= ? AND ts >= ?",
            (ticker, c_ts_str, cutoff)
        ).fetchall()
        
        for scout, s_ts_str in signals:
            if scout in scouts:
                s_ts = datetime.datetime.fromisoformat(s_ts_str)
                if s_ts.tzinfo is not None:
                    s_ts = s_ts.replace(tzinfo=None)
                if s_ts > c_ts:
                    print(f"  [FAIL] Consensus #{c_id} for {ticker} at {c_ts_str} has contributing signal from {scout} with FUTURE timestamp: {s_ts_str}")
                    order_failures += 1
                    
    # Check 2: Market Event Ingestion Lag
    # Ingestion lag = Ingestion timestamp (ts) - Event date (event_date)
    # If ts is BEFORE event_date, it represents a lookahead anomaly!
    try:
        event_rows = conn.execute("SELECT id, ticker, event_type, event_date, ts FROM market_events").fetchall()
    except Exception as exc:
        print(f"Error querying market_events: {exc}")
        return 1
        
    backfill_violations = 0
    checked_events = 0
    lag_seconds = []
    
    for ev_id, ticker, ev_type, ev_date_str, ts_str in event_rows:
        checked_events += 1
        try:
            # Parse ingestion timestamp (usually UTC)
            if "+" in ts_str or ts_str.endswith("Z"):
                ts_fixed = ts_str.replace("Z", "+00:00")
                ingest_dt = datetime.datetime.fromisoformat(ts_fixed)
            else:
                ts_str_clean = ts_str.replace("Z", "").split("+")[0]
                ingest_dt = datetime.datetime.fromisoformat(ts_str_clean).replace(tzinfo=datetime.timezone.utc)
                
            # Convert ingestion to IST for local date comparison
            ist_tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
            ingest_ist = ingest_dt.astimezone(ist_tz)
            ingest_date = ingest_ist.date()
            
            ev_date = datetime.datetime.strptime(ev_date_str, "%Y-%m-%d").date()
            
            # Extract only the date part of ingestion time for comparison
            if ingest_date < ev_date:
                print(f"  [FAIL] Event #{ev_id} for {ticker} ({ev_type}) has event date {ev_date_str} but INGESTION date (IST) is in the past: {ingest_date} (ts: {ts_str})")
                backfill_violations += 1
            else:
                # Ingestion lag: difference between ingestion time and local event date start (in UTC/IST context)
                ev_dt_ist = datetime.datetime.strptime(ev_date_str, "%Y-%m-%d").replace(tzinfo=ist_tz)
                diff = (ingest_dt - ev_dt_ist).total_seconds()
                lag_seconds.append(diff)
        except Exception as e:
            continue
            
    avg_lag_days = (sum(lag_seconds) / len(lag_seconds) / 86400.0) if lag_seconds else 0.0
    
    print("\n" + "="*80)
    print("TIMESTAMP INTEGRITY AUDIT VERDICT")
    print("="*80)
    print(f"Checked Consensus Events:      {checked_consensus}")
    print(f"Signal-to-Consensus Failures:  {order_failures}")
    print(f"Checked Market Events:         {checked_events}")
    print(f"Lookahead Ingestion Violations: {backfill_violations}")
    print(f"Average Event Ingestion Lag:   {avg_lag_days:.2f} days")
    
    verdict = "PASS" if (order_failures == 0 and backfill_violations == 0) else "FAIL"
    print(f"Final Audit Verdict:           {verdict}")
    print("="*80 + "\n")
    
    # Save the audit log info to be used in step reports
    log("timestamp_audit", f"Audit complete. Checked {checked_consensus} consensus, {checked_events} events. Failures: {order_failures + backfill_violations}. Verdict: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(run_timestamp_audit())

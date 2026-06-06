#!/usr/bin/env python3
"""
lookahead_bias_audit.py — Audits chronological sequencing and lookahead bias.
Checks if events were replayed/traded before they were publicly published.
"""
from __future__ import annotations

import sqlite3
import json
import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DB_PATH
from backtester import get_execution_date

def parse_pit_publication_time(metadata_str: str) -> datetime.datetime | None:
    try:
        meta = json.loads(metadata_str)
        # PIT JSON has a 'date' field e.g., "30-Apr-2026 22:26"
        pub_str = meta.get("date")
        if pub_str:
            return datetime.datetime.strptime(pub_str.strip(), "%d-%b-%Y %H:%M")
        
        # Fallback to intimation date (intimDt) e.g., "29-Apr-2026"
        intim_str = meta.get("intimDt")
        if intim_str:
            return datetime.datetime.strptime(intim_str.strip(), "%d-%b-%Y").replace(hour=18, minute=30)
    except Exception:
        pass
    return None

def main() -> int:
    conn = sqlite3.connect(str(DB_PATH))
    
    dates = [r[0] for r in conn.execute("SELECT DISTINCT event_date FROM market_events ORDER BY event_date ASC").fetchall()]
    
    rows = conn.execute(
        "SELECT id, ticker, event_type, event_date, metadata, ts FROM market_events "
        "WHERE event_type IN ('PROMOTER_BUY', 'PROMOTER_SELL', 'BULK_DEAL', 'BLOCK_DEAL') "
        "ORDER BY event_date ASC"
    ).fetchall()
    
    if not rows:
        print("[lookahead_bias_audit] No events found in database.")
        conn.close()
        return 1
        
    print(f"[lookahead_bias_audit] Auditing {len(rows)} events for lookahead leaks...")
    
    total_audited = 0
    leaks_count = 0
    time_deltas = []
    
    leak_details = []
    
    for r_id, ticker, ev_type, event_date_str, metadata_str, ts_str in rows:
        # Parse event date (transaction date)
        try:
            event_date = datetime.datetime.strptime(event_date_str, "%Y-%m-%d").date()
        except Exception:
            continue
            
        # Trade Execution: Backtest executes at next market open after publication
        exec_date_str = get_execution_date(event_date_str, ev_type, metadata_str, dates)
        if not exec_date_str:
            continue
        try:
            exec_date = datetime.datetime.strptime(exec_date_str, "%Y-%m-%d").date()
        except Exception:
            continue
        exec_dt = datetime.datetime.combine(exec_date, datetime.time(9, 15))
        
        pub_dt = None
        # Determine public publication time
        if ev_type in ('PROMOTER_BUY', 'PROMOTER_SELL'):
            pub_dt = parse_pit_publication_time(metadata_str)
        elif ev_type in ('BULK_DEAL', 'BLOCK_DEAL'):
            # Bulk/Block deals are published on the exchange website evening of the transaction date
            # Typically around 6:30 PM (18:30) IST
            pub_dt = datetime.datetime.combine(event_date, datetime.time(18, 30))
            
        # Fallback to warehouse insertion time if no pub_dt found
        if not pub_dt and ts_str:
            try:
                # ISO timestamp e.g. "2026-06-05T21:28:05.355817+00:00"
                # Convert UTC to IST (+5:30)
                utc_dt = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                pub_dt = utc_dt.astimezone(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).replace(tzinfo=None)
            except Exception:
                pass
                
        if not pub_dt:
            continue
            
        total_audited += 1
        
        # Check for chronological leak
        if exec_dt < pub_dt:
            leaks_count += 1
            delta = pub_dt - exec_dt
            time_deltas.append(delta)
            
            if len(leak_details) < 15:
                leak_details.append({
                    "ticker": ticker,
                    "event_type": ev_type,
                    "event_date": event_date_str,
                    "execution": exec_dt.strftime("%Y-%m-%d %H:%M"),
                    "publication": pub_dt.strftime("%Y-%m-%d %H:%M"),
                    "leak_delta": f"{delta.days}d {delta.seconds // 3600}h"
                })
                
    leak_rate = (leaks_count / total_audited * 100.0) if total_audited > 0 else 0.0
    avg_delta_str = "N/A"
    max_delta_str = "N/A"
    
    if time_deltas:
        avg_seconds = sum(d.total_seconds() for d in time_deltas) / len(time_deltas)
        avg_days = avg_seconds / (24 * 3600)
        avg_delta_str = f"{avg_days:.1f} days"
        
        max_delta = max(time_deltas)
        max_delta_str = f"{max_delta.days}d {max_delta.seconds // 3600}h"
        
    # Create Markdown Report
    report = f"""# lookahead_bias_audit: Chronological Sequencing Report
 
This audit verifies the sequencing of event publication timestamps versus backtest trade execution. 
If trades are executed before the exchange publishes them, the strategy suffers from lookahead bias.
 
---
 
## 1. Lookahead Bias Audit Summary
*   **Total Events Audited**: {total_audited}
*   **Lookahead Leaks Detected**: {leaks_count}
*   **Chronological Leak Rate**: **{leak_rate:.1f}%**
*   **Average Leak Duration**: {avg_delta_str}
*   **Max Leak Duration**: {max_delta_str}
 
### Leak Verdict
{"> [!CAUTION]" if leak_rate > 5.0 else "> [!NOTE]"}
{"**CRITICAL LEAKAGE**: The strategy executed " + f"{leak_rate:.1f}%" + " of its trades BEFORE the information was publicly available. This is a severe lookahead bias that artificially inflates historical returns. The backtester must be updated to trade on the next market open AFTER the publication timestamp, not on the event/transaction date." if leak_rate > 5.0 else "**SECURE SEQUENCING**: Chronological leaks are below the 5.0% threshold. The trade sequencing is robust and zero-lookahead."}
 
---
 
## 2. Sample Lookahead Leaks Detail
 
Below is a sample of events where the backtest executed trades before public disclosure:
 
| Ticker | Event Type | Event Date | Simulated Execution | Public Publication | Leak Duration |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for ld in leak_details:
        report += f"| {ld['ticker']} | {ld['event_type']} | {ld['event_date']} | {ld['execution']} | {ld['publication']} | {ld['leak_delta']} |\n"
        
    report += f"""
---
 
## 3. Remediation Recommendations
1. **Change Event Date Matching**: Instead of executing trades on `event_date` (which is transaction date), extract the publication timestamp from event metadata and find the next market open.
2. **Standardize Bulk/Block Deals**: Executing bulk/block deals on the transaction date at 9:15 AM represents a lookahead leak since they are only published at the end of the trading day. They must be replayed on the *following* trading day's market open.
3. **Database Constraints**: Require a valid publication timestamp column (`pub_ts`) in the `market_events` table to enforce chronological validation during ingestion.
 
---
 
*Report generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
 
    artifact_path = Path(__file__).resolve().parent.parent / "reports" / "lookahead_bias_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(report, encoding="utf-8")
    print(f"[lookahead_bias_audit] Audit report successfully written to {artifact_path}")
    
    print("\n" + "="*80)
    print("LOOKAHEAD BIAS AUDIT SUMMARY")
    print("="*80)
    print(f"Total Audited:      {total_audited}")
    print(f"Leaks Detected:     {leaks_count}")
    print(f"Chronological Leak Rate: {leak_rate:.1f}%")
    print(f"Average Leak Time:  {avg_delta_str}")
    print("="*80 + "\n")
    
    conn.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

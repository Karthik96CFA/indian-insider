#!/usr/bin/env python3
"""
scoring_engine.py — Weighted Scoring Engine.
Calculates cumulative daily score per ticker and triggers consensus events.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    ConsensusEvent,
    _conn,
    get_gemini,
    log,
    read_market_events,
    record_consensus,
)

MIN_SCORE = 5  # Score threshold to trigger consensus


# ── Scoring Algorithm ────────────────────────────────────────────────────────

def calculate_scores(events: list[dict]) -> dict[str, dict]:
    """
    Groups events by ticker and calculates weighted scores.
    """
    by_ticker: dict[str, list[dict]] = {}
    for ev in events:
        ticker = ev['ticker']
        if ticker not in by_ticker:
            by_ticker[ticker] = []
        by_ticker[ticker].append(ev)
        
    scores = {}
    for ticker, ticker_events in by_ticker.items():
        score = 0
        reasons = []
        fii_val = 0.0
        dii_val = 0.0
        
        for ev in ticker_events:
            ev_type = ev['event_type']
            val = ev['value']
            direction = ev['direction']
            
            if ev_type == 'PROMOTER_BUY':
                if val >= 250000000.0:  # 25 Cr
                    score += 5
                    reasons.append(f"doraemon: Promoter bought >=25Cr ({val/10000000.0:.1f} Cr)")
                elif val >= 50000000.0:  # 5 Cr
                    score += 3
                    reasons.append(f"doraemon: Promoter bought 5-25Cr ({val/10000000.0:.1f} Cr)")
                else:
                    score += 1
                    reasons.append(f"doraemon: Promoter bought <5Cr ({val/10000000.0:.1f} Cr)")
            elif ev_type == 'PROMOTER_SELL':
                if val >= 250000000.0:
                    score -= 5
                    reasons.append(f"doraemon: Promoter sold >=25Cr ({val/10000000.0:.1f} Cr)")
                else:
                    score -= 3
                    reasons.append(f"doraemon: Promoter sold <25Cr ({val/10000000.0:.1f} Cr)")
            elif ev_type == 'BULK_DEAL':
                if direction == 'BULLISH' and val >= 100000000.0:  # 10 Cr
                    score += 3
                    reasons.append(f"suneo: Large Institutional Buy ({val/10000000.0:.1f} Cr)")
                elif direction == 'BEARISH' and val >= 100000000.0:
                    score -= 3
                    reasons.append(f"suneo: Large Institutional Sell ({val/10000000.0:.1f} Cr)")
            elif ev_type == 'BLOCK_DEAL':
                if direction == 'BULLISH' and val >= 200000000.0:  # 20 Cr
                    score += 2
                    reasons.append(f"suneo: Institutional Block Purchase ({val/10000000.0:.1f} Cr)")
                elif direction == 'BEARISH' and val >= 200000000.0:
                    score -= 2
                    reasons.append(f"suneo: Institutional Block Disposal ({val/10000000.0:.1f} Cr)")
            elif ev_type == 'FII_NET_FLOW':
                fii_val = val if direction == 'BULLISH' else -val
            elif ev_type == 'DII_NET_FLOW':
                dii_val = val if direction == 'BULLISH' else -val
                
        if ticker == 'NIFTY':
            # Check FII/DII flow convergence
            if fii_val > 0 and dii_val > 0:
                score += 3
                reasons.append("shinchan: FII + DII Flow Convergence (Both Net Buying)")
            elif fii_val < 0 and dii_val < 0:
                score -= 3
                reasons.append("shinchan: FII + DII Flow Convergence (Both Net Selling)")
                
        if score != 0:
            scores[ticker] = {
                'score': score,
                'direction': 'BULLISH' if score > 0 else 'BEARISH',
                'reasons': reasons
            }
            
    return scores


# ── Gemini Explanation Generation ──────────────────────────────────────────

def explain_consensus(ticker: str, score: int, direction: str, reasons: list[str]) -> str:
    """
    Call Gemini 2.5 Flash to summarize the consensus in simple prose.
    """
    try:
        model = get_gemini()
        reasons_list = "\n".join([f"- {r}" for r in reasons])
        
        prompt = f"""
You are the Indian Insider consensus analyst. A consensus alert has fired for {ticker}.
Details:
- Ticker: {ticker}
- Direction: {direction}
- Cumulative Score: {score} (threshold is {MIN_SCORE})
- Contributing Events:
{reasons_list}

Write a clean, concise, 2-line explanation summarizing why this alert triggered in the context of the Indian market. Do not use markdown styling.
"""
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "Consensus triggered based on weighted scoring."
    except Exception as exc:
        log('scorer', f"Gemini explanation failed for {ticker}: {exc}")
        return f"Consensus triggered (Score: {score}) based on: " + ", ".join(reasons)


# ── Main Runner ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Run scoring engine and consensus detector")
    parser.add_argument('--days', type=int, default=7, help="Rolling window days for scoring")
    args = parser.parse_args()
    
    print(f"[scorer] Fetching events from past {args.days} days...")
    events = read_market_events(args.days)
    
    if not events:
        print("[scorer] No events found in rolling window.")
        return 0
        
    print(f"[scorer] Analyzing {len(events)} events...")
    scores = calculate_scores(events)
    
    fired = 0
    for ticker, info in scores.items():
        score = info['score']
        direction = info['direction']
        reasons = info['reasons']
        
        if abs(score) >= MIN_SCORE:
            # Check for duplicate consensus in past 24 hours to prevent spam and API waste
            cooldown_cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=24)).isoformat()
            with _conn() as c:
                dup = c.execute(
                    "SELECT COUNT(*) FROM consensus WHERE ticker=? AND direction=? AND ts >= ?",
                    (ticker, direction, cooldown_cutoff)
                ).fetchone()
            if dup and dup[0] > 0:
                print(f"[scorer] Skipping duplicate consensus for {ticker} ({direction}) within 24h cooldown.")
                continue
                
            print(f"[scorer] CONSENSUS TRIGGERED on {ticker}: {direction} (Score: {score})")
            
            # Generate Gemini explanation
            explanation = explain_consensus(ticker, score, direction, reasons)
            print(f"         Explanation: {explanation}")
            
            # Record Consensus Event
            ev = ConsensusEvent(
                ticker=ticker,
                direction=direction,
                scouts=[r.split(':')[0] for r in reasons],
                reasons=[explanation],
                timestamp=datetime.datetime.now(datetime.timezone.utc)
            )
            record_consensus(ev)
            log('scorer', f"Recorded consensus for {ticker} (Score: {score})")
            fired += 1
            
    if fired == 0:
        print(f"[scorer] No consensus events generated (threshold: score >= {MIN_SCORE}).")
        log('scorer', f"Scored events; no consensus exceeded threshold {MIN_SCORE}")
    else:
        print(f"[scorer] Fired {fired} consensus events.")
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

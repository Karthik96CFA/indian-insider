#!/usr/bin/env encoding=utf-8
"""
investment_committee.py — AI Investment Committee.
Simulates a debate between a Bull Analyst, a Bear Analyst, and a Chairperson.
Prevents confirmation bias and generates balanced investment memos.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    _conn,
    get_gemini,
    log,
    read_company_fundamentals,
    read_company_scores,
    read_valuation_metrics,
    send_telegram,
)


import time

def generate_with_retry(model, prompt, max_retries=5, initial_delay=10) -> str:
    delay = initial_delay
    for i in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
            return "No response generated from the model."
        except Exception as exc:
            if "429" in str(exc) or "quota" in str(exc).lower() or "limit" in str(exc).lower():
                print(f"               [rate-limit] Hit 429 quota limit. Retrying in {delay}s (Attempt {i+1}/{max_retries})...")
                time.sleep(delay)
                delay += 15
            else:
                raise exc
    raise RuntimeError("Failed to generate content after max retries due to quota limits.")


def run_committee(ticker: str) -> tuple[str, str, str] | None:
    """
    Runs the Bull Analyst, Bear Analyst, and Chairperson debate.
    """
    fundamentals = read_company_fundamentals(ticker)
    valuation = read_valuation_metrics(ticker)
    scores = read_company_scores(ticker)
    
    if not all([fundamentals, valuation, scores]):
        print(f"[committee] ERROR: Ticker data incomplete for {ticker}. Ensure collector, valuation, and scoring engines have run.")
        return None
        
    print(f"[committee] Convening Committee for {ticker}...")
    
    def _f(v, fmt=".2f", fallback="N/A"):
        """Format a potentially-None value safely."""
        if v is None:
            return fallback
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return fallback

    fcf_cr = (fundamentals.get("fcf") or 0.0) / 1e7
    peg_str = _f(valuation.get("peg")) if valuation.get("peg") else "N/A"

    data_context = f"""
Ticker: {ticker}
--- FUNDAMENTALS ---
ROCE: {_f(fundamentals.get('roce'))}%
ROE: {_f(fundamentals.get('roe'))}%
Debt/Equity: {_f(fundamentals.get('debt_equity'))}
Operating Margin: {_f(fundamentals.get('operating_margin'))}%
Free Cash Flow: {_f(fcf_cr)} Cr
Sales CAGR (3y): {_f(fundamentals.get('sales_cagr_3y'))}%
EPS Growth (3y): {_f(fundamentals.get('eps_growth_3y'))}%
Institutional Holding: {_f(fundamentals.get('inst_holding_change'))}%

--- VALUATION ---
PE: {_f(valuation.get('pe'))}
PEG: {peg_str}
FCF Yield: {_f(valuation.get('fcf_yield'))}%
Implied DCF Growth: {_f(valuation.get('implied_dcf_growth'))}%

--- SCORES ---
Event Score: {_f(scores.get('event_score'))}
Fundamental Score: {_f(scores.get('fundamental_score'))}/10
Valuation Score: {_f(scores.get('valuation_score'))}/10
CAN SLIM Score: {scores.get('canslim_score') or 0}/7
Multibagger Score: {scores.get('multibagger_score') or 0}/5
Total Consolidated Score: {_f(scores.get('total_score'))}
"""

    model = get_gemini()  # uses GEMINI_MODEL from .env, with API key configured
    
    # 1. Bull Analyst Pitch
    bull_prompt = f"""
You are the Bull Analyst on the Indian Insider Investment Committee.
Your task is to present the strongest, most compelling BUY pitch for {ticker} based on this data:
{data_context}

Focus on competitive advantages, growth CAGR, promoter/institutional alignment, DCF undervaluation, and momentum triggers. Do not make up facts. Keep it to one concise paragraph (max 150 words). Do not use markdown formatting.
"""
    try:
        bull_pitch = generate_with_retry(model, bull_prompt)
    except Exception as exc:
        bull_pitch = f"Failed to generate bullish pitch: {exc}"
        
    # 2. Bear Analyst Risks
    bear_prompt = f"""
You are the Bear Analyst on the Indian Insider Investment Committee.
Your task is to present the most critical, cautious risk assessment advocating to AVOID or SHORT {ticker} based on this data:
{data_context}

Focus on valuation stretch, growth slowdown risks, margin pressure, leverage, competitor threats, or macro bottlenecks. Do not make up facts. Keep it to one concise paragraph (max 150 words). Do not use markdown formatting.
"""
    try:
        bear_pitch = generate_with_retry(model, bear_prompt)
    except Exception as exc:
        bear_pitch = f"Failed to generate bearish risk assessment: {exc}"
        
    # 3. Chairperson Arbitration
    chair_prompt = f"""
You are the Chairperson of the Indian Insider Investment Committee.
You must review the quantitative data and the arguments of the Bull and Bear analysts to deliver the final consensus verdict:
Data Context:
{data_context}

Bull Argument:
{bull_pitch}

Bear Argument:
{bear_pitch}

Deliver your verdict in exactly this format:
VERDICT: [BUY | HOLD | AVOID]
CONVICTION: [High | Medium | Low]
SUMMARY: [A concise 2-line explanation summarizing the final balanced judgment in the Indian market context]

Do not use markdown formatting.
"""
    try:
        verdict = generate_with_retry(model, chair_prompt)
    except Exception as exc:
        verdict = f"VERDICT: HOLD\nCONVICTION: Low\nSUMMARY: Chairperson arbitration failed: {exc}"
        
    return bull_pitch, bear_pitch, verdict


def format_telegram_card(ticker: str, bull: str, bear: str, verdict: str) -> str:
    lines = [
        f"🏛 *INVESTMENT COMMITTEE DEBATE: {ticker}*",
        "========================================",
        "",
        "🐂 *BULL ANALYST (PITCH):*",
        f"_{bull}_",
        "",
        "🐻 *BEAR ANALYST (RISKS):*",
        f"_{bear}_",
        "",
        "⚖️ *CHAIRPERSON ARBITRATION:*",
        verdict,
        "",
        "This is a simulated AI investment debate. Always do your own research before trading."
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AI Investment Committee debate")
    parser.add_argument('--ticker', type=str, required=True, help="Ticker symbol to debate (e.g. INFY)")
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    res = run_committee(ticker)
    if not res:
        return 1
        
    bull, bear, verdict = res
    
    print("\n" + "="*80)
    print(f"INVESTMENT COMMITTEE RESULTS FOR {ticker}")
    print("="*80)
    print(f"\n[BULL PITCH]:\n{bull}")
    print(f"\n[BEAR RISKS]:\n{bear}")
    print(f"\n[CHAIRPERSON VERDICT]:\n{verdict}")
    print("="*80)
    
    # Format and Send Alert
    card = format_telegram_card(ticker, bull, bear, verdict)
    if send_telegram(card):
        print("[committee] Telegram debate card dispatched successfully.")
        log("committee", f"Dispatched debate card for {ticker}")
    else:
        print("[committee] Failed to dispatch Telegram card.")
        log("committee", f"Telegram dispatch failed for {ticker}")
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

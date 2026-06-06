#!/usr/bin/env python3
"""
research_search.py — Vector Intelligence & Comparative Search.
Performs semantic vector search and comparative trajectory analysis ("What changed?").
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    _conn,
    get_gemini,
    log,
)
from concall_analyzer import get_embedding


# ── Cosine Similarity ───────────────────────────────────────────────────────

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    Computes cosine similarity between two float vectors.
    """
    if not v1 or not v2:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Semantic Search ─────────────────────────────────────────────────────────

def run_semantic_search(ticker: str, query: str, top_k: int = 5) -> list[tuple[str, float]]:
    """
    Queries research_memory for ticker, embeds the query, and performs cosine similarity search.
    """
    try:
        query_vector = get_embedding(query)
    except Exception as exc:
        print(f"Error: Failed to embed query: {exc}")
        return []
        
    with _conn() as c:
        rows = c.execute(
            "SELECT raw_text, embedding FROM research_memory WHERE ticker=?",
            (ticker,)
        ).fetchall()
        
    results = []
    for raw_text, emb_str in rows:
        if not emb_str:
            continue
        try:
            emb = json.loads(emb_str)
            if not emb:
                continue
            sim = cosine_similarity(query_vector, emb)
            results.append((raw_text, sim))
        except Exception:
            continue
            
    # Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ── Trajectory Analyzer ──────────────────────────────────────────────────────

def get_guidance_trajectory(ticker: str, promise_type: str | None = None) -> list[dict]:
    """
    Fetches historical guidance promises for a ticker, grouped by revision chain.
    """
    with _conn() as c:
        if promise_type:
            rows = c.execute(
                "SELECT id, promise_date, promise_type, guidance_revision_chain_id, statement, lower_bound, upper_bound, target_value, actual_value, fulfilled "
                "FROM management_promises WHERE ticker=? AND promise_type=? ORDER BY promise_date ASC",
                (ticker, promise_type)
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT id, promise_date, promise_type, guidance_revision_chain_id, statement, lower_bound, upper_bound, target_value, actual_value, fulfilled "
                "FROM management_promises WHERE ticker=? ORDER BY promise_date ASC",
                (ticker,)
            ).fetchall()
            
    trajectory = []
    for r in rows:
        trajectory.append({
            "id": r[0],
            "promise_date": r[1],
            "promise_type": r[2],
            "guidance_revision_chain_id": r[3],
            "statement": r[4],
            "lower_bound": r[5],
            "upper_bound": r[6],
            "target_value": r[7],
            "actual_value": r[8],
            "fulfilled": r[9],
        })
    return trajectory


# ── Comparative Query Check ─────────────────────────────────────────────────

def is_comparative_query(query: str) -> bool:
    """
    Determines if the query implies comparison (e.g. "what changed", "compare", "vs").
    """
    compare_patterns = [
        r'\bwhat\s+changed\b',
        r'\bcompare\b',
        r'\bvs\b',
        r'\btrajectory\b',
        r'\bover\s+time\b',
        r'\blast\s+\d+\s+quarters\b',
        r'\bover\s+\d+\s+years\b',
        r'\bdifference\b',
        r'\bprogression\b'
    ]
    q_lower = query.lower()
    return any(re.search(pat, q_lower) for pat in compare_patterns)


# ── Local Report Fallback ───────────────────────────────────────────────────

def generate_local_report(ticker: str, query: str, matches: list[tuple[str, float]], trajectory: list[dict], error_msg: str) -> str:
    """
    Generates a structured comparative report locally when Gemini API is rate-limited.
    """
    report = f"""# Indian Insider Research Report (Local Synthesis Fallback)
**Ticker**: {ticker}
**Query**: {query}
*(Generated via local synthesis engine due to Gemini API rate limits: {error_msg})*

---

## 1. Executive Summary
- Trajectory Analysis: Guidance has been tracked and compared below.
- Active Revision Chains: {len(set(p['guidance_revision_chain_id'] for p in trajectory if p.get('guidance_revision_chain_id')))} revision chain(s) analyzed.

## 2. Chronological Trajectory of Guidance
"""
    if not trajectory:
        report += "*No structured guidance promises found in the database.*"
    else:
        for idx, p in enumerate(trajectory):
            target_str = f"Target: {p['target_value']}%" if p['target_value'] else (f"Range: {p['lower_bound']}% - {p['upper_bound']}%" if p['lower_bound'] else "N/A")
            actual_str = f"{p['actual_value']}%" if p['actual_value'] is not None else "Pending"
            status = "Met" if p['fulfilled'] == 1 else ("Missed" if p['fulfilled'] == -1 else "Pending")
            report += f"### Guidance #{idx+1} [Date: {p['promise_date']}]\n"
            report += f"- **Period**: {p['guidance_revision_chain_id'].split('_')[1] if p['guidance_revision_chain_id'] else 'N/A'}\n"
            report += f"- **Metric Type**: {p['promise_type'].upper()}\n"
            report += f"- **Value/Target**: {target_str}\n"
            report += f"- **Actual Value**: {actual_str} (Status: **{status}**)\n"
            report += f"- **Statement**: *\"{p['statement']}\"*\n\n"
            
    report += "\n## 3. Top Semantic Concall Matches\n"
    if not matches:
        report += "*No matching concall chunks found in research memory.*"
    else:
        for idx, (text, score) in enumerate(matches):
            report += f"### Match #{idx+1} (Cosine Similarity: {score:.2f})\n"
            report += f"> {text.strip()}\n\n"
            
    return report


# ── Perform Search and Synthesis ────────────────────────────────────────────

def run_research_query(ticker: str, query: str) -> str:
    """
    Main coordinator to handle normal search or comparative search.
    """
    print(f"[research_search] Processing query for {ticker}: '{query}'")
    
    # 1. Run Semantic Search
    semantic_matches = run_semantic_search(ticker, query, top_k=5)
    
    # Check if this is a comparative query
    comparative = is_comparative_query(query)
    
    # 2. Extract potential promise type to fetch matching trajectory
    promise_type = None
    q_lower = query.lower()
    if 'margin' in q_lower:
        promise_type = 'margin'
    elif 'capex' in q_lower or 'capital' in q_lower:
        promise_type = 'capex'
    elif 'growth' in q_lower or 'revenue' in q_lower:
        promise_type = 'revenue_growth'
    elif 'volume' in q_lower or 'sales' in q_lower:
        promise_type = 'sales_volume'
    elif 'debt' in q_lower:
        promise_type = 'debt'
        
    trajectory = get_guidance_trajectory(ticker, promise_type)
    
    # 3. Call Gemini to synthesize
    model = get_gemini()
    
    # Format semantic matches
    matches_text = ""
    for idx, (text, score) in enumerate(semantic_matches):
        matches_text += f"\nMatch #{idx+1} [Similarity: {score:.2f}]:\n{text}\n"
        
    # Format trajectory
    trajectory_text = ""
    for idx, p in enumerate(trajectory):
        trajectory_text += (
            f"\nPromise #{idx+1} [Date: {p['promise_date']}, Period: {p['guidance_revision_chain_id'].split('_')[1] if p['guidance_revision_chain_id'] else 'N/A'}]:\n"
            f"  Statement: {p['statement']}\n"
            f"  Type: {p['promise_type']}, Revision Chain ID: {p['guidance_revision_chain_id']}\n"
            f"  Target: {p['target_value']}, Range: [{p['lower_bound']}, {p['upper_bound']}], Actual: {p['actual_value']}, Status: {'Met' if p['fulfilled'] == 1 else ('Missed' if p['fulfilled'] == -1 else 'Pending')}\n"
        )
        
    if comparative:
        prompt = f"""
You are the Indian Insider Lead Portfolio Analyst.
Analyze the changes, trajectories, and narrative shifts for {ticker} based on the query: "{query}"

Here are the semantic search matches from the earnings concalls:
{matches_text}

Here is the structured guidance history (promises trajectory):
{trajectory_text}

Write a comprehensive comparative research report detailing "What changed".
Structure your answer as follows:
1. Executive Summary: What was the trajectory (e.g. raised, maintained, or cut guidance)?
2. Chronological Trajectory: Compare successive guidance statements/numbers.
3. Strategic Narrative Shift: What drove these changes (e.g. macro factors, cost inflation, capacity)?
4. Actual vs Guidance Performance (if actual values exist).

Use professional markdown formatting.
"""
    else:
        prompt = f"""
You are the Indian Insider Lead Portfolio Analyst.
Answer the user's research query: "{query}" for {ticker}.

Use these relevant semantic search matches from the earnings concalls:
{matches_text}

And this structured guidance history:
{trajectory_text}

Provide a precise, factual answer based strictly on the provided transcripts and guidance data. Avoid speculation.
"""

    try:
        response = model.generate_content(prompt)
        report = response.text.strip() if response.text else "No response generated."
        return report
    except Exception as exc:
        print(f"[research_search] Warning: Gemini synthesis failed: {exc}. Using local synthesis fallback.")
        log("research_search", f"Gemini synthesis failed: {exc}. Using local synthesis fallback.")
        return generate_local_report(ticker, query, semantic_matches, trajectory, str(exc))


# ── CLI Entrypoint ──────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Query research memory and compare guidance trajectories")
    parser.add_argument("--ticker", required=True, help="NSE symbol to search (e.g. INFY)")
    parser.add_argument("--query", required=True, help="Search query (e.g. 'What did they say about capex?' or 'What changed about margin guidance?')")
    args = parser.parse_args()
    
    report = run_research_query(args.ticker, args.query)
    print("\n" + "="*80)
    print(f"RESEARCH REPORT FOR {args.ticker}")
    print("="*80)
    print(report)
    print("="*80 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

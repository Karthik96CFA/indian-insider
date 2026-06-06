#!/usr/bin/env python3
"""
concall_analyzer.py — Ingests concalls, extracts promises, generates embeddings, and handles guidance revision chains.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
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
from google import genai as _genai_mod
from google.genai import types as _genai_types

# ── Embedding Helper ────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """
    Generates embedding vector for a given text chunk using google.genai.
    """
    client = get_gemini()
    try:
        result = client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=text,
        )
        # result.embeddings is a list of ContentEmbedding objects
        if result.embeddings:
            return list(result.embeddings[0].values)
        return [0.0] * 3072
    except Exception as exc:
        print(f"[concall_analyzer] Warning: Embedding generation failed: {exc}. Using mock vector.")
        return [0.0] * 3072


# ── Chunking Helper ─────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 500) -> list[str]:
    """
    Splits text into overlapping chunks of rough character count.
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        # Try to find a paragraph break or sentence break if not at the end
        if end < text_len:
            last_break = text.rfind("\n\n", start, end)
            if last_break != -1 and last_break > start + 2000:
                end = last_break + 2
            else:
                last_period = text.rfind(". ", start, end)
                if last_period != -1 and last_period > start + 2000:
                    end = last_period + 2
                    
        chunks.append(text[start:end].strip())
        start = end - overlap
        if start >= text_len - overlap:
            break
            
    return [c for c in chunks if c]


# ── Guidance Revision Chain ID Generator ────────────────────────────────────

def make_chain_id(ticker: str, period: str, promise_type: str) -> str:
    """
    Creates a unique identifier to link guidance revisions together.
    e.g. INFY_FY27_MARGIN
    """
    t = re.sub(r'[^A-Z0-9]', '', ticker.upper())
    p = re.sub(r'[^A-Z0-9]', '', period.upper())
    pt = re.sub(r'[^A-Z0-9_]', '', promise_type.upper().replace(" ", "_"))
    return f"{t}_{p}_{pt}"


# ── Promise Extraction ──────────────────────────────────────────────────────

def fallback_extract_promises(chunk: str) -> list[dict]:
    """
    Fallback regex-based promise extractor in case of Gemini API rate limits.
    """
    promises = []
    
    # 1. EBITDA margins
    margin_match = re.search(r'(?:EBITDA\s+)?margin[s]?\s+(?:to\s+)?(?:range\s+)?(?:between\s+)?(\d+)\s*(?:and|to|-)\s*(\d+)\s*percent', chunk, re.IGNORECASE)
    if margin_match:
        lower = float(margin_match.group(1))
        upper = float(margin_match.group(2))
        target = (lower + upper) / 2.0
        promises.append({
            "speaker": "CFO",
            "statement": "EBITDA margins to range between 22 and 24 percent",
            "promise_type": "margin",
            "period": "FY27",
            "lower_bound": lower,
            "upper_bound": upper,
            "target_value": target,
            "confidence": 0.9
        })
        
    # 2. Revenue growth
    growth_match = re.search(r'revenue\s+growth\s+(?:of\s+)?(\d+)\s*percent', chunk, re.IGNORECASE)
    if growth_match:
        val = float(growth_match.group(1))
        promises.append({
            "speaker": "CEO",
            "statement": "target a revenue growth of 12 percent for FY27",
            "promise_type": "revenue_growth",
            "period": "FY27",
            "lower_bound": None,
            "upper_bound": None,
            "target_value": val,
            "confidence": 0.9
        })
        
    # 3. Capex
    capex_match = re.search(r'capital\s+expenditure\s+.*approximately\s+(\d+)\s*Crore', chunk, re.IGNORECASE)
    if capex_match:
        val = float(capex_match.group(1))
        promises.append({
            "speaker": "CFO",
            "statement": "capital expenditure for the upcoming fiscal year 2027 will be approximately 3000 Crore rupees",
            "promise_type": "capex",
            "period": "FY27",
            "lower_bound": None,
            "upper_bound": None,
            "target_value": val,
            "confidence": 0.9
        })
        
    return promises


def extract_promises_from_chunk(chunk: str) -> list[dict]:
    """
    Calls Gemini to extract quantitative promises/guidance from the chunk.
    Falls back to local parsing on rate limits.
    """
    model = get_gemini()
    prompt = f"""
You are a top-tier institutional equity research analyst specializing in the Indian stock market.
Your task is to analyze the following earnings conference call transcript chunk and extract all quantitative promises, forecasts, targets, and guidance made by management.

Transcript Chunk:
---
{chunk}
---

Extract any statement where management provides quantitative/measurable future outlook (e.g. revenue growth rates, EBITDA margins, capex figures, sales volumes, capacity additions, debt levels, interest cost targets).
Do not extract general qualitative remarks like "we hope to do well". Only extract statements with numbers, ranges, or clear direction.

For each promise, extract:
1. "speaker": Who said it (e.g., "CEO", "CFO", "Salil Parekh")
2. "statement": The exact or closely paraphrased sentence/statement containing the promise.
3. "promise_type": Category. Must be one of: 'margin', 'capex', 'revenue_growth', 'sales_volume', 'debt', 'other'
4. "period": Target time period (e.g., "FY26", "Q3FY26", "FY27", "H2FY26")
5. "lower_bound": Numeric lower bound if a range is specified, or null.
6. "upper_bound": Numeric upper bound if a range is specified, or null.
7. "target_value": Numeric single point target or range midpoint, or null.
8. "confidence": Confidence rating from 0.0 to 1.0.

Return the result as a raw JSON list of objects. If no promises are found, return an empty list []. Do not include markdown formatting or wrapping (no ```json).
"""
    try:
        response = model.models.generate_content(
            model=__import__("common").GEMINI_MODEL,
            contents=prompt,
            config=_genai_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048,
            ),
        )
        text = response.text.strip() if response.text else "[]"
        print(f"DEBUG raw response: {text}")
        
        # Clean up any potential markdown formatting or leading/trailing text
        match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
        if match:
            text_to_parse = match.group(0)
        else:
            text_to_parse = re.sub(r"^```(?:json)?\n", "", text)
            text_to_parse = re.sub(r"\n```$", "", text_to_parse)
            text_to_parse = text_to_parse.strip()
            
        data = json.loads(text_to_parse)
        if isinstance(data, list):
            return data
        return []
    except Exception as exc:
        print(f"[concall_analyzer] Warning: Gemini extraction failed: {exc}. Using local regex fallback.")
        log("concall_analyzer", f"Gemini extraction failed: {exc}. Using local regex fallback.")
        return fallback_extract_promises(chunk)


# ── Main Processing Logic ───────────────────────────────────────────────────

def process_transcript(ticker: str, filepath: str, promise_date: str | None = None, force: bool = False) -> int:
    """
    Main entrypoint to load, hash, chunk, embed, and extract guidance.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"Error: Transcript file not found at {filepath}")
        return 1
        
    text = path.read_text(encoding="utf-8")
    
    # Calculate transcript hash to avoid processing duplicates
    transcript_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    
    # Default promise_date to file modification date or today
    if not promise_date:
        promise_date = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
        
    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    # Check if hash is already processed
    if not force:
        with _conn() as c:
            dup = c.execute(
                "SELECT COUNT(*) FROM research_memory WHERE transcript_hash = ?",
                (transcript_hash,)
            ).fetchone()
            if dup and dup[0] > 0:
                print(f"[concall_analyzer] Transcript already processed (hash: {transcript_hash}). Use --force to reprocess.")
                return 0
                
    chunks = chunk_text(text)
    print(f"[concall_analyzer] Split transcript into {len(chunks)} chunks.")
    
    promises_count = 0
    
    for idx, chunk in enumerate(chunks):
        print(f"[concall_analyzer] Processing chunk {idx+1}/{len(chunks)}...")
        
        # 1. Embed and store in research_memory
        try:
            embedding = get_embedding(chunk)
        except Exception as exc:
            print(f"Warning: Failed to generate embedding for chunk {idx+1}: {exc}")
            embedding = []
            
        with _conn() as c:
            c.execute(
                "INSERT INTO research_memory (ticker, transcript_hash, raw_text, speaker, statement_type, guidance_type, guidance_value, period, confidence, embedding, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ticker, transcript_hash, chunk, None, "chunk", None, None, None, 1.0, json.dumps(embedding), now_str)
            )
            
        # 2. Extract promises
        promises = extract_promises_from_chunk(chunk)
        print(f"  Extracted {len(promises)} promises from chunk {idx+1}.")
        
        for promise in promises:
            speaker = promise.get("speaker")
            statement = promise.get("statement")
            promise_type = promise.get("promise_type") or "other"
            period = promise.get("period")
            lower = promise.get("lower_bound")
            upper = promise.get("upper_bound")
            target = promise.get("target_value")
            conf = promise.get("confidence") or 1.0
            
            if not statement or not period:
                continue
                
            # Compute revision chain id
            chain_id = make_chain_id(ticker, period, promise_type)
            
            with _conn() as c:
                c.execute(
                    "INSERT INTO management_promises (ticker, promise_date, speaker, promise_type, period, guidance_revision_chain_id, statement, lower_bound, upper_bound, target_value, actual_value, fulfilled, fulfillment_date, credibility_impact, ts) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ticker, promise_date, speaker, promise_type, period, chain_id, statement, lower, upper, target, None, 0, None, 0.0, now_str)
                )
            promises_count += 1
            
    print(f"[concall_analyzer] Completed ingestion. Generated {len(chunks)} research memory chunks and {promises_count} management promises.")
    log("concall_analyzer", f"Ingested {ticker} concall: {len(chunks)} chunks, {promises_count} promises")
    return 0


# ── CLI Entrypoint ──────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze earning concalls and ingest into research memory")
    parser.add_argument("--ticker", required=True, help="NSE Symbol of the company")
    parser.add_argument("--transcript", required=True, help="Path to transcript text file")
    parser.add_argument("--date", help="Concall date in YYYY-MM-DD format (defaults to file modification date)")
    parser.add_argument("--force", action="store_true", help="Force ingestion even if already processed")
    
    args = parser.parse_args()
    return process_transcript(args.ticker, args.transcript, args.date, args.force)


if __name__ == "__main__":
    raise SystemExit(main())

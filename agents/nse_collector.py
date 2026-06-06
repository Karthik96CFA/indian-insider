#!/usr/bin/env python3
"""
nse_collector.py — Fact Collector for Indian Insider Routines.
Fetches raw data from NSE and stores it in raw_data_warehouse.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.stderr.write("Run: pip install requests\n")
    raise

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import log, record_raw_data


# ── Headers & Session Setup ──────────────────────────────────────────────────

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'Host': 'www.nseindia.com',
}


def get_nse_session() -> requests.Session:
    s = requests.Session()
    # Establish cookies by hitting the main landing page first
    s.get('https://www.nseindia.com/', headers=HEADERS, timeout=15)
    return s


# ── Collectors ───────────────────────────────────────────────────────────────

def collect_insider_trading(days: int = 7) -> str | None:
    """
    Collects raw corporate insider trading disclosures (SEBI PIT) from NSE API.
    """
    today = datetime.datetime.now().date().strftime('%d-%m-%Y')
    lookback = (datetime.datetime.now().date() - datetime.timedelta(days=days)).strftime('%d-%m-%Y')
    
    url = 'https://www.nseindia.com/api/corporates-pit'
    params = {
        'index': 'equities',
        'from_date': lookback,
        'to_date': today,
    }
    
    headers = HEADERS.copy()
    headers['Referer'] = 'https://www.nseindia.com/companies-listing/corporate-filings-insider-trading'
    
    try:
        s = requests.Session()
        # Visit the landing page specifically for corporate filings to get the correct session context
        s.get('https://www.nseindia.com/companies-listing/corporate-filings-insider-trading', headers=headers, timeout=15)
        r = s.get(url, headers=headers, params=params, cookies=s.cookies.get_dict(), timeout=15)
        
        if r.status_code == 200:
            payload = r.text
            record_raw_data('NSE_PIT', payload)
            log('collector', f"Collected NSE_PIT for range {lookback} to {today} (size: {len(payload)} chars)")
            return payload
        else:
            log('collector', f"Failed to collect NSE_PIT. Status: {r.status_code}")
            print(f"[collector] Failed to fetch NSE_PIT: {r.status_code}")
    except Exception as exc:
        log('collector', f"Error collecting NSE_PIT: {exc}")
        print(f"[collector] Error fetching NSE_PIT: {exc}")
    return None


def collect_fiidii() -> str | None:
    """
    Collects raw FII/DII daily flow details from NSE API.
    """
    url = 'https://www.nseindia.com/api/fiidiiTradeReact'
    headers = HEADERS.copy()
    headers['Referer'] = 'https://www.nseindia.com/'
    
    try:
        s = get_nse_session()
        r = s.get(url, headers=headers, timeout=15)
        
        if r.status_code == 200:
            payload = r.text
            record_raw_data('NSE_FII_DII', payload)
            log('collector', f"Collected NSE_FII_DII (size: {len(payload)} chars)")
            return payload
        else:
            log('collector', f"Failed to collect FII_DII. Status: {r.status_code}")
            print(f"[collector] Failed to fetch FII_DII: {r.status_code}")
    except Exception as exc:
        log('collector', f"Error collecting FII_DII: {exc}")
        print(f"[collector] Error fetching FII_DII: {exc}")
    return None


def collect_bulk_deals() -> str | None:
    """
    Downloads raw daily Bulk Deals CSV from NSE archives.
    """
    url = 'https://archives.nseindia.com/content/equities/bulk.csv'
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            payload = r.text
            record_raw_data('NSE_BULK', payload)
            log('collector', f"Collected NSE_BULK CSV (size: {len(payload)} chars)")
            return payload
        else:
            log('collector', f"Failed to collect NSE_BULK. Status: {r.status_code}")
            print(f"[collector] Failed to fetch NSE_BULK: {r.status_code}")
    except Exception as exc:
        log('collector', f"Error collecting NSE_BULK: {exc}")
        print(f"[collector] Error fetching NSE_BULK: {exc}")
    return None


def collect_block_deals() -> str | None:
    """
    Downloads raw daily Block Deals CSV from NSE archives.
    """
    url = 'https://archives.nseindia.com/content/equities/block.csv'
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            payload = r.text
            record_raw_data('NSE_BLOCK', payload)
            log('collector', f"Collected NSE_BLOCK CSV (size: {len(payload)} chars)")
            return payload
        else:
            log('collector', f"Failed to collect NSE_BLOCK. Status: {r.status_code}")
            print(f"[collector] Failed to fetch NSE_BLOCK: {r.status_code}")
    except Exception as exc:
        log('collector', f"Error collecting NSE_BLOCK: {exc}")
        print(f"[collector] Error fetching NSE_BLOCK: {exc}")
    return None


# ── Main Runner ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Collect raw NSE data")
    parser.add_argument('--days', type=int, default=7, help="Lookback days for PIT disclosures")
    args = parser.parse_args()
    
    print(f"[collector] Starting data collection (PIT lookback: {args.days} days)...")
    
    # Run all collectors
    collect_insider_trading(args.days)
    collect_fiidii()
    collect_bulk_deals()
    collect_block_deals()
    
    print("[collector] Data collection complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

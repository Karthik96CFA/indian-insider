#!/usr/bin/env encoding=utf-8
"""
sector_specific_metrics.py — Sector-Specific Scorecard Framework.
Computes a customized fundamental score (0-100) based on industry metrics.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Hardcoded high-quality metrics for watchlist companies
WATCHLIST_SECTOR_DATA = {
    "SBIN": {
        "sector": "Banking",
        "roa": 1.42,
        "roe": 14.0,
        "nim": 3.30,
        "gnpa": 2.21,
        "casa": 41.2
    },
    "INFY": {
        "sector": "IT Services",
        "margin": 20.26,
        "utilization": 82.4,
        "attrition": 12.9,
        "growth": 4.06
    },
    "CIPLA": {
        "sector": "Pharma",
        "roce": 14.86,
        "usfda_obs": 0,
        "rd_spend_pct": 6.2,
        "growth": 7.23
    },
    "DMART": {
        "sector": "Retail",
        "sssg": 11.5,
        "store_growth": 12.1,
        "inventory_turns": 14.4,
        "margin": 6.03
    }
}

def get_sector_score(ticker: str, fundamentals: dict) -> tuple[float, str, dict]:
    """
    Computes a sector-specific score (0 to 100) and returns (score, sector_name, details).
    """
    ticker = ticker.upper()
    db_sector = fundamentals.get("sector") or ""
    
    # 1. Identify Sector
    if ticker in WATCHLIST_SECTOR_DATA:
        sector_name = WATCHLIST_SECTOR_DATA[ticker]["sector"]
    elif "financial" in db_sector.lower() or "bank" in db_sector.lower():
        sector_name = "Banking"
    elif "technology" in db_sector.lower() or "software" in db_sector.lower():
        sector_name = "IT Services"
    elif "health" in db_sector.lower() or "pharma" in db_sector.lower():
        sector_name = "Pharma"
    elif "consumer" in db_sector.lower() or "retail" in db_sector.lower():
        sector_name = "Retail"
    else:
        sector_name = "General"

    score = 0.0
    details = {}

    # 2. Banking Scorecard
    if sector_name == "Banking":
        # Metrics: ROA (30 pts), ROE (25 pts), NIM (15 pts), GNPA (15 pts), CASA (15 pts)
        data = WATCHLIST_SECTOR_DATA.get(ticker, {
            "roa": fundamentals.get("roe") / 10.0 if fundamentals.get("roe") else 1.1, # fallback proxy
            "roe": fundamentals.get("roe") or 12.0,
            "nim": 3.2,
            "gnpa": 2.5,
            "casa": 38.0
        })
        
        # ROA score
        roa = data["roa"]
        roa_score = 30.0 if roa >= 1.5 else (20.0 if roa >= 1.0 else (10.0 if roa >= 0.5 else 0.0))
        # ROE score
        roe = data["roe"]
        roe_score = 25.0 if roe >= 15.0 else (15.0 if roe >= 10.0 else (5.0 if roe >= 5.0 else 0.0))
        # NIM score
        nim = data["nim"]
        nim_score = 15.0 if nim >= 3.5 else (10.0 if nim >= 3.0 else 5.0)
        # GNPA score
        gnpa = data["gnpa"]
        gnpa_score = 15.0 if gnpa < 3.0 else (10.0 if gnpa < 5.0 else 0.0)
        # CASA score
        casa = data["casa"]
        casa_score = 15.0 if casa >= 40.0 else (10.0 if casa >= 35.0 else 5.0)
        
        score = roa_score + roe_score + nim_score + gnpa_score + casa_score
        details = {"roa": roa, "roe": roe, "nim": nim, "gnpa": gnpa, "casa": casa}

    # 3. IT Services Scorecard
    elif sector_name == "IT Services":
        # Metrics: Operating Margin (30 pts), Utilization (25 pts), Attrition (25 pts), Revenue Growth (20 pts)
        margin = fundamentals.get("operating_margin") or 15.0
        data = WATCHLIST_SECTOR_DATA.get(ticker, {
            "margin": margin,
            "utilization": 80.0,
            "attrition": 14.0,
            "growth": fundamentals.get("sales_cagr_3y") or 10.0
        })
        
        margin_score = 30.0 if data["margin"] >= 20.0 else (20.0 if data["margin"] >= 15.0 else 10.0)
        util_score = 25.0 if data["utilization"] >= 82.0 else (15.0 if data["utilization"] >= 78.0 else 5.0)
        attr_score = 25.0 if data["attrition"] < 13.0 else (15.0 if data["attrition"] <= 16.0 else 5.0)
        growth_score = 20.0 if data["growth"] >= 15.0 else (15.0 if data["growth"] >= 10.0 else (10.0 if data["growth"] >= 5.0 else 0.0))
        
        score = margin_score + util_score + attr_score + growth_score
        details = {"margin": data["margin"], "utilization": data["utilization"], "attrition": data["attrition"], "growth": data["growth"]}

    # 4. Pharma Scorecard
    elif sector_name == "Pharma":
        # Metrics: ROCE (25 pts), USFDA Obs (30 pts), R&D Spend (25 pts), Growth (20 pts)
        roce = fundamentals.get("roce") or 15.0
        data = WATCHLIST_SECTOR_DATA.get(ticker, {
            "roce": roce,
            "usfda_obs": 0,
            "rd_spend_pct": 5.5,
            "growth": fundamentals.get("sales_cagr_3y") or 10.0
        })
        
        roce_score = 25.0 if data["roce"] >= 18.0 else (15.0 if data["roce"] >= 14.0 else 5.0)
        usfda_score = 30.0 if data["usfda_obs"] == 0 else (15.0 if data["usfda_obs"] <= 3 else 0.0)
        rd_score = 25.0 if data["rd_spend_pct"] >= 6.0 else (15.0 if data["rd_spend_pct"] >= 4.0 else 5.0)
        growth_score = 20.0 if data["growth"] >= 12.0 else (15.0 if data["growth"] >= 8.0 else 5.0)
        
        score = roce_score + usfda_score + rd_score + growth_score
        details = {"roce": data["roce"], "usfda_obs": data["usfda_obs"], "rd_spend_pct": data["rd_spend_pct"], "growth": data["growth"]}

    # 5. Retail & FMCG Scorecard
    elif sector_name == "Retail":
        # Metrics: SSSG (30 pts), Store Growth (25 pts), Inventory Turns (25 pts), Margins (20 pts)
        margin = fundamentals.get("operating_margin") or 8.0
        data = WATCHLIST_SECTOR_DATA.get(ticker, {
            "sssg": 9.0,
            "store_growth": 8.0,
            "inventory_turns": 10.0,
            "margin": margin
        })
        
        sssg_score = 30.0 if data["sssg"] >= 10.0 else (20.0 if data["sssg"] >= 7.0 else 10.0)
        store_score = 25.0 if data["store_growth"] >= 10.0 else (15.0 if data["store_growth"] >= 6.0 else 5.0)
        inv_score = 25.0 if data["inventory_turns"] >= 12.0 else (15.0 if data["inventory_turns"] >= 8.0 else 5.0)
        margin_score = 20.0 if data["margin"] >= 8.0 else (15.0 if data["margin"] >= 5.0 else 5.0)
        
        score = sssg_score + store_score + inv_score + margin_score
        details = {"sssg": data["sssg"], "store_growth": data["store_growth"], "inventory_turns": data["inventory_turns"], "margin": data["margin"]}

    # 6. General Scorecard (fallback)
    else:
        roce = fundamentals.get("roce") or 0.0
        roe = fundamentals.get("roe") or 0.0
        debt_equity = fundamentals.get("debt_equity") or 0.0
        margin = fundamentals.get("operating_margin") or 0.0
        
        roce_score = 30.0 if roce >= 18.0 else (20.0 if roce >= 14.0 else 10.0)
        roe_score = 30.0 if roe >= 18.0 else (20.0 if roe >= 14.0 else 10.0)
        margin_score = 20.0 if margin >= 15.0 else (10.0 if margin >= 10.0 else 5.0)
        debt_score = 20.0 if debt_equity <= 0.5 else (10.0 if debt_equity <= 1.0 else 0.0)
        
        score = roce_score + roe_score + margin_score + debt_score
        details = {"roce": roce, "roe": roe, "operating_margin": margin, "debt_equity": debt_equity}

    return score, sector_name, details

def main() -> int:
    import argparse
    from common import read_company_fundamentals
    
    parser = argparse.ArgumentParser(description="Evaluate company sector metrics")
    parser.add_argument('--ticker', type=str, required=True, help="Symbol to evaluate (e.g. INFY)")
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    fundamentals = read_company_fundamentals(ticker)
    if not fundamentals:
        print(f"Fundamentals not found for {ticker}")
        return 1
        
    score, sector, details = get_sector_score(ticker, fundamentals)
    print(f"Ticker: {ticker}")
    print(f"Sector Classification: {sector}")
    print(f"Sector Fundamental Score: {score:.1f}/100")
    print(f"Details: {details}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

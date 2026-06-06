#!/usr/bin/env python3
"""
event_detector.py — Event Detector & Typo Validator.
Parses raw data warehouse payloads and populates the structured market_events table.
"""
from __future__ import annotations

import csv
import datetime
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _conn, log, record_market_event


# ── Ticker cleaning and validation ──────────────────────────────────────────

VALID_TICKERS = {
    "360ONE",
    "3MINDIA",
    "AADHARHFC",
    "AARTIIND",
    "AAVAS",
    "ABB",
    "ABBOTINDIA",
    "ABCAPITAL",
    "ABDL",
    "ABFRL",
    "ABLBL",
    "ABREL",
    "ABSLAMC",
    "ACC",
    "ACE",
    "ACMESOLAR",
    "ACUTAAS",
    "ADANIENSOL",
    "ADANIENT",
    "ADANIGREEN",
    "ADANIPORTS",
    "ADANIPOWER",
    "AEGISLOG",
    "AEGISVOPAK",
    "AFCONS",
    "AFFLE",
    "AIAENG",
    "AIIL",
    "AJANTPHARM",
    "ALKEM",
    "AMBER",
    "AMBUJACEM",
    "ANANDRATHI",
    "ANANTRAJ",
    "ANGELONE",
    "ANTHEM",
    "ANURAS",
    "APARINDS",
    "APLAPOLLO",
    "APOLLOHOSP",
    "APOLLOTYRE",
    "APTUS",
    "ARE&M",
    "ASAHIINDIA",
    "ASHOKLEY",
    "ASIANPAINT",
    "ASTERDM",
    "ASTRAL",
    "ATGL",
    "ATHERENERG",
    "ATUL",
    "AUBANK",
    "AUROPHARMA",
    "AWL",
    "AXISBANK",
    "BAJAJFINSV",
    "BAJAJHFL",
    "BAJAJHLDNG",
    "BAJAJ_AUTO",
    "BAJFINANCE",
    "BALKRISIND",
    "BALRAMCHIN",
    "BANDHANBNK",
    "BANKBARODA",
    "BANKINDIA",
    "BANKNIFTY",
    "BATAINDIA",
    "BAYERCROP",
    "BBTC",
    "BDL",
    "BEL",
    "BELRISE",
    "BEML",
    "BERGEPAINT",
    "BHARATFORG",
    "BHARTIARTL",
    "BHARTIHEXA",
    "BHEL",
    "BIKAJI",
    "BIOCON",
    "BLS",
    "BLUEDART",
    "BLUEJET",
    "BLUESTARCO",
    "BOSCHLTD",
    "BPCL",
    "BRIGADE",
    "BRITANNIA",
    "BSE",
    "BSOFT",
    "CAMS",
    "CANBK",
    "CANFINHOME",
    "CANHLIFE",
    "CAPLIPOINT",
    "CARBORUNIV",
    "CARTRADE",
    "CASTROLIND",
    "CCL",
    "CDSL",
    "CEATLTD",
    "CEMPRO",
    "CENTRALBK",
    "CESC",
    "CGCL",
    "CGPOWER",
    "CHALET",
    "CHAMBLFERT",
    "CHENNPETRO",
    "CHOICEIN",
    "CHOLAFIN",
    "CHOLAHLDNG",
    "CIEINDIA",
    "CIPLA",
    "CLEAN",
    "COALINDIA",
    "COCHINSHIP",
    "COFORGE",
    "COHANCE",
    "COLPAL",
    "CONCOR",
    "CONCORDBIO",
    "COROMANDEL",
    "CPPLUS",
    "CRAFTSMAN",
    "CREDITACC",
    "CRISIL",
    "CROMPTON",
    "CUB",
    "CUMMINSIND",
    "CYIENT",
    "DABUR",
    "DALBHARAT",
    "DATAPATTNS",
    "DCMSHRIRAM",
    "DEEPAKFERT",
    "DEEPAKNTR",
    "DELHIVERY",
    "DEVYANI",
    "DIVISLAB",
    "DIXON",
    "DLF",
    "DMART",
    "DOMS",
    "DRREDDY",
    "DUMMYVEDL1",
    "DUMMYVEDL2",
    "DUMMYVEDL3",
    "DUMMYVEDL4",
    "ECLERX",
    "EICHERMOT",
    "EIDPARRY",
    "EIHOTEL",
    "ELECON",
    "ELGIEQUIP",
    "EMAMILTD",
    "EMCURE",
    "EMMVEE",
    "ENDURANCE",
    "ENGINERSIN",
    "ENRIN",
    "ERIS",
    "ESCORTS",
    "ETERNAL",
    "EXIDEIND",
    "FACT",
    "FEDERALBNK",
    "FINCABLES",
    "FIRSTCRY",
    "FIVESTAR",
    "FLUOROCHEM",
    "FORCEMOT",
    "FORTIS",
    "FSL",
    "GABRIEL",
    "GAIL",
    "GALLANTT",
    "GESHIP",
    "GICRE",
    "GILLETTE",
    "GLAND",
    "GLAXO",
    "GLENMARK",
    "GMDCLTD",
    "GMRAIRPORT",
    "GMRINFRA",
    "GODFRYPHLP",
    "GODIGIT",
    "GODREJCP",
    "GODREJIND",
    "GODREJPROP",
    "GOLDBEES",
    "GPIL",
    "GRANULES",
    "GRAPHITE",
    "GRASIM",
    "GRAVITA",
    "GROWW",
    "GRSE",
    "GVT&D",
    "HAL",
    "HAVELLS",
    "HBLENGINE",
    "HCLTECH",
    "HDBFS",
    "HDFCAMC",
    "HDFCBANK",
    "HDFCLIFE",
    "HEG",
    "HEROMOTOCO",
    "HEXT",
    "HFCL",
    "HINDALCO",
    "HINDCOPPER",
    "HINDPETRO",
    "HINDUNILVR",
    "HINDZINC",
    "HOMEFIRST",
    "HONASA",
    "HONAUT",
    "HSCL",
    "HUDCO",
    "HYUNDAI",
    "ICICIAMC",
    "ICICIBANK",
    "ICICIGI",
    "ICICIPRULI",
    "IDBI",
    "IDEA",
    "IDFCFIRSTB",
    "IEX",
    "IFCI",
    "IGIL",
    "IGL",
    "IIFL",
    "IKS",
    "INDGN",
    "INDHOTEL",
    "INDIACEM",
    "INDIAMART",
    "INDIANB",
    "INDIGO",
    "INDUSINDBK",
    "INDUSTOWER",
    "INFY",
    "INOXWIND",
    "INTELLECT",
    "IOB",
    "IOC",
    "IPCALAB",
    "IRB",
    "IRCON",
    "IRCTC",
    "IREDA",
    "IRFC",
    "ITC",
    "ITCHOTELS",
    "ITI",
    "J&KBANK",
    "JAINREC",
    "JBCHEPHARM",
    "JBMA",
    "JINDALSAW",
    "JINDALSTEL",
    "JIOFIN",
    "JKCEMENT",
    "JKTYRE",
    "JMFINANCIL",
    "JPPOWER",
    "JSL",
    "JSWCEMENT",
    "JSWDULUX",
    "JSWENERGY",
    "JSWINFRA",
    "JSWSTEEL",
    "JUBLFOOD",
    "JUBLINGREA",
    "JUBLPHARMA",
    "JWL",
    "JYOTICNC",
    "KAJARIACER",
    "KALYANKJIL",
    "KARURVYSYA",
    "KAYNES",
    "KEC",
    "KEI",
    "KFINTECH",
    "KIMS",
    "KIRLOSENG",
    "KOTAKBANK",
    "KPIL",
    "KPITTECH",
    "KPRMILL",
    "LALPATHLAB",
    "LATENTVIEW",
    "LAURUSLABS",
    "LEMONTREE",
    "LENSKART",
    "LGEINDIA",
    "LICHSGFIN",
    "LICI",
    "LINDEINDIA",
    "LLOYDSME",
    "LODHA",
    "LT",
    "LTF",
    "LTFOODS",
    "LTIM",
    "LTM",
    "LTTS",
    "LUPIN",
    "M&M",
    "M&MFIN",
    "MAHABANK",
    "MANAPPURAM",
    "MANKIND",
    "MAPMYINDIA",
    "MARICO",
    "MARUTI",
    "MAXHEALTH",
    "MAZDOCK",
    "MCX",
    "MEDANTA",
    "MEESHO",
    "MFSL",
    "MGL",
    "MINDACORP",
    "MMTC",
    "MOTHERSON",
    "MOTILALOFS",
    "MPHASIS",
    "MRF",
    "MRPL",
    "MSUMI",
    "MUTHOOTFIN",
    "NAM_INDIA",
    "NATCOPHARM",
    "NATIONALUM",
    "NAUKRI",
    "NAVA",
    "NAVINFLUOR",
    "NBCC",
    "NCC",
    "NESTLEIND",
    "NETWEB",
    "NEULANDLAB",
    "NEWGEN",
    "NH",
    "NHPC",
    "NIACL",
    "NIFTY",
    "NIFTYBEES",
    "NIVABUPA",
    "NLCINDIA",
    "NMDC",
    "NSLNISP",
    "NTPC",
    "NTPCGREEN",
    "NUVAMA",
    "NUVOCO",
    "NYKAA",
    "OBEROIRLTY",
    "OFSS",
    "OIL",
    "OLAELEC",
    "OLECTRA",
    "ONESOURCE",
    "ONGC",
    "PAGEIND",
    "PARADEEP",
    "PATANJALI",
    "PAYTM",
    "PCBL",
    "PERSISTENT",
    "PETRONET",
    "PFC",
    "PFIZER",
    "PGEL",
    "PHOENIXLTD",
    "PIDILITIND",
    "PIIND",
    "PINELABS",
    "PIRAMALFIN",
    "PNB",
    "PNBHOUSING",
    "POLICYBZR",
    "POLYCAB",
    "POLYMED",
    "POONAWALLA",
    "POWERGRID",
    "POWERINDIA",
    "PPLPHARMA",
    "PREMIERENE",
    "PRESTIGE",
    "PTCIL",
    "PVRINOX",
    "PWL",
    "RADICO",
    "RAILTEL",
    "RAINBOW",
    "RAMCOCEM",
    "RBLBANK",
    "RECL",
    "RECLTD",
    "REDINGTON",
    "RELIANCE",
    "RHIM",
    "RITES",
    "RKFORGE",
    "RPOWER",
    "RRKABEL",
    "RVNL",
    "SAGILITY",
    "SAIL",
    "SAILIFE",
    "SAMMAANCAP",
    "SAPPHIRE",
    "SARDAEN",
    "SAREGAMA",
    "SBFC",
    "SBICARD",
    "SBILIFE",
    "SBIN",
    "SCHAEFFLER",
    "SCHNEIDER",
    "SCI",
    "SHREECEM",
    "SHRIRAMFIN",
    "SHYAMMETL",
    "SIEMENS",
    "SIGNATURE",
    "SJVN",
    "SOBHA",
    "SOLARINDS",
    "SONACOMS",
    "SONATSOFTW",
    "SPLPETRO",
    "SRF",
    "STARHEALTH",
    "SUMICHEM",
    "SUNDARMFIN",
    "SUNPHARMA",
    "SUNTV",
    "SUPREMEIND",
    "SUZLON",
    "SWANCORP",
    "SWIGGY",
    "SYNGENE",
    "SYRMA",
    "TARIL",
    "TATACAP",
    "TATACHEM",
    "TATACOMM",
    "TATACONSUM",
    "TATAELXSI",
    "TATAINVEST",
    "TATAMOTORS",
    "TATAPOWER",
    "TATASTEEL",
    "TATATECH",
    "TBOTEK",
    "TCS",
    "TECHM",
    "TECHNOE",
    "TEGA",
    "TEJASNET",
    "TENNIND",
    "THELEELA",
    "THERMAX",
    "TIINDIA",
    "TIMKEN",
    "TITAGARH",
    "TITAN",
    "TMCV",
    "TMPV",
    "TORNTPHARM",
    "TORNTPOWER",
    "TRAVELFOOD",
    "TRENT",
    "TRENTS",
    "TRIDENT",
    "TRITURBINE",
    "TTML",
    "TVSMOTOR",
    "UBL",
    "UCOBANK",
    "ULTRACEMCO",
    "UNIONBANK",
    "UNITDSPR",
    "UNOMINDA",
    "UPL",
    "URBANCO",
    "USHAMART",
    "UTIAMC",
    "VBL",
    "VEDL",
    "VIJAYA",
    "VMM",
    "VOLTAS",
    "VTL",
    "WAAREEENER",
    "WELCORP",
    "WELSPUNLIV",
    "WHIRLPOOL",
    "WIPRO",
    "WOCKPHARMA",
    "YESBANK",
    "ZEEL",
    "ZENSARTECH",
    "ZENTEC",
    "ZFCVINDIA",
    "ZYDUSLIFE",
    "ZYDUSWELL"
}

TYPO_CORRECTIONS = {
    "RELIANEC": "RELIANCE",
    "HDFCBAN": "HDFCBANK",
    "INFYS": "INFY",
    "TCSS": "TCS",
    "REL": "RELIANCE",
    "NIFTY_50": "NIFTY",
    "NIFTY50": "NIFTY",
}


def clean_ticker(symbol: str) -> str:
    s = str(symbol).upper().strip()
    # Strip common suffixes
    if s.endswith("-EQ"):
        s = s[:-3]
    if s.endswith(".NS") or s.endswith(".BO"):
        s = s[:-3]
    # Apply manual corrections
    s = TYPO_CORRECTIONS.get(s, s)
    # Check alphanumeric structure
    s = re.sub(r'[^A-Z0-9_&]', '', s)
    return s


def is_valid_ticker(symbol: str) -> bool:
    cleaned = clean_ticker(symbol)
    # If it matches NIFTY list or looks like a valid symbol structure
    if cleaned in VALID_TICKERS:
        return True
    # Allow other symbols if they are alphanumeric and length 2 to 15
    if len(cleaned) >= 2 and len(cleaned) <= 15 and cleaned.isalpha():
        return True
    return False


# ── Parsing Logic ────────────────────────────────────────────────────────────

def parse_nse_pit(payload: str) -> list[dict]:
    """
    Parses corporate insider trading JSON (SEBI PIT).
    """
    events = []
    try:
        data_dict = json.loads(payload)
        rows = data_dict.get('data', [])
        for row in rows:
            symbol = row.get('symbol')
            if not symbol:
                continue
            
            cleaned_sym = clean_ticker(symbol)
            if not is_valid_ticker(cleaned_sym):
                continue
            
            acq_mode = str(row.get('acqMode', '')).upper()
            txn_type = str(row.get('tdpTransactionType', '')).upper()
            person_cat = str(row.get('personCategory', '')).lower()
            
            # Filter to Promoters or Promoter Group
            is_promoter = 'promoter' in person_cat
            if not is_promoter:
                continue
                
            # Filter to open market purchases / sales
            is_market = 'MARKET' in acq_mode
            if not is_market:
                continue
                
            try:
                value = float(row.get('secVal') or 0)
            except ValueError:
                value = 0.0
                
            if value <= 0:
                continue
                
            direction = 'BULLISH' if txn_type == 'BUY' else 'BEARISH'
            event_type = 'PROMOTER_BUY' if txn_type == 'BUY' else 'PROMOTER_SELL'
            
            # Parse Date
            raw_date = row.get('acqtoDt') or row.get('date') or ''
            # Convert '29-Apr-2026' or '02-May-2026 16:46' to ISO date string
            try:
                date_str = raw_date.split(' ')[0]
                parsed_date = datetime.datetime.strptime(date_str, '%d-%b-%Y').date().isoformat()
            except Exception:
                parsed_date = datetime.date.today().isoformat()
                
            events.append({
                'ticker': cleaned_sym,
                'event_type': event_type,
                'value': value,
                'direction': direction,
                'metadata': json.dumps(row),
                'event_date': parsed_date
            })
    except Exception as exc:
        log('detector', f"Error parsing NSE_PIT: {exc}")
    return events


def parse_nse_fiidii(payload: str) -> list[dict]:
    """
    Parses FII/DII daily flows JSON.
    """
    events = []
    try:
        rows = json.loads(payload)
        for row in rows:
            category = str(row.get('category', '')).upper()
            try:
                # Value is in Crores, convert to absolute INR by multiplying by 10,000,000
                net_val_cr = float(row.get('netValue') or 0)
                value = net_val_cr * 10000000.0
            except ValueError:
                value = 0.0
                
            if value == 0:
                continue
                
            event_type = 'FII_NET_FLOW' if 'FII' in category else 'DII_NET_FLOW'
            direction = 'BULLISH' if value > 0 else 'BEARISH'
            
            # Parse Date '05-Jun-2026' -> '2026-06-05'
            raw_date = row.get('date') or ''
            try:
                parsed_date = datetime.datetime.strptime(raw_date, '%d-%b-%Y').date().isoformat()
            except Exception:
                parsed_date = datetime.date.today().isoformat()
                
            events.append({
                'ticker': 'NIFTY',
                'event_type': event_type,
                'value': abs(value),
                'direction': direction,
                'metadata': json.dumps(row),
                'event_date': parsed_date
            })
    except Exception as exc:
        log('detector', f"Error parsing NSE_FII_DII: {exc}")
    return events


def parse_deals_csv(payload: str, is_bulk: bool = True) -> list[dict]:
    """
    Parses Bulk/Block deals CSV payload.
    """
    events = []
    lines = payload.splitlines()
    if not lines:
        return events
        
    reader = csv.DictReader(lines)
    for row in reader:
        symbol = row.get('Symbol')
        if not symbol:
            continue
            
        cleaned_sym = clean_ticker(symbol)
        if not is_valid_ticker(cleaned_sym):
            continue
            
        txn_type = str(row.get('Buy/Sell', '')).upper()
        try:
            qty = float(row.get('Quantity Traded') or 0)
            price = float(row.get('Trade Price / Wght. Avg. Price') or 0)
            value = qty * price
        except ValueError:
            value = 0.0
            
        if value <= 0:
            continue
            
        event_type = 'BULK_DEAL' if is_bulk else 'BLOCK_DEAL'
        direction = 'BULLISH' if txn_type == 'BUY' else 'BEARISH'
        
        # Parse Date '05-JUN-2026' -> '2026-06-05'
        raw_date = row.get('Date') or ''
        try:
            parsed_date = datetime.datetime.strptime(raw_date.upper(), '%d-%b-%Y').date().isoformat()
        except Exception:
            parsed_date = datetime.date.today().isoformat()
            
        events.append({
            'ticker': cleaned_sym,
            'event_type': event_type,
            'value': value,
            'direction': direction,
            'metadata': json.dumps(row),
            'event_date': parsed_date
        })
    return events


# ── Main Processing Loop ─────────────────────────────────────────────────────

def main() -> int:
    print("[detector] Starting event detection and normalization...")
    
    conn = _conn()
    
    # 1. Fetch raw data records from warehouse
    rows = conn.execute("SELECT id, source, payload, ts FROM raw_data_warehouse ORDER BY id ASC").fetchall()
    
    detected_events = []
    
    for r_id, source, payload, ts in rows:
        current_events = []
        if source == 'NSE_PIT':
            current_events = parse_nse_pit(payload)
        elif source == 'NSE_FII_DII':
            current_events = parse_nse_fiidii(payload)
        elif source == 'NSE_BULK':
            current_events = parse_deals_csv(payload, is_bulk=True)
        elif source == 'NSE_BLOCK':
            current_events = parse_deals_csv(payload, is_bulk=False)
            
        for ev in current_events:
            ev['ts'] = ts
            detected_events.append(ev)
            
    # 2. Clear old normalized market events
    conn.execute("DELETE FROM market_events")
    conn.commit()
    
    # 3. Store new normalized market events
    for ev in detected_events:
        record_market_event(
            ticker=ev['ticker'],
            event_type=ev['event_type'],
            value=ev['value'],
            direction=ev['direction'],
            metadata=ev['metadata'],
            event_date=ev['event_date'],
            ts=ev.get('ts')
        )
        
    print(f"[detector] Processing complete. Loaded {len(detected_events)} events into market_events.")
    log('detector', f"Processed raw warehouse data and loaded {len(detected_events)} events.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
common.py — shared foundation for the 7 Indian Insider agents.
Powered by Google Gemini 3.1 Pro via Google AI Studio.

Scouts: Doraemon, Shinchan, Nobita, Dekisugi, Suneo
Consensus: Doraemi (pure local logic, no Gemini calls)
Dispatcher: Gian (pure local logic, no Gemini calls)

State:  ~/indian-insider/.state/state.db (SQLite)
Config: ~/indian-insider/.env
"""

from __future__ import annotations

import json
import os
import smtplib
import sqlite3
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    sys.stderr.write("Run: pip install python-dotenv\n")
    raise

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    sys.stderr.write("Run: pip install google-genai\n")
    raise


# ── Paths ────────────────────────────────────────────────────────────────────

ROOT    = Path.home() / "indian-insider"
STATE   = ROOT / ".state"
LOGS    = STATE / "logs"
DB_PATH = STATE / "state.db"
ENV_PATH = ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


# ── Gemini setup ─────────────────────────────────────────────────────────────

GEMINI_MODEL      = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MODEL_FAST = GEMINI_MODEL  # alias; configure via GEMINI_MODEL env var

_gemini_client: "genai.Client | None" = None


def get_gemini() -> "genai.Client":
    """Return a cached google.genai Client instance."""
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Add it to ~/indian-insider/.env\n"
            "Get your key at: https://aistudio.google.com/app/apikey"
        )
    _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# ── Direction taxonomy ───────────────────────────────────────────────────────

BULLISH    = "BULLISH"
BEARISH    = "BEARISH"
NEUTRAL    = "NEUTRAL"
DIRECTIONS = (BULLISH, BEARISH, NEUTRAL)


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class Signal:
    scout:      str
    ticker:     str       # NSE symbol e.g. "RELIANCE", "NIFTY", or "MACRO"
    direction:  str       # BULLISH | BEARISH | NEUTRAL
    confidence: int       # 1–5
    reason:     str       # one-line plain-English reason
    raw:        str       # full model output for audit


@dataclass
class ConsensusEvent:
    ticker:    str
    direction: str
    scouts:    list[str]
    reasons:   list[str]
    timestamp: datetime


# ── State store ──────────────────────────────────────────────────────────────

class DBAdapter:
    def __init__(self, is_postgres: bool, conn_obj):
        self.is_postgres = is_postgres
        self.conn = conn_obj

    def execute(self, sql: str, params: tuple = ()):
        if self.is_postgres:
            sql = sql.replace("?", "%s")
            sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
            cur = self.conn.cursor()
            cur.execute(sql, params)
            return cur
        else:
            return self.conn.execute(sql, params)

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()
        else:
            self.commit()
        self.close()


def _ensure_dirs() -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)


def _open_conn() -> DBAdapter:
    """Open a raw DB connection. Does NOT run DDL — call initialize_db() first."""
    database_url = os.environ.get("DATABASE_URL")
    is_postgres = False
    if database_url and (database_url.startswith("postgresql://") or database_url.startswith("postgres://")):
        try:
            import psycopg2
            raw_conn = psycopg2.connect(database_url)
            is_postgres = True
        except Exception as exc:
            sys.stderr.write(f"PostgreSQL connection failed: {exc}. Falling back to SQLite.\n")
            _ensure_dirs()
            raw_conn = sqlite3.connect(DB_PATH)
    else:
        _ensure_dirs()
        raw_conn = sqlite3.connect(DB_PATH)
    return DBAdapter(is_postgres, raw_conn)


def _run_ddl(adapter: DBAdapter) -> None:
    """Create all tables and indices. Idempotent — safe to call on an existing DB."""
    adapter.execute("""CREATE TABLE IF NOT EXISTS signals (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        scout      TEXT NOT NULL,
        ticker     TEXT NOT NULL,
        direction  TEXT NOT NULL,
        confidence INTEGER NOT NULL,
        reason     TEXT NOT NULL,
        raw        TEXT NOT NULL,
        ts         TEXT NOT NULL
    )""")
    adapter.execute("""CREATE TABLE IF NOT EXISTS consensus (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker     TEXT NOT NULL,
        direction  TEXT NOT NULL,
        scouts     TEXT NOT NULL,
        reasons    TEXT NOT NULL,
        ts         TEXT NOT NULL,
        dispatched INTEGER DEFAULT 0
    )""")
    adapter.execute("""CREATE TABLE IF NOT EXISTS raw_data_warehouse (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        source     TEXT NOT NULL,
        payload    TEXT NOT NULL,
        ts         TEXT NOT NULL
    )""")
    adapter.execute("""CREATE TABLE IF NOT EXISTS market_events (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker     TEXT NOT NULL,
        event_type TEXT NOT NULL,
        value      REAL NOT NULL,
        direction  TEXT NOT NULL,
        metadata   TEXT NOT NULL,
        event_date TEXT NOT NULL,
        ts         TEXT
    )""")
    adapter.execute("""CREATE TABLE IF NOT EXISTS historical_prices (
        ticker TEXT NOT NULL,
        date   TEXT NOT NULL,
        close  REAL NOT NULL,
        PRIMARY KEY (ticker, date)
    )""")
    adapter.execute("""CREATE TABLE IF NOT EXISTS company_fundamentals (
        ticker              TEXT PRIMARY KEY,
        roce                REAL,
        roe                 REAL,
        debt_equity         REAL,
        operating_margin    REAL,
        fcf                 REAL,
        sales_cagr_3y       REAL,
        eps_growth_3y       REAL,
        inst_holding_change REAL,
        sector              TEXT,
        market_cap          REAL,
        last_updated        TEXT NOT NULL
    )""")
    adapter.execute("""CREATE TABLE IF NOT EXISTS valuation_metrics (
        ticker             TEXT PRIMARY KEY,
        pe                 REAL,
        peg                REAL,
        ev_ebitda          REAL,
        fcf_yield          REAL,
        implied_dcf_growth REAL,
        last_updated       TEXT NOT NULL
    )""")
    adapter.execute("""CREATE TABLE IF NOT EXISTS company_scores (
        ticker                  TEXT PRIMARY KEY,
        event_score             REAL DEFAULT 0.0,
        fundamental_score       REAL DEFAULT 0.0,
        valuation_score         REAL DEFAULT 0.0,
        canslim_score           INTEGER DEFAULT 0,
        multibagger_score       INTEGER DEFAULT 0,
        total_score             REAL DEFAULT 0.0,
        credibility_score       REAL DEFAULT 0.0,
        industry_tailwind_score REAL DEFAULT 0.0,
        promise_count           INTEGER DEFAULT 0,
        coverage_score          REAL DEFAULT 0.0,
        confidence_score        REAL DEFAULT 0.0,
        last_updated            TEXT NOT NULL
    )""")
    adapter.execute("""CREATE TABLE IF NOT EXISTS company_scores_history (
        ticker                  TEXT,
        event_score             REAL DEFAULT 0.0,
        fundamental_score       REAL DEFAULT 0.0,
        valuation_score         REAL DEFAULT 0.0,
        canslim_score           INTEGER DEFAULT 0,
        multibagger_score       INTEGER DEFAULT 0,
        total_score             REAL DEFAULT 0.0,
        credibility_score       REAL DEFAULT 0.0,
        industry_tailwind_score REAL DEFAULT 0.0,
        promise_count           INTEGER DEFAULT 0,
        coverage_score          REAL DEFAULT 0.0,
        confidence_score        REAL DEFAULT 0.0,
        effective_date          TEXT NOT NULL,
        PRIMARY KEY (ticker, effective_date)
    )""")
    adapter.execute("""CREATE TABLE IF NOT EXISTS management_promises (
        id                         INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker                     TEXT NOT NULL,
        promise_date               TEXT NOT NULL,
        speaker                    TEXT,
        promise_type               TEXT NOT NULL,
        period                     TEXT,
        guidance_revision_chain_id TEXT,
        statement                  TEXT NOT NULL,
        lower_bound                REAL,
        upper_bound                REAL,
        target_value               REAL,
        actual_value               REAL,
        fulfilled                  INTEGER DEFAULT 0,
        fulfillment_date           TEXT,
        credibility_impact         REAL DEFAULT 0.0,
        ts                         TEXT NOT NULL
    )""")
    adapter.execute("""CREATE TABLE IF NOT EXISTS research_memory (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker           TEXT NOT NULL,
        transcript_hash  TEXT NOT NULL,
        raw_text         TEXT NOT NULL,
        speaker          TEXT,
        statement_type   TEXT,
        guidance_type    TEXT,
        guidance_value   REAL,
        period           TEXT,
        confidence       REAL,
        embedding        TEXT,
        ts               TEXT NOT NULL
    )""")
    adapter.execute("""CREATE TABLE IF NOT EXISTS paper_trades (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker       TEXT NOT NULL,
        side         TEXT NOT NULL CHECK(side IN ('BUY','SELL')),
        qty          INTEGER NOT NULL,
        entry_price  REAL NOT NULL,
        entry_date   TEXT NOT NULL,
        exit_price   REAL,
        exit_date    TEXT,
        status       TEXT NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN','CLOSED')),
        reason       TEXT,
        ts           TEXT NOT NULL
    )""")
    adapter.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_ticker ON paper_trades(ticker)")
    adapter.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status)")

    # Backward-compat column additions — ignore errors if column already exists
    _silent_alters = [
        "ALTER TABLE company_fundamentals ADD COLUMN sector TEXT",
        "ALTER TABLE company_fundamentals ADD COLUMN market_cap REAL",
        "ALTER TABLE company_scores ADD COLUMN credibility_score REAL DEFAULT 0.0",
        "ALTER TABLE company_scores ADD COLUMN industry_tailwind_score REAL DEFAULT 0.0",
        "ALTER TABLE company_scores ADD COLUMN promise_count INTEGER DEFAULT 0",
        "ALTER TABLE company_scores ADD COLUMN coverage_score REAL DEFAULT 0.0",
        "ALTER TABLE company_scores ADD COLUMN confidence_score REAL DEFAULT 0.0",
        "ALTER TABLE company_scores_history ADD COLUMN confidence_score REAL DEFAULT 0.0",
        "ALTER TABLE market_events ADD COLUMN ts TEXT",
    ]
    for stmt in _silent_alters:
        try:
            adapter.execute(stmt)
        except Exception:
            pass
    # Indices
    adapter.execute("CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts)")
    adapter.execute("CREATE INDEX IF NOT EXISTS idx_raw_ts ON raw_data_warehouse(ts)")
    adapter.execute("CREATE INDEX IF NOT EXISTS idx_events_date ON market_events(event_date)")
    adapter.execute("CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON historical_prices(ticker, date)")
    adapter.execute("CREATE INDEX IF NOT EXISTS idx_promises_ticker_revision ON management_promises(ticker, guidance_revision_chain_id)")
    adapter.execute("CREATE INDEX IF NOT EXISTS idx_research_memory_ticker ON research_memory(ticker)")


_db_initialized = False


def initialize_db() -> None:
    """Run DDL once per process. Safe to call multiple times."""
    global _db_initialized
    if _db_initialized:
        return
    _ensure_dirs()
    adapter = _open_conn()
    try:
        _run_ddl(adapter)
        adapter.commit()
    finally:
        adapter.close()
    _db_initialized = True


def _conn() -> DBAdapter:
    """Return a DB connection, ensuring the schema is initialized."""
    initialize_db()
    return _open_conn()


def record_signal(sig: Signal) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO signals (scout,ticker,direction,confidence,reason,raw,ts) "
            "VALUES (?,?,?,?,?,?,?)",
            (sig.scout, sig.ticker, sig.direction, sig.confidence,
             sig.reason, sig.raw, datetime.now(timezone.utc).isoformat()),
        )


def read_window(days: int = 7) -> list[Signal]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _conn() as c:
        rows = c.execute(
            "SELECT scout,ticker,direction,confidence,reason,raw "
            "FROM signals WHERE ts >= ? ORDER BY ts DESC",
            (cutoff,),
        ).fetchall()
    return [Signal(*r) for r in rows]


def record_consensus(ev: ConsensusEvent) -> int:
    with _conn() as c:
        if c.is_postgres:
            cur = c.execute(
                "INSERT INTO consensus (ticker,direction,scouts,reasons,ts) VALUES (?,?,?,?,?) RETURNING id",
                (ev.ticker, ev.direction, json.dumps(ev.scouts),
                 json.dumps(ev.reasons), ev.timestamp.isoformat()),
            )
            return int(cur.fetchone()[0])
        else:
            cur = c.execute(
                "INSERT INTO consensus (ticker,direction,scouts,reasons,ts) VALUES (?,?,?,?,?)",
                (ev.ticker, ev.direction, json.dumps(ev.scouts),
                 json.dumps(ev.reasons), ev.timestamp.isoformat()),
            )
            return int(cur.lastrowid or 0)


def record_company_fundamentals(
    ticker: str,
    roce: float,
    roe: float,
    debt_equity: float,
    operating_margin: float,
    fcf: float,
    sales_cagr_3y: float,
    eps_growth_3y: float,
    inst_holding_change: float,
    sector: str | None = None,
    market_cap: float | None = None,
) -> None:
    now_str = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        if c.is_postgres:
            c.execute(
                """INSERT INTO company_fundamentals
                (ticker, roce, roe, debt_equity, operating_margin, fcf, sales_cagr_3y, eps_growth_3y, inst_holding_change, sector, market_cap, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker) DO UPDATE SET
                roce=EXCLUDED.roce, roe=EXCLUDED.roe, debt_equity=EXCLUDED.debt_equity,
                operating_margin=EXCLUDED.operating_margin, fcf=EXCLUDED.fcf, sales_cagr_3y=EXCLUDED.sales_cagr_3y,
                eps_growth_3y=EXCLUDED.eps_growth_3y, inst_holding_change=EXCLUDED.inst_holding_change,
                sector=EXCLUDED.sector, market_cap=EXCLUDED.market_cap, last_updated=EXCLUDED.last_updated""",
                (ticker, roce, roe, debt_equity, operating_margin, fcf, sales_cagr_3y, eps_growth_3y, inst_holding_change, sector, market_cap, now_str)
            )
        else:
            c.execute(
                """INSERT INTO company_fundamentals
                (ticker, roce, roe, debt_equity, operating_margin, fcf, sales_cagr_3y, eps_growth_3y, inst_holding_change, sector, market_cap, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker) DO UPDATE SET
                roce=excluded.roce, roe=excluded.roe, debt_equity=excluded.debt_equity,
                operating_margin=excluded.operating_margin, fcf=excluded.fcf, sales_cagr_3y=excluded.sales_cagr_3y,
                eps_growth_3y=excluded.eps_growth_3y, inst_holding_change=excluded.inst_holding_change,
                sector=excluded.sector, market_cap=excluded.market_cap, last_updated=excluded.last_updated""",
                (ticker, roce, roe, debt_equity, operating_margin, fcf, sales_cagr_3y, eps_growth_3y, inst_holding_change, sector, market_cap, now_str)
            )


def read_company_fundamentals(ticker: str) -> dict[str, Any] | None:
    with _conn() as c:
        row = c.execute(
            "SELECT roce, roe, debt_equity, operating_margin, fcf, sales_cagr_3y, eps_growth_3y, inst_holding_change, sector, last_updated "
            "FROM company_fundamentals WHERE ticker=?", (ticker,)
        ).fetchone()
    if not row:
        return None
    return {
        "ticker": ticker,
        "roce": row[0],
        "roe": row[1],
        "debt_equity": row[2],
        "operating_margin": row[3],
        "fcf": row[4],
        "sales_cagr_3y": row[5],
        "eps_growth_3y": row[6],
        "inst_holding_change": row[7],
        "sector": row[8],
        "last_updated": row[9],
    }


def record_valuation_metrics(
    ticker: str,
    pe: float,
    peg: float,
    ev_ebitda: float,
    fcf_yield: float,
    implied_dcf_growth: float,
) -> None:
    now_str = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        if c.is_postgres:
            c.execute(
                """INSERT INTO valuation_metrics 
                (ticker, pe, peg, ev_ebitda, fcf_yield, implied_dcf_growth, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker) DO UPDATE SET
                pe=EXCLUDED.pe, peg=EXCLUDED.peg, ev_ebitda=EXCLUDED.ev_ebitda,
                fcf_yield=EXCLUDED.fcf_yield, implied_dcf_growth=EXCLUDED.implied_dcf_growth, last_updated=EXCLUDED.last_updated""",
                (ticker, pe, peg, ev_ebitda, fcf_yield, implied_dcf_growth, now_str)
            )
        else:
            c.execute(
                """INSERT INTO valuation_metrics 
                (ticker, pe, peg, ev_ebitda, fcf_yield, implied_dcf_growth, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker) DO UPDATE SET
                pe=excluded.pe, peg=excluded.peg, ev_ebitda=excluded.ev_ebitda,
                fcf_yield=excluded.fcf_yield, implied_dcf_growth=excluded.implied_dcf_growth, last_updated=excluded.last_updated""",
                (ticker, pe, peg, ev_ebitda, fcf_yield, implied_dcf_growth, now_str)
            )


def read_valuation_metrics(ticker: str) -> dict[str, Any] | None:
    with _conn() as c:
        row = c.execute(
            "SELECT pe, peg, ev_ebitda, fcf_yield, implied_dcf_growth, last_updated "
            "FROM valuation_metrics WHERE ticker=?", (ticker,)
        ).fetchone()
    if not row:
        return None
    return {
        "ticker": ticker,
        "pe": row[0],
        "peg": row[1],
        "ev_ebitda": row[2],
        "fcf_yield": row[3],
        "implied_dcf_growth": row[4],
        "last_updated": row[5],
    }


def record_company_scores(
    ticker: str,
    event_score: float = 0.0,
    fundamental_score: float = 0.0,
    valuation_score: float = 0.0,
    canslim_score: int = 0,
    multibagger_score: int = 0,
    total_score: float = 0.0,
) -> None:
    now_str = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        if c.is_postgres:
            c.execute(
                """INSERT INTO company_scores 
                (ticker, event_score, fundamental_score, valuation_score, canslim_score, multibagger_score, total_score, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker) DO UPDATE SET
                event_score=EXCLUDED.event_score, fundamental_score=EXCLUDED.fundamental_score,
                valuation_score=EXCLUDED.valuation_score, canslim_score=EXCLUDED.canslim_score,
                multibagger_score=EXCLUDED.multibagger_score, total_score=EXCLUDED.total_score,
                last_updated=EXCLUDED.last_updated""",
                (ticker, event_score, fundamental_score, valuation_score, canslim_score, multibagger_score, total_score, now_str)
            )
        else:
            c.execute(
                """INSERT INTO company_scores 
                (ticker, event_score, fundamental_score, valuation_score, canslim_score, multibagger_score, total_score, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker) DO UPDATE SET
                event_score=excluded.event_score, fundamental_score=excluded.fundamental_score,
                valuation_score=excluded.valuation_score, canslim_score=excluded.canslim_score,
                multibagger_score=excluded.multibagger_score, total_score=excluded.total_score,
                last_updated=excluded.last_updated""",
                (ticker, event_score, fundamental_score, valuation_score, canslim_score, multibagger_score, total_score, now_str)
            )


def read_company_scores(ticker: str) -> dict[str, Any] | None:
    with _conn() as c:
        row = c.execute(
            "SELECT event_score, fundamental_score, valuation_score, canslim_score, multibagger_score, total_score, last_updated, "
            "credibility_score, industry_tailwind_score, promise_count, coverage_score, confidence_score "
            "FROM company_scores WHERE ticker=?", (ticker,)
        ).fetchone()
    if not row:
        return None
    return {
        "ticker": ticker,
        "event_score": row[0],
        "fundamental_score": row[1],
        "valuation_score": row[2],
        "canslim_score": row[3],
        "multibagger_score": row[4],
        "total_score": row[5],
        "last_updated": row[6],
        "credibility_score": row[7] if len(row) > 7 and row[7] is not None else 0.0,
        "industry_tailwind_score": row[8] if len(row) > 8 and row[8] is not None else 0.0,
        "promise_count": row[9] if len(row) > 9 and row[9] is not None else 0,
        "coverage_score": row[10] if len(row) > 10 and row[10] is not None else 0.0,
        "confidence_score": row[11] if len(row) > 11 and row[11] is not None else 0.0,
    }


def record_company_score_snapshot(ticker: str, effective_date: str = None) -> None:
    if not effective_date:
        effective_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    scores = read_company_scores(ticker)
    if not scores:
        return
    with _conn() as c:
        if c.is_postgres:
            c.execute(
                """INSERT INTO company_scores_history 
                (ticker, event_score, fundamental_score, valuation_score, canslim_score, multibagger_score, total_score, credibility_score, industry_tailwind_score, promise_count, coverage_score, confidence_score, effective_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker, effective_date) DO UPDATE SET
                event_score=EXCLUDED.event_score, fundamental_score=EXCLUDED.fundamental_score,
                valuation_score=EXCLUDED.valuation_score, canslim_score=EXCLUDED.canslim_score,
                multibagger_score=EXCLUDED.multibagger_score, total_score=EXCLUDED.total_score,
                credibility_score=EXCLUDED.credibility_score, industry_tailwind_score=EXCLUDED.industry_tailwind_score,
                promise_count=EXCLUDED.promise_count, coverage_score=EXCLUDED.coverage_score, confidence_score=EXCLUDED.confidence_score""",
                (
                    ticker,
                    scores.get("event_score", 0.0),
                    scores.get("fundamental_score", 0.0),
                    scores.get("valuation_score", 0.0),
                    scores.get("canslim_score", 0),
                    scores.get("multibagger_score", 0),
                    scores.get("total_score", 0.0),
                    scores.get("credibility_score", 0.0),
                    scores.get("industry_tailwind_score", 0.0),
                    scores.get("promise_count", 0),
                    scores.get("coverage_score", 0.0),
                    scores.get("confidence_score", 0.0),
                    effective_date
                )
            )
        else:
            c.execute(
                """INSERT INTO company_scores_history 
                (ticker, event_score, fundamental_score, valuation_score, canslim_score, multibagger_score, total_score, credibility_score, industry_tailwind_score, promise_count, coverage_score, confidence_score, effective_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker, effective_date) DO UPDATE SET
                event_score=excluded.event_score, fundamental_score=excluded.fundamental_score,
                valuation_score=excluded.valuation_score, canslim_score=excluded.canslim_score,
                multibagger_score=excluded.multibagger_score, total_score=excluded.total_score,
                credibility_score=excluded.credibility_score, industry_tailwind_score=excluded.industry_tailwind_score,
                promise_count=excluded.promise_count, coverage_score=excluded.coverage_score, confidence_score=excluded.confidence_score""",
                (
                    ticker,
                    scores.get("event_score", 0.0),
                    scores.get("fundamental_score", 0.0),
                    scores.get("valuation_score", 0.0),
                    scores.get("canslim_score", 0),
                    scores.get("multibagger_score", 0),
                    scores.get("total_score", 0.0),
                    scores.get("credibility_score", 0.0),
                    scores.get("industry_tailwind_score", 0.0),
                    scores.get("promise_count", 0),
                    scores.get("coverage_score", 0.0),
                    scores.get("confidence_score", 0.0),
                    effective_date
                )
            )


def update_company_scores(ticker: str) -> None:
    """
    Consolidates event, fundamental, valuation, canslim, and multibagger scores
    into the company_scores table and computes the weighted total_score.
    Uses a single DB connection for all reads and the final write.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    now_str = datetime.now(timezone.utc).isoformat()

    with _conn() as c:
        # 1. Event score from market_events (rolling 7 days)
        event_score = 0.0
        for ev_type, val, direction in c.execute(
            "SELECT event_type, value, direction FROM market_events WHERE ticker=? AND event_date >= ?",
            (ticker, cutoff),
        ).fetchall():
            if ev_type == "PROMOTER_BUY":
                event_score += 5.0 if val >= 2.5e8 else (3.0 if val >= 5e7 else 1.0)
            elif ev_type == "PROMOTER_SELL":
                event_score -= 5.0 if val >= 2.5e8 else 3.0
            elif ev_type == "BULK_DEAL":
                if direction == "BULLISH" and val >= 1e8:
                    event_score += 3.0
                elif direction == "BEARISH" and val >= 1e8:
                    event_score -= 3.0
            elif ev_type == "BLOCK_DEAL":
                if direction == "BULLISH" and val >= 2e8:
                    event_score += 2.0
                elif direction == "BEARISH" and val >= 2e8:
                    event_score -= 2.0

        # 2. Fundamental score
        fundamental_score = 0.0
        fund_row = c.execute(
            "SELECT roce, roe, debt_equity, operating_margin, fcf, sales_cagr_3y, eps_growth_3y, "
            "inst_holding_change, sector, last_updated FROM company_fundamentals WHERE ticker=?",
            (ticker,),
        ).fetchone()
        if fund_row:
            fundamentals = {
                "ticker": ticker, "roce": fund_row[0], "roe": fund_row[1],
                "debt_equity": fund_row[2], "operating_margin": fund_row[3],
                "fcf": fund_row[4], "sales_cagr_3y": fund_row[5],
                "eps_growth_3y": fund_row[6], "inst_holding_change": fund_row[7],
                "sector": fund_row[8], "last_updated": fund_row[9],
            }
            try:
                from sector_specific_metrics import get_sector_score  # local to avoid circular import
                sect_score, _, _ = get_sector_score(ticker, fundamentals)
                fundamental_score = sect_score / 10.0
            except Exception:
                roce = fundamentals.get("roce") or 0.0
                roe = fundamentals.get("roe") or 0.0
                debt_equity = fundamentals.get("debt_equity") or 0.0
                op_margin = fundamentals.get("operating_margin") or 0.0
                if roce >= 15.0:      fundamental_score += 3.0
                if roe >= 15.0:       fundamental_score += 3.0
                if debt_equity <= 0.5: fundamental_score += 2.0
                if op_margin >= 15.0:  fundamental_score += 2.0

        # 3. Valuation score
        valuation_score = 0.0
        val_row = c.execute(
            "SELECT pe, peg, fcf_yield FROM valuation_metrics WHERE ticker=?", (ticker,)
        ).fetchone()
        if val_row:
            pe, peg, fcf_yield = val_row[0] or 0.0, val_row[1], val_row[2] or 0.0
            all_pes  = [r[0] for r in c.execute("SELECT pe  FROM valuation_metrics WHERE pe  > 0").fetchall() if r[0]]
            all_pegs = [r[0] for r in c.execute("SELECT peg FROM valuation_metrics WHERE peg > 0").fetchall() if r[0]]
            pe_points = (sum(1 for p in all_pes if p > pe) / len(all_pes) * 4.0) if all_pes and pe > 0 else 2.0
            peg_points = (sum(1 for p in all_pegs if p > peg) / len(all_pegs) * 4.0) if all_pegs and peg and peg > 0 else 0.0
            fcf_points = 2.0 if fcf_yield >= 5.0 else (1.0 if fcf_yield >= 2.0 else 0.0)
            valuation_score = pe_points + peg_points + fcf_points

        # 4. Existing canslim / multibagger from company_scores
        canslim_score = 0
        multibagger_score = 0
        score_row = c.execute(
            "SELECT canslim_score, multibagger_score FROM company_scores WHERE ticker=?", (ticker,)
        ).fetchone()
        if score_row:
            canslim_score    = int(score_row[0] or 0)
            multibagger_score = int(score_row[1] or 0)

        # 5. Weighted total
        total_score = (
            1.0 * event_score
            + 1.5 * fundamental_score
            + 1.5 * valuation_score
            + 2.0 * float(canslim_score) / 10.0
            + 2.0 * float(multibagger_score) / 10.0
        )

        # 6. Upsert back into company_scores (single write, same connection)
        c.execute(
            """INSERT INTO company_scores
               (ticker, event_score, fundamental_score, valuation_score,
                canslim_score, multibagger_score, total_score, last_updated)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT (ticker) DO UPDATE SET
               event_score=excluded.event_score,
               fundamental_score=excluded.fundamental_score,
               valuation_score=excluded.valuation_score,
               canslim_score=excluded.canslim_score,
               multibagger_score=excluded.multibagger_score,
               total_score=excluded.total_score,
               last_updated=excluded.last_updated""",
            (ticker, event_score, fundamental_score, valuation_score,
             canslim_score, multibagger_score, total_score, now_str),
        )


def pending_consensus() -> list[tuple[int, ConsensusEvent]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id,ticker,direction,scouts,reasons,ts FROM consensus WHERE dispatched=0"
        ).fetchall()
    out: list[tuple[int, ConsensusEvent]] = []
    for r in rows:
        out.append((int(r[0]), ConsensusEvent(
            ticker=r[1], direction=r[2],
            scouts=json.loads(r[3]), reasons=json.loads(r[4]),
            timestamp=datetime.fromisoformat(r[5]),
        )))
    return out


def mark_dispatched(row_id: int) -> None:
    with _conn() as c:
        c.execute("UPDATE consensus SET dispatched=1 WHERE id=?", (row_id,))


def record_raw_data(source: str, payload: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO raw_data_warehouse (source, payload, ts) VALUES (?, ?, ?)",
            (source, payload, datetime.now(timezone.utc).isoformat()),
        )


def record_market_event(
    ticker: str,
    event_type: str,
    value: float,
    direction: str,
    metadata: str,
    event_date: str,
    ts: str | None = None,
) -> None:
    if not ts:
        ts = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            "INSERT INTO market_events (ticker, event_type, value, direction, metadata, event_date, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ticker, event_type, value, direction, metadata, event_date, ts),
        )


def read_market_events(days: int = 7) -> list[dict[str, Any]]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _conn() as c:
        rows = c.execute(
            "SELECT id, ticker, event_type, value, direction, metadata, event_date, ts "
            "FROM market_events WHERE event_date >= ? ORDER BY event_date DESC",
            (cutoff,),
        ).fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r[0],
            "ticker": r[1],
            "event_type": r[2],
            "value": r[3],
            "direction": r[4],
            "metadata": r[5],
            "event_date": r[6],
            "ts": r[7],
        })
    return out


# ── Core scout runner ─────────────────────────────────────────────────────────

def _gemini_with_retry(client, prompt: str, config, max_retries: int = 3):
    """Call client.models.generate_content with exponential back-off on 429 errors."""
    delay = 15
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            if attempt < max_retries - 1 and (
                "429" in str(exc) or "quota" in str(exc).lower() or "limit" in str(exc).lower()
            ):
                sys.stderr.write(f"[gemini] Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries})…\n")
                time.sleep(delay)
                delay *= 2
            else:
                raise


def run_scout(
    scout_name: str,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 2048,
) -> Signal:
    """
    Call Gemini with the scout's prompt. Parse the structured JSON trailer.
    Persist the signal to SQLite.

    Scout prompts MUST end with a strict JSON block:
        {"ticker": "<NSE_SYMBOL>", "direction": "BULLISH|BEARISH|NEUTRAL",
         "confidence": <1-5>, "reason": "<one line>"}
    """
    client = get_gemini()

    # Combine system + user into a single prompt (Gemini client API style)
    full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

    cfg = genai_types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=0.2,
    )
    response = _gemini_with_retry(client, full_prompt, cfg)

    raw = response.text.strip() if response.text else ""

    payload = _extract_last_json(raw)
    if payload is None:
        sig = Signal(
            scout=scout_name, ticker="MACRO", direction=NEUTRAL,
            confidence=1, reason="no qualifying signal this run", raw=raw,
        )
    else:
        sig = Signal(
            scout=scout_name,
            ticker=str(payload.get("ticker", "MACRO")).upper(),
            direction=_normalise_direction(payload.get("direction", NEUTRAL)),
            confidence=int(payload.get("confidence", 1) or 1),
            reason=str(payload.get("reason", "")).strip()[:240],
            raw=raw,
        )

    record_signal(sig)
    log(scout_name, f"signal: {sig.ticker} {sig.direction} conf={sig.confidence} :: {sig.reason}")
    return sig


def _extract_last_json(text: str) -> dict[str, Any] | None:
    depth, start, candidates = 0, -1, []
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start:i + 1])
                start = -1
    for c in reversed(candidates):
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _normalise_direction(d: Any) -> str:
    s = str(d).upper().strip()
    return s if s in DIRECTIONS else NEUTRAL


# ── Delivery ──────────────────────────────────────────────────────────────────

def send_email(subject: str, body: str) -> None:
    user = os.environ.get("GMAIL_USER")
    pw   = os.environ.get("GMAIL_APP_PASSWORD")
    to   = os.environ.get("GMAIL_TO", user)
    if not user or not pw:
        raise RuntimeError("GMAIL_USER / GMAIL_APP_PASSWORD not set in .env")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"]    = user
    msg["To"]      = to
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(user, pw)
        s.send_message(msg)


def send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat  = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    import urllib.request, urllib.parse
    url  = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode(
        {"chat_id": chat, "text": text, "parse_mode": "Markdown"}
    ).encode("utf-8")
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, data=data, method="POST"), timeout=15
        ) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


# ── Logging ───────────────────────────────────────────────────────────────────

def log(scope: str, message: str) -> None:
    _ensure_dirs()
    line = f"{datetime.now(timezone.utc).isoformat()} [{scope}] {message}\n"
    with open(LOGS / f"{scope.lower()}.log", "a", encoding="utf-8") as fh:
        fh.write(line)


# ── Alert formatter ───────────────────────────────────────────────────────────

def render_consensus(ev: ConsensusEvent) -> str:
    head = f"DORAEMI CONSENSUS — {ev.direction} on {ev.ticker}"
    rule = "=" * len(head)
    body = [head, rule, f"Time (UTC): {ev.timestamp.isoformat(timespec='minutes')}", ""]
    body.append(f"{len(ev.scouts)} of 5 scouts agree:")
    for scout, reason in zip(ev.scouts, ev.reasons):
        body.append(f"  · {scout:<12} {reason}")
    body.append("")
    body.append(textwrap.fill(
        "This is informational, not a trade instruction. Gian did not place a trade. "
        "The decision is yours. SEBI regulations apply.",
        width=72,
    ))
    body.append("\nPowered by Google Gemini 3.1 Pro via your Gemini Pro plan.")
    return "\n".join(body)


def cache_prices(ticker: str, prices: dict[str, float]) -> None:
    """Write {date_str: close} into historical_prices. Safe to call multiple times."""
    if not prices:
        return
    rows = [(ticker.upper(), d, float(v)) for d, v in prices.items() if v is not None]
    if not rows:
        return
    with _conn() as c:
        for row in rows:
            c.execute(
                "INSERT OR REPLACE INTO historical_prices (ticker, date, close) VALUES (?,?,?)",
                row,
            )


def fetch_and_cache_prices(tickers: list[str], start: str = "2020-01-01", end: str | None = None) -> int:
    """
    Fetch historical prices for a list of NSE tickers from yfinance and cache
    them in the historical_prices table. Returns the number of rows written.

    Tickers should be plain NSE symbols (e.g. "RELIANCE") — .NS suffix added automatically.
    Call this once from fundamental_collector.py or a standalone script to seed the cache.
    """
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        sys.stderr.write("yfinance / pandas not installed\n")
        return 0

    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    yf_symbols = [f"{t.upper().replace('_', '-')}.NS" for t in tickers]
    # Use the real yfinance download, bypassing the monkey-patch
    real_download = getattr(yf, "_orig_yf_download", yf.download)
    try:
        df = real_download(yf_symbols, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as exc:
        sys.stderr.write(f"[price_cache] yfinance fetch failed: {exc}\n")
        return 0

    if df is None or (hasattr(df, "empty") and df.empty):
        return 0

    written = 0
    try:
        close_df = df["Close"] if hasattr(df.columns, "levels") and "Close" in df.columns.get_level_values(0) else df
        for col in close_df.columns:
            nse_ticker = col.replace(".NS", "").replace("-", "_").upper()
            series = close_df[col].dropna()
            prices = {str(idx.date()): float(val) for idx, val in series.items()}
            cache_prices(nse_ticker, prices)
            written += len(prices)
    except Exception as exc:
        sys.stderr.write(f"[price_cache] cache write failed: {exc}\n")

    return written


def _cache_yf_result(df, cleaned_tickers: list[str], ticker_mapping: dict[str, str]) -> None:
    """Extract Close prices from a yfinance DataFrame and cache them in the DB."""
    try:
        import pandas as pd
        if df is None or (hasattr(df, "empty") and df.empty):
            return
        # yfinance returns multi-level columns when multiple tickers are requested
        close_df = None
        if hasattr(df.columns, "levels"):
            if "Close" in df.columns.get_level_values(0):
                close_df = df["Close"]
        else:
            # single-ticker result
            if "Close" in df.columns:
                close_df = df[["Close"]].rename(columns={"Close": cleaned_tickers[0]})

        if close_df is None:
            return

        for col in close_df.columns:
            nse_ticker = col.replace(".NS", "").replace("-", "_").upper()
            series = close_df[col].dropna()
            prices = {str(idx.date()): float(val) for idx, val in series.items()}
            cache_prices(nse_ticker, prices)
    except Exception:
        pass  # never let caching break the calling code


# ── yfinance monkey patch to use cached DB prices ────────────────────────────────

try:
    import yfinance as yf
    import pandas as pd
    
    _orig_yf_download = yf.download

    def patched_yf_download(tickers, start=None, end=None, *args, **kwargs):
        if isinstance(tickers, str):
            ticker_list = [tickers]
        else:
            ticker_list = list(tickers)
            
        cleaned_tickers = []
        ticker_mapping = {}
        for t in ticker_list:
            cleaned = t.replace(".NS", "").replace("-", "_").upper()
            cleaned_tickers.append(cleaned)
            ticker_mapping[cleaned] = t
            
        conn = sqlite3.connect(DB_PATH)
        start_str = start if start else "2018-01-01"
        end_str = end if end else "2026-06-15"
        
        if not isinstance(start_str, str):
            start_str = start_str.strftime("%Y-%m-%d")
        if not isinstance(end_str, str):
            end_str = end_str.strftime("%Y-%m-%d")
            
        query = """
            SELECT ticker, date, close FROM historical_prices 
            WHERE ticker IN ({}) AND date >= ? AND date <= ?
        """.format(",".join("?" for _ in cleaned_tickers))
        
        params = list(cleaned_tickers) + [start_str, end_str]
        rows = conn.execute(query, params).fetchall()
        conn.close()
        
        if not rows:
            # Cache miss — fetch from yfinance and write results back into the DB
            live_result = _orig_yf_download(tickers, start=start, end=end, *args, **kwargs)
            _cache_yf_result(live_result, cleaned_tickers, ticker_mapping)
            return live_result
            
        data_dict = {}
        for t in cleaned_tickers:
            data_dict[t] = {}
            
        for r_ticker, r_date, r_close in rows:
            data_dict[r_ticker][r_date] = r_close
            
        df = pd.DataFrame(data_dict)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.rename(columns=ticker_mapping, inplace=True)
        df = df.ffill().bfill()
        
        for t in ticker_list:
            if t not in df.columns:
                df[t] = None
        df = df[ticker_list]
        
        class YFDownloadResult(pd.DataFrame):
            _metadata = ['_close_data', '_is_single']
            
            @property
            def _constructor(self):
                return YFDownloadResult
                
            def __getitem__(self, key):
                if key == "Close":
                    if self._is_single:
                        ticker = self._close_data.columns[0]
                        return self._close_data[ticker]
                    else:
                        return self._close_data
                return super().__getitem__(key)
                
        res = YFDownloadResult(df)
        res._close_data = df
        res._is_single = (len(ticker_list) == 1)
        return res

    yf.download = patched_yf_download

except Exception as exc:
    sys.stderr.write(f"[common] yfinance monkey-patch failed: {exc}\n")

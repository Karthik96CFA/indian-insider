#!/usr/bin/env python3
"""
orchestrator.py — Master automation runner for Indian Insider.

Runs pipeline phases in dependency order. Use with cron / launchd / Task Scheduler
(see install/schedule_*.sh).

Phases
------
  tick      Every 30 min — ingest, consensus, dispatch
  morning   07:30 IST weekdays — pre-market scouts (Doraemon, Suneo)
  briefing  08:00 IST weekdays — Telegram morning digest
  portfolio 16:00 IST weekdays — Dekisugi portfolio drift check
  eod       18:00 IST weekdays — Shinchan FII/DII close analysis
  weekly    Mon 09:00 IST — Nobita RBI macro scout
  research  Sun 06:00 IST — full factor refresh (fundamentals → rankings)
  all       Run every phase sequentially (manual smoke test)

Examples
--------
  python orchestrator.py --phase tick
  python orchestrator.py --phase research --continue-on-error
  python orchestrator.py --phase all --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:
    IST = timezone(timedelta(hours=5, minutes=30))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import STATE, initialize_db, log, record_company_score_snapshot, _conn

AGENTS_DIR = Path(__file__).resolve().parent
LOCK_PATH = STATE / "orchestrator.lock"
LOCK_STALE_SEC = 7200  # 2 hours


@dataclass(frozen=True)
class Step:
    script: str
    label: str
    required: bool = True


PHASES: dict[str, list[Step]] = {
    "tick": [
        Step("nse_collector.py", "NSE collector"),
        Step("event_detector.py", "Event detector"),
        Step("doraemi.py", "Doraemi consensus"),
        Step("scoring_engine.py", "Scoring engine"),
        Step("gian.py", "Gian dispatcher"),
    ],
    "morning": [
        Step("doraemon.py", "Doraemon insider scout"),
        Step("suneo.py", "Suneo deal-flow scout"),
    ],
    "briefing": [
        Step("daily_briefing.py", "Morning briefing"),
    ],
    "portfolio": [
        Step("dekisugi.py", "Dekisugi portfolio drift", required=False),
    ],
    "eod": [
        Step("shinchan.py", "Shinchan FII/DII scout"),
    ],
    "weekly": [
        Step("nobita.py", "Nobita RBI macro scout"),
    ],
    "research": [
        Step("fundamental_collector.py", "Fundamental collector"),
        Step("valuation_engine.py", "Valuation engine"),
        Step("canslim_engine.py", "CANSLIM engine"),
        Step("multibagger_engine.py", "Multibagger engine"),
        Step("management_credibility.py", "Management credibility", required=False),
        Step("opportunity_engine.py", "Opportunity rankings"),
    ],
}


def _is_trading_weekday(now_ist: datetime | None = None) -> bool:
    """Mon–Fri in IST. NSE holiday calendar not included."""
    now = now_ist or datetime.now(IST)
    return now.weekday() < 5


def _acquire_lock(phase: str) -> bool:
    STATE.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            raw = LOCK_PATH.read_text(encoding="utf-8").strip().split("\n")
            ts = float(raw[0]) if raw else 0.0
            holder = raw[1] if len(raw) > 1 else "unknown"
            if time.time() - ts < LOCK_STALE_SEC:
                print(f"[orchestrator] Skipping — lock held by {holder} ({phase})")
                log("orchestrator", f"skip {phase}: lock held by {holder}")
                return False
            print(f"[orchestrator] Stale lock from {holder} — taking over")
        except (OSError, ValueError):
            pass
    LOCK_PATH.write_text(f"{time.time()}\n{phase}:{os.getpid()}\n", encoding="utf-8")
    return True


def _release_lock() -> None:
    try:
        LOCK_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def _run_step(
    step: Step,
    *,
    dry_run: bool,
    extra_args: list[str],
    timeout_sec: int,
) -> tuple[bool, float]:
    script_path = AGENTS_DIR / step.script
    if not script_path.exists():
        msg = f"Script not found: {step.script}"
        print(f"[orchestrator] ERROR: {msg}")
        log("orchestrator", f"ERROR {step.label}: {msg}")
        return False, 0.0

    cmd = [sys.executable, str(script_path), *extra_args]
    print(f"[orchestrator] ▶ {step.label} ({step.script})")
    if dry_run:
        print(f"             dry-run: {' '.join(cmd)}")
        return True, 0.0

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(AGENTS_DIR),
            timeout=timeout_sec,
            capture_output=False,
        )
        elapsed = time.monotonic() - t0
        ok = result.returncode == 0
        status = "OK" if ok else f"FAIL rc={result.returncode}"
        print(f"[orchestrator] ◀ {step.label}: {status} ({elapsed:.1f}s)")
        log("orchestrator", f"{step.label}: {status} ({elapsed:.1f}s)")
        return ok, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        print(f"[orchestrator] ◀ {step.label}: TIMEOUT after {timeout_sec}s")
        log("orchestrator", f"{step.label}: TIMEOUT after {timeout_sec}s")
        return False, elapsed
    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(f"[orchestrator] ◀ {step.label}: ERROR {exc}")
        log("orchestrator", f"{step.label}: ERROR {exc}")
        return False, elapsed


def _snapshot_all_scores() -> None:
    """Persist today's company_scores into company_scores_history."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _conn() as c:
        rows = c.execute("SELECT ticker FROM company_scores").fetchall()
    tickers = [r[0] for r in rows]
    if not tickers:
        print("[orchestrator] No tickers to snapshot")
        return
    for ticker in tickers:
        record_company_score_snapshot(ticker, today)
    print(f"[orchestrator] Snapshotted scores for {len(tickers)} tickers ({today})")
    log("orchestrator", f"snapshotted {len(tickers)} tickers for {today}")


def run_phase(
    phase: str,
    *,
    dry_run: bool = False,
    continue_on_error: bool = False,
    skip_trading_day_check: bool = False,
    timeout_sec: int = 3600,
    extra_args: list[str] | None = None,
) -> int:
    if phase == "all":
        order = ["tick", "morning", "briefing", "portfolio", "eod", "weekly", "research"]
        rc = 0
        for p in order:
            r = run_phase(
                p,
                dry_run=dry_run,
                continue_on_error=continue_on_error,
                skip_trading_day_check=skip_trading_day_check,
                timeout_sec=timeout_sec,
                extra_args=extra_args,
            )
            if r != 0:
                rc = r
                if not continue_on_error:
                    return rc
        return rc

    steps = PHASES.get(phase)
    if not steps:
        print(f"[orchestrator] Unknown phase: {phase}")
        print(f"             Valid: {', '.join(['all', *PHASES])}")
        return 2

    weekday_phases = {"morning", "briefing", "portfolio", "eod"}
    if phase in weekday_phases and not skip_trading_day_check and not _is_trading_weekday():
        print(f"[orchestrator] Skipping {phase} — weekend (IST)")
        log("orchestrator", f"skip {phase}: weekend")
        return 0

    if phase == "weekly" and not skip_trading_day_check:
        now_ist = datetime.now(IST)
        if now_ist.weekday() != 0:
            print("[orchestrator] Skipping weekly — not Monday (IST)")
            log("orchestrator", "skip weekly: not Monday")
            return 0

    if not dry_run and not _acquire_lock(phase):
        return 0

    initialize_db()
    print(f"[orchestrator] Phase={phase} steps={len(steps)}")
    log("orchestrator", f"start phase={phase}")

    args = extra_args or []
    failures = 0
    try:
        for step in steps:
            ok, _ = _run_step(step, dry_run=dry_run, extra_args=args, timeout_sec=timeout_sec)
            if not ok:
                failures += 1
                if step.required and not continue_on_error:
                    print(f"[orchestrator] Aborting {phase} — required step failed: {step.label}")
                    return 1

        if phase == "research" and not dry_run:
            _snapshot_all_scores()

    finally:
        if not dry_run:
            _release_lock()

    if failures:
        print(f"[orchestrator] Phase {phase} finished with {failures} failure(s)")
        log("orchestrator", f"phase {phase} done with {failures} failures")
        return 1

    print(f"[orchestrator] Phase {phase} complete ✓")
    log("orchestrator", f"phase {phase} complete")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Indian Insider pipeline orchestrator")
    parser.add_argument(
        "--phase",
        required=True,
        choices=["all", *PHASES.keys()],
        help="Pipeline phase to run",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running after non-fatal step failures",
    )
    parser.add_argument(
        "--skip-trading-day-check",
        action="store_true",
        help="Run weekday/Monday gates even on weekends",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-step timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "extra",
        nargs="*",
        help="Extra CLI args forwarded to each agent script",
    )
    args = parser.parse_args()
    return run_phase(
        args.phase,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
        skip_trading_day_check=args.skip_trading_day_check,
        timeout_sec=args.timeout,
        extra_args=args.extra,
    )


if __name__ == "__main__":
    raise SystemExit(main())

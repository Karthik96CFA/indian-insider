#!/usr/bin/env python3
"""
Dekisugi — Angel One SmartAPI portfolio drift accountant.
No Gemini call — pure local logic using live SmartAPI data.
Schedule: daily 16:00 IST.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BEARISH, BULLISH, NEUTRAL, Signal, log, record_signal

CONFIG = Path.home() / "indian-insider" / "config"
TARGET = CONFIG / "portfolio_target.json"
DRIFT_THRESHOLD_PP = 5.0


def _get_angelone_holdings() -> dict[str, float]:
    api_key     = os.environ.get("ANGELONE_API_KEY")
    client_id   = os.environ.get("ANGELONE_CLIENT_ID")
    mpin        = os.environ.get("ANGELONE_MPIN")
    totp_secret = os.environ.get("ANGELONE_TOTP_SECRET")

    if not all([api_key, client_id, mpin, totp_secret]):
        raise RuntimeError(
            "Angel One credentials missing. Set ANGELONE_API_KEY, "
            "ANGELONE_CLIENT_ID, ANGELONE_MPIN, ANGELONE_TOTP_SECRET in .env"
        )

    try:
        import pyotp, requests
    except ImportError:
        raise RuntimeError("Run: pip install pyotp requests")

    totp = pyotp.TOTP(totp_secret).now()

    session_resp = requests.post(
        "https://apiconnect.angelbroking.com/rest/auth/angelbroking/user/v1/loginByPassword",
        json={"clientcode": client_id, "password": mpin, "totp": totp},
        headers={
            "Content-Type": "application/json", "Accept": "application/json",
            "X-UserType": "USER", "X-SourceID": "WEB",
            "X-ClientLocalIP": "127.0.0.1", "X-ClientPublicIP": "127.0.0.1",
            "X-MACAddress": "00:00:00:00:00:00", "X-PrivateKey": api_key,
        },
        timeout=15,
    )
    session_resp.raise_for_status()
    session_data = session_resp.json()
    if not session_data.get("status"):
        raise RuntimeError(f"Angel One login failed: {session_data.get('message')}")

    jwt_token = session_data["data"]["jwtToken"]

    holdings_resp = requests.get(
        "https://apiconnect.angelbroking.com/rest/secure/angelbroking/portfolio/v1/getHolding",
        headers={
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json", "Accept": "application/json",
            "X-UserType": "USER", "X-SourceID": "WEB",
            "X-ClientLocalIP": "127.0.0.1", "X-ClientPublicIP": "127.0.0.1",
            "X-MACAddress": "00:00:00:00:00:00", "X-PrivateKey": api_key,
        },
        timeout=15,
    )
    holdings_resp.raise_for_status()
    holdings_data = holdings_resp.json()
    if not holdings_data.get("status"):
        raise RuntimeError(f"Holdings fetch failed: {holdings_data.get('message')}")

    result: dict[str, float] = {}
    for h in holdings_data.get("data", []):
        symbol = str(h.get("tradingsymbol", "")).upper()
        if symbol.endswith("-EQ"):
            symbol = symbol[:-3]
        qty = float(h.get("quantity", 0))
        ltp = float(h.get("ltp", 0))
        if symbol and qty > 0 and ltp > 0:
            result[symbol] = result.get(symbol, 0.0) + (qty * ltp)
    return result


def main() -> int:
    if not TARGET.exists():
        sig = Signal(scout="dekisugi", ticker="MACRO", direction=NEUTRAL,
                     confidence=1, reason="portfolio_target.json missing - skipping",
                     raw="config missing")
        record_signal(sig)
        log("dekisugi", sig.reason)
        print(f"[dekisugi] {sig.reason}")
        return 0

    target: dict[str, float] = {
        k: float(v) for k, v in json.loads(TARGET.read_text()).items() if not k.startswith("_")
    }

    try:
        current_value = _get_angelone_holdings()
        log("dekisugi", f"fetched {len(current_value)} holdings from Angel One SmartAPI")
    except Exception as exc:
        fallback = CONFIG / "portfolio_current.json"
        if fallback.exists():
            current_value = {
                k: float(v) for k, v in json.loads(fallback.read_text()).items() if not k.startswith("_")
            }
            log("dekisugi", f"Angel One failed ({exc}); using fallback JSON")
        else:
            sig = Signal(scout="dekisugi", ticker="MACRO", direction=NEUTRAL,
                         confidence=1, reason=f"Angel One fetch failed: {exc}", raw=str(exc))
            record_signal(sig)
            log("dekisugi", sig.reason)
            print(f"[dekisugi] {sig.reason}")
            return 1

    total = sum(current_value.values()) or 1.0
    current_pct = {k: 100.0 * v / total for k, v in current_value.items()}

    drifts = []
    for ticker, target_pct in target.items():
        cur   = current_pct.get(ticker, 0.0)
        drift = cur - target_pct
        if abs(drift) >= DRIFT_THRESHOLD_PP:
            drifts.append((ticker, target_pct, cur, drift))

    if not drifts:
        sig = Signal(scout="dekisugi", ticker="MACRO", direction=NEUTRAL,
                     confidence=1, reason="portfolio within tolerance - no rebalance needed",
                     raw=json.dumps({"current_pct": current_pct}))
    else:
        drifts.sort(key=lambda d: abs(d[3]), reverse=True)
        ticker, tgt, cur, drift = drifts[0]
        direction = BEARISH if drift > 0 else BULLISH
        sig = Signal(
            scout="dekisugi", ticker=ticker.upper(), direction=direction,
            confidence=min(5, 1 + int(abs(drift) // 5)),
            reason=(f"{ticker} drifted {drift:+.1f}pp "
                    f"(target {tgt:.1f}% -> current {cur:.1f}%) - "
                    f"{'trim' if drift > 0 else 'add'}"),
            raw=json.dumps({"drifts": [list(d) for d in drifts], "current_pct": current_pct}),
        )

    record_signal(sig)
    log("dekisugi", f"signal: {sig.ticker} {sig.direction} :: {sig.reason}")
    print(f"[dekisugi] {sig.ticker} {sig.direction} conf={sig.confidence}")
    print(f"           {sig.reason}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

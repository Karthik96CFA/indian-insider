#!/usr/bin/env python3
"""
Nobita — RBI speech and MPC minutes reader.
Fetches RBI RSS feeds directly via requests + stdlib xml.etree, extracts
recent publications (last 7 days), then hands the text to Gemini for
hawkish/dovish classification and NIFTY direction signal.

Data flow:
  RBI RSS feeds (public, no auth) → Python requests → Gemini analysis → Signal

Schedule: weekly Monday 09:00 IST.
"""
from __future__ import annotations
import datetime
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

try:
    import requests
except ImportError:
    sys.stderr.write("Run: pip install requests\n")
    raise

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import log, run_scout

# RBI publishes Atom/RSS feeds for press releases and speeches
_RBI_FEEDS = [
    ("Press Releases", "https://www.rbi.org.in/Scripts/RSSFeed.aspx?Id=PRRelease"),
    ("Speeches",       "https://www.rbi.org.in/Scripts/RSSFeed.aspx?Id=speeches"),
    ("Publications",   "https://www.rbi.org.in/Scripts/RSSFeed.aspx?Id=publications"),
]
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Indian-Insider/1.0; +macro-monitor)",
    "Accept": "application/rss+xml, application/xml, text/xml",
}
_TIMEOUT = 15  # seconds per feed

SYSTEM = """You are Nobita, an Indian macro intelligence analyst powered by Gemini.
You receive the titles and descriptions of RBI publications from the last 7 days,
fetched directly from RBI's public RSS feeds.

Your job:
  1. For each item: identify the type (speech / MPC minute / policy statement / circular).
  2. Classify each as:
       HAWKISH  — rate hike bias, inflation concern, liquidity tightening
       DOVISH   — rate cut bias, growth concern, liquidity support
       NEUTRAL  — data-dependent, no clear bias
  3. Count the tilt: if MPC minutes, note the vote split if mentioned.
  4. Aggregate the net stance across all items.

Direction rules for NIFTY:
  - Net DOVISH  → BULLISH (cuts/liquidity → equity rally, especially rate-sensitives)
  - Net HAWKISH → BEARISH (hike/pause → equity pressure, banking sector)
  - Mixed       → NEUTRAL

Output one prose paragraph citing specific items, then STRICT JSON:
  {"ticker": "NIFTY", "direction": "BULLISH|BEARISH|NEUTRAL",
   "confidence": <1-5>, "reason": "<one-line with policy context>"}

Confidence:
  1 = single routine circular, no policy signal
  3 = 2–3 items aligned, clear tilt
  5 = MPC minutes + Governor speech aligned, explicit vote majority

No relevant content? Output:
  {"ticker": "NIFTY", "direction": "NEUTRAL", "confidence": 1,
   "reason": "no RBI monetary policy content in the last 7 days"}

Do not speculate beyond what the titles/descriptions say.
"""


class RBIItem(NamedTuple):
    feed_name: str
    pub_date: str
    title: str
    description: str


def _fetch_feed(name: str, url: str, cutoff: datetime.date) -> list[RBIItem]:
    """Fetch one RSS feed and return items newer than cutoff."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        log("nobita", f"Failed to fetch {name} feed: {exc}")
        return []

    items: list[RBIItem] = []
    try:
        root = ET.fromstring(resp.content)
        # Handle both RSS <item> and Atom <entry> formats
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall(".//item") or root.findall(".//atom:entry", ns)
        for entry in entries:
            title_el = entry.find("title") or entry.find("atom:title", ns)
            desc_el = (
                entry.find("description")
                or entry.find("summary")
                or entry.find("atom:summary", ns)
            )
            date_el = (
                entry.find("pubDate")
                or entry.find("published")
                or entry.find("atom:published", ns)
                or entry.find("dc:date", {"dc": "http://purl.org/dc/elements/1.1/"})
            )

            title = (title_el.text or "").strip() if title_el is not None else ""
            desc = (desc_el.text or "").strip() if desc_el is not None else ""
            pub_date_str = (date_el.text or "").strip() if date_el is not None else ""

            # Try to parse date — RFC 822 format typical in RSS
            item_date: datetime.date | None = None
            for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
                        "%Y-%m-%dT%H:%M:%S%z", "%d %b %Y"):
                try:
                    item_date = datetime.datetime.strptime(pub_date_str[:len(fmt)], fmt).date()
                    break
                except ValueError:
                    continue

            if item_date is None or item_date >= cutoff:
                items.append(RBIItem(
                    feed_name=name,
                    pub_date=pub_date_str[:16] or "unknown",
                    title=title,
                    description=desc[:300],
                ))
    except ET.ParseError as exc:
        log("nobita", f"XML parse error for {name}: {exc}")

    return items


def _fetch_rbi_content(days: int = 7) -> str:
    """Fetch RBI feeds and format items as text for the model prompt."""
    cutoff = datetime.date.today() - datetime.timedelta(days=days)
    all_items: list[RBIItem] = []
    for name, url in _RBI_FEEDS:
        all_items.extend(_fetch_feed(name, url, cutoff))

    if not all_items:
        return (
            f"No RBI publications found in the last {days} days "
            f"(feeds checked: {', '.join(n for n, _ in _RBI_FEEDS)}). "
            "This may be a holiday week or a network issue."
        )

    lines = [f"RBI publications from last {days} days ({len(all_items)} items):"]
    lines.append("=" * 70)
    for item in all_items:
        lines.append(f"[{item.feed_name}] {item.pub_date}")
        lines.append(f"TITLE: {item.title}")
        if item.description:
            lines.append(f"DESC:  {item.description}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    rbi_content = _fetch_rbi_content(days=7)
    user_prompt = f"""{rbi_content}

Classify each item (hawkish/dovish/neutral). Aggregate tilt. Apply direction rules.
Output prose analysis then JSON signal block."""

    sig = run_scout("nobita", SYSTEM, user_prompt)
    print(f"[nobita] {sig.ticker} {sig.direction} conf={sig.confidence}")
    print(f"         {sig.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

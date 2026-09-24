"""
claudesports_shadow.py

Historical backtest of the "shadow bid" strategy from the 2026-09-22 Cowork handoff
(claude_code_handoff.md): claudesports (0xc2f2d0...) is a pure maker bot on Polymarket
that dumps liquidity at prices a median 16c below the market. Copying it as a taker is
a big loser, but resting bids placed just above its fill price captured a real edge in
the hour after each fill (+54% in the research sample).

This script:
  1. Pulls every historical claudesports trade via the Data API's /activity endpoint
     (data-api.polymarket.com). This was the handoff's primary suggestion, and avoids
     needing an archive-node RPC: a first attempt at reading OrderFilled events
     directly off Polygon found that free public RPCs are pruned and cannot serve logs
     back to May 4, 2026 ("History has been pruned for this block").
  2. For each fill, looks up the market's resolution via Gamma (cached per event) to
     compute realized ROI once the market has closed.
  3. Replays public trade prints for the 60 minutes after each fill to decide whether a
     simulated resting bid at (fill_price + offset) would plausibly have been hit.
  4. Logs everything to SQLite and prints a fill-rate / ROI summary, plus per-fill
     dollar-size stats (this strategy trades in ~$9-20 increments per the handoff, so
     capital requirements are small — the constraint is opportunity count, not size).

Paper only. No wallet, no signing, no order placement — this only reads public API
data. See the handoff's "Open threads" note on Polymarket's legal status before ever
wiring this to real money.

Note on the on-chain path: it isn't dead, just deferred. Once this backtest validates
the fill rate, a live/real-time version can watch OrderFilled events (verified against
github.com/Polymarket/ctf-exchange-v2's Events.sol) directly off recent blocks, which
free public RPCs handle fine since only pruned *historical* ranges are the problem.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import requests

# Concurrent requests in flight during the fetch phase. Kept moderate to avoid
# tripping rate limits on the free public APIs.
CONCURRENCY = 20

CLAUDESPORTS_ADDRESS = "0xc2f2d01b227948e59f25aeb3d49564b8975d0ff7"
START_DATE = "2026-05-04"  # claudesports's first trade per the handoff

DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"

DB_PATH = Path(__file__).parent / "claudesports_shadow.db"

# ── Shadow-bid simulation parameters ─────────────────────────────────────
BID_OFFSET_CENTS = 2           # rest at claudesports's fill price + this many cents
HOLD_MINUTES = 60              # per the handoff's "60 minutes" window
FEE_RATE = 0.05                 # Polymarket sports taker fee = 0.05 * p * (1-p) per share

# There is no historical order-book depth API, so true queue position at a price level
# is not observable after the fact. As a conservative proxy: treat the bid as filled
# once *some* volume beyond a small buffer has traded at-or-through our price, rather
# than on the very first print at that level (which we assume was ahead of us in queue).
QUEUE_AHEAD_ASSUMPTION_SHARES = 50.0

# Rough keyword buckets purely for the summary report (claudesports trades esports,
# crypto price markets, and traditional sports — not just "sports" moneylines).
CATEGORY_KEYWORDS = {
    "mlb": ["mlb", "baseball"],
    "nba": ["nba", "basketball"],
    "nhl": ["nhl", "hockey"],
    "nfl": ["nfl", "football"],
    "soccer": ["soccer", "premier league", "la liga", "champions league"],
    "tennis": ["tennis", "atp", "wta"],
    "esports": ["cs2", "counter-strike", "valorant", "league of legends", "dota"],
    "crypto_price": ["bitcoin", "ethereum", "btc", "eth "],
}


def categorize(title: str, slug: str) -> str:
    text = f"{title} {slug}".lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(k in text for k in keywords):
            return category
    return "other"


@dataclass
class ClaudesportsFill:
    condition_id: str
    asset: str
    outcome: str
    outcome_index: int
    price: float
    size: float
    usdc_size: float
    timestamp: int
    tx_hash: str
    title: str
    slug: str
    event_slug: str


def _get(url: str, params: dict, retries: int = 3) -> requests.Response | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code == 200:
                return resp
        except requests.RequestException:
            pass
        time.sleep(0.5 * (attempt + 1))
    return None


def fetch_claudesports_trades(start_ts: int) -> list[ClaudesportsFill]:
    """Page backward through claudesports's trade history via /activity until we pass
    start_ts. Each record already comes with market/outcome attribution attached."""
    fills: list[ClaudesportsFill] = []
    end_cursor = int(datetime.now(timezone.utc).timestamp())

    while True:
        resp = _get(
            f"{DATA_API}/activity",
            {
                "user": CLAUDESPORTS_ADDRESS,
                "type": "TRADE",
                "limit": 500,
                "end": end_cursor,
            },
        )
        if resp is None:
            print("  /activity request failed after retries, stopping pagination.")
            break
        batch = resp.json()
        if not batch:
            break

        for t in batch:
            ts = t["timestamp"]
            if ts < start_ts:
                continue
            fills.append(
                ClaudesportsFill(
                    condition_id=t["conditionId"],
                    asset=t["asset"],
                    outcome=t.get("outcome", ""),
                    outcome_index=t.get("outcomeIndex", 0),
                    price=float(t["price"]),
                    size=float(t["size"]),
                    usdc_size=float(t.get("usdcSize", 0)),
                    timestamp=ts,
                    tx_hash=t.get("transactionHash", ""),
                    title=t.get("title", ""),
                    slug=t.get("slug", ""),
                    event_slug=t.get("eventSlug", ""),
                )
            )

        oldest = min(t["timestamp"] for t in batch)
        print(f"  fetched {len(batch)} activity rows back to "
              f"{datetime.fromtimestamp(oldest, tz=timezone.utc).date()} "
              f"({len(fills)} fills >= start date so far)")
        if oldest < start_ts or len(batch) < 500:
            break
        end_cursor = oldest - 1
        time.sleep(0.1)

    return fills


def get_resolution_from_events(events: list[dict] | None, condition_id: str, outcome_index: int) -> float | None:
    """Return the resolved price for the specific outcome claudesports traded, or
    None if the market hasn't closed yet (or the event lookup failed)."""
    if not events:
        return None
    for ev in events:
        for m in ev.get("markets", []):
            if m.get("conditionId") != condition_id:
                continue
            if not m.get("closed"):
                return None
            raw = m.get("outcomePrices")
            if not raw:
                return None
            try:
                prices = json.loads(raw) if isinstance(raw, str) else raw
                return float(prices[outcome_index])
            except (ValueError, IndexError, TypeError):
                return None
    return None


# ── Async fetch phase ──────────────────────────────────────────────────────
# Two big speedups over a naive one-request-per-fill approach:
#   1. Fills into the same (condition_id, asset) that need overlapping trade
#      windows are merged into a single fetch (claudesports often fires several
#      small fills in a row into the same market — median fill is $1.78).
#   2. All remaining distinct fetches (merged trade windows + one per distinct
#      event, for resolutions) run concurrently instead of sequentially.

async def _get_async(session: aiohttp.ClientSession, url: str, params: dict, retries: int = 3):
    for attempt in range(retries):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status == 200:
                    return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
        await asyncio.sleep(0.4 * (attempt + 1))
    return None


def merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Standard interval merge. Keeps back-to-back rapid-fire fills into the same
    market as one fetch, while fills that are far apart in time (e.g. the same
    long-lived market traded on two different days) stay as separate, bounded
    fetches instead of pulling everything in between."""
    merged = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


async def fetch_trades_merged(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                               condition_id: str, asset: str, start_ts: int, end_ts: int) -> list[dict]:
    trades: list[dict] = []
    cursor_end = end_ts
    async with sem:
        while True:
            batch = await _get_async(
                session, f"{DATA_API}/trades",
                {"market": condition_id, "limit": 500, "end": cursor_end, "takerOnly": "false"},
            )
            if not batch:
                break
            in_window = [
                t for t in batch
                if t.get("asset") == asset and start_ts <= t.get("timestamp", 0) <= end_ts
            ]
            trades.extend(in_window)
            oldest = min(t.get("timestamp", cursor_end) for t in batch)
            if oldest <= start_ts or len(batch) < 500:
                break
            cursor_end = oldest - 1
    return trades


async def fetch_event(session: aiohttp.ClientSession, sem: asyncio.Semaphore, event_slug: str):
    async with sem:
        data = await _get_async(session, f"{GAMMA_API}/events", {"slug": event_slug})
    return event_slug, data


async def fetch_all(fills: list[ClaudesportsFill]) -> tuple[dict, dict]:
    """Returns (trades_by_group, events_by_slug) where trades_by_group maps
    (condition_id, asset) -> list of (merged_start, merged_end, trades)."""
    windows_by_group: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for f in fills:
        windows_by_group[(f.condition_id, f.asset)].append((f.timestamp, f.timestamp + HOLD_MINUTES * 60))

    merged_by_group = {key: merge_windows(w) for key, w in windows_by_group.items()}
    total_trade_fetches = sum(len(v) for v in merged_by_group.values())
    event_slugs = sorted({f.event_slug for f in fills if f.event_slug})

    print(f"  {len(windows_by_group)} distinct markets -> {total_trade_fetches} trade-window fetches "
          f"(merged from {len(fills)} fills), {len(event_slugs)} distinct events to resolve, "
          f"concurrency={CONCURRENCY}")

    sem = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        trade_tasks = {}
        for (cid, asset), windows in merged_by_group.items():
            for (s, e) in windows:
                trade_tasks[(cid, asset, s, e)] = asyncio.ensure_future(
                    fetch_trades_merged(session, sem, cid, asset, s, e)
                )
        event_tasks = {slug: asyncio.ensure_future(fetch_event(session, sem, slug)) for slug in event_slugs}

        done = 0
        total = len(trade_tasks) + len(event_tasks)
        trades_by_group: dict[tuple[str, str], list[tuple[int, int, list]]] = defaultdict(list)
        for key, task in trade_tasks.items():
            cid, asset, s, e = key
            trades = await task
            trades_by_group[(cid, asset)].append((s, e, trades))
            done += 1
            if done % 200 == 0:
                print(f"  fetched {done}/{total}")

        events_by_slug = {}
        for slug, task in event_tasks.items():
            _, data = await task
            events_by_slug[slug] = data
            done += 1
            if done % 200 == 0:
                print(f"  fetched {done}/{total}")

    return trades_by_group, events_by_slug


def simulate_shadow_fill(fill: ClaudesportsFill, trades_by_group: dict, events_by_slug: dict) -> dict:
    """Decide whether a resting bid at (claudesports_price + offset) would plausibly
    have filled within HOLD_MINUTES. See QUEUE_AHEAD_ASSUMPTION_SHARES for the
    queue-position caveat (no historical order-book depth is available)."""
    bid_price = min(fill.price + BID_OFFSET_CENTS / 100, 0.99)
    start_ts = fill.timestamp
    end_ts = start_ts + HOLD_MINUTES * 60

    trades: list[dict] = []
    for (s, e, group_trades) in trades_by_group.get((fill.condition_id, fill.asset), []):
        if s <= start_ts and end_ts <= e:
            trades = [t for t in group_trades if start_ts <= t.get("timestamp", 0) <= end_ts]
            break

    trades.sort(key=lambda t: t.get("timestamp", 0))
    cumulative_at_or_better = 0.0
    filled = False
    fill_ts = None
    for t in trades:
        t_price = float(t.get("price", 0))
        if t_price <= bid_price:
            cumulative_at_or_better += float(t.get("size", 0))
            if cumulative_at_or_better >= QUEUE_AHEAD_ASSUMPTION_SHARES:
                filled = True
                fill_ts = t.get("timestamp")
                break

    resolved_price = get_resolution_from_events(
        events_by_slug.get(fill.event_slug), fill.condition_id, fill.outcome_index
    )

    roi = None
    if filled and resolved_price is not None:
        fee = FEE_RATE * bid_price * (1 - bid_price)
        cost = bid_price + fee
        roi = (resolved_price - cost) / cost if cost else None

    return {"filled": filled, "bid_price": bid_price, "fill_ts": fill_ts,
            "resolved_price": resolved_price, "roi": roi}


# ── SQLite ────────────────────────────────────────────────────────────────

def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shadow_fills (
            tx_hash TEXT,
            condition_id TEXT,
            asset TEXT,
            category TEXT,
            title TEXT,
            outcome TEXT,
            timestamp INTEGER,
            cs_price REAL,
            cs_usdc_size REAL,
            our_bid_price REAL,
            filled INTEGER,
            fill_ts INTEGER,
            resolved_price REAL,
            roi REAL,
            PRIMARY KEY (tx_hash, asset)
        )
        """
    )
    conn.commit()


def save_result(conn: sqlite3.Connection, fill: ClaudesportsFill, sim: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO shadow_fills VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            fill.tx_hash, fill.condition_id, fill.asset,
            categorize(fill.title, fill.slug), fill.title, fill.outcome, fill.timestamp,
            fill.price, fill.usdc_size,
            sim["bid_price"], int(sim["filled"]), sim["fill_ts"],
            sim["resolved_price"], sim["roi"],
        ),
    )
    conn.commit()


def print_summary(conn: sqlite3.Connection) -> None:
    total, filled = conn.execute("SELECT COUNT(*), SUM(filled) FROM shadow_fills").fetchone()
    if not total:
        print("No fills.")
        return
    filled = filled or 0
    print(f"\n=== Overall ===")
    print(f"claudesports fills observed: {total}   "
          f"Simulated shadow bids filled: {filled}   Fill rate: {filled/total:.1%}")

    avg_size, med_size = conn.execute(
        "SELECT AVG(cs_usdc_size), "
        "(SELECT cs_usdc_size FROM shadow_fills ORDER BY cs_usdc_size "
        " LIMIT 1 OFFSET (SELECT COUNT(*) FROM shadow_fills)/2) FROM shadow_fills"
    ).fetchone()
    print(f"claudesports fill size: avg ${avg_size:.2f}, median ${med_size:.2f} "
          f"-> this is a small-capital strategy; the constraint is opportunity count.")

    print("\n=== By category ===")
    for category, n, f, avg_roi in conn.execute(
        "SELECT category, COUNT(*), SUM(filled), AVG(CASE WHEN filled=1 THEN roi END) "
        "FROM shadow_fills GROUP BY category ORDER BY 2 DESC"
    ):
        f = f or 0
        roi_str = f"{avg_roi:.1%}" if avg_roi is not None else "n/a"
        print(f"  {category:14s} n={n:5d}  filled={f:5d} ({f/n:.1%})  "
              f"avg ROI on fills={roi_str}")

    print("\n=== By price band (our bid price) ===")
    for band, n, f, avg_roi in conn.execute(
        "SELECT CAST(our_bid_price*10 AS INT), COUNT(*), SUM(filled), "
        "AVG(CASE WHEN filled=1 THEN roi END) FROM shadow_fills GROUP BY 1 ORDER BY 1"
    ):
        f = f or 0
        roi_str = f"{avg_roi:.1%}" if avg_roi is not None else "n/a"
        print(f"  {band/10:.1f}-{(band+1)/10:.1f}  n={n:5d}  filled={f:5d} "
              f"({f/n:.1%})  avg ROI on fills={roi_str}")


async def async_main():
    start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    print(f"Fetching claudesports trade history since {START_DATE}...")
    fills = fetch_claudesports_trades(start_ts)
    print(f"Found {len(fills)} claudesports fills.")

    print("Fetching trade-window replays and market resolutions concurrently...")
    t0 = time.time()
    trades_by_group, events_by_slug = await fetch_all(fills)
    print(f"  fetch phase done in {time.time() - t0:.1f}s")

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    for i, fill in enumerate(fills):
        sim = simulate_shadow_fill(fill, trades_by_group, events_by_slug)
        save_result(conn, fill, sim)
        if (i + 1) % 1000 == 0:
            print(f"  simulated {i + 1}/{len(fills)} fills")

    print_summary(conn)
    conn.close()
    print(f"\nFull results saved to {DB_PATH}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

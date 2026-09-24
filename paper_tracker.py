"""
paper_tracker.py

Forward paper-tracking of the tennis Elo fair-value model. Records what the model
and the market each said BEFORE a match starts, then later records who actually
won, so calibration and Brier score can be measured without lookahead bias.

Three one-shot commands (run by hand, or wrap in a scheduler later):
    python paper_tracker.py cycle    # snapshot open pre-match tennis moneyline markets
    python paper_tracker.py resolve  # fill in outcomes for markets that have settled
    python paper_tracker.py report   # model vs market Brier score + calibration

Design rules, each one from a bug we already hit:
  - Market selection uses Polymarket's own sportsMarketType == "moneyline", not a
    keyword blacklist (set/handicap/O-U/"Completed Match" markets all leaked through
    the blacklist at some point).
  - Only markets whose gameStartTime is still in the future are recorded. An in-play
    price against a pre-match Elo estimate would look like a huge fake edge.
  - Every covered market is logged every cycle, not just the ones the engine would
    quote. Logging only "edges" would bake selection bias into the calibration.
  - Players are matched to Elo by exact normalized name only. The fuzzy fallback in
    tennis_fair_value.py could pair the wrong player; unmatched names are counted so
    coverage stays visible instead of being papered over.

Paper only. Reads public data, places nothing. There is no odds-benchmark column
yet; it can be added without touching the recorded history.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

import tennis_fair_value as tfv

DB_PATH = Path(__file__).parent / "paper_tracker.db"
GAMMA = "https://gamma-api.polymarket.com"

SWEET_LOW, SWEET_HIGH = 0.10, 0.70   # same band as maker_engine.py
MIN_EDGE = 0.05                      # same threshold as maker_engine.py
MIN_RESOLVED_FOR_VERDICT = 200       # below this, print numbers but refuse to conclude


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshots (
            ts INTEGER, condition_id TEXT, event_slug TEXT, question TEXT,
            game_start INTEGER, player_a TEXT, player_b TEXT, surface TEXT,
            elo_a REAL, elo_b REAL,
            p_model_a REAL, p_market_a REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS resolutions (
            condition_id TEXT PRIMARY KEY, resolved_a REAL, resolved_ts INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS coverage (
            ts INTEGER, moneyline_open INTEGER, doubles_skipped INTEGER,
            covered INTEGER, unmatched_players TEXT
        )
        """
    )
    conn.commit()
    return conn


def _parse_start(raw: str | None) -> int | None:
    if not raw:
        return None
    try:
        return int(datetime.fromisoformat(raw.replace("Z", "+00:00").replace(" ", "T")).timestamp())
    except ValueError:
        return None


def _fetch_tennis_events(max_pages: int = 10) -> list[dict]:
    events: list[dict] = []
    for page in range(max_pages):
        resp = requests.get(
            f"{GAMMA}/events",
            params={"tag_slug": "tennis", "active": "true", "closed": "false",
                    "limit": 100, "offset": page * 100},
            timeout=20,
        )
        batch = resp.json()
        if not batch:
            break
        events.extend(batch)
        time.sleep(0.15)
    return events


def cmd_cycle() -> None:
    conn = _conn()
    now = int(time.time())
    elo = tfv.load_elo_ratings()

    moneyline_open = doubles = covered = 0
    unmatched: list[str] = []

    for ev in _fetch_tennis_events():
        for m in ev.get("markets", []):
            # `closed=false` on the query is not reliable, so re-check per market.
            if m.get("sportsMarketType") != "moneyline":
                continue
            if m.get("closed") or not m.get("acceptingOrders"):
                continue
            start = _parse_start(m.get("gameStartTime"))
            if start is None or start <= now:
                continue  # unknown start, or already underway: never record
            moneyline_open += 1

            try:
                outcomes = json.loads(m["outcomes"])
                prices = [float(p) for p in json.loads(m["outcomePrices"])]
            except (KeyError, ValueError, TypeError):
                continue
            if len(outcomes) != 2 or len(prices) != 2:
                continue
            a_name, b_name = outcomes
            if "/" in a_name or "/" in b_name:
                doubles += 1
                continue
            if not (0 < prices[0] < 1):
                continue

            a = elo.get(tfv._normalize_name(a_name))
            b = elo.get(tfv._normalize_name(b_name))
            if a is None or b is None:
                unmatched.extend(n for n, rec in ((a_name, a), (b_name, b)) if rec is None)
                continue

            surface = tfv.surface_from_title(m.get("question", "") + " " + ev.get("title", ""))
            elo_a = a.get(surface) if a.get(surface) == a.get(surface) else a["elo"]
            elo_b = b.get(surface) if b.get(surface) == b.get(surface) else b["elo"]
            p_model_a = 1 / (1 + 10 ** (-(elo_a - elo_b) / 400))

            conn.execute(
                "INSERT INTO snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (now, m["conditionId"], ev.get("slug"), m.get("question"), start,
                 a_name, b_name, surface, elo_a, elo_b, p_model_a, prices[0]),
            )
            covered += 1

    conn.execute(
        "INSERT INTO coverage VALUES (?,?,?,?,?)",
        (now, moneyline_open, doubles, covered, json.dumps(sorted(set(unmatched)))),
    )
    conn.commit()
    print(f"{moneyline_open} open pre-match moneyline markets; {doubles} doubles skipped; "
          f"{covered} covered by Elo ({covered / max(moneyline_open - doubles, 1):.0%} of singles).")
    print(f"{len(set(unmatched))} distinct player names had no exact Elo match "
          f"(stored in coverage table for review).")


def cmd_resolve() -> None:
    conn = _conn()
    pending = conn.execute(
        """
        SELECT DISTINCT s.condition_id, s.event_slug FROM snapshots s
        LEFT JOIN resolutions r ON r.condition_id = s.condition_id
        WHERE r.condition_id IS NULL AND s.game_start < ?
        """,
        (int(time.time()),),
    ).fetchall()
    print(f"{len(pending)} markets past their start time and not yet resolved.")

    newly = void = 0
    for condition_id, slug in pending:
        try:
            events = requests.get(f"{GAMMA}/events", params={"slug": slug}, timeout=20).json()
        except requests.RequestException:
            continue
        for ev in events:
            for m in ev.get("markets", []):
                if m.get("conditionId") != condition_id or not m.get("closed"):
                    continue
                try:
                    prices = [float(p) for p in json.loads(m["outcomePrices"])]
                except (KeyError, ValueError, TypeError):
                    continue
                if prices in ([1.0, 0.0], [0.0, 1.0]):
                    conn.execute("INSERT OR REPLACE INTO resolutions VALUES (?,?,?)",
                                 (condition_id, prices[0], int(time.time())))
                    newly += 1
                else:
                    void += 1  # 50-50 / cancelled / unexpected: excluded, not guessed
        time.sleep(0.1)
    conn.commit()
    print(f"Recorded {newly} outcomes; {void} closed with a non-binary result (excluded).")


def _brier(pairs: list[tuple[float, float]]) -> float:
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def cmd_report() -> None:
    conn = _conn()
    rows = conn.execute(
        """
        SELECT s.condition_id, s.ts, s.game_start, s.p_model_a, s.p_market_a, r.resolved_a
        FROM snapshots s JOIN resolutions r ON r.condition_id = s.condition_id
        ORDER BY s.condition_id, s.ts
        """
    ).fetchall()
    by_market: dict[str, list[tuple]] = {}
    for row in rows:
        by_market.setdefault(row[0], []).append(row)

    n_snap_markets = conn.execute("SELECT COUNT(DISTINCT condition_id) FROM snapshots").fetchone()[0]
    print(f"Markets observed: {n_snap_markets}   with recorded outcome: {len(by_market)}")
    if not by_market:
        print("Nothing resolved yet. Run `cycle` over several days, then `resolve`.")
        return

    # Two views per market: the first price we ever saw, and the last one before start.
    views = {"first snapshot": [r[0] for r in by_market.values()],
             "last pre-start snapshot": [r[-1] for r in by_market.values()]}
    for label, snaps in views.items():
        model = [(s[3], s[5]) for s in snaps]
        market = [(s[4], s[5]) for s in snaps]
        print(f"\n[{label}]  n={len(snaps)}")
        print(f"  Brier  model={_brier(model):.4f}   market={_brier(market):.4f}   "
              f"(lower is better; model - market = {_brier(model) - _brier(market):+.4f})")

    snaps = views["last pre-start snapshot"]
    print("\nModel calibration (last pre-start snapshot): model prob band -> actual win rate")
    bands: dict[int, list[float]] = {}
    for s in snaps:
        bands.setdefault(min(int(s[3] * 10), 9), []).append(s[5])
    for band in sorted(bands):
        ys = bands[band]
        print(f"  {band / 10:.1f}-{(band + 1) / 10:.1f}  n={len(ys):4d}  "
              f"model~{(band + 0.5) / 10:.0%}  actual={sum(ys) / len(ys):.1%}")

    # Hypothetical only: assumes a fill at the snapshot price, which a real resting
    # order would not get. It bounds the signal, it does not estimate real P&L.
    hyp = []
    for s in snaps:
        p_model, p_mkt, y = s[3], s[4], s[5]
        for side_model, side_mkt, side_y in ((p_model, p_mkt, y), (1 - p_model, 1 - p_mkt, 1 - y)):
            if SWEET_LOW <= side_mkt <= SWEET_HIGH and side_model - side_mkt >= MIN_EDGE:
                hyp.append((side_mkt, side_y))
    if hyp:
        cost = sum(p for p, _ in hyp)
        payout = sum(y for _, y in hyp)
        print(f"\nHypothetical (assumes fill at snapshot price, 0.10-0.70 band, edge>=5%): "
              f"n={len(hyp)}  ROI={(payout - cost) / cost:+.1%}")

    if len(snaps) < MIN_RESOLVED_FOR_VERDICT:
        print(f"\nOnly {len(snaps)} resolved markets (< {MIN_RESOLVED_FOR_VERDICT}). "
              f"These numbers are noise-level; do not draw a conclusion yet.")


if __name__ == "__main__":
    commands = {"cycle": cmd_cycle, "resolve": cmd_resolve, "report": cmd_report}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit(f"usage: python paper_tracker.py [{'|'.join(commands)}]")
    commands[sys.argv[1]]()

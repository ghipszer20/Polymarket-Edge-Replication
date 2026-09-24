"""
tennis_fair_value.py

Real fair-value model for tennis moneyline markets, using Tennis Abstract's public
ATP/WTA Elo ratings (tennisabstract.com/reports/{atp,wta}_elo_ratings.html — free,
no API key). Implements the FairValueFn interface maker_engine.py expects.

Win probability uses the standard logistic Elo formula:
    P(A beats B) = 1 / (1 + 10^(-(EloA - EloB) / 400))
This is not a guess — Tennis Abstract's own page states the exact calibration this
reproduces: "A 100-point difference implies a 64% win chance; 200 -> 76%; 300 -> 85%;
400 -> 91%; 500 -> 95%" for best-of-3 matches, which matches this formula exactly.

Known limitation, stated plainly: Elo ratings here are CURRENT, not historical
snapshots. This is fine for evaluating live/upcoming matches (the real use case),
but backtesting against old claudesports fills with today's ratings would introduce
lookahead bias — that would need a historical Elo archive, which this module doesn't
have. Don't use this file to backtest past matches without addressing that first.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

CACHE_PATH = Path(__file__).parent / "tennis_elo_cache.json"
CACHE_TTL_SECONDS = 6 * 3600  # Tennis Abstract updates weekly; refresh a few times/day at most

ATP_URL = "https://tennisabstract.com/reports/atp_elo_ratings.html"
WTA_URL = "https://tennisabstract.com/reports/wta_elo_ratings.html"


def _normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", name.lower())


def _fetch_elo_table(url: str) -> dict[str, dict]:
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", id="reportable")
    rows = table.find_all("tr")
    header = [th.get_text(strip=True) for th in rows[0].find_all("th")]
    out = {}
    for r in rows[1:]:
        cells = [td.get_text(strip=True) for td in r.find_all("td")]
        if len(cells) != len(header):
            continue
        rec = dict(zip(header, cells))
        player = rec.get("Player", "").replace("\xa0", " ")
        try:
            elo = float(rec.get("Elo", "nan"))
            helo = float(rec.get("hElo", "nan"))
            celo = float(rec.get("cElo", "nan"))
            gelo = float(rec.get("gElo", "nan"))
        except ValueError:
            continue
        out[_normalize_name(player)] = {
            "name": player, "elo": elo, "helo": helo, "celo": celo, "gelo": gelo,
        }
    return out


def load_elo_ratings(force_refresh: bool = False) -> dict[str, dict]:
    if not force_refresh and CACHE_PATH.exists():
        age = time.time() - CACHE_PATH.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            return json.loads(CACHE_PATH.read_text())

    combined = {}
    for url in (ATP_URL, WTA_URL):
        combined.update(_fetch_elo_table(url))
    CACHE_PATH.write_text(json.dumps(combined))
    return combined


def _lookup_player(elo_data: dict, name: str) -> Optional[dict]:
    key = _normalize_name(name)
    if key in elo_data:
        return elo_data[key]
    # Fall back to last-name-only match (Polymarket titles sometimes drop first names
    # or use different orderings than Tennis Abstract).
    last = key[-8:] if len(key) > 8 else key
    candidates = [v for k, v in elo_data.items() if k.endswith(last) or last.endswith(k[-8:])]
    return candidates[0] if len(candidates) == 1 else None


def parse_matchup(question: str) -> Optional[tuple[str, str]]:
    """'Sao Paulo Open: Leylah Fernandez vs Hayu Kinoshita' -> (playerA, playerB)."""
    m = re.search(r":\s*(.+?)\s+vs\.?\s+(.+?)(?:\s*\(.*)?$", question, re.IGNORECASE)
    if not m:
        m = re.search(r"^(.+?)\s+vs\.?\s+(.+?)(?:\s*\(.*)?$", question, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def surface_from_title(title: str) -> str:
    t = title.lower()
    if "french open" in t or "roland garros" in t or "clay" in t:
        return "celo"
    if "wimbledon" in t or "grass" in t:
        return "gelo"
    return "helo"  # hard court is the tour default; also our fallback for unknown surfaces


def fair_value(market) -> Optional[float]:
    """FairValueFn-compatible: takes a maker_engine.MarketSnapshot, returns the
    model's win probability for the outcome market.market_price is currently
    quoting (i.e. the specific named player in market.question's resolvable side),
    or None if we can't confidently price it."""
    matchup = parse_matchup(market.question)
    if matchup is None:
        return None
    player_a, player_b = matchup

    elo_data = load_elo_ratings()
    a = _lookup_player(elo_data, player_a)
    b = _lookup_player(elo_data, player_b)
    if a is None or b is None:
        return None

    surface_key = surface_from_title(market.question)
    elo_a = a.get(surface_key) or a["elo"]
    elo_b = b.get(surface_key) or b["elo"]
    if elo_a != elo_a or elo_b != elo_b:  # NaN check
        elo_a, elo_b = a["elo"], b["elo"]

    prob_a_wins = 1 / (1 + 10 ** (-(elo_a - elo_b) / 400))

    # We don't currently know which named outcome market.market_price refers to
    # (player_a or player_b) from the snapshot alone — caller must track that via
    # market.question's outcome ordering. This returns P(player_a wins); invert
    # (1 - value) if evaluating player_b's side.
    return prob_a_wins


if __name__ == "__main__":
    print("Loading Tennis Abstract Elo ratings...")
    elo_data = load_elo_ratings(force_refresh=True)
    print(f"Loaded {len(elo_data)} players (ATP+WTA combined).")

    test_questions = [
        "Sao Paulo Open: Leylah Fernandez vs Hayu Kinoshita",
        "Wimbledon: Novak Djokovic vs Carlos Alcaraz",
        "French Open: Jannik Sinner vs Alexander Zverev",
    ]
    for q in test_questions:
        matchup = parse_matchup(q)
        if matchup is None:
            print(f"{q!r}: could not parse matchup")
            continue
        pa, pb = matchup
        a = _lookup_player(elo_data, pa)
        b = _lookup_player(elo_data, pb)
        if a is None or b is None:
            print(f"{q!r}: player lookup failed (a={a is not None}, b={b is not None})")
            continue
        surface_key = surface_from_title(q)
        elo_a = a.get(surface_key, a["elo"])
        elo_b = b.get(surface_key, b["elo"])
        p = 1 / (1 + 10 ** (-(elo_a - elo_b) / 400))
        print(f"{pa} ({elo_a:.0f} {surface_key}) vs {pb} ({elo_b:.0f} {surface_key}) "
              f"-> P({pa} wins) = {p:.1%}")

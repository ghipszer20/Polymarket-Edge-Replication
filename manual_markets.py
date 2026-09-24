"""
manual_markets.py
Fetches active NBA and MLB moneyline markets from Polymarket Gamma API,
groups by game date using ESPN schedule.
Run before each session to generate MANUAL_MARKETS.
"""

import re
import requests
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone


# ── NBA ───────────────────────────────────────────────────────
NBA_TEAMS = {
    "hawks","celtics","nets","hornets","bulls","cavaliers","mavericks",
    "nuggets","pistons","warriors","rockets","pacers","clippers","lakers",
    "grizzlies","heat","bucks","timberwolves","pelicans","knicks","thunder",
    "magic","76ers","suns","trail blazers","kings","spurs","raptors","jazz","wizards"
}
NBA_EXCLUDE = [
    "o/u","over","under","spread","points","assists","rebounds",
    "threes","double","triple","quarter","1h","1st half","2nd half","half","total",
]
NBA_ESPN_URL = ("https://site.api.espn.com/apis/site/v2/sports"
                "/basketball/nba/scoreboard?dates={date}")
NBA_GAMMA    = ("https://gamma-api.polymarket.com/events"
                "?active=true&closed=false&tag_slug=nba&limit=100")

# ── MLB ───────────────────────────────────────────────────────
MLB_TEAMS = {
    "yankees","dodgers","mets","red sox","cubs","astros","braves",
    "padres","phillies","giants","cardinals","brewers","rays","blue jays",
    "orioles","guardians","twins","white sox","tigers","royals","angels",
    "athletics","mariners","rangers","rockies","diamondbacks","marlins",
    "nationals","pirates","reds"
}
MLB_EXCLUDE = [
    "o/u","over","under","spread","hits","strikeouts","home runs",
    "innings","series","pennant","world series","division","playoffs",
    "mvp","cy young","wins","era","batting","game 1","game 2",
    "game 3","game 4","game 5","game 6","game 7",
]
MLB_ESPN_URL = ("https://site.api.espn.com/apis/site/v2/sports"
                "/baseball/mlb/scoreboard?dates={date}")
MLB_GAMMA    = ("https://gamma-api.polymarket.com/events"
                "?active=true&closed=false&tag_slug=mlb&limit=100")


def canonical_game_key(question: str) -> frozenset:
    """
    'Hawks vs. Celtics' and 'Celtics vs. Hawks' → same frozenset key.
    Used to detect and deduplicate mirrored markets.
    """
    parts = re.split(r'\s+vs\.?\s+', question, flags=re.IGNORECASE)
    return frozenset(p.strip().lower() for p in parts)


def normalize_market(m: dict) -> dict:
    """
    Ensure YES always corresponds to the alphabetically-first team.
    If the market is already in that order, return as-is.
    If it's the mirror, swap yes/no token IDs and rewrite the question.
    """
    parts = re.split(r'\s+vs\.?\s+', m["question"], flags=re.IGNORECASE)
    if len(parts) != 2:
        return m
    a, b = parts[0].strip(), parts[1].strip()
    if a.lower() > b.lower():
        return {
            **m,
            "question":     f"{b} vs. {a}",
            "yes_token_id": m["no_token_id"],
            "no_token_id":  m["yes_token_id"],
        }
    return m


def fetch_moneyline_markets(gamma_url: str, teams: set,
                             exclude_keywords: list) -> list[dict]:
    """Fetch all active moneyline markets for a sport from Gamma API."""
    try:
        r      = requests.get(gamma_url, timeout=10)
        events = r.json()
    except Exception as e:
        print(f"  Gamma API error: {e}")
        return []

    raw_markets = []
    seen_cids   = set()

    for event in events:
        for market in event.get("markets", []):
            q   = market.get("question", "")
            cid = market.get("conditionId", "")

            if cid in seen_cids or ":" in q:
                continue
            if any(kw in q.lower() for kw in exclude_keywords):
                continue
            if " vs" not in q.lower():
                continue

            q_clean = q.replace(".", "").strip()
            parts   = q_clean.split(" vs ")
            if len(parts) != 2:
                continue
            if len(parts[0].split()) > 4 or len(parts[1].split()) > 4:
                continue
            all_teams_in_q = [t for t in teams if t in q_clean.lower()]
            if len(all_teams_in_q) < 2:
                continue

            tokens = market.get("clobTokenIds", "[]")
            if isinstance(tokens, str):
                try:
                    tokens = json.loads(tokens)
                except:
                    continue
            if len(tokens) < 2:
                continue

            raw_markets.append({
                "market_id":    cid,
                "question":     q,
                "yes_token_id": tokens[0],
                "no_token_id":  tokens[1],
            })
            seen_cids.add(cid)

    # ── Dedup mirrored markets ────────────────────────────────
    # Normalize direction first (alphabetical), then keep one per pair.
    deduped   = {}
    for m in raw_markets:
        m_norm = normalize_market(m)
        key    = canonical_game_key(m_norm["question"])
        if key not in deduped:
            deduped[key] = m_norm
        # If we've already seen this pair, the existing entry wins —
        # both normalized versions are equivalent after the token swap.

    return list(deduped.values())


def _team_tokens(name: str) -> set[str]:
    """
    Split a team display name into lowercase word tokens.
    e.g. 'Golden State Warriors' → {'golden', 'state', 'warriors'}
    """
    return set(re.findall(r'[a-z0-9]+', name.lower()))


def _market_tokens(side: str) -> set[str]:
    """
    Split one side of a market question into lowercase word tokens.
    e.g. 'Trail Blazers' → {'trail', 'blazers'}
    """
    return set(re.findall(r'[a-z0-9]+', side.lower()))


def _teams_match(market_side: str, espn_teams: list[str]) -> bool:
    """
    Return True if any ESPN team name shares at least one meaningful
    word token with the market side string.
    Filters out common short noise words.
    """
    NOISE = {"the", "at", "vs", "and", "or", "a", "of"}
    mkt   = _market_tokens(market_side) - NOISE
    for team in espn_teams:
        espn = _team_tokens(team) - NOISE
        if mkt & espn:       # non-empty intersection → match
            return True
    return False


def match_markets_to_dates(markets: list[dict],
                            espn_url_template: str) -> dict:
    """Match markets to game dates using ESPN schedule (next 7 days)."""
    by_date         = defaultdict(list)
    assigned_market = set()   # track market_ids already placed on any date

    for day_offset in range(7):
        date     = datetime.now(timezone.utc).date() + timedelta(days=day_offset)
        date_str = date.strftime("%Y%m%d")
        url      = espn_url_template.format(date=date_str)

        try:
            resp  = requests.get(url, timeout=5)
            games = resp.json().get("events", [])
        except:
            games = []

        for game in games:
            comp       = game.get("competitions", [{}])[0]
            espn_teams = [
                c.get("team", {}).get("displayName", "")
                for c in comp.get("competitors", [])
            ]

            for market in markets:
                if market["market_id"] in assigned_market:
                    continue

                parts = re.split(r'\s+vs\.?\s+', market["question"],
                                 flags=re.IGNORECASE)
                if len(parts) != 2:
                    continue

                left, right = parts[0].strip(), parts[1].strip()

                if (_teams_match(left, espn_teams) and
                        _teams_match(right, espn_teams)):
                    by_date[str(date)].append(market)
                    assigned_market.add(market["market_id"])

    return dict(by_date)


def print_by_date(by_date: dict, sport: str) -> None:
    """Print markets grouped by date."""
    print(f"\n{'='*60}")
    print(f"{sport} MARKETS BY DATE")
    print(f"{'='*60}")
    for date in sorted(by_date.keys()):
        markets = by_date[date]
        print(f"\n  DATE: {date}  ({len(markets)} markets)")
        for m in markets:
            print(f"    {m['question']}")


def print_manual_markets(markets: list[dict], expected: int | None = None) -> None:
    """Print markets in MANUAL_MARKETS format, with optional count check."""
    if expected is not None and len(markets) != expected:
        print(f"\n  WARNING: expected {expected} markets, "
              f"got {len(markets)} — check for missing or duplicate games.\n")

    print("\nMANUAL_MARKETS = [")
    for m in markets:
        print(
            f'    {{\n'
            f'        "market_id":    "{m["market_id"]}",\n'
            f'        "question":     "{m["question"]}",\n'
            f'        "yes_token_id": "{m["yes_token_id"]}",\n'
            f'        "no_token_id":  "{m["no_token_id"]}",\n'
            f'    }},'
        )
    print("]")


def main():
    print("Fetching NBA markets...")
    nba_markets = fetch_moneyline_markets(
        NBA_GAMMA, NBA_TEAMS, NBA_EXCLUDE
    )
    print(f"  Found {len(nba_markets)} NBA moneyline markets (deduped)")

    print("Fetching MLB markets...")
    mlb_markets = fetch_moneyline_markets(
        MLB_GAMMA, MLB_TEAMS, MLB_EXCLUDE
    )
    print(f"  Found {len(mlb_markets)} MLB moneyline markets (deduped)")

    print("\nMatching to ESPN schedules...")
    nba_by_date = match_markets_to_dates(nba_markets, NBA_ESPN_URL)
    mlb_by_date = match_markets_to_dates(mlb_markets, MLB_ESPN_URL)

    print_by_date(nba_by_date, "NBA")
    print_by_date(mlb_by_date, "MLB")

    print("\n" + "="*60)
    print("Generate MANUAL_MARKETS block")
    print("="*60)
    print("Sport? (nba/mlb/both):")
    sport = input().strip().lower()

    print("Date? (e.g. 2026-03-20):")
    target = input().strip()

    print("Expected game count? (press Enter to skip):")
    expected_input = input().strip()
    expected = int(expected_input) if expected_input.isdigit() else None

    combined = []
    if sport in ("nba", "both"):
        if target in nba_by_date:
            combined.extend(nba_by_date[target])
            print(f"  Added {len(nba_by_date[target])} NBA markets")
        else:
            print(f"  No NBA markets for {target}. "
                  f"Available: {sorted(nba_by_date.keys())}")

    if sport in ("mlb", "both"):
        if target in mlb_by_date:
            combined.extend(mlb_by_date[target])
            print(f"  Added {len(mlb_by_date[target])} MLB markets")
        else:
            print(f"  No MLB markets for {target}. "
                  f"Available: {sorted(mlb_by_date.keys())}")

    if combined:
        print_manual_markets(combined, expected=expected)
    else:
        print("No markets found for the selected sport/date.")


if __name__ == "__main__":
    main()
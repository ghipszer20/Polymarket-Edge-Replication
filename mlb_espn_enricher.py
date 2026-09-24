"""
mlb_espn_enricher.py
Enriches mlb_trades.parquet with ESPN game state at trade time.
Only includes trades that occurred during the game window.
Output: poly_data/processed/mlb_trades_enriched.parquet
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_DIR  = Path(r"C:\Users\24GHi\PycharmProjects\PythonProject2")
TRADES_FILE  = PROJECT_DIR / "poly_data" / "processed" / "mlb_trades.parquet"
MARKETS_FILE = PROJECT_DIR / "poly_data" / "processed" / "mlb_markets.parquet"
OUTPUT_FILE  = PROJECT_DIR / "poly_data" / "processed" / "mlb_trades_enriched.parquet"
CACHE_DIR    = PROJECT_DIR / "poly_data" / "mlb_espn_cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger(__name__)


def make_session() -> requests.Session:
    s       = requests.Session()
    retry   = Retry(total=3, backoff_factor=0.5,
                    status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s


def get_games_for_date(session: requests.Session,
                        date_str: str) -> list[dict]:
    cache_file = CACHE_DIR / f"scoreboard_{date_str}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    try:
        r = session.get(
            "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
            params={"dates": date_str},
            timeout=10
        )
        games = r.json().get("events", [])
        cache_file.write_text(json.dumps(games))
        time.sleep(0.3)
        return games
    except Exception as e:
        log.warning(f"Scoreboard fetch failed for {date_str}: {e}")
        return []


def get_game_plays(session: requests.Session,
                    game_id: str) -> tuple[list, list]:
    cache_file = CACHE_DIR / f"plays_{game_id}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        return data.get("plays", []), data.get("winprobability", [])
    try:
        r = session.get(
            "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary",
            params={"event": game_id},
            timeout=15
        )
        data     = r.json()
        plays    = data.get("plays", [])
        win_prob = data.get("winprobability", [])
        cache_file.write_text(json.dumps(
            {"plays": plays, "winprobability": win_prob}
        ))
        time.sleep(0.3)
        return plays, win_prob
    except Exception as e:
        log.warning(f"Plays fetch failed for game {game_id}: {e}")
        return [], []


def build_play_timeline(plays: list,
                         win_prob: list) -> pd.DataFrame:
    if not plays:
        return pd.DataFrame()

    wp_lookup = {}
    for wp in win_prob:
        pid = wp.get("playId", "")
        if pid:
            wp_lookup[pid] = 1.0 - float(wp.get("homeWinPercentage", 0.5))

    rows = []
    for play in plays:
        wc = play.get("wallclock", "")
        if not wc:
            continue
        try:
            ts = datetime.fromisoformat(wc.replace("Z", "+00:00"))
        except Exception:
            continue

        period      = play.get("period", {})
        inning      = int(period.get("number", 0) or 0)
        inning_type = period.get("type", "Top").lower()
        inning_half = 1 if inning_type in ("bot", "bottom", "end") else 0

        away_score = int(play.get("awayScore", 0) or 0)
        home_score = int(play.get("homeScore", 0) or 0)
        outs       = int(play.get("outs", 0) or 0)

        on_first  = 1 if play.get("onFirst") else 0
        on_second = 1 if play.get("onSecond") else 0
        on_third  = 1 if play.get("onThird") else 0

        play_id  = play.get("id", "")
        espn_wp  = wp_lookup.get(play_id)
        if espn_wp is None:
            at_bat_id = play.get("atBatId", "")
            for k, v in wp_lookup.items():
                if k.startswith(at_bat_id):
                    espn_wp = v
                    break
        if espn_wp is None:
            espn_wp = np.nan

        half_innings_done      = (inning - 1) * 2 + inning_half
        half_innings_remaining = max(0, 18 - half_innings_done)

        rows.append({
            "wallclock":              ts,
            "inning":                 inning,
            "inning_half":            inning_half,
            "away_score":             away_score,
            "home_score":             home_score,
            "score_diff":             away_score - home_score,
            "outs":                   outs,
            "on_first":               on_first,
            "on_second":              on_second,
            "on_third":               on_third,
            "espn_win_prob":          espn_wp,
            "half_innings_remaining": half_innings_remaining,
            "game_pct_done":          min(half_innings_done / 18.0, 1.0),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("wallclock").reset_index(drop=True)
    return df


def match_team_to_game(question: str, games: list) -> dict | None:
    q = question.lower().replace(".", "").replace("-", "")

    mlb_city_map = {
        "yankees":      ["yankees", "new york", "nyy"],
        "dodgers":      ["dodgers", "los angeles", "lad", "la dodgers"],
        "mets":         ["mets", "new york", "nym"],
        "red sox":      ["red sox", "boston", "bos"],
        "cubs":         ["cubs", "chicago", "chc"],
        "astros":       ["astros", "houston", "hou"],
        "braves":       ["braves", "atlanta", "atl"],
        "padres":       ["padres", "san diego", "sd"],
        "phillies":     ["phillies", "philadelphia", "phi"],
        "giants":       ["giants", "san francisco", "sf"],
        "cardinals":    ["cardinals", "st louis", "stl"],
        "brewers":      ["brewers", "milwaukee", "mil"],
        "rays":         ["rays", "tampa bay", "tb"],
        "blue jays":    ["blue jays", "toronto", "tor"],
        "orioles":      ["orioles", "baltimore", "bal"],
        "guardians":    ["guardians", "cleveland", "cle"],
        "twins":        ["twins", "minnesota", "min"],
        "white sox":    ["white sox", "chicago", "cws"],
        "tigers":       ["tigers", "detroit", "det"],
        "royals":       ["royals", "kansas city", "kc"],
        "angels":       ["angels", "los angeles", "laa", "la angels"],
        "athletics":    ["athletics", "oakland", "oak", "ath"],
        "mariners":     ["mariners", "seattle", "sea"],
        "rangers":      ["rangers", "texas", "tex"],
        "rockies":      ["rockies", "colorado", "col"],
        "diamondbacks": ["diamondbacks", "arizona", "ari"],
        "marlins":      ["marlins", "miami", "mia"],
        "nationals":    ["nationals", "washington", "wsh", "was"],
        "pirates":      ["pirates", "pittsburgh", "pit"],
        "reds":         ["reds", "cincinnati", "cin"],
    }

    for game in games:
        comp        = game.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = next((c for c in competitors
                     if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors
                     if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        home_name = home.get("team", {}).get("displayName", "").lower()
        away_name = away.get("team", {}).get("displayName", "").lower()

        home_match = False
        away_match = False

        for team, aliases in mlb_city_map.items():
            if any(alias in q for alias in aliases):
                if any(alias in home_name for alias in aliases):
                    home_match = True
                if any(alias in away_name for alias in aliases):
                    away_match = True

        for word in home_name.split():
            if len(word) > 3 and word in q:
                home_match = True
        for word in away_name.split():
            if len(word) > 3 and word in q:
                away_match = True

        if home_match and away_match:
            return {
                "game_id":          game.get("id", ""),
                "home_team":        home_name,
                "away_team":        away_name,
                "yes_is_away":      True,
                "home_score_final": int(home.get("score", 0) or 0),
                "away_score_final": int(away.get("score", 0) or 0),
                "game_date":        game.get("date", ""),
            }

    return None


def enrich_market(market_id: str, question: str,
                   trades: pd.DataFrame,
                   session: requests.Session) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    min_ts = trades["timestamp"].min()

    matched_game  = None
    matched_plays = None
    matched_wp    = None
    game_start    = None
    game_end      = None

    for day_offset in range(-1, 3):
        date  = (min_ts + timedelta(days=day_offset)).strftime("%Y%m%d")
        games = get_games_for_date(session, date)
        game  = match_team_to_game(question, games)
        if game:
            plays, wp = get_game_plays(session, game["game_id"])
            if plays:
                # Get game start from ESPN event date
                for g in games:
                    if g.get("id") == game["game_id"]:
                        start_str = g.get("date", "")
                        if start_str:
                            try:
                                game_start = datetime.fromisoformat(
                                    start_str.replace("Z", "+00:00")
                                )
                            except Exception:
                                pass
                        break

                # Get game end from last play wallclock — no buffer
                for play in reversed(plays):
                    wc = play.get("wallclock", "")
                    if wc:
                        try:
                            game_end = datetime.fromisoformat(
                                wc.replace("Z", "+00:00")
                            )
                        except Exception:
                            pass
                        break

                matched_game  = game
                matched_plays = plays
                matched_wp    = wp
                break

    if matched_game is None or not matched_plays:
        return pd.DataFrame()

    # Filter trades to strictly within game window
    if game_start and game_end:
        trades = trades[
            (trades["timestamp"] >= game_start) &
            (trades["timestamp"] <= game_end)
        ]

    if trades.empty:
        return pd.DataFrame()

    timeline       = build_play_timeline(matched_plays, matched_wp or [])
    if timeline.empty:
        return pd.DataFrame()

    enriched_rows  = []
    timeline_times = timeline["wallclock"].values

    for _, trade in trades.iterrows():
        ts = trade["timestamp"]
        if pd.isna(ts):
            continue

        ts_np   = np.datetime64(
            ts.tz_convert(None) if ts.tzinfo else ts
        )
        idx_arr  = np.searchsorted(timeline_times, ts_np, side="right") - 1
        idx      = max(0, min(idx_arr, len(timeline) - 1))
        play_row = timeline.iloc[idx]

        row = trade.to_dict()
        row.update({
            "game_id":                matched_game["game_id"],
            "home_team":              matched_game["home_team"],
            "away_team":              matched_game["away_team"],
            "yes_is_away":            matched_game["yes_is_away"],
            "home_score_final":       matched_game["home_score_final"],
            "away_score_final":       matched_game["away_score_final"],
            "game_start":             game_start.isoformat() if game_start else None,
            "game_end":               game_end.isoformat() if game_end else None,
            "inning":                 play_row["inning"],
            "inning_half":            play_row["inning_half"],
            "away_score":             play_row["away_score"],
            "home_score":             play_row["home_score"],
            "score_diff":             play_row["score_diff"],
            "outs":                   play_row["outs"],
            "on_first":               play_row["on_first"],
            "on_second":              play_row["on_second"],
            "on_third":               play_row["on_third"],
            "espn_win_prob":          play_row["espn_win_prob"],
            "half_innings_remaining": play_row["half_innings_remaining"],
            "game_pct_done":          play_row["game_pct_done"],
            "yes_won": int(
                matched_game["away_score_final"] >
                matched_game["home_score_final"]
            ),
        })
        enriched_rows.append(row)

    return pd.DataFrame(enriched_rows)


def main():
    log.info("Loading MLB trades...")
    trades  = pd.read_parquet(TRADES_FILE)
    markets = pd.read_parquet(MARKETS_FILE)
    log.info(f"  {len(trades):,} trades across "
             f"{trades['condition_id'].nunique():,} markets")

    session = make_session()

    all_enriched  = []
    markets_done  = 0
    markets_total = trades["condition_id"].nunique()
    t0            = time.time()

    for condition_id, group in trades.groupby("condition_id"):
        markets_done += 1

        market_row = markets[markets["condition_id"] == condition_id]
        if market_row.empty:
            continue
        question = market_row.iloc[0]["question"]

        enriched = enrich_market(
            condition_id, question, group, session
        )

        if not enriched.empty:
            all_enriched.append(enriched)

        if markets_done % 50 == 0:
            elapsed        = time.time() - t0
            pct            = markets_done / markets_total * 100
            enriched_count = sum(len(e) for e in all_enriched)
            log.info(
                f"  {markets_done}/{markets_total} markets "
                f"({pct:.1f}%) | "
                f"{enriched_count:,} enriched trades | "
                f"{elapsed:.0f}s elapsed"
            )

    if not all_enriched:
        log.error("No enriched trades produced.")
        return

    log.info("Concatenating results...")
    df = pd.concat(all_enriched, ignore_index=True)

    log.info(f"Saving {len(df):,} enriched trades → {OUTPUT_FILE}")
    df.to_parquet(OUTPUT_FILE, index=False, compression="snappy")

    log.info("\nSummary:")
    log.info(f"  Total enriched trades : {len(df):,}")
    log.info(f"  Markets matched       : {df['game_id'].nunique():,}")
    log.info(f"  Inning range          : "
             f"{df['inning'].min()} - {df['inning'].max()}")
    log.info(f"  Score diff range      : "
             f"{df['score_diff'].min()} - {df['score_diff'].max()}")
    log.info(f"  ESPN WP coverage      : "
             f"{df['espn_win_prob'].notna().mean()*100:.1f}%")
    log.info(f"  YES win rate          : "
             f"{df['yes_won'].mean()*100:.1f}%")

    # Sanity check — no post-game trades
    suspicious = df[
        (df["score_diff"].abs() >= 5) &
        (df["inning"] >= 8) &
        (df["price_usdc"] > 0.05) &
        (df["price_usdc"] < 0.95)
    ]
    log.info(f"\n  Suspicious trades (score>=5 inn>=8 price 0.05-0.95): "
             f"{len(suspicious):,} "
             f"(should be near 0 after fix)")


if __name__ == "__main__":
    main()
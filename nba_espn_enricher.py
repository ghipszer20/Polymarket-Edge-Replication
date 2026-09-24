"""
nba_espn_enricher.py — full data extraction
Pulls everything useful from ESPN and attaches to every trade.
Memory safe: writes incrementally, never accumulates all markets in RAM.
"""

import gc
import requests
import time
import json
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\24GHi\PycharmProjects\PythonProject2")
TRADES_FILE = PROJECT_DIR / "poly_data" / "processed" / "nba_trades.parquet"
OUTPUT_FILE = PROJECT_DIR / "poly_data" / "processed" / "nba_trades_enriched.parquet"
CACHE_DIR   = PROJECT_DIR / "poly_data" / "espn_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NBA_TEAMS = {
    "76ers","bucks","bulls","cavaliers","celtics","clippers","grizzlies",
    "hawks","heat","hornets","jazz","kings","knicks","lakers","magic",
    "mavericks","nets","nuggets","pacers","pelicans","pistons","raptors",
    "rockets","spurs","suns","thunder","timberwolves","trail blazers",
    "warriors","wizards"
}

TEAM_MAP = {
    "76ers":         "philadelphia",
    "bucks":         "milwaukee",
    "bulls":         "chicago",
    "cavaliers":     "cleveland",
    "celtics":       "boston",
    "clippers":      "la clippers",
    "grizzlies":     "memphis",
    "hawks":         "atlanta",
    "heat":          "miami",
    "hornets":       "charlotte",
    "jazz":          "utah",
    "kings":         "sacramento",
    "knicks":        "new york",
    "lakers":        "los angeles lakers",
    "magic":         "orlando",
    "mavericks":     "dallas",
    "nets":          "brooklyn",
    "nuggets":       "denver",
    "pacers":        "indiana",
    "pelicans":      "new orleans",
    "pistons":       "detroit",
    "raptors":       "toronto",
    "rockets":       "houston",
    "spurs":         "san antonio",
    "suns":          "phoenix",
    "thunder":       "oklahoma city",
    "timberwolves":  "minnesota",
    "trail blazers": "portland",
    "warriors":      "golden state",
    "wizards":       "washington",
}


def is_nba_market(question: str) -> bool:
    """Filter out non-NBA markets that slipped through."""
    q = question.lower()
    return any(team in q for team in NBA_TEAMS)


def normalize_team(name: str) -> str:
    name = name.lower().strip()
    for key, val in TEAM_MAP.items():
        if key in name:
            return val
    return name


def parse_question(question: str):
    parts = question.replace(" vs. ", " vs ").split(" vs ")
    if len(parts) != 2:
        return None, None
    return normalize_team(parts[0].strip()), normalize_team(parts[1].strip())


def teams_match(query: str, candidate: str) -> bool:
    """Match team names avoiding short word false positives."""
    query_words     = [w for w in query.split() if len(w) > 2]
    candidate_words = candidate.split()
    return any(w in candidate_words for w in query_words)


def get_scoreboard(date_str: str):
    cache_file = CACHE_DIR / f"scoreboard_{date_str}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    try:
        url = (f"https://site.api.espn.com/apis/site/v2/sports/basketball"
               f"/nba/scoreboard?dates={date_str}")
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        cache_file.write_text(json.dumps(data))
        time.sleep(0.3)
        return data
    except Exception as e:
        print(f"    Scoreboard error {date_str}: {e}")
        return None


def get_game_data(game_id: str):
    cache_file = CACHE_DIR / f"game_{game_id}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if not cached.get("plays"):
            return None
        return cached
    try:
        url = (f"https://site.api.espn.com/apis/site/v2/sports/basketball"
               f"/nba/summary?event={game_id}")
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()

        # ── Plays ─────────────────────────────────────────────
        plays = []
        for p in data.get("plays", []):
            wc = p.get("wallclock")
            if not wc:
                continue
            coord = p.get("coordinate", {})
            participants = [
                a.get("athlete", {}).get("id")
                for a in p.get("participants", [])
                if a.get("athlete", {}).get("id")
            ]
            plays.append({
                "id":               p.get("id", ""),
                "wallclock":        wc,
                "away_score":       p.get("awayScore", 0),
                "home_score":       p.get("homeScore", 0),
                "period":           p.get("period", {}).get("number", 0),
                "clock":            p.get("clock", {}).get("displayValue", "0:00"),
                "play_type_id":     p.get("type", {}).get("id", ""),
                "play_type":        p.get("type", {}).get("text", ""),
                "play_text":        p.get("text", ""),
                "short_desc":       p.get("shortDescription", ""),
                "scoring_play":     p.get("scoringPlay", False),
                "score_value":      p.get("scoreValue", 0),
                "shooting_play":    p.get("shootingPlay", False),
                "points_attempted": p.get("pointsAttempted", 0),
                "team_id":          p.get("team", {}).get("id", ""),
                "coord_x":          coord.get("x", None),
                "coord_y":          coord.get("y", None),
                "participants":     ",".join(str(x) for x in participants),
                "sequence_number":  p.get("sequenceNumber", 0),
            })

        if not plays:
            return None

        # ── Win probability ───────────────────────────────────
        winprob = []
        for wp in data.get("winprobability", []):
            winprob.append({
                "play_id":      str(wp.get("playId", "")),
                "home_win_pct": wp.get("homeWinPercentage"),
                "tie_pct":      wp.get("tiePercentage", 0),
            })

        # ── Boxscore team stats ───────────────────────────────
        boxscore = {}
        for team_entry in data.get("boxscore", {}).get("teams", []):
            side      = team_entry.get("homeAway", "unknown")
            team_name = team_entry.get("team", {}).get("displayName", "")
            stats     = {}
            for s in team_entry.get("statistics", []):
                key = s.get("name", "").replace("-", "_")
                val = s.get("displayValue", "")
                stats[key] = val
            boxscore[side] = {"team": team_name, "stats": stats}

        # ── Player boxscore ───────────────────────────────────
        players = {}
        for team_entry in data.get("boxscore", {}).get("players", []):
            side      = team_entry.get("homeAway", "unknown")
            team_name = team_entry.get("team", {}).get("displayName", "")
            player_list = []
            for stat_group in team_entry.get("statistics", []):
                keys = stat_group.get("keys", [])
                for athlete_entry in stat_group.get("athletes", []):
                    athlete = athlete_entry.get("athlete", {})
                    vals    = athlete_entry.get("stats", [])
                    player_list.append({
                        "id":       athlete.get("id"),
                        "name":     athlete.get("displayName"),
                        "position": athlete.get("position", {}).get("abbreviation", ""),
                        "starter":  athlete_entry.get("starter", False),
                        "active":   athlete_entry.get("active", True),
                        "stats":    dict(zip(keys, vals)),
                    })
            players[side] = {"team": team_name, "players": player_list}

        # ── Game info ─────────────────────────────────────────
        gi        = data.get("gameInfo", {})
        venue     = gi.get("venue", {})
        officials = [o.get("fullName") for o in gi.get("officials", [])]
        game_info = {
            "venue_name":  venue.get("fullName", ""),
            "venue_city":  venue.get("address", {}).get("city", ""),
            "venue_state": venue.get("address", {}).get("state", ""),
            "attendance":  gi.get("attendance", None),
            "officials":   officials,
        }

        # ── Injuries ──────────────────────────────────────────
        injuries = []
        for team_entry in data.get("injuries", []):
            team_name = team_entry.get("team", {}).get("displayName", "")
            for inj in team_entry.get("injuries", []):
                athlete = inj.get("athlete", {})
                injuries.append({
                    "team":        team_name,
                    "player_id":   athlete.get("id"),
                    "player_name": athlete.get("fullName"),
                    "status":      inj.get("status"),
                    "date":        inj.get("date"),
                })

        # ── Season series ─────────────────────────────────────
        series_summary = ""
        ss = data.get("seasonseries", [])
        if ss:
            series_summary = ss[0].get("summary", "")

        # ── Meta ──────────────────────────────────────────────
        meta = data.get("meta", {})

        result = {
            "plays":          plays,
            "winprob":        winprob,
            "boxscore":       boxscore,
            "players":        players,
            "game_info":      game_info,
            "injuries":       injuries,
            "series_summary": series_summary,
            "game_state":     meta.get("gameState", ""),
            "first_play_wc":  meta.get("firstPlayWallClock", ""),
            "last_play_wc":   meta.get("lastPlayWallClock", ""),
        }
        cache_file.write_text(json.dumps(result))
        time.sleep(0.3)
        return result

    except Exception as e:
        print(f"    Game data error {game_id}: {e}")
        return None


def clock_to_secs_elapsed(clock_str: str, period: int) -> float:
    try:
        parts     = clock_str.split(":")
        mins      = float(parts[0])
        secs      = float(parts[1]) if len(parts) > 1 else 0.0
        remaining = mins * 60 + secs
        if period <= 4:
            return (period - 1) * 720 + (720 - remaining)
        else:
            return 4 * 720 + (period - 5) * 300 + (300 - remaining)
    except:
        return 0.0


def find_espn_game(team1: str, team2: str, first_trade: pd.Timestamp):
    base_date = first_trade.date()
    for delta in range(-1, 10):  # extended to 10 days for early-opening markets
        date     = base_date + pd.Timedelta(days=delta)
        date_str = date.strftime("%Y%m%d")
        sb = get_scoreboard(date_str)
        if not sb:
            continue
        for event in sb.get("events", []):
            comps       = event.get("competitions", [{}])[0]
            competitors = comps.get("competitors", [])
            if len(competitors) < 2:
                continue
            home_name, away_name = "", ""
            all_names = []
            for c in competitors:
                full = c.get("team", {}).get("displayName", "").lower()
                all_names.append(full)
                if c.get("homeAway") == "home":
                    home_name = full
                elif c.get("homeAway") == "away":
                    away_name = full
            t1 = any(teams_match(team1, n) for n in all_names)
            t2 = any(teams_match(team2, n) for n in all_names)
            if t1 and t2:
                return event["id"], home_name, away_name
    return None, None, None


def enrich_market(trades_df: pd.DataFrame,
                  game_data: dict,
                  home_team: str,
                  away_team: str,
                  market_team1: str) -> pd.DataFrame:

    trades_df      = trades_df.copy()
    plays          = game_data.get("plays", [])
    winprob        = game_data.get("winprob", [])
    game_info      = game_data.get("game_info", {})
    injuries       = game_data.get("injuries", [])
    boxscore       = game_data.get("boxscore", {})
    series_summary = game_data.get("series_summary", "")

    # Robust home team matching — skip short words
    team1_words   = [w for w in market_team1.split() if len(w) > 2]
    home_words    = home_team.split()
    team1_is_home = any(w in home_words for w in team1_words)

    # ── Game-level fields ─────────────────────────────────────
    trades_df["venue_name"]     = game_info.get("venue_name", "")
    trades_df["venue_city"]     = game_info.get("venue_city", "")
    trades_df["attendance"]     = game_info.get("attendance", None)
    trades_df["series_summary"] = series_summary
    trades_df["team1_is_home"]  = team1_is_home
    trades_df["home_team"]      = home_team
    trades_df["away_team"]      = away_team

    # Injury flags
    home_injuries = [
        i["player_name"] for i in injuries
        if i.get("team", "").lower() in home_team
        and i.get("player_name")
    ]
    away_injuries = [
        i["player_name"] for i in injuries
        if i.get("team", "").lower() in away_team
        and i.get("player_name")
    ]
    trades_df["home_injuries"]  = json.dumps(home_injuries)
    trades_df["away_injuries"]  = json.dumps(away_injuries)
    trades_df["home_injury_ct"] = len(home_injuries)
    trades_df["away_injury_ct"] = len(away_injuries)

    # Boxscore final stats
    for side in ["home", "away"]:
        bs     = boxscore.get(side, {}).get("stats", {})
        prefix = f"{side}_final_"
        for stat_key in [
            "fieldGoalPct", "threePointFieldGoalPct", "freeThrowPct",
            "totalRebounds", "assists", "steals", "blocks",
            "turnovers", "totalTurnovers", "technicalFouls",
            "flagrantFouls", "fouls",
        ]:
            col = prefix + stat_key
            val = bs.get(stat_key, None)
            try:
                trades_df[col] = float(val) if val not in (None, "") else np.nan
            except:
                trades_df[col] = np.nan

    # ── Play-by-play matching ─────────────────────────────────
    if plays:
        play_df = pd.DataFrame(plays)
        play_df["ts"] = pd.to_datetime(
            play_df["wallclock"], utc=True, errors="coerce"
        )
        play_df = play_df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        play_df["secs_elapsed"] = play_df.apply(
            lambda r: clock_to_secs_elapsed(r["clock"], r["period"]), axis=1
        )

        # Win prob matched by play id
        wp_map = {
            str(wp.get("play_id", "")): wp.get("home_win_pct")
            for wp in winprob
        }
        play_df["home_win_pct"] = pd.to_numeric(
            play_df["id"].apply(lambda x: wp_map.get(str(x))),
            errors="coerce"
        ).ffill()

        play_ts  = play_df["ts"].values
        trade_ts = trades_df["timestamp"].values
        idxs     = np.clip(
            np.searchsorted(play_ts, trade_ts, side="right") - 1,
            0, len(play_df) - 1
        )

        h_score    = play_df["home_score"].values[idxs]
        a_score    = play_df["away_score"].values[idxs]
        periods    = play_df["period"].values[idxs]
        clocks     = play_df["clock"].values[idxs]
        secs_el    = play_df["secs_elapsed"].values[idxs]
        play_types = play_df["play_type"].values[idxs]
        play_texts = play_df["play_text"].values[idxs]
        scoring    = play_df["scoring_play"].values[idxs]
        score_vals = play_df["score_value"].values[idxs]
        shooting   = play_df["shooting_play"].values[idxs]
        team_ids   = play_df["team_id"].values[idxs]
        hw_pct     = play_df["home_win_pct"].values[idxs]

        # Safe coordinate extraction
        coord_x = pd.to_numeric(play_df["coord_x"].values[idxs], errors="coerce")
        coord_y = pd.to_numeric(play_df["coord_y"].values[idxs], errors="coerce")
        coord_x = np.where(np.abs(np.nan_to_num(coord_x)) > 1000, np.nan, coord_x)
        coord_y = np.where(np.abs(np.nan_to_num(coord_y)) > 1000, np.nan, coord_y)

        yes_score = h_score if team1_is_home else a_score
        no_score  = a_score if team1_is_home else h_score
        secs_rem  = np.where(
            periods <= 4,
            np.maximum(0, 4 * 720 - secs_el),
            np.maximum(0, 4 * 720 + (periods - 4) * 300 - secs_el)
        )

        trades_df["home_score"]       = h_score.astype(int)
        trades_df["away_score"]       = a_score.astype(int)
        trades_df["yes_score"]        = yes_score.astype(int)
        trades_df["no_score"]         = no_score.astype(int)
        trades_df["score_diff"]       = (yes_score - no_score).astype(int)
        trades_df["period"]           = periods.astype(int)
        trades_df["clock"]            = clocks
        trades_df["secs_elapsed"]     = secs_el.astype(float)
        trades_df["secs_remaining"]   = secs_rem.astype(float)
        trades_df["in_game"]          = (periods >= 1) & (secs_el > 30)
        trades_df["last_play_type"]   = play_types
        trades_df["last_play_text"]   = play_texts
        trades_df["last_scoring"]     = scoring.astype(bool)
        trades_df["last_score_value"] = score_vals.astype(int)
        trades_df["last_shooting"]    = shooting.astype(bool)
        trades_df["last_team_id"]     = team_ids
        trades_df["last_coord_x"]     = coord_x.astype(float)
        trades_df["last_coord_y"]     = coord_y.astype(float)

        hw_pct_float = pd.to_numeric(hw_pct, errors="coerce")
        trades_df["espn_win_prob"] = np.where(
            team1_is_home,
            hw_pct_float,
            1.0 - hw_pct_float
        )
        trades_df["poly_espn_divergence"] = (
            trades_df["price_usdc"] - trades_df["espn_win_prob"]
        )

    else:
        for col in [
            "home_score", "away_score", "yes_score", "no_score",
            "score_diff", "period", "clock", "secs_elapsed",
            "secs_remaining", "in_game", "last_play_type", "last_play_text",
            "last_scoring", "last_score_value", "last_shooting", "last_team_id",
            "last_coord_x", "last_coord_y", "espn_win_prob", "poly_espn_divergence",
        ]:
            trades_df[col] = None

    return trades_df


# ── Main ──────────────────────────────────────────────────────
def main():
    print("Loading trades...")
    df = pd.read_parquet(TRADES_FILE)
    df = df[df["market_type"] == "moneyline"].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    print(f"  {len(df):,} trades, {df['market_id'].nunique():,} markets")

    market_info = df.groupby("market_id").agg(
        question    = ("question", "first"),
        first_trade = ("timestamp", "min"),
        last_trade  = ("timestamp", "max"),
        n_trades    = ("timestamp", "count"),
    ).reset_index()
    market_info["duration_days"] = (
        market_info["last_trade"] - market_info["first_trade"]
    ).dt.days
    market_info = market_info[
        (market_info["duration_days"] <= 7) &
        (market_info["n_trades"] >= 500)
    ].reset_index(drop=True)
    print(f"  {len(market_info):,} single-game markets to enrich")

    writer    = None
    matched   = 0
    unmatched = 0
    no_plays  = 0
    skipped   = 0

    for i, row in market_info.iterrows():
        mid      = row["market_id"]
        question = row["question"]

        # Skip non-NBA markets
        if not is_nba_market(question):
            skipped += 1
            print(f"  [{i+1}/{len(market_info)}] SKIPPING non-NBA: {question}")
            continue

        team1, team2 = parse_question(question)
        if not team1 or not team2:
            unmatched += 1
            continue

        game_id, home_team, away_team = find_espn_game(
            team1, team2, row["first_trade"]
        )

        if not game_id:
            unmatched += 1
            if unmatched <= 10 or (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(market_info)}] NO MATCH: {question}")
            continue

        game_data = get_game_data(game_id)
        if not game_data or not game_data.get("plays"):
            no_plays += 1
            continue

        market_trades = df[df["market_id"] == mid].sort_values("timestamp").copy()
        enriched      = enrich_market(
            market_trades, game_data,
            home_team, away_team, team1
        )

        # Write incrementally
        table = pa.Table.from_pandas(enriched, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUTPUT_FILE, table.schema, compression="snappy")
        writer.write_table(table)
        matched += 1

        del enriched, market_trades, table
        if (i + 1) % 50 == 0:
            gc.collect()
            print(f"  [{i+1}/{len(market_info)}] "
                  f"matched={matched} unmatched={unmatched} "
                  f"skipped={skipped} no_plays={no_plays}")

    if writer:
        writer.close()

    print(f"\nFinished: matched={matched} unmatched={unmatched} "
          f"skipped={skipped} no_plays={no_plays}")
    print(f"Saved → {OUTPUT_FILE}")

    # Lightweight validation
    print("\nValidating...")
    val = pd.read_parquet(OUTPUT_FILE, columns=[
        "market_id", "price_usdc", "espn_win_prob",
        "poly_espn_divergence", "in_game", "period"
    ])
    print(f"  Total trades        : {len(val):,}")
    print(f"  Markets             : {val['market_id'].nunique():,}")
    print(f"  In-game trades      : {val['in_game'].sum():,}")
    print(f"  ESPN prob coverage  : {val['espn_win_prob'].notna().mean()*100:.1f}%")
    print(f"  Price range         : {val['price_usdc'].min():.3f} - {val['price_usdc'].max():.3f}")
    print(f"  Avg divergence      : {val['poly_espn_divergence'].abs().mean():.4f}")


if __name__ == "__main__":
    main()
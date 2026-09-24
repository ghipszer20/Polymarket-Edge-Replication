import requests

r = requests.get(
    "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json",
    headers={"User-Agent": "Mozilla/5.0"}
)
games = r.json().get("scoreboard", {}).get("games", [])
print(f"Games today: {len(games)}")
for g in games:
    print(f"  {g['awayTeam']['teamTricode']} @ {g['homeTeam']['teamTricode']} "
          f"status={g['gameStatus']} period={g['period']} "
          f"clock={g['gameClock']}")
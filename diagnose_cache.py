import json
from pathlib import Path

CACHE_DIR = Path(r"C:\Users\24GHi\PycharmProjects\PythonProject2\poly_data\espn_cache")

# Load a cached game
game_file = list(CACHE_DIR.glob("game_*.json"))[0]
data = json.loads(game_file.read_text())

print("Keys:", list(data.keys()))
print(f"\nPlays sample (first 3):")
for p in data["plays"][:3]:
    print(f"  sequence={p.get('sequence_number')}  play_text={p.get('play_text','')[:40]}")

print(f"\nWinprob sample (first 3):")
for wp in data["winprob"][:3]:
    print(f"  {wp}")

print(f"\nTotal plays:   {len(data['plays'])}")
print(f"Total winprob: {len(data['winprob'])}")
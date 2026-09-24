"""
nba_build_winprob_model.py — fixed
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
import pickle
import time

PROJECT_DIR = Path(r"C:\Users\24GHi\PycharmProjects\PythonProject2")
ENRICHED    = PROJECT_DIR / "poly_data" / "processed" / "nba_trades_enriched.parquet"
MODEL_FILE  = PROJECT_DIR / "poly_data" / "processed" / "winprob_model.pkl"

print("Loading data...")
df = pd.read_parquet(ENRICHED, columns=[
    "market_id", "timestamp", "price_usdc", "side",
    "score_diff", "secs_remaining", "secs_elapsed",
    "period", "in_game", "espn_win_prob",
    "yes_score", "no_score", "home_score", "away_score",
    "team1_is_home"
])
df = df[df["in_game"] == True].copy()
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
print(f"  {len(df):,} in-game trades")

# Get final outcome per market
# Use final scores directly — more reliable than final price
final = df.sort_values("timestamp").groupby("market_id").last()[
    ["yes_score", "no_score", "price_usdc", "espn_win_prob"]
].reset_index()
final["yes_won_by_score"] = final["yes_score"] > final["no_score"]
final["yes_won_by_price"] = final["price_usdc"] >= 0.90

print(f"\nOutcome check:")
print(f"  YES won by score : {final['yes_won_by_score'].sum():,} / {len(final):,}")
print(f"  YES won by price : {final['yes_won_by_price'].sum():,} / {len(final):,}")
print(f"  Agreement        : {(final['yes_won_by_score'] == final['yes_won_by_price']).mean()*100:.1f}%")
print(f"  Avg final YES score: {final['yes_score'].mean():.1f}")
print(f"  Avg final NO score : {final['no_score'].mean():.1f}")

# Use score-based outcome — more reliable
final["yes_won"] = final["yes_won_by_score"]

df = df.merge(final[["market_id","yes_won"]], on="market_id", how="left")
df = df.dropna(subset=["score_diff","secs_remaining","yes_won","espn_win_prob"])

print(f"\nClass distribution after merge:")
print(f"  YES won: {df['yes_won'].sum():,} ({df['yes_won'].mean()*100:.1f}%)")
print(f"  NO won:  {(~df['yes_won']).sum():,} ({(~df['yes_won']).mean()*100:.1f}%)")

# Features
df["score_diff"]             = df["score_diff"].astype(float)
df["secs_remaining"]         = df["secs_remaining"].astype(float)
df["is_ot"]                  = (df["period"] > 4).astype(float)
df["score_time_interaction"] = df["score_diff"] / np.maximum(df["secs_remaining"] ** 0.5, 1)
df["log_secs_remaining"]     = np.log1p(df["secs_remaining"])
df["q1"] = (df["period"] == 1).astype(float)
df["q2"] = (df["period"] == 2).astype(float)
df["q3"] = (df["period"] == 3).astype(float)
df["q4"] = (df["period"] == 4).astype(float)

FEATURES = [
    "score_diff",
    "secs_remaining",
    "score_time_interaction",
    "log_secs_remaining",
    "is_ot",
    "q1", "q2", "q3", "q4",
]

X = df[FEATURES].values
y = df["yes_won"].astype(int).values

# Split by market
markets = df["market_id"].unique().tolist()
np.random.seed(42)
np.random.shuffle(markets)
split         = int(len(markets) * 0.8)
train_markets = set(markets[:split])
test_markets  = set(markets[split:])

train_mask = df["market_id"].isin(train_markets).values
test_mask  = df["market_id"].isin(test_markets).values

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"\nTrain: {len(X_train):,} ({len(train_markets):,} markets)")
print(f"Test:  {len(X_test):,} ({len(test_markets):,} markets)")
print(f"Train YES rate: {y_train.mean()*100:.1f}%")
print(f"Test YES rate:  {y_test.mean()*100:.1f}%")

# Train with scaling — fixes convergence warning
print("\nTraining...")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",  # handle any remaining imbalance
        solver="lbfgs"
    ))
])
model.fit(X_train, y_train)

# Evaluate
print("Evaluating...")
y_prob     = model.predict_proba(X_test)[:, 1]
brier      = brier_score_loss(y_test, y_prob)
ll         = log_loss(y_test, y_prob)
espn_prob  = df[test_mask]["espn_win_prob"].values
espn_brier = brier_score_loss(y_test, espn_prob)
espn_ll    = log_loss(y_test, espn_prob)

print(f"\n  Our model  — Brier: {brier:.4f}  LogLoss: {ll:.4f}")
print(f"  ESPN model — Brier: {espn_brier:.4f}  LogLoss: {espn_ll:.4f}")
print(f"  Random     — Brier: 0.2500")

# Calibration
print("\nCalibration:")
df_test         = df[test_mask].copy()
df_test["prob"] = y_prob
df_test["won"]  = y_test
bins   = np.arange(0, 1.1, 0.1)
labels = [f"{int(b*100)}-{int(b*100+10)}%" for b in bins[:-1]]
df_test["bucket"] = pd.cut(df_test["prob"], bins=bins, labels=labels)
cal = df_test.groupby("bucket", observed=True).agg(
    n=("won","count"),
    pred=("prob","mean"),
    actual=("won","mean")
).round(3)
print(cal[cal["n"] > 0].to_string())

# Sample predictions — sanity check
print("\nSanity check predictions:")
scenarios = [
    ([0,   1440, 0,    7.27, 0, 1,0,0,0], "Tip-off, tied"),
    ([5,    600, 0.28, 6.40, 0, 0,0,0,1], "Up 5, Q4 10min"),
    ([-5,   300,-0.40, 5.70, 0, 0,0,0,1], "Down 5, Q4 5min"),
    ([0,    120, 0,    4.79, 0, 0,0,0,1], "Tied, Q4 2min"),
    ([10,    60, 1.26, 4.09, 0, 0,0,0,1], "Up 10, Q4 1min"),
    ([-10,   60,-1.26, 4.09, 0, 0,0,0,1], "Down 10, Q4 1min"),
    ([0,     30, 0,    3.43, 0, 0,0,0,1], "Tied, Q4 30sec"),
    ([3,     30, 0.55, 3.43, 0, 0,0,0,1], "Up 3, Q4 30sec"),
]
test_X = np.array([s[0] for s in scenarios])
probs  = model.predict_proba(test_X)[:, 1]
for (_, desc), prob in zip(scenarios, probs):
    print(f"  {desc:<25}: {prob:.3f}")

# Inference speed
print("\nInference speed:")
start = time.time()
for _ in range(10000):
    _ = model.predict_proba(test_X[:1])[:, 1]
elapsed = (time.time() - start) * 1000
print(f"  10,000 single predictions: {elapsed:.1f}ms ({elapsed/10000:.4f}ms each)")

with open(MODEL_FILE, "wb") as f:
    pickle.dump({
        "model":    model,
        "features": FEATURES,
        "brier":    brier,
        "log_loss": ll,
    }, f)
print(f"\nSaved → {MODEL_FILE}")
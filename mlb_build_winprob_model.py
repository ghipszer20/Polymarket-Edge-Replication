"""
mlb_winprob_model.py
Trains a win probability model for MLB using score_diff, inning,
inning_half, and outs as features.
Output: poly_data/processed/mlb_winprob_model.pkl
"""

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(r"C:\Users\24GHi\PycharmProjects\PythonProject2")
INPUT_FILE  = PROJECT_DIR / "poly_data" / "processed" / "mlb_trades_enriched.parquet"
OUTPUT_FILE = PROJECT_DIR / "poly_data" / "processed" / "mlb_winprob_model.pkl"

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger(__name__)

FEATURES = [
    "score_diff",
    "inning",
    "inning_half",
    "outs",
    "half_innings_remaining",
    "score_diff_x_innings",
    "abs_score_diff",
    "is_extra_innings",
    "inning_1","inning_2","inning_3","inning_4",
    "inning_5","inning_6","inning_7","inning_8","inning_9",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["half_innings_remaining"] = np.maximum(
        0, 18 - ((df["inning"] - 1) * 2 + df["inning_half"])
    )
    df["score_diff_x_innings"] = (
        df["score_diff"] * df["half_innings_remaining"]
    )
    df["abs_score_diff"]   = df["score_diff"].abs()
    df["is_extra_innings"] = (df["inning"] > 9).astype(int)
    for i in range(1, 10):
        df[f"inning_{i}"] = (df["inning"] == i).astype(int)
    return df


def main():
    log.info(f"Loading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    log.info(f"  {len(df):,} rows")

    # Filter to genuinely in-progress game states only
    df = df[
        (df["inning"] >= 1) &
        (df["inning"] <= 15) &
        (df["yes_won"].notna()) &
        (df["half_innings_remaining"] > 0) &  # game not over
        (df["price_usdc"] > 0.05) &           # not already decided on Poly
        (df["price_usdc"] < 0.95)
    ].copy()
    log.info(f"  {len(df):,} in-progress rows after filter")

    df = build_features(df)
    df = df.dropna(subset=FEATURES + ["yes_won"])
    log.info(f"  {len(df):,} rows after dropping NaN")

    X = df[FEATURES].values
    y = df["yes_won"].astype(int).values

    # Random train/test split — use all years
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    log.info(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train logistic regression
    log.info("Training model...")
    t0    = time.time()
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train, y_train)

    train_brier = brier_score_loss(y_train, model.predict_proba(X_train)[:, 1])
    test_brier  = brier_score_loss(y_test,  model.predict_proba(X_test)[:, 1])
    log.info(f"  Train Brier : {train_brier:.4f}")
    log.info(f"  Test Brier  : {test_brier:.4f}")
    log.info(f"  Trained in {time.time()-t0:.1f}s")

    # Compare to ESPN WP where available
    mid_mask = (
        (df["espn_win_prob"] > 0.05) &
        (df["espn_win_prob"] < 0.95)
    )
    if mid_mask.sum() > 1000:
        espn_sub = df[mid_mask]
        if len(espn_sub) > 100:
            espn_brier = brier_score_loss(
                espn_sub["yes_won"].astype(int),
                espn_sub["espn_win_prob"]
            )
            our_preds = model.predict_proba(
                espn_sub[FEATURES].values
            )[:, 1]
            our_brier = brier_score_loss(
                espn_sub["yes_won"].astype(int), our_preds
            )
            log.info(f"\n  On mid-range ESPN WP subset ({len(espn_sub):,} trades):")
            log.info(f"    ESPN Brier : {espn_brier:.4f}")
            log.info(f"    Our Brier  : {our_brier:.4f}")

    # Save
    data = {
        "model":    model,
        "features": FEATURES,
        "brier":    test_brier,
    }
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)
    log.info(f"\nSaved → {OUTPUT_FILE}")

    # Sample predictions
    log.info("\nSample predictions:")
    test_cases = [
        # score_diff, inning, half, outs
        (0,  1, 0, 0),   # tied, top 1st
        (0,  5, 0, 0),   # tied, top 5th
        (3,  7, 0, 2),   # up 3, top 7th
        (-3, 7, 1, 2),   # down 3, bot 7th
        (1,  9, 0, 0),   # up 1, top 9th
        (-1, 9, 0, 0),   # down 1, top 9th
        (2,  9, 1, 2),   # up 2, bot 9th
        (5,  8, 0, 0),   # up 5, top 8th
        (0,  9, 0, 2),   # tied, top 9th 2 outs
    ]

    for score_diff, inning, half, outs in test_cases:
        hi_remaining = max(0, 18 - ((inning - 1) * 2 + half))
        row = [
            score_diff, inning, half, outs,
            hi_remaining,
            score_diff * hi_remaining,
            abs(score_diff),
            int(inning > 9),
        ] + [int(inning == i) for i in range(1, 10)]
        prob = model.predict_proba([row])[0, 1]
        log.info(f"  score={score_diff:+d} inn={inning} "
                 f"half={'bot' if half else 'top'} outs={outs} "
                 f"→ P(away wins)={prob:.3f}")


if __name__ == "__main__":
    main()
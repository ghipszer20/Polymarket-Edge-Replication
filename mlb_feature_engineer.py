"""
mlb_feature_engineer.py
Builds ML features from mlb_trades_enriched.parquet.
Output: poly_data/processed/mlb_trades_features.parquet
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR   = Path(r"C:\Users\24GHi\PycharmProjects\PythonProject2")
INPUT_FILE    = PROJECT_DIR / "poly_data" / "processed" / "mlb_trades_enriched.parquet"
OUTPUT_FILE   = PROJECT_DIR / "poly_data" / "processed" / "mlb_trades_features.parquet"

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Engineering features...")

    # ── Sort by market + time ─────────────────────────────────
    df = df.sort_values(["condition_id", "timestamp"]).reset_index(drop=True)

    # ── Game state features ───────────────────────────────────
    # Innings remaining (from YES team perspective)
    # Standard game = 18 half-innings
    df["half_innings_done"]      = (df["inning"] - 1) * 2 + df["inning_half"]
    df["half_innings_remaining"] = np.maximum(0, 18 - df["half_innings_done"])
    df["game_pct_done"]          = (df["half_innings_done"] / 18.0).clip(0, 1)

    # Is late game (7th inning or later)
    df["is_late_game"]    = (df["inning"] >= 7).astype(int)
    df["is_extra_innings"] = (df["inning"] > 9).astype(int)

    # Inning dummies
    for i in range(1, 10):
        df[f"inning_{i}"] = (df["inning"] == i).astype(int)

    # ── Score features ────────────────────────────────────────
    df["abs_score_diff"]   = df["score_diff"].abs()
    df["yes_leading"]      = (df["score_diff"] > 0).astype(int)
    df["tied"]             = (df["score_diff"] == 0).astype(int)
    df["score_diff_sq"]    = df["score_diff"] ** 2

    # Score × game state interaction
    df["score_innings_interaction"] = (
        df["score_diff"] * df["half_innings_remaining"]
    )
    df["abs_score_innings"] = (
        df["abs_score_diff"] * df["half_innings_remaining"]
    )

    # ── Runners on base ───────────────────────────────────────
    df["runners_on"]    = df["on_first"] + df["on_second"] + df["on_third"]
    df["scoring_pos"]   = (df["on_second"] + df["on_third"]).clip(0, 1)
    df["bases_loaded"]  = (
        (df["on_first"] == 1) &
        (df["on_second"] == 1) &
        (df["on_third"] == 1)
    ).astype(int)

    # ── Divergence features ───────────────────────────────────
    # poly_espn_div = poly price - ESPN win prob
    df["poly_espn_div"]     = df["price_usdc"] - df["espn_win_prob"]
    df["abs_poly_espn_div"] = df["poly_espn_div"].abs()

    # ── Momentum features (price movement) ───────────────────
    # Price momentum within each market
    for window in [3, 5, 10, 20]:
        df[f"move{window}"] = (
            df.groupby("condition_id")["price_usdc"]
            .transform(lambda x: x.diff(window))
        )

    # Price volatility
    for window in [5, 10]:
        df[f"vol{window}"] = (
            df.groupby("condition_id")["price_usdc"]
            .transform(lambda x: x.rolling(window, min_periods=2).std())
        )

    # ── Price level features ──────────────────────────────────
    df["price_mid_dist"]  = (df["price_usdc"] - 0.5).abs()
    df["log_price"]       = np.log(df["price_usdc"].clip(0.001, 0.999))
    df["log_price_inv"]   = np.log(1 - df["price_usdc"].clip(0.001, 0.999))

    # ── ESPN win prob features ────────────────────────────────
    df["espn_wp_mid_dist"] = (df["espn_win_prob"] - 0.5).abs()
    df["log_espn_wp"]      = np.log(df["espn_win_prob"].clip(0.001, 0.999))

    # ── Trade size features ───────────────────────────────────
    df["trade_size"] = df["makerAmountFilled"].fillna(0)
    df["log_trade_size"] = np.log1p(df["trade_size"])

    # ── Target variable ───────────────────────────────────────
    # yes_won = 1 if away team won (YES token resolves to 1)
    # Already set in enricher

    return df


def main():
    log.info(f"Loading {INPUT_FILE}...")
    t0 = time.time()
    df = pd.read_parquet(INPUT_FILE)
    log.info(f"  {len(df):,} rows, {df.shape[1]} columns")

    df = engineer_features(df)

    log.info(f"Saving {len(df):,} rows, {df.shape[1]} columns → {OUTPUT_FILE}")
    df.to_parquet(OUTPUT_FILE, index=False, compression="snappy")

    elapsed = time.time() - t0
    log.info(f"Done in {elapsed:.0f}s")

    # Validation
    log.info("\nFeature summary:")
    log.info(f"  price_usdc mean     : {df['price_usdc'].mean():.4f}")
    log.info(f"  espn_win_prob mean  : {df['espn_win_prob'].mean():.4f}")
    log.info(f"  poly_espn_div mean  : {df['poly_espn_div'].mean():.4f}")
    log.info(f"  abs_div mean        : {df['abs_poly_espn_div'].mean():.4f}")
    log.info(f"  yes_won rate        : {df['yes_won'].mean():.4f}")
    log.info(f"  inning mean         : {df['inning'].mean():.2f}")
    log.info(f"  game_pct_done mean  : {df['game_pct_done'].mean():.4f}")
    log.info(f"  score_diff mean     : {df['score_diff'].mean():.4f}")
    log.info(f"  runners_on mean     : {df['runners_on'].mean():.4f}")

    # Check divergence distribution
    log.info("\nDivergence distribution:")
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25]:
        pct = (df["abs_poly_espn_div"] > threshold).mean() * 100
        log.info(f"  |div| > {threshold:.2f}: {pct:.1f}% of trades")


if __name__ == "__main__":
    main()
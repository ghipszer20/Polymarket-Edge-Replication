"""
mlb_signal_scanner.py
Discovers trading signals using our win probability model.
Uses Jan-Jun 2025 for discovery, Jul-Oct 2025 for validation.
"""

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(r"C:\Users\24GHi\PycharmProjects\PythonProject2")
INPUT_FILE  = PROJECT_DIR / "poly_data" / "processed" / "mlb_trades_features.parquet"
MODEL_FILE  = PROJECT_DIR / "poly_data" / "processed" / "mlb_winprob_model.pkl"
OUTPUT_FILE = PROJECT_DIR / "poly_data" / "processed" / "mlb_signal_scan.parquet"

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_FEATURES = [
    "score_diff", "inning", "inning_half", "outs",
    "half_innings_remaining", "score_diff_x_innings",
    "abs_score_diff", "is_extra_innings",
    "inning_1","inning_2","inning_3","inning_4",
    "inning_5","inning_6","inning_7","inning_8","inning_9",
]

# ── Signal definitions ────────────────────────────────────────
SIGNALS = []

for div_thresh in [0.10, 0.15, 0.20, 0.25, 0.30]:
    for inning_min in [6, 7, 8]:
        SIGNALS.append({
            "name":      f"poly_above_model_{int(div_thresh*100)}_inn{inning_min}",
            "direction": -1,
            "filters": {
                "poly_model_div": (div_thresh, None),
                "inning":         (inning_min, None),
                "price_usdc":     (0.05, 0.95),
            }
        })
        SIGNALS.append({
            "name":      f"poly_below_model_{int(div_thresh*100)}_inn{inning_min}",
            "direction": +1,
            "filters": {
                "poly_model_div": (None, -div_thresh),
                "inning":         (inning_min, None),
                "price_usdc":     (0.05, 0.95),
            }
        })

for score_thresh in [2, 3, 4, 5]:
    for inning_min in [7, 8, 9]:
        SIGNALS.append({
            "name":      f"yes_leading_{score_thresh}_inn{inning_min}",
            "direction": +1,
            "filters": {
                "score_diff": (score_thresh, None),
                "inning":     (inning_min, None),
                "price_usdc": (0.05, 0.95),
            }
        })
        SIGNALS.append({
            "name":      f"no_leading_{score_thresh}_inn{inning_min}",
            "direction": -1,
            "filters": {
                "score_diff": (None, -score_thresh),
                "inning":     (inning_min, None),
                "price_usdc": (0.05, 0.95),
            }
        })

for mom_thresh in [0.05, 0.10, 0.15]:
    for inning_min in [6, 7, 8]:
        SIGNALS.append({
            "name":      f"mom_up_{int(mom_thresh*100)}_inn{inning_min}",
            "direction": +1,
            "filters": {
                "move5":      (mom_thresh, None),
                "inning":     (inning_min, None),
                "price_usdc": (0.30, 0.70),
            }
        })
        SIGNALS.append({
            "name":      f"mom_down_{int(mom_thresh*100)}_inn{inning_min}",
            "direction": -1,
            "filters": {
                "move5":      (None, -mom_thresh),
                "inning":     (inning_min, None),
                "price_usdc": (0.30, 0.70),
            }
        })

for div_thresh in [0.15, 0.20, 0.25]:
    for inning_min in [8, 9]:
        SIGNALS.append({
            "name":      f"div_above_{int(div_thresh*100)}_inn{inning_min}_late",
            "direction": -1,
            "filters": {
                "poly_model_div": (div_thresh, None),
                "inning":         (inning_min, None),
                "price_usdc":     (0.05, 0.95),
                "is_late_game":   (1, None),
            }
        })
        SIGNALS.append({
            "name":      f"div_below_{int(div_thresh*100)}_inn{inning_min}_late",
            "direction": +1,
            "filters": {
                "poly_model_div": (None, -div_thresh),
                "inning":         (inning_min, None),
                "price_usdc":     (0.05, 0.95),
                "is_late_game":   (1, None),
            }
        })

log.info(f"Total signals to test: {len(SIGNALS)}")


def apply_signal(df: pd.DataFrame, signal: dict) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in signal["filters"].items():
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        if lo is not None:
            mask &= df[col] >= lo
        if hi is not None:
            mask &= df[col] <= hi
    return mask


def evaluate_signal(df: pd.DataFrame, signal: dict,
                     direction: int) -> dict | None:
    mask    = apply_signal(df, signal)
    matched = df[mask].copy()

    if len(matched) < 20:
        return None

    if direction == +1:
        matched["won"] = (matched["yes_won"] == 1).astype(int)
        matched["ret"] = np.where(
            matched["won"] == 1,
            1.0 - matched["price_usdc"],
            -matched["price_usdc"]
        )
    else:
        matched["won"] = (matched["yes_won"] == 0).astype(int)
        no_price = 1.0 - matched["price_usdc"]
        matched["ret"] = np.where(
            matched["won"] == 1,
            1.0 - no_price,
            -no_price
        )

    matched["ret"] -= 0.001

    win_rate     = matched["won"].mean()
    total_return = matched["ret"].sum()
    avg_ret      = matched["ret"].mean()
    std_ret      = matched["ret"].std()
    sharpe       = (avg_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

    return {
        "name":         signal["name"],
        "direction":    direction,
        "n_trades":     len(matched),
        "win_rate":     round(win_rate, 4),
        "total_return": round(total_return, 4),
        "avg_return":   round(avg_ret, 6),
        "sharpe":       round(sharpe, 4),
    }


def main():
    log.info(f"Loading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    log.info(f"  {len(df):,} rows")

    # Load win prob model and compute predictions
    log.info("Computing model win probabilities...")
    with open(MODEL_FILE, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]

    # Ensure required features exist
    df["half_innings_remaining"] = np.maximum(
        0, 18 - ((df["inning"] - 1) * 2 + df["inning_half"])
    )
    df["score_diff_x_innings"] = df["score_diff"] * df["half_innings_remaining"]
    df["abs_score_diff"]       = df["score_diff"].abs()
    df["is_extra_innings"]     = (df["inning"] > 9).astype(int)
    for i in range(1, 10):
        df[f"inning_{i}"] = (df["inning"] == i).astype(int)

    valid = df[MODEL_FEATURES].notna().all(axis=1)
    df["model_win_prob"] = np.nan
    df.loc[valid, "model_win_prob"] = model.predict_proba(
        df.loc[valid, MODEL_FEATURES].values
    )[:, 1]
    df["poly_model_div"] = df["price_usdc"] - df["model_win_prob"]

    log.info(f"  Model WP coverage      : {valid.mean()*100:.1f}%")
    log.info(f"  poly_model_div mean    : {df['poly_model_div'].mean():.4f}")
    log.info(f"  |poly_model_div| > 0.10: "
             f"{(df['poly_model_div'].abs() > 0.10).mean()*100:.1f}%")

    # Only use in-progress trades
    df = df[
        (df["inning"] >= 1) &
        (df["half_innings_remaining"] > 0) &
        (df["price_usdc"] > 0.05) &
        (df["price_usdc"] < 0.95) &
        df["model_win_prob"].notna()
    ].copy()
    log.info(f"  {len(df):,} in-progress rows for signal scan")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["year"]      = df["timestamp"].dt.year
    df["month"]     = df["timestamp"].dt.month

    log.info(f"  Year distribution:\n{df['year'].value_counts().sort_index()}")

    # Discovery: Jan-Jun 2025 | Validation: Jul-Oct 2025
    disc = df[df["month"] <= 6].copy()
    val  = df[df["month"] > 6].copy()

    log.info(f"  Discovery (Jan-Jun 2025) : {len(disc):,}")
    log.info(f"  Validation (Jul-Oct 2025): {len(val):,}")

    if len(disc) < 100:
        log.error("Discovery set too small.")
        return

    # ── Discovery ─────────────────────────────────────────────
    log.info("\nRunning discovery pass...")
    t0           = time.time()
    disc_results = []

    for sig in SIGNALS:
        result = evaluate_signal(disc, sig, sig["direction"])
        if result:
            disc_results.append(result)

    if not disc_results:
        log.error("No signals produced results.")
        return

    disc_df = pd.DataFrame(disc_results).sort_values(
        "sharpe", ascending=False
    )
    log.info(f"  {len(disc_df)} signals in {time.time()-t0:.0f}s")
    log.info(f"  Sharpe >= 2.0: {(disc_df['sharpe'] >= 2.0).sum()}")

    # ── Validation ────────────────────────────────────────────
    confirmed = []
    if len(val) > 0:
        log.info("\nRunning validation pass...")
        promising = disc_df[
            (disc_df["sharpe"] >= 2.0) &
            (disc_df["win_rate"] >= 0.50)
        ]
        log.info(f"  Promising signals: {len(promising)}")

        for _, row in promising.iterrows():
            sig    = next(s for s in SIGNALS if s["name"] == row["name"])
            result = evaluate_signal(val, sig, sig["direction"])
            if result and result["sharpe"] >= 2.0:
                confirmed.append({
                    "signal":        row["name"],
                    "direction":     row["direction"],
                    "disc_wr":       row["win_rate"],
                    "disc_sharpe":   row["sharpe"],
                    "disc_n":        row["n_trades"],
                    "val_wr":        result["win_rate"],
                    "val_sharpe":    result["sharpe"],
                    "val_n":         result["n_trades"],
                    "val_total_ret": result["total_return"],
                })

    confirmed_df = pd.DataFrame(confirmed)
    if not confirmed_df.empty:
        confirmed_df = confirmed_df.sort_values(
            "val_sharpe", ascending=False
        )

    confirmed_df.to_parquet(OUTPUT_FILE, index=False)
    log.info(f"\nSaved {len(confirmed_df)} confirmed → {OUTPUT_FILE}")

    log.info("\n" + "="*70)
    log.info("CONFIRMED SIGNALS (OOS Sharpe >= 2.0)")
    log.info("="*70)
    if confirmed_df.empty:
        log.info("No signals confirmed.")
    else:
        for _, r in confirmed_df.iterrows():
            log.info(
                f"  {r['signal']:<45} "
                f"disc_wr={r['disc_wr']:.3f} disc_sh={r['disc_sharpe']:.2f} | "
                f"val_wr={r['val_wr']:.3f} val_sh={r['val_sharpe']:.2f} "
                f"n={r['val_n']}"
            )
    log.info("="*70)

    log.info("\nTop 20 discovery signals:")
    for _, r in disc_df.head(20).iterrows():
        log.info(
            f"  {r['name']:<45} "
            f"wr={r['win_rate']:.3f} sharpe={r['sharpe']:.2f} "
            f"n={r['n_trades']}"
        )


if __name__ == "__main__":
    main()
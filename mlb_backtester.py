"""
mlb_backtester.py
Backtests confirmed MLB signals using limit order strategy.
Compares exit methods: N trades, divergence close, time limit, inning end.
Output: poly_data/processed/mlb_backtest_results.parquet
"""

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR   = Path(r"C:\Users\24GHi\PycharmProjects\PythonProject2")
FEATURES_FILE = PROJECT_DIR / "poly_data" / "processed" / "mlb_trades_features.parquet"
SIGNALS_FILE  = PROJECT_DIR / "poly_data" / "processed" / "mlb_signal_scan.parquet"
MODEL_FILE    = PROJECT_DIR / "poly_data" / "processed" / "mlb_winprob_model.pkl"
OUTPUT_FILE   = PROJECT_DIR / "poly_data" / "processed" / "mlb_backtest_results.parquet"

TAKER_FEE      = 0.001
FILL_PROB      = 0.63
INITIAL_BK     = 1000.0
KELLY_FRAC     = 0.02
EXIT_N_TRADES  = 10
EXIT_DIV_THRESH = 0.05

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


def simulate_exit(entry_row: pd.Series, future_rows: pd.DataFrame,
                   direction: int) -> dict:
    results = {}

    for method in ["exit_A_10trades", "exit_B_div_closed",
                   "exit_C_time_limit", "exit_D_inning_end"]:
        if future_rows.empty:
            exit_price = entry_row["price_usdc"]
        else:
            if method == "exit_A_10trades":
                idx      = min(EXIT_N_TRADES - 1, len(future_rows) - 1)
                exit_row = future_rows.iloc[idx]

            elif method == "exit_B_div_closed":
                close_mask = future_rows["poly_model_div"].abs() < EXIT_DIV_THRESH
                exit_row   = (future_rows[close_mask].iloc[0]
                              if close_mask.any()
                              else future_rows.iloc[-1])

            elif method == "exit_C_time_limit":
                idx      = min(EXIT_N_TRADES * 6 - 1, len(future_rows) - 1)
                exit_row = future_rows.iloc[idx]

            else:  # exit_D_inning_end
                inning_change = future_rows["inning"] > entry_row["inning"]
                exit_row      = (future_rows[inning_change].iloc[0]
                                 if inning_change.any()
                                 else future_rows.iloc[-1])

            exit_price = exit_row["price_usdc"]

        if direction == +1:
            raw_pnl = exit_price - entry_row["price_usdc"]
        else:
            entry_no = 1.0 - entry_row["price_usdc"]
            exit_no  = 1.0 - exit_price
            raw_pnl  = exit_no - entry_no

        results[method] = raw_pnl - TAKER_FEE

    return results


def parse_signal_filters(signal_name: str) -> dict | None:
    filters = {}

    if signal_name.startswith("yes_leading_"):
        # yes_leading_5_inn9 → ['yes','leading','5','inn9']
        parts      = signal_name.split("_")
        score_th   = int(parts[2])
        inning_min = int(parts[3].replace("inn", ""))
        filters = {
            "score_diff": (score_th, None),
            "inning":     (inning_min, None),
            "price_usdc": (0.05, 0.95),
            "direction":  +1,
        }

    elif signal_name.startswith("no_leading_"):
        # no_leading_5_inn9 → ['no','leading','5','inn9']
        parts      = signal_name.split("_")
        score_th   = int(parts[2])
        inning_min = int(parts[3].replace("inn", ""))
        filters = {
            "score_diff": (None, -score_th),
            "inning":     (inning_min, None),
            "price_usdc": (0.05, 0.95),
            "direction":  -1,
        }

    elif signal_name.startswith("poly_above_model_"):
        # poly_above_model_30_inn6 → ['poly','above','model','30','inn6']
        parts      = signal_name.split("_")
        div_th     = int(parts[3]) / 100
        inning_min = int(parts[4].replace("inn", ""))
        filters = {
            "poly_model_div": (div_th, None),
            "inning":         (inning_min, None),
            "price_usdc":     (0.05, 0.95),
            "direction":      -1,
        }

    elif signal_name.startswith("poly_below_model_"):
        # poly_below_model_30_inn6 → ['poly','below','model','30','inn6']
        parts      = signal_name.split("_")
        div_th     = int(parts[3]) / 100
        inning_min = int(parts[4].replace("inn", ""))
        filters = {
            "poly_model_div": (None, -div_th),
            "inning":         (inning_min, None),
            "price_usdc":     (0.05, 0.95),
            "direction":      +1,
        }

    elif signal_name.startswith("div_above_"):
        # div_above_25_inn8_late → ['div','above','25','inn8','late']
        parts      = signal_name.split("_")
        div_th     = int(parts[2]) / 100
        inning_min = int(parts[3].replace("inn", ""))
        filters = {
            "poly_model_div": (div_th, None),
            "inning":         (inning_min, None),
            "price_usdc":     (0.05, 0.95),
            "is_late_game":   (1, None),
            "direction":      -1,
        }

    elif signal_name.startswith("div_below_"):
        # div_below_25_inn8_late → ['div','below','25','inn8','late']
        parts      = signal_name.split("_")
        div_th     = int(parts[2]) / 100
        inning_min = int(parts[3].replace("inn", ""))
        filters = {
            "poly_model_div": (None, -div_th),
            "inning":         (inning_min, None),
            "price_usdc":     (0.05, 0.95),
            "is_late_game":   (1, None),
            "direction":      +1,
        }

    return filters if filters else None


def check_signal(row: pd.Series, filters: dict) -> bool:
    for col, val in filters.items():
        if col == "direction":
            continue
        if isinstance(val, tuple):
            lo, hi = val
            if lo is not None and row.get(col, 0) < lo:
                return False
            if hi is not None and row.get(col, 0) > hi:
                return False
        else:
            if row.get(col) != val:
                return False
    return True


def backtest_signal(df: pd.DataFrame, signal_name: str,
                     direction: int) -> pd.DataFrame:
    filters = parse_signal_filters(signal_name)
    if not filters:
        log.warning(f"  Could not parse filters for: {signal_name}")
        return pd.DataFrame()

    trades   = []
    df_sorted = df.sort_values(["condition_id", "timestamp"])

    for condition_id, group in df_sorted.groupby("condition_id"):
        group = group.reset_index(drop=True)

        for i in range(len(group)):
            row = group.iloc[i]

            if not check_signal(row, filters):
                continue

            if np.random.random() > FILL_PROB:
                continue

            future = group.iloc[i+1:i+1+EXIT_N_TRADES*10].copy()
            if "poly_model_div" not in future.columns:
                future["poly_model_div"] = (
                    future["price_usdc"] -
                    future.get("model_win_prob", pd.Series(0.5, index=future.index))
                )

            size_usdc    = max(1000.0 * KELLY_FRAC, 1.0)
            exit_results = simulate_exit(row, future, direction)

            for exit_method, net_pnl in exit_results.items():
                pnl_usdc = net_pnl * size_usdc
                trades.append({
                    "condition_id": condition_id,
                    "timestamp":    row["timestamp"],
                    "signal":       signal_name,
                    "direction":    direction,
                    "entry_price":  row["price_usdc"],
                    "inning":       row["inning"],
                    "score_diff":   row["score_diff"],
                    "size_usdc":    size_usdc,
                    "exit_method":  exit_method,
                    "net_pnl":      net_pnl,
                    "pnl_usdc":     pnl_usdc,
                    "won":          int(net_pnl > 0),
                    "yes_won":      int(row["yes_won"]),
                })

    return pd.DataFrame(trades)


def main():
    log.info("Loading data...")
    df      = pd.read_parquet(FEATURES_FILE)
    signals = pd.read_parquet(SIGNALS_FILE)
    log.info(f"  {len(df):,} trades, {len(signals)} confirmed signals")

    log.info("Adding model win probabilities...")
    with open(MODEL_FILE, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]

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

    # Validation period only
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["month"]     = df["timestamp"].dt.month
    val_df          = df[df["month"] > 6].copy()
    log.info(f"  Validation set: {len(val_df):,} trades")

    all_results = []
    t0          = time.time()

    for _, sig_row in signals.iterrows():
        signal_name = sig_row["signal"]
        direction   = int(sig_row["direction"])
        log.info(f"  Backtesting: {signal_name} (dir={direction})")

        result = backtest_signal(val_df, signal_name, direction)
        if not result.empty:
            all_results.append(result)

    if not all_results:
        log.error("No backtest results produced.")
        return

    results = pd.concat(all_results, ignore_index=True)
    results.to_parquet(OUTPUT_FILE, index=False)
    log.info(f"\nSaved {len(results):,} trade records → {OUTPUT_FILE}")

    # ── Summary by exit method ────────────────────────────────
    log.info("\n" + "="*70)
    log.info("BACKTEST SUMMARY BY EXIT METHOD")
    log.info("="*70)

    for method in sorted(results["exit_method"].unique()):
        sub      = results[results["exit_method"] == method]
        win_rate = (sub["pnl_usdc"] > 0).mean()
        avg_pnl  = sub["pnl_usdc"].mean()
        total    = sub["pnl_usdc"].sum()
        std      = sub["pnl_usdc"].std()
        sharpe   = (avg_pnl / std * np.sqrt(252)) if std > 0 else 0
        log.info(
            f"  {method:<25} n={len(sub):>5} "
            f"wr={win_rate*100:.1f}% "
            f"avg=${avg_pnl:.4f} "
            f"total=${total:+.2f} "
            f"sharpe={sharpe:.2f}"
        )

    # ── Summary by signal ─────────────────────────────────────
    best_method = (
        results.groupby("exit_method")["pnl_usdc"]
        .mean().idxmax()
    )
    log.info(f"\n  Best exit method: {best_method}")

    log.info("\n" + "="*70)
    log.info(f"BACKTEST BY SIGNAL ({best_method})")
    log.info("="*70)

    best = results[results["exit_method"] == best_method]
    sig_summary = []
    for sig in best["signal"].unique():
        sub      = best[best["signal"] == sig]
        win_rate = (sub["pnl_usdc"] > 0).mean()
        avg_pnl  = sub["pnl_usdc"].mean()
        std      = sub["pnl_usdc"].std()
        sharpe   = (avg_pnl / std * np.sqrt(252)) if std > 0 else 0
        sig_summary.append({
            "signal":   sig,
            "n":        len(sub),
            "win_rate": win_rate,
            "avg_pnl":  avg_pnl,
            "sharpe":   sharpe,
        })

    sig_df = pd.DataFrame(sig_summary).sort_values("sharpe", ascending=False)
    for _, r in sig_df.iterrows():
        log.info(
            f"  {r['signal']:<45} "
            f"n={r['n']:>4} "
            f"wr={r['win_rate']*100:.1f}% "
            f"avg=${r['avg_pnl']:.4f} "
            f"sharpe={r['sharpe']:.2f}"
        )

    log.info(f"\nCompleted in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
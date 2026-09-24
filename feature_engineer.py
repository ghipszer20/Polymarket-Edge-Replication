"""
feature_engineer.py
Builds per-trade features for signal discovery and live bot.
Memory safe: processes one market at a time, writes incrementally.
"""

import gc
import json
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\24GHi\PycharmProjects\PythonProject2")
ENRICHED    = PROJECT_DIR / "poly_data" / "processed" / "nba_trades_enriched.parquet"
OUTPUT      = PROJECT_DIR / "poly_data" / "processed" / "nba_trades_features.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def build_wallet_profiles(df: pd.DataFrame) -> tuple[set, set]:
    """Identify sharp and whale wallets."""
    wallet_stats = df.groupby("maker").agg(
        n_trades        = ("price_usdc", "count"),
        total_volume    = ("makerAmountFilled", "sum"),
        markets_traded  = ("market_id", "nunique"),
    ).reset_index()

    sharp_threshold = wallet_stats["n_trades"].quantile(0.999)
    whale_threshold = wallet_stats["total_volume"].quantile(0.999)

    sharp_wallets = set(wallet_stats[
        wallet_stats["n_trades"] >= sharp_threshold
    ]["maker"].tolist())
    whale_wallets = set(wallet_stats[
        wallet_stats["total_volume"] >= whale_threshold
    ]["maker"].tolist())

    return sharp_wallets, whale_wallets


def compute_run(prices: np.ndarray, window: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Compute scoring run features."""
    yes_run = np.zeros(len(prices))
    no_run  = np.zeros(len(prices))
    for i in range(window, len(prices)):
        recent = prices[i-window:i]
        if np.all(np.diff(recent) > 0):
            yes_run[i] = 1
        elif np.all(np.diff(recent) < 0):
            no_run[i] = 1
    return yes_run, no_run


def engineer_market(mdf: pd.DataFrame,
                    sharp_wallets: set,
                    whale_wallets: set) -> pd.DataFrame:
    """Engineer all features for a single market."""
    mdf = mdf.sort_values("timestamp").copy()
    n   = len(mdf)

    p   = mdf["price_usdc"].values
    ts  = mdf["timestamp"].values.astype("datetime64[s]").astype(float)
    sd  = mdf["score_diff"].values.astype(float)
    sr  = mdf["secs_remaining"].values.astype(float)
    se  = mdf["secs_elapsed"].values.astype(float)

    # ── Price lags ────────────────────────────────────────────
    for lag in [1, 2, 3, 5, 10, 20, 50]:
        col = np.full(n, np.nan)
        col[lag:] = p[:-lag]
        mdf[f"p{lag}"] = col

    # ── Price moves (NEW) ─────────────────────────────────────
    for lag in [1, 2, 3, 5, 10, 20]:
        col = np.full(n, np.nan)
        col[lag:] = p[lag:] - p[:-lag]
        mdf[f"move{lag}"] = col

    # ── Size lags ─────────────────────────────────────────────
    sz = mdf["makerAmountFilled"].values.astype(float)
    for lag in [1, 3, 5, 10]:
        col = np.full(n, np.nan)
        col[lag:] = sz[:-lag]
        mdf[f"size{lag}"] = col

    # ── Rolling windows ───────────────────────────────────────
    s = pd.Series(p)
    for w in [10, 20, 50]:
        mdf[f"vol_{w}"]  = s.rolling(w, min_periods=3).std().values
        mdf[f"ma{w}"]    = s.rolling(w, min_periods=3).mean().values

    sz_s = pd.Series(sz)
    for w in [20, 50]:
        mdf[f"avg_size{w}"] = sz_s.rolling(w, min_periods=3).mean().values
        mdf[f"std_size{w}"] = sz_s.rolling(w, min_periods=3).std().values

    # ── Sigma-adjusted moves (NEW) ────────────────────────────
    for lag in [1, 2, 3, 5, 10]:
        move_col = mdf.get(f"move{lag}")
        if move_col is not None:
            denom = np.where(mdf["vol_50"].values > 0, mdf["vol_50"].values, np.nan)
            mdf[f"sigma{lag}"] = mdf[f"move{lag}"].values / denom

    # ── Derived features ──────────────────────────────────────
    vol50 = mdf["vol_50"].values
    vol20 = mdf["vol_20"].values
    vol10 = mdf["vol_10"].values
    ma10  = mdf["ma10"].values
    ma20  = mdf["ma20"].values
    ma50  = mdf["ma50"].values

    mdf["vol_ratio"]    = np.where(vol50 > 0, vol10 / vol50, np.nan)
    mdf["vol_ratio20"]  = np.where(vol50 > 0, vol20 / vol50, np.nan)
    mdf["dev_ma10"]     = p - ma10
    mdf["dev_ma20"]     = p - ma20
    mdf["dev_ma50"]     = p - ma50

    avg20 = mdf["avg_size20"].values
    mdf["size_ratio"]   = np.where(avg20 > 0, sz / avg20, np.nan)
    mdf["size1_ratio"]  = np.where(avg20 > 0, mdf["size1"].values / avg20, np.nan)

    avg50 = mdf["avg_size50"].values
    mdf["size5_ratio"]  = np.where(avg50 > 0, mdf["size5"].values / avg50, np.nan)

    # ── Game phase ────────────────────────────────────────────
    total_secs          = 4 * 720
    mdf["game_pct_done"] = np.clip(se / total_secs, 0, 1)

    # ── Score features ────────────────────────────────────────
    mdf["abs_score_diff"] = np.abs(sd)
    mdf["proj_margin"]    = np.where(
        sr > 0,
        sd * (total_secs / np.maximum(sr, 1)),
        sd
    )
    mdf["game_close"]       = (mdf["abs_score_diff"] <= 5).astype(int)
    mdf["game_competitive"] = (mdf["abs_score_diff"] <= 10).astype(int)

    # ── Run features ──────────────────────────────────────────
    yes_run, no_run     = compute_run(p, window=5)
    mdf["run_for_yes"]  = yes_run.astype(int)
    mdf["run_for_no"]   = no_run.astype(int)
    mdf["on_a_run"]     = ((yes_run + no_run) > 0).astype(int)

    # ── Polymarket momentum ───────────────────────────────────
    poly_mom = np.full(n, np.nan)
    for i in range(1, n):
        dt = ts[i] - ts[i-1]
        if 0 < dt <= 120:
            poly_mom[i] = p[i] - p[i-1]
    mdf["poly_mom_60s"] = poly_mom

    # ── ESPN divergence features ──────────────────────────────
    ewp = mdf["espn_win_prob"].values.astype(float)
    mdf["poly_espn_div"] = p - ewp
    mdf["abs_espn_div"]  = np.abs(p - ewp)

    # ── ESPN momentum ─────────────────────────────────────────
    espn_mom = np.full(n, np.nan)
    for i in range(1, n):
        dt = ts[i] - ts[i-1]
        if 0 < dt <= 120:
            espn_mom[i] = ewp[i] - ewp[i-1]
    mdf["espn_mom_60s"] = espn_mom

    # ── Run differential ─────────────────────────────────────
    for w_secs in [120, 300]:
        rd = np.full(n, np.nan)
        for i in range(1, n):
            mask = (ts >= ts[i] - w_secs) & (ts < ts[i])
            if mask.sum() > 1:
                rd[i] = p[i] - p[mask][0]
        mdf[f"run_diff_{w_secs}s"] = rd

    # ── Wallet features ───────────────────────────────────────
    mdf["maker_is_sharp"] = mdf["maker"].isin(sharp_wallets).astype(int)
    mdf["maker_is_whale"] = mdf["maker"].isin(whale_wallets).astype(int)

    # ── Last play features ────────────────────────────────────
    lpt = mdf.get("last_play_type", pd.Series([""] * n))
    ltt = mdf.get("last_play_text", pd.Series([""] * n))

    def is_score(t):
        if pd.isna(t):
            return 0
        t = str(t).lower()
        return int(any(w in t for w in ["shot", "layup", "dunk", "free throw", "three"]))

    def is_foul(t):
        if pd.isna(t):
            return 0
        return int("foul" in str(t).lower())

    def is_3pt(t):
        if pd.isna(t):
            return 0
        return int("three" in str(t).lower() or "3pt" in str(t).lower())

    def is_timeout(t):
        if pd.isna(t):
            return 0
        return int("timeout" in str(t).lower())

    mdf["last_is_score"]   = ltt.apply(is_score).values
    mdf["last_is_foul"]    = ltt.apply(is_foul).values
    mdf["last_is_3pt"]     = ltt.apply(is_3pt).values
    mdf["last_is_timeout"] = ltt.apply(is_timeout).values

    return mdf


def main():
    log.info("Building wallet profiles...")
    df_all = pd.read_parquet(ENRICHED, columns=["maker", "price_usdc",
                                                  "makerAmountFilled", "market_id"])
    sharp_wallets, whale_wallets = build_wallet_profiles(df_all)
    del df_all
    gc.collect()
    log.info(f"  Sharp wallets : {len(sharp_wallets):,}")
    log.info(f"  Whale wallets : {len(whale_wallets):,}")

    log.info("Getting market list...")
    import duckdb
    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE VIEW e AS SELECT * FROM read_parquet('{ENRICHED}')")
    market_list = con.execute("""
        SELECT market_id, COUNT(*) as n
        FROM e
        WHERE in_game = true
        GROUP BY market_id
        HAVING COUNT(*) >= 50
        ORDER BY market_id
    """).df()
    log.info(f"  {len(market_list):,} in-game markets to process")

    writer  = None
    done    = 0
    errors  = 0

    log.info("Processing markets...")
    for _, row in market_list.iterrows():
        mid = row["market_id"]
        try:
            mdf = pd.read_parquet(ENRICHED, filters=[("market_id", "=", mid)])
            mdf = mdf[mdf["in_game"] == True].copy()
            mdf["timestamp"] = pd.to_datetime(mdf["timestamp"], utc=True)

            if len(mdf) < 50:
                continue

            mdf = engineer_market(mdf, sharp_wallets, whale_wallets)

            table = pa.Table.from_pandas(mdf, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUTPUT, table.schema, compression="snappy")
            writer.write_table(table)
            done += 1

        except Exception as ex:
            log.warning(f"  ERROR market {mid}: {ex}")
            errors += 1

        if done % 50 == 0:
            gc.collect()
            log.info(f"  [{done}/{len(market_list)}] done={done} errors={errors}")

    if writer:
        writer.close()

    log.info(f"Finished: {done} markets processed, {errors} errors")
    log.info(f"Saved → {OUTPUT}")

    log.info("Validating output...")
    val = pd.read_parquet(OUTPUT, columns=[
        "market_id", "price_usdc", "poly_espn_div",
        "move1", "move5", "sigma1", "sigma5",
        "vol_10", "vol_50", "game_pct_done"
    ])
    log.info(f"Total rows     : {len(val):,}")
    log.info(f"Markets        : {val['market_id'].nunique():,}")
    log.info(f"Avg divergence : {val['poly_espn_div'].abs().mean():.4f}")
    log.info(f"Price range    : {val['price_usdc'].min():.3f} - {val['price_usdc'].max():.3f}")
    log.info(f"move1 sample   : {val['move1'].describe().to_dict()}")
    log.info(f"sigma5 sample  : {val['sigma5'].describe().to_dict()}")


if __name__ == "__main__":
    main()
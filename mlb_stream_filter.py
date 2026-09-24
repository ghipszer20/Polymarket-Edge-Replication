"""
mlb_stream_filter.py
Filters trades.csv down to MLB moneyline trades only.
Output: poly_data/processed/mlb_trades.parquet

Price normalization: price_usdc always = P(YES/away team wins)
- YES token trades: price_usdc = raw price directly
- NO token trades: price_usdc = 1 - raw price
"""

import gc
import lzma
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_DIR  = Path(r"C:\Users\24GHi\PycharmProjects\PythonProject2")
TRADES_FILE  = PROJECT_DIR / "poly_data" / "processed" / "trades.csv"
MARKETS_FILE = PROJECT_DIR / "poly_data" / "processed" / "mlb_markets.parquet"
OUTPUT_FILE  = PROJECT_DIR / "poly_data" / "processed" / "mlb_trades.parquet"

CHUNK_SIZE = 100_000

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger(__name__)

COLS = [
    "timestamp", "maker", "makerAssetId", "makerAmountFilled",
    "taker", "takerAssetId", "takerAmountFilled", "transactionHash",
]
DTYPES = {
    "maker":             "str",
    "makerAssetId":      "str",
    "makerAmountFilled": "Int64",
    "taker":             "str",
    "takerAssetId":      "str",
    "takerAmountFilled": "Int64",
    "transactionHash":   "str",
}


def build_token_whitelist(markets_path: Path) -> tuple[set, dict]:
    log.info(f"Loading MLB markets from {markets_path} ...")
    df = pd.read_parquet(markets_path)
    log.info(f"  → {len(df):,} MLB moneyline markets")

    whitelist  = set()
    token_meta = {}

    for _, row in df.iterrows():
        yes_tok = str(row["yes_token"]).strip()
        no_tok  = str(row["no_token"]).strip()

        for tok, side in [(yes_tok, "YES"), (no_tok, "NO")]:
            if not tok or tok == "nan":
                continue
            whitelist.add(tok)
            token_meta[tok] = {
                "condition_id": row["condition_id"],
                "question":     row["question"],
                "side":         side,
                "end_date":     row.get("end_date", ""),
                "volume":       row.get("volume", 0),
            }

    log.info(f"  → {len(whitelist):,} token IDs in whitelist")
    return whitelist, token_meta


def compute_price(maker_amt: float, taker_amt: float,
                  maker_is_token: bool) -> float:
    if maker_amt == 0 or taker_amt == 0:
        return float("nan")
    if maker_is_token:
        return taker_amt / maker_amt
    else:
        return maker_amt / taker_amt


def stream_filter(trades_path: Path, whitelist: set,
                  token_meta: dict) -> None:
    writer       = None
    total_rows   = 0
    matched_rows = 0
    chunks_done  = 0
    t0           = time.time()

    log.info(f"Streaming {trades_path} ...")

    with lzma.open(trades_path, mode="rt", encoding="utf-8") as f:
        reader = pd.read_csv(
            f,
            names  = COLS,
            dtype  = DTYPES,
            chunksize = CHUNK_SIZE,
            header = 0,
            on_bad_lines = "skip",
        )

        for chunk in reader:
            chunks_done += 1
            total_rows  += len(chunk)

            mask = (
                chunk["makerAssetId"].isin(whitelist) |
                chunk["takerAssetId"].isin(whitelist)
            )

            if chunks_done % 100 == 0:
                elapsed = time.time() - t0
                log.info(
                    f"  Chunk {chunks_done:>4} | "
                    f"{total_rows/1e6:.1f}M rows scanned | "
                    f"{matched_rows:,} matched | "
                    f"{elapsed:.0f}s elapsed"
                )

            if not mask.any():
                continue

            matched = chunk[mask].copy()

            matched["maker_is_token"] = matched["makerAssetId"].isin(whitelist)
            matched["token_id"] = np.where(
                matched["maker_is_token"],
                matched["makerAssetId"],
                matched["takerAssetId"]
            )

            matched["condition_id"] = matched["token_id"].map(
                lambda t: token_meta.get(t, {}).get("condition_id"))
            matched["question"]     = matched["token_id"].map(
                lambda t: token_meta.get(t, {}).get("question"))
            matched["side"]         = matched["token_id"].map(
                lambda t: token_meta.get(t, {}).get("side"))
            matched["end_date"]     = matched["token_id"].map(
                lambda t: token_meta.get(t, {}).get("end_date"))

            matched["timestamp"] = pd.to_datetime(
                matched["timestamp"], unit="s", utc=True, errors="coerce"
            )

            matched["makerAmountFilled"] = matched["makerAmountFilled"].astype(float)
            matched["takerAmountFilled"] = matched["takerAmountFilled"].astype(float)

            matched["price_raw"] = np.vectorize(compute_price)(
                matched["makerAmountFilled"].values,
                matched["takerAmountFilled"].values,
                matched["maker_is_token"].values,
            )

            # Normalize: price_usdc = P(YES/away wins)
            matched["price_usdc"] = np.where(
                matched["side"] == "YES",
                matched["price_raw"],
                1.0 - matched["price_raw"]
            )
            matched["price_usdc"] = matched["price_usdc"].clip(0.001, 0.999)

            matched["makerAmountFilled"] = matched["makerAmountFilled"] / 1_000_000
            matched["takerAmountFilled"] = matched["takerAmountFilled"] / 1_000_000

            matched = matched.drop(columns=["maker_is_token", "price_raw"])

            matched_rows += len(matched)

            table = pa.Table.from_pandas(matched, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    OUTPUT_FILE, table.schema, compression="snappy"
                )
            writer.write_table(table)

    if writer:
        writer.close()

    elapsed = time.time() - t0
    log.info(f"Done in {elapsed:.0f}s")
    log.info(f"  Total rows scanned : {total_rows:,}")
    log.info(f"  Matched rows       : {matched_rows:,}")
    log.info(f"  Saved → {OUTPUT_FILE}")


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    whitelist, token_meta = build_token_whitelist(MARKETS_FILE)

    if len(whitelist) < 10:
        log.error(f"Whitelist too small ({len(whitelist)}) — aborting.")
        return

    stream_filter(TRADES_FILE, whitelist, token_meta)

    log.info("Deduplicating...")
    TEMP_FILE = OUTPUT_FILE.with_suffix(".tmp.parquet")
    df        = pd.read_parquet(OUTPUT_FILE)
    before    = len(df)
    df        = df.drop_duplicates(subset="transactionHash", keep="first")
    log.info(f"  Dropped {before - len(df):,} dupes, {len(df):,} remain")
    df.to_parquet(TEMP_FILE, index=False, compression="snappy")
    del df
    gc.collect()
    OUTPUT_FILE.unlink(missing_ok=True)
    TEMP_FILE.rename(OUTPUT_FILE)

    log.info("Validating...")
    sample = pd.read_parquet(OUTPUT_FILE, columns=["price_usdc", "side"])
    log.info(f"  Price range : {sample['price_usdc'].min():.4f} - "
             f"{sample['price_usdc'].max():.4f}")
    log.info(f"  Price mean  : {sample['price_usdc'].mean():.4f}")
    yes_avg = sample[sample["side"] == "YES"]["price_usdc"].mean()
    no_avg  = sample[sample["side"] == "NO"]["price_usdc"].mean()
    log.info(f"  YES avg     : {yes_avg:.4f}")
    log.info(f"  NO avg      : {no_avg:.4f}")
    log.info(f"  Expected    : both ~0.45-0.55")


if __name__ == "__main__":
    main()
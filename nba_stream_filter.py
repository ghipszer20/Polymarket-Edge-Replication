"""
nba_stream_filter.py
Filters the 150M-row trades.csv.xz down to NBA moneyline trades only.
Output: poly_data/processed/nba_trades.parquet

Price normalization: price_usdc always = P(YES/away team wins)
- YES token trades: price_usdc = P(away wins) directly
- NO token trades: price_usdc = 1 - P(home wins) = P(away wins)
"""

import gc
import json
import lzma
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Config ────────────────────────────────────────────────────
PROJECT_DIR   = Path(r"C:\Users\24GHi\PycharmProjects\PythonProject2")
TRADES_FILE   = PROJECT_DIR / "poly_data" / "processed" / "trades.csv"
METADATA_FILE = PROJECT_DIR / "poly_data" / "processed" / "markets_metadata.parquet"
OUTPUT_FILE   = PROJECT_DIR / "poly_data" / "processed" / "nba_trades.parquet"

CHUNK_SIZE   = 100_000
MARKET_TYPES = {"moneyline"}

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── CSV columns ───────────────────────────────────────────────
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


def build_token_whitelist(metadata_path: Path) -> tuple[set, dict]:
    """
    Returns:
        whitelist  — set of token IDs for moneyline markets
        token_meta — dict mapping token_id -> metadata

    Token assignment (Polymarket "Away vs. Home" format):
        token[0] = away team token  → side = "YES"
        token[1] = home team token  → side = "NO"
    """
    log.info(f"Loading metadata from {metadata_path} ...")
    df = pd.read_parquet(metadata_path)
    df = df[df["market_type"].isin(MARKET_TYPES)].copy()
    log.info(f"  → {len(df):,} moneyline markets")

    whitelist  = set()
    token_meta = {}

    for _, row in df.iterrows():
        raw = row.get("all_tokens", "")
        try:
            tokens = json.loads(raw) if isinstance(raw, str) else []
        except (json.JSONDecodeError, TypeError):
            tokens = []

        # token[0] = away team = YES, token[1] = home team = NO
        sides = ["YES", "NO"]
        for i, tok in enumerate(tokens):
            tok = str(tok).strip()
            if not tok or tok == "nan":
                continue
            side = sides[i] if i < len(sides) else f"TOKEN_{i}"
            whitelist.add(tok)
            token_meta[tok] = {
                "market_id":   row["market_id"],
                "market_type": row["market_type"],
                "question":    row["question"],
                "side":        side,
                "is_ingame":   row.get("is_ingame", False),
                "is_half":     row.get("is_half", False),
            }

    log.info(f"  → {len(whitelist):,} token IDs in whitelist")
    return whitelist, token_meta


def compute_price(maker_amt: float, taker_amt: float,
                  maker_is_token: bool) -> float:
    """
    Price of outcome token in USDC.
    price = USDC_amount / token_amount

    If maker gives tokens, taker gives USDC:
        price = taker / maker
    If maker gives USDC, taker gives tokens:
        price = maker / taker
    """
    if maker_amt == 0 or taker_amt == 0:
        return float("nan")
    if maker_is_token:
        return taker_amt / maker_amt
    else:
        return maker_amt / taker_amt


def stream_filter(trades_path: Path, whitelist: set, token_meta: dict) -> None:
    """Stream through XZ-compressed CSV, filter, save parquet."""

    writer       = None
    total_rows   = 0
    matched_rows = 0
    chunks_done  = 0
    t0           = time.time()

    log.info(f"Streaming {trades_path} ...")
    log.info(f"  Chunk size: {CHUNK_SIZE:,} rows")

    with lzma.open(trades_path, mode="rt", encoding="utf-8") as f:
        reader = pd.read_csv(
            f,
            names=COLS,
            dtype=DTYPES,
            chunksize=CHUNK_SIZE,
            header=0,
            on_bad_lines="skip",
        )

        for chunk in reader:
            chunks_done += 1
            total_rows  += len(chunk)

            # Filter rows where either asset ID is in whitelist
            mask = (
                chunk["makerAssetId"].isin(whitelist) |
                chunk["takerAssetId"].isin(whitelist)
            )

            if chunks_done % 50 == 0:
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

            # Identify which asset is the outcome token
            matched["maker_is_token"] = matched["makerAssetId"].isin(whitelist)
            matched["token_id"] = np.where(
                matched["maker_is_token"],
                matched["makerAssetId"],
                matched["takerAssetId"]
            )

            # Enrich with token metadata
            matched["market_id"]   = matched["token_id"].map(
                lambda t: token_meta.get(t, {}).get("market_id"))
            matched["market_type"] = matched["token_id"].map(
                lambda t: token_meta.get(t, {}).get("market_type"))
            matched["question"]    = matched["token_id"].map(
                lambda t: token_meta.get(t, {}).get("question"))
            matched["side"]        = matched["token_id"].map(
                lambda t: token_meta.get(t, {}).get("side"))
            matched["is_ingame"]   = matched["token_id"].map(
                lambda t: token_meta.get(t, {}).get("is_ingame", False))
            matched["is_half"]     = matched["token_id"].map(
                lambda t: token_meta.get(t, {}).get("is_half", False))

            # Parse timestamp
            matched["timestamp"] = pd.to_datetime(
                matched["timestamp"], unit="s", utc=True, errors="coerce"
            )

            # Convert amounts to float before price calculation
            matched["makerAmountFilled"] = matched["makerAmountFilled"].astype(float)
            matched["takerAmountFilled"] = matched["takerAmountFilled"].astype(float)

            # Compute raw price (USDC per token)
            matched["price_raw"] = np.vectorize(compute_price)(
                matched["makerAmountFilled"].values,
                matched["takerAmountFilled"].values,
                matched["maker_is_token"].values,
            )

            # ── Normalize price to always represent P(YES/away team wins) ──
            # YES token trades: price_raw = P(away wins) → use directly
            # NO token trades:  price_raw = P(home wins) → flip to P(away wins)
            matched["price_usdc"] = np.where(
                matched["side"] == "YES",
                matched["price_raw"],
                1.0 - matched["price_raw"]
            )

            # Clip to valid probability range
            matched["price_usdc"] = matched["price_usdc"].clip(0.001, 0.999)

            # Drop helper columns
            matched = matched.drop(columns=["maker_is_token", "price_raw"])

            # Convert amounts to USDC float (divide by 1M decimals)
            matched["makerAmountFilled"] = matched["makerAmountFilled"] / 1_000_000
            matched["takerAmountFilled"] = matched["takerAmountFilled"] / 1_000_000

            matched_rows += len(matched)

            # Write to parquet incrementally
            table = pa.Table.from_pandas(matched, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    OUTPUT_FILE, table.schema, compression="snappy"
                )
            writer.write_table(table)

    if writer:
        writer.close()

    elapsed = time.time() - t0
    log.info(f"Done streaming in {elapsed:.0f}s")
    log.info(f"  Total rows scanned : {total_rows:,}")
    log.info(f"  Matched rows       : {matched_rows:,}")
    log.info(f"  Saved → {OUTPUT_FILE}")


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    whitelist, token_meta = build_token_whitelist(METADATA_FILE)

    if len(whitelist) < 100:
        log.error(f"Whitelist too small ({len(whitelist)}) — aborting.")
        return

    stream_filter(TRADES_FILE, whitelist, token_meta)

    # Memory-safe dedup
    log.info("Deduplicating output...")
    TEMP_FILE = OUTPUT_FILE.with_suffix(".tmp.parquet")
    df = pd.read_parquet(OUTPUT_FILE)
    before = len(df)
    df = df.drop_duplicates(subset="transactionHash", keep="first")
    log.info(f"  Dropped {before - len(df):,} dupes, {len(df):,} remain")
    df.to_parquet(TEMP_FILE, index=False, compression="snappy")
    del df
    gc.collect()
    OUTPUT_FILE.unlink(missing_ok=True)
    TEMP_FILE.rename(OUTPUT_FILE)
    log.info(f"  Saved → {OUTPUT_FILE}")

    # Quick validation
    log.info("Validating...")
    sample = pd.read_parquet(OUTPUT_FILE, columns=["price_usdc", "side"])
    log.info(f"  Price range: {sample['price_usdc'].min():.4f} - {sample['price_usdc'].max():.4f}")
    log.info(f"  Price mean : {sample['price_usdc'].mean():.4f}")
    yes_prices = sample[sample["side"] == "YES"]["price_usdc"]
    no_prices  = sample[sample["side"] == "NO"]["price_usdc"]
    log.info(f"  YES avg price: {yes_prices.mean():.4f} (away team prob)")
    log.info(f"  NO avg price : {no_prices.mean():.4f} (away team prob, normalized)")
    log.info(f"  Expected: both should be ~0.45 if home teams win 55% of games")


if __name__ == "__main__":
    main()
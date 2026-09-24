"""
nba_edge_scanner.py
Scans nba_trades_features.parquet to find high-edge divergence conditions.

The key insight: move1..move20 are already computed forward price moves.
For a SHORT signal (poly above ESPN), profit = -move_N (price falls back).
For a LONG signal (poly below ESPN), profit = +move_N (price rises back).

We scan systematically across:
  - divergence magnitude buckets
  - time remaining buckets
  - score differential buckets
  - divergence direction (above vs below ESPN)
  - widening vs stable divergence (momentum filter)

Output: ranked table of (condition → avg edge) to inform new signal design.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR  = Path(r"C:\Users\24GHi\PycharmProjects\PythonProject2\poly_data\processed")
FEATURES  = DATA_DIR / "nba_trades_features.parquet"

# Cost floor — raw edge must exceed this to be viable live
# Based on: spread=0.005*2 + slippage=0.001 + avg_taker_fee≈0.005
COST_FLOOR = 0.011

# Forward horizon to measure edge (in trades, not seconds)
# move5 = price move over next 5 trades ~ 25-50 seconds at typical trade frequency
HORIZON = 5


def load_data() -> pd.DataFrame:
    print("Loading nba_trades_features.parquet...")
    df = pd.read_parquet(FEATURES)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # filter to in-game only
    df = df[df["in_game"] == True].copy()
    print(f"  {len(df):,} in-game rows")

    # drop rows without forward move data
    move_col = f"move{HORIZON}"
    df = df[df[move_col].notna()].copy()
    print(f"  {len(df):,} rows with move{HORIZON} available\n")

    return df


def compute_directional_edge(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """
    For each row compute the expected edge given the divergence direction.
    If poly_espn_div > 0: poly is overpriced → SHORT → profit if price falls → edge = -move_N
    If poly_espn_div < 0: poly is underpriced → LONG → profit if price rises → edge = +move_N
    """
    move_col = f"move{horizon}"
    df = df.copy()
    df["direction"]       = np.where(df["poly_espn_div"] > 0, -1, 1)
    df["directional_edge"] = df["direction"] * df[move_col]
    df["above_espn"]      = df["poly_espn_div"] > 0
    return df


def make_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["div_bucket"] = pd.cut(
        df["abs_espn_div"],
        bins=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 1.0],
        labels=["0.05-0.10","0.10-0.15","0.15-0.20","0.20-0.25",
                "0.25-0.30","0.30-0.35","0.35-0.40","0.40-0.50","0.50+"]
    )
    df["secs_bucket"] = pd.cut(
        df["secs_remaining"],
        bins=[0, 60, 120, 180, 300, 600, 1200, 9999],
        labels=["0-60s","60-120s","120-180s","180-300s",
                "300-600s","600-1200s","1200s+"]
    )
    df["score_bucket"] = pd.cut(
        df["abs_score_diff"],
        bins=[0, 3, 7, 12, 20, 999],
        labels=["0-3","4-7","8-12","13-20","20+"]
    )
    df["period_bucket"] = df["period"].clip(1, 5).map(
        {1:"Q1",2:"Q2",3:"Q3",4:"Q4",5:"OT"}
    )
    return df


def scan_pairwise(df: pd.DataFrame, col_a: str, col_b: str,
                  min_n: int = 30) -> pd.DataFrame:
    """Group by two bucket columns and aggregate edge stats."""
    grp = (
        df.groupby([col_a, col_b], observed=True)
        .agg(
            n              = ("directional_edge", "count"),
            avg_edge       = ("directional_edge", "mean"),
            win_rate       = ("directional_edge", lambda x: (x > 0).mean()),
            std_edge       = ("directional_edge", "std"),
            pct25          = ("directional_edge", lambda x: x.quantile(0.25)),
            pct75          = ("directional_edge", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )
    grp = grp[grp["n"] >= min_n].copy()
    grp["sharpe"]    = grp["avg_edge"] / grp["std_edge"].replace(0, np.nan)
    grp["above_cost"]= grp["avg_edge"] > COST_FLOOR
    return grp.sort_values("avg_edge", ascending=False)


def scan_triple(df: pd.DataFrame, col_a: str, col_b: str, col_c: str,
                min_n: int = 20) -> pd.DataFrame:
    grp = (
        df.groupby([col_a, col_b, col_c], observed=True)
        .agg(
            n        = ("directional_edge", "count"),
            avg_edge = ("directional_edge", "mean"),
            win_rate = ("directional_edge", lambda x: (x > 0).mean()),
            std_edge = ("directional_edge", "std"),
        )
        .reset_index()
    )
    grp = grp[grp["n"] >= min_n].copy()
    grp["sharpe"] = grp["avg_edge"] / grp["std_edge"].replace(0, np.nan)
    return grp.sort_values("avg_edge", ascending=False)


def widening_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a flag indicating whether the divergence was widening
    (increasing in magnitude) over the last 2 observations per market.
    Uses the signed divergence momentum columns already in the data.
    """
    df = df.copy()
    # poly_mom_60s > 0 means price rising; for short signals (above ESPN)
    # we want price falling (mom < 0 = divergence widening further)
    # for long signals (below ESPN) we want price rising (mom > 0)
    # widening = divergence moving further from fair value = momentum aligns with direction
    df["div_widening"] = (
        (df["above_espn"] & (df["poly_mom_60s"] < -0.005)) |
        (~df["above_espn"] & (df["poly_mom_60s"] > 0.005))
    )
    return df


def print_sep(char="=", w=80):
    print(char * w)


def print_table(df: pd.DataFrame, title: str, top_n: int = 20) -> None:
    print_sep()
    print(title)
    print_sep()
    df_show = df.head(top_n).copy()
    df_show["avg_edge"] = df_show["avg_edge"].map(lambda x: f"{x:+.4f}")
    df_show["win_rate"] = df_show["win_rate"].map(lambda x: f"{x*100:.1f}%")
    df_show["sharpe"]   = df_show["sharpe"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else "n/a"
    )
    if "above_cost" in df_show.columns:
        df_show["viable"] = df_show["above_cost"].map(
            lambda x: "YES" if x else "no"
        )
        df_show = df_show.drop(columns=["above_cost","std_edge",
                                         "pct25","pct75"], errors="ignore")
    print(df_show.to_string(index=False))
    print()


def propose_signals(df_full: pd.DataFrame,
                    div_secs: pd.DataFrame,
                    div_secs_score: pd.DataFrame) -> None:
    print_sep()
    print("PROPOSED SIGNAL CONDITIONS  (raw edge > cost floor)")
    print_sep()
    viable = div_secs[div_secs["avg_edge"].astype(float) > COST_FLOOR].copy() \
             if isinstance(div_secs["avg_edge"].iloc[0], str) \
             else div_secs[div_secs["avg_edge"] > COST_FLOOR].copy()

    if len(viable) == 0:
        print("  No pairwise conditions exceed cost floor. "
              "Check triple scan for higher-specificity rules.\n")
        return

    for _, row in viable.iterrows():
        div_b  = row["div_bucket"]
        secs_b = row["secs_bucket"]
        n      = int(row["n"])
        edge   = float(row["avg_edge"]) if not isinstance(row["avg_edge"], str) \
                 else float(row["avg_edge"])
        wr     = row["win_rate"]
        sharpe = row["sharpe"]

        # map bucket labels back to numeric conditions
        div_lo = float(str(div_b).split("-")[0].replace("0.","0.").split("+")[0])

        print(f"  abs_espn_div >= {div_lo:.2f}  AND  "
              f"secs_remaining in {secs_b}")
        print(f"    n={n}  avg_edge={edge:+.4f}  "
              f"win_rate={wr}  sharpe={sharpe}")
        print(f"    viable above cost floor ({COST_FLOOR:.3f}): YES")
        print()


def main():
    df = load_data()
    df = compute_directional_edge(df)
    df = make_buckets(df)
    df = widening_filter(df)

    # ── 1. Overall edge by divergence bucket ──────────────────
    div_only = (
        df.groupby("div_bucket", observed=True)
        .agg(
            n        = ("directional_edge", "count"),
            avg_edge = ("directional_edge", "mean"),
            win_rate = ("directional_edge", lambda x: (x > 0).mean()),
            std_edge = ("directional_edge", "std"),
        )
        .reset_index()
    )
    div_only["sharpe"]    = div_only["avg_edge"] / div_only["std_edge"]
    div_only["above_cost"]= div_only["avg_edge"] > COST_FLOOR
    div_only = div_only.sort_values("avg_edge", ascending=False)
    print_table(div_only, f"EDGE BY DIVERGENCE BUCKET  (horizon=move{HORIZON})")

    # ── 2. Divergence × time remaining ────────────────────────
    div_secs = scan_pairwise(df, "div_bucket", "secs_bucket")
    print_table(div_secs,
                f"EDGE BY DIVERGENCE × TIME REMAINING  (top 20, n≥30)")

    # ── 3. Divergence × time × score ──────────────────────────
    div_secs_score = scan_triple(df, "div_bucket", "secs_bucket", "score_bucket")
    print_table(div_secs_score,
                "EDGE BY DIVERGENCE × TIME × SCORE DIFF  (top 20, n≥20)")

    # ── 4. Divergence × period ────────────────────────────────
    div_period = scan_pairwise(df, "div_bucket", "period_bucket")
    print_table(div_period,
                "EDGE BY DIVERGENCE × PERIOD  (top 20, n≥30)")

    # ── 5. Widening filter effect ─────────────────────────────
    print_sep()
    print("WIDENING DIVERGENCE FILTER EFFECT")
    print_sep()
    for widening in [True, False]:
        sub  = df[df["div_widening"] == widening]
        n    = len(sub)
        edge = sub["directional_edge"].mean()
        wr   = (sub["directional_edge"] > 0).mean()
        label= "Widening" if widening else "Stable/narrowing"
        print(f"  {label:<20} n={n:>7,}  avg_edge={edge:+.4f}  "
              f"win_rate={wr*100:.1f}%")
    print()

    # ── 6. Above vs below ESPN ────────────────────────────────
    print_sep()
    print("ABOVE VS BELOW ESPN — does direction matter?")
    print_sep()
    for above in [True, False]:
        sub  = df[df["above_espn"] == above]
        n    = len(sub)
        edge = sub["directional_edge"].mean()
        wr   = (sub["directional_edge"] > 0).mean()
        label= "Poly ABOVE ESPN (short)" if above else "Poly BELOW ESPN (long)"
        print(f"  {label:<35} n={n:>7,}  avg_edge={edge:+.4f}  "
              f"win_rate={wr*100:.1f}%")
    print()

    # ── 7. Horizon sensitivity ────────────────────────────────
    print_sep()
    print("EDGE BY FORWARD HORIZON  (all in-game, abs_div > 0.20)")
    print_sep()
    df_high = df[df["abs_espn_div"] > 0.20].copy()
    print(f"  {'Horizon':<10} {'n':>8} {'avg_edge':>10} {'win_rate':>10}")
    print("  " + "-" * 42)
    for h in [1, 2, 3, 5, 10, 20]:
        col = f"move{h}"
        if col not in df_high.columns:
            continue
        sub = df_high[df_high[col].notna()].copy()
        sub["edge_h"] = sub["direction"] * sub[col]
        n    = len(sub)
        edge = sub["edge_h"].mean()
        wr   = (sub["edge_h"] > 0).mean()
        print(f"  move{h:<6} {n:>8,} {edge:>+10.4f} {wr*100:>9.1f}%")
    print()

    # ── 8. Sharp money filter ─────────────────────────────────
    print_sep()
    print("SHARP MONEY FILTER  (abs_div > 0.20)")
    print_sep()
    df_high = df[df["abs_espn_div"] > 0.20].copy()
    for sharp in [1, 0]:
        sub   = df_high[df_high["maker_is_sharp"] == sharp]
        n     = len(sub)
        edge  = sub["directional_edge"].mean()
        wr    = (sub["directional_edge"] > 0).mean()
        label = "Sharp maker" if sharp else "Non-sharp maker"
        print(f"  {label:<20} n={n:>6,}  avg_edge={edge:+.4f}  "
              f"win_rate={wr*100:.1f}%")
    print()

    # ── 9. Proposed signals ───────────────────────────────────
    propose_signals(df, div_secs, div_secs_score)

    # ── 10. Save top rules ────────────────────────────────────
    out = DATA_DIR / "edge_scan_results.parquet"
    div_secs_score.to_parquet(out, index=False)
    print(f"Full triple-scan results saved → {out}")


if __name__ == "__main__":
    main()
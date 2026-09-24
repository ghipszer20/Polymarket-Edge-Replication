"""
nba_live_bot.py
Reads nba_live_bot_trades.csv and produces per-signal bootstrap
confidence analysis with cost-adjusted restatement.

Cost model matches nba_signal_analysis.py:
  Entry : maker limit order = 0 fee
  Exit  : dynamic sports taker fee — peaks 0.75% at p=0.50
  Spread: PAPER_SPREAD_HALF * 2 (round-trip)
  Slippage: PAPER_SLIPPAGE on entry

Old trades (before cost model was added) are retroactively restated
using the exit_price column to compute the correct dynamic fee.
"""

import numpy as np
import pandas as pd
from pathlib import Path

TRADES_PATH = Path(
    r"C:\Users\24GHi\PycharmProjects\PythonProject2\nba_live_bot_trades.csv"
)
N_BOOTSTRAP = 10000
CONFIDENCE  = 0.95
ALPHA       = 1 - CONFIDENCE

COL_SIGNAL  = "signal"
COL_PNL     = "pnl"
COL_EXIT    = "exit_reason"

CUT_SIGNALS = {

    "yes_leading_10_with_600s",
    "poly_below_espn_20_late_300s",
    "poly_above_espn_20_late_300s",
    "poly_above_espn_10_late_300s",
    "poly_above_espn_20_late_600s",
    "poly_below_espn_20_late_600s",
    "no_leading_5_with_300s",
    "phase_final_mom_up",
    "level_below_mid_mom_up_5",
}

# Must match nba_signal_analysis.py
PAPER_SPREAD_HALF = 0.005
PAPER_SLIPPAGE    = 0.001
SPORTS_PEAK_FEE   = 0.0075

# Cost scenarios for sensitivity table
COST_SCENARIOS = {
    "optimistic":  {"spread_half": 0.003, "slippage": 0.001, "peak_fee": 0.0044},
    "base":        {"spread_half": 0.005, "slippage": 0.001, "peak_fee": 0.0075},
    "pessimistic": {"spread_half": 0.010, "slippage": 0.002, "peak_fee": 0.0075},
}


def sports_taker_fee(price: float,
                     peak_fee: float = SPORTS_PEAK_FEE) -> float:
    """Dynamic Polymarket taker fee for sports markets."""
    p = max(0.001, min(0.999, float(price)))
    return peak_fee * p * (1.0 - p) / 0.25


def total_cost_per_unit(exit_price: float,
                        spread_half: float = PAPER_SPREAD_HALF,
                        slippage: float    = PAPER_SLIPPAGE,
                        peak_fee: float    = SPORTS_PEAK_FEE) -> float:
    return (sports_taker_fee(exit_price, peak_fee)
            + spread_half * 2
            + slippage)


def load_trades(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    print(f"  Columns found: {list(df.columns)}\n")

    pnl_candidates = ["pnl", "pnl_usdc", "trade_pnl", "net_pnl", "profit",
                      "profit_loss", "p&l", "pl", "return", "pnl_usd"]
    pnl_col = next((c for c in pnl_candidates if c in df.columns), None)
    if pnl_col is None:
        for c in df.select_dtypes(include="number").columns:
            vals = df[c].dropna()
            if set(vals.unique()).issubset({-1, 0, 1}):
                continue
            if vals.min() < 0:
                pnl_col = c
                print(f"  WARNING: guessing PnL column = '{c}'")
                break
    if pnl_col is None:
        raise ValueError(f"No PnL column found. Columns: {list(df.columns)}")

    sig_col = next(
        (c for c in ["signal","signal_name","strategy","signal_type"]
         if c in df.columns), None
    )
    if sig_col is None:
        raise ValueError(f"No signal column found. Columns: {list(df.columns)}")

    exit_col = next(
        (c for c in ["exit_reason","exit_method","exit","exit_type"]
         if c in df.columns), None
    )

    rename = {pnl_col: COL_PNL, sig_col: COL_SIGNAL}
    if exit_col:
        rename[exit_col] = COL_EXIT
    df = df.rename(columns=rename)

    df[COL_PNL] = pd.to_numeric(df[COL_PNL], errors="coerce")
    df = df.dropna(subset=[COL_PNL, COL_SIGNAL])

    df_all    = df.copy()
    df_active = df[~df[COL_SIGNAL].isin(CUT_SIGNALS)].copy()
    n_cut     = len(df) - len(df_active)
    if n_cut > 0:
        print(f"  Excluded {n_cut} trades from cut signals: "
              f"{sorted(CUT_SIGNALS)}\n")
    return df_all, df_active


def apply_cost_restatement(df: pd.DataFrame,
                            spread_half: float = PAPER_SPREAD_HALF,
                            slippage: float    = PAPER_SLIPPAGE,
                            peak_fee: float    = SPORTS_PEAK_FEE) -> pd.DataFrame:
    """
    Retroactively restate PnL using the correct dynamic cost model.
    Trades already carrying cost_unit use that value directly.
    Old trades (no cost_unit) are restated using exit_price to compute
    the dynamic taker fee, then subtract the full round-trip cost.

    Old trades also had TAKER_FEE=0.001 baked in (flat, incorrect).
    We add that back first, then subtract the correct dynamic cost.
    """
    df = df.copy()

    has_cost  = "cost_unit" in df.columns
    has_exit  = "exit_price" in df.columns
    has_size  = "size_usdc" in df.columns

    if has_cost:
        old_mask = df["cost_unit"].isna()
    else:
        old_mask = pd.Series(True, index=df.index)

    n_old = old_mask.sum()
    if n_old == 0:
        print("  All trades have full cost model — no restatement needed.\n")
        return df

    if not has_exit or not has_size:
        print(f"  WARNING: cannot restate {n_old} old trades — "
              f"exit_price or size_usdc column missing.\n")
        return df

    old = df[old_mask].copy()

    # 1. Add back the flat 0.001 taker fee that was already baked in
    df.loc[old_mask, COL_PNL] += 0.001 * old["size_usdc"]

    # 2. Subtract the correct dynamic cost
    correct_cost = old["exit_price"].apply(
        lambda p: total_cost_per_unit(p, spread_half, slippage, peak_fee)
    )
    df.loc[old_mask, COL_PNL] -= correct_cost * old["size_usdc"]

    total_adj = (
        (correct_cost - 0.001) * old["size_usdc"]
    ).sum()

    print(f"  Restated {n_old} old-format trades "
          f"(net adjustment: ${-total_adj:.2f})\n")
    return df


def bootstrap_mean(pnls: np.ndarray,
                   n_resamples: int = N_BOOTSTRAP) -> dict:
    n   = len(pnls)
    obs = pnls.mean()
    if n < 5:
        return dict(n=n, mean=obs, ci_low=np.nan, ci_high=np.nan,
                    p_value=np.nan, significant=False,
                    power="insufficient data")

    resamples = np.random.choice(pnls, size=(n_resamples, n), replace=True)
    means     = resamples.mean(axis=1)
    ci_low, ci_high = np.percentile(
        means, [ALPHA / 2 * 100, (1 - ALPHA / 2) * 100]
    )
    shifted = means - obs
    p_value = (np.abs(shifted) >= np.abs(obs)).mean()
    significant = (p_value < ALPHA) and (
        (obs > 0 and ci_low > 0) or (obs < 0 and ci_high < 0)
    )
    if   n >= 200: power = "high"
    elif n >= 80:  power = "moderate"
    elif n >= 30:  power = "building"
    else:          power = "low"

    return dict(n=n, mean=obs, ci_low=ci_low, ci_high=ci_high,
                p_value=p_value, significant=significant, power=power)


def summarise_signal(df: pd.DataFrame, signal: str) -> dict:
    trades = df.loc[df[COL_SIGNAL] == signal, COL_PNL].values
    wins   = (trades > 0).sum()
    boot   = bootstrap_mean(trades)
    return dict(
        signal     = signal,
        n          = boot["n"],
        win_rate   = wins / len(trades) if len(trades) else 0,
        avg_pnl    = boot["mean"],
        total_pnl  = trades.sum(),
        ci_low     = boot["ci_low"],
        ci_high    = boot["ci_high"],
        p_value    = boot["p_value"],
        significant= boot["significant"],
        power      = boot["power"],
    )


def summarise_exit(df: pd.DataFrame, exit_method: str) -> dict:
    trades = df.loc[df[COL_EXIT] == exit_method, COL_PNL].values
    wins   = (trades > 0).sum()
    return dict(
        exit      = exit_method,
        n         = len(trades),
        win_rate  = wins / len(trades) if len(trades) else 0,
        avg_pnl   = trades.mean() if len(trades) else 0,
        total_pnl = trades.sum(),
    )


def sep(char="=", width=72):
    print(char * width)


def fp(v):
    if np.isnan(v): return "  n/a  "
    return f"${v:+.4f}"


def fci(lo, hi):
    if np.isnan(lo): return "       n/a       "
    return f"[{lo:+.4f}, {hi:+.4f}]"


def verdict(row):
    if row["n"] < 10:
        return "⚠  too few trades"
    if not row["significant"]:
        return ("~  positive but unconfirmed" if row["avg_pnl"] > 0
                else "~  negative but unconfirmed" if row["avg_pnl"] < 0
                else "~  no edge detected")
    return "✓  confirmed edge" if row["avg_pnl"] > 0 else "✗  confirmed drag — cut it"


def estimate_needed(row) -> int:
    p = row["p_value"]
    n = row["n"]
    if np.isnan(p): return n * 5
    if p <= 0.05:   return 0
    return max(10, int(n * (p / 0.05 - 1)))


def print_signal_table(rows: list[dict], label: str) -> None:
    sep()
    print(label)
    sep()
    print(f"  {'Signal':<38} {'n':>5}  {'WR':>6}  {'AvgPnL':>9}  "
          f"{'95% CI':^21}  {'p-val':>6}  {'Data':>8}  Verdict")
    print("  " + "-" * 115)
    for r in rows:
        p_str = f"{r['p_value']:.3f}" if not np.isnan(r["p_value"]) else "  nan"
        print(
            f"  {r['signal']:<38} "
            f"{r['n']:>5}  "
            f"{r['win_rate']*100:>5.1f}%  "
            f"{fp(r['avg_pnl']):>9}  "
            f"{fci(r['ci_low'], r['ci_high']):^23}  "
            f"{p_str:>6}  "
            f"{r['power']:>8}  "
            f"{verdict(r)}"
        )


def print_breakeven_table(df: pd.DataFrame) -> None:
    """
    For each signal show avg PnL under three cost scenarios and the
    half-spread at which the signal breaks even.
    """
    sep()
    print("COST SENSITIVITY — avg PnL per trade under each cost scenario")
    sep()

    # pre-compute fee examples for header
    for label, sc in COST_SCENARIOS.items():
        eg = total_cost_per_unit(0.75, sc["spread_half"],
                                 sc["slippage"], sc["peak_fee"])
        print(f"  {label:<12}: spread={sc['spread_half']*2:.3f}  "
              f"slippage={sc['slippage']:.3f}  "
              f"peak_fee={sc['peak_fee']*100:.2f}%  "
              f"→ total at p=0.75: {eg:.4f}/unit")
    print()
    print(f"  {'Signal':<38} {'Raw':>10} {'Optimistic':>12} "
          f"{'Base':>10} {'Pessimistic':>13} {'Breakeven half-spread':>22}")
    print("  " + "-" * 110)

    size_col = "size_usdc" if "size_usdc" in df.columns else None
    exit_col = "exit_price" if "exit_price" in df.columns else None

    for signal in df[COL_SIGNAL].value_counts().index:
        sub     = df[df[COL_SIGNAL] == signal]
        raw_avg = sub[COL_PNL].mean()
        n       = len(sub)
        avg_sz  = sub[size_col].mean() if size_col else 10.0

        scenario_results = {}
        for label, sc in COST_SCENARIOS.items():
            if exit_col:
                cost_per_trade = sub["exit_price"].apply(
                    lambda p: total_cost_per_unit(
                        p, sc["spread_half"], sc["slippage"], sc["peak_fee"]
                    ) * avg_sz
                ).mean()
            else:
                cost_per_trade = total_cost_per_unit(
                    0.75, sc["spread_half"], sc["slippage"], sc["peak_fee"]
                ) * avg_sz
            scenario_results[label] = raw_avg - cost_per_trade

        # breakeven: solve for spread_half where adj_avg = 0
        # raw_avg - (2*spread + slippage + dyn_fee) * avg_sz = 0
        # spread_half = (raw_avg/avg_sz - slippage - avg_dyn_fee) / 2
        base_sc     = COST_SCENARIOS["base"]
        avg_dyn_fee = (sub["exit_price"].apply(
            lambda p: sports_taker_fee(p, base_sc["peak_fee"])
        ).mean() if exit_col else sports_taker_fee(0.75, base_sc["peak_fee"]))

        numerator   = raw_avg / avg_sz - base_sc["slippage"] - avg_dyn_fee
        be_spread   = numerator / 2

        if be_spread <= 0:
            be_str = "n/a (raw≤0)"
        elif be_spread > 0.05:
            be_str = f"{be_spread:.4f} ({be_spread*100:.1f}¢)  very robust"
        elif be_spread > 0.01:
            be_str = f"{be_spread:.4f} ({be_spread*100:.1f}¢)  viable"
        elif be_spread > 0.005:
            be_str = f"{be_spread:.4f} ({be_spread*100:.1f}¢)  marginal"
        else:
            be_str = f"{be_spread:.4f} ({be_spread*100:.1f}¢)  unlikely live"

        print(
            f"  {signal:<38} "
            f"{fp(raw_avg):>10} "
            f"{fp(scenario_results['optimistic']):>12} "
            f"{fp(scenario_results['base']):>10} "
            f"{fp(scenario_results['pessimistic']):>13}  "
            f"{be_str}"
        )
    print()


def main():
    np.random.seed(42)

    print(f"\nLoading trades from {TRADES_PATH}...")
    df_all, df_active = load_trades(TRADES_PATH)

    print(f"  {len(df_all)} total trades  |  "
          f"{len(df_active)} active-signal trades across "
          f"{df_active[COL_SIGNAL].nunique()} signals\n")

    # ── Section 1: as-logged ──────────────────────────────────
    total_all  = df_all[COL_PNL].sum()
    total_act  = df_active[COL_PNL].sum()
    n          = len(df_active)
    wr         = (df_active[COL_PNL] > 0).mean()
    avg        = df_active[COL_PNL].mean()
    arr        = df_active[COL_PNL].values
    sharpe     = (arr.mean() / arr.std()) if arr.std() > 0 else 0

    sep()
    print("PAPER TRADING SUMMARY  (as logged)")
    sep()
    print(f"  [ALL]    Total PnL : {fp(total_all)}  "
          f"Final BK : {fp(1000 + total_all)}")
    print(f"  [ACTIVE] Trades    : {n}")
    print(f"           Win rate  : {wr*100:.1f}%")
    print(f"           Avg PnL   : {fp(avg)}")
    print(f"           Total PnL : {fp(total_act)}")
    print(f"           Return    : {total_act/1000*100:+.2f}%")
    print(f"           Sharpe    : {sharpe:.4f}")

    sigs_logged = df_active[COL_SIGNAL].value_counts().index.tolist()
    rows_logged = sorted(
        [summarise_signal(df_active, s) for s in sigs_logged],
        key=lambda r: r["total_pnl"], reverse=True
    )
    print()
    print_signal_table(
        rows_logged,
        f"BY SIGNAL — as logged  ({int(CONFIDENCE*100)}% CI, "
        f"active signals only, cut: {sorted(CUT_SIGNALS)})"
    )

    # ── Section 2: cost-adjusted ──────────────────────────────
    print()
    sep()
    print("COST RESTATEMENT")
    sep()
    print(f"  Dynamic sports taker fee: peak {SPORTS_PEAK_FEE*100:.2f}% "
          f"at p=0.50  →  formula: {SPORTS_PEAK_FEE} × p × (1-p) / 0.25")
    print(f"  Spread (round-trip)     : {PAPER_SPREAD_HALF*2:.3f} "
          f"({PAPER_SPREAD_HALF*100:.1f}¢ each side)")
    print(f"  Slippage                : {PAPER_SLIPPAGE:.3f}")
    print(f"  Entry fee               : 0 (maker limit order)")
    print()
    eg_prices = [0.50, 0.65, 0.75, 0.85, 0.90]
    print("  Fee at typical exit prices:")
    for p in eg_prices:
        fee   = sports_taker_fee(p)
        total = total_cost_per_unit(p)
        print(f"    p={p:.2f}  taker={fee:.4f}  "
              f"total (incl spread+slip)={total:.4f}/unit")
    print()

    df_adj     = apply_cost_restatement(df_active.copy())
    df_all_adj = apply_cost_restatement(df_all.copy())

    total_adj_act = df_adj[COL_PNL].sum()
    total_adj_all = df_all_adj[COL_PNL].sum()
    wr_adj        = (df_adj[COL_PNL] > 0).mean()
    avg_adj       = df_adj[COL_PNL].mean()
    arr_adj       = df_adj[COL_PNL].values
    sharpe_adj    = (arr_adj.mean() / arr_adj.std()
                    if arr_adj.std() > 0 else 0)

    sep()
    print("PAPER TRADING SUMMARY  (cost-adjusted)")
    sep()
    print(f"  [ALL]    Total PnL : {fp(total_adj_all)}  "
          f"Final BK : {fp(1000 + total_adj_all)}")
    print(f"  [ACTIVE] Trades    : {n}")
    print(f"           Win rate  : {wr_adj*100:.1f}%")
    print(f"           Avg PnL   : {fp(avg_adj)}")
    print(f"           Total PnL : {fp(total_adj_act)}")
    print(f"           Return    : {total_adj_act/1000*100:+.2f}%")
    print(f"           Sharpe    : {sharpe_adj:.4f}")
    delta = total_adj_act - total_act
    pct   = delta / abs(total_act) * 100 if total_act != 0 else 0
    print(f"\n  Cost impact vs as-logged: {fp(delta)}  ({pct:+.1f}%)")

    sigs_adj  = df_adj[COL_SIGNAL].value_counts().index.tolist()
    rows_adj  = sorted(
        [summarise_signal(df_adj, s) for s in sigs_adj],
        key=lambda r: r["total_pnl"], reverse=True
    )
    print()
    print_signal_table(
        rows_adj,
        f"BY SIGNAL — cost-adjusted  ({int(CONFIDENCE*100)}% CI)"
    )

    # ── Section 3: delta table ────────────────────────────────
    print()
    sep()
    print("SIGNAL IMPACT OF COST ADJUSTMENT")
    sep()
    print(f"  {'Signal':<38} {'Logged avg':>12}  "
          f"{'Adjusted avg':>14}  {'Delta':>10}  Note")
    print("  " + "-" * 85)
    lmap = {r["signal"]: r for r in rows_logged}
    amap = {r["signal"]: r for r in rows_adj}
    for sig in sorted(lmap):
        l = lmap[sig]
        a = amap.get(sig)
        if not a:
            continue
        d     = a["avg_pnl"] - l["avg_pnl"]
        flip  = (l["avg_pnl"] > 0) != (a["avg_pnl"] > 0)
        note  = "*** SIGN FLIP" if flip else ""
        print(f"  {sig:<38} {fp(l['avg_pnl']):>12}  "
              f"{fp(a['avg_pnl']):>14}  {fp(d):>10}  {note}")

    # ── Section 4: exit method ────────────────────────────────
    if COL_EXIT in df_adj.columns:
        print()
        sep()
        print("BY EXIT METHOD  (cost-adjusted, active signals only)")
        sep()
        print(f"  {'Exit':<25} {'n':>5}  {'WR':>6}  "
              f"{'AvgPnL':>9}  {'TotalPnL':>10}")
        print("  " + "-" * 62)
        for e in df_adj[COL_EXIT].value_counts().index:
            r = summarise_exit(df_adj, e)
            print(f"  {r['exit']:<25} {r['n']:>5}  "
                  f"{r['win_rate']*100:>5.1f}%  "
                  f"{fp(r['avg_pnl']):>9}  "
                  f"{fp(r['total_pnl']):>10}")

    # ── Section 5: actionable summary ────────────────────────
    print()
    sep()
    print("ACTIONABLE SUMMARY  (cost-adjusted)")
    sep()

    confirmed_pos = [r for r in rows_adj if r["significant"] and r["avg_pnl"] > 0]
    confirmed_neg = [r for r in rows_adj if r["significant"] and r["avg_pnl"] < 0]
    unconfirmed   = [r for r in rows_adj if not r["significant"] and r["n"] >= 10]
    thin          = [r for r in rows_adj if r["n"] < 10]

    if confirmed_pos:
        print("\n  KEEP (confirmed positive edge):")
        for r in confirmed_pos:
            print(f"    + {r['signal']}  "
                  f"avg={fp(r['avg_pnl'])}  "
                  f"CI={fci(r['ci_low'], r['ci_high'])}")
    if confirmed_neg:
        print("\n  CUT (confirmed drag):")
        for r in confirmed_neg:
            print(f"    - {r['signal']}  "
                  f"avg={fp(r['avg_pnl'])}  "
                  f"p={r['p_value']:.3f}")
    if unconfirmed:
        print("\n  WATCH (unconfirmed — keep collecting data):")
        for r in unconfirmed:
            needed    = estimate_needed(r)
            direction = "+" if r["avg_pnl"] >= 0 else "-"
            print(f"    {direction} {r['signal']}  "
                  f"avg={fp(r['avg_pnl'])}  "
                  f"p={r['p_value']:.3f}  "
                  f"need ~{needed} more trades")
    if thin:
        print("\n  TOO THIN (< 10 trades — no conclusion possible):")
        for r in thin:
            print(f"    . {r['signal']}  n={r['n']}  avg={fp(r['avg_pnl'])}")

    # ── Section 6: cost sensitivity ───────────────────────────
    print()
    print_breakeven_table(df_active)

    # ── Section 7: cut signal graveyard ──────────────────────
    sep()
    print("CUT SIGNALS  (historical record, cost-adjusted)")
    sep()
    print(f"  {'Signal':<38} {'n':>5}  {'WR':>6}  "
          f"{'AvgPnL':>9}  {'TotalPnL':>10}  Reason")
    print("  " + "-" * 90)
    for sig in sorted(CUT_SIGNALS):
        if sig in df_all_adj[COL_SIGNAL].values:
            sub    = df_all_adj[df_all_adj[COL_SIGNAL] == sig][COL_PNL]
            wr_cut = (sub > 0).mean()
            reason = ("structural drag — near-zero edge, high volume"
                      if sig == "yes_leading_10_with_600s"
                      else "confirmed negative avg PnL")
            print(f"  {sig:<38} {len(sub):>5}  {wr_cut*100:>5.1f}%  "
                  f"{fp(sub.mean()):>9}  "
                  f"{fp(sub.sum()):>10}  {reason}")
    print()


if __name__ == "__main__":
    main()
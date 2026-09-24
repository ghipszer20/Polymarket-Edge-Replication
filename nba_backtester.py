"""
nba_backtester.py
Simulates limit order maker strategy on confirmed signals.
Primary signal: poly_above_espn_20_late_300s
Uses Kelly-derived position sizing (2% of bankroll per trade).
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR  = Path(r"C:\Users\24GHi\PycharmProjects\PythonProject2")
FEATURES     = PROJECT_DIR / "poly_data" / "processed" / "nba_trades_features.parquet"
OUTPUT       = PROJECT_DIR / "poly_data" / "processed" / "backtest_results.parquet"

MAKER_FEE    = 0.000   # 0% maker fee on Polymarket
TAKER_FEE    = 0.001   # 0.1% taker fee on exit
CANCEL_AFTER = 30      # seconds before cancelling unfilled order
KELLY_FRAC   = 0.02    # 2% of bankroll per trade (half Kelly)
INITIAL_BK   = 1000.0  # starting bankroll in USDC
EXIT_TRADES  = 10      # exit after N trades

# ── Signal definitions ────────────────────────────────────────
SIGNALS = {
    "poly_above_espn_20_late_300s": {
        "condition":   "poly_espn_div > 0.20 AND secs_remaining < 300 AND secs_remaining > 30",
        "direction":   -1,
        "description": "Polymarket >20% above ESPN, <5min remaining"
    },
    "poly_below_espn_20_late_300s": {
        "condition":   "poly_espn_div < -0.20 AND secs_remaining < 300 AND secs_remaining > 30",
        "direction":   +1,
        "description": "Polymarket >20% below ESPN, <5min remaining"
    },
    "poly_above_espn_10_late_300s": {
        "condition":   "poly_espn_div > 0.10 AND secs_remaining < 300 AND secs_remaining > 30",
        "direction":   -1,
        "description": "Polymarket >10% above ESPN, <5min remaining"
    },
    "poly_above_espn_20_late_600s": {
        "condition":   "poly_espn_div > 0.20 AND secs_remaining < 600 AND secs_remaining > 60",
        "direction":   -1,
        "description": "Polymarket >20% above ESPN, <10min remaining"
    },
    "poly_below_espn_20_late_600s": {
        "condition":   "poly_espn_div < -0.20 AND secs_remaining < 600 AND secs_remaining > 60",
        "direction":   +1,
        "description": "Polymarket >20% below ESPN, <10min remaining"
    },
    "no_leading_5_with_300s": {
        "condition":   "score_diff < -5 AND secs_remaining < 300 AND secs_remaining > 30",
        "direction":   -1,
        "description": "NO team leading by 5+ with <5min remaining"
    },
    "yes_leading_10_with_600s": {
        "condition":   "score_diff > 10 AND secs_remaining < 600 AND secs_remaining > 60",
        "direction":   +1,
        "description": "YES team leading by 10+ with <10min remaining"
    },
    "phase_q4_mom_up_m5_s2": {
        "condition":   "game_pct_done BETWEEN 0.65 AND 0.85 AND move5 > 0.05 AND ABS(sigma5) > 2.0",
        "direction":   +1,
        "description": "Q4 momentum up 5%+ with 2+ sigma"
    },
    "phase_final_mom_up_m5_s1": {
        "condition":   "game_pct_done > 0.85 AND move5 > 0.05 AND ABS(sigma5) > 1.0",
        "direction":   +1,
        "description": "Late game momentum up 5%+ with 1+ sigma"
    },
    "espn_leading_poly_up": {
        "condition":   "espn_mom_60s > 0.05 AND poly_mom_60s < 0.02",
        "direction":   +1,
        "description": "ESPN momentum leads Polymarket up"
    },
    "espn_leading_poly_dn": {
        "condition":   "espn_mom_60s < -0.05 AND poly_mom_60s > -0.02",
        "direction":   -1,
        "description": "ESPN momentum leads Polymarket down"
    },
    "level_below_mid_mom_up_5": {
        "condition":   "price_usdc BETWEEN 0.30 AND 0.45 AND move5 > 0.05",
        "direction":   +1,
        "description": "Price 30-45%, momentum up 5%+"
    },
    "level_high_mom_dn_5": {
        "condition":   "price_usdc BETWEEN 0.70 AND 0.85 AND move5 < -0.05",
        "direction":   -1,
        "description": "Price 70-85%, momentum down 5%+"
    },
}


def run_equity_curve(filled_df: pd.DataFrame, signal_name: str,
                     year: int, year_label: str,
                     n_signals: int) -> dict:
    year_df   = filled_df[filled_df["yr"] == year].copy()
    n_trades  = len(year_df)
    if n_trades < 20:
        return None

    bankroll  = INITIAL_BK
    equity    = [bankroll]
    trade_log = []
    peak      = bankroll
    max_dd    = 0.0
    consec_losses = 0
    max_consec_losses = 0

    for _, trade in year_df.iterrows():
        position_size = bankroll * KELLY_FRAC
        pnl           = trade["ret10"] * position_size
        bankroll     += pnl
        equity.append(bankroll)
        peak          = max(peak, bankroll)
        drawdown      = (peak - bankroll) / peak
        max_dd        = max(max_dd, drawdown)

        if trade["ret10"] > 0:
            consec_losses = 0
        else:
            consec_losses += 1
            max_consec_losses = max(max_consec_losses, consec_losses)

        trade_log.append({
            "timestamp":    trade["timestamp"],
            "signal_price": trade["signal_price"],
            "ret10":        trade["ret10"],
            "ret20":        trade["ret20"],
            "pnl":          pnl,
            "bankroll":     bankroll,
            "drawdown":     drawdown,
        })

    trade_df  = pd.DataFrame(trade_log)
    n_wins    = (trade_df["ret10"] > 0).sum()
    win_rate  = n_wins / n_trades
    total_ret = (bankroll - INITIAL_BK) / INITIAL_BK
    avg_ret   = trade_df["ret10"].mean()
    std_ret   = trade_df["ret10"].std()
    sharpe    = (avg_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

    # Annualized return estimate
    # Assume ~1000 games/year, n_signals/n_games signals per game
    trades_per_year = n_trades * (365 / 365)  # scale if needed
    ann_ret = (1 + avg_ret) ** trades_per_year - 1

    return {
        "signal":             signal_name,
        "year":               year_label,
        "n_signals":          n_signals,
        "n_filled":           n_trades,
        "fill_rate":          round(n_trades / n_signals, 3) if n_signals > 0 else 0,
        "win_rate":           round(win_rate, 4),
        "avg_ret":            round(avg_ret, 6),
        "avg_ret20":          round(trade_df["ret20"].mean(), 6),
        "std_ret":            round(std_ret, 6),
        "total_return":       round(total_ret, 4),
        "final_bk":           round(bankroll, 2),
        "max_drawdown":       round(max_dd, 4),
        "max_consec_losses":  max_consec_losses,
        "sharpe":             round(sharpe, 3),
        "best_trade":         round(trade_df["ret10"].max(), 6),
        "worst_trade":        round(trade_df["ret10"].min(), 6),
        "p25_ret":            round(trade_df["ret10"].quantile(0.25), 6),
        "p75_ret":            round(trade_df["ret10"].quantile(0.75), 6),
    }


def main():
    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE VIEW f AS SELECT * FROM read_parquet('{FEATURES}')")

    # Check available columns
    cols = con.execute("SELECT column_name FROM information_schema.columns WHERE table_name='f'").df()
    available = set(cols["column_name"].tolist())
    print(f"Available feature columns: {len(available)}")

    all_results = []
    equity_curves = {}

    print("\nRunning backtests...")
    for signal_name, signal_cfg in SIGNALS.items():
        condition = signal_cfg["condition"]
        direction = signal_cfg["direction"]
        desc      = signal_cfg["description"]

        # Check all referenced columns exist
        import re
        col_refs = re.findall(r'\b([a-z][a-z0-9_]*)\b', condition)
        missing  = [c for c in col_refs if c not in available
                    and c not in ('and','or','not','between','abs','true','false')]
        if missing:
            print(f"\n  SKIP {signal_name}: missing columns {missing}")
            continue

        try:
            df = con.execute(f"""
                WITH signal AS (
                    SELECT
                        market_id,
                        timestamp,
                        price_usdc                  as signal_price,
                        poly_espn_div,
                        secs_remaining,
                        period,
                        score_diff,
                        YEAR(timestamp)             as yr,
                        LEAD(price_usdc, 1)  OVER w as entry_price,
                        EPOCH(LEAD(timestamp, 1) OVER w) -
                        EPOCH(timestamp)            as secs_to_fill,
                        LEAD(price_usdc, 5)  OVER w as exit5,
                        LEAD(price_usdc, 10) OVER w as exit10,
                        LEAD(price_usdc, 20) OVER w as exit20
                    FROM f
                    WHERE {condition}
                    AND price_usdc BETWEEN 0.05 AND 0.95
                    AND (
                        (period BETWEEN 1 AND 4 AND secs_remaining > 60)
                        OR (period > 4 AND secs_remaining > 30)
                    )
                    WINDOW w AS (
                        PARTITION BY market_id ORDER BY timestamp
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    )
                )
                SELECT * FROM signal
                WHERE entry_price IS NOT NULL
                AND exit10 IS NOT NULL
                AND exit20 IS NOT NULL
            """).df()
        except Exception as e:
            print(f"\n  ERROR {signal_name}: {e}")
            continue

        if len(df) < 50:
            print(f"\n  SKIP {signal_name}: only {len(df)} observations")
            continue

        # Simulate maker fill at signal price
        if direction == -1:
            df["filled"] = df["entry_price"] <= df["signal_price"]
            df["ret10"]  = -(df["exit10"] - df["signal_price"]) - TAKER_FEE
            df["ret20"]  = -(df["exit20"] - df["signal_price"]) - TAKER_FEE
            df["ret5"]   = -(df["exit5"]  - df["signal_price"]) - TAKER_FEE
        else:
            df["filled"] = df["entry_price"] >= df["signal_price"]
            df["ret10"]  = (df["exit10"] - df["signal_price"]) - TAKER_FEE
            df["ret20"]  = (df["exit20"] - df["signal_price"]) - TAKER_FEE
            df["ret5"]   = (df["exit5"]  - df["signal_price"]) - TAKER_FEE

        # Apply cancel rule
        df["filled"] = df["filled"] & (df["secs_to_fill"] <= CANCEL_AFTER)
        filled_df    = df[df["filled"]].copy()

        print(f"\n  {signal_name}")
        print(f"  {desc}")

        for year, year_label in [(2024, "IS"), (2025, "OOS")]:
            n_signals = len(df[df["yr"] == year])
            result    = run_equity_curve(
                filled_df, signal_name, year, year_label, n_signals
            )
            if result is None:
                continue
            all_results.append(result)
            equity_curves[f"{signal_name}_{year_label}"] = result

            print(f"    [{year_label} {year}] "
                  f"n={result['n_filled']:,}  "
                  f"fill={result['fill_rate']*100:.0f}%  "
                  f"wr={result['win_rate']*100:.1f}%  "
                  f"avg_ret={result['avg_ret']:.4f}  "
                  f"total={result['total_return']*100:.1f}%  "
                  f"BK=${result['final_bk']:.0f}  "
                  f"maxDD={result['max_drawdown']*100:.1f}%  "
                  f"sharpe={result['sharpe']:.2f}")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "="*100)
    print("SUMMARY — OUT OF SAMPLE 2025 ONLY")
    print("="*100)
    results_df = pd.DataFrame(all_results)
    oos = results_df[results_df["year"] == "OOS"].sort_values(
        "total_return", ascending=False
    )
    print(oos[[
        "signal", "n_filled", "fill_rate", "win_rate",
        "avg_ret", "total_return", "final_bk",
        "max_drawdown", "sharpe"
    ]].to_string(index=False))

    # ── Combined portfolio simulation ─────────────────────────
    print("\n" + "="*100)
    print("COMBINED PORTFOLIO (all OOS signals, equal weight, 2% each)")
    print("="*100)

    # Get all 2025 filled trades across all signals
    all_trades = []
    for signal_name, signal_cfg in SIGNALS.items():
        condition = signal_cfg["condition"]
        direction = signal_cfg["direction"]

        import re
        col_refs = re.findall(r'\b([a-z][a-z0-9_]*)\b', condition)
        missing  = [c for c in col_refs if c not in available
                    and c not in ('and','or','not','between','abs','true','false')]
        if missing:
            continue

        try:
            df = con.execute(f"""
                WITH signal AS (
                    SELECT
                        market_id,
                        timestamp,
                        price_usdc                  as signal_price,
                        YEAR(timestamp)             as yr,
                        LEAD(price_usdc, 1)  OVER w as entry_price,
                        EPOCH(LEAD(timestamp, 1) OVER w) -
                        EPOCH(timestamp)            as secs_to_fill,
                        LEAD(price_usdc, 10) OVER w as exit10
                    FROM f
                    WHERE {condition}
                    AND price_usdc BETWEEN 0.05 AND 0.95
                    AND (
                        (period BETWEEN 1 AND 4 AND secs_remaining > 60)
                        OR (period > 4 AND secs_remaining > 30)
                    )
                    WINDOW w AS (
                        PARTITION BY market_id ORDER BY timestamp
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    )
                )
                SELECT * FROM signal
                WHERE entry_price IS NOT NULL
                AND exit10 IS NOT NULL
                AND yr = 2025
            """).df()
        except:
            continue

        if len(df) < 50:
            continue

        if direction == -1:
            df["filled"] = df["entry_price"] <= df["signal_price"]
            df["ret10"]  = -(df["exit10"] - df["signal_price"]) - TAKER_FEE
        else:
            df["filled"] = df["entry_price"] >= df["signal_price"]
            df["ret10"]  = (df["exit10"] - df["signal_price"]) - TAKER_FEE

        df["filled"]     = df["filled"] & (df["secs_to_fill"] <= CANCEL_AFTER)
        df["signal_name"] = signal_name
        all_trades.append(df[df["filled"]][["timestamp","ret10","signal_name"]])

    if all_trades:
        portfolio_df = pd.concat(all_trades).sort_values("timestamp").reset_index(drop=True)
        bankroll     = INITIAL_BK
        peak         = bankroll
        max_dd       = 0.0
        equity       = []

        for _, trade in portfolio_df.iterrows():
            pos_size  = bankroll * KELLY_FRAC
            pnl       = trade["ret10"] * pos_size
            bankroll += pnl
            peak      = max(peak, bankroll)
            dd        = (peak - bankroll) / peak
            max_dd    = max(max_dd, dd)
            equity.append(bankroll)

        n       = len(portfolio_df)
        n_wins  = (portfolio_df["ret10"] > 0).sum()
        avg_ret = portfolio_df["ret10"].mean()
        std_ret = portfolio_df["ret10"].std()
        sharpe  = (avg_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

        print(f"  Total trades    : {n:,}")
        print(f"  Win rate        : {n_wins/n*100:.1f}%")
        print(f"  Avg ret/trade   : {avg_ret:.4f}")
        print(f"  Total return    : {(bankroll-INITIAL_BK)/INITIAL_BK*100:.1f}%")
        print(f"  Final bankroll  : ${bankroll:.2f}")
        print(f"  Max drawdown    : {max_dd*100:.1f}%")
        print(f"  Sharpe          : {sharpe:.2f}")
        print(f"\n  Trades by signal:")
        print(portfolio_df["signal_name"].value_counts().to_string())

    results_df.to_parquet(OUTPUT, index=False)
    print(f"\nSaved → {OUTPUT}")


if __name__ == "__main__":
    main()
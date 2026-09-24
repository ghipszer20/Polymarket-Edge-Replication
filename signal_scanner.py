"""
signal_scanner.py
Discovery on 2024, validation on 2025.
Uses nba_trades_features.parquet for game state signals.
Includes overtime handling.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

PARQUET = r"C:\Users\24GHi\PycharmProjects\PythonProject2\poly_data\processed\nba_trades_features.parquet"
OUTPUT  = Path(r"C:\Users\24GHi\PycharmProjects\PythonProject2\poly_data\processed\signal_scan.parquet")
FEE     = 0.002

con = duckdb.connect()
con.execute(f"CREATE OR REPLACE VIEW trades AS SELECT * FROM read_parquet('{PARQUET}')")

print("Building feature table...")
con.execute("""
CREATE OR REPLACE TABLE features AS
WITH game_markets AS (
    SELECT market_id
    FROM trades
    WHERE YEAR(timestamp) >= 2024
    GROUP BY market_id
    HAVING COUNT(*) >= 500
),
base AS (
    SELECT
        t.market_id,
        t.timestamp,
        t.price_usdc                  as p,
        t.makerAmountFilled           as size,
        t.score_diff,
        t.secs_remaining,
        t.secs_elapsed,
        t.espn_win_prob,
        t.poly_espn_div,
        t.espn_mom_60s,
        t.poly_mom_60s,
        t.yes_current_run,
        t.no_current_run,
        t.on_a_run,
        t.run_for_yes,
        t.run_for_no,
        t.maker_is_sharp,
        t.maker_is_whale,
        t.period,
        t.last_is_foul,
        t.last_is_score,
        t.last_is_3pt,
        t.last_is_timeout,
        t.game_close,
        t.game_competitive,
        t.abs_score_diff,
        t.proj_margin,
        t.run_diff_120s,
        t.run_diff_300s,
        YEAR(t.timestamp)             as yr,

        LAG(t.price_usdc,1)  OVER w   as p1,
        LAG(t.price_usdc,2)  OVER w   as p2,
        LAG(t.price_usdc,3)  OVER w   as p3,
        LAG(t.price_usdc,5)  OVER w   as p5,
        LAG(t.price_usdc,10) OVER w   as p10,
        LAG(t.price_usdc,20) OVER w   as p20,
        LAG(t.price_usdc,50) OVER w   as p50,

        LAG(t.makerAmountFilled,1)  OVER w as size1,
        LAG(t.makerAmountFilled,3)  OVER w as size3,
        LAG(t.makerAmountFilled,5)  OVER w as size5,
        LAG(t.makerAmountFilled,10) OVER w as size10,

        LEAD(t.price_usdc,1)  OVER w  as lead1,
        LEAD(t.price_usdc,2)  OVER w  as lead2,
        LEAD(t.price_usdc,5)  OVER w  as lead5,
        LEAD(t.price_usdc,10) OVER w  as lead10,
        LEAD(t.price_usdc,20) OVER w  as lead20,

        STDDEV(t.price_usdc) OVER w50  as vol50,
        STDDEV(t.price_usdc) OVER w20  as vol20,
        STDDEV(t.price_usdc) OVER w10  as vol10,
        AVG(t.price_usdc)    OVER w50  as ma50,
        AVG(t.price_usdc)    OVER w20  as ma20,
        AVG(t.price_usdc)    OVER w10  as ma10,

        AVG(t.makerAmountFilled)    OVER w20 as avg_size20,
        STDDEV(t.makerAmountFilled) OVER w20 as std_size20,
        AVG(t.makerAmountFilled)    OVER w50 as avg_size50,

        ROW_NUMBER() OVER w                       as rn,
        COUNT(*)     OVER (PARTITION BY t.market_id) as market_size

    FROM trades t
    INNER JOIN game_markets g ON t.market_id = g.market_id
    WHERE (
        (t.period BETWEEN 1 AND 4 AND t.secs_remaining > 60)
        OR (t.period > 4 AND t.secs_remaining > 30)
    )
    WINDOW
        w   AS (PARTITION BY t.market_id ORDER BY t.timestamp
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
        w10 AS (PARTITION BY t.market_id ORDER BY t.timestamp
                ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
        w20 AS (PARTITION BY t.market_id ORDER BY t.timestamp
                ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
        w50 AS (PARTITION BY t.market_id ORDER BY t.timestamp
                ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING)
)
SELECT *,
    p - p1   as move1,
    p - p3   as move3,
    p - p5   as move5,
    p - p10  as move10,
    p - p20  as move20,
    p - p50  as move50,
    p - ma10 as dev_ma10,
    p - ma20 as dev_ma20,
    p - ma50 as dev_ma50,
    vol10 / NULLIF(vol50, 0)      as vol_ratio,
    vol20 / NULLIF(vol50, 0)      as vol_ratio20,
    size  / NULLIF(avg_size20, 0) as size_ratio,
    size1 / NULLIF(avg_size20, 0) as size1_ratio,
    size5 / NULLIF(avg_size50, 0) as size5_ratio,
    CAST(rn AS DOUBLE) / market_size       as game_phase,
    secs_elapsed / (4.0 * 720)             as game_phase_time,
    move10 / NULLIF(vol50, 0)     as sigma10,
    move5  / NULLIF(vol50, 0)     as sigma5,
    move20 / NULLIF(vol50, 0)     as sigma20,
    lead2  - lead1 as fwd2,
    lead5  - lead1 as fwd5,
    lead10 - lead1 as fwd10,
    lead20 - lead1 as fwd20
FROM base
WHERE p50 IS NOT NULL
AND lead20 IS NOT NULL
AND vol50 > 0.001
AND vol50 < 0.8
AND p BETWEEN 0.02 AND 0.98
""")

n   = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
n24 = con.execute("SELECT COUNT(*) FROM features WHERE yr=2024").fetchone()[0]
n25 = con.execute("SELECT COUNT(*) FROM features WHERE yr=2025").fetchone()[0]
print(f"  Total rows        : {n:,}")
print(f"  2024 (discovery)  : {n24:,}")
print(f"  2025 (validation) : {n25:,}")


# ── Scanner ───────────────────────────────────────────────────
def scan(label, condition, direction, exit_col="fwd10", year=None):
    yr_filter = f"AND yr = {year}" if year else ""
    try:
        r = con.execute(f"""
            SELECT
                COUNT(*) as n,
                AVG(CASE WHEN ({direction}) * {exit_col} > 0
                         THEN 1.0 ELSE 0.0 END) as wr,
                AVG(({direction}) * fwd2)  as r2,
                AVG(({direction}) * fwd5)  as r5,
                AVG(({direction}) * fwd10) as r10,
                AVG(({direction}) * fwd20) as r20,
                STDDEV(({direction}) * fwd10) as std10
            FROM features
            WHERE {condition} {yr_filter}
        """).fetchone()
        if r is None or r[0] < 100 or r[4] is None or r[6] is None:
            return None
        net   = r[4] - FEE
        tstat = (r[4] / r[6] * np.sqrt(r[0])) if r[6] > 0 else 0
        pval  = stats.t.sf(tstat, df=r[0]-1)
        return {
            "signal":    label,
            "n":         r[0],
            "win_rate":  round(r[1]*100, 2),
            "ret_fwd2":  round(r[2],  6),
            "ret_fwd5":  round(r[3],  6),
            "ret_fwd10": round(r[4],  6),
            "ret_fwd20": round(r[5],  6),
            "net_fwd10": round(net,   6),
            "std_fwd10": round(r[6],  6),
            "tstat":     round(tstat, 3),
            "pval":      round(pval,  6),
        }
    except Exception as e:
        print(f"  ERROR {label}: {e}")
        return None


# ── Signal definitions ────────────────────────────────────────
signals = []
print("Generating signals...")

# 1. Momentum
for lag, col in [(1,"p1"),(2,"p2"),(3,"p3"),(5,"p5"),
                 (10,"p10"),(20,"p20"),(50,"p50")]:
    for thresh in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        for pmin, pmax in [(0.05,0.95),(0.10,0.90),(0.20,0.80),(0.30,0.70)]:
            signals.append((
                f"mom_up_lag{lag}_{int(thresh*100)}pct_p{int(pmin*100)}",
                f"p - {col} > {thresh} AND p BETWEEN {pmin} AND {pmax}",
                "+1"
            ))
            signals.append((
                f"mom_dn_lag{lag}_{int(thresh*100)}pct_p{int(pmin*100)}",
                f"p - {col} < -{thresh} AND p BETWEEN {pmin} AND {pmax}",
                "-1"
            ))

# 2. Reversion
for lag, col in [(3,"p3"),(5,"p5"),(10,"p10"),(20,"p20")]:
    for thresh in [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
        for pmin, pmax in [(0.05,0.95),(0.10,0.90),(0.20,0.80)]:
            signals.append((
                f"rev_lag{lag}_{int(thresh*100)}pct_p{int(pmin*100)}",
                f"ABS(p - {col}) > {thresh} AND p BETWEEN {pmin} AND {pmax}",
                f"-SIGN(p - {col})"
            ))

# 3. Sigma-adjusted momentum
for sigma in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    for pmin, pmax in [(0.05,0.95),(0.10,0.90),(0.20,0.80)]:
        signals.append((
            f"sigma_mom_up_{sigma}_p{int(pmin*100)}",
            f"sigma10 > {sigma} AND p BETWEEN {pmin} AND {pmax}",
            "+1"
        ))
        signals.append((
            f"sigma_mom_dn_{sigma}_p{int(pmin*100)}",
            f"sigma10 < -{sigma} AND p BETWEEN {pmin} AND {pmax}",
            "-1"
        ))
        signals.append((
            f"sigma_rev_{sigma}_p{int(pmin*100)}",
            f"ABS(sigma10) > {sigma} AND p BETWEEN {pmin} AND {pmax}",
            f"-SIGN(sigma10)"
        ))

# 4. Volatility regime
for vol_thresh in [0.3, 0.5, 0.7, 1.5, 2.0, 3.0]:
    for move_thresh in [0.01, 0.02, 0.03, 0.05]:
        signals.append((
            f"vol_hi_{vol_thresh}_mom_up_{int(move_thresh*100)}",
            f"vol_ratio > {vol_thresh} AND move5 > {move_thresh} AND p BETWEEN 0.05 AND 0.95",
            "+1"
        ))
        signals.append((
            f"vol_hi_{vol_thresh}_mom_dn_{int(move_thresh*100)}",
            f"vol_ratio > {vol_thresh} AND move5 < -{move_thresh} AND p BETWEEN 0.05 AND 0.95",
            "-1"
        ))
        signals.append((
            f"vol_lo_{vol_thresh}_mom_up_{int(move_thresh*100)}",
            f"vol_ratio < {1/vol_thresh:.2f} AND move5 > {move_thresh} AND p BETWEEN 0.05 AND 0.95",
            "+1"
        ))
        signals.append((
            f"vol_lo_{vol_thresh}_mom_dn_{int(move_thresh*100)}",
            f"vol_ratio < {1/vol_thresh:.2f} AND move5 < -{move_thresh} AND p BETWEEN 0.05 AND 0.95",
            "-1"
        ))

# 5. Size surge
for size_mult in [1.5, 2.0, 3.0, 5.0, 10.0]:
    for move_thresh in [0.0, 0.01, 0.02, 0.05]:
        for direction_label, dir_cond, dir_val in [
            ("up",     f"move1 > {move_thresh}",  "+1"),
            ("dn",     f"move1 < -{move_thresh}", "-1"),
            ("rev_up", f"move1 > {move_thresh}",  "-1"),
            ("rev_dn", f"move1 < -{move_thresh}", "+1"),
        ]:
            signals.append((
                f"size_{size_mult}x_{direction_label}_{int(move_thresh*100)}",
                f"size_ratio > {size_mult} AND {dir_cond} AND p BETWEEN 0.05 AND 0.95",
                dir_val
            ))

# 6. Game phase × momentum (time-based)
for phase_min, phase_max, phase_label in [
    (0.00, 0.05, "open"),
    (0.05, 0.15, "early"),
    (0.15, 0.35, "q1q2"),
    (0.35, 0.50, "half"),
    (0.50, 0.65, "q3"),
    (0.65, 0.85, "q4"),
    (0.85, 0.95, "late"),
    (0.95, 1.00, "final"),
]:
    for move_thresh in [0.02, 0.05, 0.10]:
        for sigma_thresh in [1.0, 2.0]:
            signals.append((
                f"phase_{phase_label}_mom_up_m{int(move_thresh*100)}_s{sigma_thresh}",
                f"game_phase_time BETWEEN {phase_min} AND {phase_max} "
                f"AND move10 > {move_thresh} AND ABS(sigma10) > {sigma_thresh} "
                f"AND p BETWEEN 0.05 AND 0.95",
                "+1"
            ))
            signals.append((
                f"phase_{phase_label}_mom_dn_m{int(move_thresh*100)}_s{sigma_thresh}",
                f"game_phase_time BETWEEN {phase_min} AND {phase_max} "
                f"AND move10 < -{move_thresh} AND ABS(sigma10) > {sigma_thresh} "
                f"AND p BETWEEN 0.05 AND 0.95",
                "-1"
            ))

# 7. MA deviation
for ma, ma_col in [(10,"ma10"),(20,"ma20"),(50,"ma50")]:
    for dev in [0.02, 0.05, 0.10, 0.15]:
        signals.append((
            f"above_ma{ma}_{int(dev*100)}pct_mom",
            f"p > {ma_col} * {1+dev} AND p > p5 AND p BETWEEN 0.05 AND 0.95",
            "+1"
        ))
        signals.append((
            f"below_ma{ma}_{int(dev*100)}pct_mom",
            f"p < {ma_col} * {1-dev} AND p < p5 AND p BETWEEN 0.05 AND 0.95",
            "-1"
        ))
        signals.append((
            f"above_ma{ma}_{int(dev*100)}pct_rev",
            f"p > {ma_col} * {1+dev} AND p > p5 AND p BETWEEN 0.05 AND 0.95",
            "-1"
        ))
        signals.append((
            f"below_ma{ma}_{int(dev*100)}pct_rev",
            f"p < {ma_col} * {1-dev} AND p < p5 AND p BETWEEN 0.05 AND 0.95",
            "+1"
        ))

# 8. Consecutive moves — capped at 3 (p4 does not exist in feature table)
for n_consec in [2, 3]:
    consec_up = " AND ".join([
        f"p{i} > p{i+1}" if i > 0 else "p > p1"
        for i in range(n_consec)
    ])
    consec_dn = " AND ".join([
        f"p{i} < p{i+1}" if i > 0 else "p < p1"
        for i in range(n_consec)
    ])
    for pmin, pmax in [(0.05,0.95),(0.10,0.90)]:
        signals.append((
            f"consec_up_{n_consec}_mom_p{int(pmin*100)}",
            f"{consec_up} AND p BETWEEN {pmin} AND {pmax}", "+1"
        ))
        signals.append((
            f"consec_dn_{n_consec}_mom_p{int(pmin*100)}",
            f"{consec_dn} AND p BETWEEN {pmin} AND {pmax}", "-1"
        ))
        signals.append((
            f"consec_up_{n_consec}_rev_p{int(pmin*100)}",
            f"{consec_up} AND p BETWEEN {pmin} AND {pmax}", "-1"
        ))
        signals.append((
            f"consec_dn_{n_consec}_rev_p{int(pmin*100)}",
            f"{consec_dn} AND p BETWEEN {pmin} AND {pmax}", "+1"
        ))

# 9. Price level effects
for p_lo, p_hi, label in [
    (0.05, 0.15, "very_low"),
    (0.15, 0.30, "low"),
    (0.30, 0.45, "below_mid"),
    (0.45, 0.55, "mid"),
    (0.55, 0.70, "above_mid"),
    (0.70, 0.85, "high"),
    (0.85, 0.95, "very_high"),
]:
    for move_thresh in [0.02, 0.05]:
        signals.append((
            f"level_{label}_mom_up_{int(move_thresh*100)}",
            f"p BETWEEN {p_lo} AND {p_hi} AND move10 > {move_thresh}",
            "+1"
        ))
        signals.append((
            f"level_{label}_mom_dn_{int(move_thresh*100)}",
            f"p BETWEEN {p_lo} AND {p_hi} AND move10 < -{move_thresh}",
            "-1"
        ))
        signals.append((
            f"level_{label}_rev_{int(move_thresh*100)}",
            f"p BETWEEN {p_lo} AND {p_hi} AND ABS(move10) > {move_thresh}",
            f"-SIGN(move10)"
        ))

# 10. Combined: size + sigma + phase
for size_mult in [2.0, 5.0]:
    for sigma in [1.0, 2.0]:
        for phase_max in [0.33, 0.66, 1.0]:
            signals.append((
                f"combo_size{size_mult}_sig{sigma}_phase{phase_max}_up",
                f"size_ratio > {size_mult} AND sigma10 > {sigma} "
                f"AND game_phase_time < {phase_max} AND p BETWEEN 0.05 AND 0.95",
                "+1"
            ))
            signals.append((
                f"combo_size{size_mult}_sig{sigma}_phase{phase_max}_dn",
                f"size_ratio > {size_mult} AND sigma10 < -{sigma} "
                f"AND game_phase_time < {phase_max} AND p BETWEEN 0.05 AND 0.95",
                "-1"
            ))

# 11. Game state — score differential + time
for score_thresh in [5, 10, 15]:
    for secs_thresh in [120, 300, 600]:
        signals.append((
            f"yes_leading_{score_thresh}_with_{secs_thresh}s",
            f"score_diff > {score_thresh} AND secs_remaining < {secs_thresh} "
            f"AND p BETWEEN 0.05 AND 0.95",
            "+1"
        ))
        signals.append((
            f"no_leading_{score_thresh}_with_{secs_thresh}s",
            f"score_diff < -{score_thresh} AND secs_remaining < {secs_thresh} "
            f"AND p BETWEEN 0.05 AND 0.95",
            "-1"
        ))

# 12. ESPN divergence signals
for div_thresh in [0.05, 0.10, 0.15, 0.20]:
    signals.append((
        f"poly_above_espn_{int(div_thresh*100)}",
        f"poly_espn_div > {div_thresh} AND secs_remaining > 120 "
        f"AND p BETWEEN 0.05 AND 0.95",
        "-1"
    ))
    signals.append((
        f"poly_below_espn_{int(div_thresh*100)}",
        f"poly_espn_div < -{div_thresh} AND secs_remaining > 120 "
        f"AND p BETWEEN 0.05 AND 0.95",
        "+1"
    ))

# ESPN divergence × time remaining
for div_thresh in [0.10, 0.20]:
    for secs_thresh in [300, 600]:
        signals.append((
            f"poly_above_espn_{int(div_thresh*100)}_early_{secs_thresh}s",
            f"poly_espn_div > {div_thresh} AND secs_remaining > {secs_thresh} "
            f"AND p BETWEEN 0.05 AND 0.95",
            "-1"
        ))
        signals.append((
            f"poly_above_espn_{int(div_thresh*100)}_late_{secs_thresh}s",
            f"poly_espn_div > {div_thresh} AND secs_remaining < {secs_thresh} "
            f"AND p BETWEEN 0.05 AND 0.95",
            "-1"
        ))

# 13. ESPN momentum leads Polymarket
signals.append((
    "espn_leading_poly_up",
    "espn_mom_60s > 0.05 AND poly_mom_60s < 0.02 AND p BETWEEN 0.05 AND 0.95",
    "+1"
))
signals.append((
    "espn_leading_poly_dn",
    "espn_mom_60s < -0.05 AND poly_mom_60s > -0.02 AND p BETWEEN 0.05 AND 0.95",
    "-1"
))

# 14. Sharp and whale wallet signals
signals.append((
    "sharp_mom_up",
    "maker_is_sharp = true AND move10 > 0.05 AND p BETWEEN 0.05 AND 0.95",
    "+1"
))
signals.append((
    "sharp_mom_dn",
    "maker_is_sharp = true AND move10 < -0.05 AND p BETWEEN 0.05 AND 0.95",
    "-1"
))
signals.append((
    "whale_mom_up",
    "maker_is_whale = true AND move10 > 0.05 AND p BETWEEN 0.05 AND 0.95",
    "+1"
))
signals.append((
    "whale_mom_dn",
    "maker_is_whale = true AND move10 < -0.05 AND p BETWEEN 0.05 AND 0.95",
    "-1"
))

# 15. Scoring run signals
signals.append((
    "yes_on_run_poly_above_espn",
    "run_for_yes = true AND poly_espn_div > 0.10 AND p BETWEEN 0.05 AND 0.95",
    "-1"
))
signals.append((
    "no_on_run_poly_below_espn",
    "run_for_no = true AND poly_espn_div < -0.10 AND p BETWEEN 0.05 AND 0.95",
    "+1"
))
signals.append((
    "yes_on_run_momentum",
    "run_for_yes = true AND move10 > 0.03 AND p BETWEEN 0.05 AND 0.95",
    "+1"
))
signals.append((
    "no_on_run_momentum",
    "run_for_no = true AND move10 < -0.03 AND p BETWEEN 0.05 AND 0.95",
    "-1"
))

# 16. Close game late signals
signals.append((
    "close_game_late_yes_leading",
    "game_close = true AND secs_remaining < 300 AND score_diff > 0 "
    "AND (period = 4 OR period > 4) AND p BETWEEN 0.05 AND 0.95",
    "+1"
))
signals.append((
    "close_game_late_no_leading",
    "game_close = true AND secs_remaining < 300 AND score_diff < 0 "
    "AND (period = 4 OR period > 4) AND p BETWEEN 0.05 AND 0.95",
    "-1"
))

# 17. Overtime signals
signals.append((
    "ot_yes_leading",
    "period > 4 AND score_diff > 0 AND secs_remaining > 30 "
    "AND p BETWEEN 0.05 AND 0.95",
    "+1"
))
signals.append((
    "ot_no_leading",
    "period > 4 AND score_diff < 0 AND secs_remaining > 30 "
    "AND p BETWEEN 0.05 AND 0.95",
    "-1"
))
signals.append((
    "ot_tied",
    "period > 4 AND score_diff = 0 AND secs_remaining > 30 "
    "AND p BETWEEN 0.40 AND 0.60",
    "+1"
))
signals.append((
    "ot_poly_above_espn",
    "period > 4 AND poly_espn_div > 0.10 AND secs_remaining > 30 "
    "AND p BETWEEN 0.05 AND 0.95",
    "-1"
))
signals.append((
    "ot_poly_below_espn",
    "period > 4 AND poly_espn_div < -0.10 AND secs_remaining > 30 "
    "AND p BETWEEN 0.05 AND 0.95",
    "+1"
))

print(f"  Total signals to test: {len(signals):,}")


# ── Discovery (2024) ──────────────────────────────────────────
print("\nPhase 1 — Discovery on 2024...")
discovery = []
for i, (label, cond, direction) in enumerate(signals):
    r = scan(label, cond, direction, year=2024)
    if r:
        discovery.append(r)
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(signals)} signals scanned")

df_disc = pd.DataFrame(discovery)
n_tests = len(df_disc)
df_disc["pval_corrected"]   = (df_disc["pval"] * n_tests).clip(upper=1.0)
df_disc["passes_discovery"] = (
    (df_disc["pval_corrected"] < 0.05) &
    (df_disc["net_fwd10"] > 0.001) &
    (df_disc["win_rate"] > 50) &
    (df_disc["n"] >= 500)
)

passed = df_disc[df_disc["passes_discovery"]]
print(f"\nDiscovery results:")
print(f"  Signals tested       : {len(df_disc):,}")
print(f"  Pass Bonferroni+edge : {len(passed):,}")
print(f"\nTop 20 by net return:")
print(df_disc.sort_values("net_fwd10", ascending=False).head(20)[[
    "signal","n","win_rate","net_fwd10","ret_fwd20","tstat","pval_corrected"
]].to_string(index=False))

if len(passed) == 0:
    print("\nNo signals passed Bonferroni. Relaxing to top 50 by net return...")
    print("WARNING: exploratory only — no multiple comparison correction applied")
    passed = df_disc[
        (df_disc["net_fwd10"] > 0) &
        (df_disc["win_rate"] > 50) &
        (df_disc["n"] >= 200)
    ].sort_values("net_fwd10", ascending=False).head(50)


# ── Validation (2025) ─────────────────────────────────────────
print(f"\nPhase 2 — Validation on 2025 ({len(passed)} signals)...")
validation = []
for _, row in passed.iterrows():
    label = row["signal"]
    match = [(l,c,d) for l,c,d in signals if l == label]
    if not match:
        continue
    _, cond, direction = match[0]
    r = scan(label, cond, direction, year=2025)
    if r:
        r["disc_net_fwd10"] = row["net_fwd10"]
        r["disc_win_rate"]  = row["win_rate"]
        validation.append(r)

df_val = pd.DataFrame(validation)
if len(df_val) > 0:
    df_val["passes_validation"] = (
        (df_val["net_fwd10"] > 0.001) &
        (df_val["win_rate"] > 50) &
        (df_val["pval"] < 0.05)
    )
    confirmed = df_val[df_val["passes_validation"]]

    print(f"\nValidation results:")
    print(f"  Signals validated : {len(df_val):,}")
    print(f"  Confirmed (2025)  : {len(confirmed):,}")
    print(f"\nAll validation results:")
    print(df_val.sort_values("net_fwd10", ascending=False)[[
        "signal","n","win_rate","net_fwd10","ret_fwd20",
        "tstat","pval","disc_net_fwd10","disc_win_rate","passes_validation"
    ]].to_string(index=False))

    if len(confirmed) > 0:
        print(f"\n{'='*60}")
        print(f"CONFIRMED SIGNALS — passed both 2024 and 2025")
        print(f"{'='*60}")
        print(confirmed[[
            "signal","n","win_rate","net_fwd10","ret_fwd20","tstat","pval"
        ]].to_string(index=False))
else:
    print("No validation results.")

# ── Save ──────────────────────────────────────────────────────
df_disc["dataset"] = "discovery_2024"
if len(df_val) > 0:
    df_val["dataset"]  = "validation_2025"
    combined = pd.concat([df_disc, df_val], ignore_index=True)
else:
    combined = df_disc

combined.to_parquet(OUTPUT, index=False)
print(f"\nSaved to {OUTPUT}")
print(f"Total signals generated: {len(signals):,}")
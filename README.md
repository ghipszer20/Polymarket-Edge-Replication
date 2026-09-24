# Polymarket edge replication

Research into a profitable Polymarket wallet, `claudesports`
(`0xc2f2d01b227948e59f25aeb3d49564b8975d0ff7`): where its edge comes from, why it cannot be
copy-traded, and a **paper-only** pipeline for testing whether the edge can be replicated.
Everything uses public data. Nothing here places a real order.

## What the analysis found

Based on 7,687 of the wallet's buy trades, 2026-05-04 to 2026-09-24. **All of this is
in-sample:** the conclusions were drawn from the same data they describe, with no holdout.

- **It cannot be copied.** It is a maker: it rests bids about 16c below the market. Anyone who
  reacts to a fill has to pay the higher price, and copying at market price lost -10.8% to -24%
  in the original research. A reactive "shadow bid" backtest (`claudesports_shadow.py`) also did
  not hold up as a general strategy.
- **The profit is real trading edge, not rewards.** P&L computed from its fills (+23% on cost)
  matches Polymarket's own leaderboard PnL ($33.4k). Liquidity rewards and maker rebates total
  only about $2.7k. The leaderboard's "$1.3M volume" is a share count; actual cash spent is
  about $156k.
- **It wins on discrete real-world outcomes** (weather thresholds, props and spreads, soccer,
  tennis, MLB, CS2) and **loses on crypto and commodity price-candle markets and on League of
  Legends**. NBA is unconfirmed (P&L and calibration disagree).
- **All of its profit is in the 10-70c price band.** Below 10c it loses money (-8.4% ROI); above
  ~70c it barely trades, while 28-51% of everyone else's volume in the same markets sits at
  90-100c.

Full evidence, tables and caveats: [`edge_replication_plan.md`](edge_replication_plan.md).

## What is not established

- No out-of-sample test of the band and category findings has been run.
- How much capital the wallet has at risk at one time is unknown, so no return on capital is
  quoted.
- The tennis model is unvalidated: Elo is roughly 66% accurate versus about 70% for bookmaker
  odds, so a large model-vs-market gap is more likely model error than mispricing. A forward
  paper tracker is collecting the evidence; it concludes nothing before ~200 markets resolve.
- Only tennis has a fair-value model. Weather, props, soccer, MLB and CS2 have none.

## Code

| File | Purpose |
|---|---|
| `claudesports_shadow.py` | Pulls the wallet's full trade history from the Data API, simulates the reactive shadow-bid idea, writes `claudesports_shadow.db`. About 4 minutes. |
| `maker_engine.py` | Paper-only, domain-agnostic quoting engine (10-70c band, edge threshold, exposure caps). Refuses to run in live mode. Its demo uses a **random placeholder** fair value, not a strategy. |
| `tennis_fair_value.py` | Elo win probability from Tennis Abstract's public ATP/WTA ratings. |
| `paper_tracker.py` | Forward paper tracking of model vs market for pre-match tennis: `cycle` snapshots, `resolve` records outcomes, `report` gives Brier score and calibration. |
| `schedule_tracker.ps1` | Registers two Windows scheduled tasks (cycle every 2 hours, resolve daily). |
| `edge_replication_plan.md` | Evidence trail, category verdicts, build plan, caveats. |
| `mlb_*`, `nba_*`, `signal_scanner.py`, `manual_markets.py`, ... | Pre-existing research and bot code from an earlier project. Not part of the claudesports analysis and not reviewed here. |

## Run

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
python claudesports_shadow.py
python paper_tracker.py cycle      # then: resolve, report
```

Market selection in the tracker uses Polymarket's own `sportsMarketType == "moneyline"` and a
future `gameStartTime`. Earlier keyword filters let set, handicap, over/under and "Completed
Match" markets through, and those are not match-winner markets.

## Safety

- Paper only. `maker_engine.py` raises if constructed with `paper=False`.
- The older `mlb_live_bot.py` and `nba_live_bot.py` **will trade** if the `POLYMARKET_PRIVATE_KEY`
  environment variable is set. Leave it unset.
- Keys belong in `.env` or environment variables, never in code. `.env`, databases and cached
  data are gitignored.
- The original research this project started from (not independently verified here) says
  Polymarket's international site is geoblocked for US users and that sports event contracts
  face active litigation. Check your own situation before any live trading.

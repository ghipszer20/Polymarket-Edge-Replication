# Replicating claudesports's edge — plan

Supersedes the original `claude_code_handoff.md` "shadow bid" strategy, which the
backtest in `claudesports_shadow.py` showed doesn't hold up (reactive copying loses
money; see that file's docstring for the full reasoning). This plan is based on
directly analyzing claudesports's own trading data in `claudesports_shadow.db`.

> **Read "Status and caveats" at the end first.** "Confirmed" below means confirmed on the same
> ~141 days of data the conclusions were drawn from (in-sample, no holdout). Nothing here has been
> validated out of sample or with real money.

## What claudesports actually is

Not a rewards-farming bot (checked: taker-fee pools in its niche markets are too
small — a few cents to low dollars per market lifetime — to explain its profit) and
not a pure market maker. It's a bot with **a genuinely better probability estimate
than Polymarket's crowd price**, on markets thin enough that the mispricing persists.
Confirmed two independent ways:

1. **Dollar P&L**, computed directly from claudesports's own fills + resolutions
   (zero fees assumed, since it's always the maker): overall +23.1% ROI on $144,728
   cost, matching the original handoff's headline "+$34.4k on $151k (+22.8%)" almost
   exactly.
2. **Calibration check** (price = implied probability vs. actual win rate, bucketed):
   confirmed a systematic, volume-weighted gap in claudesports's favor across every
   large-sample category it wins in. This is the stronger, more actionable proof —
   it isolates genuine mispricing from luck.

## Category verdicts (from both P&L and calibration — only trust categories where
## both agree)

**Confirmed edge — build the bot to cover the full market universe of these, not
just claudesports's specific past picks:**
| Category | n | $ ROI | Calibration gap | Notes |
|---|---|---|---|---|
| weather | 711 | +75.6% (post bug-fix) | +6.9% | Verified against real historical weather (Kuala Lumpur 34°C spot check) |
| true_other (player props, spreads, exact scores, F1) | 932 | +44.8% | +8.6% | |
| soccer (match outcomes) | 1,773 | +23.4% | +2.5% overall, but **edge concentrated in the 0.4-0.8 price band** (+20-26% gap); the 0.0-0.3 band (73% of volume) is flat-to-negative | Bot must target mid/higher-priced soccer bets, not cheap longshots |
| tennis | 130 | +20.9% | +8.5% | |
| mlb | 140 | +77.6% | +4.4% | |
| esports (CS2 specifically) | 192 (+ larger untested universe) | +53.8% | +12.4% | |

**Confirmed losers — exclude entirely:**
| Category | n | $ ROI | Calibration gap |
|---|---|---|---|
| crypto_price (hourly candle markets) | 2,833 | -10.6% | -2.1% |
| commodities (WTI oil candle markets) | 364 | -35.6% | +1.4% (contradicts P&L — likely tail-risk, not systematic; still excluded) |
| esports_misc (League of Legends specifically) | 224 | -49.2% | -2.3% overall, but -12% to -18% in the 0.1-0.3 band specifically |

**Unconfirmed — the two data sources disagree, don't build for these without more
data:**
| Category | $ ROI | Calibration gap |
|---|---|---|
| nba | -14.8% | +5.8% |

## The strategy is uniform across categories, not category-specific

Checked directly: compared claudesports's own price distribution against the general
trading population's distribution *within the same specific markets it traded*, for
every confirmed-edge category (weather, soccer, true_other/props, tennis, mlb,
esports). The same pattern holds in every single one, with no exceptions:

| Category | claudesports @ 0.8-1.0 | General population @ 0.9-1.0 alone |
|---|---|---|
| weather | 0.5% | (not directly compared; general market sample showed 67.7% of all volume in 0.9-1.0) |
| soccer_other_matches | 0.0% | 29.5% |
| true_other | 0.0% | 34.9% |
| tennis | 0.0% | 51.1% |
| mlb | 0.0% | 27.8% |
| esports | 0.0% | 50.5% |

**claudesports never buys above ~0.7-0.8 in any category, ever.** Meanwhile 28-51% of
*everyone else's* volume in these same markets sits in the 0.9-1.0 band — the crowd
piling into already-basically-decided outcomes right before resolution, where there's
no edge left to capture. claudesports simply never participates in that segment,
uniformly, across every market type tested.

**Architectural implication — the bot has two layers, and only one is
domain-specific:**
1. **Universal price-band filter** (shared infrastructure, build once): only
   evaluate/enter markets priced roughly 0.10-0.70. Dollar P&L by price band across
   the confirmed-edge categories (3,878 resolved fills, $107k cost) shows both edges
   of the band matter:

   | Price band | fills | cost | ROI | share of net profit |
   |---|---|---|---|---|
   | 0.0-0.1 | 2,162 | $14.8k | **-8.4%** | -2.9% |
   | 0.1-0.2 | 416 | $12.7k | +81.3% | 24.2% |
   | 0.2-0.3 | 643 | $23.4k | +23.1% | 12.7% |
   | 0.3-0.4 | 256 | $17.8k | +51.8% | 21.6% |
   | 0.4-0.5 | 115 | $11.6k | +46.5% | 12.6% |
   | 0.5-0.6 | 212 | $13.4k | +58.3% | 18.4% |
   | 0.6-0.7 | 46 | $8.2k | +54.0% | 10.3% |
   | 0.7-1.0 | 28 | $5.3k | +25.7% | 3.2% |

   The sub-0.10 band is 56% of its fills but only 14% of its dollars, and it LOSES money;
   claudesports barely trades above 0.70. (An earlier version of this plan said 0.0-0.4;
   that was wrong.) This rule is identical across every category.
2. **Domain-specific fair-value model** (build one per category): weather forecast
   model, soccer/tennis odds comparison, MLB win-probability extension, CS2 stats
   model. This is the part that actually varies, and only needs to operate *within*
   the segment the timing filter has already selected.

This should shape the scanning engine in phase 2 below: one shared filter stage,
then category-specific scoring only on what survives it.

## Tech stack

- **Python**, `asyncio` + `aiohttp` throughout — this is periodic scanning + resting
  limit orders, not sub-second execution, so the project's C++-for-latency-sensitive
  rule doesn't apply here.
- **Market discovery & pricing**: Polymarket Gamma API (broad scan across categories/
  leagues — not one wallet) + CLOB API (order books, midpoints).
- **Order placement**: `py-clob-client` (Polymarket's official SDK), wrapping
  `web3.py`/`eth-account` for signing. The project's venv already has the `eth_*`
  dependency chain from something else; `web3` and `py-clob-client` still need
  installing.
- **Domain fair-value models** (one module per domain, mirroring the existing
  `mlb_*`/`nba_*` file pattern in this project):
  - **Weather**: a probabilistic forecast source with a real historical archive
    (Open-Meteo Historical Forecast API, or NOAA/ECMWF reforecast data). Needs to
    produce a *distribution*, not a point estimate — these are mostly exact-degree
    or narrow-threshold markets, so point-forecast comparisons undersell the edge
    (confirmed this the hard way: a naive point-forecast test came back inconclusive
    for exactly this reason).
  - **Soccer**: an odds/stats API with deep minor-league international coverage
    (API-Football, Sportmonks, or an odds aggregator) — the edge lives in leagues
    thin enough that Polymarket's crowd is under-informed, and specifically in
    mid/higher-priced favorites, not longshots.
  - **Player props**: extend the soccer data source or a dedicated props feed.
  - **MLB/tennis**: extend the win-probability models *already built* in this
    project (`mlb_build_winprob_model.py`, `nba_build_winprob_model.py`) rather than
    starting over.
  - **CS2 esports**: needs a dedicated odds/stats source — not yet identified.
- **Storage/ops**: SQLite (matches the `claudesports_shadow.db` pattern already in
  place), `.env` + `python-dotenv` for the wallet private key and any paid API keys
  — do not repeat the Alpaca-key-in-plaintext mistake found elsewhere in this
  project's history.

## Build plan

1. **Backtest each domain's fair-value model independently, no live orders.**
   For each confirmed-edge category, pull real historical data (forecast archives,
   odds histories) for the *full market universe* in that category over the same
   historical window already used (2026-05-04 onward), compute a fair-value
   probability, and run the same calibration check used above against it. This
   validates the data source and model before spending on real-time feeds or
   wallet funding. Do NOT restrict this to markets claudesports specifically
   traded — the mispricing mechanism is structural to the market type, not tied to
   claudesports's own selection.
2. **Live scanning engine, two stages.** Stage A (shared, build once): poll Gamma
   across confirmed-edge categories only (weather, soccer, props, MLB, tennis, CS2 —
   explicitly skip crypto, commodities, NBA, LoL), and filter to markets currently
   priced 0.10-0.70 — this single rule is identical across every category
   and mirrors how claudesports itself behaves everywhere. Stage B
   (category-specific): only for what survives Stage A, run the relevant
   domain fair-value model and flag genuine edge above a threshold. Soccer's
   calibration edge concentrates in the 0.4-0.8 band while its 0.0-0.3 band is
   flat-to-negative, so soccer likely wants a narrower band (about 0.4-0.7)
   than the universal one. `maker_engine.py` implements 0.10-0.70.
3. **Execution layer.** `py-clob-client` order placement, sized via a capped/
   Kelly-fraction rule given the small-edge/high-frequency nature of these markets.
   Gated behind explicit confirmation before going live with real capital — this is
   the first component in the whole project requiring a funded, signing-capable
   wallet, categorically different risk than the read-only work done so far.
4. **Monitoring/reporting.** Extend the existing SQLite + summary-report pattern to
   track live fair-value-vs-market divergence, fill rate, and realized P&L by
   category, so any category that stops working (as crypto/commodities already have
   for claudesports) gets cut quickly.

## Status and caveats (as of 2026-09-24)

**Reconciliation against Polymarket's own numbers.**
- The leaderboard's "volume" for claudesports ($1,278,593) is a SHARE count, not dollars: our fills
  total 1,274,567 shares. Cash actually spent on BUY trades is about $156k (7,738 buys in the full
  activity history).
- The leaderboard's all-time PnL ($33,430) matches the P&L computed independently from our fills
  and resolutions ($33,372). Activity of type REWARD ($1.7k) and MAKER_REBATE ($0.9k) is about
  $2.7k in total, so rewards and rebates are a small part of the profit, not its source.

**What is NOT established.**
- Every band and category verdict above was derived from the same ~141 days of fills with no
  holdout. A time-split out-of-sample check was proposed and has not been run.
- Capital deployed at any one time is unknown: cash spent is ~$156k over 141 days, but positions
  resolve and recycle, so do not quote a return on capital from this repo.
- The tennis fair-value model (Tennis Abstract Elo) has never been validated. Published work finds
  Elo about 66% accurate versus about 70% for bookmaker odds, and ML adds only ~0.5-2.5 points over
  Elo, so large model-vs-market gaps are more likely model error than mispricing. The forward paper
  tracker (`paper_tracker.py`) exists to answer this; it reports nothing conclusive until at least
  200 markets have resolved. Elo covers about 55% of pre-match singles markets.
- No bookmaker-odds benchmark exists yet, and the other categories (weather, props, soccer, MLB,
  CS2) have no fair-value model at all.

**Safety.** Everything here is paper-only. `maker_engine.py` refuses to construct in live mode.
The older `mlb_live_bot.py` / `nba_live_bot.py` will trade if `POLYMARKET_PRIVATE_KEY` is set in the
environment; leave it unset.

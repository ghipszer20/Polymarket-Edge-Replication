"""
maker_engine.py

Domain-agnostic maker (market-making) engine for replicating claudesports's
verified strategy (see edge_replication_plan.md for the full evidence trail):
scan a market universe, compute fair value per market via a pluggable
domain-specific model, and decide whether to quote a resting BUY limit order —
only when:
  - price sits in the 10c-70c "sweet spot" (dollar-P&L verified: sub-10c loses
    money outright, 80c+ is the crowded, no-edge segment where 28-51% of everyone
    else's volume sits and claudesports never once participates)
  - the computed edge (fair_value - market_price) exceeds a minimum threshold

This is deliberately domain-agnostic: it doesn't know or care whether fair value
came from a weather model, a tennis odds API, or a win-probability model. Each
domain just needs to implement FairValueFn.

PAPER MODE ONLY. No live order placement is wired up — every quote decision is
logged to SQLite, not executed. Wiring this to py-clob-client and a funded wallet
is a separate, explicit step (edge_replication_plan.md phase 3) that needs its own
confirmation before any real capital is at risk. MakerEngine refuses to construct
in non-paper mode for exactly this reason.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# ── Sweet-spot + risk config (dollar-P&L verified, see edge_replication_plan.md) ──
SWEET_SPOT_MIN = 0.10
SWEET_SPOT_MAX = 0.70
MIN_EDGE = 0.05  # minimum (fair_value - market_price) required to bother quoting

MAX_POSITION_USD = 25.0            # cap per single market
MAX_CATEGORY_EXPOSURE_USD = 500.0  # cap per category (weather, soccer, etc.)
MAX_TOTAL_EXPOSURE_USD = 2000.0    # cap across everything

DB_PATH = Path(__file__).parent / "maker_engine.db"


@dataclass
class MarketSnapshot:
    condition_id: str
    asset: str
    category: str
    question: str
    market_price: float    # current best price for the side we'd be buying
    min_order_size: float = 5.0


@dataclass
class QuoteDecision:
    market: MarketSnapshot
    should_quote: bool
    reason: str
    fair_value: Optional[float] = None
    edge: Optional[float] = None
    quote_price: Optional[float] = None
    size_usd: Optional[float] = None


FairValueFn = Callable[[MarketSnapshot], Optional[float]]


@dataclass
class ExposureTracker:
    """Tracks paper-simulated exposure so the demo behaves like it has real risk
    limits, even though no capital is actually at risk yet."""
    total_usd: float = 0.0
    by_category: dict = field(default_factory=dict)

    def can_add(self, category: str, amount: float) -> bool:
        if self.total_usd + amount > MAX_TOTAL_EXPOSURE_USD:
            return False
        if self.by_category.get(category, 0.0) + amount > MAX_CATEGORY_EXPOSURE_USD:
            return False
        return True

    def add(self, category: str, amount: float) -> None:
        self.total_usd += amount
        self.by_category[category] = self.by_category.get(category, 0.0) + amount


class MakerEngine:
    def __init__(self, fair_value_fn: FairValueFn, paper: bool = True):
        if not paper:
            raise NotImplementedError(
                "Live order placement isn't wired up yet. This engine only runs in "
                "paper mode until py-clob-client and a funded wallet are set up as "
                "a separate, explicitly-confirmed step (edge_replication_plan.md, "
                "phase 3) — never flip this to skip that gate."
            )
        self.fair_value_fn = fair_value_fn
        self.paper = paper
        self.exposure = ExposureTracker()
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quote_decisions (
                timestamp INTEGER,
                condition_id TEXT,
                category TEXT,
                question TEXT,
                market_price REAL,
                fair_value REAL,
                edge REAL,
                should_quote INTEGER,
                reason TEXT,
                quote_price REAL,
                size_usd REAL
            )
            """
        )
        conn.commit()
        conn.close()

    def evaluate(self, market: MarketSnapshot) -> QuoteDecision:
        # Sweet-spot filter first — cheapest check, and the single most important
        # rule (this is the universal, category-independent behavior claudesports
        # shows everywhere: never below 10c, never above 70-80c).
        if not (SWEET_SPOT_MIN <= market.market_price <= SWEET_SPOT_MAX):
            return QuoteDecision(
                market, False,
                f"price {market.market_price:.2f} outside {SWEET_SPOT_MIN:.2f}-{SWEET_SPOT_MAX:.2f} sweet spot",
            )

        fair_value = self.fair_value_fn(market)
        if fair_value is None:
            return QuoteDecision(market, False, "fair value model returned no estimate")

        edge = fair_value - market.market_price
        if edge < MIN_EDGE:
            return QuoteDecision(
                market, False, f"edge {edge:+.1%} below {MIN_EDGE:.0%} threshold",
                fair_value, edge,
            )

        size_usd = max(MAX_POSITION_USD, market.min_order_size) if market.min_order_size <= MAX_POSITION_USD \
            else market.min_order_size
        size_usd = min(size_usd, MAX_POSITION_USD) if market.min_order_size <= MAX_POSITION_USD else size_usd
        if not self.exposure.can_add(market.category, size_usd):
            return QuoteDecision(market, False, "would exceed exposure caps", fair_value, edge)

        # Join the current best price rather than paying more than necessary.
        # A more refined version could price partway between market and fair
        # value to capture more edge per fill, at the cost of a lower fill rate.
        quote_price = market.market_price
        return QuoteDecision(
            market, True, f"edge {edge:+.1%} exceeds {MIN_EDGE:.0%} threshold",
            fair_value, edge, quote_price, size_usd,
        )

    def run_cycle(self, markets: list[MarketSnapshot]) -> list[QuoteDecision]:
        decisions = []
        conn = sqlite3.connect(DB_PATH)
        now = int(time.time())
        for m in markets:
            d = self.evaluate(m)
            decisions.append(d)
            conn.execute(
                "INSERT INTO quote_decisions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (now, m.condition_id, m.category, m.question, m.market_price,
                 d.fair_value, d.edge, int(d.should_quote), d.reason,
                 d.quote_price, d.size_usd),
            )
            if d.should_quote:
                self.exposure.add(m.category, d.size_usd)
                print(f"[PAPER QUOTE] {m.category:14s} {m.question[:55]:55s} "
                      f"price={m.market_price:.2f} fair={d.fair_value:.2f} "
                      f"edge={d.edge:+.1%} size=${d.size_usd:.2f}")
        conn.commit()
        conn.close()
        return decisions


# ── Demo / smoke test ───────────────────────────────────────────────────────

def _demo_fair_value_stub(market: MarketSnapshot) -> Optional[float]:
    """PLACEHOLDER ONLY — not a real model. Randomly perturbs the market price to
    demonstrate the engine's filtering/sizing/exposure logic end-to-end. Replace
    with a real domain model (weather forecast, odds API, win-probability model)
    before this means anything. Every real category needs its own version of
    this function."""
    import random
    return max(0.0, min(1.0, market.market_price + random.uniform(-0.15, 0.20)))


def _fetch_demo_markets(limit: int = 30) -> list[MarketSnapshot]:
    """Pull a handful of real, currently-open markets from confirmed-edge
    categories via Gamma, for a live smoke test of the engine (still paper —
    this only reads public data, no orders are placed)."""
    import requests

    out = []
    for tag in ["weather", "soccer", "tennis", "mlb", "esports"]:
        try:
            resp = requests.get(
                "https://gamma-api.polymarket.com/events",
                params={"tag_slug": tag, "closed": "false", "active": "true", "limit": 10},
                timeout=15,
            ).json()
        except Exception:
            continue
        for ev in resp:
            for m in ev.get("markets", [])[:2]:
                try:
                    prices = m.get("outcomePrices")
                    import json as _json
                    price = float(_json.loads(prices)[0]) if isinstance(prices, str) else float(prices[0])
                except Exception:
                    continue
                if price <= 0 or price >= 1:
                    continue
                out.append(MarketSnapshot(
                    condition_id=m.get("conditionId", ""),
                    asset="",
                    category=tag,
                    question=m.get("question", ev.get("title", "")),
                    market_price=price,
                    min_order_size=m.get("rewards", {}).get("min_size", 5.0) if isinstance(m.get("rewards"), dict) else 5.0,
                ))
                if len(out) >= limit:
                    return out
    return out


def main():
    print("Fetching real open markets from confirmed-edge categories (read-only)...")
    markets = _fetch_demo_markets()
    print(f"Got {len(markets)} markets. Running one paper scan cycle with a RANDOM "
          f"placeholder fair-value model (not a real strategy — demo only).\n")

    engine = MakerEngine(fair_value_fn=_demo_fair_value_stub, paper=True)
    decisions = engine.run_cycle(markets)

    quoted = sum(1 for d in decisions if d.should_quote)
    print(f"\n{quoted}/{len(decisions)} markets would have been quoted this cycle.")
    print(f"Simulated exposure: ${engine.exposure.total_usd:.2f} total, "
          f"by category: {engine.exposure.by_category}")
    print(f"Decisions logged to {DB_PATH}")


if __name__ == "__main__":
    main()

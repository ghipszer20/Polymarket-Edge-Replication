"""
mlb_live_bot.py
Polymarket MLB live trading bot — paper mode.
Polls ESPN MLB scoreboard + Polymarket every 30s pregame, 15s live.
Posts limit orders when signals fire.
Runs until MIN_PAPER_TRADES reached.

Setup:
1. Run manual_markets.py before each session to get MANUAL_MARKETS
2. python mlb_live_bot.py
"""

import asyncio
import aiohttp
import logging
import os
import pickle
import random
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

# ── Config ────────────────────────────────────────────────────
PROJECT_DIR           = Path(r"C:\Users\24GHi\PycharmProjects\PythonProject2")
MODEL_FILE            = PROJECT_DIR / "poly_data" / "processed" / "mlb_winprob_model.pkl"
LOG_FILE              = PROJECT_DIR / "mlb_live_bot_trades.csv"

POLL_INTERVAL_PREGAME = 30.0
POLL_INTERVAL_LIVE    = 15.0   # MLB slower than NBA — update every 15s
CANCEL_AFTER          = 60     # longer cancel window for MLB
EXIT_N_TRADES         = 10
EXIT_DIV_THRESH       = 0.05
EXIT_MINUTES          = 10     # longer exit window for MLB
INITIAL_BK            = 1000.0
MIN_BANKROLL          = 100.0
MIN_PAPER_TRADES      = 500
BETWEEN_SESSION_WAIT  = 300

# ── Manual market config ──────────────────────────────────────
# Run manual_markets.py before each session.
# token[0] = away team = YES, token[1] = home team = NO
MANUAL_MARKETS = [
    # {
    #     "market_id":    "0xabc...",
    #     "question":     "Yankees vs. Red Sox",
    #     "yes_token_id": "123...",
    #     "no_token_id":  "456...",
    # },
]

MLB_TEAMS = {
    "yankees","dodgers","mets","red sox","cubs","astros","braves",
    "padres","phillies","giants","cardinals","brewers","rays","blue jays",
    "orioles","guardians","twins","white sox","tigers","royals","angels",
    "athletics","mariners","rangers","rockies","diamondbacks","marlins",
    "nationals","pirates","reds"
}

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Signal definitions ────────────────────────────────────────
@dataclass
class Signal:
    name:      str
    direction: int
    kelly:     float = 0.02

    def check(self, state: dict) -> bool:
        raise NotImplementedError


class NoLeading3Inn7(Signal):
    def check(self, s):
        return (s["score_diff"] < -3
                and s["inning"] >= 7
                and 0.05 < s["price"] < 0.95)

class NoLeading3Inn8(Signal):
    def check(self, s):
        return (s["score_diff"] < -3
                and s["inning"] >= 8
                and 0.05 < s["price"] < 0.95)

class NoLeading4Inn7(Signal):
    def check(self, s):
        return (s["score_diff"] < -4
                and s["inning"] >= 7
                and 0.05 < s["price"] < 0.95)

class NoLeading4Inn8(Signal):
    def check(self, s):
        return (s["score_diff"] < -4
                and s["inning"] >= 8
                and 0.05 < s["price"] < 0.95)

class NoLeading5Inn7(Signal):
    def check(self, s):
        return (s["score_diff"] < -5
                and s["inning"] >= 7
                and 0.05 < s["price"] < 0.95)

class NoLeading5Inn8(Signal):
    def check(self, s):
        return (s["score_diff"] < -5
                and s["inning"] >= 8
                and 0.05 < s["price"] < 0.95)

class PolyAboveModel25Inn6(Signal):
    def check(self, s):
        return (s["poly_model_div"] > 0.25
                and s["inning"] >= 6
                and 0.05 < s["price"] < 0.95)

class PolyAboveModel25Inn8(Signal):
    def check(self, s):
        return (s["poly_model_div"] > 0.25
                and s["inning"] >= 8
                and 0.05 < s["price"] < 0.95)

class PolyAboveModel30Inn6(Signal):
    def check(self, s):
        return (s["poly_model_div"] > 0.30
                and s["inning"] >= 6
                and 0.05 < s["price"] < 0.95)

class PolyBelowModel30Inn6(Signal):
    def check(self, s):
        return (s["poly_model_div"] < -0.30
                and s["inning"] >= 6
                and 0.05 < s["price"] < 0.95)

class PolyBelowModel30Inn7(Signal):
    def check(self, s):
        return (s["poly_model_div"] < -0.30
                and s["inning"] >= 7
                and 0.05 < s["price"] < 0.95)

class PolyBelowModel30Inn8(Signal):
    def check(self, s):
        return (s["poly_model_div"] < -0.30
                and s["inning"] >= 8
                and 0.05 < s["price"] < 0.95)

class DivAbove25Inn8Late(Signal):
    def check(self, s):
        return (s["poly_model_div"] > 0.25
                and s["inning"] >= 8
                and s["is_late_game"]
                and 0.05 < s["price"] < 0.95)


SIGNALS = [
    NoLeading3Inn7(
        name="no_leading_3_inn7", direction=-1, kelly=0.02
    ),
    NoLeading3Inn8(
        name="no_leading_3_inn8", direction=-1, kelly=0.02
    ),
    NoLeading4Inn7(
        name="no_leading_4_inn7", direction=-1, kelly=0.02
    ),
    NoLeading4Inn8(
        name="no_leading_4_inn8", direction=-1, kelly=0.02
    ),
    NoLeading5Inn7(
        name="no_leading_5_inn7", direction=-1, kelly=0.02
    ),
    NoLeading5Inn8(
        name="no_leading_5_inn8", direction=-1, kelly=0.02
    ),
    PolyAboveModel25Inn6(
        name="poly_above_model_25_inn6", direction=-1, kelly=0.02
    ),
    PolyAboveModel25Inn8(
        name="poly_above_model_25_inn8", direction=-1, kelly=0.02
    ),
    PolyAboveModel30Inn6(
        name="poly_above_model_30_inn6", direction=-1, kelly=0.02
    ),
    PolyBelowModel30Inn6(
        name="poly_below_model_30_inn6", direction=+1, kelly=0.02
    ),
    PolyBelowModel30Inn7(
        name="poly_below_model_30_inn7", direction=+1, kelly=0.02
    ),
    PolyBelowModel30Inn8(
        name="poly_below_model_30_inn8", direction=+1, kelly=0.02
    ),
    DivAbove25Inn8Late(
        name="div_above_25_inn8_late", direction=-1, kelly=0.02
    ),
]


# ── Data structures ───────────────────────────────────────────
@dataclass
class GameState:
    game_id:        str
    market_id:      str
    yes_token_id:   str
    no_token_id:    str
    home_team:      str
    away_team:      str
    yes_is_away:    bool  = True
    home_score:     int   = 0
    away_score:     int   = 0
    inning:         int   = 0
    inning_half:    int   = 0   # 0=top, 1=bottom
    outs:           int   = 0
    game_status:    str   = "pre"
    yes_price:      float = 0.5
    no_price:       float = 0.5
    win_prob:       float = 0.5
    poly_model_div: float = 0.0
    poly_mom:       float = 0.0
    prev_price:     float = 0.5
    prev_ts:        float = 0.0
    active:         bool  = True

    @property
    def score_diff(self) -> int:
        # YES = away team
        return self.away_score - self.home_score

    @property
    def price(self) -> float:
        return self.yes_price

    @property
    def half_innings_remaining(self) -> float:
        done = (self.inning - 1) * 2 + self.inning_half
        return max(0, 18 - done)

    @property
    def game_pct_done(self) -> float:
        done = (self.inning - 1) * 2 + self.inning_half
        return min(done / 18.0, 1.0)

    @property
    def is_late_game(self) -> bool:
        return self.inning >= 7


@dataclass
class Position:
    market_id:   str
    signal_name: str
    direction:   int
    entry_price: float
    entry_time:  float
    entry_ts:    datetime
    size_usdc:   float
    kelly:       float
    order_id:    Optional[str]   = None
    filled:      bool            = False
    fill_price:  Optional[float] = None
    fill_time:   Optional[float] = None
    trade_count: int             = 0
    exit_price:  Optional[float] = None
    exit_time:   Optional[float] = None
    exit_reason: Optional[str]   = None
    pnl:         Optional[float] = None

    @property
    def is_open(self) -> bool:
        return self.filled and self.exit_price is None

    @property
    def is_pending(self) -> bool:
        return not self.filled and self.exit_reason is None

    @property
    def age_seconds(self) -> float:
        return time.time() - self.entry_time


# ── Win probability model ─────────────────────────────────────
class WinProbModel:
    def __init__(self, model_file: Path):
        with open(model_file, "rb") as f:
            data = pickle.load(f)
        self.model    = data["model"]
        self.features = data["features"]
        log.info(f"Loaded MLB win prob model (Brier={data['brier']:.4f})")

    def predict(self, score_diff: int, inning: int,
                inning_half: int, outs: int) -> float:
        hi_remaining        = max(0, 18 - ((inning - 1) * 2 + inning_half))
        score_x_innings     = score_diff * hi_remaining
        abs_score           = abs(score_diff)
        is_extra            = float(inning > 9)
        inning_dummies      = [float(inning == i) for i in range(1, 10)]
        X = np.array([[
            score_diff, inning, inning_half, outs,
            hi_remaining, score_x_innings, abs_score, is_extra,
            *inning_dummies,
        ]])
        prob = self.model.predict_proba(X)[0, 1]
        return float(np.clip(prob, 0.001, 0.999))


# ── Polymarket client ─────────────────────────────────────────
class PolyClient:
    TAKER_FEE = 0.001

    def __init__(self):
        self.private_key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
        self.clob_url    = "https://clob.polymarket.com"
        self._client     = None

        if self.private_key:
            try:
                from py_clob_client.client import ClobClient
                self._client = ClobClient(
                    host     = self.clob_url,
                    key      = self.private_key,
                    chain_id = 137,
                )
                log.info("py-clob-client initialized — LIVE MODE")
            except Exception as e:
                log.warning(f"py-clob-client init failed: {e} — paper mode")
        else:
            log.warning("No POLYMARKET_PRIVATE_KEY — PAPER MODE")

    @property
    def paper_mode(self) -> bool:
        return self._client is None

    async def get_midpoint(self, session: aiohttp.ClientSession,
                            token_id: str) -> Optional[float]:
        try:
            url = f"{self.clob_url}/midpoint?token_id={token_id}"
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=3)
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    mid  = data.get("mid")
                    if mid is not None:
                        return float(mid)
        except Exception:
            pass
        return None

    async def post_limit_order(self, token_id: str, price: float,
                                size: float, side: str) -> Optional[str]:
        if self.paper_mode:
            fake_id = f"paper_{int(time.time()*1000)}"
            log.info(f"  [PAPER] {side} {size:.2f} USDC @ {price:.4f} "
                     f"token={token_id[:20]}...")
            return fake_id
        try:
            from py_clob_client.clob_types import OrderArgs
            order    = self._client.create_order(OrderArgs(
                token_id     = token_id,
                price        = price,
                size         = size,
                side         = side,
                fee_rate_bps = 0,
            ))
            order_id = order.get("orderID") or order.get("id")
            log.info(f"  [LIVE] {side} {size:.2f} @ {price:.4f} id={order_id}")
            return order_id
        except Exception as e:
            log.error(f"  Order failed: {e}")
            return None

    async def cancel_order(self, order_id: str) -> bool:
        if self.paper_mode:
            return True
        try:
            self._client.cancel(order_id)
            return True
        except Exception as e:
            log.warning(f"  Cancel failed {order_id}: {e}")
            return False

    async def check_order_filled(self,
                                  order_id: str) -> tuple[bool, Optional[float]]:
        if self.paper_mode:
            return random.random() < 0.63, None
        try:
            order  = self._client.get_order(order_id)
            status = order.get("status", "")
            if status in ("MATCHED", "FILLED"):
                return True, float(order.get("price", 0))
            return False, None
        except Exception:
            return False, None


# ── ESPN MLB data ─────────────────────────────────────────────
async def fetch_mlb_scoreboard(session: aiohttp.ClientSession) -> dict:
    url = ("https://site.api.espn.com/apis/site/v2/sports"
           "/baseball/mlb/scoreboard")
    try:
        async with session.get(
            url,
            timeout = aiohttp.ClientTimeout(total=5),
            headers = {"User-Agent": "Mozilla/5.0"}
        ) as r:
            if r.status == 200:
                return await r.json(content_type=None)
    except Exception as e:
        log.warning(f"MLB scoreboard fetch failed: {e}")
    return {}


def parse_inning_detail(detail: str) -> tuple[int, int]:
    """
    Parse ESPN detail string to (inning, inning_half).
    Examples: 'Top 5th' -> (5, 0), 'Bot 7th' -> (7, 1),
              'Mid 3rd' -> (3, 0), 'End 9th' -> (9, 1),
              'Final' -> (9, 1)
    """
    detail_lower = detail.lower()
    half = 0

    if detail_lower.startswith("bot") or detail_lower.startswith("end"):
        half = 1

    # Extract inning number
    import re
    numbers = re.findall(r'\d+', detail)
    inning  = int(numbers[0]) if numbers else 0

    return inning, half


def update_game_state_from_espn(gs: GameState,
                                  espn_game: dict) -> None:
    comp   = espn_game.get("competitions", [{}])[0]
    status = comp.get("status", {})
    sit    = comp.get("situation", {})

    gs.game_status = status.get("type", {}).get("state", "pre")
    detail         = status.get("type", {}).get("detail", "")
    period         = int(status.get("period", 0) or 0)

    inning, inning_half = parse_inning_detail(detail)
    if inning == 0:
        inning = period

    gs.inning      = inning
    gs.inning_half = inning_half
    gs.outs        = int(sit.get("outs", 0) or 0)

    for comp_team in comp.get("competitors", []):
        ha    = comp_team.get("homeAway", "")
        score = int(comp_team.get("score", 0) or 0)
        if ha == "home":
            gs.home_score = score
        elif ha == "away":
            gs.away_score = score


# ── Bot ───────────────────────────────────────────────────────
class MLBLiveBot:
    TAKER_FEE = 0.001

    def __init__(self):
        self.model              = WinProbModel(MODEL_FILE)
        self.poly               = PolyClient()
        self.bankroll           = INITIAL_BK
        self.positions:   list[Position]       = []
        self.trade_log:   list[dict]           = []
        self.all_closed:  list[dict]           = []
        self.game_states: dict[str, GameState] = {}

    def compute_features(self, gs: GameState) -> dict:
        if gs.inning >= 1 and gs.game_status == "in":
            gs.win_prob = self.model.predict(
                gs.score_diff, gs.inning,
                gs.inning_half, gs.outs
            )
        gs.poly_model_div = gs.yes_price - gs.win_prob
        if gs.prev_ts > 0:
            dt = time.time() - gs.prev_ts
            gs.poly_mom = (gs.yes_price - gs.prev_price
                           if 0 < dt <= 120 else 0.0)
        return {
            "price":          gs.yes_price,
            "score_diff":     gs.score_diff,
            "inning":         gs.inning,
            "inning_half":    gs.inning_half,
            "outs":           gs.outs,
            "game_pct_done":  gs.game_pct_done,
            "poly_model_div": gs.poly_model_div,
            "win_prob":       gs.win_prob,
            "poly_mom":       gs.poly_mom,
            "is_late_game":   gs.is_late_game,
        }

    async def check_signals(self, gs: GameState, state: dict,
                             session: aiohttp.ClientSession) -> None:
        if gs.inning < 1 or gs.game_status != "in":
            return
        for signal in SIGNALS:
            if not signal.check(state):
                continue
            existing = [p for p in self.positions
                        if p.market_id == gs.market_id
                        and p.signal_name == signal.name
                        and (p.is_open or p.is_pending)]
            if existing:
                continue
            if signal.direction == +1:
                token_id = gs.yes_token_id
                side     = "BUY"
                price    = round(gs.yes_price, 4)
            else:
                token_id = gs.no_token_id
                side     = "BUY"
                price    = round(1.0 - gs.yes_price, 4)
            size_usdc = max(self.bankroll * signal.kelly, 1.0)
            log.info(f"SIGNAL: {signal.name} | "
                     f"{gs.away_team} @ {gs.home_team} | "
                     f"inn={gs.inning} score={gs.score_diff:+d} "
                     f"price={gs.yes_price:.3f} wp={gs.win_prob:.3f} "
                     f"div={gs.poly_model_div:+.3f} | "
                     f"{side} ${size_usdc:.2f}")
            order_id = await self.poly.post_limit_order(
                token_id, price, size_usdc, side
            )
            if order_id:
                self.positions.append(Position(
                    market_id   = gs.market_id,
                    signal_name = signal.name,
                    direction   = signal.direction,
                    entry_price = price,
                    entry_time  = time.time(),
                    entry_ts    = datetime.now(timezone.utc),
                    size_usdc   = size_usdc,
                    kelly       = signal.kelly,
                    order_id    = order_id,
                ))

    async def manage_positions(self, gs: GameState, state: dict,
                                session: aiohttp.ClientSession) -> None:
        for pos in self.positions:
            if pos.market_id != gs.market_id:
                continue
            if pos.is_pending and pos.order_id:
                filled, fill_price = await self.poly.check_order_filled(
                    pos.order_id
                )
                if filled:
                    pos.filled     = True
                    pos.fill_price = fill_price or pos.entry_price
                    pos.fill_time  = time.time()
                    log.info(f"  FILLED: {pos.signal_name} "
                             f"@ {pos.fill_price:.4f}")
                elif pos.age_seconds > CANCEL_AFTER:
                    await self.poly.cancel_order(pos.order_id)
                    pos.exit_reason = "cancelled_timeout"
                    pos.exit_time   = time.time()
                    log.info(f"  CANCELLED: {pos.signal_name} (timeout)")
                continue
            if not pos.is_open:
                continue
            pos.trade_count += 1
            age_minutes      = pos.age_seconds / 60
            exit_reason      = None
            if pos.trade_count >= EXIT_N_TRADES:
                exit_reason = f"exit_A_{EXIT_N_TRADES}trades"
            elif abs(state["poly_model_div"]) < EXIT_DIV_THRESH:
                exit_reason = "exit_B_div_closed"
            elif age_minutes >= EXIT_MINUTES:
                exit_reason = "exit_C_time_limit"
            elif gs.game_status == "post":
                exit_reason = "exit_game_over"
            if exit_reason:
                await self.execute_exit(
                    pos, gs, gs.yes_price, exit_reason
                )

    async def execute_exit(self, pos: Position, gs: GameState,
                            current_price: float, reason: str) -> None:
        fill = pos.fill_price if pos.fill_price is not None else pos.entry_price
        if pos.direction == +1:
            exit_price = current_price
            raw_pnl    = exit_price - fill
        else:
            exit_price = 1.0 - current_price
            raw_pnl    = exit_price - fill
        net_pnl  = raw_pnl - self.TAKER_FEE
        pnl_usdc = net_pnl * pos.size_usdc
        pos.exit_price  = exit_price
        pos.exit_time   = time.time()
        pos.exit_reason = reason
        pos.pnl         = pnl_usdc
        self.bankroll  += pnl_usdc
        log.info(f"  EXIT [{reason}]: {pos.signal_name} | "
                 f"entry={fill:.4f} exit={exit_price:.4f} | "
                 f"net={net_pnl:+.4f} | "
                 f"PnL=${pnl_usdc:+.2f} | BK=${self.bankroll:.2f}")
        record = {
            "timestamp":   pos.entry_ts.isoformat(),
            "market_id":   pos.market_id,
            "game":        f"{gs.away_team} @ {gs.home_team}",
            "signal":      pos.signal_name,
            "direction":   pos.direction,
            "entry_price": fill,
            "exit_price":  pos.exit_price,
            "exit_reason": pos.exit_reason,
            "size_usdc":   pos.size_usdc,
            "pnl_usdc":    pos.pnl,
            "bankroll":    self.bankroll,
            "trade_count": pos.trade_count,
            "age_seconds": (pos.exit_time - pos.fill_time
                            if pos.fill_time else None),
        }
        self.trade_log.append(record)

    async def discover_markets(self,
                                session: aiohttp.ClientSession) -> None:
        log.info("Discovering active MLB markets...")

        if MANUAL_MARKETS:
            log.info(f"Using {len(MANUAL_MARKETS)} manually configured markets")
            for market in MANUAL_MARKETS:
                mid   = market["market_id"]
                q     = market["question"]
                parts = q.replace(".", "").split(" vs ")
                away_team = parts[0].strip().lower() if len(parts) == 2 else "away"
                home_team = parts[1].strip().lower() if len(parts) == 2 else "home"
                self.game_states[mid] = GameState(
                    game_id      = mid,
                    market_id    = mid,
                    yes_token_id = market["yes_token_id"],
                    no_token_id  = market["no_token_id"],
                    home_team    = home_team,
                    away_team    = away_team,
                    yes_is_away  = True,
                )
                log.info(f"  Tracking: {q}")
            log.info(f"Tracking {len(self.game_states)} markets")
            return

        log.warning("MANUAL_MARKETS is empty. "
                    "Run manual_markets.py to populate it.")

    async def run_poll(self, session: aiohttp.ClientSession) -> bool:
        """Run one poll cycle. Returns True if any game is live."""
        mlb_task   = asyncio.create_task(fetch_mlb_scoreboard(session))
        book_tasks = {
            mid: asyncio.create_task(
                self.poly.get_midpoint(session, gs.yes_token_id)
            )
            for mid, gs in self.game_states.items() if gs.active
        }

        mlb_data = await mlb_task
        games    = mlb_data.get("events", [])

        # Update Polymarket prices
        for mid, task in book_tasks.items():
            midpoint = await task
            if midpoint is not None and mid in self.game_states:
                gs            = self.game_states[mid]
                gs.prev_price = gs.yes_price
                gs.prev_ts    = time.time()
                gs.yes_price  = float(midpoint)
                gs.no_price   = round(1.0 - float(midpoint), 4)

        # Match ESPN games to tracked markets
        for espn_game in games:
            comp        = espn_game.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            home_team   = next(
                (c.get("team", {}).get("displayName", "").lower()
                 for c in competitors if c.get("homeAway") == "home"), ""
            )
            away_team   = next(
                (c.get("team", {}).get("displayName", "").lower()
                 for c in competitors if c.get("homeAway") == "away"), ""
            )

            matched_gs = None
            for gs in self.game_states.values():
                if not gs.active:
                    continue
                home_match = (gs.home_team in home_team or
                              home_team in gs.home_team)
                away_match = (gs.away_team in away_team or
                              away_team in gs.away_team)
                if home_match and away_match:
                    matched_gs = gs
                    break

            if matched_gs is None:
                continue

            if matched_gs.game_id == matched_gs.market_id:
                matched_gs.game_id = espn_game.get(
                    "id", matched_gs.game_id
                )

            update_game_state_from_espn(matched_gs, espn_game)

            # Mark inactive if game is final
            status = comp.get("status", {})
            if status.get("type", {}).get("state") == "post":
                matched_gs.active = False
                log.info(f"Game over: {matched_gs.away_team} "
                         f"{matched_gs.away_score} @ "
                         f"{matched_gs.home_team} {matched_gs.home_score}")
                # Force exit all open positions
                for pos in self.positions:
                    if (pos.market_id == matched_gs.market_id
                            and pos.is_open):
                        state = self.compute_features(matched_gs)
                        await self.execute_exit(
                            pos, matched_gs,
                            matched_gs.yes_price, "exit_game_over"
                        )
                continue

            state = self.compute_features(matched_gs)
            await self.check_signals(matched_gs, state, session)
            await self.manage_positions(matched_gs, state, session)

        # Status log
        live = [
            (gs.away_team, gs.home_team, gs.inning,
             gs.inning_half, gs.away_score, gs.home_score,
             gs.yes_price, gs.poly_model_div)
            for gs in self.game_states.values()
            if gs.active and gs.game_status == "in"
        ]
        if live:
            log.info("Live: " + " | ".join(
                f"{a}@{h} inn={i}{'b' if ih else 't'} "
                f"{as_}-{hs} price={p:.3f} div={d:+.3f}"
                for a, h, i, ih, as_, hs, p, d in live
            ))

        return bool(live)

    def _load_trade_log(self) -> None:
        self.all_closed = []
        if LOG_FILE.exists():
            try:
                df              = pd.read_csv(LOG_FILE)
                df              = df[df["pnl_usdc"].notna()]
                self.all_closed = df.to_dict("records")
                if self.all_closed:
                    self.bankroll = float(df["bankroll"].iloc[-1])
                log.info(f"Resumed: {len(self.all_closed)} trades, "
                         f"BK=${self.bankroll:.2f}")
            except Exception as e:
                log.warning(f"Could not load trade log: {e}")

    def _save_trade_log(self) -> None:
        if not self.trade_log:
            return
        df           = pd.DataFrame(self.trade_log)
        write_header = not LOG_FILE.exists()
        df.to_csv(LOG_FILE, mode="a", header=write_header, index=False)
        self.all_closed.extend(self.trade_log)
        log.info(f"Saved {len(self.trade_log)} trades → {LOG_FILE}")
        self.trade_log = []

    def _print_summary(self) -> None:
        all_trades = self.all_closed + self.trade_log
        closed     = [t for t in all_trades if t.get("pnl_usdc") is not None]
        if not closed:
            log.info("No closed trades to summarize.")
            return
        df       = pd.DataFrame(closed)
        n        = len(df)
        win_rate = (df["pnl_usdc"] > 0).mean()
        avg_ret  = df["pnl_usdc"].mean()
        std_ret  = df["pnl_usdc"].std()
        sharpe   = (avg_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0
        log.info("\n" + "="*60)
        log.info("FINAL MLB PAPER TRADING SUMMARY")
        log.info(f"  Total trades  : {n}")
        log.info(f"  Win rate      : {win_rate*100:.1f}%")
        log.info(f"  Avg PnL/trade : ${avg_ret:.4f}")
        log.info(f"  Total PnL     : ${df['pnl_usdc'].sum():+.2f}")
        log.info(f"  Final BK      : ${self.bankroll:.2f}")
        log.info(f"  Return        : "
                 f"{(self.bankroll-INITIAL_BK)/INITIAL_BK*100:+.1f}%")
        log.info(f"  Sharpe        : {sharpe:.2f}")
        log.info("\n  By signal:")
        for sig in df["signal"].unique():
            sub = df[df["signal"] == sig]
            log.info(f"    {sig}: n={len(sub)} "
                     f"wr={(sub['pnl_usdc']>0).mean()*100:.1f}% "
                     f"avg=${sub['pnl_usdc'].mean():.4f}")
        log.info("\n  By exit method:")
        for reason in df["exit_reason"].unique():
            sub = df[df["exit_reason"] == reason]
            log.info(f"    {reason}: n={len(sub)} "
                     f"wr={(sub['pnl_usdc']>0).mean()*100:.1f}% "
                     f"avg=${sub['pnl_usdc'].mean():.4f}")
        log.info("="*60)
        df.to_csv(LOG_FILE, index=False)
        log.info(f"  Full log: {LOG_FILE}")

    async def run(self) -> None:
        log.info("="*60)
        log.info("MLB POLYMARKET BOT STARTING")
        log.info(f"  Mode       : {'PAPER' if self.poly.paper_mode else 'LIVE'}")
        log.info(f"  Bankroll   : ${INITIAL_BK:.2f}")
        log.info(f"  Signals    : {len(SIGNALS)}")
        log.info(f"  Target     : {MIN_PAPER_TRADES} closed trades")
        log.info("="*60)

        self._load_trade_log()
        session_num = 0

        try:
            while True:
                closed = [p for p in self.all_closed
                          if p.get("pnl_usdc") is not None]
                if len(closed) >= MIN_PAPER_TRADES:
                    log.info(f"Reached {len(closed)} trades — stopping.")
                    break

                session_num += 1
                log.info(f"\n--- Session {session_num} | "
                         f"Trades: {len(closed)}/{MIN_PAPER_TRADES} ---")

                connector = aiohttp.TCPConnector(limit=20)
                async with aiohttp.ClientSession(
                    connector=connector
                ) as session:
                    self.game_states = {}
                    self.positions   = []

                    await self.discover_markets(session)

                    if not self.game_states:
                        log.info(f"No markets. "
                                 f"Waiting {BETWEEN_SESSION_WAIT}s...")
                        await asyncio.sleep(BETWEEN_SESSION_WAIT)
                        continue

                    log.info(
                        f"Polling (pregame={POLL_INTERVAL_PREGAME}s, "
                        f"live={POLL_INTERVAL_LIVE}s)..."
                    )

                    while True:
                        t0 = time.time()

                        if self.bankroll < MIN_BANKROLL:
                            log.error(
                                f"BK ${self.bankroll:.2f} below minimum."
                            )
                            self._save_trade_log()
                            self._print_summary()
                            return

                        if all(not gs.active
                               for gs in self.game_states.values()):
                            log.info("All games finished.")
                            break

                        try:
                            any_live = await self.run_poll(session)
                        except Exception as e:
                            log.error(f"Poll error: {e}", exc_info=True)
                            any_live = False

                        elapsed       = time.time() - t0
                        poll_interval = (POLL_INTERVAL_LIVE if any_live
                                         else POLL_INTERVAL_PREGAME)
                        sleep_time    = max(0.0, poll_interval - elapsed)

                        if not any_live:
                            log.info(
                                f"No games live. "
                                f"Next check in {poll_interval:.0f}s..."
                            )

                        await asyncio.sleep(sleep_time)

                self._save_trade_log()
                log.info(f"Waiting {BETWEEN_SESSION_WAIT}s...")
                await asyncio.sleep(BETWEEN_SESSION_WAIT)

        except KeyboardInterrupt:
            log.info("Stopped by user.")
        finally:
            self._save_trade_log()
            self._print_summary()


if __name__ == "__main__":
    asyncio.run(MLBLiveBot().run())
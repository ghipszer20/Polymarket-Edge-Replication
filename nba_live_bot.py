"""
nba_live_bot.py
Polymarket NBA live trading bot.
Polls NBA Stats + Polymarket every 30 seconds before games,
switches to every 5 seconds when games go live.

Signal design based on nba_edge_scanner.py results from 842,520 in-game trades.
Key finding: divergence WIDENING is the dominant predictor — win rate 71.8% vs 35.8%.
All signals require the widening condition.

Paper mode cost model:
  Entry : maker limit order = 0 fee
  Exit  : dynamic sports taker fee (peak 0.75% at p=0.50)
  Spread: 0.5¢ half-spread (0.005) round-trip
  Slippage: 0.1% (0.001)
"""

import asyncio
import aiohttp
import logging
import os
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

# ── Config ────────────────────────────────────────────────────
PROJECT_DIR           = Path(r"C:\Users\24GHi\PycharmProjects\PythonProject2")
MODEL_FILE            = PROJECT_DIR / "poly_data" / "processed" / "winprob_model.pkl"
LOG_FILE              = PROJECT_DIR / "nba_live_bot_trades.csv"

POLL_INTERVAL_PREGAME = 30.0
POLL_INTERVAL_LIVE    = 5.0
CANCEL_AFTER          = 30
EXIT_N_TRADES         = 10
EXIT_DIV_THRESH       = 0.05
EXIT_MINUTES          = 5
INITIAL_BK            = 1000.0
MIN_BANKROLL          = 100.0
MIN_PAPER_TRADES      = 500
BETWEEN_SESSION_WAIT  = 300
SIGNAL_COOLDOWN_SECS  = 60
MAX_OPEN_PER_GAME     = 2

# ── Paper mode cost model ─────────────────────────────────────
PAPER_SPREAD_HALF = 0.005
PAPER_SLIPPAGE    = 0.001
SPORTS_PEAK_FEE   = 0.0075


def sports_taker_fee(price: float) -> float:
    p = max(0.001, min(0.999, price))
    return SPORTS_PEAK_FEE * p * (1.0 - p) / 0.25


# ── Manual market config ──────────────────────────────────────
MANUAL_MARKETS = [
    {
        "market_id": "0xb54b0241f4f33e6734a29df5d59e0b4fbb7b96cf92d2c7151622cdb294c94516",
        "question": "Bucks vs. Clippers",
        "yes_token_id": "36209974969270005929478913409127766361788224769548192629896274601121703735484",
        "no_token_id": "14601167560606426917718331855347642051223131748556197437271063122555636628254",
    },
    {
        "market_id": "0xdf99b42652984acb2679ce1729acca79ca93cbb1eb6e512cf7153eedcd0b630e",
        "question": "Heat vs. Pacers",
        "yes_token_id": "54268772394696508593449064098275296195610727610315674125039764021346976776532",
        "no_token_id": "30737869522084583250787348761921807424436212950877429392167290854319537675093",
    },
    {
        "market_id": "0xd37522f09338ac31b622cfc55f7dd0fd56f4138790b0af8824227202b1b90ac4",
        "question": "Kings vs. Nets",
        "yes_token_id": "32218104176927886479094054791215039383439201412126676759926564161259425928171",
        "no_token_id": "71703344853726352606227286727477156747988497265261389572258926895688500924890",
    },
    {
        "market_id": "0xba1c98b14a48bd9d1daeb325affc18b6f544dc3a27cb96130d783f1e49404f79",
        "question": "Celtics vs. Hornets",
        "yes_token_id": "57490801281206654852018729936736026755861016869364825432770952173135027532030",
        "no_token_id": "115188286692673541878141556629858441876946797209792642837550526323278552751048",
    },
    {
        "market_id": "0xef54425dc6d1727e47330841ae0f1a21ad050896ff88fc13a297c3d84e4aa25a",
        "question": "Magic vs. Raptors",
        "yes_token_id": "34264767528799095092721184890314240284913433712903162694603714880864452578901",
        "no_token_id": "8358691472058530639821447384002385942665528799389136840096744984014672693361",
    },
    {
        "market_id": "0xf84ca6f84dc20091c346d03a88f8644ee557e41c6b995e7809ec9a8d5d0968bf",
        "question": "Trail Blazers vs. Wizards",
        "yes_token_id": "78887580442619691321808019311514174730774054977588456230020613247014939017059",
        "no_token_id": "73825077151789743621109686158106183247921951320330568595421120222482239611866",
    },
    {
        "market_id": "0xbaf52c12a8e1d733106c87a38d5cccae22e2da3ec4e98c3d3fa17fc987d0aad0",
        "question": "Pelicans vs. Rockets",
        "yes_token_id": "90983988591044836812677395940939974740533245529447154489588830380422092094064",
        "no_token_id": "27628343019234996563184508808595312021741521549637156976363468307325922182226",
    },
    {
        "market_id": "0xc41ada612461b7090d8b19086cc4c2060e16f7eaeb985aeee089f4fe2c12de7a",
        "question": "Nuggets vs. Warriors",
        "yes_token_id": "21341380303690007692832055295311560631168332650882032609807142350179251549791",
        "no_token_id": "53258696700130508595443221744296719185974344918382674767203601812931065213093",
    },
]

NBA_TEAMS = {
    "hawks","celtics","nets","hornets","bulls","cavaliers","mavericks",
    "nuggets","pistons","warriors","rockets","pacers","clippers","lakers",
    "grizzlies","heat","bucks","timberwolves","pelicans","knicks","thunder",
    "magic","76ers","suns","trail blazers","kings","spurs","raptors","jazz","wizards"
}

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Signal definitions ────────────────────────────────────────
# All signals require div_widening=True — the single biggest predictor
# from the edge scan (win rate 71.8% vs 35.8% without it).
#
# div_widening: divergence is moving AWAY from fair value.
# If poly_espn_div > 0 (poly above ESPN): widening = price still rising = mom > 0
# If poly_espn_div < 0 (poly below ESPN): widening = price still falling = mom < 0
# Equivalently: poly_espn_div and poly_mom_60s have the same sign.
# Threshold 0.005 filters out noise in the momentum reading.

@dataclass
class Signal:
    name:        str
    direction:   int
    description: str
    kelly:       float = 0.02

    def check(self, state: dict) -> bool:
        raise NotImplementedError


def div_widening(s: dict, threshold: float = 0.005) -> bool:
    """True if the ESPN divergence is currently growing in magnitude."""
    div = s.get("poly_espn_div", 0.0)
    mom = s.get("poly_mom_60s", 0.0)
    if mom is None or np.isnan(mom):
        return False
    if div > 0:
        return mom > threshold    # poly above ESPN and still rising
    elif div < 0:
        return mom < -threshold   # poly below ESPN and still falling
    return False


def make_direction(s: dict) -> int:
    """Short if poly is overpriced, long if underpriced."""
    return -1 if s["poly_espn_div"] > 0 else +1


# ── Signal 1: extreme div, late, close game (highest edge +0.0543) ──
class ExtremeDivLateClose(Signal):
    def check(self, s):
        return (s["abs_espn_div"] >= 0.40
                and 180 < s["secs_remaining"] < 300
                and s["abs_score_diff"] <= 3
                and div_widening(s)
                and 0.05 < s["price"] < 0.95)


# ── Signal 2: extreme div, late, any score (broader version of 1) ───
class ExtremeDivLate(Signal):
    def check(self, s):
        return (s["abs_espn_div"] >= 0.40
                and 180 < s["secs_remaining"] < 300
                and s["abs_score_diff"] > 3   # non-close (close covered by sig 1)
                and div_widening(s)
                and 0.05 < s["price"] < 0.95)


# ── Signal 3: high div, final 2 min (70% win rate) ──────────────────
class HighDivFinal2Min(Signal):
    def check(self, s):
        return (s["abs_espn_div"] >= 0.35
                and 60 < s["secs_remaining"] < 120
                and div_widening(s)
                and 0.05 < s["price"] < 0.95)


# ── Signal 4: high div, final minute, close game (+0.0390) ──────────
class HighDivFinalMinClose(Signal):
    def check(self, s):
        return (s["abs_espn_div"] >= 0.30
                and s["secs_remaining"] <= 60
                and s["secs_remaining"] > 10   # avoid garbage time
                and s["abs_score_diff"] <= 3
                and div_widening(s)
                and 0.05 < s["price"] < 0.95)


# ── Signal 5: medium div, 2-3 min, close game (+0.0207) ─────────────
class MedDivLateClose(Signal):
    def check(self, s):
        return (s["abs_espn_div"] >= 0.15
                and 120 < s["secs_remaining"] < 180
                and s["abs_score_diff"] <= 7
                and div_widening(s)
                and 0.05 < s["price"] < 0.95)


# ── Signal 6: medium div, 2-3 min, blowout (+0.0272) ────────────────
# Counter-intuitive: ESPN reacts faster in blowouts, Poly lags
class MedDivLateBlowout(Signal):
    def check(self, s):
        return (s["abs_espn_div"] >= 0.15
                and 120 < s["secs_remaining"] < 180
                and 8 <= s["abs_score_diff"] <= 12
                and div_widening(s)
                and 0.05 < s["price"] < 0.95)


# ── Signal 7: high div, 5-10 min, moderate lead (+0.0153) ───────────
class HighDivMidModerateLead(Signal):
    def check(self, s):
        return (s["abs_espn_div"] >= 0.30
                and 300 < s["secs_remaining"] < 600
                and 4 <= s["abs_score_diff"] <= 7
                and div_widening(s)
                and 0.05 < s["price"] < 0.95)


# ── Signal 8: high div, 10-20 min, blowout (+0.0133) ────────────────
class HighDivEarlyBlowout(Signal):
    def check(self, s):
        return (s["abs_espn_div"] >= 0.25
                and 600 < s["secs_remaining"] < 1200
                and s["abs_score_diff"] >= 13
                and div_widening(s)
                and 0.05 < s["price"] < 0.95)


SIGNALS = [
    ExtremeDivLateClose(
        name="extreme_div_late_close",
        direction=0,       # direction computed dynamically from div sign
        description="Div≥0.40, 3-5min, close game (±3), widening",
        kelly=0.03         # highest edge — slightly higher Kelly
    ),
    ExtremeDivLate(
        name="extreme_div_late",
        direction=0,
        description="Div≥0.40, 3-5min, non-close game, widening",
        kelly=0.02
    ),
    HighDivFinal2Min(
        name="high_div_final_2min",
        direction=0,
        description="Div≥0.35, final 2min, widening",
        kelly=0.02
    ),
    HighDivFinalMinClose(
        name="high_div_final_min_close",
        direction=0,
        description="Div≥0.30, final 60s, close game, widening",
        kelly=0.02
    ),
    MedDivLateClose(
        name="med_div_late_close",
        direction=0,
        description="Div≥0.15, 2-3min, score ≤7, widening",
        kelly=0.02
    ),
    MedDivLateBlowout(
        name="med_div_late_blowout",
        direction=0,
        description="Div≥0.15, 2-3min, score 8-12, widening",
        kelly=0.02
    ),
    HighDivMidModerateLead(
        name="high_div_mid_moderate_lead",
        direction=0,
        description="Div≥0.30, 5-10min, score 4-7, widening",
        kelly=0.015
    ),
    HighDivEarlyBlowout(
        name="high_div_early_blowout",
        direction=0,
        description="Div≥0.25, 10-20min, score≥13, widening",
        kelly=0.015
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
    period:         int   = 0
    clock:          str   = "PT00M00.00S"
    secs_remaining: float = 2880.0
    secs_elapsed:   float = 0.0
    yes_price:      float = 0.5
    no_price:       float = 0.5
    win_prob:       float = 0.5
    poly_espn_div:  float = 0.0
    poly_mom_60s:   float = 0.0
    game_pct_done:  float = 0.0
    prev_price:     float = 0.5
    prev_ts:        float = 0.0
    active:         bool  = True

    @property
    def score_diff(self) -> int:
        if self.yes_is_away:
            return self.away_score - self.home_score
        return self.home_score - self.away_score

    @property
    def abs_score_diff(self) -> int:
        return abs(self.score_diff)

    @property
    def price(self) -> float:
        return self.yes_price


@dataclass
class Position:
    market_id:           str
    signal_name:         str
    direction:           int
    entry_price:         float
    entry_time:          float
    entry_ts:            datetime
    size_usdc:           float
    kelly:               float
    entry_state:         dict = field(default_factory=dict)
    order_id:            Optional[str]   = None
    filled:              bool            = False
    fill_price:          Optional[float] = None
    fill_time:           Optional[float] = None
    trade_count:         int             = 0
    exit_price:          Optional[float] = None
    exit_time:           Optional[float] = None
    exit_reason:         Optional[str]   = None
    pnl:                 Optional[float] = None
    max_favourable_move: float           = 0.0
    max_adverse_move:    float           = 0.0

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
        log.info(f"Loaded win prob model (Brier={data['brier']:.4f})")

    def predict(self, score_diff: float, secs_remaining: float,
                period: int) -> float:
        is_ot  = float(period > 4)
        sti    = score_diff / max(secs_remaining ** 0.5, 1)
        log_sr = np.log1p(secs_remaining)
        X = np.array([[
            score_diff, secs_remaining, sti, log_sr, is_ot,
            float(period == 1), float(period == 2),
            float(period == 3), float(period == 4),
        ]])
        prob = self.model.predict_proba(X)[0, 1]
        return float(np.clip(prob, 0.001, 0.999))


# ── Polymarket client ─────────────────────────────────────────
class PolyClient:

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

    async def get_fee_rate_bps(self, session: aiohttp.ClientSession,
                                token_id: str) -> int:
        try:
            url = f"{self.clob_url}/markets/{token_id}"
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=3)
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    return int(data.get("feeRateBps", 0))
        except Exception:
            pass
        return 0

    async def get_active_nba_markets(self,
                                      session: aiohttp.ClientSession) -> list[dict]:
        markets       = []
        next_cursor   = "MA=="
        pages_checked = 0
        while pages_checked < 20:
            try:
                url = (f"{self.clob_url}/markets"
                       f"?active=true&closed=false"
                       f"&next_cursor={next_cursor}&limit=100")
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=5)
                ) as r:
                    if r.status != 200:
                        break
                    data = await r.json()
                page_markets  = data.get("data", [])
                next_cursor   = data.get("next_cursor", "")
                pages_checked += 1
                for m in page_markets:
                    q = m.get("question", "").lower()
                    if " vs" in q and any(t in q for t in NBA_TEAMS):
                        tokens = m.get("tokens", [])
                        if len(tokens) >= 2:
                            markets.append({
                                "market_id":    m.get("condition_id"),
                                "question":     m.get("question"),
                                "yes_token_id": tokens[0].get("token_id"),
                                "no_token_id":  tokens[1].get("token_id"),
                            })
                if not next_cursor or next_cursor == "LTE=":
                    break
            except Exception as e:
                log.warning(f"Market discovery error: {e}")
                break
        log.info(f"Auto-discovery: {len(markets)} NBA markets")
        return markets

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
                                size: float, side: str,
                                fee_rate_bps: int = 0) -> Optional[str]:
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
                fee_rate_bps = fee_rate_bps,
            ))
            order_id = order.get("orderID") or order.get("id")
            log.info(f"  [LIVE] {side} {size:.2f} @ {price:.4f} "
                     f"fee={fee_rate_bps}bps id={order_id}")
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
            return True, None
        try:
            order  = self._client.get_order(order_id)
            status = order.get("status", "")
            if status in ("MATCHED", "FILLED"):
                return True, float(order.get("price", 0))
            return False, None
        except Exception:
            return False, None


# ── NBA data ──────────────────────────────────────────────────
async def fetch_nba_scoreboard(session: aiohttp.ClientSession) -> dict:
    url = ("https://cdn.nba.com/static/json/liveData"
           "/scoreboard/todaysScoreboard_00.json")
    try:
        async with session.get(
            url,
            timeout = aiohttp.ClientTimeout(total=5),
            headers = {"User-Agent": "Mozilla/5.0"}
        ) as r:
            if r.status == 200:
                return await r.json(content_type=None)
    except Exception as e:
        log.warning(f"NBA scoreboard fetch failed: {e}")
    return {}


def parse_clock(clock_str: str, period: int) -> tuple[float, float]:
    try:
        c = clock_str.replace("PT", "").replace("S", "")
        if "M" in c:
            parts = c.split("M")
            mins  = float(parts[0])
            secs  = float(parts[1])
        else:
            mins = 0.0
            secs = float(c) if c else 0.0
        quarter_secs_remaining = mins * 60 + secs

        if period <= 0:
            return 2880.0, 0.0

        if period <= 4:
            secs_elapsed   = (period - 1) * 720 + (720 - quarter_secs_remaining)
            secs_remaining = max(0.0, 4 * 720 - secs_elapsed)
        else:
            secs_elapsed   = (4 * 720
                              + (period - 5) * 300
                              + (300 - quarter_secs_remaining))
            secs_remaining = max(0.0, quarter_secs_remaining)

    except Exception:
        secs_elapsed   = 0.0
        secs_remaining = 2880.0

    return secs_remaining, secs_elapsed


def update_game_state_from_nba(gs: GameState, nba_game: dict) -> None:
    gs.home_score = int(nba_game.get("homeTeam", {}).get("score", 0) or 0)
    gs.away_score = int(nba_game.get("awayTeam", {}).get("score", 0) or 0)
    gs.period     = int(nba_game.get("period", 0) or 0)
    gs.clock      = nba_game.get("gameClock", "PT00M00.00S") or "PT00M00.00S"
    gs.secs_remaining, gs.secs_elapsed = parse_clock(gs.clock, gs.period)
    gs.game_pct_done = min(gs.secs_elapsed / (4 * 720), 1.0)


# ── Bot ───────────────────────────────────────────────────────
class LiveBot:

    def __init__(self):
        self.model              = WinProbModel(MODEL_FILE)
        self.poly               = PolyClient()
        self.bankroll           = INITIAL_BK
        self.positions:   list[Position]       = []
        self.trade_log:   list[dict]           = []
        self.all_closed:  list[dict]           = []
        self.game_states: dict[str, GameState] = {}

    def compute_features(self, gs: GameState) -> dict:
        if gs.period >= 1 and gs.secs_remaining > 0:
            gs.win_prob = self.model.predict(
                gs.score_diff, gs.secs_remaining, gs.period
            )
        gs.poly_espn_div = gs.yes_price - gs.win_prob

        if gs.prev_ts > 0:
            dt = time.time() - gs.prev_ts
            gs.poly_mom_60s = (gs.yes_price - gs.prev_price
                               if 0 < dt <= 120 else 0.0)

        abs_div = abs(gs.poly_espn_div)

        # widening: divergence growing in magnitude
        mom = gs.poly_mom_60s
        if mom is None or np.isnan(mom):
            widening = False
        elif gs.poly_espn_div > 0:
            widening = mom > 0.005
        elif gs.poly_espn_div < 0:
            widening = mom < -0.005
        else:
            widening = False

        return {
            "price":           gs.yes_price,
            "score_diff":      gs.score_diff,
            "abs_score_diff":  gs.abs_score_diff,
            "secs_remaining":  gs.secs_remaining,
            "period":          gs.period,
            "game_pct_done":   gs.game_pct_done,
            "poly_espn_div":   gs.poly_espn_div,
            "abs_espn_div":    abs_div,
            "espn_win_prob":   gs.win_prob,
            "poly_mom_60s":    gs.poly_mom_60s,
            "div_widening":    widening,
        }

    async def check_signals(self, gs: GameState, state: dict,
                             session: aiohttp.ClientSession) -> None:
        if gs.period < 1 or gs.secs_remaining < 10:
            return

        open_in_game = sum(
            1 for p in self.positions
            if p.market_id == gs.market_id and p.is_open
        )
        if open_in_game >= MAX_OPEN_PER_GAME:
            return

        now = time.time()

        for signal in SIGNALS:
            if not signal.check(state):
                continue

            existing = [p for p in self.positions
                        if p.market_id == gs.market_id
                        and p.signal_name == signal.name
                        and (p.is_open or p.is_pending)]
            if existing:
                continue

            recent_exit = next(
                (p for p in self.positions
                 if p.market_id == gs.market_id
                 and p.signal_name == signal.name
                 and p.exit_time is not None
                 and now - p.exit_time < SIGNAL_COOLDOWN_SECS),
                None
            )
            if recent_exit:
                continue

            # direction from divergence sign: short if poly above ESPN, long if below
            direction = make_direction(state)

            if direction == +1:
                token_id = gs.yes_token_id
                side     = "BUY"
                price    = round(gs.yes_price, 4)
            else:
                token_id = gs.no_token_id
                side     = "BUY"
                price    = round(1.0 - gs.yes_price, 4)

            size_usdc = max(self.bankroll * signal.kelly, 1.0)

            fee_rate_bps = 0
            if not self.poly.paper_mode:
                fee_rate_bps = await self.poly.get_fee_rate_bps(
                    session, token_id
                )

            log.info(f"SIGNAL: {signal.name} | "
                     f"{gs.away_team} vs {gs.home_team} | "
                     f"price={gs.yes_price:.3f} div={gs.poly_espn_div:+.3f} "
                     f"mom={gs.poly_mom_60s:+.3f} "
                     f"score={gs.score_diff:+d} "
                     f"secs={gs.secs_remaining:.0f} "
                     f"dir={'LONG' if direction==1 else 'SHORT'} | "
                     f"{side} ${size_usdc:.2f}")

            order_id = await self.poly.post_limit_order(
                token_id, price, size_usdc, side, fee_rate_bps
            )
            if order_id:
                self.positions.append(Position(
                    market_id   = gs.market_id,
                    signal_name = signal.name,
                    direction   = direction,
                    entry_price = price,
                    entry_time  = now,
                    entry_ts    = datetime.now(timezone.utc),
                    size_usdc   = size_usdc,
                    kelly       = signal.kelly,
                    order_id    = order_id,
                    entry_state = {
                        "poly_espn_div":  state["poly_espn_div"],
                        "abs_espn_div":   state["abs_espn_div"],
                        "secs_remaining": state["secs_remaining"],
                        "score_diff":     state["score_diff"],
                        "abs_score_diff": state["abs_score_diff"],
                        "period":         state["period"],
                        "espn_win_prob":  state["espn_win_prob"],
                        "poly_mom_60s":   state["poly_mom_60s"],
                        "div_widening":   state["div_widening"],
                        "price":          state["price"],
                    },
                ))

    async def manage_positions(self, gs: GameState, state: dict,
                                session: aiohttp.ClientSession) -> None:
        if gs.period < 1 or gs.secs_remaining < 1:
            return

        relevant = [
            p for p in self.positions
            if p.market_id == gs.market_id
            and (p.is_open or p.is_pending)
        ]

        for pos in relevant:
            if pos.is_pending and pos.order_id:
                filled, fill_price = await self.poly.check_order_filled(
                    pos.order_id
                )
                if filled:
                    pos.filled     = True
                    pos.fill_price = fill_price or pos.entry_price
                    pos.fill_time  = time.time()
                    log.info(f"  FILLED: {pos.signal_name} @ {pos.fill_price:.4f}")
                elif not self.poly.paper_mode and pos.age_seconds > CANCEL_AFTER:
                    await self.poly.cancel_order(pos.order_id)
                    pos.exit_reason = "cancelled_timeout"
                    pos.exit_time   = time.time()
                    log.info(f"  CANCELLED: {pos.signal_name} (timeout)")
                continue

            if not pos.is_open:
                continue

            # track intra-trade movement
            if pos.direction == +1:
                move = gs.yes_price - pos.entry_price
            else:
                move = (1.0 - gs.yes_price) - pos.entry_price

            pos.max_favourable_move = max(pos.max_favourable_move, move)
            pos.max_adverse_move    = min(pos.max_adverse_move, move)

            pos.trade_count += 1
            age_minutes      = pos.age_seconds / 60
            exit_reason      = None

            if pos.trade_count >= EXIT_N_TRADES:
                exit_reason = f"exit_A_{EXIT_N_TRADES}trades"
            elif abs(state["poly_espn_div"]) < EXIT_DIV_THRESH:
                exit_reason = "exit_B_div_closed"
            elif age_minutes >= EXIT_MINUTES:
                exit_reason = "exit_C_time_limit"
            elif gs.period >= 4 and gs.secs_remaining < 10:
                exit_reason = "exit_game_over"

            if exit_reason:
                await self.execute_exit(pos, gs, gs.yes_price,
                                        exit_reason, state)

        self.positions = [
            p for p in self.positions
            if p.is_open or p.is_pending
        ]

    async def execute_exit(self, pos: Position, gs: GameState,
                            current_price: float, reason: str,
                            state: dict) -> None:
        fill = pos.fill_price if pos.fill_price is not None else pos.entry_price

        if pos.direction == +1:
            exit_price = current_price
            raw_pnl    = exit_price - fill
        else:
            exit_price = 1.0 - current_price
            raw_pnl    = exit_price - fill

        if self.poly.paper_mode:
            taker_fee   = sports_taker_fee(exit_price)
            spread_cost = PAPER_SPREAD_HALF * 2
            cost        = taker_fee + spread_cost + PAPER_SLIPPAGE
        else:
            cost = PAPER_SPREAD_HALF + PAPER_SLIPPAGE

        net_pnl  = raw_pnl - cost
        pnl_usdc = net_pnl * pos.size_usdc

        pos.exit_price  = exit_price
        pos.exit_time   = time.time()
        pos.exit_reason = reason
        pos.pnl         = pnl_usdc
        self.bankroll  += pnl_usdc

        log.info(f"  EXIT [{reason}]: {pos.signal_name} | "
                 f"entry={fill:.4f} exit={exit_price:.4f} | "
                 f"raw={raw_pnl:+.4f} cost={cost:.4f} net={net_pnl:+.4f} | "
                 f"PnL=${pnl_usdc:+.2f} | BK=${self.bankroll:.2f}")

        record = {
            "timestamp":             pos.entry_ts.isoformat(),
            "market_id":             pos.market_id,
            "game":                  f"{gs.away_team} vs {gs.home_team}",
            "signal":                pos.signal_name,
            "direction":             pos.direction,
            "entry_price":           fill,
            "exit_price":            pos.exit_price,
            "exit_reason":           pos.exit_reason,
            "size_usdc":             pos.size_usdc,
            "pnl_usdc":              pos.pnl,
            "cost_unit":             cost,
            "cost_usdc":             cost * pos.size_usdc,
            "taker_fee_unit":        sports_taker_fee(exit_price)
                                     if self.poly.paper_mode else None,
            "bankroll":              self.bankroll,
            "trade_count":           pos.trade_count,
            "age_seconds":           (pos.exit_time - pos.fill_time
                                      if pos.fill_time else None),
            "max_favourable_move":   pos.max_favourable_move,
            "max_adverse_move":      pos.max_adverse_move,
            # entry state
            "entry_poly_espn_div":   pos.entry_state.get("poly_espn_div"),
            "entry_abs_espn_div":    pos.entry_state.get("abs_espn_div"),
            "entry_secs_remaining":  pos.entry_state.get("secs_remaining"),
            "entry_score_diff":      pos.entry_state.get("score_diff"),
            "entry_abs_score_diff":  pos.entry_state.get("abs_score_diff"),
            "entry_period":          pos.entry_state.get("period"),
            "entry_espn_win_prob":   pos.entry_state.get("espn_win_prob"),
            "entry_poly_mom_60s":    pos.entry_state.get("poly_mom_60s"),
            "entry_div_widening":    pos.entry_state.get("div_widening"),
            "entry_price_mid":       pos.entry_state.get("price"),
            # exit state
            "exit_poly_espn_div":    state.get("poly_espn_div"),
            "exit_secs_remaining":   state.get("secs_remaining"),
            "exit_score_diff":       state.get("score_diff"),
            "exit_period":           state.get("period"),
            "exit_espn_win_prob":    state.get("espn_win_prob"),
            "exit_poly_mom_60s":     state.get("poly_mom_60s"),
        }
        self.trade_log.append(record)

    async def discover_markets(self,
                                session: aiohttp.ClientSession) -> None:
        log.info("Discovering active NBA markets...")

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

        log.info("Attempting auto-discovery via CLOB API...")
        markets  = await self.poly.get_active_nba_markets(session)
        nba_data = await fetch_nba_scoreboard(session)
        games    = nba_data.get("scoreboard", {}).get("games", [])
        for g in games:
            home = g.get("homeTeam", {})
            away = g.get("awayTeam", {})
            log.info(f"  [{g.get('gameId')}] "
                     f"{away.get('teamCity')} {away.get('teamName')} "
                     f"@ {home.get('teamCity')} {home.get('teamName')}")
        for market in markets:
            q         = market["question"].lower()
            game_id   = ""
            home_team = ""
            away_team = ""
            for g in games:
                hc = g.get("homeTeam", {}).get("teamCity", "").lower()
                ac = g.get("awayTeam", {}).get("teamCity", "").lower()
                hn = g.get("homeTeam", {}).get("teamName", "").lower()
                an = g.get("awayTeam", {}).get("teamName", "").lower()
                if (any(w in q for w in hc.split()) and
                        any(w in q for w in ac.split())):
                    game_id   = g.get("gameId", "")
                    home_team = f"{hc} {hn}"
                    away_team = f"{ac} {an}"
                    break
            if not game_id:
                continue
            mid = market["market_id"]
            self.game_states[mid] = GameState(
                game_id      = game_id,
                market_id    = mid,
                yes_token_id = market["yes_token_id"],
                no_token_id  = market["no_token_id"],
                home_team    = home_team,
                away_team    = away_team,
                yes_is_away  = True,
            )
        log.info(f"Tracking {len(self.game_states)} markets")

    async def run_poll(self, session: aiohttp.ClientSession) -> bool:
        nba_task   = asyncio.create_task(fetch_nba_scoreboard(session))
        book_tasks = {
            mid: asyncio.create_task(
                self.poly.get_midpoint(session, gs.yes_token_id)
            )
            for mid, gs in self.game_states.items() if gs.active
        }

        nba_data = await nba_task
        games    = nba_data.get("scoreboard", {}).get("games", [])

        for mid, task in book_tasks.items():
            midpoint = await task
            if midpoint is not None and mid in self.game_states:
                gs            = self.game_states[mid]
                gs.prev_price = gs.yes_price
                gs.prev_ts    = time.time()
                gs.yes_price  = float(midpoint)
                gs.no_price   = round(1.0 - float(midpoint), 4)

        for nba_game in games:
            hn = nba_game.get("homeTeam", {}).get("teamName", "").lower()
            an = nba_game.get("awayTeam", {}).get("teamName", "").lower()

            matched_gs = None
            for gs in self.game_states.values():
                if not gs.active:
                    continue
                if (gs.home_team in hn or hn in gs.home_team) and \
                   (gs.away_team in an or an in gs.away_team):
                    matched_gs = gs
                    break

            if matched_gs is None:
                continue

            if matched_gs.game_id == matched_gs.market_id:
                matched_gs.game_id = nba_game.get(
                    "gameId", matched_gs.game_id
                )

            update_game_state_from_nba(matched_gs, nba_game)

            status = nba_game.get("gameStatus", 1)
            if status == 3 and matched_gs.secs_remaining < 1:
                matched_gs.active = False
                log.info(f"Game over: {matched_gs.away_team} "
                         f"{matched_gs.away_score} @ "
                         f"{matched_gs.home_team} {matched_gs.home_score}")
                continue

            state = self.compute_features(matched_gs)
            await self.check_signals(matched_gs, state, session)
            await self.manage_positions(matched_gs, state, session)

        live = [(gs.away_team, gs.home_team, gs.period,
                 gs.secs_remaining, gs.yes_price, gs.poly_espn_div,
                 gs.poly_mom_60s)
                for gs in self.game_states.values()
                if gs.active and gs.period >= 1 and gs.secs_remaining > 0]
        if live:
            log.info("Live: " + " | ".join(
                f"{a}@{h} P{p} {s:.0f}s "
                f"price={pr:.3f} div={d:+.3f} mom={m:+.3f}"
                for a, h, p, s, pr, d, m in live
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

    def _print_session_summary(self, session_num: int) -> None:
        closed = [t for t in self.all_closed if t.get("pnl_usdc") is not None]
        if not closed:
            log.info(f"Session {session_num}: no closed trades")
            return
        df = pd.DataFrame(closed)
        log.info(f"\nSession {session_num} summary:")
        log.info(f"  Trades   : {len(df)}")
        log.info(f"  Win rate : {(df['pnl_usdc']>0).mean()*100:.1f}%")
        log.info(f"  Avg PnL  : ${df['pnl_usdc'].mean():.4f}")
        log.info(f"  Total    : ${df['pnl_usdc'].sum():.2f}")
        log.info(f"  Bankroll : ${self.bankroll:.2f}")

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
        sharpe   = (avg_ret / std_ret) if std_ret > 0 else 0

        log.info("\n" + "="*60)
        log.info("FINAL PAPER TRADING SUMMARY")
        log.info(f"  Total trades  : {n}")
        log.info(f"  Win rate      : {win_rate*100:.1f}%")
        log.info(f"  Avg PnL/trade : ${avg_ret:.4f}")
        log.info(f"  Total PnL     : ${df['pnl_usdc'].sum():+.2f}")
        log.info(f"  Final BK      : ${self.bankroll:.2f}")
        log.info(f"  Return        : "
                 f"{(self.bankroll-INITIAL_BK)/INITIAL_BK*100:+.1f}%")
        log.info(f"  Sharpe        : {sharpe:.4f}  "
                 f"(raw per-trade, no annualisation)")
        if "cost_usdc" in df.columns:
            log.info(f"  Total costs   : "
                     f"${df['cost_usdc'].sum():.2f}")
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
        log.info("POLYMARKET NBA BOT STARTING")
        log.info(f"  Mode         : "
                 f"{'PAPER' if self.poly.paper_mode else 'LIVE'}")
        log.info(f"  Bankroll     : ${INITIAL_BK:.2f}")
        log.info(f"  Signals      : {len(SIGNALS)}")
        log.info(f"  Target       : {MIN_PAPER_TRADES} closed trades")
        log.info(f"  All signals use divergence-widening filter")
        log.info(f"  Cooldown     : {SIGNAL_COOLDOWN_SECS}s per signal/game")
        log.info(f"  Max open/game: {MAX_OPEN_PER_GAME}")
        log.info("  Signals active:")
        for s in SIGNALS:
            log.info(f"    {s.name}  kelly={s.kelly:.3f}  {s.description}")
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
                async with aiohttp.ClientSession(connector=connector) as session:
                    self.game_states = {}
                    self.positions   = []

                    await self.discover_markets(session)

                    if not self.game_states:
                        log.info(f"No markets to track. "
                                 f"Waiting {BETWEEN_SESSION_WAIT}s...")
                        await asyncio.sleep(BETWEEN_SESSION_WAIT)
                        continue

                    log.info(f"Polling (pregame={POLL_INTERVAL_PREGAME}s, "
                             f"live={POLL_INTERVAL_LIVE}s)...")

                    while True:
                        t0 = time.time()

                        if self.bankroll < MIN_BANKROLL:
                            log.error(
                                f"Bankroll ${self.bankroll:.2f} below "
                                f"minimum ${MIN_BANKROLL:.2f} — stopping."
                            )
                            self._save_trade_log()
                            self._print_summary()
                            return

                        if all(not gs.active
                               for gs in self.game_states.values()):
                            log.info("All games finished for this session.")
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
                self._print_session_summary(session_num)
                log.info(
                    f"Waiting {BETWEEN_SESSION_WAIT}s for next session..."
                )
                await asyncio.sleep(BETWEEN_SESSION_WAIT)

        except KeyboardInterrupt:
            log.info("Stopped by user.")
        finally:
            self._save_trade_log()
            self._print_summary()


if __name__ == "__main__":
    asyncio.run(LiveBot().run())
# main.py
# Crypto-AI-Bot + Pet (Survival-Driven Trade Hunter)
# ---------------------------------------------------
# Design goals:
# - Pet IS the bot: survival stats drive risk, filtering, and aggression.
# - Hunt “best meals”: risk-adjusted expectancy, not penny-stock style chaos.
# - Avoid death spirals: volatility targeting, cooldowns, time stops, trailing, breakeven.
# - Simple memory: reward markets that feed, penalize markets that hurt.

import os
import json
import time
import math
import logging
import traceback
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("crypto-ai-bot")

# ============================================================
# Env helpers
# ============================================================

def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v.strip() if v and v.strip() else default

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def utc_now_iso() -> str:
    return utc_now().isoformat()

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def normalize_base_url(u: str) -> str:
    """
    Accepts:
      - http://host:port
      - https://host
      - host:port
      - host
    Returns: URL with scheme, without trailing slash.
    """
    u = (u or "").strip()
    if not u:
        return ""
    u = u.rstrip("/")
    if "://" not in u:
        # default to http for internal hosts (Render private)
        u = "http://" + u
    return u.rstrip("/")

# ============================================================
# Config
# ============================================================

# IMPORTANT:
# - API_URL: where bot POSTs ingest events/trades/heartbeat + where dashboard reads /data
# - PRICE_API_BASE: where bot GETs /prices and /history (can be same as API_URL)
API_URL = normalize_base_url(env_str("API_URL", ""))  # e.g. http://crypto-ai-api-h921:10000 OR https://crypto-ai-api-h921.onrender.com
PRICE_API_BASE = normalize_base_url(env_str("PRICE_API_BASE", ""))  # optional separate price service

MARKETS = [m.strip() for m in env_str("MARKETS", "BTC-USD,ETH-USD,SOL-USD").split(",") if m.strip()]
CYCLE_SECONDS = env_int("CYCLE_SECONDS", 60)

# Paper equity
START_EQUITY = env_float("START_EQUITY", 1000.0)

# Position limits
MAX_OPEN_POSITIONS = env_int("MAX_OPEN_POSITIONS", 2)

# Indicators
RSI_PERIOD = env_int("RSI_PERIOD", 14)
SMA_FAST = env_int("SMA_FAST", 12)
SMA_SLOW = env_int("SMA_SLOW", 36)

# Risk baseline (pet will scale these)
RISK_PER_TRADE_PCT = env_float("RISK_PER_TRADE_PCT", 0.50)   # baseline % equity at risk per trade
STOP_LOSS_PCT = env_float("STOP_LOSS_PCT", 0.80)             # default stop distance (%)
TAKE_PROFIT_PCT = env_float("TAKE_PROFIT_PCT", 1.20)         # default take distance (%)

# Quality filters (avoid “penny stock crypto” behavior)
VOL_MIN = env_float("VOL_MIN", 0.004)   # 0.4% (too dead below this)
VOL_MAX = env_float("VOL_MAX", 0.030)   # 3.0% (too wild above this)
TREND_MIN = env_float("TREND_MIN", 0.0020)  # min trend strength

# Decision thresholds
MIN_CONFIDENCE = env_float("MIN_CONFIDENCE", 0.60)

# Trade management
TIME_STOP_MINUTES = env_int("TIME_STOP_MINUTES", 90)       # exit if stale
BREAKEVEN_R = env_float("BREAKEVEN_R", 0.60)               # move stop to entry after +0.6R
TRAIL_START_R = env_float("TRAIL_START_R", 1.00)           # start trailing after +1.0R
TRAIL_DIST_R = env_float("TRAIL_DIST_R", 0.60)             # trail distance measured in R

# Cooldowns after damage
MARKET_COOLDOWN_MIN = env_int("MARKET_COOLDOWN_MIN", 30)
HURT_COOLDOWN_MIN = env_int("HURT_COOLDOWN_MIN", 60)

# Memory shaping
MEMORY_HALF_LIFE = env_float("MEMORY_HALF_LIFE", 0.90)      # exponential decay factor per cycle (~0.9 good)
MEMORY_BONUS_MAX = env_float("MEMORY_BONUS_MAX", 8.0)       # max points added to edge score

# Safety
MIN_TRADE_USD = env_float("MIN_TRADE_USD", 10.0)
MAX_POS_FRACTION = env_float("MAX_POS_FRACTION", 0.25)      # max fraction of equity in one position


# ============================================================
# HTTP helpers
# ============================================================

def http_post_json(url: str, payload: dict, timeout: int = 20) -> bool:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "crypto-ai-bot/2.0"},
            method="POST",
        )
        with urlopen(req, timeout=timeout) as resp:
            _ = resp.read()
        return True
    except Exception as e:
        log.warning(f"POST failed {url}: {e}")
        return False

def http_get_json(url: str, timeout: int = 20) -> Optional[dict]:
    try:
        req = Request(url, headers={"User-Agent": "crypto-ai-bot/2.0"})
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
        log.warning(f"GET failed {url}: {e}")
        return None
    except Exception as e:
        log.warning(f"GET failed {url}: {e}")
        return None

# ============================================================
# Indicators
# ============================================================

def sma(values: List[float], period: int) -> Optional[float]:
    if period <= 0 or len(values) < period:
        return None
    return sum(values[-period:]) / period

def rsi(values: List[float], period: int) -> Optional[float]:
    if period <= 0 or len(values) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))

def volatility(values: List[float], period: int = 30) -> float:
    if len(values) < period + 1:
        return 0.0
    rets = []
    for i in range(-period, 0):
        prev = values[i - 1]
        cur = values[i]
        if prev > 0:
            rets.append((cur - prev) / prev)
    if not rets:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / len(rets)
    return math.sqrt(var)

# ============================================================
# Price feed
# Expected:
#   GET {BASE}/prices -> {"BTC-USD": 123, ...}
#   GET {BASE}/history?market=BTC-USD&limit=180 -> {"closes":[...]}  (or {"data":[{"close":...}, ...]})
# ============================================================

def price_base() -> str:
    return (PRICE_API_BASE or API_URL).rstrip("/")

def fetch_prices(markets: List[str]) -> Dict[str, float]:
    base = price_base()
    if not base:
        return {}
    data = http_get_json(f"{base}/prices")
    out: Dict[str, float] = {}
    if isinstance(data, dict):
        for m in markets:
            v = data.get(m)
            if isinstance(v, (int, float)) and v > 0:
                out[m] = float(v)
    return out

def fetch_history(market: str, limit: int = 180) -> List[float]:
    base = price_base()
    if not base:
        return []
    data = http_get_json(f"{base}/history?market={market}&limit={limit}")
    if not isinstance(data, dict):
        return []
    closes: List[float] = []

    if isinstance(data.get("closes"), list):
        for x in data["closes"]:
            if isinstance(x, (int, float)) and x > 0:
                closes.append(float(x))
        return closes

    if isinstance(data.get("data"), list):
        for row in data["data"]:
            if isinstance(row, dict):
                c = row.get("close")
                if isinstance(c, (int, float)) and c > 0:
                    closes.append(float(c))
        return closes

    return []

# ============================================================
# Models
# ============================================================

@dataclass
class Pet:
    # identity: pet == bot
    stage: str = "egg"            # egg -> hatched -> hunter -> apex
    mood: str = "sleepy"
    health: float = 100.0         # 0..100
    hunger: float = 60.0          # 0..100 (higher = hungrier)
    growth: float = 0.0           # 0..100
    focus: float = 50.0           # 0..100 (how “disciplined” it is)
    fainted_until_utc: str = ""   # temporary “injury” timeout window
    last_update_utc: str = ""

@dataclass
class Position:
    market: str
    side: str
    entry_price: float
    size_usd: float
    stop_price: float
    take_price: float
    opened_time_utc: str
    confidence: float
    reason: str

    # management fields
    r_dist: float = 0.0           # entry - stop (absolute)
    moved_to_be: bool = False
    peak_price: float = 0.0

@dataclass
class Trade:
    entry_time: str
    exit_time: str
    hold_minutes: float
    market: str
    entry_price: float
    exit_price: float
    qty: float
    pnl_usd: float
    pnl_pct: float
    take_profit_pct: float
    stop_loss_pct: float
    risk_mode: str
    trend_strength: float
    rsi: float
    volatility: float
    confidence: float
    reason: str

@dataclass
class MarketMemory:
    # “did this market feed me?”
    score: float = 0.0            # positive = good hunting ground
    recent_wins: int = 0
    recent_losses: int = 0
    last_trade_utc: str = ""
    cooldown_until_utc: str = ""  # avoid after hurt / chop

# ============================================================
# Engine (Pet-driven)
# ============================================================

class Engine:
    def __init__(self):
        self.lock = threading.Lock()

        # equity + positions
        self.equity_usd = START_EQUITY
        self.peak_equity_usd = START_EQUITY
        self.positions: List[Position] = []

        # pet state
        self.pet = Pet()

        # stats
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0

        # memory per market
        self.memory: Dict[str, MarketMemory] = {m: MarketMemory() for m in MARKETS}

        # anti-spam
        self._sound_cooldown_until = 0.0

    # ---------------- Pet / identity ----------------

    def _is_fainted(self) -> bool:
        if not self.pet.fainted_until_utc:
            return False
        try:
            until = datetime.fromisoformat(self.pet.fainted_until_utc.replace("Z", "+00:00"))
            return utc_now() < until
        except Exception:
            return False

    def _set_faint_timeout_minutes(self, minutes: int):
        until = utc_now().timestamp() + minutes * 60
        self.pet.fainted_until_utc = datetime.fromtimestamp(until, tz=timezone.utc).isoformat()

    def _set_market_cooldown(self, market: str, minutes: int):
        until = utc_now().timestamp() + minutes * 60
        self.memory.setdefault(market, MarketMemory()).cooldown_until_utc = datetime.fromtimestamp(until, tz=timezone.utc).isoformat()

    def _market_is_cooled_down(self, market: str) -> bool:
        mm = self.memory.get(market)
        if not mm or not mm.cooldown_until_utc:
            return False
        try:
            until = datetime.fromisoformat(mm.cooldown_until_utc.replace("Z", "+00:00"))
            return utc_now() < until
        except Exception:
            return False

    def _pet_tick(self):
        # time-based drift
        self.pet.last_update_utc = utc_now_iso()

        # hunger rises slowly (survival pressure)
        self.pet.hunger = clamp(self.pet.hunger + 1.5, 0.0, 100.0)

        # focus drifts down if hungry, up if healthy
        if self.pet.hunger > 80:
            self.pet.focus = clamp(self.pet.focus - 1.0, 0.0, 100.0)
        else:
            self.pet.focus = clamp(self.pet.focus + 0.3, 0.0, 100.0)

        # health reacts to hunger
        if self.pet.hunger > 88:
            self.pet.health = clamp(self.pet.health - 1.8, 0.0, 100.0)
        elif self.pet.hunger < 45:
            self.pet.health = clamp(self.pet.health + 0.6, 0.0, 100.0)

        # stage progression by “proof of life”
        # (consistent feeding is what hatches/levels-up the pet)
        if self.pet.stage == "egg" and (self.wins >= 5 or self.total_pnl > 0):
            self.pet.stage = "hatched"
            self.pet.growth = max(self.pet.growth, 10.0)
            self._emit_event("status", "pet_hatched", {"wins": self.wins, "pnl": self.total_pnl})

        if self.pet.stage == "hatched" and (self.wins >= 15 or self.total_pnl > (0.05 * START_EQUITY)):
            self.pet.stage = "hunter"
            self.pet.growth = max(self.pet.growth, 35.0)
            self._emit_event("status", "pet_became_hunter", {"wins": self.wins, "pnl": self.total_pnl})

        if self.pet.stage == "hunter" and (self.wins >= 35 or self.total_pnl > (0.15 * START_EQUITY)):
            self.pet.stage = "apex"
            self.pet.growth = max(self.pet.growth, 70.0)
            self._emit_event("status", "pet_became_apex", {"wins": self.wins, "pnl": self.total_pnl})

        # mood
        if self._is_fainted():
            self.pet.mood = "dizzy"
        elif self.pet.health < 25:
            self.pet.mood = "injured"
        elif self.pet.hunger > 90:
            self.pet.mood = "starving"
        elif self.pet.hunger > 75:
            self.pet.mood = "hungry"
        else:
            self.pet.mood = "focused"

    def survival_mode(self) -> str:
        # ordered by severity
        if self._is_fainted():
            return "DIZZY"
        if self.pet.health < 22:
            return "INJURED"
        if self.pet.hunger > 92:
            return "STARVING"
        if self.pet.hunger > 78:
            return "HUNGRY"
        return "NORMAL"

    # ---------------- Events / sounds ----------------

    def _emit_event(self, etype: str, message: str, details: dict):
        if not API_URL:
            return
        http_post_json(f"{API_URL}/ingest/event", {
            "type": etype,
            "message": message,
            "details": details,
            "time_utc": utc_now_iso(),
        })

    def _emit_sound(self, kind: str, details: dict):
        now = time.time()
        if now < self._sound_cooldown_until:
            return
        self._sound_cooldown_until = now + 20.0
        self._emit_event("sound", kind, details)

    # ---------------- Memory shaping ----------------

    def _memory_decay(self):
        # decay scores slowly each cycle so bot can “forgive” markets
        for m, mm in self.memory.items():
            mm.score *= MEMORY_HALF_LIFE

    def _memory_update_on_trade(self, market: str, pnl: float):
        mm = self.memory.setdefault(market, MarketMemory())
        mm.last_trade_utc = utc_now_iso()

        # bounded reward/punish
        # winning feeds: increases market score
        # losing harms: decreases market score more strongly
        if pnl >= 0:
            mm.recent_wins += 1
            mm.score += clamp(abs(pnl) / 10.0, 0.5, 4.0)
        else:
            mm.recent_losses += 1
            mm.score -= clamp(abs(pnl) / 8.0, 0.8, 6.0)
            # if it hurt badly, cooldown market longer
            self._set_market_cooldown(market, HURT_COOLDOWN_MIN)

        # clamp score so it doesn't blow up
        mm.score = clamp(mm.score, -20.0, 20.0)

    def memory_bonus(self, market: str) -> float:
        mm = self.memory.get(market)
        if not mm:
            return 0.0
        # convert score to small edge bonus
        return clamp(mm.score, -MEMORY_BONUS_MAX, MEMORY_BONUS_MAX)

    # ---------------- Risk control ----------------

    def drawdown_pct(self) -> float:
        self.peak_equity_usd = max(self.peak_equity_usd, self.equity_usd)
        if self.peak_equity_usd <= 0:
            return 0.0
        return max(0.0, (self.peak_equity_usd - self.equity_usd) / self.peak_equity_usd) * 100.0

    def adaptive_conf_threshold(self) -> float:
        base = MIN_CONFIDENCE

        mode = self.survival_mode()
        dd = self.drawdown_pct()

        # Injured/dizzy -> become picky
        if mode in ("DIZZY", "INJURED"):
            base += 0.06

        # hungry -> slightly less picky, but still filtered by quality rules
        if mode == "HUNGRY":
            base -= 0.02
        if mode == "STARVING":
            base -= 0.03

        # drawdown -> tighten
        if dd > 8.0:
            base += 0.04
        if dd > 15.0:
            base += 0.06

        # focus helps discipline
        if self.pet.focus < 35:
            base += 0.03

        return clamp(base, 0.52, 0.78)

    def adaptive_risk_multiplier(self, market_vol: float) -> float:
        """
        Volatility targeting + survival scaling.
        """
        mode = self.survival_mode()
        dd = self.drawdown_pct()

        # baseline multiplier
        mult = 1.0

        # survival scaling
        if mode == "STARVING":
            mult *= 1.05   # slightly more urgent
        if mode == "HUNGRY":
            mult *= 1.00
        if mode == "INJURED":
            mult *= 0.60
        if mode == "DIZZY":
            mult *= 0.35

        # drawdown protection
        if dd > 10.0:
            mult *= 0.70
        if dd > 18.0:
            mult *= 0.55

        # volatility targeting: aim around ~1.2% “comfortable” vol
        target = 0.012
        if market_vol > 0:
            vol_scale = clamp(target / market_vol, 0.35, 1.35)
            mult *= vol_scale

        # health scaling
        if self.pet.health < 30:
            mult *= 0.70
        elif self.pet.health < 50:
            mult *= 0.85

        return clamp(mult, 0.15, 1.45)

    # ---------------- Strategy scoring ----------------

    def market_quality_ok(self, vol: float, trend_strength: float) -> Tuple[bool, str]:
        if vol <= 0:
            return (False, "no_vol")
        if vol < VOL_MIN:
            return (False, "too_dead")
        if vol > VOL_MAX:
            return (False, "too_wild")
        if trend_strength < TREND_MIN:
            return (False, "no_trend")
        return (True, "ok")

    def score_market(self, market: str, closes: List[float]) -> Tuple[float, float, str, dict]:
        """
        Returns:
          edge_score (higher is better),
          confidence (0..1),
          reason_str,
          metrics dict
        """
        need = max(RSI_PERIOD + 2, SMA_SLOW + 2, 40)
        if len(closes) < need:
            return (0.0, 0.0, "not_enough_data", {"rsi": None, "trend_strength": 0.0, "vol": 0.0, "last": (closes[-1] if closes else 0.0)})

        r = rsi(closes, RSI_PERIOD)
        f = sma(closes, SMA_FAST)
        s = sma(closes, SMA_SLOW)
        vol = volatility(closes, period=30)

        if r is None or f is None or s is None:
            return (0.0, 0.0, "indicator_unavailable", {"rsi": None, "trend_strength": 0.0, "vol": vol, "last": closes[-1]})

        last = closes[-1]
        trend_up = f > s
        trend_strength = abs((f - s) / max(last, 1e-9))

        ok, whyq = self.market_quality_ok(vol, trend_strength)
        if not ok:
            # still report metrics but do not trade
            reason = f"filtered:{whyq}|rsi={r:.1f}|trendS={trend_strength:.4f}|vol={vol:.4f}|px={last:.2f}"
            metrics = {"rsi": float(r), "trend_strength": float(trend_strength), "vol": float(vol), "last": float(last), "quality": whyq}
            return (-9999.0, 0.0, reason, metrics)

        # Confidence model (simple but disciplined)
        conf = 0.52
        bits = []

        if trend_up:
            conf += 0.10
            bits.append("trend_up")
        else:
            conf -= 0.10
            bits.append("trend_down")

        # RSI: prefer dips in uptrend; avoid chasing tops
        if trend_up:
            if r < 35:
                conf += 0.14
                bits.append("dip_buy")
            elif r < 50:
                conf += 0.06
                bits.append("mild_dip")
            elif r > 70:
                conf -= 0.12
                bits.append("too_hot")
        else:
            # downtrend: do not hunt aggressively
            if r < 30:
                conf += 0.03
                bits.append("maybe_bounce")
            else:
                conf -= 0.06
                bits.append("avoid_downtrend")

        # volatility penalty within “ok” range
        # prefer moderate vol
        if vol > 0.020:
            conf -= 0.04
            bits.append("high_vol")
        elif vol < 0.007:
            conf -= 0.02
            bits.append("low_vol")

        # memory bonus
        mem = self.memory_bonus(market)
        conf += (mem / 100.0)  # tiny influence on confidence

        conf = clamp(conf, 0.0, 1.0)

        # Edge score: confidence + trend_strength + memory - vol penalty
        edge = (conf * 100.0) + (trend_strength * 1200.0) + mem - (vol * 450.0)

        reason = "|".join(bits) + f"|mem={mem:.2f}|rsi={r:.1f}|trendS={trend_strength:.4f}|vol={vol:.4f}|px={last:.2f}"
        metrics = {"rsi": float(r), "trend_strength": float(trend_strength), "vol": float(vol), "last": float(last), "quality": "ok", "mem": float(mem)}
        return (edge, conf, reason, metrics)

    # ---------------- Trading ----------------

    def can_open(self, market: str) -> bool:
        if len(self.positions) >= MAX_OPEN_POSITIONS:
            return False
        if any(p.market == market for p in self.positions):
            return False
        if self._market_is_cooled_down(market):
            return False
        return True

    def open_long(self, market: str, price: float, confidence: float, reason: str, metrics: dict):
        if not self.can_open(market):
            return

        vol = float(metrics.get("vol", 0.0))
        risk_mult = self.adaptive_risk_multiplier(vol)
        risk_pct = RISK_PER_TRADE_PCT * risk_mult
        risk_usd = self.equity_usd * (risk_pct / 100.0)

        # Stops / takes
        sl = price * (1.0 - STOP_LOSS_PCT / 100.0)
        tp = price * (1.0 + TAKE_PROFIT_PCT / 100.0)

        r_dist = max(price - sl, 1e-9)

        # position size so that loss at stop ~= risk_usd
        # loss = (r_dist/price) * size_usd  => size_usd = risk_usd * (price/r_dist)
        size_usd = risk_usd * (price / r_dist)

        # cap size
        size_usd = clamp(size_usd, MIN_TRADE_USD, self.equity_usd * MAX_POS_FRACTION)

        pos = Position(
            market=market,
            side="LONG",
            entry_price=float(price),
            size_usd=float(size_usd),
            stop_price=float(sl),
            take_price=float(tp),
            opened_time_utc=utc_now_iso(),
            confidence=float(confidence),
            reason=reason,
            r_dist=float(r_dist),
            moved_to_be=False,
            peak_price=float(price),
        )
        self.positions.append(pos)

        self._emit_event("thought", "hunt_open", {
            "market": market,
            "confidence": confidence,
            "mode": self.survival_mode(),
            "size_usd": size_usd,
            "risk_pct": risk_pct,
            "vol": vol,
            "reason": reason[:250],
        })

        log.info(f"OPEN {market} size=${size_usd:.2f} entry={price:.2f} sl={sl:.2f} tp={tp:.2f} conf={confidence:.2f} mode={self.survival_mode()}")

    def _position_age_minutes(self, pos: Position) -> float:
        try:
            opened = datetime.fromisoformat(pos.opened_time_utc.replace("Z", "+00:00"))
            return (utc_now() - opened).total_seconds() / 60.0
        except Exception:
            return 0.0

    def close_position(self, idx: int, price: float, why: str, metrics: dict):
        pos = self.positions[idx]

        pnl = ((price - pos.entry_price) / pos.entry_price) * pos.size_usd
        pnl_pct = ((price - pos.entry_price) / pos.entry_price) * 100.0

        self.total_pnl += pnl
        self.total_trades += 1
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1

        self.equity_usd += pnl
        exit_time = utc_now_iso()

        hold_minutes = self._position_age_minutes(pos)
        qty = pos.size_usd / max(pos.entry_price, 1e-9)

        # push trade to API
        if API_URL:
            trade_payload = asdict(Trade(
                entry_time=pos.opened_time_utc,
                exit_time=exit_time,
                hold_minutes=hold_minutes,
                market=pos.market,
                entry_price=pos.entry_price,
                exit_price=float(price),
                qty=float(qty),
                pnl_usd=float(pnl),
                pnl_pct=float(pnl_pct),
                take_profit_pct=float(TAKE_PROFIT_PCT),
                stop_loss_pct=float(STOP_LOSS_PCT),
                risk_mode=self.survival_mode(),
                trend_strength=float(metrics.get("trend_strength", 0.0)),
                rsi=float(metrics.get("rsi", 0.0)),
                volatility=float(metrics.get("vol", 0.0)),
                confidence=float(pos.confidence),
                reason=pos.reason,
            ))
            http_post_json(f"{API_URL}/ingest/trade", trade_payload)

        # update pet + memory
        self._pet_on_trade(pnl, pos.market)

        # emit thought
        self._emit_event("thought", "hunt_close", {
            "market": pos.market,
            "pnl": pnl,
            "why": why,
            "equity": self.equity_usd,
            "mode": self.survival_mode(),
        })

        log.info(f"CLOSE {pos.market} pnl={pnl:.2f} ({why}) equity={self.equity_usd:.2f}")
        self.positions.pop(idx)

        # cooldown after any trade to avoid overtrading the same thing
        self._set_market_cooldown(pos.market, MARKET_COOLDOWN_MIN)

    def _pet_on_trade(self, pnl: float, market: str):
        # market memory update
        self._memory_update_on_trade(market, pnl)

        # feeding / damage
        if pnl > 0:
            self.pet.hunger = clamp(self.pet.hunger - 14.0, 0.0, 100.0)
            self.pet.health = clamp(self.pet.health + 2.8, 0.0, 100.0)
            self.pet.growth = clamp(self.pet.growth + 3.2, 0.0, 100.0)
            self.pet.focus = clamp(self.pet.focus + 1.5, 0.0, 100.0)
            self._emit_sound("purr", {"pnl": pnl, "market": market})
        else:
            self.pet.hunger = clamp(self.pet.hunger + 8.0, 0.0, 100.0)
            self.pet.health = clamp(self.pet.health - 4.3, 0.0, 100.0)
            self.pet.focus = clamp(self.pet.focus - 2.0, 0.0, 100.0)
            self._emit_sound("whimper", {"pnl": pnl, "market": market})

            # faint timeout if severely hurt
            if self.pet.health <= 10 and not self._is_fainted():
                self._set_faint_timeout_minutes(20)
                self._emit_event("status", "pet_fainted_timeout", {"minutes": 20, "market": market})

    # ---------------- Position management (breakeven/trailing/time stop) ----------------

    def manage_positions(self, prices: Dict[str, float]):
        for i in range(len(self.positions) - 1, -1, -1):
            pos = self.positions[i]
            px = prices.get(pos.market)
            if px is None:
                continue

            # update peak
            pos.peak_price = max(pos.peak_price, px)

            # basic exits: stop / take
            if px <= pos.stop_price:
                self.close_position(i, px, "stop_loss", {"rsi": 0.0, "trend_strength": 0.0, "vol": 0.0})
                continue
            if px >= pos.take_price:
                self.close_position(i, px, "take_profit", {"rsi": 0.0, "trend_strength": 0.0, "vol": 0.0})
                continue

            # time stop
            age = self._position_age_minutes(pos)
            if age >= TIME_STOP_MINUTES:
                self.close_position(i, px, "time_stop", {"rsi": 0.0, "trend_strength": 0.0, "vol": 0.0})
                continue

            # R-based management (LONG only)
            r = pos.r_dist
            if r <= 0:
                continue

            # current profit in R
            r_profit = (px - pos.entry_price) / r

            # move stop to breakeven after +BREAKEVEN_R
            if (not pos.moved_to_be) and r_profit >= BREAKEVEN_R:
                pos.stop_price = max(pos.stop_price, pos.entry_price)  # never lower stop
                pos.moved_to_be = True
                self._emit_event("thought", "hunt_protect_breakeven", {
                    "market": pos.market,
                    "r_profit": r_profit,
                    "stop": pos.stop_price,
                })

            # trailing stop after +TRAIL_START_R
            if r_profit >= TRAIL_START_R:
                # trail = peak - TRAIL_DIST_R * R
                trail_stop = pos.peak_price - (TRAIL_DIST_R * r)
                pos.stop_price = max(pos.stop_price, trail_stop)

    # ---------------- Cycle ----------------

    def push_state(self, prices_ok: bool):
        hb_time = utc_now_iso()

        if API_URL:
            # heartbeat
            http_post_json(f"{API_URL}/ingest/heartbeat", {
                "status": "running",
                "time_utc": hb_time,
                "markets": MARKETS,
                "open_positions": len(self.positions),
                "equity_usd": self.equity_usd,
                "wins": self.wins,
                "losses": self.losses,
                "total_trades": self.total_trades,
                "total_pnl_usd": self.total_pnl,
                "mode": self.survival_mode(),
                "prices_ok": prices_ok,
                "drawdown_pct": self.drawdown_pct(),
            })

            # equity point
            http_post_json(f"{API_URL}/ingest/equity", {
                "time_utc": hb_time,
                "equity_usd": self.equity_usd,
            })

            # pet
            http_post_json(f"{API_URL}/ingest/pet", {
                **asdict(self.pet),
                "time_utc": hb_time,
                "mode": self.survival_mode(),
                "equity_usd": self.equity_usd,
                "drawdown_pct": self.drawdown_pct(),
            })

    def run_cycle(self):
        with self.lock:
            self._pet_tick()
            self._memory_decay()

        prices = fetch_prices(MARKETS)
        if not prices:
            self.push_state(prices_ok=False)
            log.info("No prices -> idling safely.")
            return

        with self.lock:
            # manage existing positions first
            self.manage_positions(prices)

        # score markets
        scored: List[Tuple[float, str, float, str, dict]] = []
        for m in MARKETS:
            if m not in prices:
                continue
            if self._market_is_cooled_down(m):
                # small “thought” occasionally
                continue

            closes = fetch_history(m, limit=200)
            if not closes:
                # fallback: flat synthetic (won't pass quality)
                closes = [prices[m]] * max(60, SMA_SLOW + RSI_PERIOD + 5)

            edge, conf, reason, metrics = self.score_market(m, closes)
            scored.append((edge, m, conf, reason, metrics))

        scored.sort(reverse=True, key=lambda x: x[0])

        conf_th = self.adaptive_conf_threshold()

        # choose how many new positions to open
        with self.lock:
            slots = MAX_OPEN_POSITIONS - len(self.positions)

        if slots > 0 and scored:
            opened = 0
            best = scored[0]

            for edge, m, conf, reason, metrics in scored:
                if opened >= slots:
                    break
                if edge < -1000:
                    continue
                if not self.can_open(m):
                    continue

                # Only take uptrend longs for now
                if "trend_up" not in reason:
                    continue

                # Minimum confidence
                if conf < conf_th:
                    continue

                # Extra: in INJURED/DIZZY, require a bit more “edge”
                mode = self.survival_mode()
                if mode in ("INJURED", "DIZZY") and edge < 65:
                    continue

                # open
                with self.lock:
                    self.open_long(m, prices[m], conf, reason, metrics)
                opened += 1

            if opened == 0:
                # “searching” thought when hungry
                if self.survival_mode() in ("HUNGRY", "STARVING"):
                    self._emit_event("thought", "hunt_searching", {
                        "mode": self.survival_mode(),
                        "conf_threshold": conf_th,
                        "best_market": best[1],
                        "best_edge": best[0],
                        "best_conf": best[2],
                        "best_reason": best[3][:220],
                    })

        # push state to API
        self.push_state(prices_ok=True)

# ============================================================
# Main loop
# ============================================================

ENGINE = Engine()

def bot_loop():
    log.info("Bot+Pet starting.")
    log.info(f"API_URL={API_URL or '(none)'} PRICE_API_BASE={PRICE_API_BASE or '(same as API_URL)'}")
    log.info(f"MARKETS={MARKETS} CYCLE_SECONDS={CYCLE_SECONDS}s MAX_OPEN_POSITIONS={MAX_OPEN_POSITIONS}")

    if not API_URL:
        log.warning("API_URL is empty. Set API_URL to your API service URL (internal or public).")
        log.warning("Example internal: http://crypto-ai-api-h921:10000")
        log.warning("Example public:   https://crypto-ai-api-h921.onrender.com")

    while True:
        try:
            ENGINE.run_cycle()
        except Exception as e:
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            log.error(f"Cycle crashed: {e}\n{err}")
            if API_URL:
                http_post_json(f"{API_URL}/ingest/training_event", {
                    "time_utc": utc_now_iso(),
                    "event": "cycle_crash",
                    "details": str(e)[:250],
                })
        time.sleep(max(5, CYCLE_SECONDS))

def main():
    bot_loop()

if __name__ == "__main__":
    main()

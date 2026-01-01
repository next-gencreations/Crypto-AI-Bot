import os
import json
import time
import math
import logging
import traceback
import threading
from dataclasses import dataclass, asdict
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

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ============================================================
# Config
# ============================================================

# Internal Render URL recommended:
# API_URL = http://crypto-ai-api-h921:10000
API_URL = env_str("API_URL", "").rstrip("/")

MARKETS = [m.strip() for m in env_str("MARKETS", "BTC-USD,ETH-USD").split(",") if m.strip()]
CYCLE_SECONDS = env_int("CYCLE_SECONDS", 360)

START_EQUITY = env_float("START_EQUITY", 1000.0)
MAX_OPEN_POSITIONS = env_int("MAX_OPEN_POSITIONS", 2)

# indicators
RSI_PERIOD = env_int("RSI_PERIOD", 14)
SMA_FAST = env_int("SMA_FAST", 10)
SMA_SLOW = env_int("SMA_SLOW", 30)

# risk
RISK_PER_TRADE_PCT = env_float("RISK_PER_TRADE_PCT", 0.5)  # % equity at risk per trade (baseline)
STOP_LOSS_PCT = env_float("STOP_LOSS_PCT", 0.8)
TAKE_PROFIT_PCT = env_float("TAKE_PROFIT_PCT", 1.2)

# baseline thresholds (will adapt based on "hunger")
MIN_CONFIDENCE = env_float("MIN_CONFIDENCE", 0.58)

# API paths (your price API is expected behind API_URL; if you don’t have it, bot will idle safely)
PRICE_API_BASE = env_str("PRICE_API_BASE", "").rstrip("/")  # optional separate price feed base

# ============================================================
# Models
# ============================================================

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
class Pet:
    stage: str = "egg"            # egg -> hatched
    mood: str = "sleepy"
    health: float = 100.0         # 0..100
    hunger: float = 60.0          # 0..100 (higher = hungrier)
    growth: float = 0.0           # 0..100
    fainted_until_utc: str = ""   # timeout timestamp (pet can't die permanently)
    last_update_utc: str = ""

# ============================================================
# HTTP helpers (API post)
# ============================================================

def http_post_json(url: str, payload: dict, timeout: int = 15) -> bool:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "crypto-ai-bot/1.0"},
            method="POST",
        )
        with urlopen(req, timeout=timeout) as resp:
            _ = resp.read()
        return True
    except Exception as e:
        log.warning(f"POST failed {url}: {e}")
        return False

def http_get_json(url: str, timeout: int = 15) -> Optional[dict]:
    try:
        req = Request(url, headers={"User-Agent": "crypto-ai-bot/1.0"})
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
    # simple pct-change stddev
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
# Price feed (you can point PRICE_API_BASE to your market data service)
# Expected:
#   GET {BASE}/prices -> {"BTC-USD": 123, ...}
#   GET {BASE}/history?market=BTC-USD&limit=180 -> {"closes":[...]}
# ============================================================

def price_base() -> str:
    return PRICE_API_BASE or API_URL  # fallback

def fetch_prices(markets: List[str]) -> Dict[str, float]:
    base = price_base().rstrip("/")
    if not base:
        return {}
    bulk = http_get_json(f"{base}/prices")
    out: Dict[str, float] = {}
    if isinstance(bulk, dict):
        for m in markets:
            v = bulk.get(m)
            if isinstance(v, (int, float)) and v > 0:
                out[m] = float(v)
    return out

def fetch_history(market: str, limit: int = 180) -> List[float]:
    base = price_base().rstrip("/")
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
# Engine
# ============================================================

class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.equity_usd = START_EQUITY
        self.positions: List[Position] = []
        self.pet = Pet()
        self.last_heartbeat_utc = ""

        # stats
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0

        self._sound_cooldown_until = 0.0  # avoid spamming events

    # ---------------- Pet logic ----------------

    def _is_fainted(self) -> bool:
        if not self.pet.fainted_until_utc:
            return False
        try:
            until = datetime.fromisoformat(self.pet.fainted_until_utc.replace("Z", "+00:00"))
            return datetime.now(timezone.utc) < until
        except Exception:
            return False

    def _set_faint_timeout_minutes(self, minutes: int):
        until = datetime.now(timezone.utc).timestamp() + minutes * 60
        until_iso = datetime.fromtimestamp(until, tz=timezone.utc).isoformat()
        self.pet.fainted_until_utc = until_iso

    def _pet_tick(self):
        self.pet.last_update_utc = utc_now_iso()

        # hunger rises steadily
        self.pet.hunger = clamp(self.pet.hunger + 2.0, 0.0, 100.0)

        # health reacts to hunger, but never "dies" permanently
        if self.pet.hunger > 85:
            self.pet.health = clamp(self.pet.health - 2.0, 0.0, 100.0)
        elif self.pet.hunger < 40:
            self.pet.health = clamp(self.pet.health + 0.6, 0.0, 100.0)

        # mood
        if self._is_fainted():
            self.pet.mood = "dizzy"
        elif self.pet.health < 25:
            self.pet.mood = "sad"
        elif self.pet.hunger > 85:
            self.pet.mood = "hungry"
        else:
            self.pet.mood = "focused"

        # hatch rule: when bot proves itself (total pnl positive OR wins >= 5)
        if self.pet.stage == "egg" and (self.total_pnl > 0 or self.wins >= 5):
            self.pet.stage = "hatched"
            self.pet.growth = max(self.pet.growth, 10.0)

    def _pet_on_trade(self, pnl: float):
        if pnl > 0:
            # food
            self.pet.hunger = clamp(self.pet.hunger - 12.0, 0.0, 100.0)
            self.pet.health = clamp(self.pet.health + 2.5, 0.0, 100.0)
            self.pet.growth = clamp(self.pet.growth + 3.0, 0.0, 100.0)
            self._emit_sound("purr", {"pnl": pnl})
        else:
            # pain
            self.pet.hunger = clamp(self.pet.hunger + 7.0, 0.0, 100.0)
            self.pet.health = clamp(self.pet.health - 4.0, 0.0, 100.0)
            self._emit_sound("whimper", {"pnl": pnl})

            # faint timeout if health too low, BUT bot still trades reduced-size
            if self.pet.health <= 10 and not self._is_fainted():
                self._set_faint_timeout_minutes(20)
                self._emit_event("status", "pet_fainted_timeout", {"minutes": 20})

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
        # avoid spamming every second
        now = time.time()
        if now < self._sound_cooldown_until:
            return
        self._sound_cooldown_until = now + 25.0

        self._emit_event("sound", kind, details)

    # ---------------- Survival drive / adaptive thresholds ----------------

    def survival_mode(self) -> str:
        # starving = more urgent
        if self.pet.hunger > 90:
            return "STARVING"
        if self.pet.hunger > 75:
            return "HUNGRY"
        if self._is_fainted():
            return "DIZZY"
        return "NORMAL"

    def adaptive_conf_threshold(self) -> float:
        base = MIN_CONFIDENCE
        mode = self.survival_mode()
        # starving -> slightly lower bar (take best available meal)
        if mode == "STARVING":
            return max(0.50, base - 0.06)
        if mode == "HUNGRY":
            return max(0.52, base - 0.04)
        if mode == "DIZZY":
            # fainted -> still trade but only better edges
            return min(0.70, base + 0.05)
        return base

    def adaptive_risk_multiplier(self) -> float:
        # never 0: bot must keep hunting
        mode = self.survival_mode()
        if mode == "STARVING":
            return 1.10  # more aggressive *slightly*
        if mode == "HUNGRY":
            return 1.00
        if mode == "DIZZY":
            return 0.35  # reduced size, but still trading
        # normal: depends on health
        if self.pet.health < 25:
            return 0.55
        if self.pet.health < 45:
            return 0.80
        return 1.0

    # ---------------- Strategy scoring ----------------

    def score_market(self, market: str, closes: List[float]) -> Tuple[float, float, str, dict]:
        """
        Returns:
          edge_score (higher is better),
          confidence (0..1),
          reason_str,
          extra_metrics
        """
        if len(closes) < max(RSI_PERIOD + 1, SMA_SLOW + 1):
            return (0.0, 0.0, "not_enough_data", {"rsi": None, "trend": 0.0, "vol": 0.0})

        r = rsi(closes, RSI_PERIOD)
        f = sma(closes, SMA_FAST)
        s = sma(closes, SMA_SLOW)
        vol = volatility(closes, period=30)
        if r is None or f is None or s is None:
            return (0.0, 0.0, "indicator_unavailable", {"rsi": None, "trend": 0.0, "vol": vol})

        last = closes[-1]
        trend = 1.0 if f > s else -1.0
        trend_strength = abs((f - s) / max(last, 1e-9))  # rough

        # Base confidence from trend + RSI sweet spots
        conf = 0.52
        bits = []

        if trend > 0:
            conf += 0.10
            bits.append("trend_up")
        else:
            conf -= 0.08
            bits.append("trend_down")

        # RSI preference: buy dips in uptrend
        if r < 35:
            conf += 0.12
            bits.append("rsi_oversold")
        elif r < 50:
            conf += 0.05
            bits.append("rsi_mid")
        elif r > 70:
            conf -= 0.14
            bits.append("rsi_overbought")

        # volatility penalty (too wild = harder)
        if vol > 0.03:
            conf -= 0.05
            bits.append("high_vol")

        conf = clamp(conf, 0.0, 1.0)

        # Edge score: confidence + trend_strength bias, penalize extreme RSI / high vol
        edge = (conf * 100.0) + (trend_strength * 1000.0) - (vol * 500.0)
        reason = "|".join(bits) + f"|rsi={r:.1f}|trendS={trend_strength:.4f}|vol={vol:.4f}|px={last:.2f}"

        metrics = {"rsi": float(r), "trend_strength": float(trend_strength), "vol": float(vol), "last": float(last)}
        return (edge, conf, reason, metrics)

    # ---------------- Trading ----------------

    def can_open(self, market: str) -> bool:
        if len(self.positions) >= MAX_OPEN_POSITIONS:
            return False
        if any(p.market == market for p in self.positions):
            return False
        return True

    def open_long(self, market: str, price: float, confidence: float, reason: str):
        if not self.can_open(market):
            return

        risk_mult = self.adaptive_risk_multiplier()
        risk_pct = RISK_PER_TRADE_PCT * risk_mult
        risk_usd = self.equity_usd * (risk_pct / 100.0)

        sl = price * (1.0 - STOP_LOSS_PCT / 100.0)
        tp = price * (1.0 + TAKE_PROFIT_PCT / 100.0)
        sl_dist = max(price - sl, 1e-9)

        # size such that loss to stop ~= risk_usd
        size_usd = clamp(risk_usd * (price / sl_dist), 5.0, self.equity_usd * 0.25)
        pos = Position(
            market=market,
            side="LONG",
            entry_price=float(price),
            size_usd=float(size_usd),
            stop_price=float(sl),
            take_price=float(tp),
            opened_time_utc=utc_now_iso(),
            confidence=float(confidence),
            reason=reason
        )
        self.positions.append(pos)

        self._emit_event("thought", "hunt_open", {
            "market": market,
            "confidence": confidence,
            "survival_mode": self.survival_mode(),
            "size_usd": size_usd,
            "reason": reason[:220],
        })

        log.info(f"OPEN {market} size=${size_usd:.2f} entry={price:.2f} sl={sl:.2f} tp={tp:.2f} conf={confidence:.2f}")

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

        hold_minutes = 0.0
        try:
            opened = datetime.fromisoformat(pos.opened_time_utc.replace("Z", "+00:00"))
            hold_minutes = (datetime.now(timezone.utc) - opened).total_seconds() / 60.0
        except Exception:
            pass

        # fake qty for logging (paper): qty = size_usd / entry
        qty = pos.size_usd / max(pos.entry_price, 1e-9)

        # Send trade to API
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

        self._pet_on_trade(pnl)

        self._emit_event("thought", "hunt_close", {
            "market": pos.market,
            "pnl": pnl,
            "why": why,
            "equity": self.equity_usd,
            "survival_mode": self.survival_mode(),
        })

        log.info(f"CLOSE {pos.market} pnl={pnl:.2f} ({why}) equity={self.equity_usd:.2f}")
        self.positions.pop(idx)

    # ---------------- Cycle ----------------

    def run_cycle(self):
        with self.lock:
            self._pet_tick()

        prices = fetch_prices(MARKETS)
        if not prices:
            # still heartbeat so dashboard shows alive
            self.push_state(prices_ok=False)
            log.info("No prices -> idling safely.")
            return

        # 1) manage open positions (close on stop/tp)
        with self.lock:
            for i in range(len(self.positions) - 1, -1, -1):
                pos = self.positions[i]
                px = prices.get(pos.market)
                if px is None:
                    continue
                if px <= pos.stop_price:
                    self.close_position(i, px, "stop_loss", {"rsi": 0.0, "trend_strength": 0.0, "vol": 0.0})
                elif px >= pos.take_price:
                    self.close_position(i, px, "take_profit", {"rsi": 0.0, "trend_strength": 0.0, "vol": 0.0})

        # 2) score markets and pick best "meals"
        scored: List[Tuple[float, str, float, str, dict]] = []
        for m in MARKETS:
            closes = fetch_history(m, limit=180)
            if not closes:
                closes = [prices[m]] * (SMA_SLOW + RSI_PERIOD + 5)

            edge, conf, reason, metrics = self.score_market(m, closes)
            scored.append((edge, m, conf, reason, metrics))

        scored.sort(reverse=True, key=lambda x: x[0])

        # Adaptive threshold based on survival state
        conf_th = self.adaptive_conf_threshold()

        # How many new positions to open this cycle
        slots = MAX_OPEN_POSITIONS - len(self.positions)
        if slots > 0:
            opened = 0
            for edge, m, conf, reason, metrics in scored:
                if opened >= slots:
                    break
                if not self.can_open(m):
                    continue

                # Only long bias for now (simple)
                # Require uptrend + decent confidence
                if "trend_up" in reason and conf >= conf_th and edge > 0:
                    self.open_long(m, prices[m], conf, reason)
                    opened += 1

            if opened == 0:
                # When starving, emit "searching" thought
                if self.survival_mode() in ("HUNGRY", "STARVING"):
                    best = scored[0] if scored else None
                    self._emit_event("thought", "hunt_searching", {
                        "mode": self.survival_mode(),
                        "conf_threshold": conf_th,
                        "best_candidate": (best[1] if best else None),
                        "best_edge": (best[0] if best else None),
                    })

        # 3) push state to API
        self.push_state(prices_ok=True)

    # ---------------- Push state to API ----------------

    def push_state(self, prices_ok: bool):
        self.last_heartbeat_utc = utc_now_iso()

        if API_URL:
            # heartbeat
            http_post_json(f"{API_URL}/ingest/heartbeat", {
                "status": "running",
                "time_utc": self.last_heartbeat_utc,
                "markets": MARKETS,
                "open_positions": len(self.positions),
                "equity_usd": self.equity_usd,
                "wins": self.wins,
                "losses": self.losses,
                "total_trades": self.total_trades,
                "total_pnl_usd": self.total_pnl,
                "survival_mode": self.survival_mode(),
                "prices_ok": prices_ok,
            })

            # equity point
            http_post_json(f"{API_URL}/ingest/equity", {
                "time_utc": self.last_heartbeat_utc,
                "equity_usd": self.equity_usd,
            })

            # pet
            http_post_json(f"{API_URL}/ingest/pet", {
                **asdict(self.pet),
                "time_utc": self.last_heartbeat_utc,
                "survival_mode": self.survival_mode(),
            })

# ============================================================
# Main loop
# ============================================================

ENGINE = Engine()

def bot_loop():
    log.info(f"Bot starting. api_url={API_URL or '(none)'} markets={MARKETS} cycle={CYCLE_SECONDS}s")
    if not API_URL:
        log.warning("API_URL is empty. Set API_URL to your Render internal URL: http://crypto-ai-api-h921:10000")

    while True:
        try:
            ENGINE.run_cycle()
        except Exception as e:
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            log.error(f"Cycle crashed: {e}\n{err}")
            # log a training event if possible
            if API_URL:
                http_post_json(f"{API_URL}/ingest/training_event", {
                    "time_utc": utc_now_iso(),
                    "event": "cycle_crash",
                    "details": str(e)[:250],
                })
        time.sleep(CYCLE_SECONDS)

def main():
    bot_loop()

if __name__ == "__main__":
    main()

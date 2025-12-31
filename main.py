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
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# ============================================================
# Logging (Render-friendly: goes to stdout)
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("crypto-ai-bot")

# ============================================================
# Env helpers
# ============================================================

def env_str(name: str, default: str) -> str:
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

PORT = env_int("PORT", 10000)

# If you have an external price API, keep it here.
# If not, the bot will still run and just "idle" safely.
API_URL = env_str("API_URL", "").rstrip("/")
CRYPTO_AI_API_URL = env_str("CRYPTO_AI_API_URL", API_URL).rstrip("/")

MARKETS = [m.strip() for m in env_str("MARKETS", "BTC-USD,ETH-USD").split(",") if m.strip()]
CYCLE_SECONDS = env_int("CYCLE_SECONDS", 360)      # 6 mins
MAX_OPEN_POSITIONS = env_int("MAX_OPEN_POSITIONS", 2)
START_EQUITY = env_float("START_EQUITY", 1000.0)

# Strategy params
RSI_PERIOD = env_int("RSI_PERIOD", 14)
SMA_FAST = env_int("SMA_FAST", 10)
SMA_SLOW = env_int("SMA_SLOW", 30)

# Risk management
RISK_PER_TRADE_PCT = env_float("RISK_PER_TRADE_PCT", 0.5)  # 0.5% default
STOP_LOSS_PCT = env_float("STOP_LOSS_PCT", 0.8)            # 0.8% SL
TAKE_PROFIT_PCT = env_float("TAKE_PROFIT_PCT", 1.2)        # 1.2% TP
MIN_CONFIDENCE = env_float("MIN_CONFIDENCE", 0.55)

# Storage
DATA_DIR = env_str("DATA_DIR", "data")
LOG_DIR = env_str("LOG_DIR", "logs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

STATE_FILE = os.path.join(DATA_DIR, "state.json")
TRADES_FILE = os.path.join(DATA_DIR, "trades.json")
EQUITY_FILE = os.path.join(DATA_DIR, "equity.json")
PET_FILE = os.path.join(DATA_DIR, "pet.json")

# ============================================================
# Data models
# ============================================================

@dataclass
class Position:
    market: str
    side: str              # "LONG" only (paper)
    entry_price: float
    size_usd: float
    stop_price: float
    take_price: float
    opened_time_utc: str
    confidence: float
    reason: str
    pnl_usd: Optional[float] = None

@dataclass
class Trade:
    market: str
    side: str              # "BUY"/"SELL"
    pnl_usd: float
    time: str

@dataclass
class Pet:
    stage: str = "egg"         # egg -> hatched
    mood: str = "sleepy"       # happy/ok/sad/angry/sleepy
    health: float = 100.0      # 0..100
    hunger: float = 100.0      # 0..100 (higher is hungrier)
    growth: float = 0.0        # 0..100
    fainted: bool = False
    last_update_utc: str = ""

# ============================================================
# Persistence helpers
# ============================================================

def read_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def write_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# ============================================================
# Simple indicators
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

# ============================================================
# External API fetch
# ============================================================

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

# Expected: your API might already have /prices or /candles.
# If not, we still keep bot safe and idle.
def fetch_prices_from_api(markets: List[str]) -> Dict[str, float]:
    """
    Attempts:
      1) {API_URL}/prices  -> {"BTC-USD": 43000.1, "ETH-USD": 2300.2}
      2) {API_URL}/price?market=BTC-USD -> {"market":"BTC-USD","price":...}
    """
    prices: Dict[str, float] = {}

    if not API_URL:
        return prices

    # Try bulk
    bulk = http_get_json(f"{API_URL}/prices")
    if isinstance(bulk, dict):
        for m in markets:
            v = bulk.get(m)
            if isinstance(v, (int, float)) and v > 0:
                prices[m] = float(v)

    # Fallback per market
    for m in markets:
        if m in prices:
            continue
        one = http_get_json(f"{API_URL}/price?market={m}")
        if isinstance(one, dict):
            v = one.get("price")
            if isinstance(v, (int, float)) and v > 0:
                prices[m] = float(v)

    return prices

def fetch_history_from_api(market: str, limit: int = 120) -> List[float]:
    """
    Attempts:
      {API_URL}/history?market=BTC-USD&limit=120 -> {"closes":[...]}
      or {"data":[{"close":...}, ...]}
    """
    if not API_URL:
        return []

    data = http_get_json(f"{API_URL}/history?market={market}&limit={limit}")
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
# Bot engine
# ============================================================

class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = read_json(STATE_FILE, {})
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.equity_series: List[dict] = []
        self.pet: Pet = Pet()

        # Load persisted
        self._load()

        # Ensure baseline
        if "equity_usd" not in self.state:
            self.state["equity_usd"] = float(START_EQUITY)
        self.last_heartbeat_utc = ""

        # Stats
        self.total_trades = int(self.state.get("total_trades", 0))
        self.wins = int(self.state.get("wins", 0))
        self.losses = int(self.state.get("losses", 0))
        self.total_pnl = float(self.state.get("total_pnl_usd", 0.0))

    def _load(self):
        # positions
        raw_positions = read_json(STATE_FILE, {}).get("positions", [])
        if isinstance(raw_positions, list):
            for p in raw_positions:
                if isinstance(p, dict):
                    try:
                        self.positions.append(Position(**p))
                    except Exception:
                        pass

        # trades
        raw_trades = read_json(TRADES_FILE, [])
        if isinstance(raw_trades, list):
            for t in raw_trades[-200:]:
                if isinstance(t, dict):
                    try:
                        self.trades.append(Trade(**t))
                    except Exception:
                        pass

        # equity series
        raw_eq = read_json(EQUITY_FILE, [])
        if isinstance(raw_eq, list):
            self.equity_series = raw_eq[-500:]

        # pet
        raw_pet = read_json(PET_FILE, {})
        if isinstance(raw_pet, dict) and raw_pet:
            try:
                self.pet = Pet(**raw_pet)
            except Exception:
                self.pet = Pet()

    def _persist(self):
        # Persist state + positions
        s = dict(self.state)
        s["equity_usd"] = float(s.get("equity_usd", START_EQUITY))
        s["positions"] = [asdict(p) for p in self.positions]
        s["total_trades"] = self.total_trades
        s["wins"] = self.wins
        s["losses"] = self.losses
        s["total_pnl_usd"] = self.total_pnl
        write_json(STATE_FILE, s)

        write_json(TRADES_FILE, [asdict(t) for t in self.trades][-500:])
        write_json(EQUITY_FILE, self.equity_series[-500:])
        write_json(PET_FILE, asdict(self.pet))

    def _log_equity_point(self):
        eq = float(self.state.get("equity_usd", START_EQUITY))
        self.equity_series.append({"equity_usd": eq, "time_utc": utc_now_iso()})
        self.equity_series = self.equity_series[-500:]

    def _pet_tick(self):
        """
        Pet is tied to performance:
          - wins reduce hunger and increase growth
          - losses increase hunger and reduce health
          - if total_pnl goes positive -> pet can hatch and improve mood
        Also pet influences risk: when faint/sad -> reduce risk to protect it.
        """
        now = utc_now_iso()
        self.pet.last_update_utc = now

        # natural decay: gets a bit hungrier over time
        self.pet.hunger = clamp(self.pet.hunger + 1.5, 0.0, 100.0)

        # health depends on hunger
        if self.pet.hunger > 85:
            self.pet.health = clamp(self.pet.health - 2.0, 0.0, 100.0)
        elif self.pet.hunger < 40:
            self.pet.health = clamp(self.pet.health + 0.6, 0.0, 100.0)

        # Mood
        if self.pet.health <= 10:
            self.pet.mood = "sad"
        elif self.pet.health <= 30:
            self.pet.mood = "ok"
        elif self.pet.health <= 70:
            self.pet.mood = "happy"
        else:
            self.pet.mood = "happy"

        # Fainted logic
        if self.pet.health <= 0:
            self.pet.fainted = True

        # Hatch when total pnl is positive
        if (not self.pet.fainted) and self.pet.stage == "egg" and self.total_pnl > 0:
            self.pet.stage = "hatched"
            self.pet.growth = 10.0
            self.pet.mood = "happy"

    def _pet_on_trade(self, pnl: float):
        if pnl > 0:
            self.pet.hunger = clamp(self.pet.hunger - 8.0, 0.0, 100.0)
            self.pet.health = clamp(self.pet.health + 2.0, 0.0, 100.0)
            self.pet.growth = clamp(self.pet.growth + 2.5, 0.0, 100.0)
        else:
            self.pet.hunger = clamp(self.pet.hunger + 9.0, 0.0, 100.0)
            self.pet.health = clamp(self.pet.health - 4.0, 0.0, 100.0)

        if self.pet.health <= 0:
            self.pet.fainted = True

    def effective_risk_multiplier(self) -> float:
        """
        Pet makes the bot act "responsible":
          - If pet is fainted or low health -> risk down
          - If pet is healthy -> normal
        """
        if self.pet.fainted:
            return 0.0
        if self.pet.health < 20:
            return 0.25
        if self.pet.health < 40:
            return 0.5
        return 1.0

    def win_rate(self) -> float:
        total = self.wins + self.losses
        return (self.wins / total) * 100.0 if total > 0 else 0.0

    # ---------------------------
    # Strategy
    # ---------------------------

    def generate_signal(self, market: str, closes: List[float]) -> Tuple[str, float, str]:
        """
        Returns: (action, confidence, reason)
          action: "BUY" or "HOLD"
        """
        if len(closes) < max(RSI_PERIOD + 1, SMA_SLOW + 1):
            return ("HOLD", 0.0, "not_enough_data")

        r = rsi(closes, RSI_PERIOD)
        f = sma(closes, SMA_FAST)
        s = sma(closes, SMA_SLOW)
        if r is None or f is None or s is None:
            return ("HOLD", 0.0, "indicator_unavailable")

        last = closes[-1]
        trend_up = f > s

        # Confidence scoring (simple but stable)
        conf = 0.50
        reason_bits = []

        if trend_up:
            conf += 0.10
            reason_bits.append("trend_up")
        else:
            conf -= 0.08
            reason_bits.append("trend_down")

        if r < 35:
            conf += 0.10
            reason_bits.append("rsi_oversold")
        elif r > 70:
            conf -= 0.12
            reason_bits.append("rsi_overbought")

        # Keep within bounds
        conf = clamp(conf, 0.0, 1.0)

        if trend_up and r < 60 and conf >= MIN_CONFIDENCE:
            return ("BUY", conf, "|".join(reason_bits) + f"|rsi={r:.1f}|px={last:.2f}")

        return ("HOLD", conf, "|".join(reason_bits) + f"|rsi={r:.1f}|px={last:.2f}")

    # ---------------------------
    # Position management
    # ---------------------------

    def open_position(self, market: str, price: float, confidence: float, reason: str):
        if self.pet.fainted:
            return

        if len(self.positions) >= MAX_OPEN_POSITIONS:
            return

        # Risk scales down when pet is weak
        risk_mult = self.effective_risk_multiplier()
        if risk_mult <= 0:
            return

        equity = float(self.state.get("equity_usd", START_EQUITY))
        risk_pct = RISK_PER_TRADE_PCT * risk_mult
        risk_usd = equity * (risk_pct / 100.0)

        # Use SL distance to size position (very simple sizing)
        sl = price * (1.0 - STOP_LOSS_PCT / 100.0)
        tp = price * (1.0 + TAKE_PROFIT_PCT / 100.0)
        sl_dist = max(price - sl, 0.0000001)

        # "size_usd" is amount of equity allocated, capped
        # (This is paper-trading logic, not real exchange sizing)
        size_usd = clamp(risk_usd * (price / sl_dist), 5.0, equity * 0.25)

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
            pnl_usd=None,
        )
        self.positions.append(pos)
        log.info(f"OPEN {market} LONG size_usd={size_usd:.2f} entry={price:.2f} sl={sl:.2f} tp={tp:.2f} conf={confidence:.2f}")

    def close_position(self, idx: int, price: float, why: str):
        pos = self.positions[idx]
        # PnL in USD approx: (price - entry) / entry * size_usd
        pnl = ((price - pos.entry_price) / pos.entry_price) * pos.size_usd

        self.total_pnl += pnl
        self.total_trades += 1
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1

        # Update equity
        self.state["equity_usd"] = float(self.state.get("equity_usd", START_EQUITY)) + pnl

        # Record trade
        self.trades.append(Trade(market=pos.market, side="SELL", pnl_usd=float(pnl), time=utc_now_iso()))
        self.trades = self.trades[-200:]

        # Pet reacts
        self._pet_on_trade(pnl)

        log.info(f"CLOSE {pos.market} pnl={pnl:.2f} ({why}) equity={self.state['equity_usd']:.2f}")

        # Remove
        self.positions.pop(idx)

    # ---------------------------
    # Main cycle
    # ---------------------------

    def run_cycle(self):
        with self.lock:
            self._pet_tick()

        # Fetch prices (best effort)
        prices = fetch_prices_from_api(MARKETS)

        with self.lock:
            # Mark-to-market and check exits
            for i in range(len(self.positions) - 1, -1, -1):
                pos = self.positions[i]
                px = prices.get(pos.market)
                if px is None:
                    continue
                if px <= pos.stop_price:
                    self.close_position(i, px, "stop_loss")
                elif px >= pos.take_price:
                    self.close_position(i, px, "take_profit")

        # If no API prices available, just persist + idle safely
        if not prices:
            with self.lock:
                self._log_equity_point()
                self.last_heartbeat_utc = utc_now_iso()
                self._persist()
            log.info("No prices available from API. Bot idling safely.")
            return

        # For each market, generate signal (history best-effort)
        for m in MARKETS:
            if m not in prices:
                continue
            closes = fetch_history_from_api(m, limit=180)
            if not closes:
                # fallback: create tiny history from current price (prevents crashes)
                closes = [prices[m]] * (SMA_SLOW + RSI_PERIOD + 2)

            action, confidence, reason = self.generate_signal(m, closes)

            with self.lock:
                already_open = any(p.market == m for p in self.positions)
                if action == "BUY" and (not already_open) and confidence >= MIN_CONFIDENCE:
                    self.open_position(m, prices[m], confidence, reason)

        with self.lock:
            self._log_equity_point()
            self.last_heartbeat_utc = utc_now_iso()
            self._persist()

    def snapshot_for_api(self) -> dict:
        with self.lock:
            equity = float(self.state.get("equity_usd", START_EQUITY))
            total = self.wins + self.losses
            win_rate = (self.wins / total) * 100.0 if total > 0 else 0.0
            avg_pnl = (self.total_pnl / total) if total > 0 else 0.0

            # Per-market stats (simple)
            market_stats = []
            for m in MARKETS:
                mt = [t for t in self.trades if t.market == m]
                if mt:
                    mw = len([t for t in mt if t.pnl_usd >= 0])
                    ml = len(mt) - mw
                    mp = sum(t.pnl_usd for t in mt)
                    market_stats.append({
                        "market": m,
                        "trades": len(mt),
                        "win_rate": (mw / len(mt)) * 100.0,
                        "avg_pnl": mp / len(mt),
                        "total_pnl": mp
                    })

            # Recent trades
            recent = [asdict(t) for t in self.trades[-25:]][::-1]

            return {
                "equity_series": self.equity_series[-300:],
                "total_trades": self.total_trades,
                "wins": self.wins,
                "losses": self.losses,
                "win_rate": round(win_rate, 2),
                "avg_pnl": round(avg_pnl, 4),
                "

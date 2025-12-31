import os
import json
import time
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
# Logging (Render-friendly: stdout)
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("crypto-ai-bot")

# ============================================================
# Helpers
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
# Paths / Files
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_PATH = os.path.join(BASE_DIR, "dashboard.html")

# storage dirs (Render disk or ephemeral)
DATA_DIR = env_str("DATA_DIR", os.path.join(BASE_DIR, "data"))
LOG_DIR = env_str("LOG_DIR", os.path.join(BASE_DIR, "logs"))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

STATE_FILE = os.path.join(DATA_DIR, "state.json")
TRADES_FILE = os.path.join(DATA_DIR, "trades.json")
EQUITY_FILE = os.path.join(DATA_DIR, "equity.json")
PET_FILE = os.path.join(DATA_DIR, "pet.json")

# ============================================================
# Config
# ============================================================

PORT = env_int("PORT", 10000)

# Your price API base URL (no trailing /data etc)
API_URL = env_str("API_URL", "").rstrip("/")

MARKETS = [m.strip() for m in env_str("MARKETS", "BTC-USD,ETH-USD").split(",") if m.strip()]
CYCLE_SECONDS = env_int("CYCLE_SECONDS", 360)
MAX_OPEN_POSITIONS = env_int("MAX_OPEN_POSITIONS", 2)
START_EQUITY = env_float("START_EQUITY", 1000.0)

# indicators
RSI_PERIOD = env_int("RSI_PERIOD", 14)
SMA_FAST = env_int("SMA_FAST", 10)
SMA_SLOW = env_int("SMA_SLOW", 30)

# risk
RISK_PER_TRADE_PCT = env_float("RISK_PER_TRADE_PCT", 0.5)  # % of equity at risk per trade
STOP_LOSS_PCT = env_float("STOP_LOSS_PCT", 0.8)
TAKE_PROFIT_PCT = env_float("TAKE_PROFIT_PCT", 1.2)
MIN_CONFIDENCE = env_float("MIN_CONFIDENCE", 0.55)

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
    pnl_usd: Optional[float] = None

@dataclass
class Trade:
    market: str
    side: str
    pnl_usd: float
    time: str

@dataclass
class Pet:
    stage: str = "egg"        # egg -> hatched
    mood: str = "sleepy"
    health: float = 100.0     # 0..100
    hunger: float = 100.0     # 0..100 (higher = more hungry)
    growth: float = 0.0       # 0..100
    fainted: bool = False
    last_update_utc: str = ""

# ============================================================
# Persistence
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

# ============================================================
# HTTP helpers (API fetch)
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

def fetch_prices_from_api(markets: List[str]) -> Dict[str, float]:
    """
    Tries:
      GET {API_URL}/prices -> {"BTC-USD": 123, ...}
      GET {API_URL}/price?market=BTC-USD -> {"price": 123}
    """
    prices: Dict[str, float] = {}
    if not API_URL:
        return prices

    bulk = http_get_json(f"{API_URL}/prices")
    if isinstance(bulk, dict):
        for m in markets:
            v = bulk.get(m)
            if isinstance(v, (int, float)) and v > 0:
                prices[m] = float(v)

    for m in markets:
        if m in prices:
            continue
        one = http_get_json(f"{API_URL}/price?market={m}")
        if isinstance(one, dict):
            v = one.get("price")
            if isinstance(v, (int, float)) and v > 0:
                prices[m] = float(v)

    return prices

def fetch_history_from_api(market: str, limit: int = 180) -> List[float]:
    """
    Tries:
      GET {API_URL}/history?market=BTC-USD&limit=180 -> {"closes":[...]}
      OR -> {"data":[{"close":...}, ...]}
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
# Engine
# ============================================================

class Engine:
    def __init__(self):
        self.lock = threading.Lock()

        self.state = read_json(STATE_FILE, {})
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.equity_series: List[dict] = []

        raw_pet = read_json(PET_FILE, {})
        try:
            self.pet: Pet = Pet(**raw_pet) if isinstance(raw_pet, dict) and raw_pet else Pet()
        except Exception:
            self.pet = Pet()

        self._load_positions()
        self._load_trades()
        self._load_equity()

        if "equity_usd" not in self.state:
            self.state["equity_usd"] = float(START_EQUITY)

        self.total_trades = int(self.state.get("total_trades", 0))
        self.wins = int(self.state.get("wins", 0))
        self.losses = int(self.state.get("losses", 0))
        self.total_pnl = float(self.state.get("total_pnl_usd", 0.0))

        self.last_heartbeat_utc = ""

    def _load_positions(self):
        raw_positions = self.state.get("positions", [])
        if isinstance(raw_positions, list):
            for p in raw_positions:
                if isinstance(p, dict):
                    try:
                        self.positions.append(Position(**p))
                    except Exception:
                        pass

    def _load_trades(self):
        raw_trades = read_json(TRADES_FILE, [])
        if isinstance(raw_trades, list):
            for t in raw_trades[-200:]:
                if isinstance(t, dict):
                    try:
                        self.trades.append(Trade(**t))
                    except Exception:
                        pass

    def _load_equity(self):
        raw_eq = read_json(EQUITY_FILE, [])
        if isinstance(raw_eq, list):
            self.equity_series = raw_eq[-500:]

    def _persist(self):
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

    # ---------- Pet logic ----------

    def _pet_tick(self):
        self.pet.last_update_utc = utc_now_iso()
        self.pet.hunger = clamp(self.pet.hunger + 1.5, 0.0, 100.0)

        if self.pet.hunger > 85:
            self.pet.health = clamp(self.pet.health - 2.0, 0.0, 100.0)
        elif self.pet.hunger < 40:
            self.pet.health = clamp(self.pet.health + 0.6, 0.0, 100.0)

        if self.pet.health <= 10:
            self.pet.mood = "sad"
        elif self.pet.health <= 30:
            self.pet.mood = "ok"
        else:
            self.pet.mood = "happy"

        if self.pet.health <= 0:
            self.pet.fainted = True

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
        if self.pet.fainted:
            return 0.0
        if self.pet.health < 20:
            return 0.25
        if self.pet.health < 40:
            return 0.5
        return 1.0

    # ---------- Strategy ----------

    def generate_signal(self, market: str, closes: List[float]) -> Tuple[str, float, str]:
        if len(closes) < max(RSI_PERIOD + 1, SMA_SLOW + 1):
            return ("HOLD", 0.0, "not_enough_data")

        r = rsi(closes, RSI_PERIOD)
        f = sma(closes, SMA_FAST)
        s = sma(closes, SMA_SLOW)
        if r is None or f is None or s is None:
            return ("HOLD", 0.0, "indicator_unavailable")

        last = closes[-1]
        trend_up = f > s

        conf = 0.50
        bits = []
        if trend_up:
            conf += 0.10
            bits.append("trend_up")
        else:
            conf -= 0.08
            bits.append("trend_down")

        if r < 35:
            conf += 0.10
            bits.append("rsi_oversold")
        elif r > 70:
            conf -= 0.12
            bits.append("rsi_overbought")

        conf = clamp(conf, 0.0, 1.0)

        if trend_up and r < 60 and conf >= MIN_CONFIDENCE:
            return ("BUY", conf, "|".join(bits) + f"|rsi={r:.1f}|px={last:.2f}")
        return ("HOLD", conf, "|".join(bits) + f"|rsi={r:.1f}|px={last:.2f}")

    # ---------- Positions ----------

    def open_position(self, market: str, price: float, confidence: float, reason: str):
        if self.pet.fainted:
            return
        if len(self.positions) >= MAX_OPEN_POSITIONS:
            return
        if any(p.market == market for p in self.positions):
            return

        risk_mult = self.effective_risk_multiplier()
        if risk_mult <= 0:
            return

        equity = float(self.state.get("equity_usd", START_EQUITY))
        risk_pct = RISK_PER_TRADE_PCT * risk_mult
        risk_usd = equity * (risk_pct / 100.0)

        sl = price * (1.0 - STOP_LOSS_PCT / 100.0)
        tp = price * (1.0 + TAKE_PROFIT_PCT / 100.0)
        sl_dist = max(price - sl, 1e-9)

        size_usd = clamp(risk_usd * (price / sl_dist), 5.0, equity * 0.25)

        self.positions.append(Position(
            market=market,
            side="LONG",
            entry_price=float(price),
            size_usd=float(size_usd),
            stop_price=float(sl),
            take_price=float(tp),
            opened_time_utc=utc_now_iso(),
            confidence=float(confidence),
            reason=reason,
        ))

        log.info(f"OPEN {market} size=${size_usd:.2f} entry={price:.2f} sl={sl:.2f} tp={tp:.2f} conf={confidence:.2f}")

    def close_position(self, idx: int, price: float, why: str):
        pos = self.positions[idx]
        pnl = ((price - pos.entry_price) / pos.entry_price) * pos.size_usd

        self.total_pnl += pnl
        self.total_trades += 1
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1

        self.state["equity_usd"] = float(self.state.get("equity_usd", START_EQUITY)) + pnl
        self.trades.append(Trade(market=pos.market, side="SELL", pnl_usd=float(pnl), time=utc_now_iso()))
        self.trades = self.trades[-200:]

        self._pet_on_trade(pnl)

        log.info(f"CLOSE {pos.market} pnl={pnl:.2f} ({why}) equity={self.state['equity_usd']:.2f}")
        self.positions.pop(idx)

    # ---------- Cycle ----------

    def run_cycle(self):
        with self.lock:
            self._pet_tick()

        prices = fetch_prices_from_api(MARKETS)

        with self.lock:
            for i in range(len(self.positions) - 1, -1, -1):
                pos = self.positions[i]
                px = prices.get(pos.market)
                if px is None:
                    continue
                if px <= pos.stop_price:
                    self.close_position(i, px, "stop_loss")
                elif px >= pos.take_price:
                    self.close_position(i, px, "take_profit")

        if not prices:
            with self.lock:
                self._log_equity_point()
                self.last_heartbeat_utc = utc_now_iso()
                self._persist()
            log.info("No prices from API -> idling safely.")
            return

        for m in MARKETS:
            if m not in prices:
                continue

            closes = fetch_history_from_api(m, limit=180)
            if not closes:
                closes = [prices[m]] * (SMA_SLOW + RSI_PERIOD + 2)

            action, confidence, reason = self.generate_signal(m, closes)
            with self.lock:
                if action == "BUY" and confidence >= MIN_CONFIDENCE:
                    self.open_position(m, prices[m], confidence, reason)

        with self.lock:
            self._log_equity_point()
            self.last_heartbeat_utc = utc_now_iso()
            self._persist()

    def snapshot_for_api(self) -> dict:
        with self.lock:
            total = self.wins + self.losses
            win_rate = (self.wins / total) * 100.0 if total > 0 else 0.0
            avg_pnl = (self.total_pnl / total) if total > 0 else 0.0

            return {
                "equity_series": self.equity_series[-300:],
                "total_trades": self.total_trades,
                "wins": self.wins,
                "losses": self.losses,
                "win_rate": round(win_rate, 2),
                "avg_pnl": round(avg_pnl, 4),
                "total_pnl_usd": round(self.total_pnl, 6),
                "open_positions": [asdict(p) for p in self.positions],
                "recent_trades": [asdict(t) for t in self.trades[-25:]][::-1],
                "bot_status": {
                    "running": True,
                    "last_heartbeat_utc": self.last_heartbeat_utc,
                    "api_url": API_URL,
                },
                "pet": asdict(self.pet),
            }

# ============================================================
# Dashboard loader (no more triple-quoted HTML in Python)
# ============================================================

def load_dashboard_html() -> bytes:
    try:
        with open(DASHBOARD_PATH, "rb") as f:
            return f.read()
    except Exception as e:
        log.error(f"dashboard.html missing or unreadable: {e}")
        return b"<h1>dashboard.html not found</h1>"

# ============================================================
# Web server routes
# ============================================================

ENGINE = Engine()

class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str = "text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self._send(200, load_dashboard_html(), "text/html; charset=utf-8")
            return

        if self.path == "/health":
            self._send(200, b"OK", "text/plain; charset=utf-8")
            return

        if self.path == "/data":
            data = ENGINE.snapshot_for_api()
            self._send(200, json.dumps(data).encode("utf-8"), "application/json; charset=utf-8")
            return

        self._send(404, b"Not Found", "text/plain; charset=utf-8")

    def log_message(self, fmt, *args):
        return

# ============================================================
# Main
# ============================================================

def bot_loop():
    log.info(f"Bot starting. markets={MARKETS} cycle={CYCLE_SECONDS}s api_url={API_URL or '(none)'}")
    while True:
        try:
            ENGINE.run_cycle()
        except Exception as e:
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            log.error(f"Cycle crashed: {e}\n{err}")
        time.sleep(CYCLE_SECONDS)

def main():
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()

    server = HTTPServer(("0.0.0.0", PORT), Handler)
    log.info(f"Web server listening on :{PORT}")
    server.serve_forever()

if __name__ == "__main__":
    main()

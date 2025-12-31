import os
import json
import time
import math
import traceback
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


# ============================================================
# LOGGING (prints into Render logs)
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("crypto-ai-bot")


# ============================================================
# ENV HELPERS
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


# ============================================================
# CONFIG
# ============================================================

MARKETS = [m.strip() for m in env_str("MARKETS", "BTC-USD,ETH-USD").split(",") if m.strip()]
CYCLE_SECONDS = env_int("CYCLE_SECONDS", 360)        # 6 minutes default
GRANULARITY = env_int("GRANULARITY", 360)            # 6-min candles
CANDLE_LIMIT = env_int("CANDLE_LIMIT", 120)          # enough history
RSI_PERIOD = env_int("RSI_PERIOD", 14)

START_EQUITY = env_float("START_EQUITY", 1000.0)
RISK_PER_TRADE_PCT = env_float("RISK_PER_TRADE_PCT", 1.0)  # % equity risked
STOP_LOSS_PCT = env_float("STOP_LOSS_PCT", 1.0)            # % SL
TAKE_PROFIT_PCT = env_float("TAKE_PROFIT_PCT", 0.8)        # % TP
MAX_OPEN_POSITIONS = env_int("MAX_OPEN_POSITIONS", 3)
MIN_CONFIDENCE = env_float("MIN_CONFIDENCE", 0.60)

# Disk-mount friendly defaults
DATA_DIR = env_str("DATA_DIR", "data")
LOG_DIR = env_str("LOG_DIR", "logs")

# Optional API endpoint to POST events (for your dashboard API service)
API_URL = env_str("API_URL", "")  # e.g. https://crypto-ai-api-xxxx.onrender.com/events

# Coinbase Exchange public endpoint (no key needed)
COINBASE_BASE = "https://api.exchange.coinbase.com"


# ============================================================
# FILE PATHS
# ============================================================

STATE_PATH = os.path.join(DATA_DIR, "state.json")
EVENTS_PATH = os.path.join(LOG_DIR, "events.jsonl")
EQUITY_PATH = os.path.join(LOG_DIR, "equity.jsonl")
TRADES_PATH = os.path.join(LOG_DIR, "trades.jsonl")
HEARTBEAT_PATH = os.path.join(LOG_DIR, "heartbeat.json")


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Position:
    market: str
    side: str                  # "LONG" only for now
    entry_price: float
    size: float                # units of coin (BTC etc)
    stop_price: float
    take_price: float
    opened_utc: str
    confidence: float
    reason: List[str]


# ============================================================
# UTILS
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

def write_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def safe_read_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def safe_write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ============================================================
# EVENT + HEARTBEAT
# ============================================================

def log_event(event_type: str, payload: dict) -> None:
    evt = {
        "time_utc": utc_now_iso(),
        "type": event_type,
        "payload": payload
    }
    write_jsonl(EVENTS_PATH, evt)

    # Optional POST to your API service
    if API_URL:
        try:
            body = json.dumps(evt).encode("utf-8")
            req = Request(
                API_URL,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urlopen(req, timeout=10) as r:
                _ = r.read()
        except Exception as e:
            # Do not crash the bot if API is down
            log.warning(f"API_URL post failed: {e}")

def log_trade(trade: dict) -> None:
    write_jsonl(TRADES_PATH, trade)

def log_equity(equity_usd: float, extra: Optional[dict] = None) -> None:
    row = {
        "time_utc": utc_now_iso(),
        "equity_usd": float(equity_usd)
    }
    if extra:
        row.update(extra)
    write_jsonl(EQUITY_PATH, row)

def heartbeat(status: str, info: dict) -> None:
    hb = {
        "time_utc": utc_now_iso(),
        "status": status,
        "info": info
    }
    safe_write_json(HEARTBEAT_PATH, hb)


# ============================================================
# INDICATORS
# ============================================================

def sma(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return sum(values[-period:]) / float(period)

def rsi(values: List[float], period: int = 14) -> Optional[float]:
    if len(values) <= period:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        delta = values[i] - values[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


# ============================================================
# COINBASE DATA
# ============================================================

def fetch_candles(market: str, granularity: int, limit: int) -> List[dict]:
    # Coinbase returns candles newest->oldest: [time, low, high, open, close, volume]
    url = f"{COINBASE_BASE}/products/{market}/candles?granularity={granularity}"
    req = Request(url, headers={"User-Agent": "crypto-ai-bot"})
    with urlopen(req, timeout=15) as r:
        raw = r.read().decode("utf-8")
        data = json.loads(raw)

    # Sort oldest->newest
    data.sort(key=lambda c: c[0])
    candles = []
    for row in data[-limit:]:
        candles.append({
            "time": int(row[0]),
            "low": float(row[1]),
            "high": float(row[2]),
            "open": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        })
    return candles

def fetch_last_price(market: str) -> Optional[float]:
    url = f"{COINBASE_BASE}/products/{market}/ticker"
    req = Request(url, headers={"User-Agent": "crypto-ai-bot"})
    with urlopen(req, timeout=15) as r:
        raw = r.read().decode("utf-8")
        data = json.loads(raw)
    try:
        return float(data["price"])
    except Exception:
        return None


# ============================================================
# STATE
# ============================================================

def load_state() -> Tuple[dict, List[Position]]:
    state = safe_read_json(STATE_PATH, {})
    raw_positions = state.get("positions", [])
    positions: List[Position] = []
    for p in raw_positions:
        try:
            positions.append(Position(**p))
        except Exception:
            pass
    return state, positions

def save_state(state: dict, positions: List[Position]) -> None:
    state["positions"] = [asdict(p) for p in positions]
    safe_write_json(STATE_PATH, state)


# ============================================================
# STRATEGY
# ============================================================

def score_signal(closes: List[float]) -> Tuple[float, List[str]]:
    """
    Returns (confidence 0..1, reasons)
    Simple: oversold RSI + price above SMA trend filter
    """
    reasons = []
    r = rsi(closes, RSI_PERIOD)
    s = sma(closes, 20)

    if r is None or s is None:
        return 0.0, ["not_enough_data"]

    price = closes[-1]

    # Oversold region
    if r < 30:
        reasons.append("RSI_oversold")
    elif r < 40:
        reasons.append("RSI_low")

    # Trend filter
    if price > s:
        reasons.append("trend_up")
    else:
        reasons.append("trend_down")

    # Confidence
    conf = 0.0
    if "RSI_oversold" in reasons:
        conf += 0.45
    elif "RSI_low" in reasons:
        conf += 0.25

    if "trend_up" in reasons:
        conf += 0.25
    else:
        conf -= 0.10

    # Add small signal strength from RSI distance
    # (More oversold -> slightly higher)
    conf += clamp((40 - r) / 100.0, 0.0, 0.15)

    conf = clamp(conf, 0.0, 1.0)
    return conf, reasons

def calc_position_size(equity_usd: float, entry: float, stop_price: float) -> float:
    """
    Risk model: risk $ = equity * (RISK_PER_TRADE_PCT/100)
    Size chosen so (entry - stop) * size = risk$
    """
    risk_usd = equity_usd * (RISK_PER_TRADE_PCT / 100.0)
    per_unit_risk = max(1e-9, abs(entry - stop_price))
    size = risk_usd / per_unit_risk
    # Keep size reasonable
    if size <= 0:
        return 0.0
    return float(size)

def open_long(market: str, equity: float, price: float, confidence: float, reason: List[str]) -> Optional[Position]:
    stop_price = price * (1.0 - STOP_LOSS_PCT / 100.0)
    take_price = price * (1.0 + TAKE_PROFIT_PCT / 100.0)
    size = calc_position_size(equity, price, stop_price)
    if size <= 0:
        return None

    pos = Position(
        market=market,
        side="LONG",
        entry_price=float(price),
        size=float(size),
        stop_price=float(stop_price),
        take_price=float(take_price),
        opened_utc=utc_now_iso(),
        confidence=float(confidence),
        reason=reason,
    )
    return pos


# ============================================================
# POSITION MANAGEMENT
# ============================================================

def position_pnl_usd(pos: Position, price: float) -> float:
    # LONG pnl = (price - entry) * size
    return (price - pos.entry_price) * pos.size

def close_position(pos: Position, price: float, outcome: str) -> dict:
    pnl = position_pnl_usd(pos, price)
    trade = {
        "time_utc": utc_now_iso(),
        "market": pos.market,
        "side": pos.side,
        "entry_price": pos.entry_price,
        "exit_price": float(price),
        "size": pos.size,
        "pnl_usd": float(pnl),
        "outcome": outcome,
        "opened_utc": pos.opened_utc,
        "closed_utc": utc_now_iso(),
        "confidence": pos.confidence,
        "reason": pos.reason,
    }
    return trade

def manage_positions(state: dict, positions: List[Position]) -> float:
    """
    Checks stops/targets and updates equity.
    Returns current equity.
    """
    equity = float(state.get("equity_usd", START_EQUITY))
    still_open: List[Position] = []
    for pos in positions:
        price = fetch_last_price(pos.market)
        if price is None:
            still_open.append(pos)
            continue

        if price <= pos.stop_price:
            trade = close_position(pos, price, "stop_loss")
            equity += trade["pnl_usd"]
            log_trade(trade)
            log_event("close", trade)
        elif price >= pos.take_price:
            trade = close_position(pos, price, "take_profit")
            equity += trade["pnl_usd"]
            log_trade(trade)
            log_event("close", trade)
        else:
            still_open.append(pos)

    state["equity_usd"] = float(equity)
    return float(equity), still_open


# ============================================================
# MAIN LOOP
# ============================================================

def run_cycle(state: dict, positions: List[Position]) -> List[Position]:
    """
    One cycle:
      - manage open positions
      - scan markets for new signals
      - open positions if allowed
      - log equity point
    """
    # 1) manage existing positions
    equity, positions = manage_positions(state, positions)

    # 2) scan markets for signals
    for market in MARKETS:
        try:
            # If already too many positions, stop opening
            if len(positions) >= MAX_OPEN_POSITIONS:
                break

            # skip if already open for that market
            if any(p.market == market for p in positions):
                continue

            candles = fetch_candles(market, GRANULARITY, CANDLE_LIMIT)
            closes = [c["close"] for c in candles]
            if len(closes) < RSI_PERIOD + 5:
                continue

            conf, reasons = score_signal(closes)
            price = closes[-1]

            # Decision: buy only if confidence >= MIN_CONFIDENCE and trend_up present
            action = "HOLD"
            if conf >= MIN_CONFIDENCE and "trend_up" in reasons and ("RSI_oversold" in reasons or "RSI_low" in reasons):
                action = "BUY"

            # Log the signal event (this is your "brain data")
            signal_evt = {
                "market": market,
                "action": action,
                "confidence": float(conf),
                "reason": reasons,
                "price": float(price),
            }
            log_event("signal", signal_evt)

            # 3) open position if BUY
            if action == "BUY":
                pos = open_long(market, equity, price, conf, reasons)
                if pos:
                    positions.append(pos)

                    open_evt = {
                        "market": market,
                        "action": "BUY",
                        "confidence": float(conf),
                        "reason": reasons,
                        "price": float(price),
                        "size": float(pos.size),
                        "stop_price": float(pos.stop_price),
                        "take_price": float(pos.take_price),
                        "pnl_usd": None
                    }
                    log_event("open", open_evt)

        except Exception as e:
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            log_event("market_error", {"market": market, "message": str(e), "traceback": err})

    # 4) equity point each cycle
    log_equity(float(state.get("equity_usd", START_EQUITY)), {"open_positions": len(positions)})

    return positions


def main() -> None:
    ensure_dirs()

    # boot event
    log_event("boot", {
        "markets": MARKETS,
        "cycle_seconds": CYCLE_SECONDS,
        "granularity": GRANULARITY,
        "start_equity": START_EQUITY,
        "data_dir": DATA_DIR,
        "log_dir": LOG_DIR
    })

    state, positions = load_state()

    # ensure equity exists
    if "equity_usd" not in state:
        state["equity_usd"] = float(START_EQUITY)
        save_state(state, positions)
        log_equity(float(state["equity_usd"]), {"note": "startup"})

    while True:
        try:
            heartbeat("running", {
                "markets": MARKETS,
                "open_positions": len(positions),
                "equity_usd": float(state.get("equity_usd", START_EQUITY))
            })

            log.info(
                f"Cycle running | equity=${state.get('equity_usd')} | "
                f"open_positions={len(positions)}"
            )

            positions = run_cycle(state, positions)
            save_state(state, positions)

        except Exception as e:
            err = "".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            )
            heartbeat("error", {"message": str(e)})
            log_event("error", {
                "message": str(e),
                "traceback": err
            })
            log.error(f"Cycle crashed: {e}")

        time.sleep(CYCLE_SECONDS)


if __name__ == "__main__":
    main()

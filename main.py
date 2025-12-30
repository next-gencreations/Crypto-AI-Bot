import os
import json
import time
import math
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from urllib.request import Request, urlopen

# ============================================================
# CONFIG
# ============================================================

def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v.strip() if v and v.strip() else default

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except:
        return default

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except:
        return default

MARKETS = [m.strip() for m in env_str("MARKETS", "BTC-USD,ETH-USD").split(",") if m.strip()]
CYCLE_SECONDS = env_int("CYCLE_SECONDS", 360)  # 6 minutes default
GRANULARITY = env_int("GRANULARITY", 360)      # 6-minute candles on Coinbase Exchange API
CANDLE_LIMIT = env_int("CANDLE_LIMIT", 120)    # enough history for RSI/SMA
RSI_PERIOD = env_int("RSI_PERIOD", 14)

# Risk / paper-trade settings
START_EQUITY = env_float("START_EQUITY", 1000.0)
RISK_PER_TRADE_PCT = env_float("RISK_PER_TRADE_PCT", 1.0)      # % of equity risk per trade
STOP_LOSS_PCT = env_float("STOP_LOSS_PCT", 1.0)                # SL %
TAKE_PROFIT_PCT = env_float("TAKE_PROFIT_PCT", 0.8)            # TP %
MAX_OPEN_POSITIONS = env_int("MAX_OPEN_POSITIONS", 3)
MIN_CONFIDENCE = env_float("MIN_CONFIDENCE", 0.60)

DATA_DIR = env_str("DATA_DIR", "data")
LOG_DIR = env_str("LOG_DIR", "logs")

STATE_PATH = os.path.join(DATA_DIR, "state.json")
POSITIONS_PATH = os.path.join(DATA_DIR, "positions.json")

EVENTS_PATH = os.path.join(LOG_DIR, "events.jsonl")
TRADES_PATH = os.path.join(LOG_DIR, "trades.jsonl")
EQUITY_PATH = os.path.join(LOG_DIR, "equity.jsonl")
STATUS_PATH = os.path.join(LOG_DIR, "status.json")

# ============================================================
# UTIL
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def read_json(path: str, default: dict):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default

def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def log_event(event_type: str, payload: dict):
    obj = {"time_utc": utc_now_iso(), "type": event_type}
    obj.update(payload or {})
    append_jsonl(EVENTS_PATH, obj)

def log_trade(trade_type: str, payload: dict):
    obj = {"time_utc": utc_now_iso(), "type": trade_type}
    obj.update(payload or {})
    append_jsonl(TRADES_PATH, obj)

def log_equity(equity_usd: float, payload: dict = None):
    obj = {"time_utc": utc_now_iso(), "equity_usd": float(equity_usd)}
    if payload:
        obj.update(payload)
    append_jsonl(EQUITY_PATH, obj)

def heartbeat(status: str, payload: dict = None):
    obj = {"time_utc": utc_now_iso(), "status": status}
    if payload:
        obj.update(payload)
    write_json(STATUS_PATH, obj)

# ============================================================
# COINBASE PUBLIC DATA (NO KEYS REQUIRED)
# Uses Coinbase Exchange public candles endpoint.
# ============================================================

def http_get_json(url: str, timeout: int = 20):
    req = Request(url, headers={"User-Agent": "Crypto-AI-Bot/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)

def fetch_candles(product_id: str, granularity: int, limit: int) -> List[Tuple[int, float, float, float, float, float]]:
    # Coinbase Exchange endpoint returns: [ time, low, high, open, close, volume ]
    # Descending by time. We'll reverse it.
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles?granularity={granularity}"
    data = http_get_json(url)
    if not isinstance(data, list):
        return []
    data = data[:limit]
    data = list(reversed(data))
    # Ensure types
    out = []
    for row in data:
        try:
            t, low, high, opn, cls, vol = row
            out.append((int(t), float(low), float(high), float(opn), float(cls), float(vol)))
        except:
            continue
    return out

# ============================================================
# INDICATORS
# ============================================================

def sma(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period

def rsi(values: List[float], period: int = 14) -> Optional[float]:
    if len(values) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        change = values[i] - values[i - 1]
        if change > 0:
            gains += change
        else:
            losses -= change
    if gains == 0 and losses == 0:
        return 50.0
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))

def trend_up(values: List[float]) -> bool:
    # Simple: short SMA above long SMA
    s = sma(values, 10)
    l = sma(values, 30)
    if s is None or l is None:
        return False
    return s > l

# ============================================================
# PAPER TRADING STATE
# ============================================================

@dataclass
class Position:
    market: str
    side: str               # "LONG"
    entry_price: float
    size: float             # units of coin
    stop_price: float
    take_price: float
    opened_time_utc: str

def load_state():
    state = read_json(STATE_PATH, {"equity_usd": START_EQUITY, "wins": 0, "losses": 0})
    positions_raw = read_json(POSITIONS_PATH, {"positions": []})
    positions = []
    for p in positions_raw.get("positions", []):
        try:
            positions.append(Position(**p))
        except:
            continue
    return state, positions

def save_state(state: dict, positions: List[Position]):
    write_json(STATE_PATH, state)
    write_json(POSITIONS_PATH, {"positions": [asdict(p) for p in positions]})

def position_count_by_market(positions: List[Position], market: str) -> int:
    return sum(1 for p in positions if p.market == market)

# ============================================================
# STRATEGY: RSI oversold + trend up  -> BUY
# Exit: TP or SL
# ============================================================

def compute_signal(market: str, closes: List[float]) -> dict:
    r = rsi(closes, RSI_PERIOD)
    up = trend_up(closes)
    reason = []
    confidence = 0.0

    if r is None:
        return {"action": "HOLD", "confidence": 0.0, "reason": ["Not enough candle data"]}

    if r < 30:
        reason.append("RSI oversold")
        confidence += 0.35
    if up:
        reason.append("trend up")
        confidence += 0.30

    # Mild extra confidence if RSI very low
    if r < 25:
        confidence += 0.05
    if r > 70:
        # Overbought -> discourage buying
        confidence -= 0.20
        reason.append("RSI overbought")

    confidence = max(0.0, min(1.0, confidence))

    if confidence >= MIN_CONFIDENCE and ("RSI oversold" in reason) and ("trend up" in reason):
        return {"action": "BUY", "confidence": confidence, "reason": reason, "rsi": r}
    return {"action": "HOLD", "confidence": confidence, "reason": reason or ["No setup"], "rsi": r}

def calc_position_size(equity_usd: float, entry_price: float) -> float:
    # risk per trade = equity * RISK_PER_TRADE_PCT% = (entry - stop) * size
    risk_usd = equity_usd * (RISK_PER_TRADE_PCT / 100.0)
    stop_dist = entry_price * (STOP_LOSS_PCT / 100.0)
    if stop_dist <= 0:
        return 0.0
    size = risk_usd / stop_dist
    # prevent absurd size
    return max(0.0, size)

# ============================================================
# MAIN LOOP
# ============================================================

def run_cycle(state: dict, positions: List[Position]):
    equity = float(state.get("equity_usd", START_EQUITY))

    for market in MARKETS:
        # Get candles
        candles = fetch_candles(market, GRANULARITY, CANDLE_LIMIT)
        if len(candles) < RSI_PERIOD + 2:
            log_event("data_warning", {"market": market, "message": "Not enough candles"})
            continue

        closes = [c[4] for c in candles]
        price = closes[-1]

        # 1) Manage open positions first
        still_open = []
        for p in positions:
            if p.market != market:
                still_open.append(p)
                continue

            # LONG exit checks
            if price <= p.stop_price:
                # stop loss hit
                pnl = (price - p.entry_price) * p.size
                equity += pnl
                state["equity_usd"] = equity
                state["losses"] = int(state.get("losses", 0)) + 1

                log_trade("trade_close", {
                    "market": p.market,
                    "action": "SELL",
                    "reason": ["stop loss hit"],
                    "price": price,
                    "entry_price": p.entry_price,
                    "size": p.size,
                    "pnl_usd": pnl
                })
                log_event("signal", {
                    "market": market,
                    "action": "SELL",
                    "confidence": 1.0,
                    "reason": ["stop loss hit"],
                    "price": price,
                    "size": p.size,
                    "pnl_usd": pnl
                })
                continue

            if price >= p.take_price:
                # take profit hit
                pnl = (price - p.entry_price) * p.size
                equity += pnl
                state["equity_usd"] = equity
                state["wins"] = int(state.get("wins", 0)) + 1

                log_trade("trade_close", {
                    "market": p.market,
                    "action": "SELL",
                    "reason": ["take profit hit"],
                    "price": price,
                    "entry_price": p.entry_price,
                    "size": p.size,
                    "pnl_usd": pnl
                })
                log_event("signal", {
                    "market": market,
                    "action": "SELL",
                    "confidence": 1.0,
                    "reason": ["take profit hit"],
                    "price": price,
                    "size": p.size,
                    "pnl_usd": pnl
                })
                continue

            still_open.append(p)

        positions[:] = still_open  # update in-place

        # 2) Generate new signal
        sig = compute_signal(market, closes)
        action = sig.get("action", "HOLD")
        confidence = float(sig.get("confidence", 0.0))
        reason = sig.get("reason", [])
        rsi_val = sig.get("rsi", None)

        # Log signal event every cycle (this is the "brain data")
        log_event("signal", {
            "market": market,
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "price": price,
            "rsi": rsi_val,
            "open_positions": len(positions),
            "equity_usd": equity
        })

        # 3) Enter trade if allowed
        if action == "BUY":
            if len(positions) >= MAX_OPEN_POSITIONS:
                log_event("risk_block", {"market": market, "message": "Max open positions reached"})
                continue
            if position_count_by_market(positions, market) >= 1:
                log_event("risk_block", {"market": market, "message": "Already have a position in this market"})
                continue

            size = calc_position_size(equity, price)
            if size <= 0:
                log_event("risk_block", {"market": market, "message": "Position size calculated as 0"})
                continue

            stop_price = price * (1.0 - STOP_LOSS_PCT / 100.0)
            take_price = price * (1.0 + TAKE_PROFIT_PCT / 100.0)

            pos = Position(
                market=market,
                side="LONG",
                entry_price=price,
                size=size,
                stop_price=stop_price,
                take_price=take_price,
                opened_time_utc=utc_now_iso()
            )
            positions.append(pos)

            log_trade("trade_open", {
                "market": market,
                "action": "BUY",
                "confidence": confidence,
                "reason": reason,
                "price": price,
                "size": size,
                "stop_price": stop_price,
                "take_price": take_price,
                "pnl_usd": None
            })

    # Equity point every cycle
    equity = float(state.get("equity_usd", equity))
    log_equity(equity, {"open_positions": len(positions)})

def main():
    ensure_dirs()

    # boot logs
    log_event("boot", {
        "markets": MARKETS,
        "cycle_seconds": CYCLE_SECONDS,
        "granularity": GRANULARITY,
        "start_equity": START_EQUITY
    })

    state, positions = load_state()

    # if first run, ensure we have equity + initial equity log
    if "equity_usd" not in state:
        state["equity_usd"] = START_EQUITY

    log_equity(float(state["equity_usd"]), {"note": "startup"})

    while True:
        try:
            heartbeat("running", {
                "markets": MARKETS,
                "open_positions": len(positions),
                "equity_usd": float(state.get("equity_usd", START_EQUITY))
            })

            run_cycle(state, positions)
            save_state(state, positions)

        except Exception as e:
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            heartbeat("error", {"message": str(e)})
            log_event("error", {"message": str(e), "traceback": err})

        time.sleep(CYCLE_SECONDS)

if __name__ == "__main__":
    main()

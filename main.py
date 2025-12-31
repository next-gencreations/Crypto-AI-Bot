#!/usr/bin/env python3
"""
main.py — AI Trading Bot + Tamagotchi Life + Dashboard API + Event Logging

What this gives you:
- JSONL events log: runtime/events.jsonl
- State persistence: runtime/state.json
- Dashboard status: runtime/status.json
- Brain signal output: runtime/brain_signal.json
- Simple HTTP API for your dashboard:
    GET http://localhost:<PORT>/status
    GET http://localhost:<PORT>/events?n=200

Runs in one process:
- Background bot loop
- Lightweight HTTP server for dashboard
"""

from __future__ import annotations

import json
import os
import sys
import time
import math
import signal
import random
import threading
import traceback
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# -----------------------------
# Config (via Environment Vars)
# -----------------------------
BOT_NAME = os.getenv("BOT_NAME", "TradePilot")
RUNTIME_DIR = os.getenv("RUNTIME_DIR", "runtime")

# Market + loop
MARKET = os.getenv("MARKET", "BTC-USD")
LOOP_SECONDS = int(os.getenv("LOOP_SECONDS", "15"))
HEARTBEAT_SECONDS = int(os.getenv("HEARTBEAT_SECONDS", "30"))

# Risk controls (these stop the bot from blowing up)
START_EQUITY_USD = float(os.getenv("START_EQUITY_USD", "1000"))
MAX_POSITION_USD = float(os.getenv("MAX_POSITION_USD", "150"))           # max exposure
MAX_TRADE_USD = float(os.getenv("MAX_TRADE_USD", "50"))                  # per trade cap
MAX_DAILY_LOSS_USD = float(os.getenv("MAX_DAILY_LOSS_USD", "25"))        # daily circuit breaker
MAX_DRAWDOWN_USD = float(os.getenv("MAX_DRAWDOWN_USD", "75"))            # full stop
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "60"))              # after a trade or error

# Strategy thresholds
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "30"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "70"))
CONFIDENCE_MIN = float(os.getenv("CONFIDENCE_MIN", "0.58"))

# Dashboard API
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8080"))
BIND_HOST = os.getenv("BIND_HOST", "0.0.0.0")

# Exchange wiring (YOU MUST SET THESE if you want real trading)
EXCHANGE_MODE = os.getenv("EXCHANGE_MODE", "PAPER").upper()  # PAPER or LIVE
EXCHANGE_BASE_URL = os.getenv("EXCHANGE_BASE_URL", "")       # your new URL(s)
EXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY", "")
EXCHANGE_API_SECRET = os.getenv("EXCHANGE_API_SECRET", "")

# --------------
# File locations
# --------------
EVENTS_PATH = os.path.join(RUNTIME_DIR, "events.jsonl")
STATE_PATH = os.path.join(RUNTIME_DIR, "state.json")
STATUS_PATH = os.path.join(RUNTIME_DIR, "status.json")
BRAIN_PATH = os.path.join(RUNTIME_DIR, "brain_signal.json")

# -----------------------------
# Helpers
# -----------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_runtime_dir() -> None:
    os.makedirs(RUNTIME_DIR, exist_ok=True)

def safe_write_json(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
    os.replace(tmp, path)

def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a / b) * 100.0

# -----------------------------
# Data models
# -----------------------------
@dataclass
class Trade:
    time_utc: str
    market: str
    side: str  # BUY/SELL
    price: float
    size: float
    notional_usd: float
    pnl_usd: Optional[float] = None
    confidence: float = 0.0
    reason: List[str] = field(default_factory=list)

@dataclass
class BotState:
    start_time_utc: str = field(default_factory=utc_now)
    last_heartbeat_utc: str = field(default_factory=utc_now)
    status: str = "starting"
    equity_usd: float = START_EQUITY_USD
    peak_equity_usd: float = START_EQUITY_USD
    daily_start_equity_usd: float = START_EQUITY_USD
    daily_date_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).date().isoformat())
    open_position_usd: float = 0.0
    open_position_size: float = 0.0
    open_position_avg_price: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    errors: int = 0
    last_trade_utc: Optional[str] = None
    cooldown_until_utc: Optional[str] = None

@dataclass
class Tamagotchi:
    born_utc: str = field(default_factory=utc_now)
    age_hours: float = 0.0
    health: float = 100.0          # 0..100
    mood: float = 50.0             # 0..100
    hunger: float = 0.0            # 0..100 (higher = worse)
    energy: float = 100.0          # 0..100
    streak_wins: int = 0
    streak_losses: int = 0
    alive: bool = True
    last_update_utc: str = field(default_factory=utc_now)

# -----------------------------
# Logging / Eventing
# -----------------------------
class EventLog:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()

    def event(self, kind: str, payload: Dict[str, Any]) -> None:
        item = {
            "time_utc": utc_now(),
            "kind": kind,
            "payload": payload,
        }
        with self.lock:
            append_jsonl(self.path, item)

# -----------------------------
# Indicators (simple + robust)
# -----------------------------
def rsi(prices: List[float], period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        change = prices[i] - prices[i - 1]
        if change >= 0:
            gains += change
        else:
            losses += abs(change)
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))

def sma(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return sum(values[-period:]) / float(period)

def trend_up(prices: List[float]) -> Optional[bool]:
    # simple trend: short SMA above long SMA
    s = sma(prices, 10)
    l = sma(prices, 30)
    if s is None or l is None:
        return None
    return s > l

# -----------------------------
# Exchange adapter (plug your URLs in here)
# -----------------------------
class Exchange:
    """
    PAPER mode: simulates fills at current price.
    LIVE mode: you MUST implement the 3 marked methods for your exchange URLs.
    """

    def __init__(self, mode: str, base_url: str, api_key: str, api_secret: str, elog: EventLog):
        self.mode = mode
        self.base_url = base_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.elog = elog

    def get_latest_price(self, market: str) -> float:
        if self.mode == "PAPER":
            # Replace with real price feed later; this is a harmless placeholder.
            # Uses a slow random walk so the bot can run and generate logs.
            # If you already have a market data endpoint, wire it below.
            return self._paper_price_walk(market)

        # ----- LIVE: IMPLEMENT THIS -----
        # You said you have new URLs. Put your actual HTTP call here.
        # Return latest traded price as float.
        raise NotImplementedError("LIVE mode get_latest_price() not implemented yet")

    def place_order(self, market: str, side: str, notional_usd: float, price_hint: float) -> Dict[str, Any]:
        """
        Return a dict with:
        - filled_price
        - filled_size
        - order_id
        """
        if self.mode == "PAPER":
            filled_price = price_hint
            filled_size = notional_usd / max(filled_price, 1e-9)
            return {
                "order_id": f"paper-{int(time.time()*1000)}",
                "filled_price": filled_price,
                "filled_size": filled_size,
            }

        # ----- LIVE: IMPLEMENT THIS -----
        # Place real order via your exchange API
        raise NotImplementedError("LIVE mode place_order() not implemented yet")

    def _paper_price_walk(self, market: str) -> float:
        # stable seed per market
        seed = abs(hash(market)) % 10_000
        random.seed(seed + int(time.time() // 10))
        base = 42000.0 if "BTC" in market.upper() else 100.0
        jitter = random.uniform(-80, 80)
        return max(1.0, base + jitter)

# -----------------------------
# Brain (signal generator)
# -----------------------------
def generate_signal(price: float, prices: List[float]) -> Dict[str, Any]:
    r = rsi(prices, 14)
    tu = trend_up(prices)
    reasons = []
    conf = 0.5
    action = "HOLD"

    if r is not None:
        if r <= RSI_OVERSOLD:
            reasons.append("RSI oversold")
            conf += 0.10
        elif r >= RSI_OVERBOUGHT:
            reasons.append("RSI overbought")
            conf += 0.10

    if tu is True:
        reasons.append("trend up")
        conf += 0.07
    elif tu is False:
        reasons.append("trend down")
        conf += 0.07

    # Decision logic (simple + stable)
    if r is not None and tu is not None:
        if r <= RSI_OVERSOLD and tu is True and conf >= CONFIDENCE_MIN:
            action = "BUY"
        elif r >= RSI_OVERBOUGHT and tu is False and conf >= CONFIDENCE_MIN:
            action = "SELL"
        else:
            action = "HOLD"

    conf = clamp(conf, 0.0, 0.99)

    return {
        "time": utc_now(),
        "market": MARKET,
        "action": action,
        "confidence": conf,
        "reason": reasons if reasons else ["no strong edge"],
        "price": float(price),
    }

# -----------------------------
# Metrics
# -----------------------------
def compute_advanced_metrics(trades: List[Trade], equity: float, peak_equity: float) -> Dict[str, Any]:
    wins = [t for t in trades if (t.pnl_usd is not None and t.pnl_usd > 0)]
    losses = [t for t in trades if (t.pnl_usd is not None and t.pnl_usd < 0)]
    avg_win = sum(t.pnl_usd for t in wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(t.pnl_usd for t in losses) / len(losses)) if losses else 0.0
    best_trade = max((t.pnl_usd for t in trades if t.pnl_usd is not None), default=0.0)
    worst_trade = min((t.pnl_usd for t in trades if t.pnl_usd is not None), default=0.0)

    gross_profit = sum(t.pnl_usd for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.pnl_usd for t in losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    max_dd = max(0.0, peak_equity - equity)

    return {
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "best_trade_usd": best_trade,
        "worst_trade_usd": worst_trade,
        "profit_factor": profit_factor,
        "max_drawdown_usd": max_dd,
        "wins": len(wins),
        "losses": len(losses),
        "total_trades": len([t for t in trades if t.pnl_usd is not None]),
    }

# -----------------------------
# Tamagotchi life logic
# -----------------------------
def update_tamagotchi(t: Tamagotchi, state: BotState, metrics: Dict[str, Any]) -> Tamagotchi:
    now = datetime.now(timezone.utc)
    born = datetime.fromisoformat(t.born_utc)
    t.age_hours = max(0.0, (now - born).total_seconds() / 3600.0)

    # Performance impacts
    pf = metrics.get("profit_factor")
    max_dd = float(metrics.get("max_drawdown_usd", 0.0))

    # Hunger increases over time; wins feed it; losses make it hungrier
    t.hunger = clamp(t.hunger + 0.8, 0.0, 100.0)

    # Mood responds to recent streaks and PF
    if pf is None:
        pf_score = 0.0
    else:
        # pf 1.0 is neutral. >1.5 good. <0.7 bad.
        pf_score = clamp((pf - 1.0) * 25.0, -30.0, 30.0)

    dd_penalty = clamp(max_dd * 0.6, 0.0, 50.0)

    # Base drift
    t.mood = clamp(t.mood + pf_score - (dd_penalty * 0.1) - (t.hunger * 0.05), 0.0, 100.0)

    # Health responds to drawdown and hunger
    t.health = clamp(t.health - (dd_penalty * 0.2) - (t.hunger * 0.08) + (pf_score * 0.05), 0.0, 100.0)

    # Energy fades; rest if bot is in cooldown
    in_cooldown = False
    if state.cooldown_until_utc:
        try:
            cu = datetime.fromisoformat(state.cooldown_until_utc)
            in_cooldown = now < cu
        except Exception:
            in_cooldown = False

    if in_cooldown:
        t.energy = clamp(t.energy + 2.0, 0.0, 100.0)
    else:
        t.energy = clamp(t.energy - 0.7, 0.0, 100.0)

    # Alive logic
    if t.health <= 0.0:
        t.alive = False

    t.last_update_utc = utc_now()
    return t

# -----------------------------
# State load/save
# -----------------------------
def load_state() -> Tuple[BotState, Tamagotchi, List[float]]:
    ensure_runtime_dir()
    if not os.path.exists(STATE_PATH):
        state = BotState()
        tama = Tamagotchi()
        prices: List[float] = []
        return state, tama, prices

    with open(STATE_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Backwards-safe parsing
    state_raw = raw.get("state", {})
    tama_raw = raw.get("tamagotchi", {})
    prices = raw.get("prices", [])

    # trades
    trades = []
    for tr in state_raw.get("trades", []):
        trades.append(Trade(**tr))

    state = BotState(**{k: v for k, v in state_raw.items() if k != "trades"})
    state.trades = trades
    tama = Tamagotchi(**tama_raw)
    return state, tama, prices

def save_state(state: BotState, tama: Tamagotchi, prices: List[float]) -> None:
    ensure_runtime_dir()
    payload = {
        "state": {
            **{k: v for k, v in asdict(state).items() if k != "trades"},
            "trades": [asdict(t) for t in state.trades[-2000:]],  # cap size
        },
        "tamagotchi": asdict(tama),
        "prices": prices[-3000:],  # cap size
    }
    safe_write_json(STATE_PATH, payload)

# -----------------------------
# Dashboard API server
# -----------------------------
class DashboardHandler(BaseHTTPRequestHandler):
    def _send_json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/status":
            try:
                if os.path.exists(STATUS_PATH):
                    with open(STATUS_PATH, "r", encoding="utf-8") as f:
                        self._send_json(json.load(f))
                else:
                    self._send_json({"ok": False, "error": "status not ready"}, 404)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
            return

        if parsed.path == "/events":
            qs = parse_qs(parsed.query or "")
            n = int(qs.get("n", ["200"])[0])
            n = max(1, min(n, 2000))
            items = []
            try:
                if os.path.exists(EVENTS_PATH):
                    with open(EVENTS_PATH, "r", encoding="utf-8") as f:
                        lines = f.readlines()[-n:]
                    for line in lines:
                        line = line.strip()
                        if line:
                            items.append(json.loads(line))
                self._send_json({"ok": True, "items": items})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
            return

        # simple root
        if parsed.path == "/":
            self._send_json({
                "ok": True,
                "bot": BOT_NAME,
                "endpoints": ["/status", "/events?n=200"],
            })
            return

        self._send_json({"ok": False, "error": "not found"}, 404)

def start_dashboard_server():
    server = HTTPServer((BIND_HOST, DASHBOARD_PORT), DashboardHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server

# -----------------------------
# Risk / trade helpers
# -----------------------------
def reset_daily_if_needed(state: BotState, elog: EventLog) -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    if state.daily_date_utc != today:
        state.daily_date_utc = today
        state.daily_start_equity_usd = state.equity_usd
        elog.event("daily_reset", {
            "daily_date_utc": today,
            "daily_start_equity_usd": state.daily_start_equity_usd,
        })

def daily_pnl(state: BotState) -> float:
    return state.equity_usd - state.daily_start_equity_usd

def in_cooldown(state: BotState) -> bool:
    if not state.cooldown_until_utc:
        return False
    try:
        return datetime.now(timezone.utc) < datetime.fromisoformat(state.cooldown_until_utc)
    except Exception:
        return False

def set_cooldown(state: BotState, seconds: int) -> None:
    until = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    state.cooldown_until_utc = until.isoformat()

# -----------------------------
# Core bot loop
# -----------------------------
STOP_REQUESTED = False

def handle_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True

signal.signal(signal.SIGINT, handle_stop)
signal.signal(signal.SIGTERM, handle_stop)

def write_status(state: BotState, tama: Tamagotchi, metrics: Dict[str, Any], last_signal: Dict[str, Any]) -> None:
    payload = {
        "bot_status": {
            "status": state.status,
            "last_heartbeat": state.last_heartbeat_utc,
            "mode": EXCHANGE_MODE,
            "market": MARKET,
        },
        "account": {
            "equity_usd": state.equity_usd,
            "peak_equity_usd": state.peak_equity_usd,
            "daily_pnl_usd": daily_pnl(state),
            "open_position_usd": state.open_position_usd,
        },
        "advanced_metrics": metrics,
        "tamagotchi": asdict(tama),
        "brain": last_signal,
    }
    safe_write_json(STATUS_PATH, payload)

def write_brain_signal(signal_obj: Dict[str, Any]) -> None:
    safe_write_json(BRAIN_PATH, signal_obj)

def mark_trade_pnl_simple(state: BotState, trade: Trade) -> None:
    """
    Simple PnL marking:
    - If BUY opens/increases position
    - If SELL reduces/closes position, realize pnl vs average price
    This is intentionally straightforward and robust.
    """
    if trade.side == "BUY":
        # Update weighted avg entry
        new_size = state.open_position_size + trade.size
        if new_size <= 0:
            return
        new_avg = (
            (state.open_position_avg_price * state.open_position_size) + (trade.price * trade.size)
        ) / new_size
        state.open_position_size = new_size
        state.open_position_avg_price = new_avg
        state.open_position_usd = state.open_position_size * trade.price
        trade.pnl_usd = None  # unrealized until sell
        return

    if trade.side == "SELL":
        # Realize pnl on reduced size
        sell_size = min(trade.size, state.open_position_size)
        if sell_size <= 0:
            trade.pnl_usd = 0.0
            return

        entry = state.open_position_avg_price
        realized = (trade.price - entry) * sell_size
        trade.pnl_usd = realized

        state.open_position_size = max(0.0, state.open_position_size - sell_size)
        if state.open_position_size == 0.0:
            state.open_position_avg_price = 0.0
            state.open_position_usd = 0.0
        else:
            state.open_position_usd = state.open_position_size * trade.price

        # Update equity (realized)
        state.equity_usd += realized

def bot_loop():
    global STOP_REQUESTED

    ensure_runtime_dir()
    elog = EventLog(EVENTS_PATH)
    state, tama, prices = load_state()

    # Start dashboard server
    server = start_dashboard_server()
    elog.event("startup", {
        "bot": BOT_NAME,
        "market": MARKET,
        "mode": EXCHANGE_MODE,
        "dashboard_port": DASHBOARD_PORT,
        "runtime_dir": RUNTIME_DIR,
    })

    exchange = Exchange(
        mode=EXCHANGE_MODE,
        base_url=EXCHANGE_BASE_URL,
        api_key=EXCHANGE_API_KEY,
        api_secret=EXCHANGE_API_SECRET,
        elog=elog,
    )

    last_heartbeat_ts = 0.0
    last_signal: Dict[str, Any] = {"time": utc_now(), "market": MARKET, "action": "HOLD", "confidence": 0.0, "reason": ["boot"], "price": 0.0}

    while not STOP_REQUESTED:
        try:
            state.status = "running"
            reset_daily_if_needed(state, elog)

            # Heartbeat
            now_ts = time.time()
            if now_ts - last_heartbeat_ts >= HEARTBEAT_SECONDS:
                state.last_heartbeat_utc = utc_now()
                last_heartbeat_ts = now_ts
                elog.event("heartbeat", {"equity_usd": state.equity_usd, "open_position_usd": state.open_position_usd})

            # Circuit breakers
            dd = state.peak_equity_usd - state.equity_usd
            if dd > MAX_DRAWDOWN_USD:
                state.status = "stopped_drawdown"
                elog.event("circuit_breaker", {"reason": "max_drawdown", "drawdown_usd": dd})
                set_cooldown(state, 10_000_000)  # effectively stop
            if daily_pnl(state) <= -MAX_DAILY_LOSS_USD:
                state.status = "stopped_daily_loss"
                elog.event("circuit_breaker", {"reason": "max_daily_loss", "daily_pnl_usd": daily_pnl(state)})
                set_cooldown(state, 10_000_000)

            # Cooldown = rest / avoid overtrading
            if in_cooldown(state):
                metrics = compute_advanced_metrics(state.trades, state.equity_usd, state.peak_equity_usd)
                tama = update_tamagotchi(tama, state, metrics)
                write_status(state, tama, metrics, last_signal)
                save_state(state, tama, prices)
                time.sleep(min(LOOP_SECONDS, 2))
                continue

            # Get price
            price = exchange.get_latest_price(MARKET)
            prices.append(price)

            # Update peak equity (approx with current price if position open)
            approx_equity = state.equity_usd
            if state.open_position_size > 0:
                # approximate unrealized
                unreal = (price - state.open_position_avg_price) * state.open_position_size
                approx_equity = state.equity_usd + unreal
            state.peak_equity_usd = max(state.peak_equity_usd, approx_equity)

            # Brain signal
            last_signal = generate_signal(price, prices)
            write_brain_signal(last_signal)

            # Event log signal (for dashboard brain history)
            elog.event("signal", last_signal)

            # Decide trade
            action = last_signal["action"]
            conf = float(last_signal["confidence"])
            reasons = list(last_signal.get("reason", []))

            # Position sizing (simple, safe)
            # - never exceed MAX_POSITION_USD exposure
            # - never exceed MAX_TRADE_USD per trade
            remaining_exposure = max(0.0, MAX_POSITION_USD - state.open_position_usd)
            trade_notional = min(MAX_TRADE_USD, remaining_exposure)

            # Trade rules
            did_trade = False
            if action == "BUY" and conf >= CONFIDENCE_MIN and trade_notional >= 5.0:
                order = exchange.place_order(MARKET, "BUY", trade_notional, price)
                filled_price = float(order["filled_price"])
                filled_size = float(order["filled_size"])

                tr = Trade(
                    time_utc=utc_now(),
                    market=MARKET,
                    side="BUY",
                    price=filled_price,
                    size=filled_size,
                    notional_usd=trade_notional,
                    pnl_usd=None,
                    confidence=conf,
                    reason=reasons,
                )
                state.trades.append(tr)
                mark_trade_pnl_simple(state, tr)
                state.last_trade_utc = tr.time_utc
                did_trade = True
                elog.event("trade", asdict(tr))

            elif action == "SELL" and conf >= CONFIDENCE_MIN and state.open_position_size > 0:
                # Sell up to MAX_TRADE_USD notional (or full close if smaller)
                sell_notional = min(MAX_TRADE_USD, state.open_position_size * price)
                order = exchange.place_order(MARKET, "SELL", sell_notional, price)
                filled_price = float(order["filled_price"])
                filled_size = float(order["filled_size"])

                tr = Trade(
                    time_utc=utc_now(),
                    market=MARKET,
                    side="SELL",
                    price=filled_price,
                    size=filled_size,
                    notional_usd=sell_notional,
                    pnl_usd=0.0,
                    confidence=conf,
                    reason=reasons,
                )
                state.trades.append(tr)
                mark_trade_pnl_simple(state, tr)
                state.last_trade_utc = tr.time_utc
                did_trade = True
                elog.event("trade", asdict(tr))

            # Update tamagotchi + status
            metrics = compute_advanced_metrics(state.trades, state.equity_usd, state.peak_equity_usd)
            tama = update_tamagotchi(tama, state, metrics)

            # If tamagotchi dies, stop trading
            if not tama.alive:
                state.status = "stopped_tamagotchi_dead"
                elog.event("tamagotchi_dead", {"health": tama.health, "mood": tama.mood})
                set_cooldown(state, 10_000_000)

            write_status(state, tama, metrics, last_signal)
            save_state(state, tama, prices)

            # Cooldown after a trade
            if did_trade:
                set_cooldown(state, COOLDOWN_SECONDS)

            time.sleep(LOOP_SECONDS)

        except Exception as e:
            state.errors += 1
            state.status = "error"
            err = {
                "error": str(e),
                "trace": traceback.format_exc(),
            }
            try:
                elog.event("error", err)
            except Exception:
                pass

            # write status even on error
            metrics = compute_advanced_metrics(state.trades, state.equity_usd, state.peak_equity_usd)
            try:
                tama = update_tamagotchi(tama, state, metrics)
                write_status(state, tama, metrics, last_signal)
                save_state(state, tama, prices)
            except Exception:
                pass

            # prevent crash-loop
            set_cooldown(state, max(10, COOLDOWN_SECONDS))
            time.sleep(2)

    # Shutdown
    state.status = "stopped"
    state.last_heartbeat_utc = utc_now()
    metrics = compute_advanced_metrics(state.trades, state.equity_usd, state.peak_equity_usd)
    try:
        tama = update_tamagotchi(tama, state, metrics)
        write_status(state, tama, metrics, last_signal)
        save_state(state, tama, prices)
    except Exception:
        pass

    try:
        elog.event("shutdown", {"status": state.status})
    except Exception:
        pass

def main():
    print(f"[{BOT_NAME}] starting… mode={EXCHANGE_MODE} market={MARKET} dashboard=:{DASHBOARD_PORT}")
    bot_loop()
    print(f"[{BOT_NAME}] stopped.")

if __name__ == "__main__":
    main()

from __future__ import annotations

"""
Crypto-AI-Bot – experimental self-learning paper bot.

What this script does:

- Uses ONLY public market data (NO API KEYS, NO REAL MONEY).
- Scans a randomly chosen subset of markets on a schedule.
- Opens multiple LONG positions with per-trade TP/SL.
- Logs:
    - trades to        data/trades.csv
    - equity curve to  data/equity_curve.csv
    - training data to data/training_events.csv
    - runtime logs to  logs/runtime.log
- For every closed trade, it sends a training event to the
  external Crypto-AI-API (via api_client.send_training_event_to_api).

AI “brain” integration:

- Right now, decisions are simple rule-based.
- The ONLY place you need to modify in the future to plug in a real AI
  model or your Flask API is the function:

    decide_entry_action(market, price, features, risk_mode)

- Everything else (logging, position management, CSVs, etc.)
  is already wired up.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import csv
import time
import math
import random
import logging
from decimal import Decimal, getcontext
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

from api_client import send_training_event_to_api

# ---------------------------------------------------------------------------
# Global precision
# ---------------------------------------------------------------------------

getcontext().prec = 28  # high precision for prices & PnL

# ---------------------------------------------------------------------------
# Directories & file paths
# ---------------------------------------------------------------------------

DATA_DIR = "data"
LOGS_DIR = "logs"

TRADE_LOG_PATH = os.path.join(DATA_DIR, "trades.csv")
EQUITY_LOG_PATH = os.path.join(DATA_DIR, "equity_curve.csv")
TRAINING_LOG_PATH = os.path.join(DATA_DIR, "training_events.csv")
RUNTIME_LOG_PATH = os.path.join(LOGS_DIR, "runtime.log")

TRADE_FIELDS = [
    "timestamp",
    "market",
    "direction",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "size_usd",
    "pnl_usd",
    "pnl_pct",
    "reason",
]

EQUITY_FIELDS = [
    "timestamp",
    "equity_usd",
]

TRAINING_FIELDS = [
    "entry_time",
    "exit_time",
    "hold_minutes",
    "market",
    "trend_strength",
    "rsi",
    "volatility",
    "entry_price",
    "exit_price",
    "pnl_usd",
    "pnl_pct",
    "take_profit_pct",
    "stop_loss_pct",
    "risk_mode",
]

# ---------------------------------------------------------------------------
# Bot configuration
# ---------------------------------------------------------------------------

# Equity / sizing
START_BALANCE_USD = Decimal("1000")
TRADE_SIZE_USD = Decimal("50")          # notional per trade
MAX_OPEN_POSITIONS = 5

# Risk mode label (stored in training data so the AI can know “style”)
RISK_MODE = "AGGRESSIVE"  # or "NORMAL", etc.

# Markets to sample from (Binance spot style symbols)
ALL_MARKETS: List[str] = [
    "BTC-USDT",
    "ETH-USDT",
    "BNB-USDT",
    "SOL-USDT",
    "XRP-USDT",
    "ADA-USDT",
    "AVAX-USDT",
    "LINK-USDT",
    "ATOM-USDT",
    "SAND-USDT",
]

MAX_MARKETS_PER_SCAN = 5        # max markets per cycle
SLEEP_SECONDS = 60 * 6          # 6 minutes between cycles

# Take-profit / Stop-loss per trade
TAKE_PROFIT_PCT = Decimal("1.5")  # +1.5%
STOP_LOSS_PCT = Decimal("1.0")    # -1.0%

# Public market data
BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price"

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

logger: logging.Logger
OPEN_POSITIONS: List[Dict] = []
LAST_PRICE: Dict[str, Decimal] = {}  # for simple feature calculations


# ---------------------------------------------------------------------------
# Filesystem & logging utilities
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


def setup_logging() -> logging.Logger:
    log = logging.getLogger("crypto_ai_bot")
    log.setLevel(logging.INFO)

    if not log.handlers:
        # File handler
        fh = logging.FileHandler(RUNTIME_LOG_PATH)
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        log.addHandler(fh)

        # Console handler
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        log.addHandler(sh)

    return log


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def append_csv(path: str, fieldnames: List[str], row: Dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_last_equity() -> Decimal:
    if not os.path.exists(EQUITY_LOG_PATH):
        return START_BALANCE_USD

    try:
        with open(EQUITY_LOG_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            last = None
            for last in reader:
                pass
            if last and "equity_usd" in last:
                return Decimal(last["equity_usd"])
    except Exception as exc:
        logger.error(f"Failed to read equity log, using start balance: {exc}")

    return START_BALANCE_USD


def log_equity(equity: Decimal) -> None:
    append_csv(
        EQUITY_LOG_PATH,
        EQUITY_FIELDS,
        {
            "timestamp": utc_now().isoformat(),
            "equity_usd": str(equity),
        },
    )


# ---------------------------------------------------------------------------
# Market data & features
# ---------------------------------------------------------------------------

def binance_symbol(market: str) -> str:
    """Convert 'BTC-USDT' -> 'BTCUSDT'."""
    return market.replace("-", "")


def fetch_price(market: str) -> Optional[Decimal]:
    symbol = binance_symbol(market)
    try:
        resp = requests.get(
            BINANCE_PRICE_URL,
            params={"symbol": symbol},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return Decimal(str(data["price"]))
    except Exception as exc:
        logger.error(f"Error fetching price for {market}: {exc}")
        return None


def compute_features(market: str, price: Decimal) -> Dict[str, float]:
    """
    Very lightweight features derived from last price memory:

    - trend_strength: clipped % change between current & last price
    - rsi: a fake oscillator based on recent change
    - volatility: absolute % change (clipped)
    """
    prev = LAST_PRICE.get(market, price)
    LAST_PRICE[market] = price

    if prev <= 0:
        change_pct = Decimal("0")
    else:
        change_pct = (price - prev) / prev * Decimal("100")

    # Clip trend strength to [-5, 5]
    trend_strength = float(max(Decimal("-5"), min(Decimal("5"), change_pct)))

    # Pseudo-RSI: start from 50 and push by 2x change
    rsi_base = 50 + float(change_pct) * 2
    rsi = max(0.0, min(100.0, rsi_base))

    volatility = float(min(10, abs(float(change_pct))))

    return {
        "trend_strength": trend_strength,
        "rsi": rsi,
        "volatility": volatility,
    }


# ---------------------------------------------------------------------------
# Strategy / AI decision layer
# ---------------------------------------------------------------------------

def decide_entry_action(
    market: str,
    price: Decimal,
    features: Dict[str, float],
    risk_mode: str,
) -> bool:
    """
    ENTRY decision function.

    RETURN:
        True  -> open LONG
        False -> do nothing

    CURRENT LOGIC (simple baseline):
    - Go long when trend_strength > 0 and rsi < 70
    - Later, you can replace this with:
        - a call to your Crypto-AI-API (e.g. /signal)
        - a local ML model
        - any custom AI logic

    This way, all other code (logging, positions, files) stays unchanged.
    """
    trend = features["trend_strength"]
    rsi = features["rsi"]

    # Example of very simple rule that depends on "risk_mode"
    if risk_mode == "AGGRESSIVE":
        min_trend = 0.05
        max_rsi = 75
    else:  # NORMAL / conservative
        min_trend = 0.2
        max_rsi = 70

    if trend > min_trend and rsi < max_rsi:
        return True

    return False


def can_open_more_positions() -> bool:
    return len(OPEN_POSITIONS) < MAX_OPEN_POSITIONS


# ---------------------------------------------------------------------------
# Positions: open / evaluate / close
# ---------------------------------------------------------------------------

def open_position(market: str, price: Decimal, features: Dict[str, float]) -> None:
    if not can_open_more_positions():
        return

    position = {
        "market": market,
        "direction": "LONG",
        "entry_time": utc_now(),
        "entry_price": price,
        "size_usd": TRADE_SIZE_USD,
        "take_profit_pct": TAKE_PROFIT_PCT,
        "stop_loss_pct": STOP_LOSS_PCT,
        "features": features,  # snapshot at entry
    }

    OPEN_POSITIONS.append(position)
    logger.info(f"Opened LONG {market} at {price} size {TRADE_SIZE_USD} USD")


def evaluate_exit(
    position: Dict,
    current_price: Decimal,
) -> Tuple[bool, Optional[str], Decimal, Decimal]:
    """
    EXIT decision function.

    Currently only uses TP/SL. Later you can make this more “AI-ish”
    if you like, similar to decide_entry_action.
    """
    entry_price: Decimal = position["entry_price"]
    size_usd: Decimal = position["size_usd"]

    if entry_price <= 0:
        return False, None, Decimal("0"), Decimal("0")

    change_pct = (current_price - entry_price) / entry_price * Decimal("100")

    # Take-profit?
    if change_pct >= position["take_profit_pct"]:
        pnl_pct = change_pct
        pnl_usd = size_usd * pnl_pct / Decimal("100")
        return True, "TAKE_PROFIT", pnl_usd, pnl_pct

    # Stop-loss?
    if change_pct <= -position["stop_loss_pct"]:
        pnl_pct = change_pct
        pnl_usd = size_usd * pnl_pct / Decimal("100")
        return True, "STOP_LOSS", pnl_usd, pnl_pct

    return False, None, Decimal("0"), Decimal("0")


def close_position(
    position: Dict,
    current_price: Decimal,
    reason: str,
    pnl_usd: Decimal,
    pnl_pct: Decimal,
    equity_before: Decimal,
) -> Decimal:
    """
    Close a position, log everything, send training event, return new equity.
    """
    global OPEN_POSITIONS

    market = position["market"]
    entry_time: datetime = position["entry_time"]
    entry_price: Decimal = position["entry_price"]
    size_usd: Decimal = position["size_usd"]
    features = position["features"]

    exit_time = utc_now()
    hold_minutes = (exit_time - entry_time).total_seconds() / 60.0

    # Remove from open list
    OPEN_POSITIONS = [p for p in OPEN_POSITIONS if p is not position]

    # 1) Log trade
    trade_row = {
        "timestamp": exit_time.isoformat(),
        "market": market,
        "direction": "LONG",
        "entry_time": entry_time.isoformat(),
        "exit_time": exit_time.isoformat(),
        "entry_price": str(entry_price),
        "exit_price": str(current_price),
        "size_usd": str(size_usd),
        "pnl_usd": str(pnl_usd),
        "pnl_pct": str(pnl_pct),
        "reason": reason,
    }
    append_csv(TRADE_LOG_PATH, TRADE_FIELDS, trade_row)

    # 2) Log training event
    training_row = {
        "entry_time": entry_time.isoformat(),
        "exit_time": exit_time.isoformat(),
        "hold_minutes": round(hold_minutes, 2),
        "market": market,
        "trend_strength": features["trend_strength"],
        "rsi": features["rsi"],
        "volatility": features["volatility"],
        "entry_price": str(entry_price),
        "exit_price": str(current_price),
        "pnl_usd": str(pnl_usd),
        "pnl_pct": str(pnl_pct),
        "take_profit_pct": str(position["take_profit_pct"]),
        "stop_loss_pct": str(position["stop_loss_pct"]),
        "risk_mode": RISK_MODE,
    }
    append_csv(TRAINING_LOG_PATH, TRAINING_FIELDS, training_row)

    # Send to external Crypto-AI-API for learning
    try:
        send_training_event_to_api(training_row)
    except Exception as exc:
        logger.error(f"Failed to send training event to API: {exc}")

    # 3) Update equity
    new_equity = equity_before + pnl_usd
    logger.info(
        f"Closed {market} {reason} at {current_price} PnL {pnl_usd:.2f} USD "
        f"({pnl_pct:.2f}%), equity: {equity_before:.2f} -> {new_equity:.2f}"
    )
    log_equity(new_equity)
    return new_equity


# ---------------------------------------------------------------------------
# One full cycle: scan markets + manage positions
# ---------------------------------------------------------------------------

def choose_markets_for_cycle() -> List[str]:
    """Pick a random subset of markets to scan this round."""
    return random.sample(
        ALL_MARKETS,
        k=min(MAX_MARKETS_PER_SCAN, len(ALL_MARKETS)),
    )


def scan_markets_and_trade(equity: Decimal) -> Decimal:
    """
    One cycle:
      - Pick markets
      - Get price + features
      - Decide entries
      - Update / close open positions
    """
    # 1) Entry scanning
    for market in choose_markets_for_cycle():
        price = fetch_price(market)
        if price is None:
            continue

        features = compute_features(market, price)

        if can_open_more_positions():
            want_entry = decide_entry_action(
                market=market,
                price=price,
                features=features,
                risk_mode=RISK_MODE,
            )
            if want_entry:
                open_position(market, price, features)

    # 2) Exit check for all open positions
    for position in list(OPEN_POSITIONS):
        market = position["market"]
        price = fetch_price(market)
        if price is None:
            continue

        should_close, reason, pnl_usd, pnl_pct = evaluate_exit(position, price)
        if should_close:
            equity = close_position(position, price, reason, pnl_usd, pnl_pct, equity)

    return equity


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_directories()
    global logger
    logger = setup_logging()

    logger.info("Crypto-AI-Bot starting up")

    equity = load_last_equity()
    logger.info(f"Loaded starting equity: {equity:.2f} USD")
    log_equity(equity)

    while True:
        try:
            equity = scan_markets_and_trade(equity)
        except Exception as exc:
            logger.exception(f"Unhandled error in main loop: {exc}")

        logger.info(f"Sleeping for {SLEEP_SECONDS} seconds...")
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()

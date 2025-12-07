"""
Crypto-AI-Bot - experimental self-learning paper-trading bot for Coinbase.

- Public market data only (NO API KEYS, NO REAL ORDERS)
- Scans random markets every 6 minutes
- Uses simple indicators + per-market win/loss stats ("online learning")
- Opens multiple long positions
- Each position has take-profit + stop-loss
- Logs trades to data/trades.csv
- Logs equity to data/equity_curve.csv
- Logs runtime messages to logs/runtime.log
- Logs feature snapshots to data/training_events.csv for future AI models
"""

from __future__ import annotations

import os
import csv
import time
import random
from decimal import Decimal, getcontext
from datetime import datetime, timezone, date
from typing import Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Precision / global context
# ---------------------------------------------------------------------------

getcontext().prec = 28  # high precision for crypto math

# ---------------------------------------------------------------------------
# Directories & file paths
# ---------------------------------------------------------------------------

DATA_DIR = "data"
LOGS_DIR = "logs"

TRADE_LOG_PATH = os.path.join(DATA_DIR, "trades.csv")
EQUITY_LOG_PATH = os.path.join(DATA_DIR, "equity_curve.csv")
RUNTIME_LOG_PATH = os.path.join(LOGS_DIR, "runtime.log")
TRAINING_LOG_PATH = os.path.join(DATA_DIR, "training_events.csv")


def ensure_directories() -> None:
    """Make sure data/ and logs/ exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Configuration (env + defaults)
# ---------------------------------------------------------------------------

def get_start_balance() -> Decimal:
    """Read starting USD balance from env, default 100."""
    raw = os.getenv("START_BALANCE_USD", "100").strip()
    try:
        return Decimal(raw)
    except Exception:
        return Decimal("100")


START_BALANCE_USD: Decimal = get_start_balance()

# Risk mode: SAFE / NORMAL / AGGRESSIVE (env var RISK_MODE)
RISK_MODE = os.getenv("RISK_MODE", "SAFE").upper().strip()

# Markets universe (you can change these if you like)
ALL_MARKETS: List[str] = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "AVAX-USD",
    "ADA-USD",
    "LTC-USD",
    "DOGE-USD",
    "LINK-USD",
    "MATIC-USD",
    "OP-USD",
    "ARB-USD",
    "UNI-USD",
    "RNDR-USD",
    "SAND-USD",
    "ATOM-USD",
]

# How many random markets to scan per cycle
MAX_MARKETS_PER_SCAN = 8

# Loop timing & candles
SLEEP_SECONDS = 6 * 60           # 6 minutes
CANDLE_GRANULARITY = 300         # 5-minute candles
LOOKBACK_CANDLES = 100

PUBLIC_API_BASE = "https://api.exchange.coinbase.com"

# Risk parameters by mode
if RISK_MODE == "AGGRESSIVE":
    TAKE_PROFIT_PCT = Decimal("0.015")       # +1.5%
    STOP_LOSS_PCT = Decimal("0.02")          # -2.0%
    POSITION_SIZE_FRACTION = Decimal("0.4")  # 40% of USD per new trade
    MAX_DAILY_DRAWDOWN = Decimal("0.08")     # 8%
    MAX_OPEN_POSITIONS = 5
elif RISK_MODE == "NORMAL":
    TAKE_PROFIT_PCT = Decimal("0.012")       # +1.2%
    STOP_LOSS_PCT = Decimal("0.018")         # -1.8%
    POSITION_SIZE_FRACTION = Decimal("0.35") # 35%
    MAX_DAILY_DRAWDOWN = Decimal("0.06")     # 6%
    MAX_OPEN_POSITIONS = 4
else:  # SAFE (default)
    TAKE_PROFIT_PCT = Decimal("0.010")       # +1.0%
    STOP_LOSS_PCT = Decimal("0.015")         # -1.5%
    POSITION_SIZE_FRACTION = Decimal("0.30") # 30%
    MAX_DAILY_DRAWDOWN = Decimal("0.05")     # 5%
    MAX_OPEN_POSITIONS = 3

# Indicator thresholds – *loosened* for more trades and more data
MIN_TREND_STRENGTH = Decimal("0.001")  # was 0.002
RSI_BUY_MIN = Decimal("35")            # was 40
RSI_BUY_MAX = Decimal("70")            # was 65
MIN_VOLATILITY = Decimal("0.001")      # was 0.002
MAX_VOLATILITY = Decimal("0.050")      # was 0.030

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

usd_balance: Decimal = START_BALANCE_USD

# Each position is a dict:
# {
#   "market": str,
#   "size": Decimal (coin size),
#   "entry_price": Decimal,
#   "entry_time": datetime,
#   "take_profit": Decimal,
#   "stop_loss": Decimal,
#   "features": Dict[str, Decimal],  # indicators at entry
# }
open_positions: List[Dict] = []

# Simple per-market performance stats for "online learning"
# market_stats[market] = {"wins": int, "losses": int}
market_stats: Dict[str, Dict[str, int]] = {m: {"wins": 0, "losses": 0} for m in ALL_MARKETS}

# Equity & risk tracking
equity_peak_today: Decimal = START_BALANCE_USD
today: date = datetime.now(timezone.utc).date()
trading_paused_for_today: bool = False

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    """Log to console and append to logs/runtime.log."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)

    try:
        with open(RUNTIME_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # Never crash the bot because logging failed
        pass


def append_csv_row(path: str, header: List[str], row: List) -> None:
    """Append a row to CSV, writing header first if file is new."""
    file_exists = os.path.exists(path)
    try:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)
    except Exception as e:
        log(f"WARNING: failed to write to {path}: {e}")


def log_trade_csv(
    timestamp: datetime,
    market: str,
    side: str,
    size: Decimal,
    entry_price: Decimal,
    exit_price: Decimal,
    profit_usd: Decimal,
    equity_after: Decimal,
) -> None:
    """Log closed trades to data/trades.csv."""
    header = [
        "timestamp",
        "market",
        "side",
        "size",
        "entry_price",
        "exit_price",
        "profit_usd",
        "equity_after",
    ]
    row = [
        timestamp.isoformat(),
        market,
        side,
        f"{size:.8f}",
        f"{entry_price:.2f}",
        f"{exit_price:.2f}",
        f"{profit_usd:.2f}",
        f"{equity_after:.2f}",
    ]
    append_csv_row(TRADE_LOG_PATH, header, row)


def log_equity_csv(
    timestamp: datetime,
    equity: Decimal,
    usd_balance_: Decimal,
    open_positions_count: int,
    drawdown_pct: Decimal,
) -> None:
    """Log equity curve once per loop."""
    header = ["timestamp", "equity", "usd_balance", "open_positions", "drawdown_pct"]
    row = [
        timestamp.isoformat(),
        f"{equity:.2f}",
        f"{usd_balance_:.2f}",
        str(open_positions_count),
        f"{drawdown_pct:.2f}",
    ]
    append_csv_row(EQUITY_LOG_PATH, header, row)


def log_training_event_csv(
    entry_time: datetime,
    exit_time: datetime,
    market: str,
    features: Dict[str, Decimal],
    entry_price: Decimal,
    exit_price: Decimal,
    profit_usd: Decimal,
) -> None:
    """
    Log a "learning example" for future AI models.
    Each row captures the indicators at entry + the final trade result.
    """
    hold_minutes = (exit_time - entry_time).total_seconds() / 60.0
    pnl_pct = Decimal("0")
    if entry_price > 0:
        pnl_pct = (exit_price - entry_price) / entry_price

    header = [
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

    row = [
        entry_time.isoformat(),
        exit_time.isoformat(),
        f"{hold_minutes:.2f}",
        market,
        f"{features.get('trend_strength', Decimal('0')):.6f}",
        f"{features.get('rsi', Decimal('0')):.2f}",
        f"{features.get('volatility', Decimal('0')):.6f}",
        f"{entry_price:.2f}",
        f"{exit_price:.2f}",
        f"{profit_usd:.2f}",
        f"{pnl_pct * Decimal('100'):.2f}",
        f"{TAKE_PROFIT_PCT * Decimal('100'):.2f}",
        f"{STOP_LOSS_PCT * Decimal('100'):.2f}",
        RISK_MODE,
    ]

    append_csv_row(TRAINING_LOG_PATH, header, row)


# ---------------------------------------------------------------------------
# Market data helpers
# ---------------------------------------------------------------------------

def get_recent_candles(market: str, limit: int = LOOKBACK_CANDLES) -> Optional[List[Dict]]:
    """
    Fetch recent candles using Coinbase public API.
    Returns list of candles sorted oldest -> newest:
    [{"time": int, "open": Decimal, "high": Decimal,
      "low": Decimal, "close": Decimal, "volume": Decimal}, ...]
    """
    try:
        end_ts = int(time.time())
        start_ts = end_ts - limit * CANDLE_GRANULARITY

        params = {
            "start": datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
            "end": datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
            "granularity": CANDLE_GRANULARITY,
        }
        url = f"{PUBLIC_API_BASE}/products/{market}/candles"
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()

        # Coinbase returns newest first; we want oldest first
        raw.reverse()

        candles: List[Dict] = []
        for c in raw:
            # [time, low, high, open, close, volume]
            candles.append(
                {
                    "time": int(c[0]),
                    "low": Decimal(str(c[1])),
                    "high": Decimal(str(c[2])),
                    "open": Decimal(str(c[3])),
                    "close": Decimal(str(c[4])),
                    "volume": Decimal(str(c[5])),
                }
            )
        return candles
    except Exception as e:
        log(f"Error fetching candles for {market}: {e}")
        return None


def get_latest_price(market: str) -> Optional[Decimal]:
    """Get latest ticker price using Coinbase public API."""
    try:
        url = f"{PUBLIC_API_BASE}/products/{market}/ticker"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return Decimal(str(data["price"]))
    except Exception as e:
        log(f"Error fetching price for {market}: {e}")
        return None


# ---------------------------------------------------------------------------
# Indicators & scoring
# ---------------------------------------------------------------------------

def sma(values: List[Decimal], period: int) -> Optional[Decimal]:
    if len(values) < period:
        return None
    return sum(values[-period:]) / Decimal(period)


def rsi(values: List[Decimal], period: int = 14) -> Optional[Decimal]:
    if len(values) <= period:
        return None

    gains: List[Decimal] = []
    losses: List[Decimal] = []

    for i in range(1, period + 1):
        diff = values[-i] - values[-i - 1]
        if diff > 0:
            gains.append(diff)
        else:
            losses.append(-diff)

    if not gains and not losses:
        return Decimal("50")

    avg_gain = sum(gains) / Decimal(period) if gains else Decimal(0)
    avg_loss = sum(losses) / Decimal(period) if losses else Decimal(0)

    if avg_loss == 0:
        return Decimal("100")

    rs = avg_gain / avg_loss
    return Decimal("100") - (Decimal("100") / (Decimal("1") + rs))


def estimate_volatility(prices: List[Decimal]) -> Optional[Decimal]:
    """Average absolute % move between closes."""
    if len(prices) < 2:
        return None

    moves: List[Decimal] = []
    for i in range(1, len(prices)):
        if prices[i - 1] == 0:
            continue
        move = abs(prices[i] - prices[i - 1]) / prices[i - 1]
        moves.append(move)

    if not moves:
        return None

    return sum(moves) / Decimal(len(moves))


def get_market_score(
    market: str,
) -> Tuple[Decimal, Optional[Decimal], Optional[List[Decimal]], Optional[Dict[str, Decimal]]]:
    """
    Compute a score for a market based on trend, RSI, volatility and past performance.
    Returns (score, latest_price, closes, features) where score = -999 if unusable.
    features is a small dict of indicators used later for training logs.
    """
    candles = get_recent_candles(market)
    if not candles:
        return Decimal("-999"), None, None, None

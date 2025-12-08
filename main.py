from api_client import send_training_event_to_api
"""
Crypto-AI-Bot – experimental self-learning paper trading bot

- Uses ONLY public market data (no API keys, no real money)
- Scans random markets every few minutes
- Opens simple long positions based on trend/RSI
- Logs trades to data/trades.csv
- Logs equity curve to data/equity_curve.csv
- Logs training events to data/training_events.csv
- Sends each training event to the Crypto-AI-API for learning
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------
# Precision / global context
# ---------------------------------------------------------------------
getcontext().prec = 28  # high precision for crypto prices & PnL

# ---------------------------------------------------------------------
# Directories & file paths
# ---------------------------------------------------------------------
DATA_DIR = "data"
LOGS_DIR = "logs"

TRADE_LOG_PATH = os.path.join(DATA_DIR, "trades.csv")
EQUITY_LOG_PATH = os.path.join(DATA_DIR, "equity_curve.csv")
TRAINING_LOG_PATH = os.path.join(DATA_DIR, "training_events.csv")
RUNTIME_LOG_PATH = os.path.join(LOGS_DIR, "runtime.log")

# ---------------------------------------------------------------------
# Bot parameters
# ---------------------------------------------------------------------
START_BALANCE_USD = Decimal("1000")  # starting paper balance
RISK_MODE = "AGGRESSIVE"             # saved in training events

# Markets to randomly choose from
ALL_MARKETS: List[str] = [
    "BTC-USD",
    "ETH-USD",
    "ADA-USD",
    "SOL-USD",
    "LTC-USD",
    "AVAX-USD",
    "LINK-USD",
    "MATIC-USD",
    "UNI-USD",
    "DOGE-USD",
]

MAX_MARKETS_PER_SCAN = 8
SLEEP_SECONDS = 6 * 60  # 6 minutes between scans

# Indicators
CANDLE_GRANULARITY = 60 * 60  # 1h candles
LOOKBACK_CANDLES = 100

MIN_TREND_STRENGTH = 0.20  # simple trend slope filter
RSI_BUY_MIN = 30           # oversold threshold

# Public API – you can change this if you like
# Here we use Coinbase public candles as an example
PUBLIC_API_BASE = "https://api.exchange.coinbase.com"


# ---------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------
def ensure_directories() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


def setup_runtime_logger() -> None:
    logging.basicConfig(
        filename=RUNTIME_LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def log_runtime(message: str) -> None:
    print(message)
    logging.info(message)


def ensure_csv_headers() -> None:
    """Create CSV files with headers if they don't exist yet."""

    if not os.path.exists(TRADE_LOG_PATH):
        with open(TRADE_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "entry_time",
                    "exit_time",
                    "market",
                    "entry_price",
                    "exit_price",
                    "size_usd",
                    "pnl_usd",
                    "pnl_pct",
                    "take_profit_pct",
                    "stop_loss_pct",
                    "trend_strength",
                    "rsi",
                    "volatility",
                    "risk_mode",
                ]
            )

    if not os.path.exists(EQUITY_LOG_PATH):
        with open(EQUITY_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "equity_usd"])

    if not os.path.exists(TRAINING_LOG_PATH):
        with open(TRAINING_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
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
            )


# ---------------------------------------------------------------------
# Data / indicator helpers
# ---------------------------------------------------------------------
def get_now() -> datetime:
    return datetime.now(timezone.utc)


def fetch_candles(market: str) -> Optional[List[Dict]]:
    """
    Fetch OHLCV candles from a public API.
    Returns list of candles or None on error.
    Coinbase format: [time, low, high, open, close, volume]
    """
    url = f"{PUBLIC_API_BASE}/products/{market}/candles"
    params = {
        "granularity": CANDLE_GRANULARITY,
        "limit": LOOKBACK_CANDLES,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or len(data) < 20:
            return None

        # Sort oldest → newest (Coinbase returns newest first)
        data.sort(key=lambda x: x[0])

        candles = []
        for t, low, high, open_, close, vol in data:
            candles.append(
                {
                    "time": datetime.fromtimestamp(t, tz=timezone.utc),
                    "open": Decimal(str(open_)),
                    "high": Decimal(str(high)),
                    "low": Decimal(str(low)),
                    "close": Decimal(str(close)),
                    "volume": Decimal(str(vol)),
                }
            )
        return candles
    except Exception as e:
        log_runtime(f"[{market}] Error fetching candles: {e}")
        return None


def compute_rsi(closes: List[Decimal], period: int = 14) -> Decimal:
    if len(closes) < period + 1:
        return Decimal("50")

    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = closes[-i] - closes[-i - 1]
        if diff > 0:
            gains.append(diff)
        else:
            losses.append(-diff)

    avg_gain = sum(gains) / Decimal(len(gains) or 1)
    avg_loss = sum(losses) / Decimal(len(losses) or 1)

    if avg_loss == 0:
        return Decimal("100")

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_volatility(closes: List[Decimal]) -> Decimal:
    if len(closes) < 2:
        return Decimal("0")

    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] == 0:
            continue
        r = (closes[i] - closes[i - 1]) / closes[i - 1]
        returns.append(r)

    if not returns:
        return Decimal("0")

    mean = sum(returns) / Decimal(len(returns))
    var = sum((r - mean) ** 2 for r in returns) / Decimal(len(returns))
    std = var.sqrt()
    return std


def compute_trend_strength(closes: List[Decimal]) -> Decimal:
    """Very simple trend proxy: slope of last N closes."""
    n = min(20, len(closes))
    if n < 5:
        return Decimal("0")

    xs = list(range(n))
    ys = [float(c) for c in closes[-n:]]

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs) or 1.0
    slope = num / den

    # Normalise to a rough 0–1 range
    strength = Decimal(str(math.tanh(slope)))
    return strength


# ---------------------------------------------------------------------
# Trading logic
# ---------------------------------------------------------------------
class Position:
    def __init__(
        self,
        market: str,
        entry_time: datetime,
        entry_price: Decimal,
        size_usd: Decimal,
        take_profit_pct: Decimal,
        stop_loss_pct: Decimal,
        trend_strength: Decimal,
        rsi: Decimal,
        volatility: Decimal,
    ) -> None:
        self.market = market
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.size_usd = size_usd
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.trend_strength = trend_strength
        self.rsi = rsi
        self.volatility = volatility

        self.exit_time: Optional[datetime] = None
        self.exit_price: Optional[Decimal] = None
        self.pnl_usd: Optional[Decimal] = None
        self.pnl_pct: Optional[Decimal] = None

    def maybe_close(self, last_price: Decimal) -> bool:
        """
        Simple TP/SL logic. Returns True if the position closes this tick.
        """
        change_pct = (last_price - self.entry_price) / self.entry_price

        if change_pct >= self.take_profit_pct or change_pct <= -self.stop_loss_pct:
            self.exit_time = get_now()
            self.exit_price = last_price
            self.pnl_pct = change_pct
            self.pnl_usd = self.size_usd * change_pct
            return True
        return False


def pick_markets() -> List[str]:
    count = min(MAX_MARKETS_PER_SCAN, len(ALL_MARKETS))
    return random.sample(ALL_MARKETS, count)


def open_position(
    market: str,
    candles: List[Dict],
    balance_usd: Decimal,
) -> Optional[Position]:
    closes = [c["close"] for c in candles]
    rsi = compute_rsi(closes)
    trend_strength = compute_trend_strength(closes)
    volatility = compute_volatility(closes)

    last_price = closes[-1]

    if trend_strength < Decimal(str(MIN_TREND_STRENGTH)):
        return None
    if rsi > RSI_BUY_MIN:
        return None

    # Very simple sizing: 10% of equity
    size_usd = balance_usd * Decimal("0.10")

    take_profit_pct = Decimal("0.03")  # 3%
    stop_loss_pct = Decimal("0.015")   # 1.5%

    pos = Position(
        market=market,
        entry_time=get_now(),
        entry_price=last_price,
        size_usd=size_usd,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        trend_strength=trend_strength,
        rsi=rsi,
        volatility=volatility,
    )

    log_runtime(
        f"[OPEN] {market} @ {last_price} | size ${size_usd} | trend={trend_strength:.3f} RSI={rsi:.1f}"
    )
    return pos


def log_trade_and_training_event(pos: Position) -> None:
    """
    Append trade row to trades.csv, training row to training_events.csv,
    and send the training event to the external API.
    """
    assert pos.exit_time is not None
    assert pos.exit_price is not None
    assert pos.pnl_usd is not None
    assert pos.pnl_pct is not None

    entry_time_str = pos.entry_time.isoformat()
    exit_time_str = pos.exit_time.isoformat()
    hold_minutes = (pos.exit_time - pos.entry_time).total_seconds() / 60.0

    # 1) Log trades.csv
    with open(TRADE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                entry_time_str,
                exit_time_str,
                pos.market,
                f"{pos.entry_price:f}",
                f"{pos.exit_price:f}",
                f"{pos.size_usd:f}",
                f"{pos.pnl_usd:f}",
                f"{pos.pnl_pct:f}",
                f"{pos.take_profit_pct:f}",
                f"{pos.stop_loss_pct:f}",
                f"{pos.trend_strength:f}",
                f"{pos.rsi:f}",
                f"{pos.volatility:f}",
                RISK_MODE,
            ]
        )

    # 2) Build training event dict
    event = {
        "entry_time": entry_time_str,
        "exit_time": exit_time_str,
        "hold_minutes": round(hold_minutes, 2),
        "market": pos.market,
        "trend_strength": float(pos.trend_strength),
        "rsi": float(pos.rsi),
        "volatility": float(pos.volatility),
        "entry_price": float(pos.entry_price),
        "exit_price": float(pos.exit_price),
        "pnl_usd": float(pos.pnl_usd),
        "pnl_pct": float(pos.pnl_pct),
        "take_profit_pct": float(pos.take_profit_pct),
        "stop_loss_pct": float(pos.stop_loss_pct),
        "risk_mode": RISK_MODE,
    }

    # 3) Append to training_events.csv
    with open(TRAINING_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                event["entry_time"],
                event["exit_time"],
                event["hold_minutes"],
                event["market"],
                event["trend_strength"],
                event["rsi"],
                event["volatility"],
                event["entry_price"],
                event["exit_price"],
                event["pnl_usd"],
                event["pnl_pct"],
                event["take_profit_pct"],
                event["stop_loss_pct"],
                event["risk_mode"],
            ]
        )

    # 4) Send to external API (non-blocking for the bot)
    try:
        send_training_event_to_api(event)
    except Exception as e:
        log_runtime(f"[WARN] Failed to send training event to API: {e}")


def log_equity(equity_usd: Decimal) -> None:
    with open(EQUITY_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([get_now().isoformat(), f"{equity_usd:f}"])


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def main() -> None:
    ensure_directories()
    setup_runtime_logger()
    ensure_csv_headers()

    balance_usd = START_BALANCE_USD
    open_positions: Dict[str, Position] = {}

    log_runtime("==== Crypto-AI-Bot starting ====")
    log_runtime(f"Start balance: ${balance_usd}")

    while True:
        try:
            markets = pick_markets()
            log_runtime(f"Scanning {len(markets)} markets: {', '.join(markets)}")

            # 1) Update/close existing positions with latest prices
            for market, pos in list(open_positions.items()):
                candles = fetch_candles(market)
                if not candles:
                    continue

                last_price = candles[-1]["close"]
                if pos.maybe_close(last_price):
                    # Update balance
                    assert pos.pnl_usd is not None
                    balance_usd += pos.pnl_usd
                    log_runtime(
                        f"[CLOSE] {market} @ {pos.exit_price} | "
                        f"PnL ${pos.pnl_usd:.2f} ({pos.pnl_pct:.2%}) | "
                        f"Equity ${balance_usd:.2f}"
                    )
                    log_trade_and_training_event(pos)
                    del open_positions[market]

            # 2) Try to open new positions on random markets
            for market in markets:
                if market in open_positions:
                    continue

                candles = fetch_candles(market)
                if not candles:
                    continue

                pos = open_position(market, candles, balance_usd)
                if pos:
                    open_positions[market] = pos

            # 3) Log equity
            log_equity(balance_usd)

            # 4) Sleep before next cycle
            log_runtime(
                f"Cycle complete – positions={len(open_positions)}, equity=${balance_usd:.2f}. "
                f"Sleeping {SLEEP_SECONDS} seconds..."
            )
            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            log_runtime(f"[ERROR] Main loop crashed: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()

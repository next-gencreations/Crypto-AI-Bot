from __future__ import annotations

"""
Crypto-AI-Bot - experimental self-learning paper bot.

- Public market data only (NO API KEYS, NO REAL MONEY)
- Scans random markets every few minutes
- Uses simple TP/SL per position
- Manages multiple long positions at once
- Logs trades to data/trades.csv
- Logs equity to data/equity_curve.csv
- Logs runtime messages to logs/runtime.log
- Logs feature snapshots to data/training_events.csv
- Sends each closed trade to the Crypto AI API service
"""

import os
import csv
import time
import math
import random
import logging
from decimal import Decimal, getcontext
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import requests

from api_client import send_training_event_to_api  # <- your existing helper


# ---------------------------------------------------------------------------
# Precision / global context
# ---------------------------------------------------------------------------

getcontext().prec = 28  # high precision for crypto prices & PnL


# ---------------------------------------------------------------------------
# Directories & file paths
# ---------------------------------------------------------------------------

DATA_DIR = "data"
LOGS_DIR = "logs"

TRADE_LOG_PATH = os.path.join(DATA_DIR, "trades.csv")
EQUITY_LOG_PATH = os.path.join(DATA_DIR, "equity_curve.csv")
RUNTIME_LOG_PATH = os.path.join(LOGS_DIR, "runtime.log")
TRAINING_LOG_PATH = os.path.join(DATA_DIR, "training_events.csv")


# ---------------------------------------------------------------------------
# Bot configuration
# ---------------------------------------------------------------------------

START_BALANCE_USD = Decimal("1000")

# Risk mode label (logged into training data)
RISK_MODE = "AGGRESSIVE"

# Markets to trade – Coinbase uses symbols like BTC, ETH, etc.
ALL_MARKETS: List[str] = ["BTC", "ETH", "ADA", "SOL", "LTC", "BCH"]

# How many markets we randomly pick per scan
MAX_MARKETS_PER_SCAN = 3

# How long to sleep between scans (in seconds)
SLEEP_SECONDS = 360  # 6 minutes

# Take-profit / stop-loss levels (as percentages, e.g. 0.015 = 1.5%)
DEFAULT_TAKE_PROFIT_PCT = Decimal("0.015")
DEFAULT_STOP_LOSS_PCT = Decimal("0.010")

# Percentage of equity we "risk" per position
if RISK_MODE == "AGGRESSIVE":
    RISK_PER_TRADE_PCT = Decimal("0.02")  # 2%
else:
    RISK_PER_TRADE_PCT = Decimal("0.01")  # 1%

# Coinbase public spot price endpoint template
COINBASE_URL_TEMPLATE = "https://api.coinbase.com/v2/prices/{symbol}-USD/spot"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(RUNTIME_LOG_PATH)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # Console handler (for Render logs)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root.addHandler(ch)


# ---------------------------------------------------------------------------
# Files / directories helpers
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


def init_csv_files_if_needed() -> None:
    """
    Create CSV files with headers if they don't exist yet.
    """

    if not os.path.exists(TRADE_LOG_PATH):
        with open(TRADE_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "entry_time",
                    "exit_time",
                    "hold_minutes",
                    "market",
                    "entry_price",
                    "exit_price",
                    "qty",
                    "pnl_usd",
                    "pnl_pct",
                    "take_profit_pct",
                    "stop_loss_pct",
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


# ---------------------------------------------------------------------------
# Price fetching (Coinbase public API)
# ---------------------------------------------------------------------------

def fetch_price(symbol: str) -> Optional[Decimal]:
    """
    Fetch spot price in USD for a symbol from Coinbase public API.

    symbol: "BTC", "ETH", "ADA", etc.
    Returns Decimal price or None on error.
    """
    try:
        url = COINBASE_URL_TEMPLATE.format(symbol=symbol)
        resp = requests.get(url, timeout=10)

        if resp.status_code != 200:
            logging.error(
                f"[ERROR] Coinbase {symbol}-USD status {resp.status_code}: {resp.text[:200]}"
            )
            return None

        data = resp.json()
        amount_str = data["data"]["amount"]
        price = Decimal(amount_str)

        return price

    except Exception as e:
        logging.error(f"[ERROR] Exception in fetch_price({symbol}): {e}")
        return None


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------

class Position:
    def __init__(
        self,
        market: str,
        entry_time: datetime,
        entry_price: Decimal,
        qty: Decimal,
        take_profit_pct: Decimal,
        stop_loss_pct: Decimal,
        risk_mode: str,
    ):
        self.market = market  # e.g. "BTC-USD"
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.qty = qty
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.risk_mode = risk_mode

    def evaluate(self, current_price: Decimal) -> Tuple[bool, Decimal, Decimal]:
        """
        Evaluate TP/SL.

        Returns:
            (should_close, pnl_usd, pnl_pct)
        """
        pnl_usd = (current_price - self.entry_price) * self.qty
        if self.entry_price == 0:
            pnl_pct = Decimal("0")
        else:
            pnl_pct = (current_price - self.entry_price) / self.entry_price

        # Take profit hit?
        if pnl_pct >= self.take_profit_pct:
            return True, pnl_usd, pnl_pct

        # Stop loss hit?
        if pnl_pct <= -self.stop_loss_pct:
            return True, pnl_usd, pnl_pct

        return False, pnl_usd, pnl_pct


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_equity(time_utc: datetime, equity_usd: Decimal) -> None:
    with open(EQUITY_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([time_utc.isoformat(), str(equity_usd)])


def log_trade_row(row: Dict[str, Any]) -> None:
    with open(TRADE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                row["entry_time"],
                row["exit_time"],
                row["hold_minutes"],
                row["market"],
                row["entry_price"],
                row["exit_price"],
                row["qty"],
                row["pnl_usd"],
                row["pnl_pct"],
                row["take_profit_pct"],
                row["stop_loss_pct"],
                row["risk_mode"],
            ]
        )


def log_training_event_row(row: Dict[str, Any]) -> None:
    with open(TRAINING_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                row["entry_time"],
                row["exit_time"],
                row["hold_minutes"],
                row["market"],
                row["trend_strength"],
                row["rsi"],
                row["volatility"],
                row["entry_price"],
                row["exit_price"],
                row["pnl_usd"],
                row["pnl_pct"],
                row["take_profit_pct"],
                row["stop_loss_pct"],
                row["risk_mode"],
            ]
        )


# ---------------------------------------------------------------------------
# Market selection
# ---------------------------------------------------------------------------

def pick_markets_to_scan() -> List[str]:
    """Pick a random subset of markets to scan this cycle."""
    n = min(MAX_MARKETS_PER_SCAN, len(ALL_MARKETS))
    return random.sample(ALL_MARKETS, n)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_bot() -> None:
    setup_logging()
    ensure_directories()
    init_csv_files_if_needed()

    equity = START_BALANCE_USD
    open_positions: Dict[str, Position] = {}  # key: "BTC-USD", etc.

    logging.info("Crypto-AI-Bot starting up")
    logging.info(f"Starting equity: {equity}")

    cycle = 0

    while True:
        cycle += 1
        now = datetime.now(timezone.utc)
        logging.info("=" * 70)
        logging.info(f"[CYCLE {cycle}] Starting scan at {now.isoformat()}")

        markets = pick_markets_to_scan()
        logging.info(f"[CYCLE {cycle}] Selected markets: {markets}")

        # 1. Evaluate existing positions for TP/SL exits
        for market, position in list(open_positions.items()):
            symbol = market.split("-")[0]  # "BTC-USD" -> "BTC"
            price = fetch_price(symbol)
            if price is None:
                logging.warning(f"[CYCLE {cycle}] Skipping eval for {market} (no price)")
                continue

            should_close, pnl_usd, pnl_pct = position.evaluate(price)

            if should_close:
                exit_time = now
                hold_minutes = int(
                    (exit_time - position.entry_time).total_seconds() / 60
                )

                equity += pnl_usd

                logging.info(
                    f"[EXIT] {market} "
                    f"entry={position.entry_price} exit={price} qty={position.qty} "
                    f"pnl_usd={pnl_usd:.4f} pnl_pct={pnl_pct * 100:.2f}% "
                    f"equity={equity:.2f}"
                )

                # Build row for trade + training event
                base_row: Dict[str, Any] = {
                    "entry_time": position.entry_time.isoformat(),
                    "exit_time": exit_time.isoformat(),
                    "hold_minutes": hold_minutes,
                    "market": market,
                    "entry_price": str(position.entry_price),
                    "exit_price": str(price),
                    "qty": str(position.qty),
                    "pnl_usd": str(pnl_usd),
                    "pnl_pct": str(pnl_pct),
                    "take_profit_pct": str(position.take_profit_pct),
                    "stop_loss_pct": str(position.stop_loss_pct),
                    "risk_mode": position.risk_mode,
                    # simple placeholder features for now
                    "trend_strength": "0",
                    "rsi": "0",
                    "volatility": "0",
                }

                # Log locally
                log_trade_row(base_row)
                log_training_event_row(base_row)
                log_equity(exit_time, equity)

                # Send to API (best effort)
                try:
                    send_training_event_to_api(base_row)
                except Exception as e:
                    logging.error(f"[ERROR] Failed to send training event to API: {e}")

                # Remove the position from open set
                del open_positions[market]

        # 2. Open new positions in selected markets (if not already open)
        for symbol in markets:
            market_name = f"{symbol}-USD"

            if market_name in open_positions:
                logging.info(f"[CYCLE {cycle}] Skipping {market_name}, already open")
                continue

            price = fetch_price(symbol)
            if price is None:
                logging.warning(
                    f"[CYCLE {cycle}] Could not fetch price for {market_name}"
                )
                continue

            # Calculate position size based on risk
            dollar_risk = equity * RISK_PER_TRADE_PCT
            if DEFAULT_STOP_LOSS_PCT > 0:
                # approximate position size from risk and stop distance
                qty = (dollar_risk / (price * DEFAULT_STOP_LOSS_PCT)).quantize(
                    Decimal("0.0001")
                )
            else:
                qty = (dollar_risk / price).quantize(Decimal("0.0001"))

            if qty <= 0:
                logging.warning(
                    f"[CYCLE {cycle}] Computed non-positive qty for {market_name}, "
                    f"skipping open"
                )
                continue

            pos = Position(
                market=market_name,
                entry_time=now,
                entry_price=price,
                qty=qty,
                take_profit_pct=DEFAULT_TAKE_PROFIT_PCT,
                stop_loss_pct=DEFAULT_STOP_LOSS_PCT,
                risk_mode=RISK_MODE,
            )
            open_positions[market_name] = pos

            logging.info(
                f"[ENTRY] Opened LONG {market_name} @ {price} qty={qty} "
                f"TP={DEFAULT_TAKE_PROFIT_PCT * 100:.2f}% "
                f"SL={DEFAULT_STOP_LOSS_PCT * 100:.2f}% "
                f"equity={equity:.2f}"
            )

        logging.info(
            f"[CYCLE {cycle}] Open positions: {list(open_positions.keys()) or 'None'}"
        )
        logging.info(f"[CYCLE {cycle}] Sleeping for {SLEEP_SECONDS} seconds...")
        time.sleep(SLEEP_SECONDS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_bot()

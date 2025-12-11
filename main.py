#!/usr/bin/env python3
"""
Crypto-AI-Bot worker

- Runs a simple long-only paper-trading loop.
- Fetches prices from Binance.
- Writes CSV files + heartbeat JSON that the dashboard service reads.
- Optionally sends training events to the Crypto AI API.

This file is meant to run 24/7 on a Render "worker" service.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional

import requests

# Use decent precision for PnL maths
getcontext().prec = 18

# ---------------------------------------------------------------------------
# Paths & files
# ---------------------------------------------------------------------------

DATA_DIR = os.environ.get("CRYPTO_AI_DATA_DIR", "data")

TRADE_FILE = os.path.join(DATA_DIR, "trades.csv")
EQUITY_FILE = os.path.join(DATA_DIR, "equity.csv")
TRAINING_FILE = os.path.join(DATA_DIR, "training_events.csv")
HEARTBEAT_FILE = os.path.join(DATA_DIR, "heartbeat.json")

# ---------------------------------------------------------------------------
# Config (can be overridden via env)
# ---------------------------------------------------------------------------

START_BALANCE_USD = Decimal(os.environ.get("START_BALANCE_USD", "1000"))

# Risk per trade as % of equity (e.g. 0.02 == 2%)
RISK_PER_TRADE_PCT = Decimal(os.environ.get("RISK_PER_TRADE_PCT", "0.02"))

# Take-profit / stop-loss in fractional terms (0.03 = +3%, -0.02 = -2%)
DEFAULT_TAKE_PROFIT_PCT = Decimal(os.environ.get("DEFAULT_TP_PCT", "0.03"))
DEFAULT_STOP_LOSS_PCT = Decimal(os.environ.get("DEFAULT_SL_PCT", "0.02"))

# Sleep between cycles in seconds
SLEEP_SECONDS = int(os.environ.get("SLEEP_SECONDS", "60"))

# Max distinct markets to consider per cycle
MAX_MARKETS_PER_SCAN = int(os.environ.get("MAX_MARKETS_PER_SCAN", "3"))

# Risk mode label (for training / analytics only)
RISK_MODE = os.environ.get("RISK_MODE", "basic_v1")

# Markets universe (base symbols – we treat everything as XXX-USD)
ALL_MARKETS: List[str] = [
    "BTC",
    "ETH",
    "SOL",
    "ADA",
    "LTC",
    "BCH",
    "XRP",
    "DOGE",
    "LINK",
    "AVAX",
]

# Map our symbol (e.g. "BTC") to Binance symbol (e.g. "BTCUSDT")
BINANCE_SYMBOL_MAP: Dict[str, str] = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "ADA": "ADAUSDT",
    "LTC": "LTCUSDT",
    "BCH": "BCHUSDT",
    "XRP": "XRPUSDT",
    "DOGE": "DOGEUSDT",
    "LINK": "LINKUSDT",
    "AVAX": "AVAXUSDT",
}

# Training API (optional)
TRAINING_API_URL = os.environ.get("TRAINING_API_URL")  # e.g. https://crypto-ai-api-h921.onrender.com/training
TRAINING_API_KEY = os.environ.get("TRAINING_API_KEY")  # optional bearer key

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Logging initialised (level=%s)", log_level)


# ---------------------------------------------------------------------------
# Files & directories
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def init_csv_files_if_needed() -> None:
    """
    Create CSV files with headers if they do not yet exist.
    The dashboard API expects exactly these columns.
    """
    if not os.path.exists(TRADE_FILE):
        with open(TRADE_FILE, "w", newline="", encoding="utf-8") as f:
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
                    "trend_strength",
                    "rsi",
                    "volatility",
                ]
            )

    if not os.path.exists(EQUITY_FILE):
        with open(EQUITY_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "equity_usd"])

    if not os.path.exists(TRAINING_FILE):
        with open(TRAINING_FILE, "w", newline="", encoding="utf-8") as f:
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
                    "trend_strength",
                    "rsi",
                    "volatility",
                ]
            )

    # Heartbeat file is JSON; we just ensure the directory exists
    if not os.path.exists(HEARTBEAT_FILE):
        write_heartbeat(datetime.now(timezone.utc))


def write_heartbeat(ts: datetime) -> None:
    """
    Write a simple heartbeat JSON indicating the last time the bot was alive.
    The dashboard service uses this to compute "bot status".
    """
    payload = {"last_heartbeat": ts.isoformat()}
    try:
        with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception as e:
        logging.error("[HEARTBEAT] Failed to write heartbeat: %s", e)


def log_trade_row(row: Dict[str, Any]) -> None:
    with open(TRADE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)


def log_training_event_row(row: Dict[str, Any]) -> None:
    with open(TRAINING_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)


def log_equity(ts: datetime, equity: Decimal) -> None:
    with open(EQUITY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([ts.isoformat(), str(equity)])


# ---------------------------------------------------------------------------
# Training API client (optional)
# ---------------------------------------------------------------------------

def send_training_event_to_api(row: Dict[str, Any]) -> None:
    """
    Best-effort POST to the Crypto AI training API.
    Safe to fail; dashboard will still work from CSVs.
    """
    if not TRAINING_API_URL:
        # No URL configured; silently skip.
        return

    headers = {
        "Content-Type": "application/json",
    }
    if TRAINING_API_KEY:
        headers["Authorization"] = f"Bearer {TRAINING_API_KEY}"

    try:
        resp = requests.post(TRAINING_API_URL, json=row, headers=headers, timeout=5)
        if resp.status_code >= 300:
            logging.warning(
                "[TRAINING_API] Bad status %s: %s",
                resp.status_code,
                resp.text[:200],
            )
    except Exception as e:
        logging.error("[TRAINING_API] Error sending training event: %s", e)


# ---------------------------------------------------------------------------
# Market data & position logic
# ---------------------------------------------------------------------------

def fetch_price(symbol: str) -> Optional[Decimal]:
    """
    Fetch latest price from Binance API with a proper User-Agent.
    Binance can return 451 errors if no User-Agent header is sent.
    """
    binance_symbol = BINANCE_SYMBOL_MAP.get(symbol, f"{symbol}USDT")
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}"

    headers = {
        # Looks like a normal browser – avoids 451 / blocking
        "User-Agent": "Mozilla/5.0 (compatible; CryptoAIBot/1.0; +https://example.com)",
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        price = Decimal(data["price"])
        return price
    except Exception as e:
        logging.error("[PRICE] Error fetching %s from Binance: %s", symbol, e)
        return None


@dataclass
class Position:
    market: str          # "BTC-USD"
    entry_time: datetime
    entry_price: Decimal
    qty: Decimal
    take_profit_pct: Decimal
    stop_loss_pct: Decimal
    risk_mode: str

    def evaluate(self, current_price: Decimal) -> (bool, Decimal, Decimal):
        """
        Check if TP or SL has been hit.

        Returns (should_close, pnl_usd, pnl_pct)
        """
        change = current_price - self.entry_price
        pnl_pct = change / self.entry_price if self.entry_price != 0 else Decimal("0")
        pnl_usd = pnl_pct * self.entry_price * self.qty

        if pnl_pct >= self.take_profit_pct:
            return True, pnl_usd, pnl_pct

        if pnl_pct <= -self.stop_loss_pct:
            return True, pnl_usd, pnl_pct

        return False, pnl_usd, pnl_pct


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

    equity: Decimal = START_BALANCE_USD
    open_positions: Dict[str, Position] = {}  # key: "BTC-USD", etc.

    logging.info("Crypto-AI-Bot starting up")
    logging.info("Starting equity: %.2f", equity)

    cycle = 0

    while True:
        cycle += 1
        now = datetime.now(timezone.utc)
        logging.info("=" * 70)
        logging.info("[CYCLE %s] Starting scan at %s", cycle, now.isoformat())

        markets = pick_markets_to_scan()
        logging.info("[CYCLE %s] Selected markets: %s", cycle, markets)

        # ---------------------------------------------------------------
        # 1) Evaluate existing positions for TP/SL exits
        # ---------------------------------------------------------------
        for market, position in list(open_positions.items()):
            base_symbol = market.split("-")[0]  # "BTC-USD" -> "BTC"
            price = fetch_price(base_symbol)
            if price is None:
                logging.warning(
                    "[CYCLE %s] Skipping eval for %s (no price)", cycle, market
                )
                continue

            should_close, pnl_usd, pnl_pct = position.evaluate(price)

            if should_close:
                exit_time = now
                hold_minutes = int(
                    (exit_time - position.entry_time).total_seconds() / 60
                )

                equity += pnl_usd

                logging.info(
                    "[EXIT] %s entry=%.4f exit=%.4f qty=%s pnl_usd=%.4f pnl_pct=%.2f%% equity=%.2f",
                    market,
                    position.entry_price,
                    price,
                    position.qty,
                    pnl_usd,
                    pnl_pct * 100,
                    equity,
                )

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
                    # Placeholder feature columns for future AI training
                    "trend_strength": "0",
                    "rsi": "0",
                    "volatility": "0",
                }

                # Persist locally
                log_trade_row(base_row)
                log_training_event_row(base_row)
                log_equity(exit_time, equity)

                # Try to send to training API
                try:
                    send_training_event_to_api(base_row)
                except Exception as e:
                    logging.error("[TRAINING] Failed to send event: %s", e)

                # Remove from open positions
                del open_positions[market]

        # ---------------------------------------------------------------
        # 2) Open new positions in selected markets (if not already open)
        # ---------------------------------------------------------------
        for symbol in markets:
            market_name = f"{symbol}-USD"

            if market_name in open_positions:
                logging.info(
                    "[CYCLE %s] Skipping %s, position already open", cycle, market_name
                )
                continue

            price = fetch_price(symbol)
            if price is None:
                logging.warning(
                    "[CYCLE %s] Could not fetch price for %s", cycle, market_name
                )
                continue

            # Position sizing
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
                    "[CYCLE %s] Computed non-positive qty for %s, skipping",
                    cycle,
                    market_name,
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
                "[ENTRY] Opened LONG %s @ %.4f qty=%s TP=%.2f%% SL=%.2f%% equity=%.2f",
                market_name,
                price,
                qty,
                DEFAULT_TAKE_PROFIT_PCT * 100,
                DEFAULT_STOP_LOSS_PCT * 100,
                equity,
            )

        # ---------------------------------------------------------------
        # 3) Heartbeat & sleep
        # ---------------------------------------------------------------
        write_heartbeat(now)
        logging.info(
            "[CYCLE %s] Open positions: %s",
            cycle,
            list(open_positions.keys()) or "None",
        )
        logging.info("[CYCLE %s] Sleeping for %s seconds...", cycle, SLEEP_SECONDS)
        time.sleep(SLEEP_SECONDS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_bot()

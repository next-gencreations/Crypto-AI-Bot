#!/usr/bin/env python3
"""
Crypto-AI Bot
-------------
Background worker that:

- Picks a random subset of markets each cycle
- Fetches prices from Coinbase
- Opens positions based on simple TP / SL rules
- Closes positions when TP / SL is hit
- Logs trades + equity curve + heartbeat to CSV / JSON
- Sends closed-trade "training events" to the Crypto-AI-API service

This file is meant to run on Render as a Background Worker.
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
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Paths / data files
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TRADE_FILE = DATA_DIR / "trades.csv"
EQUITY_FILE = DATA_DIR / "equity_curve.csv"
TRAINING_FILE = DATA_DIR / "training_events.csv"
HEARTBEAT_FILE = DATA_DIR / "heartbeat.json"

# ---------------------------------------------------------------------------
# Bot configuration
# ---------------------------------------------------------------------------

START_BALANCE_USD = Decimal("1000")

# risk per trade (fraction of equity)
RISK_PER_TRADE_PCT = Decimal("0.02")  # 2%

# default TP/SL in pct of entry price
DEFAULT_TAKE_PROFIT_PCT = Decimal("0.03")  # 3%
DEFAULT_STOP_LOSS_PCT = Decimal("0.02")    # 2%

RISK_MODE = "fixed_fractional"

# how often the bot wakes up
SLEEP_SECONDS = 60

# max number of markets to scan per cycle
MAX_MARKETS_PER_SCAN = 3

# markets we care about (base symbols, -USD on Coinbase)
ALL_MARKETS: List[str] = ["BTC", "ETH", "SOL", "LTC", "BCH", "ADA"]

# API for training events / dashboard
TRAINING_API_URL = os.environ.get(
    "TRAINING_API_URL",
    "https://crypto-ai-api-h921.onrender.com",
)

TRAINING_API_TIMEOUT = 5  # seconds

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# File / directory helpers
# ---------------------------------------------------------------------------


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _init_csv(path: Path, headers: List[str]) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        logging.info(f"Initialised {path}")


def init_csv_files_if_needed() -> None:
    _init_csv(
        TRADE_FILE,
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
        ],
    )
    _init_csv(
        EQUITY_FILE,
        [
            "time",
            "equity_usd",
        ],
    )
    _init_csv(
        TRAINING_FILE,
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
        ],
    )


def write_heartbeat(now: datetime) -> None:
    """Write a small JSON heartbeat file so the API dashboard can see status."""
    payload = {
        "last_heartbeat": now.isoformat(),
    }
    try:
        with HEARTBEAT_FILE.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception as e:
        logging.error(f"[HEARTBEAT] Failed to write heartbeat: {e}")


def log_trade_row(row: Dict[str, Any]) -> None:
    """Append a row to trades.csv."""
    try:
        with TRADE_FILE.open("a", newline="", encoding="utf-8") as f:
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
                    row.get("trend_strength", "0"),
                    row.get("rsi", "0"),
                    row.get("volatility", "0"),
                ]
            )
    except Exception as e:
        logging.error(f"[TRADES] Error writing trade row: {e}")


def log_training_event_row(row: Dict[str, Any]) -> None:
    """Append a row to training_events.csv (for backup / replay)."""
    try:
        with TRAINING_FILE.open("a", newline="", encoding="utf-8") as f:
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
                    row.get("trend_strength", "0"),
                    row.get("rsi", "0"),
                    row.get("volatility", "0"),
                ]
            )
    except Exception as e:
        logging.error(f"[TRAINING] Error writing training row: {e}")


def log_equity(t: datetime, equity: Decimal) -> None:
    """Append a row to equity_curve.csv."""
    try:
        with EQUITY_FILE.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    t.isoformat(),
                    str(equity),
                ]
            )
    except Exception as e:
        logging.error(f"[EQUITY] Error writing equity row: {e}")


# ---------------------------------------------------------------------------
# API client to Crypto-AI-API
# ---------------------------------------------------------------------------


def send_training_event_to_api(row: Dict[str, Any]) -> None:
    """
    Send a closed-trade training event to the Crypto-AI-API service.

    Expects the web service to expose POST /api/training-event
    that accepts a JSON body matching this row.
    """
    url = TRAINING_API_URL.rstrip("/") + "/api/training-event"

    try:
        resp = requests.post(
            url,
            json=row,
            timeout=TRAINING_API_TIMEOUT,
            headers={"User-Agent": "Crypto-AI-Bot/1.0"},
        )
        if resp.status_code >= 400:
            logging.error(
                f"[TRAINING_API] Bad status {resp.status_code} for training event: {resp.text}"
            )
    except Exception as e:
        logging.error(f"[TRAINING_API] Error sending training event: {e}")


# ---------------------------------------------------------------------------
# Market data + position logic
# ---------------------------------------------------------------------------


def fetch_price(symbol: str) -> Optional[Decimal]:
    """
    Fetch latest price from Coinbase.

    Uses the public product ticker:
    https://api.exchange.coinbase.com/products/BTC-USD/ticker
    """
    market = f"{symbol}-USD"
    url = f"https://api.exchange.coinbase.com/products/{market}/ticker"

    headers = {
        "User-Agent": "Crypto-AI-Bot/1.0",
        "Accept": "application/json",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        price_str = data.get("price")
        if price_str is None:
            logging.error(f"[PRICE] No 'price' field in Coinbase response for {market}")
            return None

        price = Decimal(price_str)
        return price

    except (requests.RequestException, InvalidOperation) as e:
        logging.error(f"[PRICE] Error fetching {market} price from Coinbase: {e}")
        return None


@dataclass
class Position:
    market: str          # e.g. "BTC-USD"
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
        pnl_usd = change * self.qty

        hit_tp = pnl_pct >= self.take_profit_pct
        hit_sl = pnl_pct <= -self.stop_loss_pct

        return hit_tp or hit_sl, pnl_usd, pnl_pct


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

        # heartbeat for the dashboard
        write_heartbeat(now)

        markets = pick_markets_to_scan()
        logging.info(f"[CYCLE {cycle}] Selected markets: {markets}")

        # ------------------------------------------------------------------
        # 1) Evaluate existing positions for TP/SL exits
        # ------------------------------------------------------------------
        for market, position in list(open_positions.items()):
            symbol = market.split("-")[0]
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
                    # placeholder features for now
                    "trend_strength": "0",
                    "rsi": "0",
                    "volatility": "0",
                }

                # Log locally
                log_trade_row(base_row)
                log_training_event_row(base_row)
                log_equity(exit_time, equity)

                # Send to API (best effort)
                send_training_event_to_api(base_row)

                # Remove the position from open set
                del open_positions[market]

        # ------------------------------------------------------------------
        # 2) Open new positions in selected markets (if not already open)
        # ------------------------------------------------------------------
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

# main.py  – Crypto AI Bot worker (Render)
# ---------------------------------------
# Paper-trading bot that:
# - Fetches BTC/ETH/SOL/LTC/ADA/BCH prices from Coinbase
# - Opens / closes positions with simple TP/SL logic
# - Logs trades to CSV for the dashboard
# - Logs equity curve
# - Writes a heartbeat.json file for bot status
# - (Optionally) POSTs training events to an API

from __future__ import annotations

import csv
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

TRADE_FILE = os.path.join(DATA_DIR, "trades.csv")
EQUITY_FILE = os.path.join(DATA_DIR, "equity_curve.csv")
TRAINING_FILE = os.path.join(DATA_DIR, "training_events.csv")
HEARTBEAT_FILE = os.path.join(DATA_DIR, "heartbeat.json")

# Trading params
START_BALANCE_USD = Decimal("1000")

RISK_PER_TRADE_PCT = Decimal("0.01")          # 1% of equity per trade
DEFAULT_TAKE_PROFIT_PCT = Decimal("0.03")     # +3%
DEFAULT_STOP_LOSS_PCT = Decimal("0.02")       # -2%
RISK_MODE = "fixed_1pct"

SLEEP_SECONDS = 60                            # wait between cycles
MAX_MARKETS_PER_SCAN = 3

# Markets to rotate through (paper trading)
ALL_MARKETS: List[str] = ["BTC", "ETH", "SOL", "LTC", "ADA", "BCH"]

# Optional: training API (can be left unset)
TRAINING_API_URL = os.environ.get("TRAINING_API_URL")

# CSV headers (must match dashboard expectations)
TRADE_HEADERS = [
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

EQUITY_HEADERS = ["time", "equity_usd"]

TRAINING_HEADERS = TRADE_HEADERS  # we log same fields for now


# ---------------------------------------------------------------------------
# Setup / filesystem helpers
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def ensure_directories() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def init_csv_files_if_needed() -> None:
    """Create CSV files with headers if they don't exist yet."""
    if not os.path.exists(TRADE_FILE):
        with open(TRADE_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_HEADERS)
            writer.writeheader()
        logging.info("Initialised %s", TRADE_FILE)

    if not os.path.exists(EQUITY_FILE):
        with open(EQUITY_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EQUITY_HEADERS)
            writer.writeheader()
        logging.info("Initialised %s", EQUITY_FILE)

    if not os.path.exists(TRAINING_FILE):
        with open(TRAINING_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRAINING_HEADERS)
            writer.writeheader()
        logging.info("Initialised %s", TRAINING_FILE)


def write_heartbeat(ts: datetime) -> None:
    """Write heartbeat.json so the dashboard can show bot status."""
    try:
        payload = {"last_heartbeat": ts.isoformat()}
        with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception as e:
        logging.error("Failed to write heartbeat: %s", e)


def log_trade_row(row: Dict[str, Any]) -> None:
    with open(TRADE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_HEADERS)
        writer.writerow(row)


def log_training_event_row(row: Dict[str, Any]) -> None:
    with open(TRAINING_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRAINING_HEADERS)
        writer.writerow(row)


def log_equity(ts: datetime, equity: Decimal) -> None:
    with open(EQUITY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EQUITY_HEADERS)
        writer.writerow(
            {
                "time": ts.isoformat(),
                "equity_usd": str(equity),
            }
        )


def send_training_event_to_api(row: Dict[str, Any]) -> None:
    """Optional: send training event to remote API."""
    if not TRAINING_API_URL:
        # Not configured, just skip.
        return

    try:
        resp = requests.post(TRAINING_API_URL, json=row, timeout=5)
        if resp.status_code >= 400:
            logging.error(
                "[TRAINING_API] Bad status %s: %s",
                resp.status_code,
                resp.text[:200],
            )
    except Exception as e:
        logging.error("[TRAINING_API] Error sending event: %s", e)


# ---------------------------------------------------------------------------
# Market data – Coinbase
# ---------------------------------------------------------------------------

def fetch_price(symbol: str) -> Optional[Decimal]:
    """
    Fetch latest price from Coinbase's public API.
    Example: BTC -> https://api.coinbase.com/v2/prices/BTC-USD/spot
    """
    pair = f"{symbol}-USD"
    url = f"https://api.coinbase.com/v2/prices/{pair}/spot"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        amount = data["data"]["amount"]
        price = Decimal(amount)
        return price
    except Exception as e:
        logging.error("[PRICE] Error fetching %s price from Coinbase: %s", symbol, e)
        return None


# ---------------------------------------------------------------------------
# Position model
# ---------------------------------------------------------------------------

@dataclass
class Position:
    market: str              # e.g. "BTC-USD"
    entry_time: datetime
    entry_price: Decimal
    qty: Decimal
    take_profit_pct: Decimal
    stop_loss_pct: Decimal
    risk_mode: str

    def evaluate(self, current_price: Decimal) -> tuple[bool, Decimal, Decimal]:
        """
        Check if TP or SL has been hit.

        Returns (should_close, pnl_usd, pnl_pct).
        """
        change = current_price - self.entry_price
        pnl_usd = change * self.qty

        if self.entry_price == 0:
            pnl_pct = Decimal("0")
        else:
            pnl_pct = (current_price / self.entry_price) - Decimal("1")

        tp_hit = pnl_pct >= self.take_profit_pct
        sl_hit = pnl_pct <= -self.stop_loss_pct

        should_close = bool(tp_hit or sl_hit)
        return should_close, pnl_usd, pnl_pct


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
    open_positions: Dict[str, Position] = {}  # key: "BTC-USD"

    logging.info("Crypto-AI-Bot starting up")
    logging.info("Starting equity: %s", equity)

    cycle = 0

    while True:
        cycle += 1
        now = datetime.now(timezone.utc)

        # Heartbeat for dashboard
        write_heartbeat(now)

        logging.info("=" * 70)
        logging.info("[CYCLE %s] Starting scan at %s", cycle, now.isoformat())

        markets = pick_markets_to_scan()
        logging.info("[CYCLE %s] Selected markets: %s", cycle, markets)

        # 1. Evaluate existing positions for TP/SL exits
        for market, position in list(open_positions.items()):
            symbol = market.split("-")[0]  # "BTC-USD" -> "BTC"
            price = fetch_price(symbol)
            if price is None:
                logging.warning("[CYCLE %s] Skipping eval for %s (no price)", cycle, market)
                continue

            should_close, pnl_usd, pnl_pct = position.evaluate(price)

            if should_close:
                exit_time = now
                hold_minutes = int((exit_time - position.entry_time).total_seconds() / 60)

                equity += pnl_usd

                logging.info(
                    "[EXIT] %s entry=%s exit=%s qty=%s pnl_usd=%.4f pnl_pct=%.2f%% equity=%.2f",
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
                    # Placeholder features for later ML:
                    "trend_strength": "0",
                    "rsi": "0",
                    "volatility": "0",
                }

                # Log locally
                log_trade_row(base_row)
                log_training_event_row(base_row)
                log_equity(exit_time, equity)

                # Best-effort send to training API
                try:
                    send_training_event_to_api(base_row)
                except Exception as e:
                    logging.error("[ERROR] Failed to send training event to API: %s", e)

                del open_positions[market]

        # 2. Open new positions in selected markets (if not already open)
        for symbol in markets:
            market_name = f"{symbol}-USD"

            if market_name in open_positions:
                logging.info("[CYCLE %s] Skipping %s, already open", cycle, market_name)
                continue

            price = fetch_price(symbol)
            if price is None:
                logging.warning(
                    "[CYCLE %s] Could not fetch price for %s", cycle, market_name
                )
                continue

            # Calculate position size based on equity and risk
            dollar_risk = equity * RISK_PER_TRADE_PCT

            if DEFAULT_STOP_LOSS_PCT > 0:
                qty = (dollar_risk / (price * DEFAULT_STOP_LOSS_PCT)).quantize(
                    Decimal("0.0001")
                )
            else:
                qty = (dollar_risk / price).quantize(Decimal("0.0001"))

            if qty <= 0:
                logging.warning(
                    "[CYCLE %s] Computed non-positive qty for %s, skipping open",
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
                "[ENTRY] Opened LONG %s @ %s qty=%s TP=%.2f%% SL=%.2f%% equity=%.2f",
                market_name,
                price,
                qty,
                DEFAULT_TAKE_PROFIT_PCT * 100,
                DEFAULT_STOP_LOSS_PCT * 100,
                equity,
            )

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

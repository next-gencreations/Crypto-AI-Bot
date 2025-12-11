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
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Config & constants
# ---------------------------------------------------------------------------

# Where persistent data lives (Render disk is mounted here)
DATA_DIR = Path(os.getenv("DATA_DIR", "/opt/render/project/src/data"))

TRADES_FILE = DATA_DIR / "trades.csv"
TRAINING_FILE = DATA_DIR / "training_events.csv"
EQUITY_FILE = DATA_DIR / "equity_curve.csv"
HEARTBEAT_FILE = DATA_DIR / "heartbeat.json"

# Core trading config (can be overridden by environment variables on Render)
START_BALANCE_USD = Decimal(os.getenv("START_BALANCE_USD", "1000"))
RISK_PER_TRADE_PCT = Decimal(os.getenv("RISK_PER_TRADE_PCT", "0.02"))  # 2%
DEFAULT_TAKE_PROFIT_PCT = Decimal(os.getenv("DEFAULT_TAKE_PROFIT_PCT", "0.03"))  # 3%
DEFAULT_STOP_LOSS_PCT = Decimal(os.getenv("DEFAULT_STOP_LOSS_PCT", "0.02"))  # 2%
RISK_MODE = os.getenv("RISK_MODE", "fixed_pct")
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "60"))

# Markets we'll trade
ALL_MARKETS = ["BTC", "ETH", "LTC", "BCH", "SOL", "ADA"]
MAX_MARKETS_PER_SCAN = int(os.getenv("MAX_MARKETS_PER_SCAN", "3"))

# Binance symbols
BINANCE_SYMBOL_MAP: Dict[str, str] = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "LTC": "LTCUSDT",
    "BCH": "BCHUSDT",
    "SOL": "SOLUSDT",
    "ADA": "ADAUSDT",
}

# Optional API endpoint to receive live training events (your crypto-ai-api)
TRAINING_API_URL = os.getenv("TRAINING_API_URL", "").strip()


# ---------------------------------------------------------------------------
# Utility / setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def init_csv_files_if_needed() -> None:
    """Create CSV files with headers if they don't exist yet."""
    if not TRADES_FILE.exists():
        with TRADES_FILE.open("w", newline="") as f:
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
        logging.info(f"Initialised {TRADES_FILE}")

    if not TRAINING_FILE.exists():
        with TRAINING_FILE.open("w", newline="") as f:
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
        logging.info(f"Initialised {TRAINING_FILE}")

    if not EQUITY_FILE.exists():
        with EQUITY_FILE.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "equity"])
        logging.info(f"Initialised {EQUITY_FILE}")


def write_heartbeat(equity: Decimal) -> None:
    """Write a small JSON file the API can read for bot status."""
    payload = {
        "last_heartbeat": datetime.now(timezone.utc).isoformat(),
        "equity": float(equity),
    }
    try:
        with HEARTBEAT_FILE.open("w") as f:
            json.dump(payload, f)
    except Exception as e:
        logging.error(f"[HEARTBEAT] Failed to write heartbeat: {e}")


def log_trade_row(row: Dict[str, Any]) -> None:
    with TRADES_FILE.open("a", newline="") as f:
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
                row["trend_strength"],
                row["rsi"],
                row["volatility"],
            ]
        )


def log_training_event_row(row: Dict[str, Any]) -> None:
    with TRAINING_FILE.open("a", newline="") as f:
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
                row["trend_strength"],
                row["rsi"],
                row["volatility"],
            ]
        )


def log_equity(ts: datetime, equity: Decimal) -> None:
    with EQUITY_FILE.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts.isoformat(), str(equity)])


def send_training_event_to_api(row: Dict[str, Any]) -> None:
    """Optional: push training events to the API directly."""
    if not TRAINING_API_URL:
        return

    try:
        response = requests.post(TRAINING_API_URL, json=row, timeout=5)
        if not response.ok:
            logging.error(
                f"[TRAINING_API] Bad status {response.status_code}: {response.text}"
            )
    except Exception as e:
        logging.error(f"[TRAINING_API] Error sending event: {e}")


# ---------------------------------------------------------------------------
# Market data + position logic
# ---------------------------------------------------------------------------

def fetch_price(symbol: str) -> Optional[Decimal]:
    """
    Fetch latest price from Binance API with a user-agent header.

    Binance returns 451 errors if no User-Agent is present.
    """
    mapped = BINANCE_SYMBOL_MAP.get(symbol)
    if not mapped:
        logging.error(f"[PRICE] No Binance mapping for symbol={symbol}")
        return None

    url = f"https://api.binance.com/api/v3/ticker/price?symbol={mapped}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; Crypto-AI-Bot/1.0; +https://example.com)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()

        price_str = data.get("price")
        if price_str is None:
            logging.error(f"[PRICE] No 'price' field in response for {symbol}: {data}")
            return None

        price = Decimal(price_str)
        return price

    except Exception as e:
        logging.error(f"[PRICE] Error fetching {symbol} from Binance: {e}")
        return None


@dataclass
class Position:
    market: str              # "BTC-USD"
    entry_time: datetime
    entry_price: Decimal
    qty: Decimal
    take_profit_pct: Decimal
    stop_loss_pct: Decimal
    risk_mode: str

    def evaluate(self, current_price: Decimal) -> Tuple[bool, Decimal, Decimal]:
        """
        Check if TP or SL has been hit.

        Returns (should_close, pnl_usd, pnl_pct).
        """
        change = current_price - self.entry_price
        pnl_pct = change / self.entry_price
        pnl_usd = change * self.qty

        tp_hit = pnl_pct >= self.take_profit_pct
        sl_hit = pnl_pct <= -self.stop_loss_pct

        return tp_hit or sl_hit, pnl_usd, pnl_pct


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
    open_positions: Dict[str, Position] = {}

    # Log initial equity so the curve isn't empty forever
    now = datetime.now(timezone.utc)
    log_equity(now, equity)
    write_heartbeat(equity)

    logging.info("Crypto-AI-Bot starting up")
    logging.info(f"Starting equity: {equity}")
    cycle = 0

    while True:
        cycle += 1
        now = datetime.now(timezone.utc)
        logging.info("=" * 70)
        logging.info(f"[CYCLE {cycle}] Starting scan at {now.isoformat()}")

        # Heartbeat for this cycle
        write_heartbeat(equity)

        markets = pick_markets_to_scan()
        logging.info(f"[CYCLE {cycle}] Selected markets: {markets}")

        # 1. Evaluate existing positions for TP/SL exits
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
                    "trend_strength": "0",
                    "rsi": "0",
                    "volatility": "0",
                }

                log_trade_row(base_row)
                log_training_event_row(base_row)
                log_equity(exit_time, equity)

                try:
                    send_training_event_to_api(base_row)
                except Exception as e:
                    logging.error(f"[ERROR] Failed to send training event to API: {e}")

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

            # Position sizing by risk
            dollar_risk = equity * RISK_PER_TRADE_PCT
            if DEFAULT_STOP_LOSS_PCT > 0:
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

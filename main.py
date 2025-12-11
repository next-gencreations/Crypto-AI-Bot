# main.py  – Crypto AI Bot (paper trading)

import os
import time
import csv
import json
import random
import logging
from dataclasses import dataclass
from decimal import Decimal, getcontext
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional

import requests

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

getcontext().prec = 28  # high precision for Decimal math

DATA_DIR = "data"
TRADE_FILE = os.path.join(DATA_DIR, "trades.csv")
EQUITY_FILE = os.path.join(DATA_DIR, "equity.csv")
TRAINING_FILE = os.path.join(DATA_DIR, "training_events.csv")
HEARTBEAT_FILE = os.path.join(DATA_DIR, "heartbeat.txt")

# Starting balance and risk settings
START_BALANCE_USD = Decimal("1000")
RISK_PER_TRADE_PCT = Decimal("0.01")          # 1% of equity per trade
DEFAULT_TAKE_PROFIT_PCT = Decimal("0.03")     # +3%
DEFAULT_STOP_LOSS_PCT = Decimal("0.02")       # -2%
RISK_MODE = "fixed_1pct"

SLEEP_SECONDS = 60                            # pause between cycles
MAX_MARKETS_PER_SCAN = 3

# Markets we allow the bot to trade (paper only)
ALL_MARKETS: List[str] = [
    "BTC",
    "ETH",
    "SOL",
    "ADA",
    "LTC",
    "BCH",
]

# Map symbols to Binance tickers
BINANCE_SYMBOL_MAP = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "ADA": "ADAUSDT",
    "LTC": "LTCUSDT",
    "BCH": "BCHUSDT",
}

# Optional: API endpoint for training events (dashboard does not require this)
TRAINING_API_URL = os.environ.get("TRAINING_API_URL")  # e.g. https://your-api/track


# ---------------------------------------------------------------------------
# Logging / filesystem helpers
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def ensure_directories() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def init_csv_files_if_needed() -> None:
    # trades.csv
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

    # equity.csv
    if not os.path.exists(EQUITY_FILE):
        with open(EQUITY_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "equity_usd"])

    # training_events.csv (optional)
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


def write_heartbeat(ts: datetime) -> None:
    """
    Update the heartbeat file so the dashboard can see
    when the bot last ran.
    """
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
            f.write(ts.isoformat())
    except Exception as e:
        logging.error(f"[HEARTBEAT] Failed to write heartbeat: {e}")


def log_trade_row(row: Dict[str, Any]) -> None:
    with open(TRADE_FILE, "a", newline="", encoding="utf-8") as f:
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


def log_training_event_row(row: Dict[str, Any]) -> None:
    with open(TRAINING_FILE, "a", newline="", encoding="utf-8") as f:
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


def log_equity(ts: datetime, equity: Decimal) -> None:
    with open(EQUITY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([ts.isoformat(), str(equity)])


def send_training_event_to_api(row: Dict[str, Any]) -> None:
    """
    Optional: send a training event to an external API.
    Dashboard does not depend on this – failures are logged only.
    """
    if not TRAINING_API_URL:
        return

    try:
        payload = json.dumps(row)
        resp = requests.post(
            TRAINING_API_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        if resp.status_code >= 300:
            logging.error(
                f"[TRAINING_API] Bad status {resp.status_code}: {resp.text[:200]}"
            )
    except Exception as e:
        logging.error(f"[TRAINING_API] Error sending event: {e}")


# ---------------------------------------------------------------------------
# Market data + position logic
# ---------------------------------------------------------------------------

def fetch_price(symbol: str) -> Optional[Decimal]:
    """
    Fetch the latest price from Binance (USDT pairs).
    Returns a Decimal or None on failure.
    """
    ticker = BINANCE_SYMBOL_MAP.get(symbol.upper())
    if not ticker:
        logging.error(f"[PRICE] No Binance mapping for symbol {symbol}")
        return None

    url = f"https://api.binance.com/api/v3/ticker/price?symbol={ticker}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        price = Decimal(data["price"])
        return price
    except Exception as e:
        logging.error(f"[PRICE] Failed to fetch {symbol} price: {e}")
        return None


@dataclass
class Position:
    market: str               # "BTC-USD"
    entry_time: datetime
    entry_price: Decimal
    qty: Decimal
    take_profit_pct: Decimal
    stop_loss_pct: Decimal
    risk_mode: str

    def evaluate(self, current_price: Decimal) -> Tuple[bool, Decimal, Decimal]:
        """
        Check if TP or SL has been hit.

        Returns (should_close, pnl_usd, pnl_pct)
        """
        change = current_price - self.entry_price
        pnl_pct = change / self.entry_price
        pnl_usd = change * self.qty

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

    equity = START_BALANCE_USD
    open_positions: Dict[str, Position] = {}  # key: "BTC-USD", etc.

    logging.info("Crypto-AI-Bot starting up")
    logging.info(f"Starting equity: {equity}")

    # Log initial equity so the curve has a starting point
    now = datetime.now(timezone.utc)
    log_equity(now, equity)
    write_heartbeat(now)

    cycle = 0

    while True:
        cycle += 1
        now = datetime.now(timezone.utc)

        # heartbeat each cycle so dashboard can see status
        write_heartbeat(now)

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

                # Send to external training API (best effort, optional)
                try:
                    send_training_event_to_api(base_row)
                except Exception as e:
                    logging.error(f"[ERROR] Failed to send training event to API: {e}")

                # Remove from open positions
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

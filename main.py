import os
import time
import json
import csv
import random
import signal
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

import requests


# =========================================================
# CONFIG (Render env vars)
# =========================================================

# Where to persist files (Render Disk mount path)
DATA_DIR = os.getenv("DATA_DIR", "/opt/render/project/src/data").rstrip("/")
STATE_FILE = os.path.join(DATA_DIR, "bot_state.json")
TRADES_FILE = os.path.join(DATA_DIR, "trades.csv")
EQUITY_FILE = os.path.join(DATA_DIR, "equity_curve.csv")
HEARTBEAT_FILE = os.path.join(DATA_DIR, "heartbeat.json")

# Your API (crypto-ai-api) ingestion base URL
TRAINING_API_URL = os.getenv("TRAINING_API_URL", "").rstrip("/")
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")

# Bot loop
CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS", "60"))
MAX_MARKETS_PER_SCAN = int(os.getenv("MAX_MARKETS_PER_SCAN", "3"))

# Paper trading
STARTING_EQUITY = float(os.getenv("STARTING_EQUITY", "1000"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "5.0"))  # position size % of equity
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "3.0"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "2.0"))

# Keep some randomness so we get a healthy dataset over weeks
OPEN_PROBABILITY = float(os.getenv("OPEN_PROBABILITY", "0.25"))  # chance to open if not open
MIN_PNL_USD_TO_CLOSE = float(os.getenv("MIN_PNL_USD_TO_CLOSE", "1.00"))

# Markets (use your internal "XXX-USD" naming)
ALL_MARKETS = os.getenv("ALL_MARKETS", "BTC-USD,ETH-USD,SOL-USD,ADA-USD,LTC-USD,BCH-USD")
MARKETS = [m.strip().upper() for m in ALL_MARKETS.split(",") if m.strip()]

# Price feed mapping for Coinbase spot endpoint
# Coinbase uses symbols like BTC-USD, ETH-USD, etc — so this is direct.
COINBASE_PRICE_URL = "https://api.coinbase.com/v2/prices/{pair}/spot"

# Timeouts & retries
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "10"))
PRICE_FAIL_SLEEP = int(os.getenv("PRICE_FAIL_SLEEP", "2"))   # short pause on price failure
MAX_CONSEC_PRICE_FAILS = int(os.getenv("MAX_CONSEC_PRICE_FAILS", "10"))

# =========================================================
# LOGGING
# =========================================================

def setup_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

setup_logging()


# =========================================================
# UTILS
# =========================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def safe_write_json(path: str, payload: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def safe_append_csv(path: str, fieldnames: List[str], row: dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def safe_post(path: str, payload: dict):
    """Never crash the bot if your API is down or redeploying."""
    if not TRAINING_API_URL:
        return
    try:
        headers = {}
        if INGEST_TOKEN:
            headers["X-INGEST-TOKEN"] = INGEST_TOKEN
        requests.post(
            f"{TRAINING_API_URL}{path}",
            json=payload,
            timeout=HTTP_TIMEOUT,
            headers=headers
        )
    except Exception:
        # silent fail to keep bot stable for weeks
        return


# =========================================================
# STATE (persisted)
# =========================================================

DEFAULT_STATE = {
    "equity_usd": STARTING_EQUITY,
    "open_positions": {},   # market -> position
    "closed_trades_count": 0,
    "last_saved_utc": None
}

state: Dict[str, Any] = {}
shutdown_requested = False


def load_state():
    global state
    ensure_data_dir()

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            # basic validation / defaults
            for k, v in DEFAULT_STATE.items():
                if k not in state:
                    state[k] = v
            if "open_positions" not in state or not isinstance(state["open_positions"], dict):
                state["open_positions"] = {}
            if "equity_usd" not in state:
                state["equity_usd"] = STARTING_EQUITY
            logging.info(f"Loaded state from disk. Equity={state['equity_usd']}, Open={list(state['open_positions'].keys())}")
            return
        except Exception as e:
            logging.error(f"Failed to load state, starting fresh: {e}")

    state = DEFAULT_STATE.copy()
    save_state()
    logging.info(f"Initialized new state. Equity={state['equity_usd']}")


def save_state():
    state["last_saved_utc"] = utc_now_iso()
    safe_write_json(STATE_FILE, state)


# =========================================================
# HEARTBEAT / EQUITY / EVENTS
# =========================================================

def write_heartbeat_local(status: str, note: str = ""):
    hb = {
        "time_utc": utc_now_iso(),
        "status": status,
        "note": note,
        "equity_usd": round(float(state["equity_usd"]), 2),
        "open_positions": list(state["open_positions"].keys())
    }
    safe_write_json(HEARTBEAT_FILE, hb)


def send_heartbeat():
    payload = {
        "time_utc": utc_now_iso(),
        "status": "running",
        "open_positions": len(state["open_positions"]),
        "equity_usd": round(float(state["equity_usd"]), 2)
    }
    safe_post("/ingest/heartbeat", payload)
    write_heartbeat_local("running")


def send_equity_point():
    payload = {
        "time_utc": utc_now_iso(),
        "equity_usd": round(float(state["equity_usd"]), 2)
    }
    safe_post("/ingest/equity", payload)
    safe_append_csv(EQUITY_FILE, ["time_utc", "equity_usd"], payload)


def send_training_event(event_type: str, market: str, payload_obj: dict):
    safe_post("/ingest/training_event", {
        "time_utc": utc_now_iso(),
        "event_type": event_type,
        "market": market,
        "payload_json": payload_obj
    })


def send_trade(trade: dict):
    safe_post("/ingest/trade", trade)
    # also persist locally for safety
    safe_append_csv(
        TRADES_FILE,
        ["time_utc", "market", "side", "entry_price", "exit_price", "qty", "pnl_usd", "pnl_pct", "reason"],
        trade
    )


# =========================================================
# PRICE FETCH (Coinbase public spot)
# =========================================================

def fetch_price_coinbase(pair: str) -> Optional[float]:
    """
    Coinbase public endpoint:
      GET https://api.coinbase.com/v2/prices/BTC-USD/spot
    Returns JSON with data.amount (string).
    """
    url = COINBASE_PRICE_URL.format(pair=pair)
    headers = {"User-Agent": "Crypto-AI-Bot/1.0 (+paper-trading)"}
    resp = requests.get(url, timeout=HTTP_TIMEOUT, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    amount = data.get("data", {}).get("amount")
    if not amount:
        return None
    return float(amount)


def fetch_price(pair: str) -> Optional[float]:
    try:
        return fetch_price_coinbase(pair)
    except Exception as e:
        logging.warning(f"[PRICE] Could not fetch {pair}: {e}")
        return None


# =========================================================
# TRADING (paper trading)
# =========================================================

def position_size_qty(price: float) -> float:
    equity = float(state["equity_usd"])
    usd_alloc = equity * (RISK_PER_TRADE_PCT / 100.0)
    if usd_alloc <= 0 or price <= 0:
        return 0.0
    return usd_alloc / price


def open_position(market: str, price: float):
    if market in state["open_positions"]:
        return

    qty = position_size_qty(price)
    if qty <= 0:
        return

    pos = {
        "market": market,
        "side": "LONG",
        "entry_price": float(price),
        "qty": float(qty),
        "opened_utc": utc_now_iso(),
        "take_profit_pct": float(TAKE_PROFIT_PCT),
        "stop_loss_pct": float(STOP_LOSS_PCT),
    }
    state["open_positions"][market] = pos

    logging.info(f"[ENTRY] Opened LONG {market} @ {price:.6f} qty={qty:.6f} TP={TAKE_PROFIT_PCT}% SL={STOP_LOSS_PCT}%")
    send_training_event("open_position", market, {"price": price, "qty": qty})


def close_position(market: str, exit_price: float, reason: str):
    pos = state["open_positions"].get(market)
    if not pos:
        return

    entry = float(pos["entry_price"])
    qty = float(pos["qty"])
    pnl_usd = (float(exit_price) - entry) * qty
    pnl_pct = 0.0
    if entry > 0:
        pnl_pct = ((float(exit_price) - entry) / entry) * 100.0

    # update equity
    state["equity_usd"] = float(state["equity_usd"]) + pnl_usd

    trade = {
        "time_utc": utc_now_iso(),
        "market": market,
        "side": pos["side"],
        "entry_price": round(entry, 8),
        "exit_price": round(float(exit_price), 8),
        "qty": round(qty, 8),
        "pnl_usd": round(pnl_usd, 2),
        "pnl_pct": round(pnl_pct, 4),
        "reason": reason
    }

    # remove position
    del state["open_positions"][market]
    state["closed_trades_count"] = int(state.get("closed_trades_count", 0)) + 1

    logging.info(f"[EXIT] Closed {market} @ {exit_price:.6f} pnl=${trade['pnl_usd']} ({trade['pnl_pct']}%) reason={reason}")
    send_trade(trade)
    send_training_event("close_position", market, {"pnl_usd": pnl_usd, "reason": reason})


def evaluate_position(market: str, current_price: float):
    pos = state["open_positions"].get(market)
    if not pos:
        return

    entry = float(pos["entry_price"])
    tp = float(pos["take_profit_pct"])
    sl = float(pos["stop_loss_pct"])

    if entry <= 0:
        return

    move_pct = ((float(current_price) - entry) / entry) * 100.0
    pnl_usd = (float(current_price) - entry) * float(pos["qty"])

    # take profit / stop loss
    if move_pct >= tp:
        close_position(market, current_price, "take_profit")
        return
    if move_pct <= -sl:
        close_position(market, current_price, "stop_loss")
        return

    # optional: allow occasional “neutral” exits to generate variety of data
    if abs(pnl_usd) >= MIN_PNL_USD_TO_CLOSE and random.random() < 0.05:
        close_position(market, current_price, "time_exit")


# =========================================================
# MAIN LOOP
# =========================================================

def pick_markets() -> List[str]:
    if MAX_MARKETS_PER_SCAN >= len(MARKETS):
        return MARKETS[:]
    return random.sample(MARKETS, k=max(1, MAX_MARKETS_PER_SCAN))


def handle_shutdown(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    logging.warning("Shutdown requested. Saving state and exiting soon...")


def run():
    global shutdown_requested

    ensure_data_dir()
    load_state()

    # write initial heartbeat so dashboard sees something quickly
    write_heartbeat_local("starting")

    logging.info("Crypto-AI-Bot running (paper trading, Coinbase prices)")
    logging.info(f"DATA_DIR={DATA_DIR}")
    logging.info(f"Markets={MARKETS}")
    logging.info(f"Equity=${state['equity_usd']}")
    logging.info(f"API={TRAINING_API_URL if TRAINING_API_URL else '(disabled)'}")

    consecutive_price_fails = 0
    last_save_ts = time.time()
    last_equity_ts = 0.0

    while not shutdown_requested:
        cycle_time = utc_now_iso()
        selected = pick_markets()

        logging.info(f"[CYCLE] {cycle_time} selected={selected} open={list(state['open_positions'].keys())}")

        # heartbeat every cycle
        send_heartbeat()

        for market in selected:
            price = fetch_price(market)
            if price is None:
                consecutive_price_fails += 1
                if consecutive_price_fails >= MAX_CONSEC_PRICE_FAILS:
                    # backoff if feed is flaky
                    logging.warning(f"[PRICE] Too many failures ({consecutive_price_fails}). Backing off...")
                    write_heartbeat_local("degraded", "price_feed_errors")
                    time.sleep(min(60, PRICE_FAIL_SLEEP * 5))
                    consecutive_price_fails = 0
                else:
                    time.sleep(PRICE_FAIL_SLEEP)
                continue

            consecutive_price_fails = 0

            # evaluate existing positions on this market
            evaluate_position(market, price)

            # maybe open new position
            if market not in state["open_positions"]:
                if random.random() < OPEN_PROBABILITY:
                    open_position(market, price)

        # equity point every cycle (or you can reduce later)
        now_ts = time.time()
        if now_ts - last_equity_ts >= CYCLE_SECONDS:
            send_equity_point()
            last_equity_ts = now_ts

        # save state periodically even if nothing happens
        if now_ts - last_save_ts >= 30:
            save_state()
            last_save_ts = now_ts

        time.sleep(CYCLE_SECONDS)

    # graceful exit
    write_heartbeat_local("stopping")
    save_state()
    logging.info("Exited cleanly.")


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    run()

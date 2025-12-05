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
SLEEP_SECONDS = 6 * 60          # 6 minutes
CANDLE_GRANULARITY = 300        # 5-minute candles
LOOKBACK_CANDLES = 100

PUBLIC_API_BASE = "https://api.exchange.coinbase.com"

# Risk parameters by mode
if RISK_MODE == "AGGRESSIVE":
    TAKE_PROFIT_PCT = Decimal("0.015")      # +1.5%
    STOP_LOSS_PCT = Decimal("0.02")         # -2.0%
    POSITION_SIZE_FRACTION = Decimal("0.4")  # 40% of USD per new trade
    MAX_DAILY_DRAWDOWN = Decimal("0.08")    # 8%
    MAX_OPEN_POSITIONS = 5
elif RISK_MODE == "NORMAL":
    TAKE_PROFIT_PCT = Decimal("0.012")      # +1.2%
    STOP_LOSS_PCT = Decimal("0.018")        # -1.8%
    POSITION_SIZE_FRACTION = Decimal("0.35")  # 35%
    MAX_DAILY_DRAWDOWN = Decimal("0.06")    # 6%
    MAX_OPEN_POSITIONS = 4
else:  # SAFE (default)
    TAKE_PROFIT_PCT = Decimal("0.010")      # +1.0%
    STOP_LOSS_PCT = Decimal("0.015")        # -1.5%
    POSITION_SIZE_FRACTION = Decimal("0.30")  # 30%
    MAX_DAILY_DRAWDOWN = Decimal("0.05")    # 5%
    MAX_OPEN_POSITIONS = 3

# Indicator thresholds
MIN_TREND_STRENGTH = Decimal("0.002")  # short MA must be > long MA by 0.2%
RSI_BUY_MIN = Decimal("40")
RSI_BUY_MAX = Decimal("65")
MIN_VOLATILITY = Decimal("0.002")      # 0.2% avg candle move
MAX_VOLATILITY = Decimal("0.030")      # 3% avg candle move

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
market_stats: Dict[str, Dict[str, int]] = {
    m: {"wins": 0, "losses": 0} for m in ALL_MARKETS
}

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

    closes: List[Decimal] = [c["close"] for c in candles]

    short_ma = sma(closes, 9)
    long_ma = sma(closes, 21)
    current_rsi = rsi(closes, 14)
    vol = estimate_volatility(closes[-20:])

    if short_ma is None or long_ma is None or current_rsi is None or vol is None:
        return Decimal("-999"), None, closes, None

    # Basic filters
    if short_ma <= long_ma * (Decimal("1") + MIN_TREND_STRENGTH):
        return Decimal("-999"), None, closes, None

    if not (RSI_BUY_MIN <= current_rsi <= RSI_BUY_MAX):
        return Decimal("-999"), None, closes, None

    if not (MIN_VOLATILITY <= vol <= MAX_VOLATILITY):
        return Decimal("-999"), None, closes, None

    price = closes[-1]

    # Base score from indicators
    trend_strength = (short_ma - long_ma) / long_ma
    rsi_centered = (Decimal("55") - abs(current_rsi - Decimal("55"))) / Decimal("55")  # best around 55
    vol_factor = (vol - MIN_VOLATILITY) / (MAX_VOLATILITY - MIN_VOLATILITY)

    base_score = trend_strength * Decimal("2.0") + rsi_centered + vol_factor

    # Online-learning tweak: reward markets with higher win-rate
    stats = market_stats.get(market, {"wins": 0, "losses": 0})
    wins = stats["wins"]
    losses = stats["losses"]
    total = wins + losses
    if total > 0:
        win_rate = Decimal(wins) / Decimal(total)
        learn_bonus = (win_rate - Decimal("0.5")) * Decimal("0.5")  # +/- 0.25
    else:
        learn_bonus = Decimal("0")

    score = base_score + learn_bonus

    features = {
        "trend_strength": trend_strength,
        "rsi": current_rsi,
        "volatility": vol,
    }

    return score, price, closes, features


def get_random_markets_to_scan() -> List[str]:
    n = min(MAX_MARKETS_PER_SCAN, len(ALL_MARKETS))
    return random.sample(ALL_MARKETS, k=n)


# ---------------------------------------------------------------------------
# Equity & risk helpers
# ---------------------------------------------------------------------------

def compute_equity() -> Decimal:
    """USD balance + market value of open positions."""
    total = usd_balance
    for pos in open_positions:
        price = get_latest_price(pos["market"])
        if price is None:
            continue
        total += pos["size"] * price
    return total


def update_daily_state() -> Tuple[Decimal, Decimal, Decimal]:
    """
    Update daily equity / drawdown tracking.
    Returns (equity, drawdown_pct, paused_flag_as_decimal)
    """
    global equity_peak_today, today, trading_paused_for_today

    now = datetime.now(timezone.utc)
    current_day = now.date()

    if current_day != today:
        # New day: reset
        today = current_day
        equity_peak_today = compute_equity()
        trading_paused_for_today = False
        log("New trading day: resetting daily drawdown and pause flags.")

    equity = compute_equity()
    if equity > equity_peak_today:
        equity_peak_today = equity

    # Drawdown relative to today's peak
    if equity_peak_today > 0:
        dd_pct = (equity_peak_today - equity) / equity_peak_today
    else:
        dd_pct = Decimal("0")

    if dd_pct >= MAX_DAILY_DRAWDOWN:
        trading_paused_for_today = True

    return equity, dd_pct, Decimal("1") if trading_paused_for_today else Decimal("0")


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------

def open_long_position(
    market: str,
    price: Decimal,
    features: Optional[Dict[str, Decimal]] = None,
) -> None:
    """Open a new long position if we have room and enough USD."""
    global usd_balance

    if len(open_positions) >= MAX_OPEN_POSITIONS:
        return

    # Fixed fraction of current USD balance
    position_usd = usd_balance * POSITION_SIZE_FRACTION
    if position_usd < Decimal("10"):
        return  # don't bother with tiny trades

    size = (position_usd / price).quantize(Decimal("0.00000001"))
    if size <= 0:
        return

    usd_balance -= position_usd

    take_profit = price * (Decimal("1") + TAKE_PROFIT_PCT)
    stop_loss = price * (Decimal("1") - STOP_LOSS_PCT)

    pos = {
        "market": market,
        "size": size,
        "entry_price": price,
        "entry_time": datetime.now(timezone.utc),
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "features": features or {},
    }
    open_positions.append(pos)

    log(
        f"OPEN LONG {market} size={size} entry={price:.2f} "
        f"TP={take_profit:.2f} SL={stop_loss:.2f} "
        f"usd_balance_after={usd_balance:.2f}"
    )


def close_position(pos: Dict, price: Decimal) -> None:
    """Close an existing long position at the given price."""
    global usd_balance

    market = pos["market"]
    size: Decimal = pos["size"]
    entry_price: Decimal = pos["entry_price"]

    # PnL in USD
    usd_change = (price - entry_price) * size
    usd_balance += size * price
    profit_usd = usd_change

    # Update learning stats
    stats = market_stats.setdefault(market, {"wins": 0, "losses": 0})
    if profit_usd > 0:
        stats["wins"] += 1
    else:
        stats["losses"] += 1

    equity_after = compute_equity()
    side = "LONG"

    log(
        f"CLOSE {side} {market} size={size} entry={entry_price:.2f} exit={price:.2f} "
        f"profit={profit_usd:.2f} equity_after={equity_after:.2f} "
        f"wins={stats['wins']} losses={stats['losses']}"
    )

    now = datetime.now(timezone.utc)
    log_trade_csv(
        timestamp=now,
        market=market,
        side=side,
        size=size,
        entry_price=entry_price,
        exit_price=price,
        profit_usd=profit_usd,
        equity_after=equity_after,
    )

    # Extra learning log with indicators at entry
    try:
        log_training_event_csv(
            entry_time=pos["entry_time"],
            exit_time=now,
            market=market,
            features=pos.get("features") or {},
            entry_price=entry_price,
            exit_price=price,
            profit_usd=profit_usd,
        )
    except Exception as e:
        log(f"WARNING: failed to log training event: {e}")


def manage_open_positions() -> None:
    """Check TP/SL for all open positions and close if hit."""
    global open_positions

    remaining: List[Dict] = []

    for pos in open_positions:
        market = pos["market"]
        price = get_latest_price(market)
        if price is None:
            remaining.append(pos)
            continue

        hit_tp = price >= pos["take_profit"]
        hit_sl = price <= pos["stop_loss"]

        if hit_tp or hit_sl:
            close_position(pos, price)
        else:
            remaining.append(pos)

    open_positions = remaining


# ---------------------------------------------------------------------------
# Main trading logic
# ---------------------------------------------------------------------------

def scan_and_open_new_positions(equity: Decimal, dd_pct: Decimal) -> None:
    """Scan random markets, score them, and open new positions if allowed."""
    if trading_paused_for_today:
        log("Daily drawdown limit hit. Pausing new entries for the rest of the day.")
        return

    if len(open_positions) >= MAX_OPEN_POSITIONS:
        log("Max open positions reached; skipping new entries this cycle.")
        return

    markets_to_scan = get_random_markets_to_scan()
    log(f"Scanning {len(markets_to_scan)} random markets this cycle...")

    best_score = Decimal("-999")
    best_market: Optional[str] = None
    best_price: Optional[Decimal] = None
    best_features: Optional[Dict[str, Decimal]] = None

    for m in markets_to_scan:
        # Skip if we already have a position in this market
        if any(pos["market"] == m for pos in open_positions):
            continue

        score, price, _, feats = get_market_score(m)
        log(f"Market {m} score {score:.4f}")
        if price is None:
            continue

        if score > best_score:
            best_score = score
            best_market = m
            best_price = price
            best_features = feats

    if best_market is None or best_price is None or best_score <= Decimal("-998"):
        log("No suitable new market found this cycle.")
        return

    # Open a long position in the best market
    open_long_position(best_market, best_price, best_features)


def main_loop() -> None:
    global usd_balance

    log("=" * 70)
    log("CRYPTO-AI-BOT PAPER TRADING")
    log("NO REAL MONEY. NO REAL ORDERS. PUBLIC DATA ONLY.")
    log(f"Starting balance: ${START_BALANCE_USD}")
    log(f"Risk mode: {RISK_MODE}")
    log("=" * 70)

    while True:
        try:
            # 1) Update equity & daily state
            equity, dd_pct, paused_flag = update_daily_state()

            # 2) Manage existing positions (TP/SL)
            manage_open_positions()

            # 3) Try to open new positions if allowed
            scan_and_open_new_positions(equity, dd_pct)

            # 4) Log equity curve
            now = datetime.now(timezone.utc)
            open_count = len(open_positions)
            log(
                f"Summary: USD=${usd_balance:.2f}, positions={open_count}, "
                f"Equity=${equity:.2f}, DD={dd_pct * 100:.2f}%, "
                f"PausedToday={'YES' if trading_paused_for_today else 'NO'}"
            )
            log_equity_csv(
                timestamp=now,
                equity=equity,
                usd_balance_=usd_balance,
                open_positions_count=open_count,
                drawdown_pct=dd_pct * Decimal("100"),
            )

        except Exception as e:
            log(f"FATAL error in main loop: {e}")

        # 5) Sleep until next cycle
        log(f"Sleeping for {SLEEP_SECONDS} seconds...")
        time.sleep(SLEEP_SECONDS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ensure_directories()
    log("Starting Crypto-AI-Bot...")
    main_loop()

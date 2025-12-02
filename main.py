from decimal import Decimal, getcontext
from datetime import datetime, timezone, date
import time
import csv
import os
import random
import requests

# Higher precision for crypto maths
getcontext().prec = 28

# ============= CONFIG =============

# Starting balance (can be overridden by Render env var START_BALANCE_USD)
def load_start_balance() -> Decimal:
    raw = os.getenv("START_BALANCE_USD", "100").strip()
    try:
        return Decimal(raw)
    except Exception:
        print(f"[WARN] Invalid START_BALANCE_USD='{raw}', using 100 instead.")
        return Decimal("100")

START_BALANCE_USD = load_start_balance()

# Big universe of possible USD markets to sample from
ALL_MARKETS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "ADA-USD",
    "LTC-USD", "DOGE-USD", "LINK-USD", "MATIC-USD", "OP-USD",
    "ARB-USD", "ATOM-USD", "SAND-USD", "UNI-USD", "RNDR-USD",
]

# How many random markets to scan each cycle
MAX_MARKETS_PER_SCAN = 8

SLEEP_SECONDS = 6 * 60          # 6 minutes
CANDLE_GRANULARITY = 300        # 5-minute candles
LOOKBACK_CANDLES = 100

# --- Risk / trade parameters (AGGRESSIVE, but not insane) ---
TAKE_PROFIT_PCT = Decimal("0.012")        # +1.2% target
STOP_LOSS_PCT   = Decimal("0.008")        # -0.8% stop
POSITION_SIZE_FRACTION = Decimal("0.6")   # use up to 60% of free USD per new position
MAX_CONCURRENT_POSITIONS = 3              # hold up to 3 simultaneous coins

# Market filters
MIN_TREND_STRENGTH = Decimal("0.002")     # short MA must be > long MA by 0.2%
RSI_BUY_MIN = Decimal("40")
RSI_BUY_MAX = Decimal("65")

MIN_VOLATILITY = Decimal("0.002")         # 0.2% avg move per candle
MAX_VOLATILITY = Decimal("0.030")         # 3% avg move per candle

# Risk controls
MAX_DAILY_DRAWDOWN = Decimal("0.08")      # 8% daily drawdown limit
MAX_LOSING_STREAK  = 4                    # after 4 losses in a row, pause new entries

PUBLIC_API_BASE = "https://api.exchange.coinbase.com"
TRADE_LOG = "live_sim_trade_history.csv"

# ============= STATE =============

usd_balance: Decimal = START_BALANCE_USD

# each position: dict(market, entry_price, size, tp, sl, opened_at, features)
positions = []

equity_peak_today: Decimal = START_BALANCE_USD
today: date = date.today()
losing_streak: int = 0
trade_count: int = 0

trading_paused_for_today: bool = False

# Simple online-learning "AI" weights
# These will slowly update as trades win/lose.
AI_WEIGHTS = {
    "trend": Decimal("0.8"),
    "rsi":   Decimal("0.6"),
    "vol":   Decimal("0.4"),
    "bias":  Decimal("0.0"),
}
AI_LEARNING_RATE = Decimal("0.05")

# Ensure trade log exists
if not os.path.exists(TRADE_LOG):
    with open(TRADE_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "market", "side", "entry_price", "exit_price",
            "size", "pnl_usd", "pnl_pct", "usd_balance_after",
            "equity_after", "win", "trend_feat", "rsi_feat", "vol_feat"
        ])


# ============= HELPERS =============

def log(msg: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} | {msg}", flush=True)


def get_random_scan_list():
    """Pick a random subset of ALL_MARKETS to scan this cycle."""
    n = min(MAX_MARKETS_PER_SCAN, len(ALL_MARKETS))
    return random.sample(ALL_MARKETS, k=n)


def get_latest_price(market: str) -> Decimal | None:
    url = f"{PUBLIC_API_BASE}/products/{market}/ticker"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        return Decimal(str(data["price"]))
    except Exception as e:
        log(f"[WARN] Failed to fetch latest price for {market}: {e}")
        return None


def get_recent_candles(market: str, limit: int = LOOKBACK_CANDLES):
    url = f"{PUBLIC_API_BASE}/products/{market}/candles"
    end_time = int(time.time())
    start_time = end_time - limit * CANDLE_GRANULARITY
    params = {
        "start": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
        "end": datetime.fromtimestamp(end_time, tz=timezone.utc).isoformat(),
        "granularity": CANDLE_GRANULARITY,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        candles = r.json()
        # [time, low, high, open, close, volume] – newest first
        closes = [Decimal(str(c[4])) for c in candles]
        closes.reverse()
        return closes
    except Exception as e:
        log(f"[WARN] Failed to fetch candles for {market}: {e}")
        return None


def sma(values, period: int) -> Decimal | None:
    if not values or len(values) < period:
        return None
    return sum(values[-period:]) / Decimal(period)


def rsi(values, period: int = 14) -> Decimal | None:
    if not values or len(values) <= period:
        return None
    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = values[-i] - values[-i - 1]
        if diff > 0:
            gains.append(diff)
        else:
            losses.append(-diff)
    if not gains and not losses:
        return Decimal(50)
    avg_gain = sum(gains) / Decimal(period) if gains else Decimal(0)
    avg_loss = sum(losses) / Decimal(period) if losses else Decimal(0)
    if avg_loss == 0:
        return Decimal(100)
    rs = avg_gain / avg_loss
    return Decimal(100) - (Decimal(100) / (Decimal(1) + rs))


def volatility(values) -> Decimal | None:
    if not values or len(values) < 2:
        return None
    moves = []
    for i in range(1, len(values)):
        if values[i - 1] == 0:
            continue
        move = (values[i] - values[i - 1]) / values[i - 1]
        moves.append(abs(move))
    if not moves:
        return None
    return sum(moves) / Decimal(len(moves))


# ============= SIMPLE "AI" MODEL =============

def ai_predict(trend_feat: Decimal, rsi_feat: Decimal, vol_feat: Decimal) -> Decimal:
    """
    Very simple logistic regression style score using our weights.
    Output ~ probability of a good trade (0..1).
    """
    # Convert to float for math.exp, then back to Decimal
    import math

    w_t = float(AI_WEIGHTS["trend"])
    w_r = float(AI_WEIGHTS["rsi"])
    w_v = float(AI_WEIGHTS["vol"])
    b   = float(AI_WEIGHTS["bias"])

    x_t = float(trend_feat)
    x_r = float(rsi_feat)
    x_v = float(vol_feat)

    z = w_t * x_t + w_r * x_r + w_v * x_v + b
    prob = 1.0 / (1.0 + math.exp(-z))
    return Decimal(str(prob))


def ai_learn(trend_feat: Decimal, rsi_feat: Decimal, vol_feat: Decimal, win: bool):
    """
    Online learning update: if the trade was a win, push probability up;
    if a loss, push it down.
    """
    target = Decimal("1.0") if win else Decimal("0.0")
    pred = ai_predict(trend_feat, rsi_feat, vol_feat)
    error = target - pred

    # Gradient-style update
    AI_WEIGHTS["trend"] += AI_LEARNING_RATE * error * trend_feat
    AI_WEIGHTS["rsi"]   += AI_LEARNING_RATE * error * rsi_feat
    AI_WEIGHTS["vol"]   += AI_LEARNING_RATE * error * vol_feat
    AI_WEIGHTS["bias"]  += AI_LEARNING_RATE * error

    log(f"[AI] Updated weights: trend={AI_WEIGHTS['trend']:.4f}, "
        f"rsi={AI_WEIGHTS['rsi']:.4f}, vol={AI_WEIGHTS['vol']:.4f}, "
        f"bias={AI_WEIGHTS['bias']:.4f}")


def build_features(short_ma: Decimal, long_ma: Decimal,
                   rsi_val: Decimal, vol_val: Decimal):
    """
    Normalise features into ~0..1 ranges for the AI.
    """
    # Trend: percentage difference
    trend_feat = (short_ma - long_ma) / long_ma if long_ma != 0 else Decimal(0)

    # RSI: best when near 50, scale 0..1
    rsi_diff = abs(rsi_val - Decimal(50))
    rsi_feat = Decimal(1) - (rsi_diff / Decimal(50))  # 1 when rsi=50, 0 when rsi=0 or 100

    # Volatility: map MIN_VOLATILITY..MAX_VOLATILITY into 0..1
    if vol_val <= MIN_VOLATILITY:
        vol_feat = Decimal(0)
    elif vol_val >= MAX_VOLATILITY:
        vol_feat = Decimal(1)
    else:
        vol_feat = (vol_val - MIN_VOLATILITY) / (MAX_VOLATILITY - MIN_VOLATILITY)

    return trend_feat, rsi_feat, vol_feat


# ============= MARKET SCORING =============

def score_market(market: str):
    """
    Analyse a market and return:
    (score, last_price, closes, (trend_feat, rsi_feat, vol_feat))
    """
    closes = get_recent_candles(market)
    if not closes:
        return Decimal("-999"), None, None, None

    short_ma = sma(closes, 9)
    long_ma  = sma(closes, 21)
    rsi_val  = rsi(closes, 14)
    vol_val  = volatility(closes)

    if any(v is None for v in (short_ma, long_ma, rsi_val, vol_val)):
        return Decimal("-999"), None, None, None

    trend_strength = (short_ma - long_ma) / long_ma if long_ma != 0 else Decimal(0)

    # Basic filters
    if trend_strength < MIN_TREND_STRENGTH:
        return Decimal("-999"), None, None, None
    if not (RSI_BUY_MIN <= rsi_val <= RSI_BUY_MAX):
        return Decimal("-999"), None, None, None
    if not (MIN_VOLATILITY <= vol_val <= MAX_VOLATILITY):
        return Decimal("-999"), None, None, None

    last_price = closes[-1]

    # Build AI features & prediction
    trend_feat, rsi_feat, vol_feat = build_features(short_ma, long_ma, rsi_val, vol_val)
    ai_score = ai_predict(trend_feat, rsi_feat, vol_feat)

    # Overall score: combine pure TA + AI view
    base_score = trend_strength * Decimal("100")   # emphasise trend
    score = base_score + ai_score * Decimal("10")

    return score, last_price, closes, (trend_feat, rsi_feat, vol_feat)


def choose_best_market():
    """
    Randomly sample some markets from ALL_MARKETS, score them, and return the best one.
    """
    best_score = Decimal("-999")
    best = None

    scan_list = get_random_scan_list()
    log(f"Scanning {len(scan_list)} random markets this cycle...")

    for m in scan_list:
        score, price, closes, feats = score_market(m)
        if price is None or feats is None:
            log(f"Market {m} score {score:.4f} (filtered out)")
            continue
        log(f"Market {m} score {score:.4f}")
        if score > best_score:
            best_score = score
            best = (m, score, price, feats)

    return best  # (market, score, price, feats) or None


# ============= PORTFOLIO & TRADING =============

def portfolio_equity(prices: dict[str, Decimal]) -> Decimal:
    total = usd_balance
    for pos in positions:
        m = pos["market"]
        p = prices.get(m)
        if p is None:
            p = pos["entry_price"]
        total += pos["size"] * p
    return total


def log_trade(timestamp: str, market: str, side: str,
              entry_price: Decimal, exit_price: Decimal,
              size: Decimal, pnl_usd: Decimal, pnl_pct: Decimal,
              equity_after: Decimal, win: bool,
              trend_feat: Decimal, rsi_feat: Decimal, vol_feat: Decimal):
    with open(TRADE_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, market, side, f"{entry_price:.8f}", f"{exit_price:.8f}",
            f"{size:.8f}", f"{pnl_usd:.2f}", f"{pnl_pct:.4f}",
            f"{usd_balance:.2f}", f"{equity_after:.2f}",
            int(win),
            f"{trend_feat:.6f}", f"{rsi_feat:.6f}", f"{vol_feat:.6f}",
        ])


def maybe_open_new_position(prices: dict[str, Decimal]):
    global usd_balance, trade_count

    if trading_paused_for_today:
        return

    if len(positions) >= MAX_CONCURRENT_POSITIONS:
        return

    candidate = choose_best_market()
    if not candidate:
        log("No suitable market found this cycle.")
        return

    market, score, price, feats = candidate

    # Use a portion of free cash
    if usd_balance <= Decimal("5"):
        return

    usd_to_use = (usd_balance * POSITION_SIZE_FRACTION).quantize(Decimal("0.01"))
    if usd_to_use <= Decimal("5"):
        return

    size = (usd_to_use / price).quantize(Decimal("0.00000001"))
    if size <= Decimal("0"):
        return

    tp_price = price * (Decimal("1") + TAKE_PROFIT_PCT)
    sl_price = price * (Decimal("1") - STOP_LOSS_PCT)

    positions.append({
        "market": market,
        "entry_price": price,
        "size": size,
        "tp": tp_price,
        "sl": sl_price,
        "opened_at": datetime.now(timezone.utc),
        "features": feats,
    })
    usd_balance -= usd_to_use
    trade_count += 1

    log(f"OPEN {market}: size={size:.8f} at {price:.4f}, TP={tp_price:.4f}, SL={sl_price:.4f} "
        f"(used ${usd_to_use:.2f}, remaining USD=${usd_balance:.2f})")


def update_positions(prices: dict[str, Decimal]):
    """
    Check all open positions for TP/SL hits, close them and update AI + stats.
    """
    global usd_balance, losing_streak, equity_peak_today

    still_open = []
    for pos in positions:
        market = pos["market"]
        price = prices.get(market)
        if price is None:
            # Can't check TP/SL without a price; keep open
            still_open.append(pos)
            continue

        entry = pos["entry_price"]
        size = pos["size"]
        tp = pos["tp"]
        sl = pos["sl"]
        feats = pos["features"]
        trend_feat, rsi_feat, vol_feat = feats

        hit_tp = price >= tp
        hit_sl = price <= sl

        if not hit_tp and not hit_sl:
            still_open.append(pos)
            continue

        # Close position
        side = "SELL"
        exit_price = price
        gross = size * exit_price
        cost = size * entry
        pnl_usd = (gross - cost).quantize(Decimal("0.01"))
        pnl_pct = (pnl_usd / cost) if cost != 0 else Decimal("0")

        usd_balance += gross

        win = pnl_usd > 0
        if win:
            losing_streak = 0
        else:
            losing_streak += 1

        # Learn from outcome
        ai_learn(trend_feat, rsi_feat, vol_feat, win)

        # Recompute equity after closing
        equity = portfolio_equity(prices)
        if equity > equity_peak_today:
            equity_peak_today = equity

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        log_trade(ts, market, side, entry, exit_price, size,
                  pnl_usd, pnl_pct, equity, win,
                  trend_feat, rsi_feat, vol_feat)

        result_str = "WIN" if win else "LOSS"
        log(f"CLOSE {market}: {result_str} pnl=${pnl_usd:.2f} "
            f"({pnl_pct * Decimal('100'):.2f}%), new USD balance=${usd_balance:.2f}")

    positions.clear()
    positions.extend(still_open)


# ============= MAIN LOOP =============

def main_loop():
    global today, equity_peak_today, trading_paused_for_today

    log("============================================================")
    log("CRYPTO PAPER-TRADING BOT (RANDOM MULTI-MARKET SCAN, AGGRESSIVE MODE)")
    log("NO REAL MONEY. NO REAL ORDERS. PUBLIC DATA ONLY.")
    log(f"Starting balance: ${START_BALANCE_USD}")
    log(f"Risk mode: AGGRESSIVE, max {MAX_CONCURRENT_POSITIONS} positions")
    log("============================================================")

    while True:
        loop_start = datetime.now(timezone.utc)

        # Reset daily stats if new day
        current_date = loop_start.date()
        if current_date != today:
            today = current_date
            equity_peak_today = portfolio_equity({})
            trading_paused_for_today = False
            log("----- New trading day, resetting daily stats -----")

        # Fetch latest prices for all relevant markets (positions + scan set)
        markets_to_price = set(m for m in ALL_MARKETS)
        for pos in positions:
            markets_to_price.add(pos["market"])

        prices: dict[str, Decimal] = {}
        for m in markets_to_price:
            p = get_latest_price(m)
            if p is not None:
                prices[m] = p

        # Update open positions (TP/SL, AI learning, balances)
        update_positions(prices)

        # Compute equity and risk metrics
        equity = portfolio_equity(prices)
        if equity > equity_peak_today:
            equity_peak_today = equity

        if equity_peak_today > 0:
            daily_dd = (equity_peak_today - equity) / equity_peak_today
        else:
            daily_dd = Decimal("0")

        # Check risk locks
        if daily_dd > MAX_DAILY_DRAWDOWN:
            trading_paused_for_today = True

        if losing_streak >= MAX_LOSING_STREAK:
            trading_paused_for_today = True

        # Summary
        dd_pct = daily_dd * Decimal("100")
        log(f"Summary: USD=${usd_balance:.2f}, positions={len(positions)}, "
            f"Equity=${equity:.2f}, DD={dd_pct:.2f}%, LosingStreak={losing_streak}")

        if trading_paused_for_today:
            log("Daily risk limits hit (DD or losing streak). "
                "Pausing NEW entries for the rest of the day, but managing open positions.")

        # Maybe open new positions (if allowed)
        if not trading_paused_for_today:
            maybe_open_new_position(prices)

        # Sleep until next cycle
        log(f"Sleeping for {SLEEP_SECONDS} seconds...\n")
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    import traceback

    try:
        main_loop()
    except Exception:
        with open("error.log", "w") as f:
            f.write("An error occurred:\n\n")
            traceback.print_exc(file=f)
        raise

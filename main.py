# main.py
import os
import time
import uuid
import math
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from event_logger import BotLogger

# If your repo already has api_client.py, we import it.
# We'll call ApiClient.get_candles() inside the adapter function below.
from api_client import ApiClient


# ----------------------------
# Helpers (indicators)
# ----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = (v * k) + (e * (1 - k))
    return e


def rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains += change
        else:
            losses += abs(change)
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100 - (100 / (1 + rs))


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


# ----------------------------
# Adapter: get candle closes
# ----------------------------
def fetch_candles_closes(client: ApiClient, market: str, granularity_sec: int, limit: int) -> List[float]:
    """
    EXPECTATION:
    Your api_client.py should provide something like:
      client.get_candles(product_id=market, granularity=granularity_sec, limit=limit)
    returning candles with close prices.

    Because I can’t see your exact function signature, this adapter tries common patterns.

    If this fails, scroll down to 'IF YOU GET AN ERROR HERE' for the 2-line fix.
    """

    # Try common method names/signatures:
    candles = None

    # 1) get_candles(product_id=..., granularity=..., limit=...)
    if candles is None and hasattr(client, "get_candles"):
        try:
            candles = client.get_candles(product_id=market, granularity=granularity_sec, limit=limit)
        except TypeError:
            pass
        except Exception:
            pass

    # 2) get_candles(market, granularity_sec, limit)
    if candles is None and hasattr(client, "get_candles"):
        try:
            candles = client.get_candles(market, granularity_sec, limit)
        except Exception:
            pass

    # 3) fetch_candles(...)
    if candles is None and hasattr(client, "fetch_candles"):
        try:
            candles = client.fetch_candles(market, granularity_sec, limit)
        except Exception:
            pass

    if candles is None:
        raise RuntimeError(
            "Could not fetch candles. Your api_client.py does not match the expected get_candles()/fetch_candles() patterns."
        )

    # Normalize candles → list of closes
    closes: List[float] = []

    # If candles are list of dicts
    if isinstance(candles, list) and candles and isinstance(candles[0], dict):
        # common keys: "close", "c"
        for c in candles[-limit:]:
            close = c.get("close", c.get("c"))
            if close is not None:
                closes.append(float(close))

    # If candles are list of lists like [time, low, high, open, close, volume]
    elif isinstance(candles, list) and candles and isinstance(candles[0], (list, tuple)):
        for row in candles[-limit:]:
            if len(row) >= 5:
                closes.append(float(row[4]))

    # If candles are something else, try last resort
    if not closes:
        # Try attribute access
        try:
            for c in candles[-limit:]:
                closes.append(float(getattr(c, "close")))
        except Exception:
            pass

    if not closes:
        raise RuntimeError("Fetched candles but could not parse close prices.")

    return closes


# ----------------------------
# Paper portfolio
# ----------------------------
class PaperPortfolio:
    def __init__(self, starting_equity: float):
        self.starting_equity = float(starting_equity)
        self.cash = float(starting_equity)
        self.open_positions: Dict[str, Dict[str, Any]] = {}  # trade_id -> position
        self.trade_pnls: List[float] = []

    def equity(self, mark_prices: Dict[str, float]) -> float:
        eq = self.cash
        for pos in self.open_positions.values():
            mkt = pos["market"]
            mark = mark_prices.get(mkt)
            if mark is None:
                continue
            # Long only in this template:
            eq += (mark - pos["entry_price"]) * pos["size"]
        return float(eq)

    def open_long(self, market: str, entry_price: float, size: float, meta: Dict[str, Any]) -> Dict[str, Any]:
        trade_id = meta.get("trade_id") or str(uuid.uuid4())
        if trade_id in self.open_positions:
            raise RuntimeError("trade_id already open")

        cost = entry_price * size
        if cost > self.cash:
            raise RuntimeError("Insufficient cash for paper buy")

        self.cash -= cost
        self.open_positions[trade_id] = {
            "trade_id": trade_id,
            "market": market,
            "side": "BUY",
            "entry_price": float(entry_price),
            "size": float(size),
            "opened_utc": utc_now_iso(),
            **meta
        }
        return self.open_positions[trade_id]

    def close_long(self, trade_id: str, exit_price: float, fees_usd: float = 0.0) -> Dict[str, Any]:
        if trade_id not in self.open_positions:
            raise RuntimeError("trade_id not open")

        pos = self.open_positions.pop(trade_id)
        entry_price = pos["entry_price"]
        size = pos["size"]

        proceeds = exit_price * size
        pnl = (exit_price - entry_price) * size - float(fees_usd)

        self.cash += proceeds
        self.trade_pnls.append(float(pnl))

        closed = {
            "trade_id": trade_id,
            "market": pos["market"],
            "side": "SELL",
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "size": float(size),
            "pnl_usd": float(pnl),
            "fees_usd": float(fees_usd),
            "closed_utc": utc_now_iso(),
        }
        return closed


# ----------------------------
# Strategy (simple + stable)
# ----------------------------
def decide_signal(closes: List[float]) -> Dict[str, Any]:
    """
    Returns:
      action: BUY/SELL/HOLD
      confidence: 0..1
      reasons: list[str]
    """
    if len(closes) < 60:
        return {"action": "HOLD", "confidence": 0.0, "reasons": ["not_enough_data"]}

    current = closes[-1]
    r = rsi(closes, 14)
    ema_fast = ema(closes[-60:], 20)
    ema_slow = ema(closes[-60:], 50)

    reasons = []
    confidence = 0.50

    trend_up = (ema_fast is not None and ema_slow is not None and ema_fast > ema_slow)
    oversold = (r is not None and r < 30)

    if trend_up:
        reasons.append("trend_up (EMA20>EMA50)")
        confidence += 0.08
    if oversold:
        reasons.append("RSI_oversold (<30)")
        confidence += 0.12

    # Buy logic
    if trend_up and (oversold or (r is not None and r < 40)):
        return {"action": "BUY", "confidence": min(confidence, 0.95), "reasons": reasons or ["trend_up"]}

    # Sell logic (simple exit condition)
    if r is not None and r > 65:
        reasons.append("RSI_overbought (>65)")
        return {"action": "SELL", "confidence": 0.60, "reasons": reasons}

    return {"action": "HOLD", "confidence": 0.45, "reasons": reasons or ["no_edge"]}


# ----------------------------
# Main loop
# ----------------------------
def main():
    # ---- Config via env ----
    markets = os.getenv("MARKETS", "BTC-USD,ETH-USD").split(",")
    markets = [m.strip() for m in markets if m.strip()]

    cycle_seconds = int(os.getenv("CYCLE_SECONDS", "360"))  # 6 min default
    granularity_sec = int(os.getenv("GRANULARITY_SEC", "300"))  # 5m candles
    candle_limit = int(os.getenv("CANDLE_LIMIT", "120"))

    starting_equity = float(os.getenv("STARTING_EQUITY_USD", "1000"))
    trade_size_units = float(os.getenv("TRADE_SIZE_UNITS", "0.01"))  # size in base units (BTC etc)
    max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS", "3"))

    # Risk/exit rules
    take_profit_pct = float(os.getenv("TAKE_PROFIT_PCT", "0.008"))   # 0.8%
    stop_loss_pct = float(os.getenv("STOP_LOSS_PCT", "0.006"))       # 0.6%

    # ---- Init ----
    logger = BotLogger(log_dir="logs")
    client = ApiClient()
    portfolio = PaperPortfolio(starting_equity=starting_equity)

    logger.event("config", {
        "markets": markets,
        "cycle_seconds": cycle_seconds,
        "granularity_sec": granularity_sec,
        "candle_limit": candle_limit,
        "starting_equity_usd": starting_equity,
        "trade_size_units": trade_size_units,
        "max_open_positions": max_open_positions,
        "take_profit_pct": take_profit_pct,
        "stop_loss_pct": stop_loss_pct
    })

    cycle = 0

    while True:
        cycle += 1
        try:
            logger.heartbeat("running", {"cycle": cycle})

            mark_prices: Dict[str, float] = {}

            # 1) Fetch data and generate signals
            for market in markets:
                closes = fetch_candles_closes(client, market, granularity_sec, candle_limit)
                price = float(closes[-1])
                mark_prices[market] = price

                sig = decide_signal(closes)
                action = sig["action"]
                confidence = float(sig["confidence"])
                reasons = sig["reasons"]

                # Event log the signal (THIS is your proper events log)
                logger.event("signal", {
                    "market": market,
                    "action": action,
                    "confidence": confidence,
                    "reason": reasons,
                    "price": price,
                    "size": trade_size_units,
                })

                # 2) Manage existing positions: stop-loss / take-profit
                # (We iterate positions safely by copying items)
                for trade_id, pos in list(portfolio.open_positions.items()):
                    if pos["market"] != market:
                        continue
                    entry = pos["entry_price"]
                    size = pos["size"]

                    # Long PnL% (approx)
                    pnl_pct = (price - entry) / entry

                    if pnl_pct >= take_profit_pct:
                        closed = portfolio.close_long(trade_id, exit_price=price, fees_usd=0.0)
                        logger.trade_close(closed)

                    elif pnl_pct <= -stop_loss_pct:
                        closed = portfolio.close_long(trade_id, exit_price=price, fees_usd=0.0)
                        logger.trade_close(closed)

                # 3) Entry / Exit decisions
                # SELL signal: close one position for this market (if any)
                if action == "SELL":
                    # close first open pos on this market
                    for trade_id, pos in list(portfolio.open_positions.items()):
                        if pos["market"] == market:
                            closed = portfolio.close_long(trade_id, exit_price=price, fees_usd=0.0)
                            logger.trade_close(closed)
                            break

                # BUY signal: open if under max positions
                if action == "BUY":
                    if len(portfolio.open_positions) < max_open_positions:
                        # Open a long
                        opened = portfolio.open_long(
                            market=market,
                            entry_price=price,
                            size=trade_size_units,
                            meta={
                                "strategy": "ema_rsi_template",
                                "confidence": confidence,
                                "reasons": reasons,
                            }
                        )
                        logger.trade_open(opened)

            # 4) Equity logging (THIS fixes your time_utc null problem)
            eq = portfolio.equity(mark_prices)
            logger.equity_point(eq, payload={
                "cycle": cycle,
                "open_positions": len(portfolio.open_positions)
            })

            # 5) sleep until next cycle
            time.sleep(cycle_seconds)

        except Exception as e:
            logger.event("error", {"message": str(e)})
            logger.heartbeat("error", {"cycle": cycle, "error": str(e)})
            # Re-raise so you can see the crash in logs/console too
            raise


if __name__ == "__main__":
    main()

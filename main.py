import os
import csv
import json
import time
import random
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

try:
    import requests
except Exception:
    requests = None  # requirements.txt should include it on Render


# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("crypto-ai-bot")


# ---------------------------
# Paths / Config
# ---------------------------
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.getcwd(), "data"))

BOT_STATE_FILE = os.path.join(DATA_DIR, "bot_state.json")
HEARTBEAT_JSON = os.path.join(DATA_DIR, "heartbeat.json")
HEARTBEAT_TXT = os.path.join(DATA_DIR, "heartbeat.txt")
TRADES_CSV = os.path.join(DATA_DIR, "trades.csv")
EQUITY_CSV = os.path.join(DATA_DIR, "equity_curve.csv")
TRAINING_FILE = os.path.join(DATA_DIR, "training_events.csv")  # optional

DEFAULT_MARKETS = ["BTC-USD", "ETH-USD", "LTC-USD", "BCH-USD", "SOL-USD", "ADA-USD"]
MARKETS = [m.strip() for m in os.environ.get("MARKETS", ",".join(DEFAULT_MARKETS)).split(",") if m.strip()]

START_EQUITY = float(os.environ.get("START_EQUITY", "1000"))
CYCLE_SECONDS = int(os.environ.get("CYCLE_SECONDS", "60"))

MAX_OPEN_POSITIONS = int(os.environ.get("MAX_OPEN_POSITIONS", "2"))
POSITION_USD = float(os.environ.get("POSITION_USD", "100"))

TAKE_PROFIT_PCT = float(os.environ.get("TAKE_PROFIT_PCT", "3.0"))
STOP_LOSS_PCT = float(os.environ.get("STOP_LOSS_PCT", "2.0"))
MAX_HOLD_MINUTES = int(os.environ.get("MAX_HOLD_MINUTES", "180"))

ENTRY_CHANCE = float(os.environ.get("ENTRY_CHANCE", "0.06"))
RISK_MODE = os.environ.get("RISK_MODE", "normal")

# IMPORTANT: This must be set in the WORKER environment variables
API_URL = os.environ.get("API_URL", "").strip().rstrip("/")
API_KEY = os.environ.get("API_KEY", "").strip()

COINBASE_SPOT = "https://api.coinbase.com/v2/prices/{symbol}/spot"


# ---------------------------
# Helpers
# ---------------------------
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _minutes_between(iso_a: str, iso_b: str) -> float:
    try:
        a = datetime.fromisoformat(iso_a.replace("Z", "+00:00"))
        b = datetime.fromisoformat(iso_b.replace("Z", "+00:00"))
        return (b - a).total_seconds() / 60.0
    except Exception:
        return 0.0


def _ensure_csv_header(path: str, header: List[str]) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)


def _append_csv_row(path: str, row: List[Any]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row)


# ---------------------------
# API Push (THIS IS THE DASHBOARD FIX)
# ---------------------------
def _api_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-KEY"] = API_KEY
    return h


def api_post(path: str, payload: Dict[str, Any]) -> bool:
    """Post payload to your crypto-ai-api service. Returns True if success."""
    if not API_URL or requests is None:
        return False
    try:
        url = f"{API_URL}{path}"
        r = requests.post(url, json=payload, headers=_api_headers(), timeout=10)
        if r.status_code >= 200 and r.status_code < 300:
            return True
        log.warning(f"API POST failed {r.status_code}: {r.text[:200]}")
        return False
    except Exception as e:
        log.warning(f"API POST exception: {e}")
        return False


def push_heartbeat(payload: Dict[str, Any]) -> None:
    api_post("/ingest/heartbeat", payload)


def push_trade(row: Dict[str, Any]) -> None:
    api_post("/ingest/trade", row)


# ---------------------------
# State
# ---------------------------
@dataclass
class Position:
    market: str
    entry_time: str
    entry_price: float
    qty: float
    take_profit_pct: float = TAKE_PROFIT_PCT
    stop_loss_pct: float = STOP_LOSS_PCT
    risk_mode: str = RISK_MODE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BotState:
    equity_usd: float = START_EQUITY
    # NOTE: keep compatibility: could contain dicts OR old string markets
    open_positions: List[Union[Dict[str, Any], str]] = field(default_factory=list)

    def normalize_positions(self) -> None:
        """
        Make sure open_positions is a list of dicts with at least a 'market' key.
        If older state stored positions as strings, convert them safely.
        """
        normalized: List[Dict[str, Any]] = []
        for p in self.open_positions:
            if isinstance(p, dict):
                if "market" in p and isinstance(p["market"], str):
                    normalized.append(p)
                elif "symbol" in p and isinstance(p["symbol"], str):
                    p["market"] = p.pop("symbol")
                    normalized.append(p)
        elif isinstance(p, str):
    # Old/invalid legacy entry (string market) — drop it.
    continue
        self.open_positions = normalized

    def to_json(self) -> Dict[str, Any]:
        self.normalize_positions()
        return {
            "equity_usd": float(self.equity_usd),
            "open_positions": self.open_positions,
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "BotState":
        s = BotState(
            equity_usd=_safe_float(d.get("equity_usd"), START_EQUITY) or START_EQUITY,
            open_positions=d.get("open_positions") or [],
        )
        s.normalize_positions()
        return s


def load_state() -> BotState:
    ensure_data_dir()
    if not os.path.exists(BOT_STATE_FILE):
        return BotState()
    try:
        with open(BOT_STATE_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        return BotState.from_json(d)
    except Exception as e:
        log.warning(f"Failed to load state, starting fresh: {e}")
        return BotState()


def save_state(state: BotState) -> None:
    ensure_data_dir()
    tmp = BOT_STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state.to_json(), f, indent=2)
    os.replace(tmp, BOT_STATE_FILE)


# ---------------------------
# Heartbeat
# ---------------------------
def write_heartbeat(state: BotState, status: str, note: str = "") -> Dict[str, Any]:
    """
    Writes local data/heartbeat.json and also RETURNS the payload.
    We then POST this payload to the API (crypto-ai-api) so the dashboard updates.
    """
    ensure_data_dir()
    state.normalize_positions()

    payload = {
        "time_utc": now_utc_iso(),
        "status": status,
        "note": note,
        "equity_usd": float(state.equity_usd),
        # robust for dict positions
        "open_positions": [p["market"] if isinstance(p, dict) else str(p) for p in state.open_positions],
    }

    tmp = HEARTBEAT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, HEARTBEAT_JSON)

    with open(HEARTBEAT_TXT, "w", encoding="utf-8") as f:
        f.write(f"{payload['time_utc']} | {status} | equity={payload['equity_usd']}\n")

    # DASHBOARD FIX: push to API
    push_heartbeat(payload)

    return payload


# ---------------------------
# CSV outputs
# ---------------------------
def record_equity(equity_usd: float) -> None:
    _ensure_csv_header(EQUITY_CSV, ["time", "equity_usd"])
    _append_csv_row(EQUITY_CSV, [now_utc_iso(), float(equity_usd)])


def record_trade(row: Dict[str, Any]) -> None:
    _ensure_csv_header(
        TRADES_CSV,
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

    _append_csv_row(
        TRADES_CSV,
        [
            row.get("entry_time", ""),
            row.get("exit_time", ""),
            row.get("hold_minutes", 0.0),
            row.get("market", ""),
            row.get("entry_price", 0.0),
            row.get("exit_price", 0.0),
            row.get("qty", 0.0),
            row.get("pnl_usd", 0.0),
            row.get("pnl_pct", 0.0),
            row.get("take_profit_pct", TAKE_PROFIT_PCT),
            row.get("stop_loss_pct", STOP_LOSS_PCT),
            row.get("risk_mode", RISK_MODE),
            row.get("trend_strength", ""),
            row.get("rsi", ""),
            row.get("volatility", ""),
        ],
    )

    # DASHBOARD FIX: push to API
    push_trade(row)


# ---------------------------
# Prices
# ---------------------------
def fetch_coinbase_spot(market: str) -> Optional[float]:
    if requests is None:
        return None
    try:
        url = COINBASE_SPOT.format(symbol=market)
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        amt = data.get("data", {}).get("amount")
        return _safe_float(amt, None)
    except Exception:
        return None


def fetch_prices(markets: List[str]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    for m in markets:
        p = fetch_coinbase_spot(m)
        if p is not None and p > 0:
            prices[m] = p
    return prices


# ---------------------------
# Simple paper strategy
# ---------------------------
def should_enter_long(market: str) -> bool:
    return random.random() < ENTRY_CHANCE


def close_reason(pos: Dict[str, Any], price: float) -> Optional[str]:
    entry = _safe_float(pos.get("entry_price"), None) or 0.0
    if entry <= 0:
        return None

    tp = float(pos.get("take_profit_pct", TAKE_PROFIT_PCT)) / 100.0
    sl = float(pos.get("stop_loss_pct", STOP_LOSS_PCT)) / 100.0

    if price >= entry * (1.0 + tp):
        return "TP"
    if price <= entry * (1.0 - sl):
        return "SL"

    hold = _minutes_between(pos.get("entry_time", now_utc_iso()), now_utc_iso())
    if hold >= MAX_HOLD_MINUTES:
        return "TIME"

    return None


def open_position(state: BotState, market: str, price: float) -> None:
    qty = 0.0 if price <= 0 else POSITION_USD / price

    pos = Position(
        market=market,
        entry_time=now_utc_iso(),
        entry_price=float(price),
        qty=float(qty),
        take_profit_pct=TAKE_PROFIT_PCT,
        stop_loss_pct=STOP_LOSS_PCT,
        risk_mode=RISK_MODE,
    ).to_dict()

    state.open_positions.append(pos)

    log.info(
        f"[ENTRY] Opened LONG {market} @ {price:.6f} qty={qty:.6f} "
        f"TP={TAKE_PROFIT_PCT:.1f}% SL={STOP_LOSS_PCT:.1f}%"
    )


def close_position(state: BotState, pos: Dict[str, Any], exit_price: float, reason: str) -> None:
    entry_price = float(pos.get("entry_price", 0.0) or 0.0)
    qty = float(pos.get("qty", 0.0) or 0.0)

    if entry_price <= 0 or qty <= 0:
        pnl = 0.0
        pnl_pct = 0.0
    else:
        pnl = (exit_price - entry_price) * qty
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0

    exit_time = now_utc_iso()
    hold_minutes = _minutes_between(pos.get("entry_time", exit_time), exit_time)

    state.equity_usd = float(state.equity_usd) + float(pnl)

    record_trade(
        {
            "entry_time": pos.get("entry_time", ""),
            "exit_time": exit_time,
            "hold_minutes": hold_minutes,
            "market": pos.get("market", ""),
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "qty": qty,
            "pnl_usd": float(pnl),
            "pnl_pct": float(pnl_pct),
            "take_profit_pct": pos.get("take_profit_pct", TAKE_PROFIT_PCT),
            "stop_loss_pct": pos.get("stop_loss_pct", STOP_LOSS_PCT),
            "risk_mode": pos.get("risk_mode", RISK_MODE),
            "trend_strength": pos.get("trend_strength", ""),
            "rsi": pos.get("rsi", ""),
            "volatility": pos.get("volatility", ""),
        }
    )

    log.info(f"[EXIT] Closed {pos.get('market')} reason={reason} pnl=${pnl:.2f} equity=${state.equity_usd:.2f}")


# ---------------------------
# Main loop
# ---------------------------
def main() -> None:
    ensure_data_dir()
    state = load_state()
    state.normalize_positions()

    _ensure_csv_header(EQUITY_CSV, ["time", "equity_usd"])
    _ensure_csv_header(
        TRADES_CSV,
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
    if not os.path.exists(TRAINING_FILE):
        _ensure_csv_header(TRAINING_FILE, ["time_utc", "event", "detail"])

    log.info("Crypto-AI-Bot running (paper trading, Coinbase prices)")
    log.info(f"DATA_DIR={DATA_DIR}")
    log.info(f"Markets={MARKETS}")
    log.info(f"API_URL={API_URL if API_URL else '(none)'} API_KEY={'(set)' if API_KEY else '(disabled)'}")
    log.info(f"Equity=${state.equity_usd:.2f}")

    # Boot heartbeat (stops 'heartbeat not found')
    write_heartbeat(state, "running", note="boot")
    record_equity(state.equity_usd)
    save_state(state)

    while True:
        try:
            state.normalize_positions()

            prices = fetch_prices(MARKETS)

            # Close logic
            still_open: List[Dict[str, Any]] = []
            for p in state.open_positions:
                if not isinstance(p, dict):
                    still_open.append(
                        {"market": str(p), "entry_time": now_utc_iso(), "entry_price": 0.0, "qty": 0.0}
                    )
                    continue

                m = p.get("market")
                if not m or m not in prices:
                    still_open.append(p)
                    continue

                reason = close_reason(p, prices[m])
                if reason:
                    close_position(state, p, prices[m], reason)
                else:
                    still_open.append(p)

            state.open_positions = still_open

            # Entry logic
            open_markets = set(pp.get("market") for pp in state.open_positions if isinstance(pp, dict))
            if len(state.open_positions) < MAX_OPEN_POSITIONS:
                candidates = [m for m in MARKETS if m in prices and m not in open_markets]
                random.shuffle(candidates)
                for m in candidates:
                    if len(state.open_positions) >= MAX_OPEN_POSITIONS:
                        break
                    if should_enter_long(m):
                        open_position(state, m, prices[m])

            # Persist
            record_equity(state.equity_usd)
            save_state(state)
            write_heartbeat(state, "running", note="cycle")

            log.info(
                f"[CYCLE] {now_utc_iso()} open={[p.get('market') for p in state.open_positions if isinstance(p, dict)]} "
                f"equity=${state.equity_usd:.2f}"
            )

        except Exception as e:
            log.exception(f"Unhandled error in cycle: {e}")
            try:
                write_heartbeat(state, "error", note=str(e)[:200])
                save_state(state)
            except Exception:
                pass

        time.sleep(CYCLE_SECONDS)


if __name__ == "__main__":
    main()

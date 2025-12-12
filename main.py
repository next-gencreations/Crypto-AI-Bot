import os
import json
import time
import random
import signal
import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import requests


# ---------------------------
# Config
# ---------------------------

DATA_DIR = os.environ.get("DATA_DIR", "/opt/render/project/src/data")
API_BASE = os.environ.get("CRYPTO_AI_API_URL", "https://crypto-ai-api-h921.onrender.com").rstrip("/")
API_KEY = os.environ.get("CRYPTO_AI_API_KEY", "").strip()  # optional
BOT_NAME = os.environ.get("BOT_NAME", "Crypto-AI-Bot")

# cycle timing
CYCLE_SECONDS = int(os.environ.get("CYCLE_SECONDS", "60"))

# paper trading params
START_EQUITY = float(os.environ.get("START_EQUITY", "1000"))
RISK_MODE = os.environ.get("RISK_MODE", "normal")  # "safe" | "normal" | "aggressive"
MAX_OPEN_POSITIONS = int(os.environ.get("MAX_OPEN_POSITIONS", "2"))

TP_PCT = float(os.environ.get("TAKE_PROFIT_PCT", "3.0"))      # take profit percent
SL_PCT = float(os.environ.get("STOP_LOSS_PCT", "2.0"))         # stop loss percent
POSITION_SIZE_PCT = float(os.environ.get("POSITION_SIZE_PCT", "0.25"))  # per trade size of equity

# Universe
MARKETS = [m.strip().upper() for m in os.environ.get(
    "MARKETS",
    "BTC-USD,ETH-USD,LTC-USD,BCH-USD,SOL-USD,ADA-USD"
).split(",") if m.strip()]


# ---------------------------
# Helpers
# ---------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def append_csv(path: str, header: List[str], row: Dict[str, Any]) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def coinbase_spot_price_usd(market: str) -> Optional[float]:
    # market format like "BTC-USD"
    try:
        url = f"https://api.coinbase.com/v2/prices/{market}/spot"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        amt = data.get("data", {}).get("amount")
        return safe_float(amt, None)
    except Exception:
        return None


# ---------------------------
# API client with queue/retry
# ---------------------------

class ApiClient:
    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.session = requests.Session()

        # a simple queue persisted on disk so we can survive restarts
        self.queue_path = os.path.join(DATA_DIR, "api_queue.jsonl")

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            # keep flexible; app.py can read either header name if you add it later
            h["X-API-KEY"] = self.api_key
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def enqueue(self, endpoint: str, payload: Dict[str, Any]) -> None:
        record = {"endpoint": endpoint, "payload": payload, "ts": utc_now_iso()}
        ensure_dir(DATA_DIR)
        with open(self.queue_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def flush(self, max_items: int = 50) -> Tuple[int, int]:
        """
        Try to send queued items.
        Returns (sent, remaining).
        """
        if not os.path.exists(self.queue_path):
            return (0, 0)

        # read all lines
        with open(self.queue_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            return (0, 0)

        sent = 0
        keep: List[str] = []
        for i, line in enumerate(lines):
            if i >= max_items:
                keep.append(line)
                continue

            try:
                rec = json.loads(line)
                ok = self._post(rec["endpoint"], rec["payload"])
                if ok:
                    sent += 1
                else:
                    keep.append(line)
            except Exception:
                keep.append(line)

        # rewrite file with remaining queue
        tmp = self.queue_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(keep)
        os.replace(tmp, self.queue_path)

        return (sent, len(keep))

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> bool:
        try:
            url = f"{self.base_url}{endpoint}"
            r = self.session.post(url, headers=self._headers(), data=json.dumps(payload), timeout=15)
            return 200 <= r.status_code < 300
        except Exception:
            return False


# ---------------------------
# Trading model (paper)
# ---------------------------

@dataclass
class Position:
    market: str
    side: str  # "LONG" only in this simple version
    entry_time: str
    entry_price: float
    qty: float
    take_profit_pct: float
    stop_loss_pct: float

    def tp_price(self) -> float:
        return self.entry_price * (1.0 + self.take_profit_pct / 100.0)

    def sl_price(self) -> float:
        return self.entry_price * (1.0 - self.stop_loss_pct / 100.0)


@dataclass
class BotState:
    equity_usd: float
    open_positions: List[Dict[str, Any]]
    wins: int
    losses: int
    total_trades: int

    @staticmethod
    def default() -> "BotState":
        return BotState(
            equity_usd=START_EQUITY,
            open_positions=[],
            wins=0,
            losses=0,
            total_trades=0
        )


# ---------------------------
# File paths
# ---------------------------

STATE_FILE = os.path.join(DATA_DIR, "bot_state.json")
HEARTBEAT_JSON = os.path.join(DATA_DIR, "heartbeat.json")
HEARTBEAT_TXT = os.path.join(DATA_DIR, "heartbeat.txt")
EQUITY_CSV = os.path.join(DATA_DIR, "equity_curve.csv")
TRADES_CSV = os.path.join(DATA_DIR, "trades.csv")
TRAINING_CSV = os.path.join(DATA_DIR, "training_events.csv")


TRADES_HEADER = [
    "entry_time", "exit_time", "hold_minutes", "market",
    "entry_price", "exit_price", "qty",
    "pnl_usd", "pnl_pct",
    "take_profit_pct", "stop_loss_pct",
    "risk_mode", "trend_strength", "rsi", "volatility"
]

EQUITY_HEADER = ["time", "equity_usd"]

TRAINING_HEADER = ["time_utc", "event", "market", "details"]


# ---------------------------
# Graceful shutdown
# ---------------------------

shutdown_requested = False

def handle_signal(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    print("[WARNING] Shutdown requested. Saving state and exiting soon...")

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


# ---------------------------
# Strategy helpers
# ---------------------------

def choose_candidates(markets: List[str]) -> List[str]:
    # For now random shuffle; you can replace with your ML selector later
    m = markets[:]
    random.shuffle(m)
    return m

def compute_mock_features() -> Tuple[float, float, float]:
    # placeholder features for training_events.csv / dashboard analytics
    trend_strength = round(random.uniform(0.0, 1.0), 3)
    rsi = round(random.uniform(30.0, 70.0), 2)
    volatility = round(random.uniform(0.5, 3.0), 3)
    return trend_strength, rsi, volatility

def position_size_usd(equity: float) -> float:
    # adjust sizing slightly by risk mode
    mult = 1.0
    if RISK_MODE.lower() == "safe":
        mult = 0.6
    elif RISK_MODE.lower() == "aggressive":
        mult = 1.4
    usd = equity * POSITION_SIZE_PCT * mult
    return max(usd, 10.0)


# ---------------------------
# Main loop
# ---------------------------

def load_state() -> BotState:
    obj = read_json(STATE_FILE, None)
    if not obj:
        return BotState.default()
    try:
        return BotState(
            equity_usd=safe_float(obj.get("equity_usd"), START_EQUITY),
            open_positions=obj.get("open_positions", []) or [],
            wins=int(obj.get("wins", 0)),
            losses=int(obj.get("losses", 0)),
            total_trades=int(obj.get("total_trades", 0)),
        )
    except Exception:
        return BotState.default()

def save_state(state: BotState) -> None:
    write_json(STATE_FILE, asdict(state))

def write_heartbeat(state: BotState, status: str, note: str = "") -> Dict[str, Any]:
    hb = {
        "time_utc": utc_now_iso(),
        "status": status,
        "note": note,
        "equity_usd": round(state.equity_usd, 4),
        "open_positions": [p.get("market") for p in state.open_positions],
        "bot": BOT_NAME,
    }
    write_json(HEARTBEAT_JSON, hb)
    with open(HEARTBEAT_TXT, "w", encoding="utf-8") as f:
        f.write(f"{hb['time_utc']} {hb['status']} equity={hb['equity_usd']} open={len(hb['open_positions'])}\n")
    return hb

def append_equity(state: BotState) -> Dict[str, Any]:
    row = {"time": utc_now_iso(), "equity_usd": round(state.equity_usd, 4)}
    append_csv(EQUITY_CSV, EQUITY_HEADER, row)
    return row

def record_training_event(event: str, market: str, details: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "time_utc": utc_now_iso(),
        "event": event,
        "market": market,
        "details": json.dumps(details, ensure_ascii=False)
    }
    append_csv(TRAINING_CSV, TRAINING_HEADER, row)
    return row

def close_position(state: BotState, pos: Position, exit_price: float, exit_time: str,
                   trend_strength: float, rsi: float, volatility: float) -> Dict[str, Any]:
    pnl_usd = (exit_price - pos.entry_price) * pos.qty
    pnl_pct = ((exit_price / pos.entry_price) - 1.0) * 100.0

    state.equity_usd += pnl_usd
    state.total_trades += 1
    if pnl_usd >= 0:
        state.wins += 1
    else:
        state.losses += 1

    # hold time in minutes
    try:
        entry_dt = datetime.fromisoformat(pos.entry_time.replace("Z", "+00:00"))
        exit_dt = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))
        hold_minutes = max(0.0, (exit_dt - entry_dt).total_seconds() / 60.0)
    except Exception:
        hold_minutes = 0.0

    trade_row = {
        "entry_time": pos.entry_time,
        "exit_time": exit_time,
        "hold_minutes": round(hold_minutes, 4),
        "market": pos.market,
        "entry_price": round(pos.entry_price, 6),
        "exit_price": round(exit_price, 6),
        "qty": round(pos.qty, 8),
        "pnl_usd": round(pnl_usd, 6),
        "pnl_pct": round(pnl_pct, 6),
        "take_profit_pct": pos.take_profit_pct,
        "stop_loss_pct": pos.stop_loss_pct,
        "risk_mode": RISK_MODE,
        "trend_strength": trend_strength,
        "rsi": rsi,
        "volatility": volatility,
    }

    append_csv(TRADES_CSV, TRADES_HEADER, trade_row)
    return trade_row

def main():
    ensure_dir(DATA_DIR)

    api = ApiClient(API_BASE, API_KEY)
    state = load_state()

    print(f"[INFO] {BOT_NAME} running (paper trading, Coinbase prices)")
    print(f"[INFO] DATA_DIR={DATA_DIR}")
    print(f"[INFO] Markets={MARKETS}")
    print(f"[INFO] API_URL={API_BASE}  API_KEY={'(set)' if API_KEY else '(disabled)'}")
    print(f"[INFO] Equity=${state.equity_usd:.2f}")

    # initial heartbeat + equity
    hb = write_heartbeat(state, "running", note="boot")
    eq = append_equity(state)

    # enqueue initial snapshots
    api.enqueue("/ingest/heartbeat", hb)
    api.enqueue("/ingest/equity", eq)
    api.flush()

    while not shutdown_requested:
        cycle_start = time.time()
        now = utc_now_iso()

        # prices
        prices: Dict[str, float] = {}
        for m in MARKETS:
            p = coinbase_spot_price_usd(m)
            if p is not None:
                prices[m] = p

        # rebuild open positions list into Position objects
        open_positions: List[Position] = []
        for op in state.open_positions:
            try:
                open_positions.append(Position(
                    market=op["market"],
                    side=op.get("side", "LONG"),
                    entry_time=op["entry_time"],
                    entry_price=float(op["entry_price"]),
                    qty=float(op["qty"]),
                    take_profit_pct=float(op.get("take_profit_pct", TP_PCT)),
                    stop_loss_pct=float(op.get("stop_loss_pct", SL_PCT)),
                ))
            except Exception:
                continue

        # --- Close logic (TP/SL) ---
        still_open: List[Position] = []
        closed_trades: List[Dict[str, Any]] = []

        for pos in open_positions:
            price = prices.get(pos.market)
            if price is None:
                still_open.append(pos)
                continue

            trend_strength, rsi, volatility = compute_mock_features()

            hit_tp = price >= pos.tp_price()
            hit_sl = price <= pos.sl_price()

            if hit_tp or hit_sl:
                trade_row = close_position(
                    state=state,
                    pos=pos,
                    exit_price=price,
                    exit_time=now,
                    trend_strength=trend_strength,
                    rsi=rsi,
                    volatility=volatility
                )
                closed_trades.append(trade_row)

                # training event
                record_training_event(
                    event="close_trade",
                    market=pos.market,
                    details={"hit": "TP" if hit_tp else "SL", "exit_price": price}
                )
            else:
                still_open.append(pos)

        # --- Open logic ---
        if len(still_open) < MAX_OPEN_POSITIONS and prices:
            candidates = choose_candidates([m for m in MARKETS if m in prices])
            for m in candidates:
                if len(still_open) >= MAX_OPEN_POSITIONS:
                    break
                if any(p.market == m for p in still_open):
                    continue

                # very simple entry condition (random chance)
                if random.random() < 0.25:
                    entry_price = prices[m]
                    usd_size = position_size_usd(state.equity_usd)
                    qty = usd_size / entry_price

                    pos = Position(
                        market=m,
                        side="LONG",
                        entry_time=now,
                        entry_price=entry_price,
                        qty=qty,
                        take_profit_pct=TP_PCT,
                        stop_loss_pct=SL_PCT
                    )
                    still_open.append(pos)

                    trend_strength, rsi, volatility = compute_mock_features()
                    record_training_event(
                        event="open_trade",
                        market=m,
                        details={
                            "entry_price": entry_price,
                            "qty": qty,
                            "tp_pct": TP_PCT,
                            "sl_pct": SL_PCT,
                            "trend_strength": trend_strength,
                            "rsi": rsi,
                            "volatility": volatility
                        }
                    )

                    print(f"[ENTRY] Opened LONG {m} @ {entry_price:.6f} qty={qty:.8f} TP={TP_PCT}% SL={SL_PCT}%")

        # persist open positions back to state
        state.open_positions = [asdict(p) for p in still_open]
        save_state(state)

        # heartbeat + equity (every cycle)
        hb = write_heartbeat(state, "running")
        eq = append_equity(state)

        # enqueue to API
        api.enqueue("/ingest/heartbeat", hb)
        api.enqueue("/ingest/equity", eq)

        for tr in closed_trades:
            api.enqueue("/ingest/trade", tr)

        # also send a periodic training ping (keeps total_events moving)
        api.enqueue("/ingest/training", {
            "time_utc": utc_now_iso(),
            "event": "cycle",
            "market": "",
            "details": json.dumps({"open_positions": len(still_open)}, ensure_ascii=False)
        })

        sent, remaining = api.flush(max_items=100)
        if sent or remaining:
            print(f"[INFO] API flush: sent={sent} remaining_queue={remaining}")

        # cycle log line like your existing output
        selected = list(prices.keys())[:3]
        open_mkts = [p.market for p in still_open]
        print(f"[CYCLE] {now} selected={selected} open={open_mkts} equity={state.equity_usd:.2f}")

        # sleep remainder
        elapsed = time.time() - cycle_start
        sleep_for = max(1, CYCLE_SECONDS - int(elapsed))
        for _ in range(sleep_for):
            if shutdown_requested:
                break
            time.sleep(1)

    # shutdown
    hb = write_heartbeat(state, "stopping", note="shutdown")
    api.enqueue("/ingest/heartbeat", hb)
    api.flush()
    save_state(state)
    print("[INFO] Worker exited cleanly.")


if __name__ == "__main__":
    main()

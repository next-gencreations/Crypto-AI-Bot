
import os
import time
import random
import logging
import requests
from datetime import datetime, timezone, timedelta

# =========================
# CONFIG
# =========================

API_URL = (os.getenv("API_URL") or "https://crypto-ai-api-1-7cte.onrender.com").rstrip("/")
CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS", "30"))

START_EQUITY = float(os.getenv("START_EQUITY", "1000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))  # 1% paper risk

# cryo rules (tweak later)
MAX_LOSS_STREAK = int(os.getenv("MAX_LOSS_STREAK", "4"))
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "3.5"))  # % from peak
CRYO_SECONDS = int(os.getenv("CRYO_SECONDS", "600"))  # 10 min

MARKETS = [m.strip() for m in (os.getenv("MARKETS", "BTCUSDT,ETHUSDT").split(",")) if m.strip()]

# =========================
# LOGGING
# =========================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("Crypto-AI-Bot")

# =========================
# STATE
# =========================

state = {
    "equity": START_EQUITY,
    "peak_equity": START_EQUITY,
    "wins": 0,
    "losses": 0,
    "loss_streak": 0,
    "total_trades": 0,
    "total_pnl_usd": 0.0,
    "pet": {
        "name": "TradePet",
        "sex": os.getenv("PET_SEX", "boy"),  # boy/girl (cosmetic)
        "stage": "egg",
        "health": 100.0,
        "mood": "focused",
        "hunger": 40.0,
        "growth": 0.0,
        "fainted_until_utc": "",
        "time_utc": None,
    }
}

# =========================
# API HELPERS
# =========================

def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _post(path, payload):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        log.warning(f"POST {path} failed: {e}")
        return False

def _get(path):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        log.warning(f"GET {path} failed: {e}")
        return None

def api_control():
    c = _get("/control") or {}
    return {
        "state": (c.get("state") or "ACTIVE").upper(),
        "pause_until_utc": c.get("pause_until_utc", ""),
        "cryo_until_utc": c.get("cryo_until_utc", ""),
        "pause_reason": c.get("pause_reason", ""),
        "cryo_reason": c.get("cryo_reason", ""),
    }

def enter_cryo(reason: str, details: dict):
    # Tell API we entered cryo
    _post("/control/cryo", {"seconds": CRYO_SECONDS, "reason": reason})

    # Record a "death" log (really: cryo/death history)
    _post("/ingest/death", {
        "time_utc": utc_now(),
        "source": "bot",
        "reason": "CRYO_TRIGGERED",
        "details": {"why": reason, **(details or {})}
    })

# =========================
# PET LOGIC
# =========================

def pet_tick():
    p = state["pet"]
    now = utc_now()
    p["time_utc"] = now

    # hunger rises
    p["hunger"] = min(100.0, p["hunger"] + 2.0)

    # health reacts to hunger
    if p["hunger"] > 85:
        p["health"] = max(0.0, p["health"] - 2.0)
    elif p["hunger"] < 35:
        p["health"] = min(100.0, p["health"] + 0.6)

    # mood
    if p["health"] < 20:
        p["mood"] = "sick"
    elif p["hunger"] > 85:
        p["mood"] = "hungry"
    else:
        p["mood"] = "focused"

    # hatch rule
    if p["stage"] == "egg" and (state["total_pnl_usd"] > 0 or state["wins"] >= 5):
        p["stage"] = "hatched"
        p["growth"] = max(p["growth"], 10.0)

def pet_on_trade(pnl):
    p = state["pet"]
    if pnl > 0:
        p["hunger"] = max(0.0, p["hunger"] - 12.0)
        p["health"] = min(100.0, p["health"] + 2.5)
        p["growth"] = min(100.0, p["growth"] + 2.0)
        p["mood"] = "happy"
        # “sound”
        _post("/ingest/event", {"time_utc": utc_now(), "type": "sound", "message": "purr", "details": {"pnl": pnl}})
    else:
        p["hunger"] = min(100.0, p["hunger"] + 8.0)
        p["health"] = max(0.0, p["health"] - 4.0)
        p["mood"] = "sad"
        _post("/ingest/event", {"time_utc": utc_now(), "type": "sound", "message": "whimper", "details": {"pnl": pnl}})

# =========================
# TRADING LOGIC (SAFE PAPER SIM)
# =========================

def simulate_trade():
    risk_usd = state["equity"] * RISK_PER_TRADE

    # slightly “edgey” win chance (you'll replace later with real logic)
    win = random.random() > 0.47

    pnl = risk_usd * random.uniform(0.6, 1.6) if win else -risk_usd

    state["equity"] += pnl
    state["peak_equity"] = max(state["peak_equity"], state["equity"])
    state["total_pnl_usd"] += pnl
    state["total_trades"] += 1

    if pnl > 0:
        state["wins"] += 1
        state["loss_streak"] = 0
    else:
        state["losses"] += 1
        state["loss_streak"] += 1

    return pnl

def survival_mode():
    p = state["pet"]
    if p["hunger"] > 90:
        return "STARVING"
    if p["hunger"] > 75:
        return "HUNGRY"
    if p["health"] < 25:
        return "SICK"
    return "NORMAL"

def drawdown_pct():
    peak = max(1e-9, state["peak_equity"])
    dd = (peak - state["equity"]) / peak * 100.0
    return dd

# =========================
# PRICE TICKS (FOR OHLC)
# =========================

_last_prices = {"BTCUSDT": 42000.0, "ETHUSDT": 2200.0}

def generate_fake_prices():
    # simple random walk so candles look real-ish
    for m in MARKETS:
        base = _last_prices.get(m, 1000.0)
        step = base * random.uniform(-0.0018, 0.0018)
        base = max(0.1, base + step)
        _last_prices[m] = base
    return dict(_last_prices)

# =========================
# MAIN LOOP
# =========================

def run():
    log.info(f"🚀 Bot started. API_URL={API_URL} cycle={CYCLE_SECONDS}s markets={MARKETS}")

    while True:
        # 1) tick pet always
        pet_tick()

        # 2) read API control state
        ctrl = api_control()
        mode = ctrl["state"]

        # 3) always push prices (so candles keep forming even during cryo)
        prices = generate_fake_prices()
        _post("/ingest/prices", {"time_utc": utc_now(), "prices": prices})

        # 4) If CRYO or PAUSED, do NOT trade — just heartbeat/pet
        if mode in ("CRYO", "PAUSED"):
            # In cryo, pet recovers slowly
            if mode == "CRYO":
                state["pet"]["mood"] = "cryo"
                state["pet"]["health"] = min(100.0, state["pet"]["health"] + 0.8)
                state["pet"]["hunger"] = min(100.0, state["pet"]["hunger"] + 0.5)

            # heartbeat + pet + equity point
            _post("/ingest/heartbeat", {
                "time_utc": utc_now(),
                "status": "running",
                "survival_mode": survival_mode(),
                "equity_usd": state["equity"],
                "wins": state["wins"],
                "losses": state["losses"],
                "total_trades": state["total_trades"],
                "total_pnl_usd": state["total_pnl_usd"],
                "markets": MARKETS,
                "open_positions": 0,
                "prices_ok": True,
            })
            _post("/ingest/pet", {**state["pet"], "time_utc": utc_now(), "survival_mode": survival_mode()})
            _post("/ingest/equity", {"time_utc": utc_now(), "equity_usd": state["equity"]})

            log.info(f"🧊 {mode} active ({ctrl.get('cryo_reason') or ctrl.get('pause_reason')}). No trading.")
            time.sleep(CYCLE_SECONDS)
            continue

        # 5) Trade cycle (paper sim)
        pnl = simulate_trade()
        pet_on_trade(pnl)

        dd = drawdown_pct()

        log.info(
            f"Trade #{state['total_trades']} | PnL {pnl:.2f} | Equity {state['equity']:.2f} | "
            f"DD {dd:.2f}% | LossStreak {state['loss_streak']} | PetHP {state['pet']['health']:.1f}"
        )

        # 6) Cryo trigger rules (THIS is the “cryo tube” safety net)
        if state["loss_streak"] >= MAX_LOSS_STREAK:
            enter_cryo("loss_streak", {"loss_streak": state["loss_streak"], "drawdown_pct": dd, "equity": state["equity"]})

        if dd >= MAX_DRAWDOWN_PCT:
            enter_cryo("max_drawdown", {"drawdown_pct": dd, "peak": state["peak_equity"], "equity": state["equity"]})

        # 7) push state to API
        _post("/ingest/heartbeat", {
            "time_utc": utc_now(),
            "status": "running",
            "survival_mode": survival_mode(),
            "equity_usd": state["equity"],
            "wins": state["wins"],
            "losses": state["losses"],
            "total_trades": state["total_trades"],
            "total_pnl_usd": state["total_pnl_usd"],
            "markets": MARKETS,
            "open_positions": 0,
            "prices_ok": True,
        })

        _post("/ingest/pet", {**state["pet"], "time_utc": utc_now(), "survival_mode": survival_mode()})
        _post("/ingest/equity", {"time_utc": utc_now(), "equity_usd": state["equity"]})

        # trade log
        _post("/ingest/trade", {
            "time_utc": utc_now(),
            "market": "BTCUSDT",
            "side": "buy" if pnl >= 0 else "sell",
            "size_usd": state["equity"] * RISK_PER_TRADE,
            "price": prices.get("BTCUSDT", 0),
            "pnl_usd": pnl,
            "confidence": 0.60,
            "reason": "paper_sim",
        })

        time.sleep(CYCLE_SECONDS)

if __name__ == "__main__":
    run()

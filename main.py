import os
import time
import json
import random
import logging
import traceback
import requests
from datetime import datetime, timezone

# =========================
# CONFIG
# =========================

API_URL = os.environ.get("API_URL", "https://crypto-ai-api-1-7cte.onrender.com")
CYCLE_SECONDS = int(os.environ.get("CYCLE_SECONDS", "30"))

START_EQUITY = float(os.environ.get("START_EQUITY", "1000"))
RISK_PER_TRADE_DEFAULT = float(os.environ.get("RISK_PER_TRADE", "0.01"))

# Recovery / survival
DEATH_COOLDOWN_SECONDS = int(os.environ.get("DEATH_COOLDOWN_SECONDS", "600"))  # 10 minutes
MAX_CONSECUTIVE_ERRORS_BEFORE_COOLDOWN = int(os.environ.get("MAX_ERRORS", "1"))

# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("Crypto-AI-Bot")

session = requests.Session()

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# =========================
# STATE
# =========================

state = {
    "equity": START_EQUITY,
    "wins": 0,
    "losses": 0,
    "total_trades": 0,
    "total_pnl_usd": 0.0,
    "consecutive_errors": 0,

    # control from API
    "paused": False,
    "pause_reason": "",
    "paused_until_utc": "",

    # adaptive risk (can reduce after deaths)
    "risk_per_trade": RISK_PER_TRADE_DEFAULT,
    "death_count": 0,
    "last_death_reason": "",
    "last_death_utc": "",

    "pet": {
        "name": "TradePet",
        "stage": "egg",        # egg -> active -> etc (you can expand later)
        "health": 100.0,
        "mood": "neutral",
        "hunger": 0.0,
        "growth": 0.0,
        "alive": True,
        "fainted_until_utc": "",
        "survival_mode": "NORMAL",
        "time_utc": utc_now_iso()
    }
}

# =========================
# API HELPERS
# =========================

def api_post(endpoint: str, payload: dict) -> bool:
    url = f"{API_URL}{endpoint}"
    try:
        r = session.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            return True
        log.warning(f"POST {endpoint} failed: {r.status_code} {r.text[:200]}")
        return False
    except Exception as e:
        log.warning(f"POST {endpoint} error: {e}")
        return False

def api_get(endpoint: str):
    url = f"{API_URL}{endpoint}"
    try:
        r = session.get(url, timeout=10)
        if r.status_code != 200:
            log.warning(f"GET {endpoint} failed: {r.status_code} {r.text[:200]}")
            return None
        return r.json()
    except Exception as e:
        log.warning(f"GET {endpoint} error: {e}")
        return None

def ingest_event(msg: str, type_: str = "info", details: dict | None = None):
    api_post("/ingest/event", {
        "time_utc": utc_now_iso(),
        "type": type_,
        "message": msg,
        "details": details or {}
    })

def ingest_death(reason: str, details: dict | None = None):
    # API must implement /ingest/death for this to persist (safe if missing)
    api_post("/ingest/death", {
        "time_utc": utc_now_iso(),
        "reason": reason,
        "details": details or {}
    })

def fetch_control():
    """
    Expected API /control response (example):
    {
      "paused": false,
      "pause_reason": "",
      "paused_until_utc": ""
    }
    """
    data = api_get("/control")
    if not data:
        return
    state["paused"] = bool(data.get("paused", False))
    state["pause_reason"] = str(data.get("pause_reason", "") or "")
    state["paused_until_utc"] = str(data.get("paused_until_utc", "") or "")

# =========================
# PET LOGIC
# =========================

def set_pet_egg_recovering(until_utc: str, reason: str):
    pet = state["pet"]
    pet["stage"] = "egg"
    pet["mood"] = "recovering"
    pet["alive"] = True
    pet["survival_mode"] = "PAUSED"
    pet["fainted_until_utc"] = until_utc
    pet["time_utc"] = utc_now_iso()
    ingest_event("🥚 Pet reverted to egg for recovery pause.", "warn", {
        "until_utc": until_utc,
        "reason": reason
    })

def revive_pet_ready():
    pet = state["pet"]
    pet["stage"] = "active"
    pet["mood"] = "neutral"
    pet["health"] = 100.0
    pet["hunger"] = 0.0
    pet["alive"] = True
    pet["survival_mode"] = "NORMAL"
    pet["fainted_until_utc"] = ""
    pet["time_utc"] = utc_now_iso()
    ingest_event("✅ Recovery complete. Pet back to active.", "info", {})

def update_pet_after_trade(pnl: float):
    pet = state["pet"]

    pet["stage"] = pet.get("stage") or "active"
    if pet["stage"] == "egg":
        pet["stage"] = "active"

    pet["hunger"] = float(pet.get("hunger", 0.0)) + 5.0

    if pnl > 0:
        pet["health"] = min(100.0, float(pet["health"]) + 5.0)
        pet["mood"] = "happy"
        pet["hunger"] = max(0.0, float(pet["hunger"]) - 10.0)
        pet["growth"] = float(pet.get("growth", 0.0)) + 1.0
    else:
        pet["health"] = float(pet["health"]) - 5.0
        pet["mood"] = "sad"
        pet["growth"] = max(0.0, float(pet.get("growth", 0.0)) - 0.5)

    if float(pet["hunger"]) > 80.0:
        pet["health"] = float(pet["health"]) - 5.0
        pet["mood"] = "hungry"

    pet["alive"] = float(pet["health"]) > 0.0
    pet["survival_mode"] = "NORMAL"
    pet["time_utc"] = utc_now_iso()

# =========================
# TRADING LOGIC (SAFE SIM)
# =========================

def simulate_trade():
    """
    Returns a dict containing trade info.
    """
    risk = max(1.0, state["equity"] * state["risk_per_trade"])
    market = "BTCUSDT"
    side = "buy" if random.random() > 0.5 else "sell"
    confidence = round(random.uniform(0.35, 0.95), 2)

    # Slight edge/variance
    win = random.random() > 0.45
    pnl = (risk * random.uniform(0.8, 1.5)) if win else (-risk)

    price = random.uniform(30000, 60000)
    size_usd = risk  # simple sizing

    reason = "simulated"
    if confidence < 0.45:
        reason = "low_confidence_sim"
    elif confidence > 0.8:
        reason = "high_confidence_sim"

    return {
        "market": market,
        "side": side,
        "confidence": confidence,
        "price": price,
        "size_usd": size_usd,
        "pnl_usd": pnl,
        "reason": reason,
        "time_utc": utc_now_iso(),
    }

def apply_trade_result(trade: dict):
    pnl = float(trade["pnl_usd"])
    state["equity"] += pnl
    state["total_trades"] += 1
    state["total_pnl_usd"] += pnl

    if pnl > 0:
        state["wins"] += 1
    else:
        state["losses"] += 1

# =========================
# SURVIVAL / COOLDOWN
# =========================

def in_local_cooldown() -> bool:
    until = state["pet"].get("fainted_until_utc") or ""
    if not until:
        return False
    try:
        until_dt = datetime.fromisoformat(until.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) < until_dt
    except Exception:
        return False

def start_cooldown(reason: str, extra: dict | None = None):
    # record death
    state["death_count"] += 1
    state["last_death_reason"] = reason
    state["last_death_utc"] = utc_now_iso()

    # “learn”: reduce risk a bit after each death, floor at 0.002 (0.2%)
    state["risk_per_trade"] = max(0.002, state["risk_per_trade"] * 0.8)

    until_dt = datetime.now(timezone.utc) + timedelta_seconds(DEATH_COOLDOWN_SECONDS)
    until_utc = until_dt.isoformat()

    ingest_event("💀 Pet died / bot failure detected. Entering 10 min recovery pause.", "error", {
        "reason": reason,
        "cooldown_seconds": DEATH_COOLDOWN_SECONDS,
        "new_risk_per_trade": state["risk_per_trade"],
        "extra": extra or {}
    })
    ingest_death(reason, {
        "cooldown_seconds": DEATH_COOLDOWN_SECONDS,
        "new_risk_per_trade": state["risk_per_trade"],
        **(extra or {})
    })

    set_pet_egg_recovering(until_utc, reason)

def timedelta_seconds(seconds: int):
    from datetime import timedelta
    return timedelta(seconds=seconds)

# =========================
# MAIN LOOP
# =========================

def send_ingest_payloads(trade: dict | None, status: str):
    # heartbeat
    api_post("/ingest/heartbeat", {
        "time_utc": utc_now_iso(),
        "status": status,
        "survival_mode": state["pet"].get("survival_mode", "NORMAL"),
        "equity_usd": state["equity"],
        "open_positions": 0,
        "prices_ok": True,
        "markets": ["BTCUSDT"],
        "losses": state["losses"],
        "total_trades": state["total_trades"],
        "wins": state["wins"],
        "total_pnl_usd": state["total_pnl_usd"],
    })

    # pet
    api_post("/ingest/pet", state["pet"])

    # equity timeline
    api_post("/ingest/equity", {
        "time_utc": utc_now_iso(),
        "equity_usd": state["equity"]
    })

    # trade
    if trade is not None:
        api_post("/ingest/trade", {
            "time_utc": trade["time_utc"],
            "market": trade["market"],
            "side": trade["side"],
            "size_usd": trade["size_usd"],
            "price": trade["price"],
            "pnl_usd": trade["pnl_usd"],
            "reason": trade["reason"],
            "confidence": trade["confidence"],
        })

    # prices
    api_post("/ingest/prices", {
        "time_utc": utc_now_iso(),
        "prices": {
            "BTCUSDT": random.uniform(30000, 60000)
        }
    })

def run():
    log.info("🚀 Crypto-AI-Bot started")
    ingest_event("🚀 Crypto-AI-Bot started", "info", {
        "cycle_seconds": CYCLE_SECONDS,
        "start_equity": START_EQUITY
    })

    # set initial pet state
    state["pet"]["stage"] = "active"
    state["pet"]["time_utc"] = utc_now_iso()
    api_post("/ingest/pet", state["pet"])

    while True:
        try:
            # 1) Check remote control (pause/resume)
            fetch_control()

            # 2) If API says paused, honour it
            if state["paused"]:
                state["pet"]["stage"] = "egg"
                state["pet"]["mood"] = "paused"
                state["pet"]["survival_mode"] = "PAUSED"
                state["pet"]["time_utc"] = utc_now_iso()

                send_ingest_payloads(trade=None, status="paused")
                log.info(f"⏸️ PAUSED by control. Reason: {state['pause_reason']}")
                time.sleep(CYCLE_SECONDS)
                continue

            # 3) If we’re in local cooldown after a death/crash, do nothing but report
            if in_local_cooldown():
                state["pet"]["survival_mode"] = "PAUSED"
                state["pet"]["time_utc"] = utc_now_iso()
                send_ingest_payloads(trade=None, status="cooldown")
                log.info("🥚 In recovery cooldown (egg). No trading this cycle.")
                time.sleep(CYCLE_SECONDS)
                continue
            else:
                # if cooldown ended and we were egg, revive
                if state["pet"].get("stage") == "egg" and state["pet"].get("fainted_until_utc"):
                    revive_pet_ready()

            # 4) Trade (safe sim)
            trade = simulate_trade()
            apply_trade_result(trade)
            update_pet_after_trade(trade["pnl_usd"])

            # 5) If pet “dies”, trigger cooldown + death logging (instead of halting forever)
            if not state["pet"]["alive"]:
                start_cooldown("pet_health_zero", {
                    "equity": state["equity"],
                    "total_trades": state["total_trades"],
                    "wins": state["wins"],
                    "losses": state["losses"]
                })
                send_ingest_payloads(trade=None, status="cooldown")
                time.sleep(CYCLE_SECONDS)
                continue

            # 6) Normal ingest
            send_ingest_payloads(trade=trade, status="running")

            log.info(
                f"Trade #{state['total_trades']} | "
                f"{trade['market']} {trade['side']} | "
                f"PnL: {trade['pnl_usd']:.2f} | "
                f"Equity: {state['equity']:.2f} | "
                f"HP: {state['pet']['health']:.1f} | "
                f"Risk: {state['risk_per_trade']:.4f}"
            )

            state["consecutive_errors"] = 0
            time.sleep(CYCLE_SECONDS)

        except Exception as e:
            state["consecutive_errors"] += 1
            tb = traceback.format_exc()

            log.error(f"💥 Crash in main loop: {e}")
            ingest_event("💥 Crash in main loop", "error", {
                "error": str(e),
                "traceback": tb[:4000],  # keep it bounded
                "consecutive_errors": state["consecutive_errors"],
            })

            # Put into cooldown so it “reverts to egg” and pauses to “learn”
            if state["consecutive_errors"] >= MAX_CONSECUTIVE_ERRORS_BEFORE_COOLDOWN:
                start_cooldown("exception_crash", {
                    "error": str(e),
                    "traceback": tb[:4000],
                    "consecutive_errors": state["consecutive_errors"],
                })

            # still report status while cooling down
            send_ingest_payloads(trade=None, status="error")
            time.sleep(CYCLE_SECONDS)

# =========================
# ENTRY
# =========================

if __name__ == "__main__":
    run()

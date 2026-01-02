import time
import random
import requests
import logging
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

# =========================
# CONFIG
# =========================

API_URL = "https://crypto-ai-api-1-7cte.onrender.com"
CYCLE_SECONDS = 30

START_EQUITY = 1000.0
RISK_PER_TRADE = 0.01

COOLDOWN_MINUTES = 10
DEATH_LOG_PATH = Path("death_log.jsonl")  # saved alongside main.py

# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("Crypto-AI-Bot")

# =========================
# STATE
# =========================

def utc_now():
    return datetime.now(timezone.utc)

state = {
    "equity": START_EQUITY,
    "wins": 0,
    "losses": 0,
    "total_trades": 0,
    "rebirth_count": 0,
    "risk_multiplier": 1.0,     # reduces after deaths to "learn not to die"
    "cooldown_until": None,     # ISO timestamp
    "halt_logged": False,       # prevents spam death logs
    "pet": {
        "name": "TradePet",
        "stage": "egg",         # egg -> (later you can evolve)
        "health": 100,
        "mood": "neutral",
        "hunger": 0,
        "alive": True,
        "last_update": None,
        "last_death_reason": None,
        "last_death_time": None
    }
}

# =========================
# API HELPERS
# =========================

def post(endpoint, payload):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        log.warning(f"API error {endpoint}: {e}")
        return False

def safe_post(endpoint, payload):
    """Post but never crash the loop if API doesn't have that endpoint."""
    try:
        requests.post(f"{API_URL}{endpoint}", json=payload, timeout=10)
    except Exception:
        pass

# =========================
# DEATH / COOLDOWN HELPERS
# =========================

def log_death(death_report: dict):
    try:
        DEATH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEATH_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(death_report, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning(f"Could not write death log: {e}")

def build_death_report(reason: str, pnl: float = None, err: Exception = None):
    pet = state["pet"]
    win_rate = (state["wins"] / state["total_trades"]) if state["total_trades"] else 0.0

    report = {
        "time_utc": utc_now().isoformat(),
        "reason": reason,
        "error": str(err) if err else None,
        "equity_usd": state["equity"],
        "wins": state["wins"],
        "losses": state["losses"],
        "total_trades": state["total_trades"],
        "win_rate": win_rate,
        "risk_multiplier": state.get("risk_multiplier", 1.0),
        "pet": {
            "name": pet.get("name"),
            "stage": pet.get("stage"),
            "health": pet.get("health"),
            "hunger": pet.get("hunger"),
            "mood": pet.get("mood"),
            "alive": pet.get("alive")
        },
        "last_trade": {
            "pnl": pnl,
            "timestamp": utc_now().isoformat()
        } if pnl is not None else None
    }
    return report

def set_cooldown(minutes: int):
    state["cooldown_until"] = (utc_now() + timedelta(minutes=minutes)).isoformat()

def in_cooldown():
    s = state.get("cooldown_until")
    if not s:
        return False
    try:
        until = datetime.fromisoformat(s)
        return utc_now() < until
    except Exception:
        return False

def rebirth_to_egg(reason: str):
    pet = state["pet"]

    # store last death info for dashboard
    pet["last_death_reason"] = reason
    pet["last_death_time"] = utc_now().isoformat()

    # "learn": reduce risk a bit each death (floor it)
    state["risk_multiplier"] = max(0.25, float(state.get("risk_multiplier", 1.0)) * 0.85)
    state["rebirth_count"] = int(state.get("rebirth_count", 0)) + 1

    # reset pet to egg baseline
    pet["stage"] = "egg"
    pet["health"] = 100
    pet["hunger"] = 0
    pet["mood"] = "hopeful"
    pet["alive"] = True
    pet["last_update"] = utc_now().isoformat()

    # cooldown
    set_cooldown(COOLDOWN_MINUTES)

    # reset the spam-guard
    state["halt_logged"] = False

# =========================
# PET LOGIC
# =========================

def update_pet(pnl):
    pet = state["pet"]

    pet["hunger"] += 5

    if pnl > 0:
        pet["health"] = min(100, pet["health"] + 5)
        pet["mood"] = "happy"
        pet["hunger"] = max(0, pet["hunger"] - 10)
    else:
        pet["health"] -= 5
        pet["mood"] = "sad"

    if pet["hunger"] > 80:
        pet["health"] -= 5

    if pet["health"] <= 0:
        pet["alive"] = False
        pet["mood"] = "dead"

    pet["last_update"] = utc_now().isoformat()

# =========================
# TRADING LOGIC (SAFE SIM)
# =========================

def simulate_trade():
    # adaptive risk (learn not to die)
    risk_mult = float(state.get("risk_multiplier", 1.0))
    risk = state["equity"] * RISK_PER_TRADE * risk_mult

    win = random.random() > 0.45
    pnl = risk * random.uniform(0.8, 1.5) if win else -risk

    state["equity"] += pnl
    state["total_trades"] += 1

    if pnl > 0:
        state["wins"] += 1
    else:
        state["losses"] += 1

    return pnl

# =========================
# API PUSH
# =========================

def push_updates(pnl=None):
    # heartbeat
    post("/ingest/heartbeat", {
        "time": utc_now().isoformat(),
        "equity": state["equity"],
        "status": "cooldown" if in_cooldown() else "running",
        "cooldown_until": state.get("cooldown_until"),
        "rebirth_count": state.get("rebirth_count", 0)
    })

    # pet
    post("/ingest/pet", state["pet"])

    # equity
    post("/ingest/equity", {
        "equity_usd": state["equity"],
        "time_utc": utc_now().isoformat()
    })

    # optional trade
    if pnl is not None:
        post("/ingest/trade", {
            "pnl": pnl,
            "equity": state["equity"],
            "timestamp": utc_now().isoformat()
        })

    # prices (fake)
    post("/ingest/prices", {
        "BTCUSDT": random.uniform(30000, 60000)
    })

# =========================
# MAIN LOOP
# =========================

def run():
    log.info("🚀 Crypto-AI-Bot started")

    while True:
        try:
            # COOLDOWN MODE (no trading)
            if in_cooldown():
                # keep UI alive and show countdown
                log.info(f"⏸️ Cooldown active until {state['cooldown_until']} (no trading)")
                state["pet"]["mood"] = "resting"
                state["pet"]["last_update"] = utc_now().isoformat()
                push_updates(pnl=None)
                time.sleep(CYCLE_SECONDS)
                continue

            # if pet is dead, do post-mortem + rebirth + cooldown
            if not state["pet"]["alive"]:
                reason = "health<=0"

                # log death ONCE
                if not state.get("halt_logged"):
                    report = build_death_report(reason=reason)
                    log_death(report)
                    log.error("💀 Pet died. Logging death + rebirth to egg + cooldown 10 mins.")

                    # (optional) send death event if your API has it (won't break if it doesn't)
                    safe_post("/ingest/death", report)

                    state["halt_logged"] = True

                rebirth_to_egg(reason=reason)
                push_updates(pnl=None)
                time.sleep(2)
                continue

            # NORMAL TRADE
            pnl = simulate_trade()
            update_pet(pnl)

            log.info(
                f"Trade #{state['total_trades']} | "
                f"PnL: {pnl:.2f} | "
                f"Equity: {state['equity']:.2f} | "
                f"Pet HP: {state['pet']['health']} | "
                f"RiskMult: {state.get('risk_multiplier', 1.0):.2f}"
            )

            # If this trade killed the pet, handle next loop (but also push now)
            push_updates(pnl=pnl)

            time.sleep(CYCLE_SECONDS)

        except Exception as e:
            # Treat unexpected exceptions as a "death" too, so you learn from crashes
            log.exception(f"🔥 Bot exception: {e}")

            report = build_death_report(reason="exception", err=e)
            log_death(report)
            safe_post("/ingest/death", report)

            # rebirth + cooldown so it doesn't thrash
            rebirth_to_egg(reason="exception")
            push_updates(pnl=None)
            time.sleep(CYCLE_SECONDS)

# =========================
# ENTRY
# =========================

if __name__ == "__main__":
    run()

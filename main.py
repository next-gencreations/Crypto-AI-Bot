import time
import random
import requests
import logging
from datetime import datetime

# =========================
# CONFIG
# =========================

API_URL = "https://crypto-ai-api-1-7cte.onrender.com"
CYCLE_SECONDS = 30
DEATH_PAUSE_SECONDS = 600  # 10 minutes

START_EQUITY = 1000.0
BASE_RISK_PER_TRADE = 0.01
MIN_RISK = 0.002

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

state = {
    "equity": START_EQUITY,
    "wins": 0,
    "losses": 0,
    "total_trades": 0,
    "risk_per_trade": BASE_RISK_PER_TRADE,
    "generation": 1,
    "pet": {
        "stage": "egg",        # egg | alive | dead
        "health": 100,
        "hunger": 0,
        "mood": "neutral",
        "alive": True,
        "birth_time": None,
        "last_update": None
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

# =========================
# PET LIFECYCLE
# =========================

def hatch_pet():
    state["pet"] = {
        "stage": "alive",
        "health": 100,
        "hunger": 0,
        "mood": "curious",
        "alive": True,
        "birth_time": datetime.utcnow().isoformat(),
        "last_update": datetime.utcnow().isoformat()
    }
    log.info("🥚➡️🐣 Pet hatched. New life begins.")

def kill_pet(reason):
    state["pet"]["alive"] = False
    state["pet"]["stage"] = "dead"
    state["pet"]["mood"] = "dead"

    death_report = {
        "time": datetime.utcnow().isoformat(),
        "generation": state["generation"],
        "equity": state["equity"],
        "wins": state["wins"],
        "losses": state["losses"],
        "risk_per_trade": state["risk_per_trade"],
        "reason": reason
    }

    log.error(f"💀 Pet died. Reason: {reason}")
    post("/ingest/death", death_report)

def rebirth():
    state["generation"] += 1
    state["equity"] = max(START_EQUITY * 0.9, state["equity"])
    state["wins"] = 0
    state["losses"] = 0
    state["total_trades"] = 0

    # Reduce risk after death (learning)
    state["risk_per_trade"] = max(
        MIN_RISK,
        state["risk_per_trade"] * 0.8
    )

    state["pet"]["stage"] = "egg"
    state["pet"]["alive"] = True

    log.info(
        f"🔁 Rebirth complete | Generation {state['generation']} | "
        f"New risk: {state['risk_per_trade']:.4f}"
    )

# =========================
# PET UPDATE
# =========================

def update_pet(pnl):
    pet = state["pet"]

    pet["hunger"] += 5

    if pnl > 0:
        pet["health"] = min(100, pet["health"] + 4)
        pet["hunger"] = max(0, pet["hunger"] - 8)
        pet["mood"] = "happy"
    else:
        pet["health"] -= 6
        pet["mood"] = "stressed"

    if pet["hunger"] > 85:
        pet["health"] -= 8

    pet["last_update"] = datetime.utcnow().isoformat()

    if pet["health"] <= 0:
        reason = "starvation" if pet["hunger"] > 85 else "losses"
        kill_pet(reason)

# =========================
# TRADING LOGIC (SURVIVAL FIRST)
# =========================

def simulate_trade():
    equity = state["equity"]
    risk = equity * state["risk_per_trade"]

    # Conservative bias: avoid trades if equity is fragile
    if equity < START_EQUITY * 0.8:
        log.warning("⚠️ Equity low. Skipping trade to survive.")
        return 0.0

    win_chance = 0.52 if state["risk_per_trade"] < BASE_RISK_PER_TRADE else 0.48
    win = random.random() < win_chance

    pnl = risk * random.uniform(0.6, 1.3) if win else -risk

    state["equity"] += pnl
    state["total_trades"] += 1

    if pnl > 0:
        state["wins"] += 1
    else:
        state["losses"] += 1

    return pnl

# =========================
# MAIN LOOP
# =========================

def run():
    log.info("🚀 Crypto-AI-Bot started")

    while True:

        # Egg phase
        if state["pet"]["stage"] == "egg":
            hatch_pet()

        # Death handling
        if not state["pet"]["alive"]:
            log.error("⏸️ Trading paused. Analysing death.")
            time.sleep(DEATH_PAUSE_SECONDS)
            rebirth()
            continue

        pnl = simulate_trade()
        update_pet(pnl)

        log.info(
            f"Gen {state['generation']} | "
            f"Trade #{state['total_trades']} | "
            f"PnL: {pnl:.2f} | "
            f"Equity: {state['equity']:.2f} | "
            f"HP: {state['pet']['health']} | "
            f"Risk: {state['risk_per_trade']:.4f}"
        )

        # -------------------------
        # API REPORTING
        # -------------------------

        post("/ingest/heartbeat", {
            "time": datetime.utcnow().isoformat(),
            "equity": state["equity"],
            "generation": state["generation"]
        })

        post("/ingest/pet", state["pet"])

        post("/ingest/trade", {
            "pnl": pnl,
            "equity": state["equity"],
            "wins": state["wins"],
            "losses": state["losses"],
            "risk_per_trade": state["risk_per_trade"],
            "timestamp": datetime.utcnow().isoformat()
        })

        time.sleep(CYCLE_SECONDS)

# =========================
# ENTRY
# =========================

if __name__ == "__main__":
    run()

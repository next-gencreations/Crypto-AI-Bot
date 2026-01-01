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

START_EQUITY = 1000.0
RISK_PER_TRADE = 0.01

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
    "pet": {
        "name": "TradePet",
        "health": 100,
        "mood": "neutral",
        "hunger": 0,
        "alive": True,
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

    pet["last_update"] = datetime.utcnow().isoformat()

# =========================
# TRADING LOGIC (SAFE SIM)
# =========================

def simulate_trade():
    risk = state["equity"] * RISK_PER_TRADE
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
# MAIN LOOP
# =========================

def run():
    log.info("🚀 Crypto-AI-Bot started")

    while True:
        if not state["pet"]["alive"]:
            log.error("💀 Pet died. Trading halted.")
            time.sleep(60)
            continue

        pnl = simulate_trade()
        update_pet(pnl)

        log.info(
            f"Trade #{state['total_trades']} | "
            f"PnL: {pnl:.2f} | "
            f"Equity: {state['equity']:.2f} | "
            f"Pet HP: {state['pet']['health']}"
        )

        # -------------------------
        # SEND DATA TO API
        # -------------------------

        post("/ingest/heartbeat", {
            "time": datetime.utcnow().isoformat(),
            "equity": state["equity"]
        })

        post("/ingest/pet", state["pet"])

        post("/ingest/equity", {
            "equity_usd": state["equity"]
        })

        post("/ingest/trade", {
            "pnl": pnl,
            "equity": state["equity"],
            "timestamp": datetime.utcnow().isoformat()
        })

        post("/ingest/prices", {
            "BTCUSDT": random.uniform(30000, 60000)
        })

        time.sleep(CYCLE_SECONDS)

# =========================
# ENTRY
# =========================

if __name__ == "__main__":
    run()

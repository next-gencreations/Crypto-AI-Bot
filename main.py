# main.py
import os
import time
import random
import requests

from brain_v2 import BrainV2

API_BASE = os.getenv("API_BASE", "https://crypto-ai-api-1-7cte.onrender.com").rstrip("/")
PIN = os.getenv("VAULT_PIN", "4567")

SLEEP_SEC = float(os.getenv("BOT_SLEEP_SEC", "5"))

def unlock_vault():
    r = requests.post(f"{API_BASE}/vault/unlock", json={"pin": PIN}, timeout=15)
    r.raise_for_status()
    j = r.json()
    if not j.get("ok"):
        raise RuntimeError(f"Vault unlock failed: {j}")
    return j["token"]

def get_data():
    r = requests.get(f"{API_BASE}/data", timeout=15)
    r.raise_for_status()
    return r.json()

def post_heartbeat(open_positions, equity_usd, last_pnl_usd):
    requests.post(
        f"{API_BASE}/bot/heartbeat",
        json={
            "open_positions": open_positions,
            "equity_usd": equity_usd,
            "last_pnl_usd": last_pnl_usd,
        },
        timeout=15
    )

def post_paper_event(pnl_usd, reason="paper_trade"):
    requests.post(
        f"{API_BASE}/paper/event",
        json={"pnl_usd": pnl_usd, "reason": reason},
        timeout=15
    )

def simulate_signal():
    # placeholder signal: -1..+1
    return random.uniform(-1, 1)

def simulate_trade_outcome(signal_score, risk_mult):
    """
    Very simple paper sim:
    - stronger signals win more often
    - risk_mult scales pnl size
    """
    strength = abs(signal_score)
    win_prob = 0.45 + 0.40 * strength  # 0.45..0.85
    win = random.random() < win_prob

    base = 5.0 + 20.0 * strength  # $5..$25
    pnl = base * risk_mult
    return pnl if win else -pnl

def main():
    print("Starting Crypto-AI-Bot (paper mode)")
    brain = BrainV2()

    # (Optional) unlock to prove vault works; bot itself doesn't need token for /data
    try:
        token = unlock_vault()
        print("Vault unlocked OK (token acquired)")
    except Exception as e:
        print("Vault unlock failed (continuing anyway):", e)
        token = None

    equity_usd = 0.0
    open_positions = 0
    last_pnl = 0.0

    while True:
        try:
            data = get_data()
            pet = data.get("pet") or {}
            settings = (data.get("settings") or {})

            # companion influence
            signal = simulate_signal()

            decision = brain.decide(signal_score=signal, companion=pet)

            if decision.allow_trade:
                # paper trade
                pnl = simulate_trade_outcome(signal, decision.risk_multiplier)
                last_pnl = pnl
                equity_usd += pnl

                brain.update_result(pnl)

                # feed the companion system
                post_paper_event(pnl_usd=pnl, reason=decision.reason)

                print(f"TRADE: signal={signal:+.2f} conf={decision.confidence:.2f} "
                      f"risk={decision.risk_multiplier:.2f} pnl={pnl:+.2f} equity={equity_usd:+.2f} "
                      f"pet(h={pet.get('health')},hu={pet.get('hunger')},m={pet.get('mood')})")
            else:
                last_pnl = 0.0
                print(f"SKIP: signal={signal:+.2f} conf={decision.confidence:.2f} reason={decision.reason} "
                      f"pet(h={pet.get('health')},hu={pet.get('hunger')},m={pet.get('mood')})")

            # heartbeat for dashboard
            post_heartbeat(open_positions=open_positions, equity_usd=equity_usd, last_pnl_usd=last_pnl)

        except Exception as e:
            print("Loop error:", e)

        time.sleep(SLEEP_SEC)

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Crypto-AI-Bot (paper mode)
Universe scanning via Coinbase public API

Features:
- Live prices from Coinbase
- Momentum signal engine
- BrainV2 gate + risk multiplier
- Vault unlock
- Pet pressure system
- Dashboard heartbeat + paper events
"""

import os
import time
import math
import random
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
import requests

# ===================== ENV =====================

API_BASE = os.getenv("API_BASE", "https://crypto-ai-api-1-7cte.onrender.com")
VAULT_PIN = os.getenv("VAULT_PIN", "4567")

COINBASE_BASE = "https://api.exchange.coinbase.com"

CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS", "15"))
HISTORY_LEN = int(os.getenv("HISTORY_LEN", "30"))

RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.015"))
MAX_RISK_MULT = float(os.getenv("MAX_RISK_MULT", "1.50"))
MIN_CONF = float(os.getenv("MIN_CONF", "0.20"))

USE_BRAIN = os.getenv("USE_BRAIN", "1") != "0"

DEFAULT_UNIVERSE = (
    "BTC-USD","ETH-USD","SOL-USD","ADA-USD","XRP-USD",
    "DOGE-USD","AVAX-USD","LINK-USD","DOT-USD",
    "LTC-USD","BCH-USD","UNI-USD","ARB-USD","OP-USD","POL-USD"
)

raw_uni = os.getenv("UNIVERSE")
if raw_uni:
    UNIVERSE = [x.strip().upper() for x in raw_uni.split(",")]
else:
    UNIVERSE = list(DEFAULT_UNIVERSE)

# ===================== LOGGING =====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("bot")

# ===================== HELPERS =====================

def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def safe_get_json(url, headers=None, timeout=10):
    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"GET failed: {url} :: {e}")
        return None

def safe_post_json(url, payload, headers=None, timeout=10):
    try:
        r = requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
        r.raise_for_status()
        return r.json() if r.text else {"ok": True}
    except Exception as e:
        log.warning(f"POST failed: {url} :: {e}")
        return None

# ===================== VAULT + DASHBOARD =====================

def unlock_vault():
    j = safe_post_json(f"{API_BASE}/vault/unlock", {"pin": VAULT_PIN})
    if not j or not j.get("ok"):
        return None
    return j.get("token")

def get_data():
    return safe_get_json(f"{API_BASE}/data", timeout=15)

def post_heartbeat(open_positions, equity, last_pnl):
    safe_post_json(
        f"{API_BASE}/bot/heartbeat",
        {
            "open_positions": open_positions,
            "equity_usd": equity,
            "last_pnl_usd": last_pnl,
            "time_utc": utc_now(),
        },
    )

def post_paper_event(pnl, reason, market, side, confidence, risk_mult):
    safe_post_json(
        f"{API_BASE}/paper/event",
        {
            "pnl_usd": float(pnl),
            "reason": reason,
            "market": market,
            "side": side,
            "confidence": float(confidence),
            "risk_mult": float(risk_mult),
            "time_utc": utc_now(),
        },
    )

# ===================== COINBASE =====================

_CB_HEADERS = {"User-Agent": "Crypto-AI-Bot/1.0"}

def fetch_coinbase_price(product_id):
    j = safe_get_json(f"{COINBASE_BASE}/products/{product_id}/ticker", headers=_CB_HEADERS)
    if not j:
        return None
    try:
        return float(j.get("price"))
    except:
        return None

# ===================== PRICE STORE =====================

class PriceStore:
    def __init__(self, history_len):
        self.history_len = history_len
        self.prices: Dict[str, List[float]] = {}

    def add(self, market, price):
        arr = self.prices.setdefault(market, [])
        arr.append(float(price))
        if len(arr) > self.history_len:
            del arr[0:len(arr) - self.history_len]

    def ready(self, market, min_len=8):
        return len(self.prices.get(market, [])) >= min_len

    def momentum_signal(self, market):
        arr = self.prices.get(market, [])
        if len(arr) < 8:
            return 0.0

        n = len(arr)
        short = max(3, n // 4)
        long = max(6, n // 2)

        ma_s = sum(arr[-short:]) / short
        ma_l = sum(arr[-long:]) / long

        if ma_l <= 0:
            return 0.0

        raw = (ma_s - ma_l) / ma_l
        return clamp(raw * 80.0, -1.0, 1.0)

# ===================== PAPER SIM =====================

def simulate_pnl(risk_usd, confidence, side, signal):
    conf = clamp(abs(confidence), 0.0, 1.0)
    win_prob = 0.48 + 0.22 * conf
    win = random.random() < win_prob

    alignment = 1.0
    if (side == "BUY" and signal > 0) or (side == "SELL" and signal < 0):
        alignment = 1.05

    if win:
        return abs(risk_usd) * random.uniform(0.4, 1.2) * alignment
    else:
        return -abs(risk_usd) * random.uniform(0.4, 1.1)

# ===================== PET PRESSURE =====================

def pet_pressure(data):
    try:
        pet = (data or {}).get("pet") or {}
        h = float(pet.get("health", 100))
        hu = float(pet.get("hunger", 100))
    except:
        return 1.0

    pressure = 1.0
    if h < 60:
        pressure += (60 - h) / 200.0
    if hu < 60:
        pressure += (60 - hu) / 250.0

    return clamp(pressure, 1.0, 1.35)

# ===================== MAIN LOOP =====================

def run():
    log.info("Starting Crypto-AI-Bot (paper mode) - Coinbase universe")
    token = unlock_vault()
    if token:
        log.info("Vault unlocked OK")
    else:
        log.warning("Vault unlock failed (continuing anyway)")

    store = PriceStore(HISTORY_LEN)
    equity = 1000.0
    last_pnl = 0.0

    d0 = get_data()
    try:
        eq0 = float((d0 or {}).get("equity", {}).get("usd", equity))
        if math.isfinite(eq0) and eq0 > 0:
            equity = eq0
    except:
        pass

    while True:
        data = get_data()
        pressure = pet_pressure(data)

        prices = {}
        ok = 0
        for m in UNIVERSE:
            p = fetch_coinbase_price(m)
            if p is None:
                continue
            prices[m] = p
            store.add(m, p)
            ok += 1

        if ok == 0:
            log.warning("No prices fetched")
            post_heartbeat(0, equity, last_pnl)
            time.sleep(CYCLE_SECONDS)
            continue

        best_market = None
        best_signal = 0.0

        for m in UNIVERSE:
            if not store.ready(m):
                continue
            s = store.momentum_signal(m)
            if abs(s) > abs(best_signal):
                best_signal = s
                best_market = m

        if not best_market:
            log.info("Warming up price history...")
            post_heartbeat(0, equity, last_pnl)
            time.sleep(CYCLE_SECONDS)
            continue

        confidence = clamp(abs(best_signal), 0.0, 1.0)
        side = "BUY" if best_signal > 0 else "SELL"

        if confidence < MIN_CONF:
            log.info(f"SKIP low confidence {confidence:.2f}")
            post_heartbeat(0, equity, last_pnl)
            time.sleep(CYCLE_SECONDS)
            continue

        risk_mult = clamp(pressure, 0.25, MAX_RISK_MULT)

        risk_usd = equity * RISK_PER_TRADE * risk_mult
        pnl = simulate_pnl(risk_usd, confidence, side, best_signal)
        equity += pnl
        last_pnl = pnl

        log.info(
            f"TRADE {best_market} {side} "
            f"conf={confidence:.2f} "
            f"risk_mult={risk_mult:.2f} "
            f"pnl={pnl:+.2f} "
            f"equity={equity:.2f}"
        )

        post_paper_event(
            pnl=pnl,
            reason="universe_trade",
            market=best_market,
            side=side,
            confidence=confidence,
            risk_mult=risk_mult,
        )

        post_heartbeat(0, equity, last_pnl)
        time.sleep(CYCLE_SECONDS)

if __name__ == "__main__":
    run()

cat > main.py <<'PY'
# -*- coding: utf-8 -*-
"""
Crypto-AI-Bot (paper mode) - Option A: Universe scanning via Coinbase public prices

- No dependency on API /signal endpoint (avoids your 404 issue)
- Scans many coins, picks best by simple momentum score
- Feeds confidence into BrainV2 (gate + risk multiplier)
- Posts heartbeat + paper events to your Render API so Vault/Dashboard updates

ENV:
  API_BASE        default: https://crypto-ai-api-1-7cte.onrender.com
  VAULT_PIN       default: 4567
  CYCLE_SECONDS   default: 15
  UNIVERSE        comma list, default provided
  HISTORY_LEN     default: 30
  RISK_PER_TRADE  default: 0.015 (1.5% of equity)
  MAX_RISK_MULT   default: 1.50
  MIN_CONF        default: 0.20 (below this = no trade)
  USE_BRAIN       default: 1  (0 disables BrainV2 gate for testing)

Notes:
- Uses Coinbase Exchange public endpoints:
    https://api.exchange.coinbase.com/products/<PRODUCT>/ticker
  where PRODUCT like BTC-USD, ETH-USD etc
"""

import os
import time
import math
import random
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

# ----------------------------
# Brain V2 import
# ----------------------------
try:
    from brain_v2 import BrainV2
except Exception as e:
    BrainV2 = None
    _brain_import_error = str(e)
else:
    _brain_import_error = ""

# ----------------------------
# Config
# ----------------------------
API_BASE = os.getenv("API_BASE", "https://crypto-ai-api-1-7cte.onrender.com").rstrip("/")
VAULT_PIN = os.getenv("VAULT_PIN", "4567")

COINBASE_BASE = "https://api.exchange.coinbase.com"

CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS", "15"))
HISTORY_LEN = int(os.getenv("HISTORY_LEN", "30"))

RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.015"))
MAX_RISK_MULT = float(os.getenv("MAX_RISK_MULT", "1.50"))
MIN_CONF = float(os.getenv("MIN_CONF", "0.20"))

USE_BRAIN = os.getenv("USE_BRAIN", "1").strip() != "0"

DEFAULT_UNIVERSE = (
    "BTC-USD,ETH-USD,SOL-USD,ADA-USD,XRP-USD,DOGE-USD,AVAX-USD,LINK-USD,DOT-USD,ATOM-USD,"
    "LTC-USD,BCH-USD,UNI-USD,OP-USD,ARB-USD,APT-USD,INJ-USD,NEAR-USD,MATIC-USD,TRX-USD"
)
UNIVERSE = [x.strip().upper() for x in os.getenv("UNIVERSE", DEFAULT_UNIVERSE).split(",") if x.strip()]

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("bot")

# ----------------------------
# Helpers
# ----------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_get_json(url: str, headers: Optional[dict] = None, timeout: int = 10) -> Optional[dict]:
    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"GET failed: {url} :: {e}")
        return None

def safe_post_json(url: str, payload: dict, headers: Optional[dict] = None, timeout: int = 10) -> Optional[dict]:
    try:
        r = requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
        r.raise_for_status()
        return r.json() if r.text else {"ok": True}
    except Exception as e:
        log.warning(f"POST failed: {url} :: {e}")
        return None

# ----------------------------
# Render API calls (Vault + paper)
# ----------------------------
def unlock_vault() -> Optional[str]:
    j = safe_post_json(f"{API_BASE}/vault/unlock", {"pin": VAULT_PIN}, timeout=15)
    if not j or not j.get("ok"):
        return None
    return j.get("token")

def get_data() -> Optional[dict]:
    return safe_get_json(f"{API_BASE}/data", timeout=15)

def post_heartbeat(open_positions: int, equity_usd: float, last_pnl_usd: float) -> None:
    safe_post_json(
        f"{API_BASE}/bot/heartbeat",
        {"open_positions": open_positions, "equity_usd": equity_usd, "last_pnl_usd": last_pnl_usd},
        timeout=10,
    )

def post_paper_event(pnl_usd: float, reason: str, market: str, side: str, confidence: float, risk_mult: float) -> None:
    safe_post_json(
        f"{API_BASE}/paper/event",
        {
            "pnl_usd": float(pnl_usd),
            "reason": reason,
            "market": market,
            "side": side,
            "confidence": float(confidence),
            "risk_mult": float(risk_mult),
            "time_utc": utc_now(),
        },
        timeout=10,
    )

# ----------------------------
# Coinbase price fetch
# ----------------------------
_CB_HEADERS = {"User-Agent": "Crypto-AI-Bot/1.0", "Accept": "application/json"}

def fetch_coinbase_price(product_id: str) -> Optional[float]:
    j = safe_get_json(f"{COINBASE_BASE}/products/{product_id}/ticker", headers=_CB_HEADERS, timeout=10)
    if not j:
        return None
    p = j.get("price")
    try:
        return float(p)
    except Exception:
        return None

# ----------------------------
# Signal engine (simple momentum)
# ----------------------------
class PriceStore:
    def __init__(self, history_len: int):
        self.history_len = history_len
        self.prices: Dict[str, List[float]] = {}

    def add(self, market: str, price: float) -> None:
        arr = self.prices.setdefault(market, [])
        arr.append(float(price))
        if len(arr) > self.history_len:
            del arr[0 : len(arr) - self.history_len]

    def ready(self, market: str, min_len: int = 8) -> bool:
        return len(self.prices.get(market, [])) >= min_len

    def momentum_signal(self, market: str) -> float:
        """
        Returns signal in [-1, +1] using short-vs-long MA slope.
        """
        arr = self.prices.get(market, [])
        if len(arr) < 8:
            return 0.0

        n = len(arr)
        short = max(3, n // 4)
        long = max(6, n // 2)

        ma_s = sum(arr[-short:]) / short
        ma_l = sum(arr[-long:]) / long

        # normalized diff
        if ma_l <= 0:
            return 0.0
        raw = (ma_s - ma_l) / ma_l  # e.g. 0.002 = 0.2%

        # squash into [-1,1]
        return clamp(raw * 80.0, -1.0, 1.0)

# ----------------------------
# Paper outcome model (biased by confidence)
# ----------------------------
def simulate_pnl(risk_usd: float, confidence: float, side: str, signal: float) -> float:
    """
    Paper sim:
      - win probability increases with confidence
      - pnl magnitude scales with risk_usd
    """
    conf = clamp(abs(confidence), 0.0, 1.0)
    win_prob = 0.48 + 0.22 * conf  # 48% .. 70%
    win = (random.random() < win_prob)

    # small edge: if side matches signal direction, tiny boost
    alignment = 1.0
    if (side == "BUY" and signal > 0) or (side == "SELL" and signal < 0):
        alignment = 1.05

    # payout distribution
    if win:
        return abs(risk_usd) * random.uniform(0.4, 1.2) * alignment
    else:
        return -abs(risk_usd) * random.uniform(0.4, 1.1)

# ----------------------------
# Pet pressure (make bot “try harder” when pet is unwell)
# ----------------------------
def pet_pressure(data: Optional[dict]) -> float:
    """
    Returns extra risk multiplier based on pet health/hunger.
    If pet is sick/hungry -> allow slightly more aggression (still capped).
    """
    try:
        pet = (data or {}).get("pet") or {}
        h = float(pet.get("health", 100))
        hu = float(pet.get("hunger", 100))
    except Exception:
        return 1.0

    # if health/hunger drop, pressure increases
    pressure = 1.0
    if h < 60:
        pressure += (60 - h) / 200.0   # up to +0.30
    if hu < 60:
        pressure += (60 - hu) / 250.0  # up to +0.24
    return clamp(pressure, 1.0, 1.35)

# ----------------------------
# Main loop
# ----------------------------
def run() -> None:
    if not UNIVERSE:
        raise RuntimeError("UNIVERSE is empty. Set UNIVERSE env var.")

    if USE_BRAIN and BrainV2 is None:
        log.warning(f"BrainV2 import failed. Running without brain. error={_brain_import_error}")

    brain = BrainV2() if (USE_BRAIN and BrainV2 is not None) else None

    log.info("Starting Crypto-AI-Bot (paper mode) - Coinbase universe scanning")
    token = unlock_vault()
    if token:
        log.info("Vault unlocked OK (token acquired)")
    else:
        log.warning("Vault unlock failed (continuing anyway)")

    store = PriceStore(HISTORY_LEN)

    equity = 1000.0
    last_pnl = 0.0

    # Try to seed equity from API /data if present
    d0 = get_data()
    try:
        eq0 = float((d0 or {}).get("equity", {}).get("usd", equity))
        if math.isfinite(eq0) and eq0 > 0:
            equity = eq0
    except Exception:
        pass

    while True:
        data = get_data()
        pressure = pet_pressure(data)

        # 1) Fetch prices for universe + update history
        prices: Dict[str, float] = {}
        ok_count = 0
        for m in UNIVERSE:
            p = fetch_coinbase_price(m)
            if p is None:
                continue
            prices[m] = p
            store.add(m, p)
            ok_count += 1

        if ok_count == 0:
            log.warning("No prices fetched (network or Coinbase issue). Sleeping.")
            post_heartbeat(0, equity, last_pnl)
            time.sleep(CYCLE_SECONDS)
            continue

        # 2) Compute signal for each market and pick best by abs(signal)
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
            log.info("Warming up price history... (need a few cycles)")
            post_heartbeat(0, equity, last_pnl)
            time.sleep(CYCLE_SECONDS)
            continue

        confidence = clamp(abs(best_signal), 0.0, 1.0)
        side = "BUY" if best_signal > 0 else "SELL"

        # 3) Gate + risk via BrainV2
        brain_state = "allow"
        brain_reason = "no_brain"
        brain_risk_mult = 1.0

        if brain is not None:
            # BrainV2 should accept confidence and return (state, reason, risk_mult)
            try:
                brain_state, brain_reason, brain_risk_mult = brain.evaluate(confidence)
            except Exception as e:
                brain_state, brain_reason, brain_risk_mult = ("allow", f"brain_error:{e}", 1.0)

        # Apply pet pressure, cap risk mult
        risk_mult = clamp(float(brain_risk_mult) * pressure, 0.25, MAX_RISK_MULT)

        # 4) Decide trade
        if confidence < MIN_CONF:
            log.info(f"SKIP (low_conf): best={best_market} conf={confidence:.2f} sig={best_signal:+.2f}")
            post_heartbeat(0, equity, last_pnl)
            time.sleep(CYCLE_SECONDS)
            continue

        if str(brain_state).lower().startswith("block") or str(brain_state).lower().startswith("hold"):
            log.info(f"SKIP ({brain_state}): best={best_market} conf={confidence:.2f} reason={brain_reason}")
            post_heartbeat(0, equity, last_pnl)
            time.sleep(CYCLE_SECONDS)
            continue

        # 5) Paper trade outcome
        risk_usd = equity * RISK_PER_TRADE * risk_mult
        pnl = simulate_pnl(risk_usd=risk_usd, confidence=confidence, side=side, signal=best_signal)
        equity += pnl
        last_pnl = pnl

        log.info(
            f"TRADE: market={best_market} side={side} conf={confidence:.2f} "
            f"risk_mult={risk_mult:.2f} pnl={pnl:+.2f} equity={equity:.2f} "
            f"brain={brain_state} reason={brain_reason}"
        )

        # 6) Post updates so Vault/Dashboard changes
        post_paper_event(
            pnl_usd=pnl,
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
PY

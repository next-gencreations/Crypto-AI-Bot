cat > main.py <<'PY'
# -*- coding: utf-8 -*-
"""
Crypto-AI-Bot (paper mode) — Universe Scanning (Option A)

What this does:
- Each cycle, scans many markets (UNIVERSE) by asking the API for /signal
- Picks the best market by confidence (abs(signal))
- Passes that best pick through BrainV2 gate + risk multiplier
- Simulates a paper trade and posts events/heartbeat so dashboard/vault updates

Works with your existing API endpoints:
- /signal?market=...
- /ingest/heartbeat
- /ingest/pet
- /ingest/equity
- /ingest/prices
- /vault/unlock
"""

import os
import time
import random
import logging
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

# =========================
# BRAIN V2 IMPORT
# =========================
try:
    from brain_v2 import BrainV2
except Exception as e:
    BrainV2 = None
    _brain_import_error = str(e)
else:
    _brain_import_error = ""

# =========================
# CONFIG
# =========================

API_URL = os.getenv("API_BASE") or os.getenv("API_URL") or "https://crypto-ai-api-1-7cte.onrender.com"
VAULT_PIN = os.getenv("VAULT_PIN") or os.getenv("VAULT_CODE") or os.getenv("PIN") or os.getenv("VAULT_PASS") or os.getenv("VAULT_PW") or os.getenv("VAULT_P") or os.getenv("VAULT") or os.getenv("VAULT_KEY") or os.getenv("VAULT_SECRET") or os.getenv("VAULT_TOKEN") or os.getenv("VAULT_UNLOCK") or os.getenv("VAULT_PASSWORD") or os.getenv("VAULT_PASSCODE") or os.getenv("VAULT_PIN_CODE") or os.getenv("VAULTPIN") or os.getenv("PIN_CODE") or os.getenv("PASSCODE") or os.getenv("PASSWORD") or os.getenv("PASS") or os.getenv("CODE") or os.getenv("VAULTPINCODE") or os.getenv("VAULTPASSCODE") or os.getenv("VAULT_PASS_CODE") or os.getenv("VAULT_PINCODE") or os.getenv("VAULTPIN_CODE") or os.getenv("VAULTPASS_CODE") or os.getenv("VAULTPASS") or os.getenv("VAULTPASSWD") or os.getenv("VAULTPWD") or os.getenv("VAULTPW") or os.getenv("VAULT-PIN") or os.getenv("VAULT-PASS") or os.getenv("VAULT-PASSWORD") or os.getenv("VAULT-PASSCODE") or os.getenv("VAULT-PINCODE") or os.getenv("VAULT-PIN-CODE") or os.getenv("VAULT-PASS-CODE") or os.getenv("VAULT_PASSWORD_CODE") or os.getenv("VAULT_PIN_CODE") or os.getenv("VAULTPASSCODE") or os.getenv("VAULTPINCODE") or os.getenv("VAULT_UNLOCK_PIN") or os.getenv("VAULT_UNLOCK_CODE") or os.getenv("VAULT_UNLOCK_PASS") or os.getenv("VAULT_UNLOCK_PASSWORD") or os.getenv("VAULT_UNLOCK_PASSCODE") or os.getenv("VAULT_UNLOCK_PINCODE") or os.getenv("VAULT_UNLOCK_PIN_CODE") or os.getenv("VAULT_UNLOCK_PASS_CODE") or os.getenv("VAULT_UNLOCK_PASSWORD_CODE") or os.getenv("VAULT_UNLOCK_PINCODE") or os.getenv("VAULT_UNLOCK_PASSCODE") or os.getenv("VAULT_UNLOCKPIN") or os.getenv("VAULT_UNLOCKPASS") or os.getenv("VAULT_UNLOCKPASSWORD") or os.getenv("VAULT_UNLOCKPASSCODE") or os.getenv("VAULT_UNLOCKPINCODE") or os.getenv("VAULT_UNLOCK_PINCODE") or os.getenv("VAULT_UNLOCK_PASSCODE") or os.getenv("VAULT_UNLOCK_PIN_CODE") or os.getenv("VAULT_UNLOCK_PASS_CODE") or os.getenv("VAULT_UNLOCK_PASSWORD_CODE") or os.getenv("VAULT_UNLOCK_PIN_CODE") or os.getenv("VAULT_UNLOCK_PASS_CODE") or os.getenv("VAULT_UNLOCK_PASSWORD_CODE") or os.getenv("VAULT_UNLOCK_PINCODE") or os.getenv("VAULT_UNLOCK_PASSCODE") or os.getenv("VAULT_UNLOCK_PIN") or os.getenv("VAULT_UNLOCK_CODE") or os.getenv("VAULT_UNLOCK_PASS") or os.getenv("VAULT_UNLOCK_PASSWORD") or os.getenv("VAULT_UNLOCK_PASSCODE") or os.getenv("VAULT_UNLOCK_PINCODE") or "4567"

CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS", os.getenv("BOT_SLEEP_SEC", "15")))
CONF_MIN_TRADE = float(os.getenv("CONF_MIN_TRADE", "0.62"))

# Paper sim settings
START_EQUITY = float(os.getenv("START_EQUITY", "1000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.015"))  # 1.5% base
WIN_THRESHOLD = float(os.getenv("WIN_THRESHOLD", "0.52"))     # higher = more losses; 0.52 is slightly tough

# Cryo safety
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "30"))
MAX_LOSS_STREAK_DEFAULT = int(os.getenv("MAX_LOSS_STREAK", "6"))
CRYO_SECONDS = int(os.getenv("CRYO_SECONDS", "1800"))

# Universe scanning
# Use same market codes your API uses (your screenshots show BTCUSDT/ETHUSDT)
UNIVERSE = os.getenv(
    "UNIVERSE",
    "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,LTCUSDT,BNBUSDT,LINKUSDT,TRXUSDT,DOTUSDT,ATOMUSDT,NEARUSDT,OPUSDT,ARBUSDT,MATICUSDT,ETCUSDT,BCHUSDT,UNIUSDT"
)
MARKETS: List[str] = [m.strip() for m in UNIVERSE.split(",") if m.strip()]
SCAN_TOP_N = int(os.getenv("SCAN_TOP_N", "12"))   # how many markets to scan each cycle (random sample)
SCAN_SHUFFLE = os.getenv("SCAN_SHUFFLE", "1") != "0"

USE_BRAIN_V2 = os.getenv("USE_BRAIN_V2", "1") != "0"
USE_SIGNAL = os.getenv("USE_SIGNAL", "1") != "0"  # call API /signal

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("crypto-ai-bot")

# =========================
# STATE
# =========================
state: Dict[str, Any] = {
    "equity": START_EQUITY,
    "peak_equity": START_EQUITY,
    "wins": 0,
    "losses": 0,
    "loss_streak": 0,
    "total_trades": 0,
    "total_pnl_usd": 0.0,
    "pet": {
        "name": os.getenv("COMPANION_NAME", "TradePet"),
        "stage": "active",
        "mood": "idle",
        "health": 100.0,
        "hunger": 0.0,
        "growth": 0.0,
        "updated_utc": None,
    }
}

brain = None
if USE_BRAIN_V2 and BrainV2 is not None:
    try:
        brain = BrainV2()
        log.info("BrainV2 loaded OK")
    except Exception as e:
        brain = None
        log.warning("BrainV2 init failed: %s", e)

# =========================
# HELPERS
# =========================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _get(path: str, params: Optional[dict] = None, timeout: int = 15) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning("GET %s failed: %s", path, e)
        return None

def _post(path: str, payload: Dict[str, Any], timeout: int = 15) -> None:
    try:
        requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    except Exception as e:
        log.warning("POST %s failed: %s", path, e)

def unlock_vault() -> Optional[str]:
    try:
        r = requests.post(f"{API_URL}/vault/unlock", json={"pin": VAULT_PIN}, timeout=15)
        r.raise_for_status()
        j = r.json()
        if not j.get("ok"):
            raise RuntimeError(str(j))
        return j.get("token")
    except Exception as e:
        log.warning("Vault unlock failed (continuing anyway): %s", e)
        return None

def api_control() -> Dict[str, Any]:
    # If API has /control, use it. Else default to ACTIVE.
    j = _get("/control")
    if j and isinstance(j, dict) and j.get("ok"):
        return {
            "state": (j.get("state") or "ACTIVE").upper(),
            "pause_reason": j.get("pause_reason"),
            "cryo_reason": j.get("cryo_reason"),
        }
    return {"state": "ACTIVE", "pause_reason": None, "cryo_reason": None}

def log_event(kind: str, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
    _post("/ingest/log", {
        "time_utc": utc_now(),
        "level": "info",
        "kind": kind,
        "msg": msg,
        "extra": extra or {}
    })

# =========================
# PET LOGIC (simple but stable)
# =========================

def pet_tick() -> None:
    p = state["pet"]

    # hunger rises slowly over time
    p["hunger"] = min(100.0, p["hunger"] + 0.35)

    # health drifts down if hunger is high
    if p["hunger"] > 80:
        p["health"] = max(0.0, p["health"] - 0.35)

    # mood from vitals
    if p["health"] <= 5:
        p["stage"] = "cryo"
        p["mood"] = "cryo"
    else:
        p["stage"] = "active"
        if p["health"] < 35:
            p["mood"] = "sick"
        elif p["hunger"] > 90:
            p["mood"] = "starving"
        elif p["hunger"] > 75:
            p["mood"] = "hungry"
        else:
            p["mood"] = "idle"

    p["updated_utc"] = utc_now()

def pet_on_trade(pnl: float, confidence: float) -> None:
    p = state["pet"]
    conf01 = max(0.0, min(1.0, float(confidence)))

    if pnl >= 0:
        # good trades heal and reduce hunger
        p["health"] = min(100.0, p["health"] + (2.0 + 3.0 * conf01))
        p["hunger"] = max(0.0, p["hunger"] - (3.0 + 4.0 * conf01))
        p["growth"] = min(100.0, p["growth"] + (0.3 + 0.6 * conf01))
        p["mood"] = "thriving"
    else:
        # bad trades hurt more when confidence was high
        hurt = 1.5 + 4.0 * conf01
        p["health"] = max(0.0, p["health"] - hurt)
        p["hunger"] = min(100.0, p["hunger"] + (1.0 + 2.0 * conf01))
        p["mood"] = "weak"

def survival_mode() -> str:
    p = state["pet"]
    if p["stage"] == "cryo" or p["mood"] == "cryo":
        return "CRYO"
    if p["health"] < 25:
        return "SICK"
    if p["hunger"] > 90:
        return "STARVING"
    if p["hunger"] > 75:
        return "HUNGRY"
    return "NORMAL"

def drawdown_pct() -> float:
    peak = max(1e-9, float(state["peak_equity"]))
    return (peak - float(state["equity"])) / peak * 100.0

# =========================
# SIGNAL + SCANNING
# =========================

def get_signal(market: str) -> Dict[str, Any]:
    """
    Calls API /signal for a market.
    Expected-ish shape:
      { ok:true, market:"BTCUSDT", side:"buy"/"sell"/"hold", confidence:0..1, reason:"...", features:{...} }
    We sanitize heavily so bot never crashes.
    """
    if not USE_SIGNAL:
        # fallback random signal
        s = random.uniform(-1, 1)
        return {
            "ok": True,
            "market": market,
            "side": "buy" if s > 0 else "sell",
            "confidence": abs(s),
            "reason": "random_fallback",
            "features": {}
        }

    j = _get("/signal", params={"market": market}, timeout=15) or {}
    side = str(j.get("side") or "hold").lower()
    conf = float(j.get("confidence") or 0.0)
    features = j.get("features") or {}
    reason = str(j.get("reason") or j.get("msg") or "api_signal")

    if side not in ("buy", "sell", "hold"):
        side = "hold"
    conf = max(0.0, min(1.0, conf))

    return {
        "ok": bool(j.get("ok", True)),
        "market": market,
        "side": side,
        "confidence": conf,
        "reason": reason,
        "features": features,
    }

def choose_scan_list() -> List[str]:
    if not MARKETS:
        return ["BTCUSDT"]
    markets = MARKETS[:]
    if SCAN_SHUFFLE:
        random.shuffle(markets)
    return markets[:max(1, min(SCAN_TOP_N, len(markets)))]

def scan_best_market() -> Tuple[str, Dict[str, Any]]:
    """
    Option A:
    - scan N markets each cycle
    - pick best by confidence (abs signal)
    - if all are hold/low confidence, return best anyway (bot will gate later)
    """
    scan = choose_scan_list()
    best_sig = None
    best_score = -1.0

    for m in scan:
        sig = get_signal(m)
        score = float(sig.get("confidence") or 0.0)

        # prefer non-hold if same confidence
        if score > best_score:
            best_score = score
            best_sig = sig
        elif best_sig is not None and score == best_score:
            if best_sig.get("side") == "hold" and sig.get("side") != "hold":
                best_sig = sig

    if best_sig is None:
        best_sig = get_signal("BTCUSDT")

    return str(best_sig.get("market") or "BTCUSDT"), best_sig

# =========================
# BrainV2 adapter
# =========================

def build_indicators_from_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    BrainV2 expects: atr, ema_slope, adx, atr_danger.
    We build safe proxies from whatever we get.
    """
    features = features or {}
    close = float(features.get("close") or 0.0) or 1.0
    ema_fast = float(features.get("ema_fast") or 0.0)
    ema_slow = float(features.get("ema_slow") or 0.0)
    trend = float(features.get("trend") or 0.0)

    atr_proxy = abs(ema_fast - ema_slow) / max(1.0, close)
    adx_proxy = min(50.0, abs(trend) * 1000.0)

    return {
        "atr": float(atr_proxy),
        "ema_slope": float(trend),
        "adx": float(adx_proxy),
        "atr_danger": float(os.getenv("BRAIN_ATR_DANGER", "0.03")),
    }

# =========================
# Paper trade
# =========================

def simulate_trade(risk_usd: float) -> float:
    risk_usd = max(0.0, float(risk_usd))
    win = random.random() > WIN_THRESHOLD
    pnl = risk_usd * random.uniform(0.6, 1.6) if win else -risk_usd

    state["equity"] += pnl
    state["peak_equity"] = max(float(state["peak_equity"]), float(state["equity"]))
    state["total_pnl_usd"] += pnl
    state["total_trades"] += 1

    if pnl > 0:
        state["wins"] += 1
        state["loss_streak"] = 0
    else:
        state["losses"] += 1
        state["loss_streak"] += 1

    return pnl

# =========================
# Prices (for dashboard charts / sanity)
# =========================

_last_prices: Dict[str, float] = {}

def generate_fake_prices(markets: List[str]) -> Dict[str, float]:
    out = {}
    for m in markets:
        base = _last_prices.get(m, 1000.0)
        step = base * random.uniform(-0.002, 0.002)
        base = max(0.1, base + step)
        _last_prices[m] = base
        out[m] = base
    return out

# =========================
# Push helpers
# =========================

def push_heartbeat(prices_ok: bool = True, best_market: str = "") -> None:
    _post("/ingest/heartbeat", {
        "time_utc": utc_now(),
        "status": "running",
        "survival_mode": survival_mode(),
        "equity_usd": float(state["equity"]),
        "wins": int(state["wins"]),
        "losses": int(state["losses"]),
        "loss_streak": int(state["loss_streak"]),
        "total_trades": int(state["total_trades"]),
        "total_pnl_usd": float(state["total_pnl_usd"]),
        "markets": MARKETS,
        "best_market": best_market,
        "open_positions": 0,
        "prices_ok": bool(prices_ok),
    })

def push_pet_and_equity() -> None:
    _post("/ingest/pet", {**state["pet"], "time_utc": utc_now(), "survival_mode": survival_mode()})
    _post("/ingest/equity", {"time_utc": utc_now(), "equity_usd": float(state["equity"])})

# =========================
# CRYO triggers
# =========================

def enter_cryo(reason: str, extra: Optional[Dict[str, Any]] = None) -> None:
    state["pet"]["stage"] = "cryo"
    state["pet"]["mood"] = "cryo"
    log_event("cryo", reason, extra or {})
    # Tell API if it supports /vault/lock (optional)
    _post("/vault/lock", {"reason": reason, "extra": extra or {}})

# =========================
# MAIN LOOP
# =========================

def run() -> None:
    log.info("Bot started. API_URL=%s cycle=%ss universe=%d scan_top_n=%d use_signal=%s use_brain=%s",
             API_URL, CYCLE_SECONDS, len(MARKETS), SCAN_TOP_N, USE_SIGNAL, USE_BRAIN_V2)

    if USE_BRAIN_V2 and brain is None:
        log.warning("BrainV2 NOT active. Import/init error: %s", _brain_import_error or "unknown")

    unlock_vault()

    while True:
        pet_tick()

        ctrl = api_control()
        mode = (ctrl.get("state") or "ACTIVE").upper()

        # Push prices for charting (fake for now)
        prices = generate_fake_prices(MARKETS[:max(1, min(len(MARKETS), 30))])
        _post("/ingest/prices", {"time_utc": utc_now(), "prices": prices})

        # No trading during CRYO/PAUSED (if your API uses it)
        if mode in ("CRYO", "PAUSED"):
            if mode == "CRYO":
                state["pet"]["mood"] = "cryo"
                state["pet"]["health"] = min(100.0, float(state["pet"]["health"]) + 0.6)
                state["pet"]["hunger"] = min(100.0, float(state["pet"]["hunger"]) + 0.3)

            push_heartbeat(prices_ok=True)
            push_pet_and_equity()
            log.info("%s active (%s). No trading.", mode, ctrl.get("cryo_reason") or ctrl.get("pause_reason"))
            time.sleep(CYCLE_SECONDS)
            continue

        # =========================
        # SCAN + PICK BEST
        # =========================
        best_market, sig = scan_best_market()
        side = str(sig.get("side") or "hold").lower()
        confidence = float(sig.get("confidence") or 0.0)
        reason = str(sig.get("reason") or "signal")
        features = sig.get("features") or {}

        # Always update dashboard with life signs
        push_heartbeat(prices_ok=True, best_market=best_market)
        push_pet_and_equity()

        # Basic gates
        if side == "hold":
            log.info("SKIP (hold): best=%s conf=%.2f reason=%s", best_market, confidence, reason)
            time.sleep(CYCLE_SECONDS)
            continue

        if confidence < CONF_MIN_TRADE:
            log.info("SKIP (low conf): best=%s conf=%.2f < %.2f reason=%s", best_market, confidence, CONF_MIN_TRADE, reason)
            time.sleep(CYCLE_SECONDS)
            continue

        # =========================
        # BRAIN V2 GATE + RISK
        # =========================
        brain_state = "OFF"
        brain_reason = ""
        brain_risk_mult = 1.0
        allow = True

        if USE_BRAIN_V2 and brain is not None:
            try:
                indicators = build_indicators_from_features(features)
                decision = brain.decide(indicators=indicators, signal_score=float(confidence))
                allow = bool(decision.allow_trade)
                brain_state = decision.brain_state
                brain_reason = decision.reason
                brain_risk_mult = float(decision.risk_multiplier or 1.0)
            except Exception as e:
                allow = True
                brain_state = "BRAIN_ERROR"
                brain_reason = str(e)
                brain_risk_mult = 1.0
                log.warning("BrainV2 decide error (fail-open): %s", e)

        if not allow:
            log.info("BLOCKED by brain: best=%s conf=%.2f brain=%s (%s)", best_market, confidence, brain_state, brain_reason)
            time.sleep(CYCLE_SECONDS)
            continue

        # Risk sizing (simple)
        base_risk_usd = float(state["equity"]) * RISK_PER_TRADE
        risk_usd = base_risk_usd * brain_risk_mult

        pnl = simulate_trade(risk_usd=risk_usd)

        if USE_BRAIN_V2 and brain is not None:
            try:
                brain.record_trade(float(pnl), equity_after=float(state["equity"]))
            except Exception as e:
                log.warning("BrainV2 record_trade failed: %s", e)

        pet_on_trade(pnl, confidence)
        dd = drawdown_pct()

        # Post trade to API
        _post("/paper/event", {
            "pnl_usd": float(pnl),
            "reason": "universe_trade",
            "market": best_market,
            "side": side,
            "confidence": float(confidence),
            "risk_usd": float(risk_usd),
            "brain_state": brain_state,
            "brain_reason": brain_reason,
            "brain_risk_multiplier": float(brain_risk_mult),
            "time_utc": utc_now(),
        })

        log.info("TRADE: market=%s side=%s conf=%.2f risk=%.2f pnl=%.2f equity=%.2f brain=%s",
                 best_market, side.upper(), confidence, risk_usd, pnl, float(state["equity"]), brain_state)

        # Safety: cryo triggers
        max_loss_streak = MAX_LOSS_STREAK_DEFAULT
        if state["loss_streak"] >= max_loss_streak:
            enter_cryo("loss_streak", {
                "loss_streak": int(state["loss_streak"]),
                "threshold": int(max_loss_streak),
                "equity": float(state["equity"]),
                "best_market": best_market,
            })

        if dd >= MAX_DRAWDOWN_PCT:
            enter_cryo("max_drawdown", {
                "drawdown_pct": float(dd),
                "threshold": float(MAX_DRAWDOWN_PCT),
                "equity": float(state["equity"]),
                "best_market": best_market,
            })

        time.sleep(CYCLE_SECONDS)

if __name__ == "__main__":
    run()
PY

# -*- coding: utf-8 -*-
import os
import time
import random
import logging
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# =========================
# CONFIG
# =========================

API_URL = (os.getenv("API_URL") or "https://crypto-ai-api-1-7cte.onrender.com").rstrip("/")
CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS", "60"))

START_EQUITY = float(os.getenv("START_EQUITY", "1000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.015"))  # base risk % of equity

# Cryo rules
MAX_LOSS_STREAK = int(os.getenv("MAX_LOSS_STREAK", "5"))
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "5.0"))  # % from peak
CRYO_SECONDS = int(os.getenv("CRYO_SECONDS", "420"))  # 7 min

# IMPORTANT: keep these in USDT format to match your API/dashboard
MARKETS = [m.strip() for m in (os.getenv("MARKETS", "BTCUSDT,ETHUSDT").split(",")) if m.strip()]

# =========================
# SIGNAL + BRAIN SETTINGS
# =========================

USE_SIGNAL = (os.getenv("USE_SIGNAL", "1").strip().lower() not in ("0", "false", "no"))
USE_BRAIN_V2 = (os.getenv("USE_BRAIN_V2", "1").strip().lower() not in ("0", "false", "no"))

# Gate: if signal confidence is below this, skip before brain (keeps spam down)
CONF_MIN_TRADE = float(os.getenv("CONF_MIN_TRADE", "0.50"))

# Safety caps
MIN_EQUITY_HARD_STOP = float(os.getenv("MIN_EQUITY_HARD_STOP", "100"))
MAX_RISK_CAP_PCT = float(os.getenv("MAX_RISK_CAP_PCT", "3.0"))  # cap risk per trade as % equity (after all multipliers)

# Paper-sim "edge"
WIN_THRESHOLD = float(os.getenv("WIN_THRESHOLD", "0.42"))

# =========================
# LOGGING
# =========================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("Crypto-AI-Bot")

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
        "name": "TradePet",
        "sex": os.getenv("PET_SEX", "boy"),
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
# BRAIN V2 (IMPORT)
# =========================

brain = None
if USE_BRAIN_V2:
    try:
        from brain_v2 import BrainV2  # must be in same folder
        brain = BrainV2()
        log.info("BrainV2 enabled.")
    except Exception as e:
        brain = None
        log.warning("BrainV2 import failed, running without it: %s", e)

# =========================
# API HELPERS
# =========================

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _post(path: str, payload: Dict[str, Any]) -> bool:
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        log.warning("POST %s failed: %s", path, e)
        return False

def _get(path: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_URL}{path}", timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        log.warning("GET %s failed: %s", path, e)
        return None

def api_control() -> Dict[str, str]:
    c = _get("/control") or {}
    return {
        "state": (c.get("state") or "ACTIVE").upper(),
        "pause_until_utc": c.get("pause_until_utc", "") or "",
        "cryo_until_utc": c.get("cryo_until_utc", "") or "",
        "pause_reason": c.get("pause_reason", "") or "",
        "cryo_reason": c.get("cryo_reason", "") or "",
    }

def enter_cryo(reason: str, details: Dict[str, Any]) -> None:
    _post("/control/cryo", {"seconds": CRYO_SECONDS, "reason": reason})
    _post("/ingest/death", {
        "time_utc": utc_now(),
        "source": "bot",
        "reason": "CRYO_TRIGGERED",
        "details": {"why": reason, **(details or {})}
    })

def log_event(ev_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
    _post("/ingest/event", {
        "time_utc": utc_now(),
        "type": ev_type,
        "message": message,
        "details": details or {}
    })

# =========================
# SIGNAL HELPER
# =========================

def get_signal(market: str = "BTCUSDT") -> Dict[str, Any]:
    """
    GET /signal?market=BTCUSDT
    Expected:
      { side: "buy|sell|hold", confidence: 0..1, reason: "...", features: {...} }
    """
    s = _get(f"/signal?market={market}") or {}
    side = (s.get("side") or "hold").lower().strip()
    if side not in ("buy", "sell", "hold"):
        side = "hold"

    try:
        conf = float(s.get("confidence") or 0.5)
    except Exception:
        conf = 0.5

    reason = s.get("reason") or "unknown"
    features = s.get("features") or {}

    return {"side": side, "confidence": conf, "reason": reason, "features": features}

# =========================
# PET LOGIC
# =========================

def pet_tick() -> None:
    p = state["pet"]
    p["time_utc"] = utc_now()

    p["hunger"] = min(100.0, p["hunger"] + 2.0)

    if p["hunger"] > 85:
        p["health"] = max(0.0, p["health"] - 2.0)
    elif p["hunger"] < 35:
        p["health"] = min(100.0, p["health"] + 0.6)

    if p["health"] < 20:
        p["mood"] = "sick"
    elif p["hunger"] > 85:
        p["mood"] = "hungry"
    else:
        p["mood"] = "focused"

    if p["stage"] == "egg" and (state["total_pnl_usd"] > 0 or state["wins"] >= 5):
        p["stage"] = "hatched"
        p["growth"] = max(p["growth"], 10.0)

def pet_on_trade(pnl: float) -> None:
    p = state["pet"]
    if pnl > 0:
        p["hunger"] = max(0.0, p["hunger"] - 10.0)
        p["health"] = min(100.0, p["health"] + 2.0)
        p["growth"] = min(100.0, p["growth"] + 2.0)
        p["mood"] = "happy"
        log_event("sound", "purr", {"pnl": pnl})
    else:
        p["hunger"] = min(100.0, p["hunger"] + 8.0)
        p["health"] = max(0.0, p["health"] - 4.0)
        p["mood"] = "sad"
        log_event("sound", "whimper", {"pnl": pnl})

# =========================
# TRADING LOGIC (SAFE PAPER SIM)
# =========================

def simulate_trade(risk_usd: float) -> float:
    risk_usd = max(0.0, float(risk_usd))

    win = random.random() > WIN_THRESHOLD
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

def drawdown_pct() -> float:
    peak = max(1e-9, state["peak_equity"])
    return (peak - state["equity"]) / peak * 100.0

def survival_mode() -> str:
    p = state["pet"]
    if p["hunger"] > 90:
        return "STARVING"
    if p["hunger"] > 75:
        return "HUNGRY"
    if p["health"] < 25:
        return "SICK"
    return "NORMAL"

# =========================
# PRICE TICKS (FOR OHLC)
# =========================

_last_prices: Dict[str, float] = {"BTCUSDT": 42000.0, "ETHUSDT": 2200.0}

def generate_fake_prices() -> Dict[str, float]:
    for m in MARKETS:
        base = _last_prices.get(m, 1000.0)
        step = base * random.uniform(-0.0018, 0.0018)
        base = max(0.1, base + step)
        _last_prices[m] = base
    return dict(_last_prices)

# =========================
# PUSH HELPERS
# =========================

def push_heartbeat(prices_ok: bool = True) -> None:
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
        "prices_ok": bool(prices_ok),
    })

def push_pet_and_equity() -> None:
    _post("/ingest/pet", {**state["pet"], "time_utc": utc_now(), "survival_mode": survival_mode()})
    _post("/ingest/equity", {"time_utc": utc_now(), "equity_usd": state["equity"]})

# =========================
# MAIN LOOP
# =========================

_market_i = 0

def pick_market() -> str:
    global _market_i
    if not MARKETS:
        return "BTCUSDT"
    m = MARKETS[_market_i % len(MARKETS)]
    _market_i += 1
    return m

def run() -> None:
    log.info("Bot started. API_URL=%s cycle=%ss markets=%s USE_SIGNAL=%s USE_BRAIN_V2=%s",
             API_URL, CYCLE_SECONDS, MARKETS, USE_SIGNAL, bool(brain))

    while True:
        pet_tick()

        ctrl = api_control()
        mode = ctrl["state"]

        prices = generate_fake_prices()
        _post("/ingest/prices", {"time_utc": utc_now(), "prices": prices})

        # No trading during CRYO/PAUSED
        if mode in ("CRYO", "PAUSED"):
            if mode == "CRYO":
                state["pet"]["mood"] = "cryo"
                state["pet"]["health"] = min(100.0, state["pet"]["health"] + 0.8)
                state["pet"]["hunger"] = min(100.0, state["pet"]["hunger"] + 0.5)

            push_heartbeat(prices_ok=True)
            push_pet_and_equity()
            log.info("%s active (%s). No trading.", mode, ctrl.get("cryo_reason") or ctrl.get("pause_reason"))
            time.sleep(CYCLE_SECONDS)
            continue

        # Hard stop safety
        if state["equity"] <= MIN_EQUITY_HARD_STOP:
            enter_cryo("min_equity_hard_stop", {"equity": state["equity"], "threshold": MIN_EQUITY_HARD_STOP})
            push_heartbeat(prices_ok=True)
            push_pet_and_equity()
            time.sleep(CYCLE_SECONDS)
            continue

        market = pick_market()

        # -------------------------
        # Get signal
        # -------------------------
        if USE_SIGNAL:
            sig = get_signal(market)
            side = sig["side"]
            conf = float(sig["confidence"])
            reason = sig["reason"]
            features = sig.get("features") or {}
        else:
            side = "buy" if random.random() > 0.5 else "sell"
            conf = float(random.uniform(0.5, 0.9))
            reason = "random_fallback"
            features = {}

        # Basic gate
        if side == "hold" or conf < CONF_MIN_TRADE:
            log_event("decision", "hold_or_low_confidence", {
                "market": market, "side": side, "confidence": conf, "reason": reason, "features": features
            })
            push_heartbeat(prices_ok=True)
            push_pet_and_equity()
            log.info("Skip trade | %s %s conf=%.3f reason=%s", market, side, conf, reason)
            time.sleep(CYCLE_SECONDS)
            continue

        # -------------------------
        # BrainV2 decision (YOUR API)
        # -------------------------
        brain_used = False
        brain_decision = None

        # Build indicators dict from features if present (safe defaults)
        indicators = {
            "atr": float(features.get("atr", 0.0) or 0.0),
            "ema_slope": float(features.get("ema_slope", 0.0) or 0.0),
            "adx": float(features.get("adx", 0.0) or 0.0),
            # Provide a default atr_danger if the feature doesn’t supply it
            "atr_danger": float(features.get("atr_danger", 0.05) or 0.05),
        }

        allow_trade = True
        effective_conf = conf
        risk_mult = 1.0
        brain_state = ""
        brain_reason = ""

        if brain is not None:
            try:
                brain_decision = brain.decide(indicators=indicators, signal_score=float(conf))
                brain_used = True
                allow_trade = bool(brain_decision.allow_trade)
                effective_conf = float(brain_decision.confidence)
                risk_mult = float(brain_decision.risk_multiplier)
                brain_state = str(brain_decision.brain_state)
                brain_reason = str(brain_decision.reason)
            except Exception as e:
                # If Brain fails, fall back to simple trading
                brain_used = False
                log.warning("BrainV2 decide failed: %s", e)

        # Apply brain veto
        if not allow_trade or risk_mult <= 0.0:
            log_event("decision", "brain_veto", {
                "market": market,
                "side": side,
                "signal_confidence": conf,
                "effective_confidence": effective_conf,
                "brain_used": brain_used,
                "brain_state": brain_state,
                "brain_reason": brain_reason,
                "indicators": indicators,
                "reason": reason,
            })
            push_heartbeat(prices_ok=True)
            push_pet_and_equity()
            log.info("Brain veto | %s %s sig=%.3f eff=%.3f state=%s", market, side, conf, effective_conf, brain_state)
            time.sleep(CYCLE_SECONDS)
            continue

        # -------------------------
        # Risk sizing (paper-safe)
        # -------------------------
        base_risk_usd = float(state["equity"]) * float(RISK_PER_TRADE)
        risk_usd = base_risk_usd * risk_mult

        # Cap risk
        max_risk = float(state["equity"]) * (MAX_RISK_CAP_PCT / 100.0)
        if risk_usd > max_risk:
            risk_usd = max_risk

        # Log decision
        log_event("decision", "trade_decision", {
            "market": market,
            "side": side,
            "signal_confidence": conf,
            "effective_confidence": effective_conf,
            "risk_usd": risk_usd,
            "risk_multiplier": risk_mult,
            "brain_used": brain_used,
            "brain_state": brain_state,
            "brain_reason": brain_reason,
            "indicators": indicators,
            "signal_reason": reason,
            "features": features,
        })

        # Execute paper trade
        pnl = simulate_trade(risk_usd=risk_usd)
        pet_on_trade(pnl)

        # Feed back to brain
        if brain is not None:
            try:
                brain.record_trade(float(pnl))
            except Exception:
                pass

        dd = drawdown_pct()

        log.info(
            "Trade #%d | %s %s | PnL %.2f | Equity %.2f | DD %.2f%% | LossStreak %d | Brain=%s (%s)",
            state["total_trades"], market, side.upper(), pnl, state["equity"], dd,
            state["loss_streak"], brain_state, "on" if brain_used else "off"
        )

        # Cryo triggers (system level)
        if state["loss_streak"] >= MAX_LOSS_STREAK:
            enter_cryo("loss_streak", {
                "market": market,
                "loss_streak": state["loss_streak"],
                "threshold": MAX_LOSS_STREAK,
                "drawdown_pct": dd,
                "equity": state["equity"],
                "side": side,
                "brain_state": brain_state,
            })

        if dd >= MAX_DRAWDOWN_PCT:
            enter_cryo("max_drawdown", {
                "market": market,
                "drawdown_pct": dd,
                "peak": state["peak_equity"],
                "equity": state["equity"],
                "side": side,
                "brain_state": brain_state,
            })

        push_heartbeat(prices_ok=True)
        push_pet_and_equity()

        # Trade log
        _post("/ingest/trade", {
            "time_utc": utc_now(),
            "market": market,
            "side": side,
            "size_usd": float(risk_usd),
            "price": float(prices.get(market, 0)),
            "pnl_usd": float(pnl),
            "confidence": float(effective_conf),
            "reason": f"{reason} | {brain_reason}".strip(),
        })

        time.sleep(CYCLE_SECONDS)

if __name__ == "__main__":
    run()

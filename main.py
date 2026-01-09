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
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.015"))  # confident but sane

# Cryo rules
MAX_LOSS_STREAK = int(os.getenv("MAX_LOSS_STREAK", "5"))
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "5.0"))  # % from peak
CRYO_SECONDS = int(os.getenv("CRYO_SECONDS", "420"))  # 7 min

# IMPORTANT: keep these in USDT format to match your API/dashboard
MARKETS = [m.strip() for m in (os.getenv("MARKETS", "BTCUSDT,ETHUSDT").split(",")) if m.strip()]

# =========================
# CONFIDENCE -> BEHAVIOUR
# =========================

# Gate: skip trades below this confidence (lower = trades more often)
CONF_MIN_TRADE = float(os.getenv("CONF_MIN_TRADE", "0.52"))

# Confidence band used for scaling size and pet effects
CONF_LOW = float(os.getenv("CONF_LOW", "0.58"))
CONF_HIGH = float(os.getenv("CONF_HIGH", "0.78"))

# Size multiplier at low/high confidence
SIZE_MIN_MULT = float(os.getenv("SIZE_MIN_MULT", "0.80"))
SIZE_MAX_MULT = float(os.getenv("SIZE_MAX_MULT", "1.80"))

# Stricter freezing when confidence is low
STRICT_STREAK_LOWCONF = int(os.getenv("STRICT_STREAK_LOWCONF", "2"))
STRICT_STREAK_MIDCONF = int(os.getenv("STRICT_STREAK_MIDCONF", "3"))
STRICT_STREAK_LOW_CUTOFF = float(os.getenv("STRICT_STREAK_LOW_CUTOFF", "0.52"))
STRICT_STREAK_MID_CUTOFF = float(os.getenv("STRICT_STREAK_MID_CUTOFF", "0.60"))

# TEMP confidence generator until real model plugged in (used if /signal fails)
CONF_GEN_MIN = float(os.getenv("CONF_GEN_MIN", "0.55"))
CONF_GEN_MAX = float(os.getenv("CONF_GEN_MAX", "0.90"))

# Paper-sim "edge" (higher = wins more often). 0.42 => ~58% win rate.
WIN_THRESHOLD = float(os.getenv("WIN_THRESHOLD", "0.42"))

# Use /signal endpoint (turn off if needed)
USE_SIGNAL = (os.getenv("USE_SIGNAL", "1").strip().lower() not in ("0", "false", "no"))

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
# SIGNAL HELPER (NEW)
# =========================

def get_signal(market: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Calls the API brain: GET /signal?market=BTCUSDT
    Expected response shape:
      { side: "buy|sell|hold", confidence: 0..1, reason: "...", features: {...} }
    Safe defaults if missing.
    """
    # IMPORTANT: _get expects a path starting with /
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

    return {
        "side": side,
        "confidence": conf,
        "reason": reason,
        "features": features,
    }

# =========================
# CONFIDENCE HELPERS
# =========================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def confidence_to_size_mult(conf: float) -> float:
    c = clamp(conf, CONF_LOW, CONF_HIGH)
    if CONF_HIGH == CONF_LOW:
        return 1.0
    t = (c - CONF_LOW) / (CONF_HIGH - CONF_LOW)  # 0..1
    return SIZE_MIN_MULT + t * (SIZE_MAX_MULT - SIZE_MIN_MULT)

def confidence_to_strict_loss_streak(conf: float) -> int:
    if conf < STRICT_STREAK_LOW_CUTOFF:
        return STRICT_STREAK_LOWCONF
    if conf < STRICT_STREAK_MID_CUTOFF:
        return STRICT_STREAK_MIDCONF
    return MAX_LOSS_STREAK

def generate_confidence() -> float:
    # fallback only
    return float(random.uniform(CONF_GEN_MIN, CONF_GEN_MAX))

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

def pet_on_trade(pnl: float, confidence: float) -> None:
    p = state["pet"]

    conf01 = 0.0
    if CONF_HIGH != CONF_LOW:
        conf01 = (clamp(confidence, CONF_LOW, CONF_HIGH) - CONF_LOW) / (CONF_HIGH - CONF_LOW)

    if pnl > 0:
        hunger_drop = 10.0 + (6.0 * conf01)
        hp_gain = 1.5 + (2.0 * conf01)
        growth_gain = 1.0 + (2.5 * conf01)

        p["hunger"] = max(0.0, p["hunger"] - hunger_drop)
        p["health"] = min(100.0, p["health"] + hp_gain)
        p["growth"] = min(100.0, p["growth"] + growth_gain)
        p["mood"] = "happy"
        log_event("sound", "purr", {"pnl": pnl, "confidence": confidence})
    else:
        low_conf_penalty = 1.0 - conf01
        hunger_up = 6.0 + (6.0 * low_conf_penalty)
        hp_drop = 3.0 + (3.0 * low_conf_penalty)

        p["hunger"] = min(100.0, p["hunger"] + hunger_up)
        p["health"] = max(0.0, p["health"] - hp_drop)
        p["mood"] = "sad"
        log_event("sound", "whimper", {"pnl": pnl, "confidence": confidence})

# =========================
# TRADING LOGIC (SAFE PAPER SIM)
# =========================

def simulate_trade(risk_usd: float) -> float:
    risk_usd = max(0.0, float(risk_usd))

    # "Winning-style" paper sim edge
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

def survival_mode() -> str:
    p = state["pet"]
    if p["hunger"] > 90:
        return "STARVING"
    if p["hunger"] > 75:
        return "HUNGRY"
    if p["health"] < 25:
        return "SICK"
    return "NORMAL"

def drawdown_pct() -> float:
    peak = max(1e-9, state["peak_equity"])
    return (peak - state["equity"]) / peak * 100.0

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
    log.info("Bot started. API_URL=%s cycle=%ss markets=%s USE_SIGNAL=%s", API_URL, CYCLE_SECONDS, MARKETS, USE_SIGNAL)

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

        # =========================
        # SIGNAL DECISION (NEW)
        # =========================
        market = pick_market()

        side = "hold"
        confidence = 0.5
        reason = "fallback"
        features: Dict[str, Any] = {}

        if USE_SIGNAL:
            sig = get_signal(market)
            side = sig["side"]
            confidence = float(sig["confidence"])
            reason = sig["reason"]
            features = sig.get("features") or {}

            # if the API is down or missing endpoint, /signal might return None -> handled above
            # If signal seems "bad", we'll still gate it below
        else:
            confidence = generate_confidence()
            side = "buy" if random.random() > 0.5 else "sell"
            reason = "random_fallback"

        # Gate: HOLD or low confidence => skip trade
        if side == "hold" or confidence < CONF_MIN_TRADE:
            log_event("decision", "hold_or_low_confidence", {
                "market": market,
                "side": side,
                "confidence": confidence,
                "min": CONF_MIN_TRADE,
                "reason": reason,
                "features": features,
            })
            push_heartbeat(prices_ok=True)
            push_pet_and_equity()

            log.info("Skip trade | market=%s side=%s conf=%.3f reason=%s", market, side, confidence, reason)
            time.sleep(CYCLE_SECONDS)
            continue

        # Size/risk scaling
        size_mult = confidence_to_size_mult(confidence)
        base_risk_usd = state["equity"] * RISK_PER_TRADE
        risk_usd = base_risk_usd * size_mult

        strict_loss_streak = confidence_to_strict_loss_streak(confidence)

        log_event("decision", "trade_decision", {
            "market": market,
            "side": side,
            "confidence": confidence,
            "size_mult": size_mult,
            "risk_usd": risk_usd,
            "strict_loss_streak": strict_loss_streak,
            "reason": reason,
            "features": features,
        })

        pnl = simulate_trade(risk_usd=risk_usd)
        pet_on_trade(pnl, confidence)

        dd = drawdown_pct()

        log.info(
            "Trade #%d | %s %s | PnL %.2f | Equity %.2f | DD %.2f%% | LossStreak %d (thr %d) | Conf %.2f | SizeMult %.2f | Reason %s",
            state["total_trades"], market, side.upper(), pnl, state["equity"], dd,
            state["loss_streak"], strict_loss_streak, confidence, size_mult, reason
        )

        # Cryo triggers
        if state["loss_streak"] >= strict_loss_streak:
            enter_cryo("loss_streak", {
                "market": market,
                "loss_streak": state["loss_streak"],
                "threshold": strict_loss_streak,
                "confidence": confidence,
                "drawdown_pct": dd,
                "equity": state["equity"],
                "side": side,
                "reason": reason,
            })

        if dd >= MAX_DRAWDOWN_PCT:
            enter_cryo("max_drawdown", {
                "market": market,
                "drawdown_pct": dd,
                "peak": state["peak_equity"],
                "equity": state["equity"],
                "confidence": confidence,
                "side": side,
                "reason": reason,
            })

        push_heartbeat(prices_ok=True)
        push_pet_and_equity()

        # Trade log driven by API brain
        _post("/ingest/trade", {
            "time_utc": utc_now(),
            "market": market,
            "side": side,
            "size_usd": float(risk_usd),
            "price": float(prices.get(market, 0)),
            "pnl_usd": float(pnl),
            "confidence": float(confidence),
            "reason": reason,
        })

        time.sleep(CYCLE_SECONDS)

if __name__ == "__main__":
    run()

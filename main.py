import os
import time
import random
import logging
import requests
from datetime import datetime, timezone

# =========================
# CONFIG
# =========================

API_URL = (os.getenv("API_URL") or "https://crypto-ai-api-1-7cte.onrender.com").rstrip("/")
CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS", "30"))

START_EQUITY = float(os.getenv("START_EQUITY", "1000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))  # 1% paper risk

# cryo rules
MAX_LOSS_STREAK = int(os.getenv("MAX_LOSS_STREAK", "4"))
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "3.5"))  # % from peak
CRYO_SECONDS = int(os.getenv("CRYO_SECONDS", "600"))  # 10 min

MARKETS = [m.strip() for m in (os.getenv("MARKETS", "BTCUSDT,ETHUSDT").split(",")) if m.strip()]

# =========================
# CONFIDENCE -> BEHAVIOUR (NEW)
# =========================
# Trade gating
CONF_MIN_TRADE = float(os.getenv("CONF_MIN_TRADE", "0.58"))

# Scaling band for size
CONF_LOW = float(os.getenv("CONF_LOW", "0.60"))
CONF_HIGH = float(os.getenv("CONF_HIGH", "0.80"))
SIZE_MIN_MULT = float(os.getenv("SIZE_MIN_MULT", "0.50"))  # 50% size at low confidence
SIZE_MAX_MULT = float(os.getenv("SIZE_MAX_MULT", "1.50"))  # 150% size at high confidence

# Strictness band (lower confidence => freeze sooner)
STRICT_STREAK_LOWCONF = int(os.getenv("STRICT_STREAK_LOWCONF", "2"))
STRICT_STREAK_MIDCONF = int(os.getenv("STRICT_STREAK_MIDCONF", "3"))
STRICT_STREAK_LOW_CUTOFF = float(os.getenv("STRICT_STREAK_LOW_CUTOFF", "0.58"))
STRICT_STREAK_MID_CUTOFF = float(os.getenv("STRICT_STREAK_MID_CUTOFF", "0.65"))

# Optional: confidence generation range (until real model plugged in)
CONF_GEN_MIN = float(os.getenv("CONF_GEN_MIN", "0.50"))
CONF_GEN_MAX = float(os.getenv("CONF_GEN_MAX", "0.90"))

# =========================
# LOGGING
# =========================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("Crypto-AI-Bot")

# =========================
# STATE
# =========================

state = {
    "equity": START_EQUITY,
    "peak_equity": START_EQUITY,
    "wins": 0,
    "losses": 0,
    "loss_streak": 0,
    "total_trades": 0,
    "total_pnl_usd": 0.0,
    "pet": {
        "name": "TradePet",
        "sex": os.getenv("PET_SEX", "boy"),  # boy/girl (cosmetic)
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

def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _post(path, payload):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        log.warning(f"POST {path} failed: {e}")
        return False

def _get(path):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        log.warning(f"GET {path} failed: {e}")
        return None

def api_control():
    c = _get("/control") or {}
    return {
        "state": (c.get("state") or "ACTIVE").upper(),
        "pause_until_utc": c.get("pause_until_utc", ""),
        "cryo_until_utc": c.get("cryo_until_utc", ""),
        "pause_reason": c.get("pause_reason", ""),
        "cryo_reason": c.get("cryo_reason", ""),
    }

def enter_cryo(reason: str, details: dict):
    # Tell API we entered cryo
    _post("/control/cryo", {"seconds": CRYO_SECONDS, "reason": reason})

    # Record a "death" log (really: cryo/death history)
    _post("/ingest/death", {
        "time_utc": utc_now(),
        "source": "bot",
        "reason": "CRYO_TRIGGERED",
        "details": {"why": reason, **(details or {})}
    })

def log_event(ev_type: str, message: str, details: dict | None = None):
    _post("/ingest/event", {
        "time_utc": utc_now(),
        "type": ev_type,
        "message": message,
        "details": details or {}
    })

# =========================
# CONFIDENCE HELPERS (NEW)
# =========================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def confidence_to_size_mult(conf: float) -> float:
    c = clamp(conf, CONF_LOW, CONF_HIGH)
    if CONF_HIGH == CONF_LOW:
        return 1.0
    t = (c - CONF_LOW) / (CONF_HIGH - CONF_LOW)  # 0..1
    return SIZE_MIN_MULT + t * (SIZE_MAX_MULT - SIZE_MIN_MULT)

def confidence_to_strict_loss_streak(conf: float) -> int:
    # lower confidence -> stricter (freeze sooner)
    if conf < STRICT_STREAK_LOW_CUTOFF:
        return STRICT_STREAK_LOWCONF
    if conf < STRICT_STREAK_MID_CUTOFF:
        return STRICT_STREAK_MIDCONF
    return MAX_LOSS_STREAK

def generate_confidence() -> float:
    # TEMP until real AI signal is plugged in
    return float(random.uniform(CONF_GEN_MIN, CONF_GEN_MAX))

# =========================
# PET LOGIC
# =========================

def pet_tick():
    p = state["pet"]
    p["time_utc"] = utc_now()

    # hunger rises
    p["hunger"] = min(100.0, p["hunger"] + 2.0)

    # health reacts to hunger
    if p["hunger"] > 85:
        p["health"] = max(0.0, p["health"] - 2.0)
    elif p["hunger"] < 35:
        p["health"] = min(100.0, p["health"] + 0.6)

    # mood
    if p["health"] < 20:
        p["mood"] = "sick"
    elif p["hunger"] > 85:
        p["mood"] = "hungry"
    else:
        p["mood"] = "focused"

    # hatch rule
    if p["stage"] == "egg" and (state["total_pnl_usd"] > 0 or state["wins"] >= 5):
        p["stage"] = "hatched"
        p["growth"] = max(p["growth"], 10.0)

def pet_on_trade(pnl: float, confidence: float):
    """
    Confidence affects pet reward/penalty:
      - high confidence wins = bigger growth
      - low confidence losses = bigger health hit
    """
    p = state["pet"]

    # normalize conf into 0..1 across the band
    conf01 = 0.0
    if CONF_HIGH != CONF_LOW:
        conf01 = (clamp(confidence, CONF_LOW, CONF_HIGH) - CONF_LOW) / (CONF_HIGH - CONF_LOW)

    if pnl > 0:
        # reward scales with confidence
        hunger_drop = 10.0 + (6.0 * conf01)
        hp_gain = 1.5 + (2.0 * conf01)
        growth_gain = 1.0 + (2.5 * conf01)

        p["hunger"] = max(0.0, p["hunger"] - hunger_drop)
        p["health"] = min(100.0, p["health"] + hp_gain)
        p["growth"] = min(100.0, p["growth"] + growth_gain)
        p["mood"] = "happy"
        log_event("sound", "purr", {"pnl": pnl, "confidence": confidence})
    else:
        # penalty increases when confidence is low
        low_conf_penalty = 1.0 - conf01  # 1 when low, 0 when high
        hunger_up = 6.0 + (6.0 * low_conf_penalty)
        hp_drop = 3.0 + (3.0 * low_conf_penalty)

        p["hunger"] = min(100.0, p["hunger"] + hunger_up)
        p["health"] = max(0.0, p["health"] - hp_drop)
        p["mood"] = "sad"
        log_event("sound", "whimper", {"pnl": pnl, "confidence": confidence})

# =========================
# TRADING LOGIC (SAFE PAPER SIM)
# =========================

def simulate_trade(risk_usd: float):
    """
    Paper sim uses a risk amount per trade (USD).
    risk_usd is already confidence-scaled.
    """
    risk_usd = max(0.0, float(risk_usd))

    # slightly “edgey” win chance (replace later with real logic)
    win = random.random() > 0.47

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

def survival_mode():
    p = state["pet"]
    if p["hunger"] > 90:
        return "STARVING"
    if p["hunger"] > 75:
        return "HUNGRY"
    if p["health"] < 25:
        return "SICK"
    return "NORMAL"

def drawdown_pct():
    peak = max(1e-9, state["peak_equity"])
    return (peak - state["equity"]) / peak * 100.0

# =========================
# PRICE TICKS (FOR OHLC)
# =========================

_last_prices = {"BTCUSDT": 42000.0, "ETHUSDT": 2200.0}

def generate_fake_prices():
    # simple random walk so candles look real-ish
    for m in MARKETS:
        base = _last_prices.get(m, 1000.0)
        step = base * random.uniform(-0.0018, 0.0018)
        base = max(0.1, base + step)
        _last_prices[m] = base
    return dict(_last_prices)

# =========================
# MAIN LOOP
# =========================

def push_heartbeat(prices_ok=True):
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

def push_pet_and_equity():
    _post("/ingest/pet", {**state["pet"], "time_utc": utc_now(), "survival_mode": survival_mode()})
    _post("/ingest/equity", {"time_utc": utc_now(), "equity_usd": state["equity"]})

def run():
    log.info(f"🚀 Bot started. API_URL={API_URL} cycle={CYCLE_SECONDS}s markets={MARKETS}")

    while True:
        # 1) tick pet always
        pet_tick()

        # 2) read API control state
        ctrl = api_control()
        mode = ctrl["state"]

        # 3) always push prices (so candles keep forming even during cryo)
        prices = generate_fake_prices()
        _post("/ingest/prices", {"time_utc": utc_now(), "prices": prices})

        # 4) If CRYO or PAUSED, do NOT trade — just heartbeat/pet
        if mode in ("CRYO", "PAUSED"):
            if mode == "CRYO":
                state["pet"]["mood"] = "cryo"
                state["pet"]["health"] = min(100.0, state["pet"]["health"] + 0.8)
                state["pet"]["hunger"] = min(100.0, state["pet"]["hunger"] + 0.5)

            push_heartbeat(prices_ok=True)
            push_pet_and_equity()

            log.info(f"🧊 {mode} active ({ctrl.get('cryo_reason') or ctrl.get('pause_reason')}). No trading.")
            time.sleep(CYCLE_SECONDS)
            continue

        # =========================
        # 5) CONFIDENCE DECISION (NEW)
        # =========================
        confidence = generate_confidence()

        # Gate: skip trade if too low confidence
        if confidence < CONF_MIN_TRADE:
            log_event("decision", "skip_trade_low_confidence", {"confidence": confidence, "min": CONF_MIN_TRADE})

            # still push heartbeat/pet/equity so dashboard updates
            push_heartbeat(prices_ok=True)
            push_pet_and_equity()

            log.info(f"⛔ Skip trade | confidence={confidence:.3f} < {CONF_MIN_TRADE:.3f}")
            time.sleep(CYCLE_SECONDS)
            continue

        # Scale risk/size by confidence
        size_mult = confidence_to_size_mult(confidence)
        base_risk_usd = state["equity"] * RISK_PER_TRADE
        risk_usd = base_risk_usd * size_mult

        # Stricter loss streak threshold when confidence is low
        strict_loss_streak = confidence_to_strict_loss_streak(confidence)

        log_event("decision", "trade_decision", {
            "confidence": confidence,
            "size_mult": size_mult,
            "risk_usd": risk_usd,
            "strict_loss_streak": strict_loss_streak
        })

        # 6) Trade (paper sim) using confidence-scaled risk
        pnl = simulate_trade(risk_usd=risk_usd)
        pet_on_trade(pnl, confidence)

        dd = drawdown_pct()

        log.info(
            f"Trade #{state['total_trades']} | PnL {pnl:.2f} | Equity {state['equity']:.2f} | "
            f"DD {dd:.2f}% | LossStreak {state['loss_streak']} (thr {strict_loss_streak}) | "
            f"Conf {confidence:.2f} | SizeMult {size_mult:.2f} | PetHP {state['pet']['health']:.1f}"
        )

        # 7) Cryo triggers (updated with confidence strictness)
        if state["loss_streak"] >= strict_loss_streak:
            enter_cryo("loss_streak", {
                "loss_streak": state["loss_streak"],
                "threshold": strict_loss_streak,
                "confidence": confidence,
                "drawdown_pct": dd,
                "equity": state["equity"]
            })

        if dd >= MAX_DRAWDOWN_PCT:
            enter_cryo("max_drawdown", {
                "drawdown_pct": dd,
                "peak": state["peak_equity"],
                "equity": state["equity"],
                "confidence": confidence
            })

        # 8) push state to API
        push_heartbeat(prices_ok=True)
        push_pet_and_equity()

        # trade log (confidence + scaled size)
        _post("/ingest/trade", {
            "time_utc": utc_now(),
            "market": "BTCUSDT",
            "side": "buy" if pnl >= 0 else "sell",
            "size_usd": risk_usd,  # confidence-scaled size
            "price": prices.get("BTCUSDT", 0),
            "pnl_usd": pnl,
            "confidence": float(confidence),
            "reason": "paper_sim_confidence_scaled",
        })

        time.sleep(CYCLE_SECONDS)

if __name__ == "__main__":
    run()
```0

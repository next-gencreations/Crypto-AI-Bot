# -*- coding: utf-8 -*-
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

API_URL = (os.getenv("API_URL") or "https://crypto-ai-api-1-7cte.onrender.com").rstrip("/")
CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS", "60"))

START_EQUITY = float(os.getenv("START_EQUITY", "1000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.015"))

# Cryo rules
MAX_LOSS_STREAK = int(os.getenv("MAX_LOSS_STREAK", "5"))
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "5.0"))
CRYO_SECONDS = int(os.getenv("CRYO_SECONDS", "420"))

# Universe (Coinbase format)
# Example: BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LINK-USD
MARKETS = [m.strip() for m in (os.getenv("MARKETS", "BTC-USD,ETH-USD,SOL-USD").split(",")) if m.strip()]

# Coinbase public candles (no auth, no Binance)
COINBASE_EXCHANGE_BASE = os.getenv("COINBASE_EXCHANGE_BASE", "https://api.exchange.coinbase.com").rstrip("/")

# Scan settings (safe defaults)
SCAN_GRANULARITY = int(os.getenv("SCAN_GRANULARITY", "300"))  # 300 = 5m candles
SCAN_CANDLES = int(os.getenv("SCAN_CANDLES", "30"))           # how many candles to score with
SCAN_TOP_N = int(os.getenv("SCAN_TOP_N", "1"))               # keep 1 = best coin only (recommended)

# =========================
# CONFIDENCE -> BEHAVIOUR
# =========================

CONF_MIN_TRADE = float(os.getenv("CONF_MIN_TRADE", "0.52"))

CONF_LOW = float(os.getenv("CONF_LOW", "0.58"))
CONF_HIGH = float(os.getenv("CONF_HIGH", "0.78"))

SIZE_MIN_MULT = float(os.getenv("SIZE_MIN_MULT", "0.80"))
SIZE_MAX_MULT = float(os.getenv("SIZE_MAX_MULT", "1.80"))

STRICT_STREAK_LOWCONF = int(os.getenv("STRICT_STREAK_LOWCONF", "2"))
STRICT_STREAK_MIDCONF = int(os.getenv("STRICT_STREAK_MIDCONF", "3"))
STRICT_STREAK_LOW_CUTOFF = float(os.getenv("STRICT_STREAK_LOW_CUTOFF", "0.52"))
STRICT_STREAK_MID_CUTOFF = float(os.getenv("STRICT_STREAK_MID_CUTOFF", "0.60"))

# Paper sim edge
WIN_THRESHOLD = float(os.getenv("WIN_THRESHOLD", "0.42"))

# Use /signal endpoint (optional; Option A does NOT require it)
USE_SIGNAL = (os.getenv("USE_SIGNAL", "0").strip().lower() not in ("0", "false", "no"))

USE_BRAIN_V2 = (os.getenv("USE_BRAIN_V2", "1").strip().lower() not in ("0", "false", "no"))

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

# Create Brain V2
brain = None
if USE_BRAIN_V2 and BrainV2 is not None:
    try:
        brain = BrainV2()
    except Exception as e:
        brain = None
        log.warning("BrainV2 init failed, continuing without it: %s", e)

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
# OPTIONAL /signal
# =========================

def get_signal(market: str) -> Dict[str, Any]:
    """
    Optional: Calls API signal: GET /signal?market=BTC-USD
    Expected: { side, confidence, reason, features }
    """
    s = _get(f"/signal?market={market}") or {}
    side = (s.get("side") or "hold").lower().strip()
    if side not in ("buy", "sell", "hold"):
        side = "hold"
    try:
        conf = float(s.get("confidence") or 0.5)
    except Exception:
        conf = 0.5
    return {
        "side": side,
        "confidence": conf,
        "reason": s.get("reason") or "api_signal",
        "features": s.get("features") or {}
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
    t = (c - CONF_LOW) / (CONF_HIGH - CONF_LOW)
    return SIZE_MIN_MULT + t * (SIZE_MAX_MULT - SIZE_MIN_MULT)

def confidence_to_strict_loss_streak(conf: float) -> int:
    if conf < STRICT_STREAK_LOW_CUTOFF:
        return STRICT_STREAK_LOWCONF
    if conf < STRICT_STREAK_MID_CUTOFF:
        return STRICT_STREAK_MIDCONF
    return MAX_LOSS_STREAK

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
# BRAIN V2 ADAPTER
# =========================

def build_indicators_from_features(features: Dict[str, Any]) -> Dict[str, Any]:
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
# COINBASE SCAN (Option A)
# =========================

def fetch_coinbase_candles(product_id: str, granularity: int, limit: int) -> Optional[List[List[float]]]:
    """
    Coinbase Exchange candles:
    GET /products/{product_id}/candles?granularity=300
    returns: [[time, low, high, open, close, volume], ...] newest-first
    """
    url = f"{COINBASE_EXCHANGE_BASE}/products/{product_id}/candles"
    try:
        r = requests.get(url, params={"granularity": int(granularity)}, timeout=12)
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        return data[: max(1, int(limit))]
    except Exception:
        return None

def score_market_from_candles(candles: List[List[float]]) -> Optional[Dict[str, float]]:
    """
    candles newest-first: [t, low, high, open, close, volume]
    score = abs(momentum) * (0.5 + volatility_boost)
    """
    if not candles or len(candles) < 10:
        return None

    # Reverse to oldest->newest for momentum
    c = list(reversed(candles))

    opens = [float(x[3]) for x in c if len(x) >= 5]
    closes = [float(x[4]) for x in c if len(x) >= 5]
    highs = [float(x[2]) for x in c if len(x) >= 3]
    lows = [float(x[1]) for x in c if len(x) >= 2]

    if not opens or not closes or not highs or not lows:
        return None

    first = float(opens[0]) if opens[0] else 1.0
    last = float(closes[-1]) if closes[-1] else 1.0

    # momentum (signed)
    momentum = (last - first) / max(1e-9, first)

    # volatility proxy
    ranges = [(h - l) / max(1e-9, cl) for h, l, cl in zip(highs, lows, closes) if cl]
    vol = sum(ranges) / max(1, len(ranges))

    # score
    vol_boost = min(1.0, vol * 50.0)  # scale
    score = abs(momentum) * (0.5 + vol_boost)

    return {
        "close": last,
        "momentum": float(momentum),
        "vol": float(vol),
        "score": float(score),
    }

def score_to_confidence(score: float) -> float:
    # Simple curve: small scores ~0.50, bigger scores approach 0.95
    # You can tune SCAN_CONF_SCALE in env.
    scale = float(os.getenv("SCAN_CONF_SCALE", "6.0"))
    c = 0.50 + min(0.45, max(0.0, score * scale))
    return clamp(c, 0.0, 1.0)

def scan_universe(markets: List[str]) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
    """
    Returns:
      best_market,
      features dict,
      prices dict for ingest
    """
    best_market = markets[0] if markets else "BTC-USD"
    best_score = -1.0
    best_feat: Dict[str, Any] = {}
    prices: Dict[str, float] = {}

    scanned = 0
    ok_count = 0

    for m in markets:
        scanned += 1
        candles = fetch_coinbase_candles(m, SCAN_GRANULARITY, SCAN_CANDLES)
        if not candles:
            continue
        scored = score_market_from_candles(candles)
        if not scored:
            continue

        ok_count += 1
        prices[m] = float(scored["close"])

        if scored["score"] > best_score:
            best_score = scored["score"]
            best_market = m
            best_feat = dict(scored)

    # Include scan meta
    best_feat = best_feat or {}
    best_feat["scan_markets"] = scanned
    best_feat["scan_ok"] = ok_count
    best_feat["scan_granularity"] = SCAN_GRANULARITY
    best_feat["scan_candles"] = SCAN_CANDLES
    best_feat["scan_best_score"] = float(best_score if best_score > 0 else 0.0)

    return best_market, best_feat, prices

def derive_signal_from_scan(best_market: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """
    side + confidence come from momentum + score
    """
    momentum = float(features.get("momentum") or 0.0)
    score = float(features.get("score") or 0.0)

    if abs(momentum) < 1e-6:
        side = "hold"
    else:
        side = "buy" if momentum > 0 else "sell"

    confidence = score_to_confidence(score)
    reason = f"scan_best score={score:.4f} mom={momentum:.4f}"

    # extra fields that BrainV2 adapter can use
    close = float(features.get("close") or 0.0)
    vol = float(features.get("vol") or 0.0)

    out_features = {
        "close": close,
        # simple proxies for ema_fast/ema_slow/trend
        "ema_fast": close * (1.0 + momentum * 0.35),
        "ema_slow": close * (1.0 + momentum * 0.15),
        "trend": momentum,
        "vol": vol,
        "score": score,
        "market": best_market,
        **features,
    }

    return {"side": side, "confidence": confidence, "reason": reason, "features": out_features}

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

def run() -> None:
    log.info(
        "Bot started. API_URL=%s cycle=%ss markets=%d USE_SIGNAL=%s USE_BRAIN_V2=%s COINBASE=%s",
        API_URL, CYCLE_SECONDS, len(MARKETS), USE_SIGNAL, USE_BRAIN_V2, COINBASE_EXCHANGE_BASE
    )
    if USE_BRAIN_V2 and brain is None:
        log.warning("BrainV2 NOT active. Import/init error: %s", _brain_import_error or "unknown")

    if not MARKETS:
        log.warning("MARKETS is empty. Set MARKETS env like: BTC-USD,ETH-USD,SOL-USD")
        time.sleep(5)

    while True:
        pet_tick()

        ctrl = api_control()
        mode = ctrl["state"]

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
        # UNIVERSE SCAN
        # =========================
        best_market, best_feat, prices = scan_universe(MARKETS)

        # Ingest prices to API for dashboard display (best effort)
        if prices:
            _post("/ingest/prices", {"time_utc": utc_now(), "prices": prices})
        else:
            _post("/ingest/prices", {"time_utc": utc_now(), "prices": {}})

        # Optional: log scan summary
        log_event("scan", "universe_scan", {
            "best_market": best_market,
            "best_score": best_feat.get("scan_best_score", 0.0),
            "scan_ok": best_feat.get("scan_ok", 0),
            "scan_markets": best_feat.get("scan_markets", 0),
        })

        # =========================
        # SIGNAL
        # =========================
        if USE_SIGNAL:
            sig = get_signal(best_market)
            side = sig["side"]
            confidence = float(sig["confidence"])
            reason = sig["reason"]
            features = sig.get("features") or {}
        else:
            sig = derive_signal_from_scan(best_market, best_feat)
            side = sig["side"]
            confidence = float(sig["confidence"])
            reason = sig["reason"]
            features = sig.get("features") or {}

        # =========================
        # BASIC GATES
        # =========================
        if side == "hold":
            log_event("decision", "hold_signal", {
                "market": best_market, "side": side, "confidence": confidence, "reason": reason, "features": features
            })
            push_heartbeat(prices_ok=True)
            push_pet_and_equity()
            log.info("Skip trade | market=%s side=%s conf=%.3f reason=%s", best_market, side, confidence, reason)
            time.sleep(CYCLE_SECONDS)
            continue

        if confidence < CONF_MIN_TRADE:
            log_event("decision", "low_confidence_gate", {
                "market": best_market, "side": side, "confidence": confidence, "min": CONF_MIN_TRADE,
                "reason": reason, "features": features
            })
            push_heartbeat(prices_ok=True)
            push_pet_and_equity()
            log.info("Skip trade (low confidence) | market=%s side=%s conf=%.3f min=%.3f", best_market, side, confidence, CONF_MIN_TRADE)
            time.sleep(CYCLE_SECONDS)
            continue

        # =========================
        # BRAIN V2 (gate + risk multiplier)
        # =========================
        brain_state = "OFF"
        brain_reason = ""
        brain_risk_mult = 1.0

        if USE_BRAIN_V2 and brain is not None:
            try:
                indicators = build_indicators_from_features(features)
                decision = brain.decide(indicators=indicators, signal_score=float(confidence))
                brain_state = decision.brain_state
                brain_reason = decision.reason
                brain_risk_mult = float(decision.risk_multiplier or 1.0)

                if not bool(decision.allow_trade):
                    log_event("decision", "brain_blocked_trade", {
                        "market": best_market,
                        "side": side,
                        "confidence": confidence,
                        "reason": reason,
                        "features": features,
                        "brain_state": brain_state,
                        "brain_reason": brain_reason,
                        "brain_risk_mult": brain_risk_mult,
                    })
                    push_heartbeat(prices_ok=True)
                    push_pet_and_equity()
                    log.info("Brain blocked | market=%s side=%s conf=%.3f brain=%s", best_market, side, confidence, brain_state)
                    time.sleep(CYCLE_SECONDS)
                    continue

            except Exception as e:
                brain_state = "BRAIN_ERROR"
                brain_reason = f"{e}"
                brain_risk_mult = 1.0
                log.warning("BrainV2 decide error (fail-open): %s", e)

        # =========================
        # RISK / SIZE
        # =========================
        size_mult = confidence_to_size_mult(confidence)
        base_risk_usd = state["equity"] * RISK_PER_TRADE
        risk_usd = base_risk_usd * size_mult * brain_risk_mult

        strict_loss_streak = confidence_to_strict_loss_streak(confidence)

        log_event("decision", "trade_decision", {
            "market": best_market,
            "side": side,
            "confidence": confidence,
            "size_mult": size_mult,
            "risk_usd": risk_usd,
            "strict_loss_streak": strict_loss_streak,
            "reason": reason,
            "features": features,
            "brain_active": bool(USE_BRAIN_V2 and brain is not None),
            "brain_state": brain_state,
            "brain_reason": brain_reason,
            "brain_risk_multiplier": brain_risk_mult,
        })

        # =========================
        # PAPER TRADE + FEEDBACK
        # =========================
        pnl = simulate_trade(risk_usd=risk_usd)

        if USE_BRAIN_V2 and brain is not None:
            try:
                brain.record_trade(float(pnl), equity_after=float(state["equity"]))
            except Exception as e:
                log.warning("BrainV2 record_trade failed: %s", e)

        pet_on_trade(pnl, confidence)
        dd = drawdown_pct()

        log.info(
            "SCAN→SELECT %s | %s %s | conf=%.2f score=%.4f | PnL %.2f | Eq %.2f | DD %.2f%% | LS %d | Brain %s",
            best_market,
            best_market, side.upper(),
            confidence,
            float(features.get("score") or 0.0),
            pnl, state["equity"], dd,
            state["loss_streak"],
            brain_state
        )

        # =========================
        # CRYO TRIGGERS
        # =========================
        if state["loss_streak"] >= strict_loss_streak:
            enter_cryo("loss_streak", {
                "market": best_market,
                "loss_streak": state["loss_streak"],
                "threshold": strict_loss_streak,
                "confidence": confidence,
                "drawdown_pct": dd,
                "equity": state["equity"],
                "side": side,
                "reason": reason,
                "brain_state": brain_state,
                "brain_reason": brain_reason,
            })

        if dd >= MAX_DRAWDOWN_PCT:
            enter_cryo("max_drawdown", {
                "market": best_market,
                "drawdown_pct": dd,
                "peak": state["peak_equity"],
                "equity": state["equity"],
                "confidence": confidence,
                "side": side,
                "reason": reason,
                "brain_state": brain_state,
                "brain_reason": brain_reason,
            })

        push_heartbeat(prices_ok=True)
        push_pet_and_equity()

        # Trade log (+ brain metadata)
        _post("/ingest/trade", {
            "time_utc": utc_now(),
            "market": best_market,
            "side": side,
            "size_usd": float(risk_usd),
            "price": float(prices.get(best_market, 0.0)),
            "pnl_usd": float(pnl),
            "confidence": float(confidence),
            "reason": reason,
            "brain_state": brain_state,
            "brain_reason": brain_reason,
            "brain_risk_multiplier": float(brain_risk_mult),
        })

        time.sleep(CYCLE_SECONDS)

if __name__ == "__main__":
    run()

from dataclasses import dataclass
from typing import Dict, List
import math
import time

# ==========================
# Brain State Models
# ==========================

@dataclass
class MarketState:
    regime: str              # trending | ranging | volatile | danger
    volatility: float
    trend_strength: float
    confidence: float


@dataclass
class BrainDecision:
    allow_trade: bool
    confidence: float
    risk_multiplier: float
    brain_state: str
    reason: str


# ==========================
# Brain V2 Core
# ==========================

class BrainV2:
    def __init__(self):
        self.trade_history: List[Dict] = []
        self.last_decision_time = 0
        self.cooldown_sec = 20

        # survival memory
        self.loss_streak = 0
        self.win_streak = 0
        self.drawdown = 0.0

    # --------------------------
    # Market Regime Detection
    # --------------------------
    def detect_regime(self, indicators: Dict) -> MarketState:
        atr = indicators.get("atr", 0)
        ema_slope = indicators.get("ema_slope", 0)
        adx = indicators.get("adx", 0)

        if atr > indicators.get("atr_danger", 0.05):
            regime = "volatile"
        elif adx < 15:
            regime = "ranging"
        elif adx > 25 and abs(ema_slope) > 0:
            regime = "trending"
        else:
            regime = "neutral"

        confidence = min(1.0, max(0.1, adx / 50))
        return MarketState(
            regime=regime,
            volatility=atr,
            trend_strength=adx,
            confidence=confidence,
        )

    # --------------------------
    # Survival Logic
    # --------------------------
    def survival_mode(self) -> str:
        if self.drawdown > 0.15:
            return "EMERGENCY"
        if self.loss_streak >= 3:
            return "DEFENSIVE"
        if self.win_streak >= 3:
            return "AGGRESSIVE"
        return "NORMAL"

    # --------------------------
    # Decision Engine
    # --------------------------
    def decide(self, indicators: Dict, signal_score: float) -> BrainDecision:
        now = time.time()
        if now - self.last_decision_time < self.cooldown_sec:
            return BrainDecision(
                False, 0.0, 0.0, "COOLDOWN", "Decision cooldown"
            )

        market = self.detect_regime(indicators)
        survival = self.survival_mode()

        confidence = signal_score * market.confidence

        allow = confidence > 0.55
        risk_mult = 1.0

        # survival overrides
        if survival == "DEFENSIVE":
            risk_mult = 0.5
            allow = confidence > 0.65
        elif survival == "EMERGENCY":
            allow = False
            risk_mult = 0.0
        elif survival == "AGGRESSIVE":
            risk_mult = 1.25

        # regime filters
        if market.regime in ("volatile", "danger"):
            allow = False

        self.last_decision_time = now

        return BrainDecision(
            allow_trade=allow,
            confidence=round(confidence, 3),
            risk_multiplier=risk_mult,
            brain_state=f"{market.regime.upper()} / {survival}",
            reason=f"Signal={signal_score:.2f}, Regime={market.regime}, Survival={survival}",
        )

    # --------------------------
    # Learning Feedback
    # --------------------------
    def record_trade(self, pnl: float):
        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0

        self.drawdown = max(0.0, self.drawdown - pnl)

        self.trade_history.append({
            "pnl": pnl,
            "time": time.time()
        })

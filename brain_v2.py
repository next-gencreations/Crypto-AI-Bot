# brain_v2.py
from dataclasses import dataclass
from typing import Dict, Optional, Any
import math

@dataclass
class BrainDecision:
    allow_trade: bool
    confidence: float
    risk_multiplier: float
    reason: str

class BrainV2:
    """
    Decision brain with:
    - confidence gating
    - risk scaling
    - loss-streak protection
    - companion-pressure modifier (health/hunger)
    """

    def __init__(
        self,
        base_conf_min: float = 0.62,
        base_risk_mult: float = 1.0,
        max_risk_mult: float = 1.6,
        loss_streak_soft_limit: int = 3,
        loss_streak_hard_limit: int = 6,
    ):
        self.base_conf_min = base_conf_min
        self.base_risk_mult = base_risk_mult
        self.max_risk_mult = max_risk_mult
        self.loss_streak_soft_limit = loss_streak_soft_limit
        self.loss_streak_hard_limit = loss_streak_hard_limit

        self.loss_streak = 0
        self.win_streak = 0

    def update_result(self, pnl: float):
        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
        elif pnl < 0:
            self.loss_streak += 1
            self.win_streak = 0

    def _companion_pressure(self, companion: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """
        Convert companion needs into trade behaviour changes.
        SAFE version:
          - low health/hunger => raise confidence threshold
          - low health/hunger => reduce risk
        """
        if not companion:
            return {"conf_boost": 0.0, "risk_scale": 1.0, "pressure": 0.0}

        health = float(companion.get("health", 100))
        hunger = float(companion.get("hunger", 100))

        # pressure 0..1 (1 = urgent)
        # hunger <= 20 or health <= 30 begins to bite
        p_h = max(0.0, (30.0 - health) / 30.0)
        p_u = max(0.0, (20.0 - hunger) / 20.0)
        pressure = max(p_h, p_u)
        pressure = max(0.0, min(1.0, pressure))

        # Make it *more selective* when pressure high
        conf_boost = 0.10 * pressure  # up to +0.10 required confidence

        # Risk down when pressure high (do NOT size up)
        risk_scale = 1.0 - 0.35 * pressure  # down to 0.65x

        return {"conf_boost": conf_boost, "risk_scale": risk_scale, "pressure": pressure}

    def decide(
        self,
        signal_score: float,
        indicators: Optional[Dict[str, float]] = None,
        companion: Optional[Dict[str, Any]] = None,
    ) -> BrainDecision:
        """
        signal_score: -1..+1 (direction+strength)
        indicators: optional dict (rsi, trend, vol, etc) for reason text
        companion: {health,hunger,stage,mood}
        """

        # Base confidence from signal magnitude
        confidence = min(1.0, max(0.0, abs(signal_score)))

        # Companion pressure modifies behaviour
        cp = self._companion_pressure(companion)
        conf_min = min(0.90, self.base_conf_min + cp["conf_boost"])

        # Loss-streak protection: if losing, get stricter
        if self.loss_streak >= self.loss_streak_hard_limit:
            return BrainDecision(False, confidence, 0.0, f"blocked: hard loss streak ({self.loss_streak})")

        if self.loss_streak >= self.loss_streak_soft_limit:
            conf_min = min(0.92, conf_min + 0.06)

        # Decide allow/deny
        if confidence < conf_min:
            why = f"blocked: confidence {confidence:.2f} < {conf_min:.2f}"
            if cp["pressure"] > 0:
                why += f" (companion pressure {cp['pressure']:.2f})"
            return BrainDecision(False, confidence, 0.0, why)

        # Risk multiplier: small boost on win streak, reduce on loss streak, and scale down if companion pressure high
        risk = self.base_risk_mult

        if self.win_streak >= 3:
            risk *= 1.10
        if self.loss_streak >= 1:
            risk *= max(0.70, 1.0 - 0.12 * self.loss_streak)

        risk *= cp["risk_scale"]
        risk = max(0.25, min(self.max_risk_mult, risk))

        reason = "allowed"
        if cp["pressure"] > 0:
            reason += f" (companion pressure {cp['pressure']:.2f}, safer mode)"
        if self.loss_streak:
            reason += f" (loss_streak={self.loss_streak})"
        if self.win_streak:
            reason += f" (win_streak={self.win_streak})"

        return BrainDecision(True, confidence, risk, reason)

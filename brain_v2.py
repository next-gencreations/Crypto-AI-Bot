from dataclasses import dataclass
from typing import Dict, List, Optional
import time

# ==========================
# Brain State Models
# ==========================

@dataclass
class MarketState:
    regime: str              # trending | ranging | volatile | danger | neutral
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
    """
    BrainV2 is a gate + risk controller.
    It does NOT place trades. It only decides:
      - allow_trade (True/False)
      - risk_multiplier (0..)
      - meta: confidence, brain_state, reason
    """

    def __init__(self):
        self.trade_history: List[Dict] = []
        self.last_decision_time = 0.0

        # Base cooldown (will be adjusted dynamically)
        self.base_cooldown_sec = 20

        # survival memory
        self.loss_streak = 0
        self.win_streak = 0

        # equity-aware drawdown tracking (percent)
        self.equity = 1.0
        self.peak_equity = 1.0
        self.drawdown_pct = 0.0

        # behavior tuning
        self.min_conf_normal = 0.55
        self.min_conf_defensive = 0.65

        # volatility thresholds (ATR assumed as fraction, e.g. 0.01 = 1%)
        self.atr_volatile = 0.03
        self.atr_danger = 0.06

    # --------------------------
    # Market Regime Detection
    # --------------------------
    def detect_regime(self, indicators: Dict) -> MarketState:
        atr = float(indicators.get("atr", 0.0) or 0.0)
        ema_slope = float(indicators.get("ema_slope", 0.0) or 0.0)
        adx = float(indicators.get("adx", 0.0) or 0.0)

        # Allow external override thresholds
        atr_volatile = float(indicators.get("atr_volatile", self.atr_volatile) or self.atr_volatile)
        atr_danger = float(indicators.get("atr_danger", self.atr_danger) or self.atr_danger)

        if atr >= atr_danger:
            regime = "danger"
        elif atr >= atr_volatile:
            regime = "volatile"
        elif adx < 15:
            regime = "ranging"
        elif adx > 25 and abs(ema_slope) > 0:
            regime = "trending"
        else:
            regime = "neutral"

        # Market confidence from ADX (0.1..1.0)
        market_conf = min(1.0, max(0.1, adx / 50.0))

        return MarketState(
            regime=regime,
            volatility=atr,
            trend_strength=adx,
            confidence=market_conf,
        )

    # --------------------------
    # Survival Logic
    # --------------------------
    def survival_mode(self) -> str:
        if self.drawdown_pct >= 15.0:
            return "EMERGENCY"
        if self.loss_streak >= 3:
            return "DEFENSIVE"
        if self.win_streak >= 3:
            return "AGGRESSIVE"
        return "NORMAL"

    # --------------------------
    # Dynamic Cooldown
    # --------------------------
    def cooldown_for(self, market: MarketState, survival: str) -> int:
        cd = self.base_cooldown_sec

        if market.regime == "trending":
            cd = int(cd * 0.75)
        elif market.regime == "ranging":
            cd = int(cd * 1.10)
        elif market.regime == "volatile":
            cd = int(cd * 1.60)
        elif market.regime == "danger":
            cd = int(cd * 3.00)

        if survival == "DEFENSIVE":
            cd = int(cd * 1.50)
        elif survival == "EMERGENCY":
            cd = int(cd * 10.0)

        return max(5, cd)

    # --------------------------
    # Decision Engine
    # --------------------------
    def decide(self, indicators: Dict, signal_score: float) -> BrainDecision:
        now = time.time()

        market = self.detect_regime(indicators)
        survival = self.survival_mode()
        cooldown_sec = self.cooldown_for(market, survival)

        if now - self.last_decision_time < cooldown_sec:
            return BrainDecision(
                allow_trade=False,
                confidence=0.0,
                risk_multiplier=0.0,
                brain_state="COOLDOWN",
                reason=f"Decision cooldown ({cooldown_sec}s)"
            )

        # Combine signal with market quality
        s = float(signal_score or 0.0)
        confidence = s * market.confidence

        # Loss-streak penalty
        if self.loss_streak >= 2:
            confidence *= 0.90
        if self.loss_streak >= 3:
            confidence *= 0.80

        allow = confidence >= self.min_conf_normal
        risk_mult = 1.0

        # survival overrides
        if survival == "DEFENSIVE":
            risk_mult = 0.5
            allow = confidence >= self.min_conf_defensive
        elif survival == "EMERGENCY":
            allow = False
            risk_mult = 0.0
        elif survival == "AGGRESSIVE":
            risk_mult = 1.25

        # regime filters
        if market.regime == "volatile":
            allow = allow and (confidence >= 0.70)
            risk_mult *= 0.75
        if market.regime == "danger":
            allow = False
            risk_mult = 0.0

        self.last_decision_time = now

        return BrainDecision(
            allow_trade=bool(allow),
            confidence=round(float(confidence), 3),
            risk_multiplier=float(risk_mult),
            brain_state=f"{market.regime.upper()} / {survival}",
            reason=(
                f"Signal={s:.2f}, MarketConf={market.confidence:.2f}, "
                f"Regime={market.regime}, Survival={survival}, "
                f"DD={self.drawdown_pct:.1f}%, LS={self.loss_streak}, WS={self.win_streak}"
            ),
        )

    # --------------------------
    # Learning Feedback
    # --------------------------
    def record_trade(self, pnl: float, equity_after: Optional[float] = None):
        pnl = float(pnl or 0.0)

        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0

        # Equity-aware drawdown tracking (recommended)
        if equity_after is not None:
            self.equity = float(equity_after)
            self.peak_equity = max(self.peak_equity, self.equity)
            if self.peak_equity > 0:
                self.drawdown_pct = (self.peak_equity - self.equity) / self.peak_equity * 100.0
            else:
                self.drawdown_pct = 0.0

        self.trade_history.append({
            "pnl": pnl,
            "time": time.time(),
            "equity": self.equity,
            "drawdown_pct": self.drawdown_pct,
            "loss_streak": self.loss_streak,
            "win_streak": self.win_streak,
        })

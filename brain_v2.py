# brain_v2.py
# Simple online-learning gate for paper trading.
# Learns per-market outcomes and can reduce risk or veto trades when performance is poor.

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

@dataclass
class MarketStats:
    n: int = 0
    ewma_pnl: float = 0.0     # exponentially weighted average pnl in USD
    ewma_win: float = 0.5     # exponentially weighted win rate
    last_ts: float = 0.0

class BrainV2:
    """
    evaluate() returns:
      - allow: bool (gate trade)
      - confidence: float (adjusted 0..1)
      - risk_mult: float (risk multiplier adjustment)
      - note: str
    """

    def __init__(self,
                 enabled: bool = True,
                 state_path: str = "brain_state.json",
                 alpha: float = 0.12,           # learning rate for EWMA
                 min_trades_to_trust: int = 12, # needs some history before strong gating
                 hard_veto_winrate: float = 0.42,
                 soft_veto_winrate: float = 0.46):
        self.enabled = enabled
        self.state_path = state_path
        self.alpha = alpha
        self.min_trades_to_trust = min_trades_to_trust
        self.hard_veto_winrate = hard_veto_winrate
        self.soft_veto_winrate = soft_veto_winrate
        self.stats: Dict[str, MarketStats] = {}
        self._dirty = False
        self._last_save = 0.0
        self.load()

    def load(self) -> None:
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                raw = data.get("stats", {})
                for m, s in raw.items():
                    self.stats[m] = MarketStats(**s)
        except Exception:
            # Safe fail: no crash
            self.stats = {}

    def save(self, force: bool = False) -> None:
        if not self._dirty and not force:
            return
        now = time.time()
        if not force and (now - self._last_save) < 30:
            return
        try:
            payload = {"stats": {m: asdict(s) for m, s in self.stats.items()}}
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            self._dirty = False
            self._last_save = now
        except Exception:
            # Safe fail: no crash
            pass

    def evaluate(self,
                 market: str,
                 base_confidence: float,
                 pressure: float,
                 equity: float) -> dict:
        # If disabled, pass-through
        if not self.enabled:
            return {
                "allow": True,
                "confidence": clamp(base_confidence, 0.0, 1.0),
                "risk_mult": 1.0,
                "note": "brain_disabled",
            }

        base_confidence = clamp(base_confidence, 0.0, 1.0)

        s = self.stats.get(market, MarketStats())
        n = s.n
        win = clamp(s.ewma_win, 0.0, 1.0)

        # Start with neutral adjustments
        allow = True
        conf = base_confidence
        risk_adj = 1.0
        note = "neutral"

        # Don’t overreact until we have enough trades
        if n < self.min_trades_to_trust:
            # small gentle shaping only
            conf *= (0.95 + 0.10 * win)  # win 0.5 -> 1.0x, win 0.8 -> 1.03x
            risk_adj *= (0.90 + 0.25 * win)  # win 0.5 -> 1.02x, win 0.3 -> 0.98x
            note = f"warmup n={n} win={win:.2f}"
        else:
            # If winrate is poor, reduce risk and maybe veto weak signals
            if win <= self.hard_veto_winrate and base_confidence < 0.35:
                allow = False
                note = f"hard_veto n={n} win={win:.2f}"
            elif win <= self.soft_veto_winrate and base_confidence < 0.28:
                allow = False
                note = f"soft_veto n={n} win={win:.2f}"
            else:
                # Adjust confidence and risk based on winrate
                conf *= (0.85 + 0.30 * win)  # win 0.5 -> 1.00x, win 0.7 -> 1.06x, win 0.3 -> 0.94x
                # Stronger risk reduction when winrate is low
                risk_adj *= clamp(0.65 + 0.85 * win, 0.65, 1.25)
                note = f"learned n={n} win={win:.2f}"

        # Additional safety: if “pressure” is high (pet unhealthy), reduce risk slightly
        # pressure in your system is 1.0..1.35
        risk_adj *= clamp(1.05 - (pressure - 1.0) * 0.35, 0.75, 1.05)

        return {
            "allow": allow,
            "confidence": clamp(conf, 0.0, 1.0),
            "risk_mult": clamp(risk_adj, 0.50, 1.35),
            "note": note,
        }

    def update(self, market: str, pnl_usd: float) -> None:
        if not self.enabled:
            return
        s = self.stats.get(market)
        if not s:
            s = MarketStats()
            self.stats[market] = s

        s.n += 1
        win = 1.0 if pnl_usd > 0 else 0.0

        a = self.alpha
        # EWMA updates
        s.ewma_pnl = (1 - a) * s.ewma_pnl + a * float(pnl_usd)
        s.ewma_win = (1 - a) * s.ewma_win + a * win
        s.last_ts = time.time()

        self._dirty = True
        self.save()

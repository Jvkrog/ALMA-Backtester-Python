"""
╔══════════════════════════════════════════════════════════╗
║  risk_manager.py — Module 4: Position Sizing & Stops     ║
╚══════════════════════════════════════════════════════════╝

ATR-based dynamic stop-loss / take-profit with risk-percentage
position sizing.  Stops widen during volatile markets and
tighten in quiet markets — keeping risk constant in rupee terms.

    Stop distance = ATR × atr_multiplier
    TP   distance = Stop distance × reward_ratio
    Shares        = (capital × risk_pct%) / stop_distance_per_share
"""

import pandas as pd

from config import ALMAConfig


class RiskManager:
    """
    Calculates position sizes, stop-losses, and take-profits.

    Args:
        config: ALMAConfig with risk_pct / atr_period /
                atr_multiplier / reward_ratio set.

    Example:
        rm    = RiskManager(cfg)
        atr   = rm.compute_atr(df["high"], df["low"], df["close"])
        sl, tp = rm.compute_levels(entry_price=450.0,
                                    atr=8.5, direction="LONG")
        qty   = rm.position_size(capital=100_000,
                                  entry=450.0, stop=sl)
    """

    def __init__(self, config: ALMAConfig):
        self.cfg = config

    # ── Public API ────────────────────────────

    def compute_atr(
        self,
        high:  pd.Series,
        low:   pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """
        Average True Range (ATR) — Wilder's method.

        True Range = max of:
          • High − Low
          • |High − Prev Close|
          • |Low  − Prev Close|

        Args:
            high:  High price series.
            low:   Low price series.
            close: Close price series.

        Returns:
            pd.Series of ATR values named "ATR".
        """
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low  - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)

        return tr.rolling(self.cfg.atr_period).mean().rename("ATR")

    def compute_levels(
        self,
        entry:     float,
        atr:       float,
        direction: str,
    ) -> tuple[float, float]:
        """
        Calculate stop-loss and take-profit for a new trade.

        Args:
            entry:     Entry price.
            atr:       Current ATR value for this bar.
            direction: "LONG" or "SHORT".

        Returns:
            (stop_loss, take_profit) as a tuple of floats.
        """
        sl_dist = atr * self.cfg.atr_multiplier
        tp_dist = sl_dist * self.cfg.reward_ratio

        if direction == "LONG":
            stop_loss   = entry - sl_dist
            take_profit = entry + tp_dist
        else:
            stop_loss   = entry + sl_dist
            take_profit = entry - tp_dist

        return round(stop_loss, 4), round(take_profit, 4)

    def position_size(
        self,
        capital: float,
        entry:   float,
        stop:    float,
    ) -> float:
        """
        Risk-based position sizing.

        Formula:
            shares = (capital × risk_pct / 100) / |entry − stop|

        Args:
            capital: Available trading capital in ₹.
            entry:   Planned entry price per share.
            stop:    Stop-loss price per share.

        Returns:
            Number of shares (float; round down in live trading).
            Returns 0 if entry == stop (degenerate case).
        """
        risk_amount    = capital * (self.cfg.risk_pct / 100)
        per_share_risk = abs(entry - stop)
        if per_share_risk == 0:
            return 0.0
        return risk_amount / per_share_risk

    def max_loss(self, capital: float) -> float:
        """Maximum rupee loss allowed per trade at current config."""
        return capital * (self.cfg.risk_pct / 100)

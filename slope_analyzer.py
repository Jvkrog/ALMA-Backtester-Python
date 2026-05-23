"""
╔══════════════════════════════════════════════════════════╗
║  slope_analyzer.py — Module 2: Slope Strength & Color    ║
╚══════════════════════════════════════════════════════════╝

Measures the rate-of-change of the ALMA line and classifies
each bar as GREEN (bullish), RED (bearish), or GREY (sideways).

Slope is expressed as a percentage of the previous ALMA value
so that it is comparable across instruments of different price
magnitudes (e.g. RELIANCE ≈ ₹3000 vs SBIN ≈ ₹600).

    slope_pct[i] = (ALMA[i] − ALMA[i−1]) / ALMA[i−1] × 100
"""

import pandas as pd

from config import ALMAConfig, Color


class SlopeAnalyzer:
    """
    Computes slope and assigns a color label to each bar.

    Args:
        config: ALMAConfig with slope_bull_pct / slope_bear_pct set.

    Example:
        analyzer   = SlopeAnalyzer(cfg)
        slope      = analyzer.compute_slope(alma_series)
        colors     = analyzer.classify_series(slope)
        strength   = slope.apply(analyzer.slope_strength)
    """

    def __init__(self, config: ALMAConfig):
        self.cfg = config

    # ── Public API ────────────────────────────

    def compute_slope(self, alma: pd.Series) -> pd.Series:
        """
        Return percentage slope of the ALMA line.

        Args:
            alma: pd.Series of ALMA values.

        Returns:
            pd.Series named "slope_pct".
        """
        return (alma.pct_change() * 100).rename("slope_pct")

    def classify_color(self, slope_pct: float) -> Color:
        """
        Map a single slope value to a Color enum.

        Args:
            slope_pct: Float percentage slope.

        Returns:
            Color.GREEN  if slope_pct >  slope_bull_pct  → bullish
            Color.RED    if slope_pct <  slope_bear_pct  → bearish
            Color.GREY   otherwise                       → sideways
        """
        if slope_pct > self.cfg.slope_bull_pct:
            return Color.GREEN
        elif slope_pct < self.cfg.slope_bear_pct:
            return Color.RED
        return Color.GREY

    def classify_series(self, slope_series: pd.Series) -> pd.Series:
        """
        Apply classify_color to every element of a slope series.

        Args:
            slope_series: pd.Series of slope_pct values.

        Returns:
            pd.Series of Color enum values named "color".
        """
        return slope_series.apply(self.classify_color).rename("color")

    def slope_strength(self, slope_pct: float) -> float:
        """
        Normalise slope into a [0, 1] strength score.

        Strong trends (steep slope) → score near 1.0
        Flat / sideways              → score near 0.0

        Args:
            slope_pct: Float percentage slope.

        Returns:
            float in [0.0, 1.0]
        """
        threshold = max(
            abs(self.cfg.slope_bull_pct),
            abs(self.cfg.slope_bear_pct),
        )
        return min(abs(slope_pct) / (threshold * 10), 1.0)

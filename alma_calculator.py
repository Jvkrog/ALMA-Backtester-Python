"""
╔══════════════════════════════════════════════════════════╗
║  alma_calculator.py — Module 1: Pure ALMA Math Engine    ║
╚══════════════════════════════════════════════════════════╝

Arnaud Legoux Moving Average (ALMA)
────────────────────────────────────
Uses a Gaussian distribution kernel to weight prices.
Unlike EMA/SMA, the phase offset shifts the peak of the
bell curve, balancing lag vs noise.

Formula:
    m    = floor(offset × (window − 1))
    s    = window / sigma
    w[k] = exp(−(k − m)² / (2 × s²))
    ALMA = Σ(price[k] × w[k]) / Σ(w[k])
"""

import numpy as np
import pandas as pd

from config import ALMAConfig


class ALMACalculator:
    """
    Computes the Arnaud Legoux Moving Average for a price series.

    The Gaussian weights are pre-computed once at construction time
    and reused for every calculation, making batch and streaming
    use equally efficient.

    Args:
        config: ALMAConfig instance with window / sigma / offset set.

    Example:
        calc = ALMACalculator(ALMAConfig(window=14))
        alma_series = calc.calculate(df["close"])
        latest_alma = calc.update_single(df["close"].values[-14:])
    """

    def __init__(self, config: ALMAConfig):
        self.cfg      = config
        self._weights = self._precompute_weights()

    # ── Public API ────────────────────────────

    def calculate(self, prices: pd.Series) -> pd.Series:
        """
        Compute ALMA over the full price series (batch mode).

        Args:
            prices: pd.Series of close prices indexed by datetime.

        Returns:
            pd.Series named "ALMA" with NaN for the first window−1 bars.
        """
        n   = len(prices)
        w   = self.cfg.window
        arr = prices.values.astype(float)
        out = np.full(n, np.nan)

        for i in range(w - 1, n):
            out[i] = np.dot(arr[i - w + 1 : i + 1], self._weights)

        return pd.Series(out, index=prices.index, name="ALMA")

    def update_single(self, recent_prices: np.ndarray) -> float:
        """
        Compute ALMA for the latest bar only (live / streaming mode).

        Args:
            recent_prices: Last `window` prices as a numpy array.
                           Must have at least `window` elements.

        Returns:
            float ALMA value, or np.nan if insufficient data.
        """
        if len(recent_prices) < self.cfg.window:
            return float(np.nan)
        return float(np.dot(recent_prices[-self.cfg.window:], self._weights))

    # ── Private helpers ───────────────────────

    def _precompute_weights(self) -> np.ndarray:
        """Build and normalise the Gaussian weight vector."""
        w   = self.cfg.window
        m   = self.cfg.offset * (w - 1)
        s   = w / self.cfg.sigma
        k   = np.arange(w)
        wts = np.exp(-((k - m) ** 2) / (2 * s ** 2))
        return wts / wts.sum()          # normalise so weights sum to 1

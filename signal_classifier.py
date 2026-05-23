"""
╔══════════════════════════════════════════════════════════╗
║  signal_classifier.py — Module 3: Signal Generation     ║
╚══════════════════════════════════════════════════════════╝

Converts a series of Color labels into confirmed trade signals.

Confirmation rules:
  • BUY      → GREEN for ≥ confirm_bars consecutive bars
  • SELL     → RED   for ≥ confirm_bars consecutive bars
  • SIDEWAYS → any GREY bar (also resets confirmation counter)
  • NEUTRAL  → building confirmation, no actionable signal yet

Entry filter:
  get_signal_entries() further filters to only NEW signal
  transitions (GREEN→RED or RED→GREEN), preventing repeated
  entry signals on every bar of an ongoing trend.
"""

import pandas as pd

from config import ALMAConfig, Color, Signal


class SignalClassifier:
    """
    Converts Color sequences into confirmed BUY / SELL signals.

    Args:
        config: ALMAConfig with confirm_bars set.

    Example:
        classifier = SignalClassifier(cfg)
        signals    = classifier.generate_signals(colors)
        entries    = classifier.get_signal_entries(signals)
    """

    def __init__(self, config: ALMAConfig):
        self.cfg = config

    # ── Public API ────────────────────────────

    def generate_signals(self, colors: pd.Series) -> pd.Series:
        """
        Generate confirmed signals from a Color series.

        Args:
            colors: pd.Series of Color enum values (from SlopeAnalyzer).

        Returns:
            pd.Series of Signal enum values named "signal".
        """
        signals      = []
        streak_color = None
        streak_count = 0

        for color in colors:
            if pd.isna(color):
                signals.append(Signal.NEUTRAL)
                streak_color = None
                streak_count = 0
                continue

            if color == Color.GREY:
                signals.append(Signal.SIDEWAYS)
                streak_color = None
                streak_count = 0

            elif color == streak_color:
                streak_count += 1
                if streak_count >= self.cfg.confirm_bars:
                    signals.append(
                        Signal.BUY if color == Color.GREEN else Signal.SELL
                    )
                else:
                    signals.append(Signal.NEUTRAL)

            else:
                # New color — start building streak
                streak_color = color
                streak_count = 1
                signals.append(Signal.NEUTRAL)

        return pd.Series(signals, index=colors.index, name="signal")

    def get_signal_entries(self, signals: pd.Series) -> pd.Series:
        """
        Filter signals to ONLY new transitions (entry bars).

        Prevents re-entering on every bar of an existing trend:
        once a BUY fires, subsequent BUY bars are marked NEUTRAL
        until a SELL or SIDEWAYS interrupts the sequence.

        Args:
            signals: pd.Series from generate_signals().

        Returns:
            pd.Series of Signal values named "entry_signal"
            (only BUY / SELL on transition bars; NEUTRAL elsewhere).
        """
        entries = []
        prev    = Signal.NEUTRAL

        for sig in signals:
            if sig != prev and sig in (Signal.BUY, Signal.SELL):
                entries.append(sig)
            else:
                entries.append(Signal.NEUTRAL)
            prev = sig

        return pd.Series(entries, index=signals.index, name="entry_signal")

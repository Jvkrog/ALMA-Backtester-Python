"""
╔══════════════════════════════════════════════════════════╗
║  alma_strategy.py — Module 7: Orchestrator               ║
╚══════════════════════════════════════════════════════════╝

Wires all modules together and provides a clean public API
for both backtesting and live signal generation.

Typical usage
─────────────
    from config       import ALMAConfig
    from alma_strategy import ALMAStrategy

    cfg      = ALMAConfig(window=14, sigma=6.0, offset=0.85)
    strategy = ALMAStrategy(cfg)

    # ── Backtest with Kite historical data ──
    results = strategy.backtest_kite(
        symbol="RELIANCE", exchange="NSE", interval="day",
        days=365, capital=100_000
    )
    strategy.print_summary(results)

    # ── Live signal (latest bar) ────────────
    sig = strategy.live_signal(
        symbol="RELIANCE", exchange="NSE", interval="5minute"
    )
    print(sig["signal"], sig["entry"])
"""

from typing import Optional

import pandas as pd

from config           import ALMAConfig, Signal
from alma_calculator  import ALMACalculator
from slope_analyzer   import SlopeAnalyzer
from signal_classifier import SignalClassifier
from risk_manager     import RiskManager
from backtester       import Backtester


class ALMAStrategy:
    """
    Main entry point — orchestrates the full ALMA pipeline.

    Args:
        config:       ALMAConfig instance (or None for defaults).
        kite_fetcher: Optional KiteDataFetcher.  Required for
                      backtest_kite() and live_signal().

    Modules wired internally:
        ALMACalculator  → SlopeAnalyzer → SignalClassifier
        RiskManager     (used by Backtester)
        KiteDataFetcher (injected; used for data access)
    """

    def __init__(
        self,
        config:       Optional[ALMAConfig] = None,
        kite_fetcher = None,
    ):
        self.cfg          = config or ALMAConfig()
        self.alma_calc    = ALMACalculator(self.cfg)
        self.slope_anal   = SlopeAnalyzer(self.cfg)
        self.sig_class    = SignalClassifier(self.cfg)
        self.risk_manager = RiskManager(self.cfg)
        self.fetcher      = kite_fetcher        # KiteDataFetcher | None

    # ── Core pipeline ─────────────────────────

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full ALMA pipeline over a price DataFrame.

        Args:
            df: Must contain at least a 'close' column.
                'high' and 'low' are used by RiskManager.

        Returns:
            pd.DataFrame with columns:
                close, ALMA, slope_pct, color, strength,
                signal, entry_signal
        """
        close    = df["close"]
        alma     = self.alma_calc.calculate(close)
        slope    = self.slope_anal.compute_slope(alma)
        colors   = self.slope_anal.classify_series(slope)
        strength = slope.apply(self.slope_anal.slope_strength)
        signals  = self.sig_class.generate_signals(colors)
        entries  = self.sig_class.get_signal_entries(signals)

        return pd.DataFrame({
            "close":        close,
            "ALMA":         alma,
            "slope_pct":    slope,
            "color":        colors,
            "strength":     strength,
            "signal":       signals,
            "entry_signal": entries,
        })

    # ── Signal helpers ────────────────────────

    def latest_signal(self, df: pd.DataFrame) -> dict:
        """
        Get the most recent bar's signal from an existing DataFrame.

        Args:
            df: OHLCV DataFrame (pre-fetched).

        Returns:
            dict with keys: bar, close, alma, slope_pct,
                            color, strength, signal, entry
        """
        result = self.run(df)
        last   = result.iloc[-1]
        return {
            "bar":       df.index[-1],
            "close":     last["close"],
            "alma":      last["ALMA"],
            "slope_pct": last["slope_pct"],
            "color":     last["color"],
            "strength":  last["strength"],
            "signal":    last["signal"],
            "entry":     last["entry_signal"],
        }

    # ── Kite-integrated methods ───────────────

    def live_signal(
        self,
        symbol:   str,
        exchange: str  = "NSE",
        interval: str  = "day",
        n_bars:   int  = 50,
    ) -> dict:
        """
        Fetch latest bars from Kite and return the current signal.

        Args:
            symbol:   Trading symbol, e.g. "RELIANCE".
            exchange: "NSE" | "BSE" | "NFO".
            interval: Kite interval string.
            n_bars:   Minimum bars needed (buffer added internally).

        Returns:
            Same dict as latest_signal().

        Raises:
            RuntimeError if no kite_fetcher was provided.
        """
        self._require_fetcher()
        df = self.fetcher.fetch_latest_bars(
            symbol=symbol, interval=interval,
            n_bars=n_bars, exchange=exchange,
        )
        return self.latest_signal(df)

    def backtest_kite(
        self,
        symbol:   str,
        exchange: str   = "NSE",
        interval: str   = "day",
        days:     int   = 365,
        capital:  float = 100_000,
    ) -> dict:
        """
        Fetch historical data from Kite and run a full backtest.

        Args:
            symbol:   Trading symbol.
            exchange: Exchange name.
            interval: Kite interval string.
            days:     Calendar days of history to fetch.
            capital:  Initial capital in ₹.

        Returns:
            Backtest results dict (same as backtest()).
        """
        self._require_fetcher()
        df = self.fetcher.fetch(
            symbol=symbol, interval=interval,
            days=days, exchange=exchange,
        )
        return self.backtest(df, capital=capital)

    def backtest(
        self,
        df:      pd.DataFrame,
        capital: float = 100_000,
    ) -> dict:
        """
        Run backtest on a pre-loaded OHLCV DataFrame.

        Args:
            df:      OHLCV DataFrame.
            capital: Initial capital in ₹.

        Returns:
            Backtest results dict from Backtester._stats().
        """
        bt = Backtester(self)
        return bt.run(df, initial_capital=capital)

    def print_summary(self, results: dict) -> None:
        """Delegate pretty-print to Backtester."""
        bt = Backtester(self)
        bt.print_summary(results)

    # ── Private helpers ───────────────────────

    def _require_fetcher(self):
        if self.fetcher is None:
            raise RuntimeError(
                "A KiteDataFetcher instance must be passed to ALMAStrategy "
                "as `kite_fetcher=` to use live / Kite-integrated methods."
            )

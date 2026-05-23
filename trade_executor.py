"""
╔══════════════════════════════════════════════════════════╗
║  trade_executor.py — Module 5: Order & P&L Management   ║
╚══════════════════════════════════════════════════════════╝

Manages the lifecycle of a single open trade:
  • enter_trade   — open a LONG or SHORT position
  • check_exit    — test SL / TP hits bar-by-bar
  • force_close   — close at market on signal reversal
  • P&L           — recorded on the TradeRecord

Note: This module tracks trades in memory for backtesting.
      Yahoo Finance is a data-only API and does not support
      live order placement.  For live trading, wire the
      signals from ALMAStrategy into your broker's API
      (e.g. Alpaca, Interactive Brokers, Angel One, etc.).
"""

import logging
from typing import Optional

from config import TradeRecord

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  BACKTEST TRADE EXECUTOR
# ─────────────────────────────────────────────

class TradeExecutor:
    """
    Manages open trades and P&L tracking during backtesting.

    Attributes:
        trades:     List of all TradeRecord objects (open & closed).
        open_trade: The currently open TradeRecord, or None.

    Example:
        ex = TradeExecutor()
        ex.enter_trade(bar=10, price=450.0, direction="LONG",
                       stop=440.0, tp=470.0)
        closed = ex.check_exit(bar=11, high=471.0, low=448.0, close=470.0)
    """

    def __init__(self):
        self.trades:     list[TradeRecord]       = []
        self.open_trade: Optional[TradeRecord]   = None

    # ── Public API ────────────────────────────

    def enter_trade(
        self,
        bar:       int,
        price:     float,
        direction: str,
        stop:      float,
        tp:        float,
    ) -> TradeRecord:
        """
        Open a new trade.  If a trade is already open, force-close it first.

        Args:
            bar:       Bar index (integer position in DataFrame).
            price:     Entry price.
            direction: "LONG" or "SHORT".
            stop:      Stop-loss price.
            tp:        Take-profit price.

        Returns:
            The new TradeRecord.
        """
        if self.open_trade and self.open_trade.status == "OPEN":
            logger.debug("Forcing close of existing trade before new entry.")
            self.force_close(bar, price)

        trade = TradeRecord(
            entry_bar   = bar,
            entry_price = price,
            direction   = direction,
            stop_loss   = stop,
            take_profit = tp,
        )
        self.open_trade = trade
        self.trades.append(trade)
        logger.debug("Entered %s at %.4f | SL=%.4f TP=%.4f",
                     direction, price, stop, tp)
        return trade

    def check_exit(
        self,
        bar:   int,
        high:  float,
        low:   float,
        close: float,
    ) -> Optional[TradeRecord]:
        """
        Check whether the open trade hit its stop-loss or take-profit.

        Args:
            bar:   Current bar index.
            high:  Bar high price.
            low:   Bar low price.
            close: Bar close price (used as fallback exit).

        Returns:
            Closed TradeRecord if an exit was triggered, else None.
        """
        t = self.open_trade
        if not t or t.status != "OPEN":
            return None

        hit_tp = hit_sl = False

        if t.direction == "LONG":
            hit_sl = low  <= t.stop_loss
            hit_tp = high >= t.take_profit
        else:
            hit_sl = high >= t.stop_loss
            hit_tp = low  <= t.take_profit

        if hit_tp:
            return self._close(bar, t.take_profit, "WIN")
        elif hit_sl:
            return self._close(bar, t.stop_loss, "LOSS")
        return None

    def force_close(self, bar: int, price: float) -> Optional[TradeRecord]:
        """
        Close the open trade at market price (signal reversal or EOD).

        Args:
            bar:   Current bar index.
            price: Market price at which to close.

        Returns:
            Closed TradeRecord, or None if no trade was open.
        """
        if self.open_trade and self.open_trade.status == "OPEN":
            return self._close(bar, price, "NEUTRAL")
        return None

    # ── Private helpers ───────────────────────

    def _close(self, bar: int, price: float, status: str) -> TradeRecord:
        t            = self.open_trade
        t.exit_bar   = bar
        t.exit_price = price
        t.status     = status
        mult         = 1 if t.direction == "LONG" else -1
        t.pnl        = mult * (price - t.entry_price)
        self.open_trade = None
        logger.debug("Closed %s @ %.4f | P&L=%.4f | %s",
                     t.direction, price, t.pnl, status)
        return t

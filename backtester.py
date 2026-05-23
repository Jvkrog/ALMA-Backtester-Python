"""
╔══════════════════════════════════════════════════════════╗
║  backtester.py — Module 6: Historical Performance        ║
╚══════════════════════════════════════════════════════════╝

Runs the full ALMA strategy pipeline over historical OHLCV data
on HOURLY bars, and prints a per-day summary at market close.

Stats returned:
  • Equity curve (one point per hour bar)
  • Daily EOD summary table  (date, trades, day P&L, cum capital)
  • Total trades, win rate
  • Total return %, max drawdown %
  • Avg win / avg loss / reward-risk ratio
  • Final capital
"""

import numpy as np
import pandas as pd

from config         import Signal
from risk_manager   import RiskManager
from trade_executor import TradeExecutor


class Backtester:
    """
    Simulates the ALMA strategy on a historical OHLCV DataFrame.

    When the DataFrame has sub-daily frequency (e.g. hourly), the
    backtester automatically groups bars by calendar date and prints
    an EOD summary after each day's last bar is processed.

    Args:
        strategy: An initialised ALMAStrategy orchestrator instance.

    Example:
        bt      = Backtester(strategy)
        results = bt.run(df, initial_capital=100_000)
        bt.print_summary(results)
    """

    def __init__(self, strategy):
        self.strategy = strategy

    # ── Public API ────────────────────────────

    def run(
        self,
        df:              pd.DataFrame,
        initial_capital: float = 100_000,
    ) -> dict:
        """
        Run the backtest bar-by-bar.

        Args:
            df:              DataFrame with [open, high, low, close, volume]
                             at any frequency (hourly recommended).
            initial_capital: Starting equity.

        Returns:
            dict with keys:
                signals, equity_curve, trades, daily_summary,
                total_trades, win_rate_pct, total_return_pct,
                max_drawdown_pct, avg_win, avg_loss,
                reward_risk_ratio, final_capital
        """
        s       = self.strategy
        signals = s.run(df)
        rm: RiskManager = s.risk_manager
        ex      = TradeExecutor()

        capital = initial_capital
        equity  = [capital]

        atr = rm.compute_atr(df["high"], df["low"], df["close"])

        # ── Detect whether bars are intraday ──────────────────────
        dates       = df.index.normalize()
        is_intraday = (dates != df.index).any()

        daily_log: list[dict] = []
        day_start_capital     = capital
        day_trades_closed: list = []
        current_date          = dates[0] if is_intraday else None

        for i, (ts, row) in enumerate(df.iterrows()):
            bar_date = pd.Timestamp(ts).normalize()

            # ── Day boundary: flush previous day EOD ──────────────
            if is_intraday and bar_date != current_date and current_date is not None:
                daily_log.append(self._day_entry(
                    current_date, day_trades_closed,
                    day_start_capital, capital,
                ))
                day_start_capital = capital
                day_trades_closed = []
                current_date      = bar_date

            bar_atr = atr.iloc[i]
            sig     = signals["entry_signal"].iloc[i]

            # ── 1. Check exits first ───────────────────────────────
            closed = ex.check_exit(i, row["high"], row["low"], row["close"])
            if closed and closed.pnl is not None:
                qty      = rm.position_size(capital, closed.entry_price,
                                            closed.stop_loss)
                capital += closed.pnl * qty
                day_trades_closed.append(closed)

            # ── 2. New entries ─────────────────────────────────────
            if sig == Signal.BUY and not ex.open_trade:
                sl, tp = rm.compute_levels(row["close"], bar_atr, "LONG")
                ex.enter_trade(i, row["close"], "LONG", sl, tp)

            elif sig == Signal.SELL and not ex.open_trade:
                sl, tp = rm.compute_levels(row["close"], bar_atr, "SHORT")
                ex.enter_trade(i, row["close"], "SHORT", sl, tp)

            # ── 3. Signal reversal → force close ───────────────────
            elif (sig == Signal.BUY and ex.open_trade
                  and ex.open_trade.direction == "SHORT"):
                fc = ex.force_close(i, row["close"])
                if fc and fc.pnl is not None:
                    qty      = rm.position_size(capital, fc.entry_price, fc.stop_loss)
                    capital += fc.pnl * qty
                    day_trades_closed.append(fc)

            elif (sig == Signal.SELL and ex.open_trade
                  and ex.open_trade.direction == "LONG"):
                fc = ex.force_close(i, row["close"])
                if fc and fc.pnl is not None:
                    qty      = rm.position_size(capital, fc.entry_price, fc.stop_loss)
                    capital += fc.pnl * qty
                    day_trades_closed.append(fc)

            equity.append(capital)

        # ── Flush the last day ─────────────────────────────────────
        if is_intraday and current_date is not None:
            daily_log.append(self._day_entry(
                current_date, day_trades_closed,
                day_start_capital, capital,
            ))

        daily_summary = pd.DataFrame(daily_log) if daily_log else pd.DataFrame()

        return self._stats(
            ex.trades, equity, initial_capital, signals, daily_summary
        )

    # ── Print helpers ─────────────────────────

    def print_summary(self, results: dict) -> None:
        """Pretty-print EOD daily table + overall backtest statistics."""
        r   = results
        sep = "═" * 72

        # ── Per-day EOD table ──────────────────────────────────────
        ds: pd.DataFrame = r.get("daily_summary", pd.DataFrame())
        if not ds.empty:
            print(f"\n{sep}")
            print("  ALMA STRATEGY — DAILY EOD SUMMARY  (Hourly Bars)")
            print(sep)
            print(
                f"  {'Date':<12}  {'Trades':>6}  {'Wins':>4}  {'Losses':>6}  "
                f"{'Day P&L':>12}  {'Day Ret%':>9}  {'Cum Capital':>14}"
            )
            print("  " + "─" * 68)
            for _, row in ds.iterrows():
                pnl_sign = "+" if row["day_pnl"] >= 0 else ""
                ret_sign = "+" if row["day_ret_pct"] >= 0 else ""
                print(
                    f"  {str(row['date']):<12}  "
                    f"{int(row['trades']):>6}  "
                    f"{int(row['wins']):>4}  "
                    f"{int(row['losses']):>6}  "
                    f"{pnl_sign}{row['day_pnl']:>11,.2f}  "
                    f"{ret_sign}{row['day_ret_pct']:>8.2f}%  "
                    f"{row['cum_capital']:>14,.2f}"
                )

        # ── Overall stats ──────────────────────────────────────────
        print(f"\n{sep}")
        print("  ALMA STRATEGY — OVERALL BACKTEST RESULTS")
        print(sep)
        print(f"  Total Trades          : {r['total_trades']}")
        print(f"  Win Rate              : {r['win_rate_pct']:.2f}%")
        print(f"  Total Return          : {r['total_return_pct']:.2f}%")
        print(f"  Max Drawdown          : {r['max_drawdown_pct']:.2f}%")
        print(f"  Avg Win / Avg Loss    : {r['avg_win']:.4f} / {r['avg_loss']:.4f}")
        print(f"  Reward : Risk         : {r['reward_risk_ratio']:.2f}")
        print(f"  Final Capital         : {r['final_capital']:>14,.2f}")
        print(f"{sep}\n")

    # ── Private helpers ───────────────────────

    @staticmethod
    def _day_entry(
        date,
        closed_trades: list,
        start_capital: float,
        end_capital:   float,
    ) -> dict:
        """Build one row for the daily_summary DataFrame."""
        wins    = [t for t in closed_trades if t.status == "WIN"]
        losses  = [t for t in closed_trades if t.status == "LOSS"]
        day_pnl = end_capital - start_capital
        day_ret = (day_pnl / start_capital * 100) if start_capital else 0.0
        return {
            "date":        str(date.date()),
            "trades":      len(closed_trades),
            "wins":        len(wins),
            "losses":      len(losses),
            "day_pnl":     round(day_pnl, 2),
            "day_ret_pct": round(day_ret, 4),
            "cum_capital": round(end_capital, 2),
        }

    @staticmethod
    def _stats(trades, equity, initial_capital, signals, daily_summary) -> dict:
        wins   = [t for t in trades if t.status == "WIN"]
        losses = [t for t in trades if t.status == "LOSS"]
        total  = len(trades)

        equity_arr = np.array(equity)
        win_rate   = len(wins) / total * 100 if total else 0.0
        final_eq   = equity[-1] if equity else initial_capital
        total_ret  = (final_eq - initial_capital) / initial_capital * 100

        peak   = np.maximum.accumulate(equity_arr)
        dd     = (equity_arr - peak) / peak * 100
        max_dd = float(dd.min())

        avg_win  = float(np.mean([t.pnl for t in wins]))   if wins   else 0.0
        avg_loss = float(np.mean([t.pnl for t in losses]))  if losses else 0.0
        rr       = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        return {
            "signals":           signals,
            "equity_curve":      equity,
            "trades":            trades,
            "daily_summary":     daily_summary,
            "total_trades":      total,
            "win_rate_pct":      round(win_rate, 2),
            "total_return_pct":  round(total_ret, 2),
            "max_drawdown_pct":  round(max_dd, 2),
            "avg_win":           round(avg_win, 4),
            "avg_loss":          round(avg_loss, 4),
            "reward_risk_ratio": round(rr, 2),
            "final_capital":     round(final_eq, 2),
        }

"""
╔══════════════════════════════════════════════════════════╗
║  plotter.py — Chart Visualisation                        ║
╚══════════════════════════════════════════════════════════╝

Produces a 3-panel chart identical to the reference UI:

  Panel 1 (top)    — Price + ALMA line (green/red/grey colored)
                     with BUY ▲ / SELL ▼ entry markers
                     and shaded background per regime

  Panel 2 (middle) — Slope strength histogram
                     (green = bullish, red = bearish, grey = sideways)

  Panel 3 (bottom) — Equity curve

Install:
    pip install matplotlib

Usage:
    from plotter import Plotter
    plotter = Plotter()
    plotter.plot(results, title="RELIANCE.NS — 1h")
    # or save to file:
    plotter.plot(results, title="RELIANCE.NS", save_path="chart.png")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

from config import Color, Signal


# ── Dark theme colours (matches the reference UI) ─────────
_BG        = "#0d1117"
_PANEL_BG  = "#161b22"
_GRID      = "#21262d"
_TEXT      = "#c9d1d9"
_GREEN     = "#00ff88"
_RED       = "#ff3366"
_GREY      = "#555566"
_PRICE     = "#888899"
_ALMA_COL  = "#9b59b6"   # purple — neutral ALMA reference
_EQUITY    = "#ff4444"
_BUY_MRK   = "#00ff88"
_SELL_MRK  = "#ff3366"


class Plotter:
    """
    Visualises backtest results produced by ALMAStrategy / Backtester.

    Args:
        figsize:  Matplotlib figure size (width, height) in inches.
        dpi:      Figure resolution.

    Example:
        from plotter import Plotter
        p = Plotter()
        p.plot(results, title="RELIANCE.NS — 60m backtest")
    """

    def __init__(self, figsize=(16, 11), dpi=130):
        self.figsize = figsize
        self.dpi     = dpi

    # ── Public API ────────────────────────────

    def plot(
        self,
        results:   dict,
        title:     str  = "ALMA Strategy",
        save_path: str  = "",
        show:      bool = True,
    ) -> None:
        """
        Render the 3-panel chart.

        Args:
            results:   Dict returned by ALMAStrategy.backtest() or
                       Backtester.run().  Must contain keys:
                         'signals'     — DataFrame from ALMAStrategy.run()
                         'equity_curve' — list of float
                         'trades'      — list of TradeRecord
            title:     Chart title string.
            save_path: If non-empty, saves PNG to this path.
            show:      If True, calls plt.show() (set False for headless).
        """
        sig_df  = results["signals"]
        equity  = results["equity_curve"]
        trades  = results["trades"]

        # ── Stats for header ──────────────────────────────────────
        total_ret = results.get("total_return_pct", 0.0)
        win_rate  = results.get("win_rate_pct", 0.0)
        max_dd    = results.get("max_drawdown_pct", 0.0)
        n_trades  = results.get("total_trades", 0)

        # ── Figure setup ──────────────────────────────────────────
        plt.style.use("dark_background")
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi,
                         facecolor=_BG)
        fig.subplots_adjust(hspace=0.08, top=0.91, bottom=0.06,
                            left=0.07, right=0.97)

        gs = GridSpec(3, 1, figure=fig,
                      height_ratios=[3, 1, 1.2])

        ax_price  = fig.add_subplot(gs[0])
        ax_slope  = fig.add_subplot(gs[1], sharex=ax_price)
        ax_equity = fig.add_subplot(gs[2])

        for ax in (ax_price, ax_slope, ax_equity):
            ax.set_facecolor(_PANEL_BG)
            ax.tick_params(colors=_TEXT, labelsize=8)
            ax.spines[:].set_color(_GRID)
            ax.yaxis.label.set_color(_TEXT)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

        x = np.arange(len(sig_df))

        # ─────────────────────────────────────────────────────────
        #  PANEL 1 — Price + ALMA
        # ─────────────────────────────────────────────────────────
        self._draw_price_panel(ax_price, sig_df, x, trades)

        # ─────────────────────────────────────────────────────────
        #  PANEL 2 — Slope Strength Histogram
        # ─────────────────────────────────────────────────────────
        self._draw_slope_panel(ax_slope, sig_df, x)

        # ─────────────────────────────────────────────────────────
        #  PANEL 3 — Equity Curve
        # ─────────────────────────────────────────────────────────
        self._draw_equity_panel(ax_equity, equity)

        # Hide x-tick labels on top two panels (shared x-axis)
        plt.setp(ax_price.get_xticklabels(), visible=False)
        plt.setp(ax_slope.get_xticklabels(), visible=False)

        # ── Master title + stats bar ──────────────────────────────
        ret_color = _GREEN if total_ret >= 0 else _RED
        dd_color  = _RED

        fig.text(0.07, 0.955, title,
                 color=_TEXT, fontsize=13, fontweight="bold")

        stats_str = (
            f"Trades: {n_trades}    "
            f"Win Rate: {win_rate:.1f}%    "
        )
        fig.text(0.07, 0.935, stats_str, color=_TEXT, fontsize=9)

        fig.text(0.07 + len(stats_str) * 0.0058, 0.935,
                 f"Return: {total_ret:+.2f}%",
                 color=ret_color, fontsize=9, fontweight="bold")

        fig.text(0.07 + len(stats_str) * 0.0058 + 0.115, 0.935,
                 f"    Max DD: {max_dd:.2f}%",
                 color=dd_color, fontsize=9)

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight",
                        facecolor=_BG)
            print(f"Chart saved → {save_path}")

        if show:
            plt.show()

        plt.close(fig)

    # ── Panel drawing helpers ─────────────────

    def _draw_price_panel(self, ax, sig_df, x, trades):
        """Price line, coloured ALMA segments, background shading, markers."""

        close = sig_df["close"].values
        alma  = sig_df["ALMA"].values
        color_series = sig_df["color"].values

        # ── Background shading per regime ─────────────────────────
        i = 0
        while i < len(color_series):
            c = color_series[i]
            j = i
            while j < len(color_series) and color_series[j] == c:
                j += 1
            if c == Color.GREEN:
                ax.axvspan(i, j, alpha=0.07, color=_GREEN, linewidth=0)
            elif c == Color.RED:
                ax.axvspan(i, j, alpha=0.07, color=_RED,   linewidth=0)
            i = j

        # ── Price line ────────────────────────────────────────────
        ax.plot(x, close, color=_PRICE, linewidth=0.9,
                alpha=0.7, zorder=2, label="Price")

        # ── ALMA line — coloured segments ─────────────────────────
        _color_map = {
            Color.GREEN: _GREEN,
            Color.RED:   _RED,
            Color.GREY:  _GREY,
        }
        i = 0
        while i < len(x):
            c = color_series[i]
            j = i
            while j < len(color_series) and color_series[j] == c:
                j += 1
            seg_x    = x[i:j+1]
            seg_alma = alma[i:j+1]
            col      = _color_map.get(c, _GREY)
            ax.plot(seg_x, seg_alma, color=col,
                    linewidth=2.0, zorder=3)
            i = j if j > i else j + 1

        # ── Trade entry markers ───────────────────────────────────
        entry_sig = sig_df["entry_signal"].values
        for xi, sig in zip(x, entry_sig):
            if sig == Signal.BUY:
                ax.scatter(xi, close[xi], marker="^",
                           color=_BUY_MRK, s=60, zorder=5)
            elif sig == Signal.SELL:
                ax.scatter(xi, close[xi], marker="v",
                           color=_SELL_MRK, s=60, zorder=5)

        # ── Legend ────────────────────────────────────────────────
        legend_elements = [
            Line2D([0], [0], color=_GREEN,   lw=2,  label="BUY (green slope)"),
            Line2D([0], [0], color=_RED,     lw=2,  label="SELL (red slope)"),
            Line2D([0], [0], color=_GREY,    lw=2,  label="SIDEWAYS (grey)"),
            Line2D([0], [0], color=_PRICE,   lw=1,  label="Price"),
            Line2D([0], [0], marker="^", color=_BUY_MRK,
                   lw=0, markersize=7, label="BUY entry"),
            Line2D([0], [0], marker="v", color=_SELL_MRK,
                   lw=0, markersize=7, label="SELL entry"),
        ]
        ax.legend(handles=legend_elements, loc="upper left",
                  fontsize=7.5, framealpha=0.3,
                  facecolor=_PANEL_BG, edgecolor=_GRID,
                  labelcolor=_TEXT, ncol=3)

        ax.set_ylabel("Price", color=_TEXT, fontsize=9)
        ax.grid(color=_GRID, linewidth=0.4, alpha=0.6)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:,.2f}")
        )

    def _draw_slope_panel(self, ax, sig_df, x):
        """Slope strength histogram coloured green / red / grey."""

        slope  = sig_df["slope_pct"].values
        colors = sig_df["color"].values

        bar_colors = []
        for c in colors:
            if c == Color.GREEN:
                bar_colors.append(_GREEN)
            elif c == Color.RED:
                bar_colors.append(_RED)
            else:
                bar_colors.append(_GREY)

        ax.bar(x, slope, color=bar_colors, width=0.8, alpha=0.85, zorder=2)
        ax.axhline(0, color=_GRID, linewidth=0.6)
        ax.set_ylabel("Slope %", color=_TEXT, fontsize=8)
        ax.grid(color=_GRID, linewidth=0.3, alpha=0.5, axis="y")

        ax.text(0.01, 0.92,
                "Slope strength  ·  green=bullish  ·  red=bearish  ·  grey=sideways",
                transform=ax.transAxes, color=_TEXT,
                fontsize=7.5, va="top")

        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.2f}%")
        )

    def _draw_equity_panel(self, ax, equity):
        """Equity curve as filled area."""

        eq = np.array(equity)
        x  = np.arange(len(eq))

        ax.plot(x, eq, color=_EQUITY, linewidth=1.5, zorder=3)
        ax.fill_between(x, eq, eq.min(), alpha=0.15,
                        color=_EQUITY, zorder=2)

        # Peak line
        peak = np.maximum.accumulate(eq)
        ax.plot(x, peak, color=_TEXT, linewidth=0.6,
                linestyle="--", alpha=0.4, zorder=2)

        ax.set_ylabel("Equity", color=_TEXT, fontsize=9)
        ax.set_xlabel("Bar", color=_TEXT, fontsize=9)
        ax.grid(color=_GRID, linewidth=0.4, alpha=0.6)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k")
        )

        ax.text(0.01, 0.92, "Equity curve",
                transform=ax.transAxes, color=_TEXT,
                fontsize=7.5, va="top")

"""
╔══════════════════════════════════════════════════════════╗
║  config.py — Enums, Data Classes, ALMAConfig             ║
╚══════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────
#  ENUMS
# ─────────────────────────────────────────────

class Signal(Enum):
    BUY      = "BUY"
    SELL     = "SELL"
    SIDEWAYS = "SIDEWAYS"
    NEUTRAL  = "NEUTRAL"


class Color(Enum):
    GREEN = "#00ff88"   # Positive slope → BUY
    RED   = "#ff3366"   # Negative slope → SELL
    GREY  = "#888899"   # Flat slope     → SIDEWAYS


# ─────────────────────────────────────────────
#  ALMA CONFIG
# ─────────────────────────────────────────────

@dataclass
class ALMAConfig:
    """All tunable strategy parameters in one place."""

    # ── ALMA core ──────────────────────────────
    window:          int   = 14       # Lookback period
    sigma:           float = 6.0      # Gaussian width  (higher = smoother)
    offset:          float = 0.85     # Phase shift     (0–1; 0.85 = responsive)

    # ── Slope thresholds (as % of ALMA price) ──
    slope_bull_pct:  float = 0.05     # Above this  → GREEN / BUY
    slope_bear_pct:  float = -0.05    # Below this  → RED   / SELL

    # ── Signal confirmation ─────────────────────
    confirm_bars:    int   = 2        # Consecutive same-color bars before signal fires

    # ── Risk management ─────────────────────────
    risk_pct:        float = 1.0      # % of capital risked per trade
    atr_period:      int   = 14       # ATR lookback for stop-loss
    atr_multiplier:  float = 2.0      # Stop = entry ± ATR × multiplier
    reward_ratio:    float = 2.0      # Take-profit = stop distance × ratio

    # ── Yahoo Finance data ──────────────────────
    # exchange is kept for ticker-suffix resolution (NSE → .NS, BSE → .BO)
    exchange:        str   = "NSE"    # Exchange hint: NSE / BSE / NYSE / NASDAQ / ""
    interval:        str   = "day"    # Interval: minute/5minute/15minute/30minute/
                                      #   60minute/day/week/month  — or native yfinance
                                      #   strings: 1m/5m/15m/30m/1h/1d/1wk/1mo


# ─────────────────────────────────────────────
#  TRADE RECORD
# ─────────────────────────────────────────────

@dataclass
class TradeRecord:
    entry_bar:   int
    entry_price: float
    direction:   str            # "LONG" | "SHORT"
    stop_loss:   float
    take_profit: float
    exit_bar:    Optional[int]   = None
    exit_price:  Optional[float] = None
    pnl:         Optional[float] = None
    status:      str = "OPEN"   # "OPEN" | "WIN" | "LOSS" | "NEUTRAL"

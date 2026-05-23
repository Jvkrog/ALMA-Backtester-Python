"""
╔══════════════════════════════════════════════════════════╗
║  main.py — Entry Point & Usage Examples                  ║
╚══════════════════════════════════════════════════════════╝

Run modes
─────────
    python main.py --mode demo          # synthetic data, no network
    python main.py --mode backtest      # Yahoo Finance hourly backtest + EOD summary
    python main.py --mode live          # Yahoo Finance latest hourly signal

Prerequisites
─────────────
    pip install yfinance pandas numpy matplotlib

─────────────────────────────────────────────────────────
  INSTRUMENT REFERENCE — change SYMBOL below
─────────────────────────────────────────────────────────

  INDIAN STOCKS (NSE)        INDIAN STOCKS (BSE)
  ─────────────────────      ─────────────────────
  RELIANCE.NS                RELIANCE.BO
  TCS.NS                     TCS.BO
  INFY.NS                    INFY.BO
  HDFCBANK.NS                HDFCBANK.BO
  ICICIBANK.NS               SBIN.BO
  WIPRO.NS
  BAJFINANCE.NS

  INDIAN INDICES             US STOCKS
  ─────────────────────      ─────────────────────
  ^NSEI   (Nifty 50)         AAPL   MSFT   GOOGL
  ^BSESN  (Sensex)           AMZN   META   NVDA
  ^NSEBANK (Bank Nifty)      SPY    QQQ    TSLA

  US INDICES                 CRYPTO
  ─────────────────────      ─────────────────────
  ^GSPC  (S&P 500)           BTC-USD
  ^DJI   (Dow Jones)         ETH-USD
  ^IXIC  (Nasdaq)            SOL-USD

  FOREX                      COMMODITIES
  ─────────────────────      ─────────────────────
  EURUSD=X                   GC=F   (Gold)
  USDINR=X                   CL=F   (Crude Oil)
  GBPUSD=X                   SI=F   (Silver)

─────────────────────────────────────────────────────────
  INTERVAL REFERENCE
─────────────────────────────────────────────────────────
  "1m"   1-minute    (last 7 days only)
  "5m"   5-minute    (last 60 days)
  "15m"  15-minute   (last 60 days)
  "30m"  30-minute   (last 60 days)
  "60m"  1-hour  ←── default here  (last 730 days)
  "1d"   daily       (years of history)
  "1wk"  weekly
  "1mo"  monthly
─────────────────────────────────────────────────────────
"""

import argparse
import logging

import numpy as np
import pandas as pd

from config        import ALMAConfig, Color
from data_fetcher  import YahooDataFetcher
from alma_strategy import ALMAStrategy
from plotter       import Plotter

# ─────────────────────────────────────────────
#  ANSI COLOR CODES
# ─────────────────────────────────────────────

_ANSI = {
    "green":  "\033[92m",
    "red":    "\033[91m",
    "grey":   "\033[90m",
    "yellow": "\033[93m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

def _colorize(text: str, color: str) -> str:
    return f"{_ANSI.get(color, '')}{text}{_ANSI['reset']}"


# ─────────────────────────────────────────────
#  ALMA COLOR → terminal color mapping
# ─────────────────────────────────────────────

_ALMA_COLOR_MAP = {
    Color.GREEN: "green",
    Color.RED:   "red",
    Color.GREY:  "grey",
}

def _alma_color_str(color) -> str:
    """Return a terminal-colored ALMA color label."""
    if color is None or (isinstance(color, float) and np.isnan(color)):
        return _colorize("GREY  (SIDEWAYS)", "grey")
    ansi = _ALMA_COLOR_MAP.get(color, "grey")
    label = {
        Color.GREEN: "GREEN  (BULLISH ▲)",
        Color.RED:   "RED    (BEARISH ▼)",
        Color.GREY:  "GREY   (SIDEWAYS ─)",
    }.get(color, str(color))
    return _colorize(label, ansi)


# ─────────────────────────────────────────────
#  LOGGING — clean format
# ─────────────────────────────────────────────

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────
#  ← CHANGE THESE TO SWITCH INSTRUMENTS
# ─────────────────────────────────────────────

SYMBOL   = "AAPL"   # See instrument table above
INTERVAL = "60m"           # See interval table above
DAYS     = 60              # How many calendar days of history to fetch
CAPITAL  = 100_000         # Starting capital

# ─────────────────────────────────────────────
#  STRATEGY CONFIG
# ─────────────────────────────────────────────

DEFAULT_CONFIG = ALMAConfig(
    window         = 14,
    sigma          = 6.0,
    offset         = 0.85,
    slope_bull_pct = 0.04,
    slope_bear_pct = -0.04,
    confirm_bars   = 2,
    risk_pct       = 1.0,
    atr_period     = 14,
    atr_multiplier = 2.0,
    reward_ratio   = 2.0,
    exchange       = "",
    interval       = INTERVAL,
)


# ─────────────────────────────────────────────
#  SIGNAL PRINTER  (shared by all modes)
# ─────────────────────────────────────────────

def print_signal(sig: dict, symbol: str = "", interval: str = ""):
    sep = "─" * 46
    print(f"\n{sep}")
    if symbol:
        print(f"  {_colorize('Instrument', 'bold')} : {symbol}  [{interval}]")
    print(f"  {_colorize('Bar        ', 'bold')} : {sig['bar']}")
    print(f"  {_colorize('Close      ', 'bold')} : {sig['close']:.2f}")
    print(f"  {_colorize('ALMA       ', 'bold')} : {sig['alma']:.2f}")
    print(f"  {_colorize('Slope      ', 'bold')} : {sig['slope_pct']:.4f}%")
    print(f"  {_colorize('ALMA Color ', 'bold')} : {_alma_color_str(sig['color'])}")
    print(f"  {_colorize('Strength   ', 'bold')} : {sig['strength']:.3f}")

    signal_val = sig['signal'].value
    entry_val  = sig['entry'].value
    sig_color  = "green" if signal_val == "BUY" else ("red" if signal_val == "SELL" else "grey")
    ent_color  = "green" if entry_val  == "BUY" else ("red" if entry_val  == "SELL" else "grey")
    print(f"  {_colorize('Signal     ', 'bold')} : {_colorize(signal_val, sig_color)}")
    print(f"  {_colorize('Entry Flag ', 'bold')} : {_colorize(entry_val,  ent_color)}")
    print(f"{sep}\n")


# ─────────────────────────────────────────────
#  MODE: DEMO  (synthetic hourly, no network)
# ─────────────────────────────────────────────

def run_demo():
    logger.info("Running DEMO mode with synthetic hourly data …")

    np.random.seed(42)
    trading_days = pd.bdate_range("2024-01-01", periods=30)
    hours        = range(9, 16)
    timestamps   = [
        pd.Timestamp(f"{d.date()} {h:02d}:00:00")
        for d in trading_days for h in hours
    ]
    n      = len(timestamps)
    close  = 1000 + np.cumsum(np.random.randn(n) * 3)
    spread = np.abs(np.random.randn(n)) * 1.5

    df = pd.DataFrame({
        "open":   close - spread * 0.3,
        "high":   close + spread,
        "low":    close - spread,
        "close":  close,
        "volume": np.random.randint(100_000, 1_000_000, n),
    }, index=pd.DatetimeIndex(timestamps))

    strategy = ALMAStrategy(config=DEFAULT_CONFIG)
    sig      = strategy.latest_signal(df)
    print_signal(sig, symbol="DEMO", interval="1h")

    results = strategy.backtest(df, capital=CAPITAL)
    strategy.print_summary(results)

    Plotter().plot(results, title="ALMA Strategy — DEMO (synthetic hourly)")


# ─────────────────────────────────────────────
#  MODE: BACKTEST  (Yahoo Finance)
# ─────────────────────────────────────────────

def run_backtest():
    logger.info("Fetching %s | interval=%s | %d days …", SYMBOL, INTERVAL, DAYS)

    fetcher  = YahooDataFetcher()
    strategy = ALMAStrategy(config=DEFAULT_CONFIG, kite_fetcher=fetcher)

    results = strategy.backtest_kite(
        symbol   = SYMBOL,
        exchange = "",
        interval = INTERVAL,
        days     = DAYS,
        capital  = CAPITAL,
    )
    strategy.print_summary(results)

    Plotter().plot(
        results,
        title     = f"ALMA Strategy — {SYMBOL}  [{INTERVAL}]",
        save_path = f"chart_{SYMBOL.replace('.','_').replace('^','')}.png",
    )

    # Show last 10 bars with colored ALMA line in terminal
    sig_df = results["signals"]
    print(_colorize("  Last 10 hourly bars:", "bold"))
    print(f"  {'Timestamp':<22} {'Close':>8}  {'ALMA':>8}  {'Slope%':>8}  {'Color':<22}  Signal")
    print("  " + "─" * 82)
    for ts, row in sig_df.tail(10).iterrows():
        color_str = _alma_color_str(row["color"]).ljust(22)
        sig_val   = row["entry_signal"].value
        sig_color = "green" if sig_val == "BUY" else ("red" if sig_val == "SELL" else "grey")
        print(
            f"  {str(ts):<22}  "
            f"{row['close']:>8.2f}  "
            f"{row['ALMA']:>8.2f}  "
            f"{row['slope_pct']:>8.4f}  "
            f"{color_str}  "
            f"{_colorize(sig_val, sig_color)}"
        )
    print()


# ─────────────────────────────────────────────
#  MODE: LIVE  (Yahoo Finance latest bar)
# ─────────────────────────────────────────────

def run_live():
    logger.info("Fetching live signal: %s | %s", SYMBOL, INTERVAL)

    fetcher  = YahooDataFetcher()
    strategy = ALMAStrategy(config=DEFAULT_CONFIG, kite_fetcher=fetcher)

    sig = strategy.live_signal(
        symbol   = SYMBOL,
        exchange = "",
        interval = INTERVAL,
        n_bars   = DEFAULT_CONFIG.window + 20,
    )
    print_signal(sig, symbol=SYMBOL, interval=INTERVAL)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ALMA Strategy — Yahoo Finance Edition"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "backtest", "live"],
        default="demo",
        help="Run mode (default: demo)",
    )
    args = parser.parse_args()

    {
        "demo":     run_demo,
        "backtest": run_backtest,
        "live":     run_live,
    }[args.mode]()


if __name__ == "__main__":
    main()

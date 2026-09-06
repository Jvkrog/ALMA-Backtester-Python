# ALMA Backtester — Python

A modular Python backtesting framework for an **ALMA (Arnaud Legoux Moving Average) trend-following strategy**.

The project turns ALMA slope into market-state colors, confirms directional signals, applies ATR-based risk management, simulates trade execution, and produces performance statistics and charts. It also supports fetching market data through **Yahoo Finance / yfinance** and generating the latest strategy signal.

## What it does

The core pipeline is:

```text
OHLCV Data
    ↓
ALMA Calculator
    ↓
Slope Analyzer
    ↓
Color Classification
    ↓
Signal Confirmation
    ↓
Risk Manager
    ↓
Backtester / Trade Executor
    ↓
Performance Statistics + Charts
```

### ALMA

The ALMA implementation uses a Gaussian weighting kernel with configurable **window**, **sigma**, and **offset** parameters. The Gaussian weights are pre-computed once and reused for subsequent calculations.

### Slope-based regime detection

The slope of ALMA is expressed as a percentage change from the previous ALMA value. It is then classified into three states:

- **GREEN** — bullish slope
- **RED** — bearish slope
- **GREY** — sideways / flat slope

A normalized slope-strength value is also produced on a 0–1 scale.

### Signal confirmation

Signals are not triggered by a single color change. The classifier requires a configurable number of consecutive bars of the same directional color before producing a BUY or SELL signal.

Entry signals are further filtered so an ongoing BUY or SELL state does not repeatedly generate new entries on every bar.

### Risk management

Risk management is ATR-based and configurable through a single `ALMAConfig` object.

- ATR-based stop-loss
- ATR × multiplier stop distance
- Reward-to-risk based take-profit
- Risk-per-trade position sizing
- Maximum rupee risk calculation

### Backtesting

The backtester processes historical OHLCV bars sequentially and simulates a single open trade at a time.

It supports:

- Stop-loss and take-profit exits
- Signal-reversal exits
- Equity-curve tracking
- Daily end-of-day summaries for intraday data
- Win rate
- Total return
- Maximum drawdown
- Average win / average loss
- Reward-to-risk ratio
- Final capital

### Visualisation

`plotter.py` produces a three-panel chart containing:

1. **Price + ALMA** with directional coloring and BUY/SELL markers
2. **ALMA slope strength** histogram
3. **Equity curve** with peak reference

Charts can be displayed interactively or saved as PNG files.

## Project structure

```text
ALMA-Backtester-Python/
│
├── alma_calculator.py    # ALMA mathematical engine
├── alma_strategy.py      # Main strategy orchestrator
├── backtester.py         # Historical simulation + performance stats
├── config.py             # Configuration, enums, and trade records
├── data_fetcher.py       # Yahoo Finance / yfinance OHLCV provider
├── plotter.py            # Strategy, slope, and equity visualisation
├── risk_manager.py       # ATR, stops, targets, position sizing
├── signal_classifier.py  # Signal confirmation and entry filtering
├── slope_analyzer.py     # ALMA slope and regime classification
├── trade_executor.py     # Backtest trade lifecycle and P&L
├── symbols.txt           # Example supported Yahoo Finance symbols
├── images/               # Project images / generated charts
└── main.py               # CLI entry point and usage examples
```

## Requirements

Python 3.9+ is recommended.

Install the main dependencies with:

```bash
pip install yfinance pandas numpy matplotlib
```

## Quick start

### 1. Run the synthetic demo

The demo uses generated hourly OHLCV data and does not require network access.

```bash
python main.py --mode demo
```

### 2. Run a historical backtest

```bash
python main.py --mode backtest
```

The default configuration in `main.py` uses:

```text
Symbol      : AAPL
Interval    : 60m
History     : 60 days
Capital     : 100,000
ALMA Window : 14
Sigma       : 6.0
Offset      : 0.85
Confirmation: 2 bars
Risk        : 1.0% per trade
ATR Period  : 14
ATR Stop    : 2.0 × ATR
Reward/Risk : 2.0
```

Change `SYMBOL`, `INTERVAL`, `DAYS`, `CAPITAL`, or the `ALMAConfig` values in `main.py` to test another market or parameter set.

### 3. Generate the latest signal

```bash
python main.py --mode live
```

This fetches recent bars and prints the latest:

- Close
- ALMA
- ALMA slope
- ALMA color
- Slope strength
- Signal
- Entry flag

## Using the library directly

The modules can also be used without the CLI.

```python
from config import ALMAConfig
from alma_strategy import ALMAStrategy

cfg = ALMAConfig(
    window=14,
    sigma=6.0,
    offset=0.85,
    slope_bull_pct=0.04,
    slope_bear_pct=-0.04,
    confirm_bars=2,
    risk_pct=1.0,
    atr_period=14,
    atr_multiplier=2.0,
    reward_ratio=2.0,
)

strategy = ALMAStrategy(cfg)

# df must contain OHLCV columns
results = strategy.backtest(df, capital=100_000)
strategy.print_summary(results)
```

For data-backed operation, `ALMAStrategy` accepts a fetcher instance and exposes `backtest_kite()` / `live_signal()` methods that use the repository's Yahoo Finance data provider implementation.

## Supported symbols

The repository examples include markets from several asset classes:

| Asset class | Examples |
|---|---|
| Indian stocks | `TCS.NS`, `HDFCBANK.NS`, `INFY.NS` |
| Indian indices | `^NSEI`, `^NSEBANK` |
| US stocks | `AAPL`, `NVDA` |
| Crypto | `BTC-USD`, `ETH-USD` |
| Commodities | `GC=F`, `CL=F` |
| Forex | `EURUSD=X`, `USDINR=X` |

`data_fetcher.py` also maps common exchange hints such as NSE/BSE to Yahoo Finance suffixes.

## Strategy parameters

All main strategy parameters live in the `ALMAConfig` dataclass, keeping the strategy configuration centralized and easy to experiment with.

### ALMA parameters

| Parameter | Purpose |
|---|---|
| `window` | ALMA lookback period |
| `sigma` | Gaussian width / smoothing |
| `offset` | Phase offset of the Gaussian kernel |

### Signal parameters

| Parameter | Purpose |
|---|---|
| `slope_bull_pct` | Minimum slope for GREEN classification |
| `slope_bear_pct` | Maximum slope for RED classification |
| `confirm_bars` | Consecutive directional bars required for confirmation |

### Risk parameters

| Parameter | Purpose |
|---|---|
| `risk_pct` | Capital percentage risked per trade |
| `atr_period` | ATR lookback |
| `atr_multiplier` | Stop distance multiplier |
| `reward_ratio` | Take-profit distance relative to stop distance |

## Design notes

The repository is intentionally split into small modules rather than putting the entire strategy into one script. This makes it easier to test or replace individual components such as the ALMA calculation, signal logic, risk model, data provider, or execution simulator.

The ALMA calculator supports both batch calculation and latest-window calculation for streaming-style use. The backtester is designed around bar-by-bar processing, while the trade executor keeps the simulated open-trade lifecycle explicit.

## Data and execution limitations

This project is a **research and backtesting tool**. The Yahoo Finance provider supplies market data only; it does not place live orders.

Backtest results are simulations and depend on the supplied historical data, strategy parameters, and execution assumptions. They should not be interpreted as a guarantee of future performance.

The repository currently models trade exits from OHLC bar information. Real execution can differ because of slippage, spread, liquidity, order latency, partial fills, gaps, and broker-specific behavior.

## Development focus

This project is part of an ongoing exploration of systematic trading and strategy engineering, with emphasis on:

- deterministic signal generation
- modular strategy architecture
- explicit risk management
- reproducible backtesting
- clear separation between market data, strategy logic, and execution

## License

No license file is currently defined in the repository. Treat the source as **all rights reserved** unless a license is added by the repository owner.

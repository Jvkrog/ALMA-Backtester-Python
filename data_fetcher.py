"""
╔══════════════════════════════════════════════════════════╗
║  data_fetcher.py — Yahoo Finance OHLCV Data Provider     ║
╚══════════════════════════════════════════════════════════╝

Wraps yfinance to fetch historical OHLCV bars and return a
clean pandas DataFrame ready for the ALMA pipeline.

Install:
    pip install yfinance pandas

Symbol format:
    Indian stocks  → "RELIANCE.NS"  (NSE)  or "RELIANCE.BO" (BSE)
    US stocks      → "AAPL", "MSFT", "SPY"
    Indices        → "^NSEI" (Nifty 50), "^BSESN" (Sensex)
    Crypto         → "BTC-USD"
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  INTERVAL MAPPING  (yfinance strings)
# ─────────────────────────────────────────────

# Maps human-friendly names → yfinance interval strings
# yfinance valid intervals:
#   1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo 3mo
_INTERVAL_MAP = {
    # Kite-style aliases kept for backward compat
    "minute":   "1m",
    "3minute":  "2m",    # closest yfinance equivalent
    "5minute":  "5m",
    "15minute": "15m",
    "30minute": "30m",
    "60minute": "60m",
    "day":      "1d",
    "week":     "1wk",
    "month":    "1mo",
    # Native yfinance strings pass through unchanged
    "1m":  "1m",  "2m":  "2m",  "5m":  "5m",
    "15m": "15m", "30m": "30m", "60m": "60m", "90m": "90m",
    "1h":  "1h",  "1d":  "1d",  "5d":  "5d",
    "1wk": "1wk", "1mo": "1mo", "3mo": "3mo",
}

# yfinance caps intraday history at 60 days (1m → 7 days)
_INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}


def _suffix_for_exchange(exchange: str) -> str:
    """Return Yahoo Finance ticker suffix for common exchanges."""
    mapping = {
        "NSE": ".NS",
        "BSE": ".BO",
        "NYSE": "",
        "NASDAQ": "",
        "": "",
    }
    return mapping.get(exchange.upper(), "")


# ─────────────────────────────────────────────
#  YAHOO FINANCE DATA FETCHER
# ─────────────────────────────────────────────

class YahooDataFetcher:
    """
    Fetches OHLCV data from Yahoo Finance (via yfinance) and returns
    a normalised DataFrame with columns: [open, high, low, close, volume].

    Args:
        auto_adjust: If True (default), prices are adjusted for splits &
                     dividends.  Set False to get raw prices.

    Example:
        fetcher = YahooDataFetcher()
        df = fetcher.fetch("RELIANCE.NS", interval="day", days=365)

        # Or for US stocks
        df = fetcher.fetch("AAPL", interval="1d", days=180)
    """

    def __init__(self, auto_adjust: bool = True):
        try:
            import yfinance  # noqa: F401, PLC0415
        except ImportError as exc:
            raise ImportError(
                "yfinance package not found. "
                "Install it with: pip install yfinance"
            ) from exc
        self._auto_adjust = auto_adjust
        logger.info("YahooDataFetcher initialised (auto_adjust=%s).", auto_adjust)

    # ── Public API ────────────────────────────

    def fetch(
        self,
        symbol:    str,
        interval:  str  = "day",
        days:      int  = 365,
        exchange:  str  = "",
        from_date: Optional[datetime] = None,
        to_date:   Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a symbol.

        Args:
            symbol:    Yahoo Finance ticker, e.g. "RELIANCE.NS", "AAPL".
                       If `exchange` is provided and the symbol has no dot-
                       suffix, the suffix is appended automatically.
            interval:  Interval string — accepts both Kite-style ("day",
                       "5minute") and native yfinance ("1d", "5m").
            days:      Calendar days of history when from/to are not given.
            exchange:  Optional exchange hint: "NSE" → ".NS", "BSE" → ".BO".
                       Ignored when the symbol already contains a dot suffix.
            from_date: Override start date.
            to_date:   Override end date (default = today).

        Returns:
            pd.DataFrame with DatetimeIndex and columns
            [open, high, low, close, volume]

        Raises:
            ValueError: on invalid interval or empty response.
        """
        import yfinance as yf  # noqa: PLC0415

        yf_interval = self._resolve_interval(interval)
        ticker      = self._resolve_ticker(symbol, exchange)

        to_dt   = to_date   or datetime.now()
        from_dt = from_date or (to_dt - timedelta(days=days))

        # yfinance needs string dates for daily+; datetime objects for intraday
        start = from_dt.strftime("%Y-%m-%d")
        end   = to_dt.strftime("%Y-%m-%d")

        logger.info(
            "Fetching %s | interval=%s | %s → %s",
            ticker, yf_interval, start, end,
        )

        raw = yf.download(
            tickers     = ticker,
            start       = start,
            end         = end,
            interval    = yf_interval,
            auto_adjust = self._auto_adjust,
            progress    = False,
        )

        # Older yfinance versions return a MultiIndex columns when a single
        # ticker is passed — flatten to simple column names.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        if raw is None or raw.empty:
            raise ValueError(
                f"No data returned for '{ticker}' "
                f"(interval={yf_interval}, {start} – {end}). "
                "Check the ticker symbol and date range."
            )

        return self._to_dataframe(raw)

    def fetch_latest_bars(
        self,
        symbol:   str,
        interval: str = "day",
        n_bars:   int = 50,
        exchange: str = "",
    ) -> pd.DataFrame:
        """
        Convenience wrapper: fetch the last `n_bars` bars.
        Adds a 20-bar safety buffer so ALMA warmup never runs short.

        Args:
            symbol:   Ticker symbol.
            interval: Interval string.
            n_bars:   Minimum bars needed (20 added internally).
            exchange: Optional exchange suffix hint.
        """
        yf_interval  = self._resolve_interval(interval)
        buffer_bars  = n_bars + 20

        # Rough days-per-bar estimates for calendar day calculation
        interval_days_map = {
            "1m":  1,  "2m":  1,  "5m":  1,
            "15m": 2,  "30m": 3,  "60m": 5,  "90m": 5,  "1h": 5,
            "1d":  2,  "5d": 10,
            "1wk": 14, "1mo": 45, "3mo": 135,
        }
        days_per_bar  = interval_days_map.get(yf_interval, 2)
        calendar_days = buffer_bars * days_per_bar

        return self.fetch(symbol, interval=interval,
                          days=calendar_days, exchange=exchange)

    def get_info(self, symbol: str, exchange: str = "") -> dict:
        """
        Return basic ticker metadata (name, sector, currency, etc.).

        Useful for verifying a symbol is valid before fetching history.
        """
        import yfinance as yf  # noqa: PLC0415
        ticker = self._resolve_ticker(symbol, exchange)
        return yf.Ticker(ticker).info

    # ── Private helpers ───────────────────────

    def _resolve_ticker(self, symbol: str, exchange: str) -> str:
        """Append exchange suffix if the symbol has none."""
        if "." in symbol or "-" in symbol or "^" in symbol:
            # Already a fully-qualified Yahoo ticker
            return symbol.upper()
        suffix = _suffix_for_exchange(exchange)
        return (symbol + suffix).upper()

    @staticmethod
    def _resolve_interval(interval: str) -> str:
        """Map Kite-style or native interval → canonical yfinance string."""
        yf_interval = _INTERVAL_MAP.get(interval.lower())
        if yf_interval is None:
            raise ValueError(
                f"Invalid interval '{interval}'. "
                f"Supported: {sorted(_INTERVAL_MAP.keys())}"
            )
        return yf_interval

    @staticmethod
    def _to_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise yfinance output to a clean OHLCV DataFrame.

        yfinance returns columns: Open, High, Low, Close, Volume
        (Adj Close is merged into Close when auto_adjust=True).
        """
        df = raw.copy()

        # Lowercase all column names for consistency
        df.columns = [c.lower() for c in df.columns]

        # Keep only the columns we need
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in ohlcv_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Expected columns missing from Yahoo Finance data: {missing}. "
                f"Available: {list(df.columns)}"
            )

        df = df[ohlcv_cols].copy()
        df = df.apply(pd.to_numeric, errors="coerce")
        df.dropna(subset=["close"], inplace=True)

        # Ensure index is a proper DatetimeIndex (tz-naive for simplicity)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df.sort_index(inplace=True)

        logger.info("Fetched %d bars.", len(df))
        return df


# ─────────────────────────────────────────────
#  BACKWARD-COMPAT ALIAS
# ─────────────────────────────────────────────
# Old code that imported KiteDataFetcher will still work:
#   from data_fetcher import KiteDataFetcher
KiteDataFetcher = YahooDataFetcher

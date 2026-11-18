"""Daily bars from Alpaca (free IEX feed), cached to CSV."""
from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import CACHE_DIR


def fetch_daily_bars(client, symbols: list[str], history_days: int) -> dict[str, pd.DataFrame]:
    """Return {symbol: DataFrame[open, high, low, close, volume]} indexed by date."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=datetime.now(timezone.utc) - timedelta(days=history_days),
        adjustment=Adjustment.ALL,   # split + dividend adjusted
        feed=DataFeed.IEX,           # free plan
    )
    raw = client.get_stock_bars(req).df
    out = {}
    for sym in symbols:
        df = raw.xs(sym, level="symbol")[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_convert("America/New_York").normalize().tz_localize(None)
        df.index.name = "date"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_DIR / f"{sym}.csv")
        out[sym] = df
    return out


def load_cached(symbol: str) -> pd.DataFrame:
    return pd.read_csv(CACHE_DIR / f"{symbol}.csv", index_col="date", parse_dates=True)

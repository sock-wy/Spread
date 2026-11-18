"""Download 1-minute SIP bars from Alpaca and save RTH bars as CSV.

    python scripts/fetch_alpaca_1m.py --symbols IWM SPY --start 2019-01-01 --end 2026-11-14

Output: data/<SYMBOL>_1m.csv with columns datetime, open, high, low, close, volume.
`datetime` = bar END time, America/New_York, naive (09:31 ... 16:00), matching
imif.data.load_csv / filter_rth. Prices are RAW (not dividend-adjusted): the
strategy is flat overnight, so raw prices give honest intraday P&L.
Needs APCA_API_KEY_ID / APCA_API_SECRET_KEY in the environment.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imif.data import filter_rth  # noqa: E402


def fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"])
    chunks = []
    # one request per year keeps responses manageable
    for y0 in pd.date_range(start, end, freq="YS").append(pd.DatetimeIndex([pd.Timestamp(start)])).unique().sort_values():
        y1 = min(pd.Timestamp(end), pd.Timestamp(year=y0.year + 1, month=1, day=1))
        if y0 >= y1:
            continue
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=y0, end=y1,
                               feed=DataFeed.SIP, adjustment=Adjustment.RAW)
        df = client.get_stock_bars(req).df
        if len(df):
            chunks.append(df.xs(symbol, level="symbol"))
        print(f"  {symbol} {y0.date()} -> {y1.date()}: {len(df):,} bars")
    df = pd.concat(chunks)
    df = df[~df.index.duplicated()].sort_index()
    idx = df.index.tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(minutes=1)
    df.index = pd.DatetimeIndex(idx, name="datetime")
    return filter_rth(df[["open", "high", "low", "close", "volume"]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["IWM", "SPY"])
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in a.symbols:
        df = fetch(s, a.start, a.end)
        df.to_csv(out / f"{s}_1m.csv")
        print(f"{s}: {len(df):,} RTH bars, {df.index[0]} -> {df.index[-1]}")


if __name__ == "__main__":
    main()

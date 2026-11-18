import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imif import StrategyConfig, build_spread, run_backtest  # noqa: E402
from imif.data import align_legs, synthetic_legs, validate_bars  # noqa: E402
from imif.indicators import cci, donchian, wilder_rsi  # noqa: E402
from imif.spread import cumulative_log_spread  # noqa: E402
from imif.threshold import engle_granger, manual_critical_value  # noqa: E402


@pytest.fixture(scope="module")
def legs():
    return align_legs(*synthetic_legs(n_days=30, seed=3))


def test_synthetic_bars_are_valid(legs):
    im, if_ = legs
    assert validate_bars(im, "IM") == [] and validate_bars(if_, "IF") == []
    assert len(im) == 30 * 240


def test_cumulative_log_spread_equals_log_ratio(legs):
    im, if_ = legs
    s = cumulative_log_spread(im["close"], if_["close"])
    direct = (np.log(im["close"] / im["close"].iloc[0]) - np.log(if_["close"] / if_["close"].iloc[0])) * 1e4
    assert np.allclose(s.values, direct.values)


def test_spread_ohlc_consistent(legs):
    sp = build_spread(*legs)
    assert (sp["high"] >= sp[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (sp["low"] <= sp[["open", "close"]].min(axis=1) + 1e-9).all()


def test_rsi_bounds_and_trend():
    up = pd.Series(np.arange(100, dtype=float))
    rsi = wilder_rsi(up, 14).dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all() and rsi.iloc[-1] == 100.0


def test_donchian_excludes_current_bar():
    h = pd.Series([1, 2, 3, 10], dtype=float)
    up, _ = donchian(h, h, 3)
    assert up.iloc[3] == 3.0  # the 10 on the current bar is a breakout, not part of the band


def test_cci_zero_on_flat_series():
    s = pd.Series(np.full(50, 5.0))
    assert cci(s, s, s, 20).dropna().empty  # zero mean deviation -> undefined, not inf


def test_backtest_legs_synchronised(legs):
    res = run_backtest(*legs, StrategyConfig())
    st = res.stats
    assert st["leg_trades"] == 2 * st["trades"]
    if st["trades"]:
        tr = res.trades
        assert (tr["exit_time"] > tr["entry_time"]).all()
        # never held across sessions
        assert (tr["entry_time"].dt.date == tr["exit_time"].dt.date).all()
        assert np.isclose(tr["net"].sum(), st["net_profit"], atol=1e-6)


def test_entry_cutoffs_respected(legs):
    res = run_backtest(*legs, StrategyConfig())
    tr = res.trades
    if len(tr):
        # signals before cutoff fill one bar later at most
        short_t = tr.loc[tr["side"] == "short", "entry_time"].dt.time
        assert all(t <= pd.Timestamp("14:01").time() for t in short_t)


def test_threshold_helpers(legs):
    im, if_ = legs
    s = build_spread(im, if_)["close"]
    days = s.index.normalize().unique()
    cv = manual_critical_value(s, [(str(days[0].date()), str(days[2].date())), (str(days[5].date()), str(days[7].date()))])
    assert cv > 0
    eg = engle_granger(np.log(im["close"]), np.log(if_["close"]))
    assert 0.5 < eg["beta"] < 2.0


def test_us_preset_sizes_both_legs_and_flattens_before_close():
    im, if_ = align_legs(*synthetic_legs(n_days=20, seed=5, session="us", im0=2400, if0=6000, tick=0.25))
    cfg = StrategyConfig.us_rty_es()
    res = run_backtest(im, if_, cfg)
    if len(res.trades):
        t = res.trades
        assert (t["n_im"] >= 4).all() and (t["n_if"] >= 1).all()      # ~$60k per leg on M2K / MES
        assert (t["exit_time"].dt.time <= cfg.flat_time).all()
        assert (t["commission"] > 0).all()


def test_filter_rth_drops_overnight_bars():
    from imif.data import filter_rth
    idx = pd.date_range("2025-03-03 08:00", "2025-03-03 17:00", freq="1min")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}, index=idx)
    out = filter_rth(df)
    assert out.index[0].strftime("%H:%M") == "09:31" and out.index[-1].strftime("%H:%M") == "16:00"

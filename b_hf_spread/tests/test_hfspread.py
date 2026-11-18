import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfspread import leadlag, pairs, quotes  # noqa: E402
from hfspread.synthetic import synthetic_pair  # noqa: E402


def test_leadlag_detects_a_leading_b():
    q = synthetic_pair(n_days=1, lag=2)
    cc = leadlag.cross_corr(q["mid_a"], q["mid_b"], max_lag=5)
    assert int(cc.abs().idxmax()) == 2


def test_rolling_beta_has_no_lookahead():
    q = synthetic_pair(n_days=1)
    z_full = pairs.zscore(q, pairs.PairsConfig())["z"]
    z_cut = pairs.zscore(q.iloc[:15000], pairs.PairsConfig())["z"]
    pd.testing.assert_series_equal(z_full.iloc[:15000], z_cut)


def test_backtest_runs_and_flattens_each_day():
    q = synthetic_pair(n_days=2)
    t, st = pairs.backtest(q, pairs.PairsConfig(z_in=2.0))
    assert st["trades"] > 0
    assert (t["entry"].dt.normalize() == t["exit"].dt.normalize()).all()
    assert (t["cost"] > 0).all()


def test_to_mid_and_align():
    idx = pd.date_range("2025-03-03 09:30", periods=4, freq="500ms")
    a = pd.DataFrame({"bid_px_00": [1, 1, 2, 2.0], "ask_px_00": [1.25, 1.25, 2.25, 2.25]}, index=idx)
    out = quotes.align(quotes.to_mid(a), quotes.to_mid(a))
    assert list(out.columns) == ["mid_a", "spread_a", "mid_b", "spread_b"] and len(out) == 2

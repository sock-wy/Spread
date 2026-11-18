"""Parameter search that prefers plateaus over spikes.

The MultiCharts optimizer was used to render a 3-D surface over pairs of
parameters; a robust setting sits on a broad plateau, not an isolated peak.
Here every grid point gets a "plateau score" (the median Sharpe of its
neighbourhood), and selection uses that score instead of the raw Sharpe.
Trade-count stability across neighbours is reported as well, since a large
jump in trade count usually means the logic flipped, not that it improved.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import StrategyConfig


def grid_search(im, if_, grid: dict[str, list], base: StrategyConfig | None = None) -> pd.DataFrame:
    base = base or StrategyConfig()
    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = replace(base, **dict(zip(keys, combo)))
        st = run_backtest(im, if_, cfg).stats
        rows.append({**dict(zip(keys, combo)), "sharpe": st["sharpe"], "net": st["net_profit"],
                     "mdd": st["max_drawdown"], "trades": st["trades"]})
    return pd.DataFrame(rows)


def plateau_scores(res: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Median Sharpe and trade-count dispersion over each point's grid neighbours."""
    levels = {k: sorted(res[k].unique()) for k in keys}
    pos = {k: {v: i for i, v in enumerate(levels[k])} for k in keys}
    coords = res[keys].apply(lambda r: tuple(pos[k][r[k]] for k in keys), axis=1)
    lookup = dict(zip(coords, res.index))
    scores, stab = [], []
    for c in coords:
        nb = []
        for d in itertools.product(*([-1, 0, 1] for _ in keys)):
            j = lookup.get(tuple(a + b for a, b in zip(c, d)))
            if j is not None:
                nb.append(j)
        sub = res.loc[nb]
        scores.append(float(np.nanmedian(sub["sharpe"])))
        t = sub["trades"].replace(0, np.nan)
        stab.append(float(t.std() / t.mean()) if t.mean() == t.mean() and t.mean() > 0 else float("nan"))
    out = res.copy()
    out["plateau_sharpe"] = scores
    out["trade_count_cv"] = stab
    return out.sort_values("plateau_sharpe", ascending=False)


def walk_forward_split(im, if_, train_frac: float = 0.7):
    """Chronological in-sample / out-of-sample split, by trading day."""
    days = im.index.normalize().unique()
    cut = days[int(len(days) * train_frac)]
    ins = im.index < cut
    return (im[ins], if_[ins]), (im[~ins], if_[~ins])

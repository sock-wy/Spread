"""Charts: spread with entries/exits, and the equity curve."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_result(res, path: str | Path, title: str = "IM/IF spread backtest") -> None:
    df, tr = res.frame, res.trades
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=False, gridspec_kw={"height_ratios": [2, 1]})
    x = range(len(df))
    pos = {t: i for i, t in enumerate(df.index)}
    ax1.plot(x, df["close"], lw=0.6, color="#3b5b7a", label="spread (bp)")
    ax1.plot(x, df["ema"], lw=0.6, color="#c07a2c", label="EMA")
    ax1.plot(x, df["center"], lw=0.6, ls="--", color="#777777", label="centre")
    if len(tr):
        for side, m, col in (("long", "^", "#2a7a3b"), ("short", "v", "#b03a2e")):
            sub = tr[tr["side"] == side]
            ax1.scatter([pos[t] for t in sub["entry_time"]], sub["spread_in"], marker=m, s=18, color=col, label=f"{side} entry", zorder=3)
        ax1.scatter([pos[t] for t in tr["exit_time"]], tr["spread_out"], marker="x", s=12, color="#222222", label="exit", zorder=3)
    ax1.set_ylabel("cumulative log-return spread (bp)")
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize=8, ncol=3)
    ax2.plot(range(len(res.equity)), res.equity.values, lw=0.8, color="#3b5b7a")
    ax2.set_ylabel("equity (CNY)")
    ax2.set_xlabel("bar (1-minute, trading time only)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

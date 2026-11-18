"""Charts + markdown report for tune_us.py."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"           # categorical slots 1-3
DIVERGE = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", "#2a78d6"])
DOLLAR = lambda x, _=None: f"${x / 1000:,.0f}k" if abs(x) >= 1000 else f"${x:,.0f}"  # noqa: E731

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "lines.linewidth": 2, "legend.frameon": False,
})


def _shade_holdout(ax, dev_end, idx):
    ax.axvspan(dev_end, idx[-1], color="#f0efec", zorder=0)
    ax.text(dev_end, 1.0, "  holdout (never used to choose)", transform=ax.get_xaxis_transform(),
            va="top", ha="left", color=INK2, fontsize=8)


def _heat(ax, df, x, y, val, title, fmt="{:.2f}", mark=None):
    piv = df.pivot_table(index=y, columns=x, values=val, aggfunc="median", dropna=False)
    piv = piv.reindex(index=sorted(df[y].unique()), columns=sorted(df[x].unique()))
    v = piv.to_numpy(float)
    lim = np.nanmax(np.abs(v)) if np.isfinite(v).any() else 1
    im = ax.imshow(v, cmap=DIVERGE, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto", origin="lower")
    ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{c:g}" for c in piv.index])
    ax.set_xlabel(x), ax.set_ylabel(y), ax.set_title(title), ax.grid(False)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isfinite(v[i, j]):
                ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", fontsize=7, color=INK)
    if mark is not None:
        j, i = list(piv.columns).index(mark[0]), list(piv.index).index(mark[1])
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=INK, lw=2))
    return im


def make_report(out: Path, cfg, base, pnl_f, pnl_b, pnl_wf, tr, dev_end, cap, stage_runs, g1, g2) -> None:
    out = Path(out)

    # 1 equity curves
    fig, ax = plt.subplots(figsize=(10, 4.2))
    _shade_holdout(ax, dev_end, pnl_f.index)
    ax.plot(pnl_f.index, pnl_f.cumsum(), color=S1, label="Tuned params")
    ax.plot(pnl_b.index, pnl_b.cumsum(), color=S2, label="Original IM/IF params", lw=1.5)
    if len(pnl_wf):
        ax.plot(pnl_wf.index, pnl_wf.cumsum(), color=S3, label="Walk-forward (re-picked quarterly, OOS only)", lw=1.5)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.yaxis.set_major_formatter(DOLLAR)
    ax.set_title(f"Cumulative net P&L, IWM vs SPY, ${cap / 1000:.0f}k capital")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93))
    fig.tight_layout(), fig.savefig(out / "equity_curve.png", dpi=150), plt.close(fig)

    # 2 drawdown
    eq = cap + pnl_f.cumsum()
    dd = eq / eq.cummax() - 1
    fig, ax = plt.subplots(figsize=(10, 2.6))
    _shade_holdout(ax, dev_end, pnl_f.index)
    ax.fill_between(dd.index, dd * 100, 0, color=S1, alpha=0.35, lw=0)
    ax.plot(dd.index, dd * 100, color=S1, lw=1)
    ax.set_ylabel("%"), ax.set_title("Drawdown, tuned params")
    fig.tight_layout(), fig.savefig(out / "drawdown.png", dpi=150), plt.close(fig)

    # 3 heatmaps (dev-period Sharpe)
    s1 = pd.read_csv(out / "stage1.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _heat(axes[0], s1[s1.len_don == cfg.len_don], "len_cci", "len_ema", "sharpe",
          f"Stage 1 dev Sharpe, len_don={cfg.len_don}", mark=(cfg.len_cci, cfg.len_ema))
    _heat(axes[1], s1[s1.len_ema == cfg.len_ema], "len_cci", "len_don", "sharpe",
          f"Stage 1 dev Sharpe, len_ema={cfg.len_ema}", mark=(cfg.len_cci, cfg.len_don))
    fig.tight_layout(), fig.savefig(out / "heatmap_stage1.png", dpi=150), plt.close(fig)

    s2 = pd.read_csv(out / "stage2.csv")
    sub = s2[(s2.rsi_long_min == cfg.rsi_long_min) & (s2.rsi_short_min == cfg.rsi_short_min)]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _heat(ax, sub, "critical_value_bp", "cci_low", "sharpe",
          f"Stage 2 dev Sharpe, RSI gates {cfg.rsi_short_min:g}/{cfg.rsi_long_min:g}",
          mark=(cfg.critical_value_bp, cfg.cci_low))
    fig.tight_layout(), fig.savefig(out / "heatmap_stage2.png", dpi=150), plt.close(fig)

    # 4 monthly returns (% of capital)
    m = (pnl_f.groupby([pnl_f.index.year, pnl_f.index.month]).sum() / cap * 100).unstack()
    m = m.reindex(columns=range(1, 13))
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(m) + 1.2))
    lim = np.nanmax(np.abs(m.to_numpy()))
    ax.imshow(m.to_numpy(), cmap=DIVERGE, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_xticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(m)), m.index), ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(12):
            v = m.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7.5)
    ax.set_title("Monthly net return, % of capital (tuned params)")
    fig.tight_layout(), fig.savefig(out / "monthly_returns.png", dpi=150), plt.close(fig)

    # 5 breakdowns
    tr = tr.copy()
    tr["period"] = np.where(tr["exit_time"] < dev_end, "dev", "holdout")
    tr["hour"] = tr["entry_time"].dt.hour
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, col, title in ((axes[0], "reason", "Net P&L by exit reason"),
                           (axes[1], "hour", "Net P&L by entry hour (ET)"),
                           (axes[2], "side", "Net P&L by side")):
        g = tr.groupby([col, "period"])["net"].sum().unstack().fillna(0)
        g = g.reindex(columns=["dev", "holdout"]).fillna(0)
        x = np.arange(len(g))
        ax.bar(x - 0.2, g["dev"], 0.38, color=S1, label="dev")
        ax.bar(x + 0.2, g["holdout"], 0.38, color=S2, label="holdout")
        ax.set_xticks(x, [str(i) for i in g.index], rotation=30 if col == "reason" else 0, ha="right" if col == "reason" else "center")
        ax.axhline(0, color=INK2, lw=0.8), ax.yaxis.set_major_formatter(DOLLAR), ax.set_title(title)
    axes[0].legend()
    fig.tight_layout(), fig.savefig(out / "breakdown.png", dpi=150), plt.close(fig)

    # 6 markdown
    js = json.loads((out / "summary.json").read_text())
    S = js["summary"]
    cost = pd.read_csv(out / "cost_sensitivity.csv")

    def row(name, d):
        return (f"| {name} | {d.get('ann_return', float('nan')):.1%} | {d.get('sharpe', float('nan')):.2f} | "
                f"{d.get('max_dd', float('nan')):.1%} | {DOLLAR(d.get('net', 0))} | {d.get('trades', '–')} | "
                f"{d.get('win_rate', float('nan')):.0%} | {d.get('avg_minutes', float('nan')):.0f} |")

    params = "\n".join(f"| `{k}` | {getattr(base, k, 1 if k == 'bar_minutes' else '-')} | **{v}** |" for k, v in S["final_params"].items())
    stages = "\n".join(f"| {k} | {', '.join(f'{a}={b}' for a, b in v['chosen'].items())} | "
                       f"{v['dev_sharpe']:.2f} | {v['dev_plateau_sharpe']:.2f} | {v['dev_trades']} |"
                       for k, v in js["stages"].items())
    s0 = "\n".join(f"| {r['tf']} | {r['best_plateau_sharpe']:.2f} | {r['median_sharpe']:.2f} | {r['trades_at_best']} |" for r in js["stage0"])
    costs = "\n".join(f"| {r.slippage_ticks:g} | {r.dev_sharpe:.2f} | {r.dev_ann_return:.1%} | "
                      f"{r.hold_sharpe:.2f} | {r.hold_ann_return:.1%} |" for r in cost.itertuples())
    md = f"""# Project A tuning report: IWM vs SPY (ETF proxy for RTY vs ES)

Dev period (parameters chosen here): start -> {js['dev_end']}. Holdout: {js['dev_end']} -> end, never used to choose.
Costs: 1 tick ($0.01) slippage per leg per fill, no commission, ~$50k per leg, $100k capital.
Bar size chosen in stage 0: **{js['bar_minutes']} min** (original IM/IF used 1 min).

Avg net per trade: dev ${S['dev'].get('avg_net', float('nan')):.2f}, holdout ${S['holdout'].get('avg_net', float('nan')):.2f}.
See "Costs": the zero-slippage row shows whether there is any edge before trading costs.

## Results

| | Ann. return | Sharpe | Max DD | Net | Trades | Win rate | Avg min |
|---|---|---|---|---|---|---|---|
{row("Tuned, dev", S["dev"])}
{row("Tuned, holdout", S["holdout"])}
{row("Original params, dev", S["baseline_cn_params_dev"])}
{row("Original params, holdout", S["baseline_cn_params_holdout"])}
{row("Walk-forward OOS", S["walk_forward_oos"])}

![equity](equity_curve.png)
![drawdown](drawdown.png)

## Parameters

| Parameter | Original (IM/IF) | Tuned |
|---|---|---|
{params}

Spread scale on dev (bp): {', '.join(f'{k}={v:.2f}' for k, v in js['scale_bp'].items())}

Stage 0, bar size (stage-1 grid at original thresholds):

| Bar (min) | Best plateau Sharpe | Grid median Sharpe | Trades at best |
|---|---|---|---|
{s0}

| Stage | Pick | Dev Sharpe | Plateau Sharpe | Dev trades |
|---|---|---|---|---|
{stages}

![stage1](heatmap_stage1.png)
![stage2](heatmap_stage2.png)

## Costs

| Slippage (ticks/leg/fill) | Dev Sharpe | Dev ann. | Holdout Sharpe | Holdout ann. |
|---|---|---|---|---|
{costs}

## Breakdown

![monthly](monthly_returns.png)
![breakdown](breakdown.png)
"""
    (out / "report.md").write_text(md)

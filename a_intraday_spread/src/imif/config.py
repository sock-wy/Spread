"""Strategy and cost parameters.

Defaults follow the design described in docs/strategy_summary.md. Exact
production values were not preserved, so thresholds marked "reconstructed"
are reasonable starting points to re-optimize, not the original settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import time


@dataclass
class CostConfig:
    """Futures contract specs and trading frictions (CFFEX index futures)."""

    capital: float = 1_000_000.0
    mult_im: float = 200.0          # currency per index point, leg A ("IM" = small-cap leg)
    mult_if: float = 300.0          # currency per index point, leg B ("IF" = large-cap leg)
    tick: float = 0.2               # minimum price move, leg A
    tick_if: float | None = None    # minimum price move, leg B (None = same as tick)
    slippage_ticks: float = 1.0     # per leg, per fill
    fee_per_contract: float = 0.0   # flat fee per contract per side (US futures style)
    # Sizing. None = original rule (1 lot of leg A, leg B matched by notional).
    # A number = size BOTH legs to roughly this notional each (needed when leg A
    # is the smaller contract, e.g. M2K vs MES).
    target_notional: float | None = None
    # Commission as a fraction of notional. The original backtest used the
    # close-yesterday rate on both sides; set close_rate to 3.45e-4 to model
    # the much higher close-today fee instead.
    open_rate: float = 0.23e-4
    close_rate: float = 0.23e-4


@dataclass
class StrategyConfig:
    # Indicator lengths (bars)
    len_rsi: int = 14
    len_cci: int = 20
    len_don: int = 20
    len_ema: int = 30

    # RSI / CCI thresholds (reconstructed)
    rsi_short_min: float = 20.0     # short entry gate: RSI > 20
    rsi_long_min: float = 80.0      # long entry gate: RSI > 80
    rsi_overbought: float = 80.0    # long take-profit
    rsi_oversold: float = 20.0      # short take-profit (mirror)
    cci_low: float = -100.0         # "prior bar breaks the CCI low"
    cci_exit_low: float = -150.0    # ranging-market exit threshold from the summary
    cci_exit_floor: float = -400.0  # below this the CCI exit is suppressed (trend regime)
    cci_high: float = 100.0
    cci_ceiling: float = 200.0      # "CCI hitting its ceiling" stop for short spread

    # Spread-level controls, in basis points of cumulative log return
    hard_stop_bp: float = 40.0
    max_close_to_ema_bp: float = 15.0
    use_critical_value: bool = True
    critical_value_bp: float = 20.0
    center_window: int = 240        # bars used for the equilibrium centre (~1 trading day)

    # Intraday timing
    short_cutoff: time = time(14, 0)
    long_cutoff: time = time(14, 15)
    flat_time: time = time(14, 55)

    enable_long: bool = True
    enable_short: bool = True

    costs: CostConfig = field(default_factory=CostConfig)

    @classmethod
    def us_rty_es(cls, micro: bool = True, **overrides) -> "StrategyConfig":
        """Preset for the US version: Russell 2000 (leg A) vs S&P 500 (leg B), CME, RTH only.

        micro=True  -> M2K / MES ($5 per point each), small enough for a test account.
        micro=False -> RTY / ES  ($50 per point each).
        Ticks: RTY 0.10, ES 0.25. Fee is an all-in estimate per contract per side;
        check your broker. Indicator thresholds are the CN defaults and MUST be
        re-optimised on US data (scripts/optimize.py).
        """
        mult = 5.0 if micro else 50.0
        costs = CostConfig(
            capital=100_000.0 if micro else 1_000_000.0,
            mult_im=mult, mult_if=mult, tick=0.10, tick_if=0.25,
            slippage_ticks=1.0, fee_per_contract=0.85 if micro else 2.25,
            open_rate=0.0, close_rate=0.0,
            target_notional=60_000.0 if micro else 600_000.0,
        )
        base = cls(
            short_cutoff=time(15, 0), long_cutoff=time(15, 15), flat_time=time(15, 55),
            center_window=390, costs=costs,     # 390 one-minute bars = one RTH session
        )
        for k, v in overrides.items():
            setattr(base, k, v)
        return base

    @classmethod
    def us_iwm_spy(cls, **overrides) -> "StrategyConfig":
        """ETF proxy of the RTY/ES pair: IWM (leg A) vs SPY (leg B), 1-minute SIP bars.

        Used for research because ~8 years of minute data are free from Alpaca and
        the same pair can be paper-traded there. Shares: multiplier 1, tick $0.01,
        1 tick slippage per leg per fill, no commission (Alpaca); ~$50k per leg.
        """
        base = cls.us_rty_es()
        base.costs = CostConfig(capital=100_000.0, mult_im=1.0, mult_if=1.0, tick=0.01, tick_if=0.01,
                                slippage_ticks=1.0, fee_per_contract=0.0, open_rate=0.0, close_rate=0.0,
                                target_notional=50_000.0)
        for k, v in overrides.items():
            setattr(base, k, v)
        return base

    def to_dict(self) -> dict:
        d = asdict(self)  # noqa
        for k in ("short_cutoff", "long_cutoff", "flat_time"):
            d[k] = getattr(self, k).strftime("%H:%M")
        return d

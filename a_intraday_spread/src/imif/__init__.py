"""IM/IF stock-index futures spread statistical arbitrage.

Reconstructed research codebase for a CSI 1000 (IM) vs CSI 300 (IF) spread
strategy built during a quantitative-trading internship. See README.md.
"""

from .config import StrategyConfig, CostConfig
from .spread import build_spread
from .indicators import wilder_rsi, cci, donchian, ema
from .backtest import run_backtest, BacktestResult

__all__ = [
    "StrategyConfig",
    "CostConfig",
    "build_spread",
    "wilder_rsi",
    "cci",
    "donchian",
    "ema",
    "run_backtest",
    "BacktestResult",
]

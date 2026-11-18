"""Pre-trade checks. Every order passes through here before it is sent."""
from dataclasses import dataclass


@dataclass
class Order:
    symbol: str
    qty: int          # positive = buy, negative = sell
    price: float      # reference price (last close)

    @property
    def notional(self) -> float:
        return abs(self.qty) * self.price


def check(orders: list[Order], risk_cfg: dict) -> tuple[list[Order], list[str]]:
    """Return (approved orders, rejection messages)."""
    approved, rejected = [], []
    for o in orders:
        if o.qty == 0:
            continue
        if o.notional < risk_cfg["min_trade_notional"]:
            rejected.append(f"{o.symbol}: skipped, notional {o.notional:.0f} below minimum")
        elif o.notional > risk_cfg["max_order_notional"]:
            rejected.append(f"{o.symbol}: REJECTED, notional {o.notional:.0f} above max_order_notional")
        else:
            approved.append(o)
    if len(approved) > risk_cfg["max_orders_per_run"]:
        rejected.append(f"REJECTED all: {len(approved)} orders exceeds max_orders_per_run")
        approved = []
    return approved, rejected

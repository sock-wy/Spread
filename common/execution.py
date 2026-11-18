"""Compare target vs current positions and send orders to Alpaca."""
from .risk import Order


def current_quantities(trading, symbols: list[str]) -> dict[str, int]:
    held = {p.symbol: int(float(p.qty)) for p in trading.get_all_positions()}
    return {s: held.get(s, 0) for s in symbols}


def diff_orders(target: dict[str, int], current: dict[str, int], prices: dict[str, float]) -> list[Order]:
    """Sells first so their cash is available for buys."""
    orders = [Order(s, target[s] - current.get(s, 0), prices[s]) for s in target]
    return sorted([o for o in orders if o.qty != 0], key=lambda o: o.qty)


def submit(trading, orders: list[Order]) -> list[dict]:
    """Market DAY orders. Sent after the close, they queue for the next open."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    results = []
    for o in orders:
        req = MarketOrderRequest(
            symbol=o.symbol,
            qty=abs(o.qty),
            side=OrderSide.BUY if o.qty > 0 else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        resp = trading.submit_order(req)
        results.append({"symbol": o.symbol, "qty": o.qty, "order_id": str(resp.id), "status": str(resp.status)})
    return results

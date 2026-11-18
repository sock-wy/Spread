"""Price-level (L2 / MBP) order book.

Feed it incremental updates `(side, price, size)` where size is the NEW total
resting size at that price (0 = level removed). This is how CME MDP3 market-by-
price and Databento `mbp-*` updates work. If your data is already snapshots
(10 levels per row), skip this class and go straight to `features.py`.

Prices are stored as integer ticks to avoid float-key bugs.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class L2Book:
    tick: float
    bids: dict[int, float] = field(default_factory=dict)   # price_ticks -> size
    asks: dict[int, float] = field(default_factory=dict)

    def _key(self, price: float) -> int:
        return int(round(price / self.tick))

    def update(self, side: str, price: float, size: float) -> None:
        book = self.bids if side in ("B", "bid", "buy") else self.asks
        k = self._key(price)
        if size <= 0:
            book.pop(k, None)
        else:
            book[k] = size

    def clear(self) -> None:
        self.bids.clear()
        self.asks.clear()

    def best_bid(self) -> tuple[float, float] | None:
        if not self.bids:
            return None
        k = max(self.bids)
        return k * self.tick, self.bids[k]

    def best_ask(self) -> tuple[float, float] | None:
        if not self.asks:
            return None
        k = min(self.asks)
        return k * self.tick, self.asks[k]

    def depth(self, n: int = 5) -> dict[str, list[tuple[float, float]]]:
        b = sorted(self.bids.items(), reverse=True)[:n]
        a = sorted(self.asks.items())[:n]
        return {"bids": [(k * self.tick, v) for k, v in b], "asks": [(k * self.tick, v) for k, v in a]}

    def is_crossed(self) -> bool:
        bb, ba = self.best_bid(), self.best_ask()
        return bb is not None and ba is not None and bb[0] >= ba[0]

    def snapshot(self, n: int = 5) -> dict[str, float]:
        """Flat row in the same column layout features.py expects: bid_px_00, bid_sz_00, ..."""
        d = self.depth(n)
        row: dict[str, float] = {}
        for side, lv in (("bid", d["bids"]), ("ask", d["asks"])):
            for i in range(n):
                px, sz = lv[i] if i < len(lv) else (float("nan"), 0.0)
                row[f"{side}_px_{i:02d}"], row[f"{side}_sz_{i:02d}"] = px, sz
        return row

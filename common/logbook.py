"""Append-only CSV logs: one row per symbol per run, plus daily equity."""
import csv
from datetime import datetime

from .config import LOG_DIR


def append(filename: str, rows: list[dict]) -> None:
    if not rows:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / filename
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")

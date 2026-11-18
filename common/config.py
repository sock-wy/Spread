"""Shared paths and Alpaca credentials (paper account only)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
LOG_DIR = ROOT / "logs"


def credentials() -> tuple[str, str]:
    """Read keys from the environment. Never hard-code them."""
    try:
        return os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"]
    except KeyError as e:
        raise RuntimeError(f"Missing environment variable {e}. See README.") from None

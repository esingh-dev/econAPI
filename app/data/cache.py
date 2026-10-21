"""Thread-safe high-speed in-memory TTL caching for market and macro data."""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional
from cachetools import TTLCache


class MarketDataCache:
    """In-memory thread-safe cache with configurable Time-To-Live (TTL)."""

    def __init__(self, maxsize: int = 2048, default_ttl_seconds: int = 300) -> None:
        self._cache = TTLCache(maxsize=maxsize, ttl=default_ttl_seconds)
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def get_or_set(self, key: str, supplier: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        # Compute outside main lock to minimize lock contention
        val = supplier()
        with self._lock:
            self._cache[key] = val
            return val


# Global shared cache instances
market_cache = MarketDataCache(maxsize=1024, default_ttl_seconds=300)      # 5 min for equities/options
macro_cache = MarketDataCache(maxsize=256, default_ttl_seconds=1800)       # 30 min for Treasury/FRED rates

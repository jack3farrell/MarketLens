import time
from typing import Any, Dict, Tuple, Optional

class TTLCache:
    def __init__(self, ttl_seconds: int):
        self.ttl_seconds = ttl_seconds
        self._store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if not item:
            return None
        ts, val = item
        if time.time() - ts > self.ttl_seconds:
            self._store.pop(key, None)
            return None
        return val

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time(), value)
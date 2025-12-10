import time
from typing import Dict, Any, Optional
import datetime

class TtlCache:
    """
    A simple in-memory Time-to-Live (TTL) cache with stats tracking.
    """
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        # --- NEW: Stats Tracking ---
        self.hits = 0
        self.misses = 0
        self.last_cleanup = None

    def set(self, key: str, value: Any):
        """Sets a value in the cache with an expiration timestamp."""
        self.cache[key] = {
            "value": value,
            "expires_at": time.time() + self.ttl
        }

    def get(self, key: str) -> Optional[Any]:
        """Gets a value from the cache if it exists and has not expired."""
        if key not in self.cache:
            self.misses += 1 # Track miss
            return None
        
        item = self.cache[key]
        if time.time() > item["expires_at"]:
            # Item expired, delete it
            del self.cache[key]
            self.misses += 1 # Track miss (as it was expired)
            return None
            
        self.hits += 1 # Track hit
        return item["value"]

    def get_stats(self) -> Dict[str, Any]:
        """
        Cleans expired items and returns a dictionary of cache statistics.
        """
        expired_count = self.clear_expired()
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) * 100 if total_requests > 0 else 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "current_items": len(self.cache),
            "expired_items_cleared": expired_count,
            "last_cleanup": self.last_cleanup
        }

    def clear_expired(self) -> int:
        """
        Manually clears all expired items from the cache.
        Returns the number of items cleared.
        """
        now = time.time()
        expired_keys = [key for key, item in self.cache.items() if now > item["expires_at"]]
        count = 0
        for key in expired_keys:
            try:
                del self.cache[key]
                count += 1
            except KeyError:
                pass
        
        if count > 0:
            self.last_cleanup = datetime.datetime.now().isoformat()
            
        return count
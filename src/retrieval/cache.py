"""
Retrieval result cache.

Prevents re-embedding and re-querying the vector database for identical queries.
Uses Redis if available (shared cache across workers), otherwise falls back
to an in-memory LRU cache.
"""
import os
import json
import hashlib
from collections import OrderedDict
from typing import Tuple, List, Optional

from src.utils.logger import logger

_REDIS_AVAILABLE = False
try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    pass


class RetrievalCache:
    """Cache for retrieval results (context string + sources list)."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        self._redis: Optional[object] = None
        redis_url = os.getenv("REDIS_URL")
        if redis_url and _REDIS_AVAILABLE:
            try:
                self._redis = _redis_lib.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("RetrievalCache: using Redis backend")
            except Exception as e:
                logger.warning(f"RetrievalCache: Redis unavailable ({e}), falling back to in-memory")
                self._redis = None
        
        if self._redis is None:
            # In-memory LRU cache: query_hash -> (context, sources)
            self._cache: OrderedDict = OrderedDict()
            logger.info("RetrievalCache: using in-memory LRU backend")

    def _hash_query(self, query: str) -> str:
        """Create a deterministic hash for the query."""
        # Lowercase and strip to normalize slightly (e.g. "Hello " == "hello")
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def get(self, query: str) -> Optional[Tuple[str, List[str]]]:
        """Get cached retrieval results if available."""
        q_hash = self._hash_query(query)
        
        if self._redis:
            try:
                key = f"smartroute:retrieval:{q_hash}"
                raw = self._redis.get(key)
                if raw:
                    data = json.loads(raw)
                    return data["context"], data["sources"]
            except Exception as e:
                logger.warning(f"RetrievalCache Redis get failed: {e}")
            return None
        
        # In-memory LRU
        if q_hash in self._cache:
            # Move to end to mark as recently used
            self._cache.move_to_end(q_hash)
            return self._cache[q_hash]
            
        return None

    def set(self, query: str, context: str, sources: List[str]) -> None:
        """Cache retrieval results."""
        q_hash = self._hash_query(query)
        
        if self._redis:
            try:
                key = f"smartroute:retrieval:{q_hash}"
                payload = json.dumps({"context": context, "sources": sources})
                self._redis.setex(key, self.ttl_seconds, payload)
            except Exception as e:
                logger.warning(f"RetrievalCache Redis set failed: {e}")
        else:
            # In-memory LRU
            self._cache[q_hash] = (context, sources)
            self._cache.move_to_end(q_hash)
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)  # pop least recently used (first item)

    def clear(self) -> None:
        """Clear the cache (useful after adding new documents)."""
        if self._redis:
            try:
                # Use SCAN to find all retrieval keys and delete them
                cursor = '0'
                while cursor != 0:
                    cursor, keys = self._redis.scan(cursor=cursor, match="smartroute:retrieval:*", count=100)
                    if keys:
                        self._redis.delete(*keys)
            except Exception as e:
                logger.warning(f"RetrievalCache Redis clear failed: {e}")
        else:
            self._cache.clear()

"""
Retrieval result cache — Upstash Redis only.

Prevents re-embedding and re-querying the vector database for identical queries.
REDIS_URL is REQUIRED. The app will refuse to start without it.

Get your free Upstash Redis URL at: https://upstash.com
"""

import hashlib
import json
import os
from typing import List, Optional, Tuple

import redis as sync_redis

from src.utils.logger import logger


class RetrievalCache:
    """Cache for retrieval results (context string + sources list) using Upstash Redis.

    REDIS_URL must be set. Raises RuntimeError on startup if missing.
    Eviction (LRU) is handled by the Redis server config (allkeys-lru).
    """

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds

        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            raise RuntimeError(
                "REDIS_URL is not set.\n"
                "This app requires Upstash Redis for retrieval caching.\n"
                "1. Create a free database at https://upstash.com\n"
                "2. Copy the Redis URL (starts with rediss://).\n"
                "3. Set REDIS_URL=rediss://... in your .env or Render env vars."
            )

        self._redis = sync_redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            ssl_cert_reqs=None,
        )
        self._redis.ping()
        logger.info("RetrievalCache → Upstash Redis connected")

    def _hash_query(self, query: str) -> str:
        """Create a deterministic hash for the query."""
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, query: str) -> Optional[Tuple[str, List[str]]]:
        """Get cached retrieval results if available."""
        q_hash = self._hash_query(query)
        try:
            key = f"smartroute:retrieval:{q_hash}"
            raw = self._redis.get(key)
            if raw:
                data = json.loads(raw)
                return data["context"], data["sources"]
        except Exception as e:
            logger.warning(f"RetrievalCache get failed: {e}")
        return None

    def set(self, query: str, context: str, sources: List[str]) -> None:
        """Cache retrieval results."""
        q_hash = self._hash_query(query)
        try:
            key = f"smartroute:retrieval:{q_hash}"
            payload = json.dumps({"context": context, "sources": sources})
            self._redis.setex(key, self.ttl_seconds, payload)
        except Exception as e:
            logger.warning(f"RetrievalCache set failed: {e}")

    def clear(self) -> None:
        """Clear the cache (useful after adding new documents)."""
        try:
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(
                    cursor=cursor, match="smartroute:retrieval:*", count=100
                )
                if keys:
                    self._redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning(f"RetrievalCache clear failed: {e}")

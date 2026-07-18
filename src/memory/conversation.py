"""
ConversationMemory — Upstash Redis only.

REDIS_URL is REQUIRED. The app will refuse to start without it.
No in-memory fallback — this system runs in the cloud with multiple workers.

Get your free Upstash Redis URL at: https://upstash.com
"""

import json
import os
from typing import List, Dict

import redis as sync_redis

from src.utils.logger import logger

MAX_TURNS = 10
SESSION_TTL_SECONDS = 30 * 60  # 30 minutes


class ConversationMemory:
    """Per-session conversation history backed by Upstash Redis.

    REDIS_URL must be set. Raises RuntimeError on startup if missing.
    Thread-safe and multi-worker safe — Redis is the single source of truth.
    """

    def __init__(
        self,
        max_turns: int = MAX_TURNS,
        session_ttl: int = SESSION_TTL_SECONDS,
    ):
        self.max_turns = max_turns
        self.session_ttl = session_ttl

        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            raise RuntimeError(
                "REDIS_URL is not set.\n"
                "This app requires Upstash Redis for conversation memory.\n"
                "1. Create a free database at https://upstash.com\n"
                "2. Copy the Redis URL (starts with rediss://).\n"
                "3. Set REDIS_URL=rediss://... in your .env or Render env vars."
            )

        self._redis = sync_redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            ssl_cert_reqs=None,  # required for Upstash TLS
        )
        self._redis.ping()  # fail fast if Redis is unreachable
        logger.info("ConversationMemory → Upstash Redis connected")

    def _key(self, session_id: str) -> str:
        return f"smartroute:memory:{session_id}"

    def get_history(self, session_id: str) -> List[Dict]:
        try:
            raw = self._redis.get(self._key(session_id))
            return json.loads(raw) if raw else []
        except Exception as e:
            logger.warning(f"ConversationMemory get failed: {e}")
            return []

    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        try:
            key = self._key(session_id)
            raw = self._redis.get(key)
            history: List[Dict] = json.loads(raw) if raw else []

            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": assistant_msg})

            # FIFO eviction — keep newest MAX_TURNS * 2 messages
            max_messages = self.max_turns * 2
            if len(history) > max_messages:
                history = history[-max_messages:]

            self._redis.setex(key, self.session_ttl, json.dumps(history))
        except Exception as e:
            logger.warning(f"ConversationMemory add failed: {e}")

    def clear(self, session_id: str) -> None:
        try:
            self._redis.delete(self._key(session_id))
        except Exception as e:
            logger.warning(f"ConversationMemory clear failed: {e}")

    def session_exists(self, session_id: str) -> bool:
        try:
            return bool(self._redis.exists(self._key(session_id)) > 0)
        except Exception:
            return False

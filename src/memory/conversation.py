"""
ConversationMemory — Upstash Redis only.

REDIS_URL is REQUIRED. The app will refuse to start without it.
No in-memory fallback — this system runs in the cloud with multiple workers.

Get your free Upstash Redis URL at: https://upstash.com
"""

import json
from typing import Dict, List

from src.core.dependencies import get_redis_client
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
        self._redis = get_redis_client()

    def _key(self, session_id: str) -> str:
        return f"smartroute:memory:{session_id}"

    async def get_history(self, session_id: str) -> List[Dict]:
        try:
            raw = await self._redis.get(self._key(session_id))
            return json.loads(raw) if raw else []
        except Exception as e:
            logger.warning(f"ConversationMemory get failed: {e}")
            return []

    async def add_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        try:
            key = self._key(session_id)
            raw = await self._redis.get(key)
            history: List[Dict] = json.loads(raw) if raw else []

            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": assistant_msg})

            # FIFO eviction — keep newest MAX_TURNS * 2 messages
            max_messages = self.max_turns * 2
            if len(history) > max_messages:
                history = history[-max_messages:]

            await self._redis.setex(key, self.session_ttl, json.dumps(history))
        except Exception as e:
            logger.warning(f"ConversationMemory add failed: {e}")

    async def clear(self, session_id: str) -> None:
        try:
            await self._redis.delete(self._key(session_id))
        except Exception as e:
            logger.warning(f"ConversationMemory clear failed: {e}")

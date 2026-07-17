"""
Conversation memory for multi-turn RAG sessions.

Enterprise upgrade:
- Uses `redis.asyncio` (non-blocking) when REDIS_URL is set (Upstash or self-hosted).
- Falls back to a thread-safe in-memory dict for local development.
- Upstash Redis is accessed via standard redis-py over TLS — no special SDK needed.

Upstash free tier: https://upstash.com  (10,000 commands/day, zero ops)
Set REDIS_URL=rediss://:password@hostname:6380 in your .env.
"""
import json
import os
import time
import threading
from typing import List, Dict, Optional

from src.utils.logger import logger

MAX_TURNS = 10
SESSION_TTL_SECONDS = 30 * 60  # 30 minutes

_ASYNC_REDIS_AVAILABLE = False
try:
    from redis import asyncio as aioredis
    _ASYNC_REDIS_AVAILABLE = True
except ImportError:
    pass


class ConversationMemory:
    """Stores and retrieves per-session conversation history.

    Public API is fully synchronous for backwards-compatibility with the
    existing pipeline. The underlying Redis calls are run via asyncio.run()
    when needed, or an event loop is reused when inside an async context.
    """

    def __init__(
        self,
        max_turns: int = MAX_TURNS,
        session_ttl: int = SESSION_TTL_SECONDS,
    ):
        self.max_turns = max_turns
        self.session_ttl = session_ttl
        self._redis = None

        redis_url = os.getenv("REDIS_URL")
        if redis_url and _ASYNC_REDIS_AVAILABLE:
            try:
                # Upstash uses rediss:// (TLS). ssl_cert_reqs=None is required
                # because Upstash uses self-signed certs on the free tier.
                self._redis = aioredis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=3,
                    ssl_cert_reqs=None,
                )
                logger.info("ConversationMemory: Upstash/Redis async backend connected")
            except Exception as e:
                logger.warning(f"ConversationMemory: Redis failed ({e}), using in-memory")
                self._redis = None

        if self._redis is None:
            self._store: Dict[str, dict] = {}
            self._lock = threading.Lock()
            logger.info("ConversationMemory: in-memory backend (single-node only)")

    def _key(self, session_id: str) -> str:
        return f"smartroute:memory:{session_id}"

    # ------------------------------------------------------------------
    # Async Redis helpers (called internally via asyncio)
    # ------------------------------------------------------------------

    async def _aget(self, session_id: str) -> List[Dict]:
        try:
            raw = await self._redis.get(self._key(session_id))
            return json.loads(raw) if raw else []
        except Exception as e:
            logger.warning(f"ConversationMemory Redis get failed: {e}")
            return []

    async def _aadd(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        try:
            key = self._key(session_id)
            raw = await self._redis.get(key)
            history: List[Dict] = json.loads(raw) if raw else []

            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": assistant_msg})

            max_messages = self.max_turns * 2
            if len(history) > max_messages:
                history = history[-max_messages:]

            await self._redis.setex(key, self.session_ttl, json.dumps(history))
        except Exception as e:
            logger.warning(f"ConversationMemory Redis add failed: {e}")

    async def _aclear(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))

    # ------------------------------------------------------------------
    # Sync public API (unchanged signature — no callers need to change)
    # ------------------------------------------------------------------

    def _run_async(self, coro):
        """Run a coroutine, reusing the running loop if we are inside one."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            return asyncio.run(coro)

    def get_history(self, session_id: str) -> List[Dict]:
        if self._redis:
            return self._run_async(self._aget(session_id))
        return self._memory_get(session_id)

    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        if self._redis:
            self._run_async(self._aadd(session_id, user_msg, assistant_msg))
        else:
            self._memory_add(session_id, user_msg, assistant_msg)

    def clear(self, session_id: str) -> None:
        if self._redis:
            self._run_async(self._aclear(session_id))
        else:
            with self._lock:
                self._store.pop(session_id, None)

    def session_exists(self, session_id: str) -> bool:
        if self._redis:
            async def _check():
                return await self._redis.exists(self._key(session_id)) > 0
            return self._run_async(_check())
        with self._lock:
            return session_id in self._store

    # ------------------------------------------------------------------
    # In-memory backend
    # ------------------------------------------------------------------

    def _memory_get(self, session_id: str) -> List[Dict]:
        with self._lock:
            entry = self._store.get(session_id)
            if not entry:
                return []
            if time.time() - entry["last_access"] > self.session_ttl:
                del self._store[session_id]
                return []
            entry["last_access"] = time.time()
            return list(entry["messages"])

    def _memory_add(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = {"messages": [], "last_access": time.time()}
            entry = self._store[session_id]
            entry["messages"].append({"role": "user", "content": user_msg})
            entry["messages"].append({"role": "assistant", "content": assistant_msg})
            entry["last_access"] = time.time()
            max_messages = self.max_turns * 2
            if len(entry["messages"]) > max_messages:
                entry["messages"] = entry["messages"][-max_messages:]

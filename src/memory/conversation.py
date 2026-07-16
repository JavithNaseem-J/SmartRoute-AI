"""
Conversation memory for multi-turn RAG sessions.

Design decisions:
- Backend: in-memory dict (default) or Redis (when REDIS_URL is set).
- In-memory is sufficient for single-node deployments and development.
  Redis is required for multi-node (multiple uvicorn workers/containers),
  because each worker has its own process memory — sessions created in
  worker A are invisible to worker B.
- Thread safety: Python dict reads/writes are protected by the GIL for
  single operations. We use a threading.Lock only around the compound
  read-modify-write in add_turn() to prevent lost updates under concurrent
  requests for the same session_id.
- Memory limits: history is capped at MAX_TURNS pairs (default 10).
  Without a cap, long sessions accumulate tokens indefinitely and will
  eventually exceed the model's context window, causing API errors.
  We drop the oldest turn first (FIFO eviction) — the most recent context
  is always preserved.
- Session TTL: Redis entries expire automatically via key TTL.
  In-memory sessions expire lazily on next access (checked in get_history).
"""
import os
import json
import time
import threading
from typing import List, Dict, Optional

from src.utils.logger import logger

# Maximum number of user+assistant turn PAIRS stored per session.
# 10 pairs = 20 messages. At ~200 tokens/message = ~4000 tokens of history,
# leaving ~4000 tokens for context + query + answer in an 8k context window.
MAX_TURNS = 10

# Session TTL in seconds. Sessions inactive longer than this are evicted.
# 30 minutes is a reasonable conversational session window.
SESSION_TTL_SECONDS = 30 * 60  # 30 minutes

_REDIS_AVAILABLE = False
try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    pass


class ConversationMemory:
    """Stores and retrieves per-session conversation history.

    Usage:
        memory = ConversationMemory()
        memory.add_turn(session_id, user_msg, assistant_msg)
        history = memory.get_history(session_id)
        # history → [{"role": "user", ...}, {"role": "assistant", ...}, ...]
    """

    def __init__(
        self,
        max_turns: int = MAX_TURNS,
        session_ttl: int = SESSION_TTL_SECONDS,
    ):
        self.max_turns = max_turns
        self.session_ttl = session_ttl

        # Try Redis first — required for multi-node deployments.
        self._redis: Optional[object] = None
        redis_url = os.getenv("REDIS_URL")
        if redis_url and _REDIS_AVAILABLE:
            try:
                self._redis = _redis_lib.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("ConversationMemory: using Redis backend")
            except Exception as e:
                logger.warning(
                    f"ConversationMemory: Redis unavailable ({e}), "
                    "falling back to in-memory store"
                )
                self._redis = None

        if self._redis is None:
            # In-memory store: {session_id: {"messages": [...], "last_access": float}}
            self._store: Dict[str, dict] = {}
            self._lock = threading.Lock()
            logger.info("ConversationMemory: using in-memory backend (single-node only)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> List[Dict]:
        """Return the message history for a session as a list of dicts.

        Returns:
            List of {"role": "user"|"assistant", "content": str} dicts,
            in chronological order, ready to insert into a messages list.
            Returns [] for unknown or expired sessions.
        """
        if self._redis:
            return self._redis_get(session_id)
        return self._memory_get(session_id)

    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """Append a user+assistant turn to the session history.

        Enforces MAX_TURNS by evicting the oldest turn pair when the cap
        is reached. This keeps memory bounded regardless of session length.
        """
        if self._redis:
            self._redis_add(session_id, user_msg, assistant_msg)
        else:
            self._memory_add(session_id, user_msg, assistant_msg)

    def clear(self, session_id: str) -> None:
        """Delete all history for a session."""
        if self._redis:
            self._redis.delete(self._key(session_id))
        else:
            with self._lock:
                self._store.pop(session_id, None)

    def session_exists(self, session_id: str) -> bool:
        """Return True if the session has any history."""
        if self._redis:
            return self._redis.exists(self._key(session_id)) > 0
        with self._lock:
            return session_id in self._store

    # ------------------------------------------------------------------
    # Redis backend
    # ------------------------------------------------------------------

    def _key(self, session_id: str) -> str:
        return f"smartroute:memory:{session_id}"

    def _redis_get(self, session_id: str) -> List[Dict]:
        try:
            raw = self._redis.get(self._key(session_id))
            if not raw:
                return []
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"ConversationMemory Redis get failed: {e}")
            return []

    def _redis_add(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        try:
            key = self._key(session_id)
            raw = self._redis.get(key)
            history: List[Dict] = json.loads(raw) if raw else []

            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": assistant_msg})

            # Enforce cap: each turn = 2 messages
            max_messages = self.max_turns * 2
            if len(history) > max_messages:
                history = history[-max_messages:]  # keep newest

            self._redis.setex(key, self.session_ttl, json.dumps(history))
        except Exception as e:
            logger.warning(f"ConversationMemory Redis add failed: {e}")

    # ------------------------------------------------------------------
    # In-memory backend
    # ------------------------------------------------------------------

    def _memory_get(self, session_id: str) -> List[Dict]:
        with self._lock:
            entry = self._store.get(session_id)
            if not entry:
                return []
            # Lazy TTL eviction
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

            # Enforce cap
            max_messages = self.max_turns * 2
            if len(entry["messages"]) > max_messages:
                entry["messages"] = entry["messages"][-max_messages:]

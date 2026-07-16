"""
Tests for ConversationMemory — in-memory backend only.
Redis backend is integration-tested separately when REDIS_URL is set.
"""
import time
import pytest
from src.memory.conversation import ConversationMemory


@pytest.fixture
def mem():
    """Return a ConversationMemory with tight limits for testing."""
    return ConversationMemory(max_turns=2, session_ttl=2)


def test_get_history_empty_session(mem):
    """Unknown session returns empty list."""
    history = mem.get_history("nonexistent-session")
    assert history == []


def test_add_and_get_single_turn(mem):
    """A single turn produces exactly 2 messages in the correct order."""
    mem.add_turn("s1", "Hello", "Hi there!")
    history = mem.get_history("s1")

    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hello"}
    assert history[1] == {"role": "assistant", "content": "Hi there!"}


def test_add_multiple_turns(mem):
    """Two turns produce 4 messages."""
    mem.add_turn("s2", "Q1", "A1")
    mem.add_turn("s2", "Q2", "A2")

    history = mem.get_history("s2")
    assert len(history) == 4


def test_fifo_eviction_on_max_turns(mem):
    """When max_turns=2 is exceeded the oldest turn is dropped (FIFO)."""
    mem.add_turn("s3", "Q1", "A1")  # turn 1
    mem.add_turn("s3", "Q2", "A2")  # turn 2
    mem.add_turn("s3", "Q3", "A3")  # turn 3 → should evict turn 1

    history = mem.get_history("s3")
    # max_turns=2 → max 4 messages
    assert len(history) == 4
    # Oldest pair (Q1/A1) must be gone
    contents = [m["content"] for m in history]
    assert "Q1" not in contents
    assert "A1" not in contents
    # Newest pair must be present
    assert "Q3" in contents
    assert "A3" in contents


def test_clear_session(mem):
    """clear() removes all history for the session."""
    mem.add_turn("s4", "Hello", "Hi")
    mem.clear("s4")
    assert mem.get_history("s4") == []


def test_session_exists(mem):
    """session_exists returns True only when history is present."""
    assert not mem.session_exists("s5")
    mem.add_turn("s5", "Q", "A")
    assert mem.session_exists("s5")


def test_ttl_eviction(mem):
    """Sessions expire after session_ttl seconds (set to 2 in fixture)."""
    mem.add_turn("s6", "Q", "A")
    assert len(mem.get_history("s6")) == 2

    time.sleep(3)  # Wait past TTL

    evicted = mem.get_history("s6")
    assert evicted == [], f"Expected eviction, got: {evicted}"


def test_multiple_independent_sessions(mem):
    """Different session IDs have isolated histories."""
    mem.add_turn("alice", "Q_alice", "A_alice")
    mem.add_turn("bob", "Q_bob", "A_bob")

    alice_h = mem.get_history("alice")
    bob_h = mem.get_history("bob")

    assert any(m["content"] == "Q_alice" for m in alice_h)
    assert not any(m["content"] == "Q_bob" for m in alice_h)
    assert any(m["content"] == "Q_bob" for m in bob_h)

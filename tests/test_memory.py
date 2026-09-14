import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.conversation import ConversationMemory


@pytest.fixture
async def mem(mock_redis):
    """Return a ConversationMemory backed by the conftest fakeredis instance."""
    return ConversationMemory(max_turns=2, session_ttl=2)


async def test_get_history_empty_session(mem):
    """Unknown session returns empty list."""
    history = await mem.get_history("user-a", "nonexistent-session")
    assert history == []


async def test_add_and_get_single_turn(mem):
    """A single turn produces exactly 2 messages in the correct order."""
    await mem.add_turn("user-a", "s1", "Hello", "Hi there!")
    history = await mem.get_history("user-a", "s1")

    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hello"}
    assert history[1] == {"role": "assistant", "content": "Hi there!"}


async def test_add_multiple_turns(mem):
    """Two turns produce 4 messages."""
    await mem.add_turn("user-a", "s2", "Q1", "A1")
    await mem.add_turn("user-a", "s2", "Q2", "A2")

    history = await mem.get_history("user-a", "s2")
    assert len(history) == 4


async def test_fifo_eviction_on_max_turns(mem):
    """When max_turns=2 is exceeded the oldest turn is dropped (FIFO)."""
    await mem.add_turn("user-a", "s3", "Q1", "A1")  # turn 1
    await mem.add_turn("user-a", "s3", "Q2", "A2")  # turn 2
    await mem.add_turn("user-a", "s3", "Q3", "A3")  # turn 3

    history = await mem.get_history("user-a", "s3")
    # max_turns=2 → max 4 messages
    assert len(history) == 4
    contents = [m["content"] for m in history]
    assert "Q1" not in contents
    assert "A1" not in contents
    assert "Q3" in contents
    assert "A3" in contents


async def test_clear_session(mem):
    """clear() removes all history for the session."""
    await mem.add_turn("user-a", "s4", "Hello", "Hi")
    await mem.clear("user-a", "s4")
    assert await mem.get_history("user-a", "s4") == []


async def test_ttl_eviction(mem):
    """Sessions expire after session_ttl seconds (set to 2 in fixture)."""
    await mem.add_turn("user-a", "s6", "Q", "A")
    assert len(await mem.get_history("user-a", "s6")) == 2

    time.sleep(3)

    evicted = await mem.get_history("user-a", "s6")
    assert evicted == [], f"Expected eviction, got: {evicted}"


async def test_multiple_independent_sessions(mem):
    """Different session IDs have isolated histories."""
    await mem.add_turn("alice", "s1", "Q_alice", "A_alice")
    await mem.add_turn("bob", "s1", "Q_bob", "A_bob")

    alice_h = await mem.get_history("alice", "s1")
    bob_h = await mem.get_history("bob", "s1")

    assert any(m["content"] == "Q_alice" for m in alice_h)
    assert not any(m["content"] == "Q_bob" for m in alice_h)
    assert any(m["content"] == "Q_bob" for m in bob_h)


async def test_same_session_id_isolated_by_user(mem):
    """The same client session ID cannot cross user boundaries."""
    await mem.add_turn("alice", "shared-session", "alice question", "alice answer")

    assert await mem.get_history("bob", "shared-session") == []

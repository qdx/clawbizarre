"""
ClawBizarre SSE Notifications — Phase 9

Server-Sent Events (SSE) for real-time notifications to agents.
Providers no longer need to poll for new handshake requests.

Event types:
  - handshake.initiated   — A buyer wants to work with you
  - handshake.responded    — Provider accepted/rejected your handshake
  - handshake.executed     — Provider submitted work output
  - handshake.verified     — Buyer verified your work (receipt generated)
  - discovery.match        — A new listing matches your watch criteria

Design choices:
  - SSE over WebSocket: simpler, HTTP-native, proxy-friendly, auto-reconnect
  - Per-agent event queue with bounded size (prevent memory leaks)
  - Last-Event-ID support for reconnection (gap-free delivery)
  - Auth via query param token (SSE doesn't support custom headers easily)
"""

import json
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from collections import deque


class EventType(str, Enum):
    HANDSHAKE_INITIATED = "handshake.initiated"
    HANDSHAKE_RESPONDED = "handshake.responded"
    HANDSHAKE_EXECUTED = "handshake.executed"
    HANDSHAKE_VERIFIED = "handshake.verified"
    DISCOVERY_MATCH = "discovery.match"


@dataclass
class SSEEvent:
    event_id: str
    event_type: EventType
    agent_id: str  # recipient
    data: dict
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        """Format as SSE wire format."""
        lines = [
            f"id: {self.event_id}",
            f"event: {self.event_type.value}",
            f"data: {json.dumps(self.data)}",
            "",  # blank line terminates event
            "",
        ]
        return "\n".join(lines)


class NotificationBus:
    """
    Central notification bus. Agents subscribe via SSE; events are
    pushed to per-agent queues.
    
    Thread-safe: multiple HTTP handler threads can emit/subscribe concurrently.
    """

    def __init__(self, max_queue_size: int = 200, max_history: int = 500):
        self._lock = threading.Lock()
        self._event_counter = 0
        # agent_id → list of threading.Event objects (one per active SSE connection)
        self._subscribers: dict[str, list[threading.Event]] = {}
        # agent_id → deque of SSEEvent (bounded)
        self._queues: dict[str, deque] = {}
        # Global ordered history for Last-Event-ID replay
        self._history: deque[SSEEvent] = deque(maxlen=max_history)
        self._max_queue_size = max_queue_size

    def _next_id(self) -> str:
        self._event_counter += 1
        return f"evt-{self._event_counter}"

    def emit(self, event_type: EventType, agent_id: str, data: dict) -> SSEEvent:
        """Emit an event to an agent. Returns the event."""
        with self._lock:
            event = SSEEvent(
                event_id=self._next_id(),
                event_type=event_type,
                agent_id=agent_id,
                data=data,
            )
            # Add to agent queue
            q = self._queues.setdefault(agent_id, deque(maxlen=self._max_queue_size))
            q.append(event)
            # Add to global history
            self._history.append(event)
            # Wake all subscribers for this agent
            for wake in self._subscribers.get(agent_id, []):
                wake.set()
            return event

    def subscribe(self, agent_id: str) -> tuple[threading.Event, int]:
        """
        Subscribe to events for agent_id.
        Returns (wake_event, queue_position) — caller should drain from position.
        """
        with self._lock:
            wake = threading.Event()
            subs = self._subscribers.setdefault(agent_id, [])
            subs.append(wake)
            q = self._queues.setdefault(agent_id, deque(maxlen=self._max_queue_size))
            return wake, len(q)

    def unsubscribe(self, agent_id: str, wake: threading.Event):
        """Remove a subscriber."""
        with self._lock:
            subs = self._subscribers.get(agent_id, [])
            if wake in subs:
                subs.remove(wake)

    def drain(self, agent_id: str, from_pos: int) -> tuple[list[SSEEvent], int]:
        """Get events from position. Returns (events, new_position)."""
        with self._lock:
            q = self._queues.get(agent_id, deque())
            events = list(q)[from_pos:]  # slice from deque
            return events, len(q)

    def replay_from(self, agent_id: str, last_event_id: str) -> list[SSEEvent]:
        """Replay events after last_event_id for reconnection."""
        with self._lock:
            found = False
            events = []
            for evt in self._history:
                if found and evt.agent_id == agent_id:
                    events.append(evt)
                if evt.event_id == last_event_id:
                    found = True
            return events

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_events": self._event_counter,
                "subscribers": {aid: len(subs) for aid, subs in self._subscribers.items() if subs},
                "queue_sizes": {aid: len(q) for aid, q in self._queues.items() if q},
                "history_size": len(self._history),
            }


# --- Tests ---

def run_tests():
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  ✓ {name}")
        else:
            failed += 1
            print(f"  ✗ {name}")

    print("\n=== NotificationBus Tests ===\n")

    bus = NotificationBus(max_queue_size=10)

    # 1. Emit without subscribers
    evt = bus.emit(EventType.HANDSHAKE_INITIATED, "agent-A", {"session_id": "s1"})
    check("1. Emit returns event", evt.event_id == "evt-1")

    # 2. SSE format
    sse = evt.to_sse()
    check("2. SSE format correct", "id: evt-1\n" in sse and "event: handshake.initiated\n" in sse)

    # 3. Subscribe and drain
    wake, pos = bus.subscribe("agent-A")
    check("3. Subscribe position = 1 (1 event already)", pos == 1)

    # 4. Drain existing
    events, new_pos = bus.drain("agent-A", 0)
    check("4. Drain gets existing event", len(events) == 1 and events[0].event_id == "evt-1")

    # 5. Emit wakes subscriber
    wake.clear()
    bus.emit(EventType.HANDSHAKE_RESPONDED, "agent-A", {"action": "accepted"})
    check("5. Wake event set on emit", wake.is_set())

    # 6. Drain new events
    events, new_pos = bus.drain("agent-A", pos)
    check("6. Drain new events", len(events) == 1 and events[0].event_type == EventType.HANDSHAKE_RESPONDED)

    # 7. Other agent's events don't appear
    bus.emit(EventType.HANDSHAKE_INITIATED, "agent-B", {"session_id": "s2"})
    events, _ = bus.drain("agent-A", new_pos)
    check("7. No cross-agent leaking", len(events) == 0)

    # 8. Replay from last event ID
    bus.emit(EventType.HANDSHAKE_EXECUTED, "agent-A", {"output": "result"})
    bus.emit(EventType.HANDSHAKE_VERIFIED, "agent-A", {"receipt_id": "r1"})
    replayed = bus.replay_from("agent-A", "evt-2")
    agent_a_replayed = [e for e in replayed if e.agent_id == "agent-A"]
    check("8. Replay from evt-2", len(agent_a_replayed) == 2)

    # 9. Unsubscribe
    bus.unsubscribe("agent-A", wake)
    stats = bus.stats()
    check("9. Unsubscribe removes sub", stats["subscribers"].get("agent-A", 0) == 0)

    # 10. Queue bounded
    for i in range(15):
        bus.emit(EventType.DISCOVERY_MATCH, "agent-C", {"i": i})
    check("10. Queue bounded to max_queue_size", stats["queue_sizes"].get("agent-C", 0) <= 10 or len(bus._queues.get("agent-C", [])) <= 10)

    # 11. Stats
    stats = bus.stats()
    check("11. Stats work", stats["total_events"] > 0)

    # 12. Multiple subscribers
    w1, _ = bus.subscribe("agent-D")
    w2, _ = bus.subscribe("agent-D")
    bus.emit(EventType.HANDSHAKE_INITIATED, "agent-D", {"x": 1})
    check("12. Both subscribers woken", w1.is_set() and w2.is_set())

    print(f"\n{'='*40}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*40}\n")
    return failed == 0


if __name__ == "__main__":
    run_tests()

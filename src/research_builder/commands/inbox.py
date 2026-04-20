"""Process-wide inbox for commands routed to a specific agent.

This is the seed of the in-process event bus we agreed to defer until needed.
For now it's the *minimum* needed to support chat: per-agent_id queues plus a
single registered handler for orchestrator-targeted messages (which are
handled live, not queued, since the orchestrator is always available).

Phase-targeted messages are queued and drained by ExecutionLoop right before
the next SubAgent attempt for that phase. This means a chat message you type
while a phase is running will be delivered as additional user context on the
phase's next attempt — exactly the behavior the user asked for.

When parallel phase execution lands, this same inbox keeps working: each
SubAgent drains its own queue. When we eventually want a real pub/sub bus
(multiple subscribers per agent_id, fan-out to metrics/logging consumers),
the natural refactor is to replace the dict-of-queues with a small bus and
have ExecutionLoop subscribe to its own inbox topic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

# An async handler the listener calls when a message targets the orchestrator.
OrchestratorHandler = Callable[[str], Awaitable[None]]


class Inbox:
    def __init__(self) -> None:
        self._queues: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()
        self._orchestrator_handler: OrchestratorHandler | None = None

    def register_orchestrator_handler(self, handler: OrchestratorHandler) -> None:
        self._orchestrator_handler = handler

    async def deliver(self, agent_id: str, text: str) -> None:
        """Deliver a chat message to its target."""
        if agent_id == "orchestrator":
            if self._orchestrator_handler is None:
                logger.warning("inbox: orchestrator message received but no handler registered")
                return
            try:
                await self._orchestrator_handler(text)
            except Exception as e:
                logger.warning("inbox: orchestrator handler raised: %s", e)
            return

        async with self._lock:
            self._queues.setdefault(agent_id, []).append(text)
        logger.info("inbox: queued message for %s (%d pending)", agent_id, len(self._queues[agent_id]))

    def drain(self, agent_id: str) -> list[str]:
        """Synchronous drain — safe to call from inside ExecutionLoop without
        awaiting because writes are append-only and the GIL serializes them."""
        msgs = self._queues.pop(agent_id, [])
        return msgs


_singleton: Inbox | None = None


def get_inbox() -> Inbox:
    global _singleton
    if _singleton is None:
        _singleton = Inbox()
    return _singleton

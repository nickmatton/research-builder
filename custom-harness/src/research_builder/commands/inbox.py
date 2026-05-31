"""Process-wide inbox for commands routed to a specific agent.

This is the seed of the in-process event bus we agreed to defer until needed.
It carries two payloads today:

1. **Chat messages** — per-agent_id text queues, drained by ExecutionLoop right
   before the next SubAgent attempt for that phase. A chat message you type
   while a phase is running will be delivered as additional user context on
   the phase's next attempt.

2. **Interventions** (Stage 3a) — typed operator commands (edit_refined_spec,
   force_retry, inject_note, jump_back). Routed into bucketed queues keyed by
   ``(phase_id, hook_point)`` and drained at well-defined safe points in the
   execution loop (pre_refiner / pre_builder / pre_verifier / between_phases).

Both share the same singleton Inbox so the file-tailing CommandListener has
exactly one routing target.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# An async handler the listener calls when a message targets the orchestrator.
OrchestratorHandler = Callable[[str], Awaitable[None]]

# Bucket key for an intervention: (phase_id, hook) where hook is one of
# "pre_refiner" | "pre_researcher" | "pre_builder" | "pre_verifier" | "between_phases".
# Global-scope interventions use phase_id="*".
InterventionKey = tuple[str, str]


class Inbox:
    def __init__(self) -> None:
        self._queues: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()
        self._orchestrator_handler: OrchestratorHandler | None = None
        # Intervention buckets keyed by (phase_id, hook). Each holds the raw
        # command dicts in arrival order. Drained synchronously at hook
        # boundaries — no awaiting from inside _execute_phase.
        self._interventions: dict[InterventionKey, list[dict]] = {}
        # cmd_id LRU dedupe: if the listener replays after a crash or a
        # client double-submits, we drop repeats. Bounded so it doesn't grow
        # without bound during long sessions.
        self._seen_cmd_ids: "OrderedDict[str, None]" = OrderedDict()
        self._seen_cap = 1024

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

    # ─── Interventions (Stage 3a) ─────────────────────────────────────────

    # Each intervention type advertises which hook point should fire it. For
    # types that fan out to multiple hooks (inject_note targeting several
    # agents), the listener enqueues one copy per (phase_id, hook).
    INTERVENTION_TYPES = {
        "edit_refined_spec",
        "force_retry",
        "inject_note",
        "jump_back",
    }

    def post_intervention(self, cmd: dict) -> bool:
        """Route a typed intervention into the matching bucket(s).

        Returns False if the command is a duplicate (by cmd_id) or malformed.
        """
        cmd_id = cmd.get("cmd_id")
        if cmd_id is not None:
            if cmd_id in self._seen_cmd_ids:
                logger.info("inbox: dropping duplicate intervention cmd_id=%s", cmd_id)
                return False
            self._seen_cmd_ids[cmd_id] = None
            if len(self._seen_cmd_ids) > self._seen_cap:
                self._seen_cmd_ids.popitem(last=False)

        ctype = cmd.get("type")
        if ctype not in self.INTERVENTION_TYPES:
            logger.warning("inbox: post_intervention got unknown type=%r", ctype)
            return False

        payload = cmd.get("payload", {})
        phase_id = payload.get("phase_id") or payload.get("to_phase_id") or "*"

        # Decide which hook(s) consume this command.
        keys: list[InterventionKey] = []
        if ctype == "edit_refined_spec":
            before = payload.get("before_agent", "builder")
            keys.append((phase_id, f"pre_{before}"))
        elif ctype == "force_retry" or ctype == "jump_back":
            keys.append((phase_id, "between_phases"))
        elif ctype == "inject_note":
            scope = payload.get("scope", "phase")
            targets = payload.get("target_agents") or ["builder"]
            target_pid = "*" if scope == "global" else phase_id
            for agent in targets:
                keys.append((target_pid, f"pre_{agent}"))

        if not keys:
            logger.warning("inbox: post_intervention could not route cmd=%s", cmd)
            return False

        for key in keys:
            self._interventions.setdefault(key, []).append(cmd)
        logger.info("inbox: queued intervention type=%s keys=%s", ctype, keys)
        return True

    def drain_interventions(self, phase_id: str, hook: str) -> list[dict]:
        """Drain all queued interventions for ``(phase_id, hook)``.

        Also drains the wildcard bucket ``("*", hook)`` so global-scope notes
        fire for every phase that reaches the hook.
        """
        out: list[dict] = []
        out.extend(self._interventions.pop((phase_id, hook), []))
        out.extend(self._interventions.pop(("*", hook), []))
        return out


_singleton: Inbox | None = None


def get_inbox() -> Inbox:
    global _singleton
    if _singleton is None:
        _singleton = Inbox()
    return _singleton

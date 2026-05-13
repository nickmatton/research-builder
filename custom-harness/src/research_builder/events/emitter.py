"""Append-only JSONL event sink for external observers.

Design notes
------------
This is the *streaming sink* half of the observability story. It is intentionally
the dumbest possible thing that works:

- One JSON object per line, written with a single ``write()`` + ``flush()`` call.
  POSIX guarantees small (<PIPE_BUF) appends to ``O_APPEND`` files are atomic, so
  this is safe even when phases run in parallel from multiple coroutines/threads.
- No in-process pub/sub. If/when we want a second consumer (e.g. a metrics
  aggregator that needs sub-millisecond reaction), the natural refactor is to
  introduce a tiny ``EventBus`` here that fans out to N subscribers — the
  JSONL writer becoming one of them. Until then, YAGNI.
- Disabled by default. ``get_emitter()`` returns ``None`` unless the environment
  variable ``RESEARCH_BUILDER_EVENT_LOG`` is set. Callers wrap emissions with
  ``if emitter:`` so the production path pays zero cost when no viewer is
  attached.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_VAR = "RESEARCH_BUILDER_EVENT_LOG"


class EventEmitter:
    """Append structured events as JSONL to a file AND fan out to in-process subscribers."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Touch the file so tailers can open it immediately.
        self.path.touch(exist_ok=True)
        # In-process subscribers — each receives the full event dict.
        # Used by the inline TUI viewer to render styled output in the same
        # terminal as the running harness.
        self._subscribers: list[Any] = []

    def subscribe(self, callback) -> None:
        """Register a callback(event_dict) called on every emit().

        Exceptions in subscribers are caught and logged; they never block
        emission or affect other subscribers.
        """
        self._subscribers.append(callback)

    def unsubscribe(self, callback) -> None:
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass

    def emit(
        self,
        type: str,
        agent_id: str,
        parent_id: str | None = None,
        **payload: Any,
    ) -> None:
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "type": type,
            "agent_id": agent_id,
            "parent_id": parent_id,
            **payload,
        }
        try:
            line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
        except Exception as e:
            logger.warning("event_emitter: failed to serialize %s: %s", type, e)
            return
        with self._lock:
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
            except Exception as e:
                logger.warning("event_emitter: failed to write %s: %s", type, e)
        # Fan out to in-process subscribers (after the file write so the
        # forensic trail captures the event even if a subscriber crashes).
        for cb in list(self._subscribers):
            try:
                cb(record)
            except Exception:
                logger.warning("event_emitter: subscriber %r raised", cb, exc_info=True)


_singleton: EventEmitter | None = None
_singleton_path: str | None = None


def get_emitter() -> EventEmitter | None:
    """Return a process-wide emitter if RESEARCH_BUILDER_EVENT_LOG is set, else None."""
    global _singleton, _singleton_path
    path = os.environ.get(ENV_VAR)
    if not path:
        return None
    if _singleton is None or _singleton_path != path:
        try:
            _singleton = EventEmitter(Path(path))
            _singleton_path = path
        except Exception as e:
            logger.warning("event_emitter: could not initialize at %s: %s", path, e)
            return None
    return _singleton

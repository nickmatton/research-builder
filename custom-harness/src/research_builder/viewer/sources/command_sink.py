"""Append commands to commands.jsonl — the inbound counterpart to events.jsonl.

The viewer is otherwise read-only; this is the only place it writes to the
filesystem. We use the same single-line append + flush pattern as
research-builder's EventEmitter so writes from multiple processes interleave
safely.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


class CommandSink:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()

    def send_chat(self, agent_id: str, text: str) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "type": "chat_message",
            "agent_id": agent_id,
            "text": text,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()

"""Async tail of commands.jsonl — the inbound counterpart to events.jsonl.

Symmetric with research_builder.events.emitter: dumbest possible thing that
works. The viewer appends one JSON line per command; we tail the file with a
periodic poll, parse, and dispatch into the in-process Inbox.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from .inbox import get_inbox

logger = logging.getLogger(__name__)


class CommandListener:
    def __init__(self, path: Path, *, poll_interval: float = 0.1) -> None:
        self.path = Path(path)
        self.poll_interval = poll_interval
        self._stop = asyncio.Event()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        """Tail the commands file forever (until stop() or task cancellation)."""
        inbox = get_inbox()
        # Start at end-of-file: commands typed before the listener started are
        # ignored on purpose. We don't want to replay stale chat at startup.
        try:
            pos = self.path.stat().st_size
        except FileNotFoundError:
            pos = 0
        inode = _inode(self.path)
        buffer = ""

        while not self._stop.is_set():
            try:
                st = self.path.stat()
            except FileNotFoundError:
                await asyncio.sleep(self.poll_interval)
                continue

            cur_inode = _inode(self.path)
            if cur_inode != inode or st.st_size < pos:
                pos = 0
                inode = cur_inode
                buffer = ""

            if st.st_size > pos:
                with self.path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    buffer += f.read()
                    pos = f.tell()
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cmd = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("command_listener: skipping invalid line: %r", line[:200])
                        continue
                    await self._dispatch(inbox, cmd)
            else:
                await asyncio.sleep(self.poll_interval)

    async def _dispatch(self, inbox, cmd: dict) -> None:
        ctype = cmd.get("type")
        if ctype == "chat_message":
            agent_id = cmd.get("agent_id")
            text = cmd.get("text")
            if not agent_id or not text:
                return
            await inbox.deliver(agent_id, text)
            return

        if ctype in inbox.INTERVENTION_TYPES:
            inbox.post_intervention(cmd)
            return

        logger.info("command_listener: ignoring unknown type=%r", ctype)


def _inode(path: Path) -> int:
    try:
        return os.stat(path).st_ino
    except FileNotFoundError:
        return -1

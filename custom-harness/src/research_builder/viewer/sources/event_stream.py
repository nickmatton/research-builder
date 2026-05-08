"""Async tail of a JSONL event stream.

No inotify, no fsevents — just a periodic poll that tracks file position
and yields any new complete lines. Robust to file truncation and rotation
(detected via inode change or shrinking size).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import AsyncIterator


async def tail_jsonl(
    path: Path,
    *,
    from_start: bool = True,
    poll_interval: float = 0.1,
    stop_event: asyncio.Event | None = None,
) -> AsyncIterator[dict]:
    """Yield JSON records appended to ``path``.

    Args:
        path: File to tail. Created if it does not exist.
        from_start: If True, replay every existing line before tailing. If
            False, seek to end first.
        poll_interval: Seconds between polls when no new data is available.
        stop_event: Optional asyncio.Event; when set, the iterator stops.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    pos = 0 if from_start else path.stat().st_size
    inode = _inode(path)
    buffer = ""

    while True:
        if stop_event is not None and stop_event.is_set():
            return

        try:
            st = path.stat()
        except FileNotFoundError:
            await asyncio.sleep(poll_interval)
            continue

        # Detect rotation/truncation: inode changed or file shrank.
        cur_inode = _inode(path)
        if cur_inode != inode or st.st_size < pos:
            pos = 0
            inode = cur_inode
            buffer = ""

        if st.st_size > pos:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        else:
            await asyncio.sleep(poll_interval)


def _inode(path: Path) -> int:
    try:
        return os.stat(path).st_ino
    except FileNotFoundError:
        return -1

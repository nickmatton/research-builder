"""WebSocket endpoint that streams the workspace's events.jsonl to clients.

On connect:
  1. Replays every existing event in events.jsonl (so a freshly-loaded
     UI sees the full history of a completed or in-progress run).
  2. Tails the file, emitting each new line as soon as it lands.

If no active workspace yet (launcher mode), the WS waits for one to be
set instead of erroring — the page typically opens the socket before the
user uploads a paper.

Symmetric with ``research_builder.commands.listener``: polls every 100ms,
handles file rotation (inode change) and truncation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .state import WebState

logger = logging.getLogger(__name__)

POLL_INTERVAL = 0.1
WORKSPACE_WAIT_INTERVAL = 0.5


def build_events_router(state: WebState) -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws/events")
    async def events_ws(ws: WebSocket) -> None:
        await ws.accept()
        state.register_websocket(ws)
        try:
            # Wait until a workspace is active. The frontend opens this
            # connection on page load (before the user uploads), so we
            # don't want to bounce them — just idle until /api/launch
            # sets the workspace.
            while state.workspace is None:
                if await _sleep_or_shutdown(state, WORKSPACE_WAIT_INTERVAL):
                    return
            workspace = state.workspace
            events_path = workspace / "logs" / "events.jsonl"
            events_path.parent.mkdir(parents=True, exist_ok=True)
            events_path.touch(exist_ok=True)
            await _tail_and_stream(state, ws, events_path)
        except WebSocketDisconnect:
            logger.debug("events_ws: client disconnected")
        except asyncio.CancelledError:
            # App shutdown — bubble the cancellation so uvicorn's lifespan
            # graceful-shutdown completes promptly. Do NOT swallow.
            raise
        except Exception:
            logger.exception("events_ws: stream failed")
            try:
                await ws.close()
            except Exception:
                pass
        finally:
            state.unregister_websocket(ws)

    return router


async def _sleep_or_shutdown(state: WebState, timeout: float) -> bool:
    """Sleep up to ``timeout`` seconds, or return early if shutdown fires.

    Returns ``True`` if the shutdown_event was set during the wait (caller
    should bail), ``False`` if the full sleep elapsed normally.

    This is what makes the tail loop responsive to app shutdown. Plain
    ``asyncio.sleep`` would happily wait out the full interval even if
    ``ws.close()`` had been called server-side — the WS handle doesn't
    interrupt an already-awaiting sleep — so uvicorn's lifespan teardown
    would hang on "Waiting for background tasks to complete."
    """
    try:
        await asyncio.wait_for(state.shutdown_event.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False


async def _tail_and_stream(state: WebState, ws: WebSocket, path: Path) -> None:
    """Read existing file → stream → tail forever (or until shutdown)."""
    pos = 0
    inode = _inode(path)
    buffer = ""

    while True:
        if state.shutdown_event.is_set():
            return
        try:
            st = path.stat()
        except FileNotFoundError:
            if await _sleep_or_shutdown(state, POLL_INTERVAL):
                return
            continue

        # Rotation or truncation → rewind.
        cur_inode = _inode(path)
        if cur_inode != inode or st.st_size < pos:
            pos = 0
            inode = cur_inode
            buffer = ""

        if st.st_size > pos:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                buffer += f.read()
                pos = f.tell()
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("events_ws: skipping invalid line: %r", line[:200])
                    continue
                await ws.send_json(payload)
        else:
            if await _sleep_or_shutdown(state, POLL_INTERVAL):
                return


def _inode(path: Path) -> int:
    try:
        return os.stat(path).st_ino
    except FileNotFoundError:
        return -1

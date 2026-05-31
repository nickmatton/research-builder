"""Shared mutable state for the web app.

The original single-workspace model bound the workspace path at app-create
time. To support an upload-then-launch UX (``research-builder-app`` with
no workspace argument), routers now read the active workspace
dynamically from this state singleton instead of closing over a path.

Lifecycle for a single ``research-builder-app`` process:

  1. Boot — runs_dir set, workspace=None, proc=None.
  2. User uploads a paper → workspace=<runs_dir>/<stem>/, paper saved.
  3. ``launch_pipeline`` spawns ``research-builder --auto`` subprocess.
  4. Pipeline emits events.jsonl → frontend streams them.
  5. On app shutdown, ``terminate`` SIGTERMs the subprocess tree.

Only one active workspace + pipeline at a time. Re-uploading a paper
while a run is in progress is rejected by the API layer.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import WebSocket

logger = logging.getLogger(__name__)


def paper_stem(name: str) -> str:
    """Derive a directory-safe folder name from a paper filename."""
    stem = Path(name).stem
    sanitized = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return sanitized or "paper"


@dataclass
class WebState:
    runs_dir: Path
    workspace: Path | None = None
    proc: subprocess.Popen | None = None
    pipeline_log: Path | None = None
    # When True, the spawned ``research-builder`` subprocess is launched
    # with ``--dev`` — routes through the Claude Code subscription rather
    # than requiring ANTHROPIC_API_KEY. Set once at boot from the CLI flag.
    dev_mode: bool = False
    # Extra dirs to forward to the spawned pipeline via --allow-dir. Set
    # once at boot from the CLI flag; each entry is an absolute path.
    extra_allowed_dirs: list[Path] = field(default_factory=list)
    # Open /ws/events WebSocket connections. Tracked so app shutdown can
    # close them cleanly — otherwise uvicorn waits forever for the tail
    # loop to drop.
    websockets: set["WebSocket"] = field(default_factory=set)
    # Cooperative shutdown signal. The events tail loop waits on this
    # alongside its poll-interval sleep, so when shutdown sets it the loop
    # returns immediately instead of finishing whichever ``asyncio.sleep``
    # it happened to be in (which can be up to POLL_INTERVAL late, but
    # more importantly was being missed entirely because ``ws.close()``
    # server-side doesn't interrupt an already-awaiting sleep).
    shutdown_event: asyncio.Event = field(default_factory=asyncio.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def register_websocket(self, ws: "WebSocket") -> None:
        with self._lock:
            self.websockets.add(ws)

    def unregister_websocket(self, ws: "WebSocket") -> None:
        with self._lock:
            self.websockets.discard(ws)

    def set_workspace(self, workspace: Path) -> None:
        with self._lock:
            self.workspace = workspace.resolve()

    def clear_workspace(self) -> None:
        with self._lock:
            self.workspace = None

    def pipeline_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def pipeline_status(self) -> dict:
        if self.proc is None:
            return {"state": "idle", "pid": None, "exit_code": None}
        rc = self.proc.poll()
        if rc is None:
            return {"state": "running", "pid": self.proc.pid, "exit_code": None}
        return {"state": "finished", "pid": self.proc.pid, "exit_code": rc}

    def set_proc(self, proc: subprocess.Popen, log_path: Path) -> None:
        with self._lock:
            self.proc = proc
            self.pipeline_log = log_path

    def terminate(self) -> None:
        """Kill the pipeline subprocess and all its descendants. Idempotent."""
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return

        # Snapshot descendants before SIGTERM so we can clean up children
        # that escaped the root's process group (e.g. via the claude CLI).
        descendants = _descendant_pids(proc.pid)

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass

        alive: list[int] = []
        for pid in descendants:
            try:
                os.kill(pid, signal.SIGTERM)
                alive.append(pid)
            except ProcessLookupError:
                pass

        if not alive:
            return

        import time
        time.sleep(1)
        for pid in alive:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _descendant_pids(root_pid: int) -> list[int]:
    """Return all descendant PIDs of *root_pid*, recursively."""
    try:
        result = subprocess.run(
            ["pgrep", "-P", str(root_pid)],
            capture_output=True, text=True, timeout=5,
        )
        children = [int(p) for p in result.stdout.strip().split() if p.strip()]
    except Exception:
        return []
    descendants = list(children)
    for child in children:
        descendants.extend(_descendant_pids(child))
    return descendants

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
import re
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
        # Pre-write snapshots keyed by process_id. Populated when a Write/
        # Edit-class tool is about to execute, drained when the matching
        # process_result fires. See ``capture_file_before`` /
        # ``emit_file_write`` below. Bounded by the lifetime of an active
        # tool call, but we cap the size as a belt-and-suspenders measure
        # in case a crash leaves stragglers — old entries get evicted when
        # the dict hits MAX_PENDING_SNAPSHOTS.
        self._pending_before: dict[str, dict[str, Any]] = {}

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


_PAGES_RE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$")


def _parse_pages(raw: Any) -> tuple[int, int] | None:
    """Parse the Read tool's ``pages`` argument into (start, end) inclusive.

    Accepts ``"3"``, ``"3-5"``, or an int. Returns None if unparseable.
    """
    if raw is None:
        return None
    if isinstance(raw, int):
        return (raw, raw)
    if not isinstance(raw, str):
        return None
    m = _PAGES_RE.match(raw)
    if not m:
        return None
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else start
    if end < start:
        end = start
    return (start, end)


def _paths_match(read_path: str, paper_path: str | Path) -> bool:
    """True iff the Read tool's file_path refers to the paper PDF.

    Tries resolved-absolute match first; falls back to basename equality.
    Basename fallback is safe in practice — workspaces only ever contain a
    single paper PDF, and the agent has no other source of PDFs (the
    citation tool returns markdown, not files).
    """
    try:
        if Path(read_path).resolve() == Path(paper_path).resolve():
            return True
    except Exception:
        pass
    return Path(read_path).name == Path(paper_path).name


def maybe_emit_paper_read(
    emitter: "EventEmitter | None",
    *,
    agent_id: str,
    parent_id: str | None,
    tool_name: str,
    tool_input: dict[str, Any],
    paper_path: str | Path,
) -> None:
    """If this tool call is a Read on the paper PDF, emit a ``paper_read`` event.

    The event carries ``page_start`` / ``page_end`` (inclusive, 1-indexed) so the
    UI can highlight which slice of the paper an agent is loading as context.
    If ``pages`` is absent, emits ``page_start=null, page_end=null`` to mean
    "whole document" — caller's UI decides how to render that.
    """
    if emitter is None or tool_name != "Read":
        return
    file_path = tool_input.get("file_path") or ""
    if not file_path or not _paths_match(file_path, paper_path):
        return
    parsed = _parse_pages(tool_input.get("pages"))
    page_start = parsed[0] if parsed else None
    page_end = parsed[1] if parsed else None
    emitter.emit(
        "paper_read",
        agent_id=agent_id,
        parent_id=parent_id,
        page_start=page_start,
        page_end=page_end,
        paper_path=str(paper_path),
    )


def emit_artifact_created(
    emitter: "EventEmitter | None",
    *,
    agent_id: str,
    parent_id: str | None = None,
    artifact_type: str,
    path: str | Path,
    producer: str,
    **extra: Any,
) -> None:
    """Unified emission for "a new on-disk artifact exists".

    Lets the frontend show a generic "new artifact" notification driven by a
    single event type, regardless of which producer wrote it (skeleton author,
    section author, critic, claims extractor, builder, verifier, …).

    ``artifact_type`` is a short tag like ``"top_level_spec"`` /
    ``"section_spec"`` / ``"section_critique"`` / ``"claims_ledger"`` /
    ``"verification_report"`` / ``"reproduction_report"`` etc.
    """
    if emitter is None:
        return
    emitter.emit(
        "artifact_created",
        agent_id=agent_id,
        parent_id=parent_id,
        artifact_type=artifact_type,
        path=str(path),
        producer=producer,
        **extra,
    )


# ─── File-write diff snapshots ──────────────────────────────────────────
#
# When the agent invokes a Write/Edit-class tool, we want the UI to be able
# to show a before/after diff. To do that we snapshot the file's pre-state
# the moment the tool call lands (process_started) and re-read it after the
# result lands (process_result), emitting a single ``file_write`` event
# tying the two via process_id.
#
# Constraints:
# - Cap content at 256KB. Larger files emit truncated=true with the head;
#   the UI can still show a partial diff and a "truncated" indicator.
# - Skip binary content (presence of NUL byte in first 8KB heuristic). The
#   harness's agents do write the occasional notebook with binary blobs;
#   sending them through the event log balloons the JSONL pointlessly.
# - Pre-existing file missing → before=null (Write creating a new file).
# - Post-state missing → after=null (Edit on a path that errored out).

WRITE_TOOL_NAMES = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
_MAX_DIFF_BYTES = 256 * 1024  # 256KB per side
_BINARY_PROBE_BYTES = 8 * 1024  # check first 8KB for NUL
_MAX_PENDING_SNAPSHOTS = 256


def _is_probably_binary(data: bytes) -> bool:
    return b"\x00" in data[:_BINARY_PROBE_BYTES]


def _read_text_capped(path: Path) -> tuple[str | None, bool, bool]:
    """Return (text, truncated, binary). text is None if file missing."""
    try:
        if not path.exists() or not path.is_file():
            return None, False, False
        raw = path.read_bytes()
    except OSError:
        return None, False, False
    if _is_probably_binary(raw):
        return None, False, True
    truncated = len(raw) > _MAX_DIFF_BYTES
    if truncated:
        raw = raw[:_MAX_DIFF_BYTES]
    try:
        return raw.decode("utf-8"), truncated, False
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), truncated, False


def capture_file_before(
    emitter: "EventEmitter | None",
    *,
    process_id: str,
    tool_name: str,
    file_path: str | None,
    cwd: str | Path | None = None,
) -> None:
    """Snapshot a file's pre-write state.

    Call at ``process_started`` emission. Pairs with ``emit_file_write``
    at ``process_result``. Silently no-ops when:
    - emitter is None,
    - tool isn't a write tool,
    - file_path is empty / unparseable,
    - file is binary or unreadable (the after-pass will skip too).
    """
    if emitter is None or tool_name not in WRITE_TOOL_NAMES:
        return
    if not file_path:
        return
    path = Path(file_path)
    if not path.is_absolute() and cwd is not None:
        path = Path(cwd) / path
    text, truncated, binary = _read_text_capped(path)
    if binary:
        return
    # Cap dict growth — drop oldest in FIFO order if a crash left strays.
    if len(emitter._pending_before) >= _MAX_PENDING_SNAPSHOTS:
        try:
            oldest = next(iter(emitter._pending_before))
            emitter._pending_before.pop(oldest, None)
        except StopIteration:
            pass
    emitter._pending_before[process_id] = {
        "tool_name": tool_name,
        "file_path": str(path),
        "before": text,
        "before_truncated": truncated,
    }


def emit_file_write(
    emitter: "EventEmitter | None",
    *,
    agent_id: str,
    parent_id: str | None,
    process_id: str,
    is_error: bool = False,
) -> None:
    """Emit a ``file_write`` event with before + after content.

    Call at ``process_result`` emission. Looks up the matching
    ``capture_file_before`` entry by ``process_id``; if none exists (not a
    write tool, or the before-pass skipped it), no-ops.
    """
    if emitter is None:
        return
    snap = emitter._pending_before.pop(process_id, None)
    if snap is None:
        return
    file_path = snap["file_path"]
    # Re-read post-state. On errored tool calls the after may equal before;
    # the UI renders "no change" in that case, which is informative.
    after, after_truncated, _ = _read_text_capped(Path(file_path))
    emitter.emit(
        "file_write",
        agent_id=agent_id,
        parent_id=parent_id,
        process_id=process_id,
        tool_name=snap["tool_name"],
        file_path=file_path,
        before=snap["before"],
        before_truncated=snap["before_truncated"],
        after=after,
        after_truncated=after_truncated,
        is_error=is_error,
    )


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

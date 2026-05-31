"""Structured event emission for external observers (e.g. agent-terminal viewer)."""

from .emitter import EventEmitter, emit_artifact_created, get_emitter, maybe_emit_paper_read

__all__ = [
    "EventEmitter",
    "emit_artifact_created",
    "get_emitter",
    "maybe_emit_paper_read",
]

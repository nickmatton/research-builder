"""Local web UI for the research-builder harness.

A single-workspace, single-user FastAPI app served on 127.0.0.1. Reads
``canonical_spec/`` + ``phases/`` + ``paper/`` + tails ``logs/events.jsonl``;
writes operator commands to ``logs/commands.jsonl`` (Phase 3+).

Lazy import boundary: nothing in this package is loaded by the rest of
the harness — uvicorn/fastapi imports only fire when
``research-builder-app`` runs.
"""

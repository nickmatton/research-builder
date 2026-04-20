"""Unit tests for the JSONL tail loop."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from research_builder.viewer.sources.event_stream import tail_jsonl


@pytest.mark.asyncio
async def test_tail_yields_existing_then_appended_lines(tmp_path: Path):
    log = tmp_path / "events.jsonl"
    log.write_text(json.dumps({"type": "agent_started", "agent_id": "a", "parent_id": None}) + "\n")

    stop = asyncio.Event()
    received: list[dict] = []

    async def consume():
        async for evt in tail_jsonl(log, from_start=True, poll_interval=0.02, stop_event=stop):
            received.append(evt)
            if len(received) >= 3:
                stop.set()
                return

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0.1)

    with log.open("a") as f:
        f.write(json.dumps({"type": "agent_thinking", "agent_id": "a", "parent_id": None, "text": "hi"}) + "\n")
        f.write(json.dumps({"type": "agent_completed", "agent_id": "a", "parent_id": None, "status": "completed"}) + "\n")

    await asyncio.wait_for(consumer, timeout=2.0)
    assert [e["type"] for e in received] == ["agent_started", "agent_thinking", "agent_completed"]


@pytest.mark.asyncio
async def test_tail_skips_invalid_lines(tmp_path: Path):
    log = tmp_path / "events.jsonl"
    log.write_text("not-json\n" + json.dumps({"type": "agent_started", "agent_id": "a", "parent_id": None}) + "\n")

    stop = asyncio.Event()
    received: list[dict] = []

    async def consume():
        async for evt in tail_jsonl(log, from_start=True, poll_interval=0.02, stop_event=stop):
            received.append(evt)
            stop.set()
            return

    await asyncio.wait_for(consume(), timeout=2.0)
    assert received and received[0]["type"] == "agent_started"

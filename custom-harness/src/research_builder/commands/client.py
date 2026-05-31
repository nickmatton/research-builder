"""Tiny client for appending intervention commands to commands.jsonl.

This is the only writer-side helper for the inbound command channel. Any
external tool (or the inline/browse viewer) appends a JSON line via
``append_command`` and the running pipeline's ``CommandListener`` picks it
up at its 100ms poll cadence.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


def make_command(cmd_type: str, payload: dict[str, Any], *, issuer: str = "operator") -> dict:
    """Construct a command envelope with a fresh ``cmd_id`` and ``ts``."""
    return {
        "cmd_id": uuid.uuid4().hex,
        "ts": datetime.now().isoformat(timespec="milliseconds"),
        "issuer": issuer,
        "type": cmd_type,
        "payload": payload,
    }


def append_command(commands_path: Path | str, cmd: dict) -> None:
    """Atomically append one command line to ``commands.jsonl``.

    Small (<PIPE_BUF) ``O_APPEND`` writes on POSIX are atomic, so this is
    safe to call from multiple writers without coordination.
    """
    path = Path(commands_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(cmd, default=str, ensure_ascii=False) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


# ─── High-level helpers used by the browse-mode UI ────────────────────────

def edit_refined_spec(
    commands_path: Path | str,
    *,
    phase_id: str,
    content: str,
    before_agent: str = "builder",
    mode: str = "replace",
    rationale: str = "",
) -> dict:
    """Append an edit_refined_spec command. Returns the command dict."""
    cmd = make_command("edit_refined_spec", {
        "phase_id": phase_id,
        "before_agent": before_agent,
        "mode": mode,
        "content": content,
        "rationale": rationale,
    })
    append_command(commands_path, cmd)
    return cmd


def force_retry(
    commands_path: Path | str,
    *,
    phase_id: str,
    reset_refined_spec: bool = False,
    reset_research_cache: bool = False,
    rationale: str = "",
) -> dict:
    cmd = make_command("force_retry", {
        "phase_id": phase_id,
        "reset_refined_spec": reset_refined_spec,
        "reset_research_cache": reset_research_cache,
        "rationale": rationale,
    })
    append_command(commands_path, cmd)
    return cmd


def inject_note(
    commands_path: Path | str,
    *,
    text: str,
    scope: str = "phase",
    phase_id: str | None = None,
    target_agents: list[str] | None = None,
    rationale: str = "",
) -> dict:
    payload: dict[str, Any] = {
        "scope": scope,
        "text": text,
        "target_agents": target_agents or ["builder"],
        "rationale": rationale,
    }
    if phase_id is not None:
        payload["phase_id"] = phase_id
    cmd = make_command("inject_note", payload)
    append_command(commands_path, cmd)
    return cmd


def jump_back(
    commands_path: Path | str,
    *,
    to_phase_id: str,
    preserve_artifacts: bool = True,
    rationale: str = "",
) -> dict:
    cmd = make_command("jump_back", {
        "to_phase_id": to_phase_id,
        "preserve_artifacts": preserve_artifacts,
        "rationale": rationale,
    })
    append_command(commands_path, cmd)
    return cmd

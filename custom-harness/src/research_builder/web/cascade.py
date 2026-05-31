"""Compute the cascade preview for a per-phase refined_spec edit.

Given a proposed new ``refined_spec.md`` for phase X and the agent the
edit must take effect before (``before_agent``), figure out:

  - The unified diff of old vs new content.
  - Which (phase, role) pairs need to re-run:
      • The edited phase itself, from ``before_agent`` forward through
        the refiner → researcher → builder → verifier chain.
      • Every downstream phase that transitively depends on the edited
        one (those re-run end-to-end).

The cascade computation reuses the dependency graph stored in
``canonical_spec/state.json`` — same source the harness's
``orchestrator.dependency`` uses, so the preview matches what actually
happens when the edit is applied.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any


# Agent chain ordering. Matches the order each phase's manifest records
# sub-agent steps and the harness's execution loop.
ROLE_CHAIN = ["refiner", "researcher", "builder", "verifier"]


def compute_cascade(
    workspace: Path,
    phase_id: str,
    new_content: str,
    before_agent: str = "builder",
) -> dict[str, Any]:
    """Return ``{phase_id, before_agent, diff, invalidated}``.

    ``invalidated`` is a list of ``{phase_id, roles, reason}`` entries,
    with the edited phase first (reason="direct") then every downstream
    phase (reason="cascade").
    """
    refined_path = workspace / "phases" / phase_id / "context" / "refined_spec.md"
    old_content = refined_path.read_text() if refined_path.exists() else ""

    diff = _make_diff(old_content, new_content)

    state_path = workspace / "canonical_spec" / "state.json"
    state = json.loads(state_path.read_text() or "{}") if state_path.exists() else {}
    deps: dict[str, list[str]] = (state or {}).get("dependency_graph", {}) or {}
    phases_in_state = {
        (p.get("phase_id") or ""): p for p in (state or {}).get("phases", []) or []
    }

    if phase_id not in phases_in_state:
        return {
            "phase_id": phase_id,
            "before_agent": before_agent,
            "diff": diff,
            "invalidated": [],
            "error": f"unknown phase: {phase_id}",
        }

    # Direct: roles from ``before_agent`` onward on the edited phase.
    try:
        start_idx = ROLE_CHAIN.index(before_agent)
    except ValueError:
        start_idx = ROLE_CHAIN.index("builder")
    direct_roles = ROLE_CHAIN[start_idx:]

    # Cascade: every transitive downstream phase.
    downstream = sorted(_get_downstream(deps, phase_id))

    invalidated: list[dict[str, Any]] = [
        {
            "phase_id": phase_id,
            "title": phases_in_state[phase_id].get("title", phase_id),
            "roles": direct_roles,
            "reason": "direct",
        },
    ]
    for ds in downstream:
        invalidated.append({
            "phase_id": ds,
            "title": phases_in_state.get(ds, {}).get("title", ds),
            "roles": list(ROLE_CHAIN),
            "reason": "cascade",
        })

    return {
        "phase_id": phase_id,
        "before_agent": before_agent,
        "diff": diff,
        "invalidated": invalidated,
    }


def _get_downstream(deps: dict[str, list[str]], start: str) -> set[str]:
    """Transitive set of phases that depend on ``start``. Excludes ``start``."""
    out: set[str] = set()
    queue = [start]
    while queue:
        cur = queue.pop()
        for pid, ds in deps.items():
            if cur in ds and pid not in out:
                out.add(pid)
                queue.append(pid)
    return out


def _make_diff(old: str, new: str) -> list[dict[str, Any]]:
    """Word-equal-line-different diff in structured form.

    Returns a flat list of ``{type, text}`` entries where ``type`` is
    one of "context" | "add" | "remove". Replace-tagged blocks become
    paired remove/add runs — easiest for the UI to render as side-by-
    side row pairs.
    """
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    out: list[dict[str, Any]] = []
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line in old_lines[i1:i2]:
                out.append({"type": "context", "text": line})
        elif tag == "delete":
            for line in old_lines[i1:i2]:
                out.append({"type": "remove", "text": line})
        elif tag == "insert":
            for line in new_lines[j1:j2]:
                out.append({"type": "add", "text": line})
        elif tag == "replace":
            for line in old_lines[i1:i2]:
                out.append({"type": "remove", "text": line})
            for line in new_lines[j1:j2]:
                out.append({"type": "add", "text": line})
    return out

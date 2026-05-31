"""Tests for the web UI backend.

Scope of this file:
- cascade.compute_cascade: diff structure + direct/cascade invalidation set.
- /api/spec/apply-edit: writes the two expected commands.jsonl entries.
- /api/workspace + /api/spec basic shape on a minimal workspace.

We don't exercise the WS endpoints here (they need a running event loop +
file-tail timing); the chat WS was smoke-tested live in Phase 2. Adding
async WS tests is the natural next step but is not load-bearing for the
correctness of the new edit/cascade machinery, which is the highest-risk
new code in this surface.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from research_builder.web.app import create_app
from research_builder.web.cascade import compute_cascade


def _make_workspace(tmp_path: Path, state_yaml: str) -> Path:
    import yaml
    spec_dir = tmp_path / "canonical_spec"
    spec_dir.mkdir(parents=True)
    parsed = yaml.safe_load(textwrap.dedent(state_yaml))
    (spec_dir / "state.json").write_text(json.dumps(parsed))
    (tmp_path / "logs").mkdir()
    return tmp_path


# ─── cascade.compute_cascade ─────────────────────────────────────────────


def test_cascade_unknown_phase_carries_error(tmp_path: Path):
    _make_workspace(tmp_path, """
    phases: []
    dependency_graph: {}
    """)
    result = compute_cascade(tmp_path, "nonexistent", "x")
    assert result["error"].startswith("unknown phase")
    # Diff against empty-old/x-new still computes.
    assert result["diff"] == [{"type": "add", "text": "x"}]


def test_cascade_direct_roles_respect_before_agent(tmp_path: Path):
    _make_workspace(tmp_path, """
    phases:
      - phase_id: a
        title: A
    dependency_graph:
      a: []
    """)
    result = compute_cascade(tmp_path, "a", "content", before_agent="researcher")
    direct = result["invalidated"][0]
    assert direct["phase_id"] == "a"
    assert direct["reason"] == "direct"
    assert direct["roles"] == ["researcher", "builder", "verifier"]


def test_cascade_transitive_downstream(tmp_path: Path):
    _make_workspace(tmp_path, """
    phases:
      - phase_id: a
        title: A
      - phase_id: b
        title: B
      - phase_id: c
        title: C
    dependency_graph:
      a: []
      b: [a]
      c: [b]
    """)
    result = compute_cascade(tmp_path, "a", "new", before_agent="builder")
    by_id = {p["phase_id"]: p for p in result["invalidated"]}
    assert by_id["a"]["reason"] == "direct"
    assert by_id["a"]["roles"] == ["builder", "verifier"]
    assert by_id["b"]["reason"] == "cascade"
    assert by_id["c"]["reason"] == "cascade"
    # Cascade phases always re-run the full chain.
    assert by_id["b"]["roles"] == ["refiner", "researcher", "builder", "verifier"]


def test_cascade_diff_structure(tmp_path: Path):
    _make_workspace(tmp_path, """
    phases:
      - phase_id: a
        title: A
    dependency_graph:
      a: []
    """)
    (tmp_path / "phases" / "a" / "context").mkdir(parents=True)
    (tmp_path / "phases" / "a" / "context" / "refined_spec.md").write_text(
        "line 1\nline 2\nline 3\n"
    )
    result = compute_cascade(tmp_path, "a", "line 1\nline TWO\nline 3\n")
    types = [d["type"] for d in result["diff"]]
    assert types == ["context", "remove", "add", "context"]


# ─── /api ────────────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    _make_workspace(tmp_path, """
    phases:
      - phase_id: a
        title: Phase A
      - phase_id: b
        title: Phase B
    dependency_graph:
      a: []
      b: [a]
    """)
    # Two-arg signature (runs_dir, workspace): legacy tests pin a fixed workspace
    # so they don't have to round-trip through /api/launch.
    return TestClient(create_app(tmp_path / "_runs", workspace=tmp_path))


def test_workspace_endpoint(client: TestClient, tmp_path: Path):
    r = client.get("/api/workspace")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == tmp_path.name
    assert body["has_spec"] is True


def test_phases_endpoint(client: TestClient):
    r = client.get("/api/phases")
    assert r.status_code == 200
    body = r.json()
    assert [p["phase_id"] for p in body["phases"]] == ["a", "b"]
    assert body["phases"][1]["dependencies"] == ["a"]


def test_apply_edit_writes_both_commands(client: TestClient, tmp_path: Path):
    r = client.post(
        "/api/spec/apply-edit",
        json={
            "phase_id": "a",
            "content": "new content\n",
            "before_agent": "builder",
            "rationale": "test",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True

    cmd_path = tmp_path / "logs" / "commands.jsonl"
    assert cmd_path.exists()
    lines = [json.loads(l) for l in cmd_path.read_text().strip().splitlines()]
    assert [c["type"] for c in lines] == ["edit_refined_spec", "jump_back"]
    assert lines[0]["payload"]["content"] == "new content\n"
    assert lines[1]["payload"]["preserve_artifacts"] is True


def test_files_endpoint_rejects_escape(client: TestClient):
    r = client.get("/api/files?path=../../etc")
    assert r.status_code == 400


def test_agents_endpoint(client: TestClient):
    r = client.get("/api/agents")
    assert r.status_code == 200
    body = r.json()
    roles = {r["role"] for r in body["roles"]}
    assert roles == {"refiner", "researcher", "builder", "verifier"}

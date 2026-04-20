"""Unit tests for AgentTree event application + navigation."""

from __future__ import annotations

from research_builder.viewer.store import AgentTree


def _evt(type_: str, agent_id: str, parent_id=None, **payload):
    return {"type": type_, "agent_id": agent_id, "parent_id": parent_id, **payload}


def test_orchestrator_then_subagent_builds_tree():
    tree = AgentTree()
    tree.apply_event(_evt(
        "agent_started", "orchestrator",
        kind="orchestrator", title="Orchestrator",
    ))
    tree.apply_event(_evt(
        "agent_started", "phase:data", parent_id="orchestrator",
        kind="subagent", title="Data",
        planned_files=[{"path": "outputs/train.pt", "role": "output"}],
    ))

    assert tree.root_id == "orchestrator"
    assert tree.nodes["orchestrator"].kind == "orchestrator"
    assert "phase:data" in tree.nodes["orchestrator"].children_ids

    data = tree.nodes["phase:data"]
    assert data.parent_id == "orchestrator"
    assert data.status == "running"
    assert len(data.files) == 1 and data.files[0].status == "planned"


def test_file_lifecycle():
    tree = AgentTree()
    tree.apply_event(_evt(
        "agent_started", "phase:data", parent_id="orchestrator",
        kind="subagent", title="Data",
    ))
    tree.apply_event(_evt("file_planned", "phase:data", path="a.txt", role="output"))
    tree.apply_event(_evt("file_created", "phase:data", path="a.txt", role="output"))

    data = tree.nodes["phase:data"]
    assert len(data.files) == 1
    assert data.files[0].status == "created"


def test_thinking_buffers_and_activity_logged():
    tree = AgentTree()
    tree.apply_event(_evt("agent_started", "phase:x", parent_id="orchestrator"))
    tree.apply_event(_evt("agent_thinking", "phase:x", text="loading parquet"))
    tree.apply_event(_evt("agent_tool", "phase:x", summary="bash(pytest -q)"))

    node = tree.nodes["phase:x"]
    assert "loading parquet" in node.reasoning_buffer
    assert any("loading parquet" in a.text for a in node.activity)
    assert any("bash(pytest -q)" in a.text for a in node.activity)


def test_navigation_breadcrumb_and_go_up():
    tree = AgentTree()
    tree.apply_event(_evt("agent_started", "orchestrator", kind="orchestrator", title="Orch"))
    tree.apply_event(_evt("agent_started", "phase:data", parent_id="orchestrator"))
    tree.set_current("phase:data")

    chain = tree.breadcrumb()
    assert [n.id for n in chain] == ["orchestrator", "phase:data"]

    tree.go_up()
    assert tree.current_id == "orchestrator"
    tree.go_up()  # already at root, no-op
    assert tree.current_id == "orchestrator"


def test_agent_completed_marks_node():
    tree = AgentTree()
    tree.apply_event(_evt("agent_started", "phase:y", parent_id="orchestrator"))
    tree.apply_event(_evt("agent_completed", "phase:y", status="completed"))
    assert tree.nodes["phase:y"].status == "completed"


def test_dag_updated_seeds_missing_children():
    tree = AgentTree()
    tree.apply_event(_evt(
        "dag_updated", "orchestrator", parent_id=None,
        nodes=[
            {"id": "phase:data", "title": "Data", "status": "pending",
             "parent_id": "orchestrator"},
            {"id": "phase:train", "title": "Train", "status": "in_progress",
             "parent_id": "orchestrator"},
        ],
        edges=[{"from": "phase:data", "to": "phase:train"}],
    ))
    assert "phase:data" in tree.nodes
    assert tree.nodes["phase:train"].status == "running"

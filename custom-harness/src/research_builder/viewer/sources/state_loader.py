"""Bootstrap an :class:`AgentTree` from research-builder's on-disk state.

Now that the viewer lives in-tree, we use :class:`SpecStore` and the typed
``PhaseStatus`` / ``FileStatus`` enums directly instead of re-parsing YAML/JSON
ourselves. If the on-disk schema ever changes, this loader breaks loudly at
import time rather than silently mis-rendering.
"""

from __future__ import annotations

from pathlib import Path

from ...models.spec import FileStatus, PhaseStatus
from ...storage.spec_store import SpecStore
from ..models import FileEntry
from ..store import AgentTree


# research-builder PhaseStatus → viewer AgentNode status
_STATUS_MAP: dict[PhaseStatus, str] = {
    PhaseStatus.pending: "pending",
    PhaseStatus.in_progress: "running",
    PhaseStatus.completed: "completed",
    PhaseStatus.failed: "failed",
}

# research-builder FileStatus → viewer FileEntry status
_FILE_STATUS_MAP: dict[FileStatus, str] = {
    FileStatus.planned: "planned",
    FileStatus.in_progress: "in_progress",
    FileStatus.written: "created",
    FileStatus.verified: "created",
}


def bootstrap_from_workspace(workspace: Path, tree: AgentTree) -> None:
    """Read the canonical spec dir and seed ``tree``.

    Missing files are tolerated — this is best-effort. The JSONL stream is
    expected to fill in anything we couldn't reconstruct.
    """
    spec_dir = _resolve_spec_dir(Path(workspace))
    store = SpecStore(spec_dir)

    # Always create the orchestrator root.
    tree.ensure_node(
        "orchestrator",
        parent_id=None,
        title="Orchestrator",
        kind="orchestrator",
        status="pending",
    )

    state = None
    try:
        if store.state_path.exists():
            state = store.load_state()
    except Exception:
        state = None

    if state is not None:
        for phase in state.phases:
            agent_id = f"phase:{phase.phase_id}"
            node = tree.ensure_node(
                agent_id,
                parent_id="orchestrator",
                title=phase.title or phase.phase_id,
                kind="subagent",
            )
            node.status = _STATUS_MAP.get(phase.status, "pending")  # type: ignore[assignment]

    plan = None
    try:
        plan = store.load_plan()
    except Exception:
        plan = None

    if plan is not None:
        for n in plan.nodes:
            agent_id = f"phase:{n.phase_id}"
            node = tree.ensure_node(agent_id, parent_id="orchestrator")
            if not node.title:
                node.title = n.title or n.phase_id
        for f in plan.files:
            agent_id = f"phase:{f.owning_phase}"
            node = tree.ensure_node(agent_id, parent_id="orchestrator")
            if any(fe.path == f.rel_path for fe in node.files):
                continue
            node.files.append(FileEntry(
                path=f.rel_path,
                file_id=f.file_id,
                role=f.role.value,
                status=_FILE_STATUS_MAP.get(f.status, "planned"),  # type: ignore[arg-type]
            ))

    if tree.current_id is None:
        tree.current_id = "orchestrator"


def _resolve_spec_dir(workspace: Path) -> Path:
    """research-builder writes spec files under <workspace>/canonical_spec/."""
    candidates = [workspace / "canonical_spec", workspace]
    for c in candidates:
        if (c / "state.yaml").exists() or (c / "dag.json").exists():
            return c
    return workspace / "canonical_spec"

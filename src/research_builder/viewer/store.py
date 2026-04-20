"""In-memory store: a tree of AgentNodes mutated by structured events.

The store is the single source of truth for the UI. The Textual app reads
from it; the event-stream task writes to it via :meth:`AgentTree.apply_event`.
"""

from __future__ import annotations

from typing import Any

from .models import (
    ActivityLine,
    AgentNode,
    ChatMessage,
    ComputeEntry,
    FileEntry,
    ProcessEntry,
)

# Cap how much per-agent state we keep in memory. The viewer is for live
# observation, not forensic archaeology — anything past these limits gets
# trimmed from the front so the most recent activity stays visible.
MAX_ACTIVITY_LINES = 500
MAX_CHAT_MESSAGES = 200
MAX_REASONING_CHARS = 4000
MAX_PROCESSES = 200


class AgentTree:
    """Mutable tree of agents keyed by id."""

    def __init__(self) -> None:
        self.nodes: dict[str, AgentNode] = {}
        self.root_id: str | None = None
        self.current_id: str | None = None
        self.compute_resources: list[ComputeEntry] = []

    # ----- bootstrap helpers -------------------------------------------------

    def ensure_node(
        self,
        agent_id: str,
        parent_id: str | None = None,
        **defaults: Any,
    ) -> AgentNode:
        node = self.nodes.get(agent_id)
        if node is None:
            node = AgentNode(id=agent_id, parent_id=parent_id, **defaults)
            self.nodes[agent_id] = node
            if parent_id and parent_id in self.nodes:
                parent = self.nodes[parent_id]
                if agent_id not in parent.children_ids:
                    parent.children_ids.append(agent_id)
            if parent_id is None and self.root_id is None:
                self.root_id = agent_id
            if self.current_id is None:
                self.current_id = self.root_id or agent_id
        else:
            # Late-arriving parent_id: link up if we now know it.
            if parent_id and node.parent_id is None:
                node.parent_id = parent_id
                if parent_id in self.nodes:
                    parent = self.nodes[parent_id]
                    if agent_id not in parent.children_ids:
                        parent.children_ids.append(agent_id)
        return node

    # ----- navigation --------------------------------------------------------

    def set_current(self, agent_id: str | None) -> None:
        if agent_id and agent_id in self.nodes:
            self.current_id = agent_id

    def go_up(self) -> None:
        if not self.current_id:
            return
        node = self.nodes.get(self.current_id)
        if node and node.parent_id:
            self.current_id = node.parent_id

    def breadcrumb(self) -> list[AgentNode]:
        chain: list[AgentNode] = []
        cur_id = self.current_id
        while cur_id:
            node = self.nodes.get(cur_id)
            if node is None:
                break
            chain.append(node)
            cur_id = node.parent_id
        return list(reversed(chain))

    def current(self) -> AgentNode | None:
        if self.current_id is None:
            return None
        return self.nodes.get(self.current_id)

    # ----- event application -------------------------------------------------

    def apply_event(self, event: dict[str, Any]) -> None:
        """Mutate state in response to a single JSONL event record."""
        evt_type = event.get("type")
        agent_id = event.get("agent_id")
        if not evt_type or not agent_id:
            return
        parent_id = event.get("parent_id")

        node = self.ensure_node(agent_id, parent_id=parent_id)

        handler = _HANDLERS.get(evt_type)
        if handler is None:
            return
        handler(self, node, event)

    # ----- internal mutators (used by handlers) ------------------------------

    def _add_activity(self, node: AgentNode, kind: str, text: str) -> None:
        node.activity.append(ActivityLine(kind=kind, text=text))  # type: ignore[arg-type]
        if len(node.activity) > MAX_ACTIVITY_LINES:
            del node.activity[: len(node.activity) - MAX_ACTIVITY_LINES]

    def _append_chat(self, node: AgentNode, role: str, text: str) -> None:
        node.chat.append(ChatMessage(role=role, text=text))
        if len(node.chat) > MAX_CHAT_MESSAGES:
            del node.chat[: len(node.chat) - MAX_CHAT_MESSAGES]


# ---------- event handlers ---------------------------------------------------

def _h_agent_started(tree: AgentTree, node: AgentNode, e: dict[str, Any]) -> None:
    node.title = e.get("title") or node.title or node.id
    kind = e.get("kind")
    if kind in ("orchestrator", "subagent"):
        node.kind = kind  # type: ignore[assignment]
    node.status = "running"
    node.attempt = e.get("attempt", node.attempt)
    for pf in e.get("planned_files") or []:
        if not any(f.path == pf.get("path") for f in node.files):
            node.files.append(FileEntry(
                path=pf.get("path", ""),
                role=pf.get("role"),
                file_id=pf.get("file_id"),
                status="planned",
            ))


def _h_agent_thinking(tree: AgentTree, node: AgentNode, e: dict[str, Any]) -> None:
    text = e.get("text", "")
    if not text:
        return
    node.reasoning_buffer = (node.reasoning_buffer + "\n" + text)[-MAX_REASONING_CHARS:]
    tree._add_activity(node, "thinking", text[:200])
    # Accumulate consecutive thinking fragments into a single chat message
    # so streaming deltas don't flood the pane with tiny messages.
    if node.chat and node.chat[-1].role == "thinking":
        last = node.chat[-1]
        last.text = (last.text + " " + text)[-MAX_REASONING_CHARS:]
    else:
        tree._append_chat(node, "thinking", text)


def _h_agent_tool(tree: AgentTree, node: AgentNode, e: dict[str, Any]) -> None:
    summary = e.get("summary") or e.get("tool") or ""
    if summary:
        tree._add_activity(node, "tool", str(summary))
        tree._append_chat(node, "tool", str(summary))


def _h_agent_message(tree: AgentTree, node: AgentNode, e: dict[str, Any]) -> None:
    role = e.get("role", "assistant")
    text = e.get("text", "")
    if not text:
        return
    node.chat.append(ChatMessage(role=role, text=text))
    if len(node.chat) > MAX_CHAT_MESSAGES:
        del node.chat[: len(node.chat) - MAX_CHAT_MESSAGES]


def _h_file_planned(tree: AgentTree, node: AgentNode, e: dict[str, Any]) -> None:
    path = e.get("path", "")
    if not path:
        return
    for f in node.files:
        if f.path == path:
            return
    node.files.append(FileEntry(
        path=path,
        role=e.get("role"),
        file_id=e.get("file_id"),
        status="planned",
    ))


def _h_file_created(tree: AgentTree, node: AgentNode, e: dict[str, Any]) -> None:
    path = e.get("path", "")
    if not path:
        return
    for f in node.files:
        if f.path == path:
            f.status = "created"
            return
    node.files.append(FileEntry(
        path=path,
        role=e.get("role"),
        file_id=e.get("file_id"),
        status="created",
    ))


def _h_agent_completed(tree: AgentTree, node: AgentNode, e: dict[str, Any]) -> None:
    status = e.get("status", "completed")
    node.status = "failed" if status == "failed" else "completed"
    summary = e.get("summary")
    if summary:
        tree._add_activity(node, "done", str(summary))
        tree._append_chat(node, "system", f"✓ {summary}")


def _h_dag_updated(tree: AgentTree, node: AgentNode, e: dict[str, Any]) -> None:
    """Snapshot of the orchestrator-level DAG. Ensures every node exists."""
    for n in e.get("nodes", []):
        nid = n.get("id")
        if not nid:
            continue
        child = tree.ensure_node(nid, parent_id=n.get("parent_id") or node.id)
        child.title = n.get("title") or child.title or nid
        status = n.get("status")
        if status:
            child.status = _PHASE_TO_AGENT_STATUS.get(status, child.status)


def _h_process_started(tree: AgentTree, node: AgentNode, e: dict[str, Any]) -> None:
    process_id = e.get("process_id", "")
    if not process_id:
        return
    node.processes.append(ProcessEntry(
        process_id=process_id,
        agent_id=node.id,
        tool_name=e.get("tool_name", ""),
        summary=e.get("summary", ""),
        command=e.get("command"),
        file_path=e.get("file_path"),
        compute_ip=e.get("compute_ip"),
        status="running",
    ))
    if len(node.processes) > MAX_PROCESSES:
        del node.processes[: len(node.processes) - MAX_PROCESSES]


def _h_process_result(tree: AgentTree, node: AgentNode, e: dict[str, Any]) -> None:
    process_id = e.get("process_id", "")
    if not process_id:
        return
    for proc in reversed(node.processes):
        if proc.process_id == process_id:
            proc.output = e.get("output")
            proc.is_error = bool(e.get("is_error", False))
            proc.status = "errored" if proc.is_error else "completed"
            return


def _h_compute_provisioned(tree: AgentTree, node: AgentNode, e: dict[str, Any]) -> None:
    instance_id = e.get("instance_id", "")
    if not instance_id:
        return
    tree.compute_resources.append(ComputeEntry(
        instance_id=instance_id,
        instance_type=e.get("instance_type", ""),
        public_ip=e.get("public_ip", ""),
        agent_id=node.id,
    ))


def _h_compute_terminated(tree: AgentTree, node: AgentNode, e: dict[str, Any]) -> None:
    instance_id = e.get("instance_id", "")
    if not instance_id:
        return
    for cr in tree.compute_resources:
        if cr.instance_id == instance_id:
            cr.status = "terminated"
            return


_PHASE_TO_AGENT_STATUS = {
    "pending": "pending",
    "in_progress": "running",
    "completed": "completed",
    "failed": "failed",
}


_HANDLERS = {
    "agent_started": _h_agent_started,
    "agent_thinking": _h_agent_thinking,
    "agent_tool": _h_agent_tool,
    "agent_message": _h_agent_message,
    "file_planned": _h_file_planned,
    "file_created": _h_file_created,
    "agent_completed": _h_agent_completed,
    "dag_updated": _h_dag_updated,
    "process_started": _h_process_started,
    "process_result": _h_process_result,
    "compute_provisioned": _h_compute_provisioned,
    "compute_terminated": _h_compute_terminated,
}

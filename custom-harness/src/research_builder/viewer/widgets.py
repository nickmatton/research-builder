"""Textual widgets used by AgentTerminalApp.

Kept in a single module to minimize wiring overhead. Each widget exposes a
``render_for(node)`` method (or equivalent) that the app calls whenever the
store changes or the user navigates to a different agent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import DataTable, RichLog, Static, Tree
from textual.widgets.tree import TreeNode

if TYPE_CHECKING:
    from .models import AgentNode, ProcessEntry
    from .store import AgentTree


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


class HeaderBar(Static):
    """Title + breadcrumb path + status."""

    DEFAULT_CSS = """
    HeaderBar {
        height: 1;
        background: $accent;
        color: $text;
        padding: 0 1;
    }
    """

    def render_for(self, tree: "AgentTree") -> None:
        chain = tree.breadcrumb()
        if not chain:
            self.update("agent-terminal")
            return
        crumbs = " › ".join(_short_id(n.id) for n in chain)
        current = chain[-1]
        status = current.status
        text = Text()
        text.append("agent-terminal  ", style="bold")
        text.append(crumbs, style="bold white")
        text.append("    status: ", style="dim")
        text.append(status, style=_status_style(status))
        self.update(text)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class ChatPane(RichLog):
    DEFAULT_CSS = """
    ChatPane {
        border: round $primary;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(highlight=False, markup=False, wrap=True, **kwargs)
        self._last_signature: tuple | None = None

    def render_for(self, node: "AgentNode | None", tree: "AgentTree | None" = None) -> None:
        # Build a merged transcript: this node's messages plus every descendant's,
        # ordered by timestamp, indented by depth so sub-agent activity nests
        # underneath the parent that spawned it (Claude Code's Task pattern).
        merged: list[tuple] = []  # (ts, depth, short_id, msg)
        if node is not None:
            stack: list[tuple["AgentNode", int]] = [(node, 0)]
            while stack:
                cur, depth = stack.pop()
                short = _short_id(cur.id)
                for m in cur.chat:
                    merged.append((m.ts, depth, short, m))
                if tree is not None:
                    for cid in cur.children_ids:
                        child = tree.nodes.get(cid)
                        if child is not None:
                            stack.append((child, depth + 1))
        merged.sort(key=lambda x: x[0])

        sig = (node.id if node else "", len(merged), merged[-1][0] if merged else None)
        if sig == self._last_signature:
            return
        self._last_signature = sig
        self.clear()
        if node is None:
            return
        for ts, depth, short, msg in merged[-300:]:
            indent = "  " * depth
            t = Text()
            if depth > 0:
                t.append(f"{indent}└ ", style="dim")
                t.append(f"{short} ", style="dim magenta")
            role = msg.role
            if role == "thinking":
                t.append("✻ thinking  ", style="italic magenta")
                t.append(msg.text, style="italic dim")
            elif role == "tool":
                t.append("● tool  ", style="bold cyan")
                t.append(msg.text, style="cyan")
            elif role == "system":
                t.append(msg.text, style="dim italic")
            elif role == "user":
                t.append("you  ", style="bold green")
                t.append(msg.text)
            elif role == "assistant":
                t.append("agent  ", style="bold white")
                t.append(msg.text)
            elif role.startswith("subagent-"):
                t.append(f"{role}  ", style="bold magenta")
                t.append(msg.text)
            else:
                t.append(f"{role}: ", style="bold white")
                t.append(msg.text)
            self.write(t)


# ---------------------------------------------------------------------------
# DAG (Tree)
# ---------------------------------------------------------------------------


class DagSelected(Message):
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        super().__init__()


class FileSelected(Message):
    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__()


class DagPane(Tree):
    DEFAULT_CSS = """
    DagPane {
        border: round $primary;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("agents", **kwargs)
        self.show_root = False
        self.guide_depth = 2
        self._id_to_node: dict[str, TreeNode] = {}

    def render_for(self, tree: "AgentTree") -> None:
        # Rebuild the entire tree on each render. Cheap for our sizes.
        self.clear()
        self._id_to_node.clear()
        if tree.root_id is None:
            return
        root_agent = tree.nodes.get(tree.root_id)
        if root_agent is None:
            return
        root_tn = self.root.add(_format_node(root_agent, tree.current_id), data=root_agent.id, expand=True)
        self._id_to_node[root_agent.id] = root_tn
        self._add_children(tree, root_agent.id, root_tn)

    def _add_children(self, tree: "AgentTree", agent_id: str, tn: TreeNode) -> None:
        agent = tree.nodes.get(agent_id)
        if agent is None:
            return
        for child_id in agent.children_ids:
            child = tree.nodes.get(child_id)
            if child is None:
                continue
            child_tn = tn.add(_format_node(child, tree.current_id), data=child_id, expand=True)
            self._id_to_node[child_id] = child_tn
            self._add_children(tree, child_id, child_tn)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        agent_id = event.node.data
        if isinstance(agent_id, str):
            self.post_message(DagSelected(agent_id))


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


class FilesPane(Tree):
    """Filesystem-style tree of planned/created files for the focused agent."""

    DEFAULT_CSS = """
    FilesPane {
        border: round $primary;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("files", **kwargs)
        self.show_root = False
        self.guide_depth = 2

    def render_for(self, node: "AgentNode | None", tree: "AgentTree | None" = None) -> None:
        self.clear()
        if node is None:
            return
        files = _collect_files(tree, node)
        if not files:
            self.root.add_leaf(Text("(no files)", style="dim"))
            return

        # Build a nested dict: {dirname: {...subdirs..., "__files__": [FileEntry, ...]}}
        root: dict = {}
        for f in files:
            parts = f.path.split("/")
            dirs, name = parts[:-1], parts[-1]
            cur = root
            for d in dirs:
                cur = cur.setdefault(d, {})
            cur.setdefault("__files__", []).append((name, f))

        self._add_dir(self.root, root)
        self.root.expand_all()

    def _add_dir(self, parent_tn: TreeNode, tree_dict: dict) -> None:
        # Subdirectories first (sorted), then files (sorted by name).
        dirs = sorted(k for k in tree_dict.keys() if k != "__files__")
        for d in dirs:
            label = Text()
            label.append("📁 ", style="bold blue")
            label.append(d, style="bold blue")
            child_tn = parent_tn.add(label, expand=True)
            self._add_dir(child_tn, tree_dict[d])
        for name, f in sorted(tree_dict.get("__files__", []), key=lambda x: x[0]):
            parent_tn.add_leaf(_format_file_leaf(name, f), data=f.path)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        path = event.node.data
        if isinstance(path, str) and path:
            self.post_message(FileSelected(path))


def _format_file_leaf(name: str, f) -> Text:
    if f.status == "created":
        icon, style = "✓", "green"
    elif f.status == "in_progress":
        icon, style = "⌛", "yellow"
    else:
        icon, style = "·", "dim"
    text = Text()
    text.append(f"{icon} ", style=style)
    text.append(name, style=style)
    if f.role:
        text.append(f"  ({f.role})", style="dim")
    return text


def _collect_files(tree: "AgentTree | None", node: "AgentNode") -> list:
    """Gather files for FilesPane.

    - At the orchestrator root: union of files from every node in the tree.
    - At any other node: union of that node's files plus all descendants'.
    Deduped on (file_id or path) and sorted so created files float above
    in_progress, then planned, then by path within each bucket.
    """
    seen: dict[str, "FileEntry"] = {}  # noqa: F821

    def add_node(n: "AgentNode") -> None:
        for f in n.files:
            key = f.file_id or f.path
            existing = seen.get(key)
            # Prefer the entry with the more "advanced" status if duplicated.
            if existing is None or _status_rank(f.status) < _status_rank(existing.status):
                seen[key] = f

    if tree is not None and node.id == tree.root_id:
        for n in tree.nodes.values():
            add_node(n)
    else:
        # Walk node + descendants.
        stack = [node]
        while stack:
            cur = stack.pop()
            add_node(cur)
            if tree is not None:
                for cid in cur.children_ids:
                    child = tree.nodes.get(cid)
                    if child is not None:
                        stack.append(child)

    return sorted(seen.values(), key=lambda f: (_status_rank(f.status), f.path))


def _status_rank(status: str) -> int:
    return {"created": 0, "in_progress": 1, "planned": 2}.get(status, 3)


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------


class OutputPane(RichLog):
    """Live tail of pipeline.out — surfaces subprocess stdout/stderr in the TUI.

    Without this, a crash in the pipeline subprocess (e.g. a missing
    dependency) is completely invisible; the TUI just shows "pending" forever.
    """

    DEFAULT_CSS = """
    OutputPane {
        border: round $primary;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(highlight=False, markup=False, wrap=True, **kwargs)
        self._pos: int = 0
        self._path: str | None = None

    def set_path(self, path: str) -> None:
        self._path = path
        self._pos = 0

    def poll(self) -> None:
        """Read any new bytes from the log file and append to the widget."""
        if self._path is None:
            return
        try:
            import os
            st = os.stat(self._path)
            if st.st_size <= self._pos:
                return
            with open(self._path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._pos)
                chunk = f.read()
                self._pos = f.tell()
            for line in chunk.splitlines():
                t = Text(line)
                # Highlight errors so they jump out visually
                lower = line.lower()
                if "error" in lower or "traceback" in lower or "exception" in lower:
                    t = Text(line, style="bold red")
                elif "warning" in lower:
                    t = Text(line, style="yellow")
                self.write(t)
        except FileNotFoundError:
            pass
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------


class ProcessesPane(Vertical):
    """Master-detail view of tool calls / processes run by agents."""

    DEFAULT_CSS = """
    ProcessesPane {
        border: round $primary;
        padding: 0 1;
    }
    ProcessesPane #proc-list {
        height: 2fr;
    }
    ProcessesPane #proc-detail {
        height: 1fr;
        border-top: solid $primary;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._last_signature: tuple | None = None
        self._selected_process_id: str | None = None
        self._processes: list["ProcessEntry"] = []

    def compose(self):
        self._table = DataTable(id="proc-list")
        self._table.cursor_type = "row"
        self._table.add_columns("", "Tool", "Summary", "Agent")
        self._detail = RichLog(id="proc-detail", highlight=False, markup=False, wrap=True)
        yield self._table
        yield self._detail

    def render_for(self, node: "AgentNode | None", tree: "AgentTree | None" = None) -> None:
        processes = _collect_processes(tree, node)

        sig = (
            node.id if node else "",
            len(processes),
            processes[-1].process_id if processes else None,
            # Also detect result updates on selected process
            self._selected_process_id,
            next((p.status for p in processes if p.process_id == self._selected_process_id), None),
        )
        if sig == self._last_signature:
            return
        self._last_signature = sig
        self._processes = processes

        # Rebuild table
        self._table.clear()
        for p in processes[-100:]:
            icon = {"running": "⏳", "completed": "✓", "errored": "✗"}.get(p.status, "?")
            style = {"running": "yellow", "completed": "green", "errored": "red"}.get(p.status, "white")
            summary = p.summary if len(p.summary) <= 60 else p.summary[:57] + "..."
            self._table.add_row(
                Text(icon, style=style),
                Text(p.tool_name, style="cyan"),
                summary,
                Text(_short_id(p.agent_id), style="dim"),
                key=p.process_id,
            )

        # Show compute resources when no process is selected
        if tree is not None and not self._selected_process_id:
            active = [c for c in tree.compute_resources if c.status == "active"]
            if active:
                self._detail.clear()
                for c in active:
                    self._detail.write(
                        Text(f"☁ {c.instance_type} @ {c.public_ip}  ({_short_id(c.agent_id)})", style="bold cyan")
                    )

        # Re-render detail for selected process
        if self._selected_process_id:
            self._render_detail(self._selected_process_id)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        key = event.row_key
        self._selected_process_id = key.value if key else None
        if self._selected_process_id:
            self._render_detail(self._selected_process_id)

    def _render_detail(self, process_id: str) -> None:
        self._detail.clear()
        proc = next((p for p in self._processes if p.process_id == process_id), None)
        if proc is None:
            self._detail.write(Text("(process not found)", style="dim"))
            return

        status_style = {"running": "yellow", "completed": "green", "errored": "red"}.get(proc.status, "white")
        self._detail.write(Text(f"{proc.tool_name}  [{proc.status}]", style=f"bold {status_style}"))
        self._detail.write(Text(proc.summary))
        if proc.command:
            self._detail.write(Text(f"$ {proc.command}", style="cyan"))
        if proc.file_path:
            self._detail.write(Text(f"File: {proc.file_path}", style="blue"))
        if proc.compute_ip:
            self._detail.write(Text(f"Cloud: {proc.compute_ip}", style="magenta"))
        if proc.output:
            self._detail.write(Text("--- output ---", style="dim"))
            style = "red" if proc.is_error else ""
            for line in proc.output[:2000].splitlines():
                self._detail.write(Text(line, style=style))
        elif proc.status == "running":
            self._detail.write(Text("(waiting for output...)", style="dim italic"))


def _collect_processes(tree: "AgentTree | None", node: "AgentNode | None") -> list["ProcessEntry"]:
    """Gather processes from a node and its descendants, sorted by timestamp."""
    if node is None:
        return []
    all_procs: list["ProcessEntry"] = []

    if tree is not None and node.id == tree.root_id:
        for n in tree.nodes.values():
            all_procs.extend(n.processes)
    else:
        stack = [node]
        while stack:
            cur = stack.pop()
            all_procs.extend(cur.processes)
            if tree is not None:
                for cid in cur.children_ids:
                    child = tree.nodes.get(cid)
                    if child is not None:
                        stack.append(child)

    return sorted(all_procs, key=lambda p: p.ts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _short_id(agent_id: str) -> str:
    return agent_id.split(":", 1)[1] if ":" in agent_id else agent_id


def _status_style(status: str) -> str:
    return {
        "running": "bold yellow",
        "completed": "bold green",
        "failed": "bold red",
        "pending": "dim",
    }.get(status, "white")


def _format_node(agent: "AgentNode", current_id: str | None) -> Text:
    is_current = agent.id == current_id
    marker = "● " if is_current else "○ "
    label = _short_id(agent.id)
    text = Text()
    text.append(marker, style=_status_style(agent.status))
    text.append(label, style=("bold reverse" if is_current else "bold"))
    text.append(f"  [{agent.status}]", style="dim")
    return text

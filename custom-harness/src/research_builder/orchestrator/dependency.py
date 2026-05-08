"""Dependency graph computation and traversal (spec_v4 §2.3)."""

from __future__ import annotations

from ..models.spec import PhaseStatus, SpecState


class DependencyGraph:
    """Manages phase dependency relationships.

    The graph is stored as a dict mapping phase_id -> list of phase_ids it depends on.
    E.g. {"training": ["data", "arch"]} means training depends on data and arch.
    """

    def __init__(self, graph: dict[str, list[str]]) -> None:
        self.graph = graph

    @classmethod
    def from_spec_state(cls, state: SpecState) -> DependencyGraph:
        return cls(dict(state.dependency_graph))

    def get_dependencies(self, phase_id: str) -> list[str]:
        """Return the phases that must complete before this phase can start."""
        return self.graph.get(phase_id, [])

    def get_downstream(self, phase_id: str) -> set[str]:
        """Return all phases that transitively depend on this phase."""
        downstream: set[str] = set()
        queue = [phase_id]
        while queue:
            current = queue.pop()
            for pid, deps in self.graph.items():
                if current in deps and pid not in downstream:
                    downstream.add(pid)
                    queue.append(pid)
        return downstream

    def get_upstream(self, phase_id: str) -> set[str]:
        """Return all phases this phase transitively depends on."""
        upstream: set[str] = set()
        queue = list(self.get_dependencies(phase_id))
        while queue:
            current = queue.pop()
            if current not in upstream:
                upstream.add(current)
                queue.extend(self.get_dependencies(current))
        return upstream

    def get_runnable(self, state: SpecState) -> list[str]:
        """Return phase_ids that are pending and have all dependencies completed."""
        runnable: list[str] = []
        for phase in state.phases:
            if phase.status != PhaseStatus.pending:
                continue
            deps = self.get_dependencies(phase.phase_id)
            all_deps_done = all(
                state.get_phase(d) is not None and state.get_phase(d).status == PhaseStatus.completed
                for d in deps
            )
            if all_deps_done:
                runnable.append(phase.phase_id)
        return runnable

    def validate(self, phase_ids: set[str]) -> list[str]:
        """Check for issues: unknown deps, cycles. Returns list of error messages."""
        errors: list[str] = []

        # Check for unknown dependencies
        for pid, deps in self.graph.items():
            if pid not in phase_ids:
                errors.append(f"Phase '{pid}' in dependency graph but not in phase list")
            for dep in deps:
                if dep not in phase_ids:
                    errors.append(f"Phase '{pid}' depends on unknown phase '{dep}'")

        # Check for cycles via DFS
        visited: set[str] = set()
        in_stack: set[str] = set()

        def has_cycle(node: str) -> bool:
            if node in in_stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            in_stack.add(node)
            for dep in self.graph.get(node, []):
                if has_cycle(dep):
                    return True
            in_stack.discard(node)
            return False

        for pid in self.graph:
            if pid not in visited:
                if has_cycle(pid):
                    errors.append(f"Cycle detected involving phase '{pid}'")

        return errors

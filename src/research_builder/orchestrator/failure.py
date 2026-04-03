"""Failure handling and retry budget tracking (spec_v4 §4.3)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..models.results import SubAgentResult

logger = logging.getLogger(__name__)


@dataclass
class FailureHandler:
    """Tracks orchestrator retry budgets per phase.

    Two separate budgets exist (§4.3):
    - Sub-agent debug attempts: internal to each sub-agent invocation (not tracked here)
    - Orchestrator retries: tracked here — how many fresh sub-agent invocations per phase

    Spec-issue returns do NOT count against the retry budget.
    """

    max_retries: int = 3
    _retry_counts: dict[str, int] = field(default_factory=dict)
    _results: dict[str, list[SubAgentResult]] = field(default_factory=dict)

    def record_result(self, phase_id: str, result: SubAgentResult) -> None:
        """Record a sub-agent result for a phase."""
        if phase_id not in self._results:
            self._results[phase_id] = []
        self._results[phase_id].append(result)

        # Only count non-spec-issue failures against retry budget
        if result.status.value == "failure" and not result.is_spec_issue:
            self._retry_counts[phase_id] = self._retry_counts.get(phase_id, 0) + 1

    def can_retry(self, phase_id: str) -> bool:
        """Check if the orchestrator can retry this phase."""
        return self._retry_counts.get(phase_id, 0) < self.max_retries

    def retries_used(self, phase_id: str) -> int:
        """Return how many retries have been used (excluding spec-issue returns)."""
        return self._retry_counts.get(phase_id, 0)

    def get_prior_results(self, phase_id: str) -> list[SubAgentResult]:
        """Return all prior results for a phase (for retry context)."""
        return self._results.get(phase_id, [])

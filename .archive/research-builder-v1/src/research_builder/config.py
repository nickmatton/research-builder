"""Run configuration for the research builder harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Top-level configuration for a reproduction run."""

    # Paper
    paper_path: Path = Path("paper/paper.pdf")

    # Project root (all outputs written under here)
    project_root: Path = Path(".")

    # Model
    model: str = "claude-opus-4-6"

    # Orchestrator budgets
    max_retries: int = 3

    # Sub-agent budgets
    max_debug_attempts: int = 10

    # Timeout for sub-agent bash commands (seconds)
    bash_timeout: int = 300

    # Phases to run (None = all standard phases)
    phases: list[str] | None = None

    # Interactive mode (human-in-the-loop checkpoints)
    interactive: bool = True

    # Per-run hard cap on Lambda Cloud GPU spend (USD). Provisioning that would
    # exceed this triggers an operator approval prompt (asks to raise the cap).
    gpu_budget_usd: float = 30.0

    @property
    def paper_dir(self) -> Path:
        return self.project_root / "paper"

    @property
    def spec_dir(self) -> Path:
        return self.project_root / "canonical_spec"

    @property
    def phases_dir(self) -> Path:
        return self.project_root / "phases"

    @property
    def report_dir(self) -> Path:
        return self.project_root / "report"

    @property
    def logs_dir(self) -> Path:
        """Human-readable run artifacts: post-mortems, spec-amendment records."""
        return self.project_root / "logs"

    @property
    def context_dir(self) -> Path:
        """Per-attempt snapshots of what each sub-agent actually saw (system_prompt, sub_spec, kickoff)."""
        return self.project_root / "context"

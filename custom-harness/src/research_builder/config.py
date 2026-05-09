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

    # ---- paper-repo–shape paths (Phase 2 of the dual-interface refactor) ----
    # These are the user-facing artifacts that match what the Claude Code
    # skill workflow produces. The internal harness still writes its own
    # canonical_spec/+phases/ format above; the paper-repo projection writes
    # to these alongside, so either interface can read either.

    @property
    def notes_dir(self) -> Path:
        return self.project_root / "notes"

    @property
    def runs_dir(self) -> Path:
        return self.project_root / "runs"

    @property
    def claude_md_path(self) -> Path:
        """The user-facing reproduction spec — projection of canonical_spec/spec.md."""
        return self.project_root / "CLAUDE.md"

    @property
    def journal_path(self) -> Path:
        """Append-only run log — projection of canonical_spec/revision_log.yaml events."""
        return self.notes_dir / "journal.md"

    @property
    def claims_yaml_path(self) -> Path:
        """User-facing claims ledger — copy of canonical_spec/claims.yaml in skill-workflow shape."""
        return self.notes_dir / "claims.yaml"

    @property
    def post_mortems_dir(self) -> Path:
        """User-facing post-mortems — projection of logs/postmortems/<phase>/retry_N.md."""
        return self.notes_dir / "post-mortems"

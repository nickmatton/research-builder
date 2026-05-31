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

    # Model — defaulting to Opus 4.6 with the 1M context window. Running
    # through Claude Code charges these calls against the subscription, so
    # the cost case for falling back to Haiku doesn't apply here. Override
    # per-run with --model.
    #
    # NOTE: not 4.7 — Opus 4.7 returns "API Error: 400 role 'system' is not
    # supported on this model" via the bundled claude-agent-sdk, which sends
    # the system prompt as a message role rather than the top-level system
    # parameter. 4.6 still tolerates the legacy format. Revisit when the SDK
    # is upgraded.
    model: str = "claude-opus-4-6[1m]"

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

    # Long-phase approval gate: if the plan refiner estimates a phase will
    # take longer than this (wall-clock minutes), and ``interactive`` is True,
    # prompt the operator before dispatching the builder. In non-interactive
    # mode the harness logs a warning and proceeds. Set to 0 to disable.
    long_phase_threshold_minutes: int = 30

    # Per-run hard cap on Lambda Cloud GPU spend (USD). Provisioning that would
    # exceed this triggers an operator approval prompt (asks to raise the cap).
    gpu_budget_usd: float = 30.0

    # Per-run hard cap on LLM spend (USD) — covers BOTH sub-agent and
    # orchestrator query costs. When total spend crosses this, the run aborts
    # cleanly (remaining phases marked failed). Set to 0 to disable. A single
    # runaway section can easily burn $20+ (we have measured this when an
    # agent polls a long-running training job in a loop).
    llm_spend_cap_usd: float = 20.0

    # Extra directories the agent sandbox may read/write outside of cwd. Passed
    # through to ClaudeAgentOptions.add_dirs for both orchestrator and sub-agent.
    # Populate via --allow-dir on the CLI / web app. Each entry should be an
    # absolute path; we don't resolve here (callers do at flag-parse time).
    # The agent can also call mcp__access__read_outside_workspace to read paths
    # NOT in this list, which surfaces an approval prompt in interactive mode.
    extra_allowed_dirs: list[Path] = field(default_factory=list)

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
        """Append-only run log — projection of canonical_spec/revision_log.json events."""
        return self.notes_dir / "journal.md"

    @property
    def claims_yaml_path(self) -> Path:
        """User-facing claims ledger — projection of canonical_spec/claims.json in skill-workflow shape (YAML)."""
        return self.notes_dir / "claims.yaml"

    @property
    def post_mortems_dir(self) -> Path:
        """User-facing post-mortems — projection of logs/postmortems/<phase>/retry_N.md."""
        return self.notes_dir / "post-mortems"

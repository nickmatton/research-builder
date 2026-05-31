"""Project harness state into the paper-repo shape (Phase 2 of dual-interface refactor).

The harness internally writes its old canonical_spec/+phases/ layout (so the
existing orchestrator/loop/sub_agent code keeps working). This module mirrors
the user-facing artifacts to the same per-paper-repo paths the Claude Code
skill workflow uses, so EITHER interface can read EITHER set of artifacts:

    Internal (harness)                  →  Paper-repo (user-facing, shared)
    ─────────────────────────────────────────────────────────────────
    canonical_spec/spec.md              →  CLAUDE.md
    canonical_spec/claims.json          →  notes/claims.yaml  (YAML output is intentional —
                                              the skill workflow's compare-claims.py reads YAML)
    canonical_spec/revision_log.json    →  notes/journal.md   (markdown row blocks)
    logs/postmortems/<phase>/retry_N.md →  notes/post-mortems/<phase>-retry-<N>.md

Called at meaningful checkpoints from orchestrator/loop.py:
- After spec creation: write CLAUDE.md + notes/claims.yaml.
- After each phase completes/fails: append a journal row.
- After each post-mortem: copy to notes/post-mortems/.
- At run completion: write the final summary journal row.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Config
from ..models.claims import ClaimsLedger
from ..models.results import SubAgentResult
from ..models.spec import PhaseStatus, SpecState

logger = logging.getLogger(__name__)


def project_spec_to_claude_md(config: Config, spec_md: str, state: SpecState) -> None:
    """Write CLAUDE.md at the project root from the canonical spec_md.

    Light header insertion to identify the source; rest of the markdown is
    passed through verbatim (the LLM-authored spec.md is already in the
    shape the skill workflow expects for CLAUDE.md).
    """
    config.project_root.mkdir(parents=True, exist_ok=True)
    paper_id = state.metadata.paper_id if state.metadata else "(unknown)"
    paper_title = state.metadata.paper_title if state.metadata else "(unknown)"
    header = (
        f"<!-- Authored by research-builder-harness. paper_id={paper_id}. "
        f"Edit freely; the harness re-projects from canonical_spec/spec.md "
        f"on every spec amendment. -->\n\n"
    )
    if not spec_md.lstrip().startswith("#"):
        # Add a title if the spec.md doesn't lead with a heading.
        spec_md = f"# {paper_title}\n\n{spec_md}"
    config.claude_md_path.write_text(header + spec_md)
    logger.debug("Projected CLAUDE.md (%d chars) → %s", len(spec_md), config.claude_md_path)


def project_claims_to_notes(config: Config, ledger: ClaimsLedger) -> None:
    """Write notes/claims.yaml in the shape paper-template/notes/claims.yaml uses
    (top-level `claims:` list, one entry per claim)."""
    import yaml
    config.notes_dir.mkdir(parents=True, exist_ok=True)
    body = (
        "# Claims Ledger — projected from canonical_spec/claims.json by the harness.\n"
        "# The Claude Code skill workflow's compare-claims.py reads this same format.\n\n"
    )
    payload = {"claims": [c.model_dump(mode="json") for c in ledger.claims]}
    config.claims_yaml_path.write_text(body + yaml.dump(payload, default_flow_style=False, sort_keys=False))
    logger.debug("Projected %d claims → %s", len(ledger.claims), config.claims_yaml_path)


def ensure_journal_header(config: Config) -> None:
    """Create notes/journal.md with the documented header if it doesn't exist."""
    if config.journal_path.exists():
        return
    config.notes_dir.mkdir(parents=True, exist_ok=True)
    header = """# Run Journal — harness invocations

Append-only log of every harness run. Most-recent at the bottom.

The custom-harness writes a row per phase-complete and a summary row per run-complete.
The Claude Code skill workflow appends rows here too (via /reproduce). Same format.

## Format

```
## <run-id>  (<ISO 8601 timestamp>)
**Type:** harness | smoke | overfit-one-batch | short-train | full
**Git SHA:** <short sha>
**Phase:** <phase-id>  (per-phase rows; omitted on summary rows)
**Status:** completed | failed
**Duration:** <wall-clock>

**Notes**
<one or two sentences. What did this run prove or fail to prove?>
```

---

## Runs

"""
    config.journal_path.write_text(header)


def append_journal_row(
    config: Config,
    *,
    run_id: str,
    row_type: str,
    status: str,
    phase_id: str | None = None,
    duration_seconds: float | None = None,
    notes: str = "",
    extra_lines: list[str] | None = None,
) -> None:
    """Append a markdown row block to notes/journal.md.

    Format matches what the skill workflow's /reproduce produces.
    """
    ensure_journal_header(config)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    git_sha = _git_sha(config.project_root)

    lines = [f"## {run_id}  ({ts})"]
    lines.append(f"**Type:** {row_type}")
    lines.append(f"**Git SHA:** {git_sha}")
    if phase_id:
        lines.append(f"**Phase:** {phase_id}")
    lines.append(f"**Status:** {status}")
    if duration_seconds is not None:
        if duration_seconds < 60:
            lines.append(f"**Duration:** {duration_seconds:.1f} s")
        else:
            lines.append(f"**Duration:** {duration_seconds / 60:.1f} min")
    if extra_lines:
        lines.append("")
        lines.extend(extra_lines)
    if notes:
        lines.append("")
        lines.append("**Notes**")
        lines.append(notes)
    lines.append("")
    block = "\n".join(lines)

    with config.journal_path.open("a") as f:
        f.write(block + "\n")
    logger.debug("Appended journal row run_id=%s type=%s status=%s", run_id, row_type, status)


def project_post_mortem(
    config: Config,
    *,
    phase_id: str,
    retry_num: int,
    internal_path: Path,
) -> None:
    """Copy a post-mortem from the harness's logs/postmortems/<phase>/retry_N.md
    location to the user-facing notes/post-mortems/<phase>-retry-<N>.md."""
    if not internal_path.exists():
        logger.debug("Post-mortem source not found: %s", internal_path)
        return
    config.post_mortems_dir.mkdir(parents=True, exist_ok=True)
    dest = config.post_mortems_dir / f"{phase_id}-retry-{retry_num}.md"
    shutil.copy2(internal_path, dest)
    logger.debug("Projected post-mortem → %s", dest)


def project_phase_complete(
    config: Config,
    *,
    phase_id: str,
    result: SubAgentResult,
    duration_seconds: float | None = None,
    run_id: str | None = None,
) -> None:
    """One row per phase completion, written to notes/journal.md."""
    rid = run_id or f"phase-{phase_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    extra: list[str] = []
    if result.test_report and result.test_report.tests_run:
        tr = result.test_report
        extra.append(f"**Tests:** {tr.tests_passed}/{tr.tests_run} passed, {tr.tests_failed} failed")
    if result.outputs:
        out_summary = ", ".join(o.name for o in result.outputs[:5])
        if len(result.outputs) > 5:
            out_summary += f", … ({len(result.outputs)} total)"
        extra.append(f"**Outputs:** {out_summary}")
    if result.cost_usd:
        extra.append(f"**Cost:** ${result.cost_usd:.2f}")

    notes = (result.summary or "").strip()
    if len(notes) > 500:
        notes = notes[:497] + "…"

    append_journal_row(
        config,
        run_id=rid,
        row_type="harness",
        phase_id=phase_id,
        status=("completed" if result.status.value == "success" else "failed"),
        duration_seconds=duration_seconds,
        notes=notes,
        extra_lines=extra,
    )


def project_run_complete(
    config: Config,
    *,
    run_id: str,
    state: SpecState,
    total_cost_usd: float,
    duration_seconds: float | None = None,
) -> None:
    """Final summary row — one per harness invocation."""
    completed = sum(1 for p in state.phases if p.status == PhaseStatus.completed)
    failed = sum(1 for p in state.phases if p.status == PhaseStatus.failed)
    total = len(state.phases)
    extra = [f"**Phases:** {completed}/{total} completed, {failed} failed"]
    if total_cost_usd:
        extra.append(f"**Total cost:** ${total_cost_usd:.2f}")

    status = "completed" if failed == 0 and completed == total else "failed"
    notes = (
        f"Harness-driven reproduction of {state.metadata.paper_title if state.metadata else '(unknown paper)'}. "
        f"See canonical_spec/ for full per-phase machine state."
    )

    append_journal_row(
        config,
        run_id=run_id,
        row_type="harness-summary",
        status=status,
        duration_seconds=duration_seconds,
        notes=notes,
        extra_lines=extra,
    )


# ---- helpers ----------------------------------------------------------------


def _git_sha(repo_root: Path) -> str:
    """Best-effort short git SHA. Returns 'unknown' if not in a repo."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"

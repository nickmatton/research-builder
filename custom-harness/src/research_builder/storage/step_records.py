"""Per-attempt agent step records.

For every refiner / researcher / builder / verifier (and post-mortem / refine-spec)
call inside a phase, we persist one JSON file under
``phases/<phase_id>/attempts/<retry_num>/<role>.json`` plus an append-only
``manifest.json`` summarising the ordered sequence of steps.

The browse-mode TUI reads these files to render a phase's history without
having to re-parse ``events.jsonl``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """One agent step inside one phase attempt.

    ``parsed`` holds the structured output the orchestrator extracted from the
    raw ``response_text`` (e.g. the refiner's dict with ``refined_spec_md`` +
    ``research_questions``). ``extra`` is a free-form bag for role-specific
    extras (test report counts for the builder, verifier_payload, etc.).
    """

    role: str
    phase_id: str
    retry_num: int
    started_at: float
    ended_at: float
    duration_s: float
    cost_usd: float | None = None
    model: str | None = None
    prompt_role: str | None = None
    system_prompt: str | None = None
    prompt: str | None = None
    response_text: str | None = None
    parsed: Any = None
    messages_received: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"  # ok | error | timeout

    def to_dict(self) -> dict:
        return asdict(self)


def attempt_dir(phase_work_dir: Path, retry_num: int) -> Path:
    d = phase_work_dir / "attempts" / str(retry_num)
    d.mkdir(parents=True, exist_ok=True)
    return d


def step_record_path(phase_work_dir: Path, retry_num: int, role: str) -> Path:
    return attempt_dir(phase_work_dir, retry_num) / f"{role}.json"


def manifest_path(phase_work_dir: Path, retry_num: int) -> Path:
    return attempt_dir(phase_work_dir, retry_num) / "manifest.json"


def write_step_record(phase_work_dir: Path, record: StepRecord) -> Path:
    """Persist ``record`` and append a summary line to the attempt manifest.

    Returns the path the record was written to.
    """
    out = step_record_path(phase_work_dir, record.retry_num, record.role)
    try:
        out.write_text(json.dumps(record.to_dict(), indent=2, default=str))
    except Exception:
        logger.exception("step_records: failed to write %s", out)
        return out

    _append_to_manifest(phase_work_dir, record)
    return out


def _append_to_manifest(phase_work_dir: Path, record: StepRecord) -> None:
    """Append a one-line entry to the attempt manifest (atomic write)."""
    mpath = manifest_path(phase_work_dir, record.retry_num)
    entry = {
        "role": record.role,
        "started_at": record.started_at,
        "ended_at": record.ended_at,
        "duration_s": record.duration_s,
        "cost_usd": record.cost_usd,
        "status": record.status,
        "record_path": str(step_record_path(phase_work_dir, record.retry_num, record.role).name),
    }
    existing: list[dict] = []
    if mpath.exists():
        try:
            existing = json.loads(mpath.read_text())
            if not isinstance(existing, list):
                existing = []
        except Exception:
            logger.warning("step_records: manifest unreadable at %s, replacing", mpath)
            existing = []
    existing.append(entry)
    tmp = mpath.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(existing, indent=2, default=str))
        os.replace(tmp, mpath)
    except Exception:
        logger.exception("step_records: failed to update manifest %s", mpath)


def now() -> float:
    return time.time()

"""End-of-run error summary.

Scans ``logs/run.log`` for ERROR and WARNING lines, groups them by
(module, function, message-signature), and writes a single markdown
file at ``notes/run_errors.md``. This is a flat run-scoped post-mortem
that complements the per-phase ``logs/postmortems/`` files: those fire
only on phase-level failure; this fires every run (success or fail) so
transient SDK errors, dropped plan files, and other "warnings that
mattered" are visible in one place.

Cheap to compute (a few thousand log lines), so we always run it from
``main.run_pipeline`` regardless of success/failure.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


# Format produced by main.py's file handler:
#   "YYYY-MM-DD HH:MM:SS module.path LEVEL funcName:lineno message"
# Anchor at the timestamp to avoid matching the level word inside messages.
_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(?P<module>\S+)\s+"
    r"(?P<level>WARNING|ERROR)\s+"
    r"(?P<func>\S+?):(?P<lineno>\d+)\s+"
    r"(?P<msg>.*)$"
)

# Aggressively normalize varying parts so different occurrences of "the
# same problem" group together. Order matters: longer patterns first.
_NORMALIZERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?\b"), "<ts>"),
    (re.compile(r"\b/[\w./-]+\.(?:py|json|yaml|md|jsonl|pdf|pt|csv)\b"), "<path>"),
    (re.compile(r"\b\$[0-9.]+\b"), "<$>"),
    (re.compile(r"\b\d+\.\d{2,}\b"), "<float>"),
    (re.compile(r"\b\d{4,}\b"), "<num>"),
    (re.compile(r"'[^']{8,}'"), "<str>"),
]


def _signature(msg: str) -> str:
    """Collapse variable spans so similar lines bucket together."""
    out = msg
    for pat, repl in _NORMALIZERS:
        out = pat.sub(repl, out)
    return out[:240]


def write_run_summary(log_path: Path, output_path: Path) -> bool:
    """Read ``log_path``, group warn/err lines, write a markdown summary.

    Returns True if a summary was written, False if the log was missing
    or empty. Never raises — best-effort.
    """
    try:
        if not log_path.exists() or log_path.stat().st_size == 0:
            return False
    except OSError:
        return False

    by_level: dict[str, dict[tuple[str, str], list[dict]]] = {
        "ERROR": defaultdict(list),
        "WARNING": defaultdict(list),
    }

    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _LINE.match(line)
                if not m:
                    continue
                level = m.group("level")
                func = m.group("func")
                module = m.group("module").split(".")[-1]
                msg = m.group("msg").strip()
                key = (f"{module}.{func}", _signature(msg))
                by_level[level][key].append({
                    "ts": m.group("ts"),
                    "lineno": m.group("lineno"),
                    "msg": msg,
                })
    except OSError as e:
        logger.warning("write_run_summary: failed to read %s: %s", log_path, e)
        return False

    total_err = sum(len(v) for v in by_level["ERROR"].values())
    total_warn = sum(len(v) for v in by_level["WARNING"].values())
    if total_err == 0 and total_warn == 0:
        return False

    out: list[str] = [
        "# Run Error Summary",
        "",
        f"Source: `{log_path}`",
        f"Errors: {total_err}, Warnings: {total_warn}",
        "",
    ]

    for level in ("ERROR", "WARNING"):
        buckets = by_level[level]
        if not buckets:
            continue
        # Sort buckets by count desc — surface the most-frequent issues first.
        ranked = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
        out.append(f"## {level} ({sum(len(v) for v in buckets.values())} total, {len(buckets)} distinct)")
        out.append("")
        for (origin, sig), occurrences in ranked:
            out.append(f"### {origin} × {len(occurrences)}")
            out.append("")
            # Sample: first + last so chronology is visible.
            sample = occurrences[0]
            out.append(f"- First at `{sample['ts']}` (run.log:{sample['lineno']}):")
            out.append(f"  > {sample['msg'][:600]}")
            if len(occurrences) > 1:
                last = occurrences[-1]
                out.append(f"- Last at `{last['ts']}`")
            out.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out))
    return True

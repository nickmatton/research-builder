"""Section verifier — deterministic checks + tool-free LLM judge prompt.

Replaces the old tool-loop LLM verifier. The old one's job was to gate each
section's outputs against its acceptance criteria, but in practice it:

  * burned compute investigating files via Read/Bash for many turns,
  * frequently hit ``max_turns`` mid-investigation,
  * fail-OPENed on every failure mode (parse fail, crash, no JSON), so a
    verifier that crashed was indistinguishable from one that approved.

This module is structured as two layers:

  1. :func:`run_deterministic_checks` — cheap, reliable, no LLM. Catches the
     bulk of real defects (missing files, syntax errors, failing tests,
     vacuous test assertions).
  2. :func:`build_judge_user_prompt` — builds the user-turn prompt for a
     tool-free LLM judge call. The orchestrator wires this into a
     single-turn inference (no Read/Bash/Glob/Grep, no tool loop). The
     judge decides accept/reject purely from the inlined acceptance
     criteria, file contents, and test report.

Both layers are fail-closed: anything that can't be verified is rejected,
not auto-accepted.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..models.results import SubAgentResult, TestReport
from ..models.spec import Artifact


# Per-file content cap when building the LLM judge prompt. Keeps the prompt
# bounded even for large multi-file sections.
PER_FILE_CHAR_CAP = 8_000
# Hard cap across all files. Prevents a section with 50 small files from
# ballooning the prompt.
TOTAL_CONTENT_CAP = 50_000


@dataclass
class CheckResult:
    """One deterministic check outcome.

    ``name`` is a short identifier (``output_exists:model.py``) suitable for
    grouping in dashboards or step records. ``detail`` is the human-readable
    explanation that ends up in the Builder's retry feedback when the
    check fails.
    """

    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def _resolve(work_dir: Path | None, file_path: str) -> Path:
    """Resolve an artifact's file_path. Treats relative paths as living
    under ``work_dir``; absolute paths pass through unchanged.

    The spec authoring convention is ``phases/<phase_id>/outputs/<file>``
    (project-root-relative), but ``work_dir`` is already
    ``<project_root>/phases/<phase_id>/`` — so a naive join double-nests
    the prefix and the file looks "missing" even when it exists. When the
    file_path starts with the work_dir's last two path segments
    (``phases/<phase_id>/``), strip them before joining. Builder-relative
    paths like ``outputs/foo.py`` pass through unchanged.
    """
    p = Path(file_path)
    if p.is_absolute() or work_dir is None:
        return p
    if len(work_dir.parts) >= 2:
        prefix = Path(work_dir.parts[-2]) / work_dir.parts[-1]
        try:
            p = p.relative_to(prefix)
        except ValueError:
            pass
    return work_dir / p


_PLACEHOLDER_COMMENT_RE = re.compile(
    r"#\s*(TODO|FIXME|XXX|HACK|placeholder)\b"
    r"|would be implemented"
    r"|implementation depends on"
    r"|left as an exercise"
    r"|implement(ed)? (later|here|me)"
    r"|fill in (the )?(rest|implementation|details)",
    re.IGNORECASE,
)


def _find_stub_bodies(tree: ast.AST) -> list[str]:
    """Return names of functions/methods whose body is only ``pass`` or ``...``
    or solely ``raise NotImplementedError`` — the textbook signatures of a
    placeholder implementation.

    Abstract methods (decorated with ``@abstractmethod`` / ``@abc.abstractmethod``)
    are excluded — a real ABC legitimately has stub bodies.
    """
    stubs: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorators = {
            (d.attr if isinstance(d, ast.Attribute) else getattr(d, "id", ""))
            for d in node.decorator_list
        }
        if "abstractmethod" in decorators:
            continue
        body = node.body
        # Skip leading docstring when judging whether the body is a stub.
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        if not body:
            stubs.append(node.name)
            continue
        if len(body) == 1:
            stmt = body[0]
            if isinstance(stmt, ast.Pass):
                stubs.append(node.name)
                continue
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is Ellipsis
            ):
                stubs.append(node.name)
                continue
            if isinstance(stmt, ast.Raise):
                exc = stmt.exc
                exc_name = ""
                if isinstance(exc, ast.Call):
                    exc_name = getattr(exc.func, "id", "") or getattr(exc.func, "attr", "")
                elif isinstance(exc, ast.Name):
                    exc_name = exc.id
                if exc_name == "NotImplementedError":
                    stubs.append(node.name)
    return stubs


# Name-shaped sweep markers. Two flavors:
#   - prefix patterns (grid_search_*, sweep_*, …) — naming the whole entry
#     point as a sweep.
#   - "verb + hyperparam" patterns (select_lr, pick_alpha, tune_step_size, …)
#     — naming the local mini-search the model writes inside a "single" run.
# The second is what slipped through on the Adam paper run: the model wrote
# ``select_lr(model_factory, LR_CANDIDATES, ...)`` thinking it wasn't a
# "sweep" because it's just picking one number — but those 1500 wasted
# training steps per config ARE the sweep we want to kill.
_SWEEP_NAME_RE = re.compile(
    r"^(?:"
    r"grid_search|sweep|hyperparameter_search|hp_search|param_sweep|lr_sweep"
    r"|(?:select|pick|tune|find|search|sweep|auto)_+"
    r"(?:lr|lrs|alpha|alphas|step_size|step_sizes|hp|hyperparam|hyperparameter|"
    r"learning_rate|learning_rates|optim|optimizer)"
    r")",
    re.IGNORECASE,
)

# Module-level constants whose names announce a list of hyperparameters.
# Used together with the for-loop scan below: ``for lr in LR_CANDIDATES:``
# is the sweep shape, ``LR_CANDIDATES = [0.01, 0.02, 0.1]`` is the proof.
_HP_LIST_NAME_RE = re.compile(
    r"^(?:lr|lrs|alpha|alphas|step_size|step_sizes|"
    r"learning_rate|learning_rates|"
    r"(?:lr|alpha|step_size|hp|hyperparam|param)_(?:candidates|grid|options|choices|values|set|list)"
    r")$",
    re.IGNORECASE,
)

# Function-name fragments that indicate a training entry point. We use these
# to judge whether a `for x in [...]` loop body is iterating a training call
# (which makes the loop a sweep) vs something benign (iterating a list to
# accumulate features).
_TRAIN_CALL_NAME_RE = re.compile(
    r"(?:^|_)(train|fit|run_one|run_trial|one_run|train_one|step_through|optimize_step)(?:$|_)",
    re.IGNORECASE,
)


def _loop_iterates_hyperparams(node: ast.For) -> str | None:
    """If this for-loop is iterating a list of hyperparameters around a
    training call, return a human-readable marker; else None.

    Detects three iter shapes:
      1. ``for lr in [0.01, 0.02, 0.1]:`` — list literal of numeric constants.
      2. ``for lr in LR_CANDIDATES:`` — Name reference matching the HP-list
         name pattern.
      3. ``for lr, bs in zip(LRS, BATCH_SIZES):`` — zip over two HP-named iterables.
    And requires the loop body to contain a Call whose function name (or
    attr-chain tail) matches the training-entry-point pattern. Without the
    body check we'd flag a benign ``for lr in scheduler.lrs(): log(lr)``.
    """
    iter_node = node.iter
    looks_like_hp_iter = False
    iter_label = ""

    if isinstance(iter_node, ast.List):
        if iter_node.elts and all(
            isinstance(e, ast.Constant) and isinstance(e.value, (int, float))
            for e in iter_node.elts
        ):
            looks_like_hp_iter = True
            iter_label = "[<numeric literals>]"
    elif isinstance(iter_node, ast.Name):
        if _HP_LIST_NAME_RE.match(iter_node.id):
            looks_like_hp_iter = True
            iter_label = iter_node.id
    elif isinstance(iter_node, ast.Call):
        # zip(LRS, BATCH_SIZES) and similar.
        func = iter_node.func
        if isinstance(func, ast.Name) and func.id == "zip":
            hp_names = [
                a.id for a in iter_node.args
                if isinstance(a, ast.Name) and _HP_LIST_NAME_RE.match(a.id)
            ]
            if len(hp_names) >= 2:
                looks_like_hp_iter = True
                iter_label = f"zip({', '.join(hp_names)})"

    if not looks_like_hp_iter:
        return None

    # Body must call something train-shaped.
    body_calls_train = False
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            name = ""
            if isinstance(f, ast.Name):
                name = f.id
            elif isinstance(f, ast.Attribute):
                name = f.attr
            if name and _TRAIN_CALL_NAME_RE.search(name):
                body_calls_train = True
                break
    if not body_calls_train:
        return None
    return f"`for {ast.unparse(node.target)} in {iter_label}:` wraps a training call"


def _find_sweep_patterns(tree: ast.AST) -> list[str]:
    """Return source markers that look like hyperparameter sweeps.

    The harness's "no sweeps" rule is in the Builder prompt, but the model
    sometimes ignores it — and a single bad section can burn an hour of GPU
    budget before anyone notices. This walker catches the giveaway shapes
    so the verifier can reject before retry, not after the cloud bill:

      - function names like ``grid_search_*`` / ``sweep_*`` / ``select_lr`` /
        ``pick_alpha`` / ``tune_step_size`` (a local lr-selection still IS a
        sweep — observed on the VAE run that we tried to forbid in prompts)
      - calls to ``itertools.product`` near a training entry-point
      - ``for lr in [0.01, 0.02, 0.1]:`` or ``for lr in LR_CANDIDATES:``
        loops whose body calls a training-shaped function

    Returns human-readable strings; empty list ⇒ no sweep markers found.
    """
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _SWEEP_NAME_RE.match(node.name):
                findings.append(f"function `{node.name}` (sweep-named entry point)")
        elif isinstance(node, ast.For):
            marker = _loop_iterates_hyperparams(node)
            if marker:
                findings.append(marker)
        elif isinstance(node, ast.Call):
            func = node.func
            # itertools.product(...) — overwhelmingly used for hyperparameter
            # cartesian products in reproduction code; the rare legitimate use
            # is in data preprocessing, but we'd rather flag and let the model
            # explain than miss a sweep.
            if isinstance(func, ast.Attribute):
                if (
                    func.attr == "product"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "itertools"
                ):
                    findings.append("`itertools.product(...)` — cartesian product over configs")
    return findings


def _find_starved_dataloaders(tree: ast.AST) -> list[str]:
    """Return source markers for ``DataLoader(...)`` calls that will starve the GPU.

    Detects two failure modes we've actually paid for:
      1. ``num_workers`` omitted entirely — PyTorch's default of 0 means the
         main thread blocks on every batch fetch, GPU sits idle.
      2. ``num_workers=0`` set explicitly — same outcome, different shape.

    We DON'T flag low ``num_workers`` values (1, 2) — there are legitimate
    reasons (debugging, tiny datasets) and we'd rather under-flag than nag.
    A run that's worker-starved at num_workers=2 will still show ``gpu_util``
    drops in the Compute tab; the operator can catch it there.

    Returns empty if no DataLoader calls present.
    """
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = ""
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name != "DataLoader":
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        nw = kwargs.get("num_workers")
        if nw is None:
            findings.append(
                f"DataLoader(...) on line {node.lineno} omits `num_workers` "
                "(default 0 starves the GPU)"
            )
            continue
        # Catch num_workers=0 / num_workers=False literals; non-literal
        # expressions (e.g. config.num_workers) get the benefit of the doubt.
        if isinstance(nw, ast.Constant) and nw.value in (0, False):
            findings.append(
                f"DataLoader(...) on line {node.lineno} has num_workers=0 "
                "(main thread will block on each batch — set ≥4)"
            )
    return findings


def _has_nontrivial_assert(tree: ast.AST) -> bool:
    """True iff the AST contains at least one ``assert`` whose condition is
    not a trivially-true constant.

    Vacuous (returns False if these are the only asserts present):
      - ``assert True``
      - ``assert 1``
      - ``assert "non-empty string"``

    Real (returns True): comparisons, function calls, attribute access, name
    references, anything that could actually fail at runtime.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            test = node.test
            if isinstance(test, ast.Constant) and bool(test.value):
                continue
            return True
    return False


def run_deterministic_checks(
    phase_outputs: list[Artifact],
    builder_result: SubAgentResult,
    work_dir: Path | None,
) -> list[CheckResult]:
    """Run all cheap deterministic checks.

    Returns one :class:`CheckResult` per check, in execution order. The
    caller decides whether to accept/reject based on any ``passed=False``
    entries.
    """
    results: list[CheckResult] = []

    # 1. Every spec-declared output must exist on disk, be non-empty, and
    #    pass type-specific structural checks (python parses, json parses,
    #    test files have real asserts).
    for art in phase_outputs:
        path = _resolve(work_dir, art.file_path)
        if not path.exists():
            results.append(CheckResult(
                name=f"output_exists:{art.name}",
                passed=False,
                detail=f"Expected output `{art.name}` not found at `{path}`",
            ))
            # No point running downstream checks on a missing file.
            continue
        results.append(CheckResult(name=f"output_exists:{art.name}", passed=True))

        if path.stat().st_size == 0:
            results.append(CheckResult(
                name=f"output_nonempty:{art.name}",
                passed=False,
                detail=f"Output `{art.name}` at `{path}` is zero bytes",
            ))
            continue
        results.append(CheckResult(name=f"output_nonempty:{art.name}", passed=True))

        if path.suffix == ".py":
            try:
                source = path.read_text(errors="replace")
                tree = ast.parse(source)
            except SyntaxError as e:
                results.append(CheckResult(
                    name=f"python_parses:{art.name}",
                    passed=False,
                    detail=f"Syntax error in `{path}` line {e.lineno}: {e.msg}",
                ))
                continue
            results.append(CheckResult(name=f"python_parses:{art.name}", passed=True))

            # Placeholder / stub detection. Production outputs must be real
            # implementations, not scaffolding with `NotImplementedError`,
            # `pass`-only bodies, or comments like "would be implemented".
            # Skip declared test files — those legitimately have helper
            # stubs and the vacuous-assert check below covers their quality.
            is_test_file = path.name.startswith("test_") or path.name.endswith("_test.py")
            if not is_test_file:
                stubs = _find_stub_bodies(tree)
                placeholder_lines = [
                    (i + 1, line.strip())
                    for i, line in enumerate(source.splitlines())
                    if _PLACEHOLDER_COMMENT_RE.search(line)
                ]
                if stubs or placeholder_lines:
                    detail_parts = []
                    if stubs:
                        detail_parts.append(
                            f"stub bodies (pass/.../NotImplementedError) in: "
                            + ", ".join(stubs[:8])
                            + ("…" if len(stubs) > 8 else "")
                        )
                    if placeholder_lines:
                        first = placeholder_lines[:3]
                        rendered = "; ".join(f"L{n}: {t[:80]}" for n, t in first)
                        more = f" (+{len(placeholder_lines) - 3} more)" if len(placeholder_lines) > 3 else ""
                        detail_parts.append(f"placeholder comments — {rendered}{more}")
                    results.append(CheckResult(
                        name=f"no_placeholders:{art.name}",
                        passed=False,
                        detail=(
                            f"`{path}` contains placeholder/stub code: "
                            + " | ".join(detail_parts)
                            + ". Ship a real implementation or call "
                            "report_result with is_spec_issue=true / "
                            "status=failure — do not report success with "
                            "scaffolding in production paths."
                        ),
                    ))
                else:
                    results.append(CheckResult(
                        name=f"no_placeholders:{art.name}", passed=True,
                    ))

                sweep_markers = _find_sweep_patterns(tree)
                if sweep_markers:
                    results.append(CheckResult(
                        name=f"no_hp_sweep:{art.name}",
                        passed=False,
                        detail=(
                            f"`{path}` contains hyperparameter-sweep patterns: "
                            + "; ".join(sweep_markers[:5])
                            + (f" (+{len(sweep_markers) - 5} more)" if len(sweep_markers) > 5 else "")
                            + ". Reproduce the paper's reported configuration "
                            "(or, for sensitivity figures, the exact point-sampled "
                            "configs the paper plotted) — never a grid. Each "
                            "deliverable should map to one explicit `train_one_run` "
                            "call with hard-coded hyperparameters; delete the loop "
                            "and replace with literal calls. See the "
                            "\"No hyperparameter sweeps\" rule in the system prompt."
                        ),
                    ))
                else:
                    results.append(CheckResult(
                        name=f"no_hp_sweep:{art.name}", passed=True,
                    ))

                starved_loaders = _find_starved_dataloaders(tree)
                if starved_loaders:
                    results.append(CheckResult(
                        name=f"dataloader_keeps_gpu_fed:{art.name}",
                        passed=False,
                        detail=(
                            f"`{path}` has GPU-starving DataLoader(s): "
                            + "; ".join(starved_loaders[:3])
                            + (f" (+{len(starved_loaders) - 3} more)" if len(starved_loaders) > 3 else "")
                            + ". Every training DataLoader must set "
                            "num_workers≥4, pin_memory=True, "
                            "persistent_workers=True, prefetch_factor=4 — see the "
                            "\"keep the GPU fed (data pipeline)\" rule in the "
                            "Training phase prompt. A starved GPU bills $1–3/hr "
                            "while doing nothing."
                        ),
                    ))
                else:
                    results.append(CheckResult(
                        name=f"dataloader_keeps_gpu_fed:{art.name}", passed=True,
                    ))

            # Vacuous-test check. Only applies to files the spec declared as
            # outputs whose name follows the pytest convention — secret test
            # files the builder may have written but not declared are out of
            # scope for v1.
            if is_test_file:
                if _has_nontrivial_assert(tree):
                    results.append(CheckResult(
                        name=f"test_has_real_assert:{art.name}",
                        passed=True,
                    ))
                else:
                    results.append(CheckResult(
                        name=f"test_has_real_assert:{art.name}",
                        passed=False,
                        detail=(
                            f"Test file `{path}` has no non-trivial "
                            "assertions (only `assert True` / `assert 1` "
                            "or no asserts at all). Real tests must check "
                            "something that could fail."
                        ),
                    ))

        elif path.suffix == ".json":
            try:
                json.loads(path.read_text())
            except json.JSONDecodeError as e:
                results.append(CheckResult(
                    name=f"json_parses:{art.name}",
                    passed=False,
                    detail=f"Invalid JSON in `{path}` line {e.lineno}: {e.msg}",
                ))
                continue
            results.append(CheckResult(name=f"json_parses:{art.name}", passed=True))

    # 2. If the phase produced any Python code, it should have run tests.
    #    Skipping this check for pure-spec / pure-data phases that don't
    #    declare any ``.py`` outputs.
    tr: TestReport = builder_result.test_report
    has_python_output = any(
        Path(a.file_path).suffix == ".py" for a in phase_outputs
    )
    if has_python_output:
        if tr.tests_run == 0:
            results.append(CheckResult(
                name="tests_ran",
                passed=False,
                detail=(
                    "Builder reported 0 tests run but the phase produced "
                    "Python code. Sections with code must have tests that "
                    "exercise the acceptance criteria — a zero-test report "
                    "means nothing was actually verified."
                ),
            ))
        else:
            results.append(CheckResult(
                name="tests_ran",
                passed=True,
                detail=f"{tr.tests_run} tests run",
            ))

    if tr.tests_failed > 0:
        failed_names = [
            t.test_name
            for t in tr.test_details
            if t.status.value in ("failed", "error")
        ]
        results.append(CheckResult(
            name="tests_passed",
            passed=False,
            detail=(
                f"{tr.tests_failed} failing test(s): "
                + (", ".join(failed_names[:5]) if failed_names else "<names not reported>")
            ),
        ))
    else:
        results.append(CheckResult(
            name="tests_passed",
            passed=True,
            detail=f"{tr.tests_passed} passed",
        ))

    return results


def summarize_failures(checks: Iterable[CheckResult]) -> str:
    """Render the failing checks as a single feedback string for the
    Builder's retry context. Returns empty string when nothing failed."""
    failed = [c for c in checks if not c.passed]
    if not failed:
        return ""
    lines = [f"Deterministic verification failed ({len(failed)} issue(s)):"]
    for c in failed:
        lines.append(f"- [{c.name}] {c.detail}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool-free LLM judge prompt construction
# ---------------------------------------------------------------------------


def build_judge_user_prompt(
    phase_outputs: list[Artifact],
    builder_result: SubAgentResult,
    work_dir: Path | None,
    acceptance_criteria_md: str,
) -> str:
    """Build the user-turn prompt for the tool-free LLM judge.

    Inlines acceptance criteria, the builder's summary + test report, and
    truncated contents of every declared output. The judge then decides
    accept/reject from this single message — no tools, no follow-up turns.
    """
    parts: list[str] = []

    parts.append("## Acceptance criteria\n")
    parts.append(acceptance_criteria_md.strip() or "(no criteria provided)")

    parts.append("\n\n## Builder summary\n")
    parts.append((builder_result.summary or "(no summary)").strip())

    tr = builder_result.test_report
    parts.append("\n\n## Test report\n")
    parts.append(
        f"- tests_run: {tr.tests_run}\n"
        f"- tests_passed: {tr.tests_passed}\n"
        f"- tests_failed: {tr.tests_failed}\n"
    )
    if tr.test_details:
        parts.append("- per-test (first 30):\n")
        for t in tr.test_details[:30]:
            mark = "PASS" if t.status.value == "passed" else "FAIL"
            desc = (t.description or "").strip()
            parts.append(f"  - [{mark}] {t.test_name}{(': ' + desc) if desc else ''}\n")
            if t.message:
                msg = t.message.strip().replace("\n", " ⏎ ")
                parts.append(f"           ↳ {msg[:300]}\n")

    parts.append("\n\n## Output files\n")
    total = 0
    for art in phase_outputs:
        path = _resolve(work_dir, art.file_path)
        if not path.exists():
            parts.append(f"\n### {art.name} (`{path}`) — MISSING\n")
            continue
        try:
            content = path.read_text(errors="replace")
        except Exception as e:  # noqa: BLE001 — boundary, anything from disk goes
            parts.append(f"\n### {art.name} (`{path}`) — UNREADABLE: {e}\n")
            continue

        if len(content) > PER_FILE_CHAR_CAP:
            head_len = PER_FILE_CHAR_CAP // 2
            head = content[:head_len]
            tail = content[-(PER_FILE_CHAR_CAP - head_len):]
            omitted = len(content) - PER_FILE_CHAR_CAP
            content = f"{head}\n\n…<truncated {omitted} chars>…\n\n{tail}"

        remaining = TOTAL_CONTENT_CAP - total
        if remaining <= 0:
            parts.append(
                f"\n### {art.name} (`{path}`) — omitted (prompt budget exhausted)\n"
            )
            continue
        if len(content) > remaining:
            content = content[:remaining] + "\n…<prompt budget cap reached>…"

        ext = path.suffix.lstrip(".") or ""
        parts.append(f"\n### {art.name} (`{path}`)\n```{ext}\n{content}\n```\n")
        total += len(content)

    return "".join(parts)

"""Tests for the deterministic section verifier checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_builder.models.results import (
    ResultStatus,
    SubAgentResult,
    TestReport,
    TestResult,
    TestStatus,
)
from research_builder.models.spec import Artifact
from research_builder.orchestrator.section_verifier import (
    build_judge_user_prompt,
    run_deterministic_checks,
    summarize_failures,
)


def _result(
    *,
    outputs: list[Artifact] | None = None,
    tests_run: int = 1,
    tests_passed: int = 1,
    tests_failed: int = 0,
    test_details: list[TestResult] | None = None,
) -> SubAgentResult:
    """Helper: build a SubAgentResult with sensible defaults."""
    return SubAgentResult(
        status=ResultStatus.success,
        phase_id="phase_x",
        outputs=outputs or [],
        summary="ok",
        test_report=TestReport(
            tests_run=tests_run,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            test_details=test_details or [],
        ),
    )


class TestOutputExistence:
    def test_missing_output_fails(self, tmp_path: Path):
        outputs = [Artifact(name="model", file_path="model.py")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        names = {c.name: c for c in checks}
        assert names["output_exists:model"].passed is False
        assert "not found" in names["output_exists:model"].detail

    def test_existing_output_passes(self, tmp_path: Path):
        (tmp_path / "model.py").write_text("x = 1\n")
        outputs = [Artifact(name="model", file_path="model.py")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        names = {c.name: c for c in checks}
        assert names["output_exists:model"].passed is True

    def test_absolute_path_resolves_outside_workdir(self, tmp_path: Path):
        somewhere = tmp_path / "other"
        somewhere.mkdir()
        (somewhere / "x.py").write_text("x = 1\n")
        outputs = [Artifact(name="x", file_path=str(somewhere / "x.py"))]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        names = {c.name: c for c in checks}
        assert names["output_exists:x"].passed is True

    def test_spec_relative_path_does_not_double_nest(self, tmp_path: Path):
        # Spec convention: artifact file_path is `phases/<phase_id>/outputs/<file>`
        # (project-root-relative). work_dir is already `<root>/phases/<phase_id>`,
        # so a naive join would produce
        # `<root>/phases/<phase_id>/phases/<phase_id>/outputs/<file>` and the
        # file would falsely look missing.
        phase_id = "section_4_lstm_architecture"
        work_dir = tmp_path / "phases" / phase_id
        (work_dir / "outputs").mkdir(parents=True)
        (work_dir / "outputs" / "lstm_cell.py").write_text("x = 1\n")
        outputs = [Artifact(
            name="lstm_cell",
            file_path=f"phases/{phase_id}/outputs/lstm_cell.py",
        )]
        checks = run_deterministic_checks(outputs, _result(), work_dir)
        names = {c.name: c for c in checks}
        assert names["output_exists:lstm_cell"].passed is True


class TestNonEmpty:
    def test_zero_byte_output_fails(self, tmp_path: Path):
        (tmp_path / "model.py").write_text("")
        outputs = [Artifact(name="model", file_path="model.py")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        names = {c.name: c for c in checks}
        assert names["output_nonempty:model"].passed is False


class TestPythonParses:
    def test_syntax_error_fails(self, tmp_path: Path):
        (tmp_path / "broken.py").write_text("def f(:\n  pass\n")
        outputs = [Artifact(name="broken", file_path="broken.py")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        names = {c.name: c for c in checks}
        assert names["python_parses:broken"].passed is False
        assert "Syntax error" in names["python_parses:broken"].detail

    def test_valid_python_passes(self, tmp_path: Path):
        (tmp_path / "ok.py").write_text("def f():\n    return 1\n")
        outputs = [Artifact(name="ok", file_path="ok.py")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        names = {c.name: c for c in checks}
        assert names["python_parses:ok"].passed is True


class TestJsonParses:
    def test_broken_json_fails(self, tmp_path: Path):
        (tmp_path / "data.json").write_text("{not valid")
        outputs = [Artifact(name="data", file_path="data.json")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        names = {c.name: c for c in checks}
        assert names["json_parses:data"].passed is False

    def test_valid_json_passes(self, tmp_path: Path):
        (tmp_path / "data.json").write_text('{"a": 1}')
        outputs = [Artifact(name="data", file_path="data.json")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        names = {c.name: c for c in checks}
        assert names["json_parses:data"].passed is True


class TestPlaceholderDetection:
    def test_not_implemented_error_fails(self, tmp_path: Path):
        (tmp_path / "stub.py").write_text(
            "def forward(x):\n    raise NotImplementedError\n"
        )
        outputs = [Artifact(name="stub", file_path="stub.py")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        names = {c.name: c for c in checks}
        assert names["no_placeholders:stub"].passed is False
        assert "forward" in names["no_placeholders:stub"].detail

    def test_pass_only_body_fails(self, tmp_path: Path):
        (tmp_path / "skel.py").write_text(
            'def backward(grad):\n    """Backward pass."""\n    pass\n'
        )
        outputs = [Artifact(name="skel", file_path="skel.py")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        names = {c.name: c for c in checks}
        assert names["no_placeholders:skel"].passed is False

    def test_ellipsis_body_fails(self, tmp_path: Path):
        (tmp_path / "elip.py").write_text("def step():\n    ...\n")
        outputs = [Artifact(name="elip", file_path="elip.py")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        names = {c.name: c for c in checks}
        assert names["no_placeholders:elip"].passed is False

    def test_would_be_implemented_comment_fails(self, tmp_path: Path):
        (tmp_path / "todo.py").write_text(
            "def f():\n"
            "    # The internal-state update would be implemented here\n"
            "    return 0\n"
        )
        outputs = [Artifact(name="todo", file_path="todo.py")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        names = {c.name: c for c in checks}
        assert names["no_placeholders:todo"].passed is False

    def test_todo_fixme_comment_fails(self, tmp_path: Path):
        (tmp_path / "x.py").write_text("# TODO: wire this up\nX = 1\n")
        outputs = [Artifact(name="x", file_path="x.py")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        names = {c.name: c for c in checks}
        assert names["no_placeholders:x"].passed is False

    def test_real_implementation_passes(self, tmp_path: Path):
        (tmp_path / "real.py").write_text(
            "def add(a, b):\n    return a + b\n"
            "def mul(a, b):\n    return a * b\n"
        )
        outputs = [Artifact(name="real", file_path="real.py")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        names = {c.name: c for c in checks}
        assert names["no_placeholders:real"].passed is True

    def test_abstract_method_not_flagged(self, tmp_path: Path):
        # An @abstractmethod legitimately has an empty body.
        (tmp_path / "abc_iface.py").write_text(
            "from abc import ABC, abstractmethod\n"
            "class Base(ABC):\n"
            "    @abstractmethod\n"
            "    def step(self):\n"
            "        ...\n"
        )
        outputs = [Artifact(name="iface", file_path="abc_iface.py")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        names = {c.name: c for c in checks}
        assert names["no_placeholders:iface"].passed is True

    def test_test_files_exempt_from_placeholder_check(self, tmp_path: Path):
        # Test files are governed by the vacuous-assert check, not the
        # stub check. A test file that contains a `pass`-only helper but
        # has real asserts should not be flagged as a placeholder.
        (tmp_path / "test_thing.py").write_text(
            "def _setup():\n    pass\n"
            "def test_one():\n    assert 1 + 1 == 2\n"
        )
        outputs = [Artifact(name="t", file_path="test_thing.py")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        names = {c.name for c in checks}
        assert "no_placeholders:t" not in names


class TestVacuousAssertDetection:
    def test_assert_true_only_fails(self, tmp_path: Path):
        (tmp_path / "test_vacuous.py").write_text(
            "def test_one():\n    assert True\n"
            "def test_two():\n    assert 1\n"
        )
        outputs = [Artifact(name="tests", file_path="test_vacuous.py")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        names = {c.name: c for c in checks}
        assert names["test_has_real_assert:tests"].passed is False
        assert "vacuous" in names["test_has_real_assert:tests"].detail.lower() or \
               "non-trivial" in names["test_has_real_assert:tests"].detail.lower()

    def test_real_assert_passes(self, tmp_path: Path):
        (tmp_path / "test_real.py").write_text(
            "def test_one():\n    assert 1 + 1 == 2\n"
        )
        outputs = [Artifact(name="tests", file_path="test_real.py")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        names = {c.name: c for c in checks}
        assert names["test_has_real_assert:tests"].passed is True

    def test_non_test_file_skipped(self, tmp_path: Path):
        # A non-test file containing only `assert True` shouldn't be
        # flagged by the vacuous-assert check (the check only fires for
        # files matching the pytest naming convention).
        (tmp_path / "helper.py").write_text("def f():\n    assert True\n")
        outputs = [Artifact(name="helper", file_path="helper.py")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        names = {c.name for c in checks}
        assert "test_has_real_assert:helper" not in names


class TestTestsRanCheck:
    def test_zero_tests_with_python_output_fails(self, tmp_path: Path):
        (tmp_path / "model.py").write_text("x = 1\n")
        outputs = [Artifact(name="model", file_path="model.py")]
        checks = run_deterministic_checks(
            outputs, _result(tests_run=0, tests_passed=0), tmp_path,
        )
        names = {c.name: c for c in checks}
        assert names["tests_ran"].passed is False

    def test_zero_tests_with_no_python_outputs_skipped(self, tmp_path: Path):
        # Pure-doc / pure-json phases legitimately have no tests; the
        # check shouldn't fire.
        (tmp_path / "spec.json").write_text("{}")
        outputs = [Artifact(name="spec", file_path="spec.json")]
        checks = run_deterministic_checks(
            outputs, _result(tests_run=0, tests_passed=0), tmp_path,
        )
        names = {c.name for c in checks}
        assert "tests_ran" not in names


class TestTestsPassedCheck:
    def test_failing_tests_rejected_with_names(self, tmp_path: Path):
        details = [
            TestResult(test_name="test_alpha", status=TestStatus.passed),
            TestResult(test_name="test_beta", status=TestStatus.failed),
        ]
        checks = run_deterministic_checks(
            [],
            _result(tests_run=2, tests_passed=1, tests_failed=1, test_details=details),
            tmp_path,
        )
        names = {c.name: c for c in checks}
        assert names["tests_passed"].passed is False
        assert "test_beta" in names["tests_passed"].detail

    def test_all_pass(self, tmp_path: Path):
        checks = run_deterministic_checks(
            [],
            _result(tests_run=3, tests_passed=3),
            tmp_path,
        )
        names = {c.name: c for c in checks}
        assert names["tests_passed"].passed is True


class TestSummarize:
    def test_no_failures_returns_empty(self, tmp_path: Path):
        (tmp_path / "ok.py").write_text("x = 1\n")
        outputs = [Artifact(name="ok", file_path="ok.py")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        assert summarize_failures(checks) == ""

    def test_failures_render_with_details(self, tmp_path: Path):
        outputs = [Artifact(name="missing", file_path="nope.py")]
        checks = run_deterministic_checks(outputs, _result(), tmp_path)
        summary = summarize_failures(checks)
        assert "Deterministic verification failed" in summary
        assert "output_exists:missing" in summary
        assert "nope.py" in summary


class TestJudgePrompt:
    def test_prompt_includes_criteria_and_file_contents(self, tmp_path: Path):
        (tmp_path / "model.py").write_text("class LSTM:\n    pass\n")
        outputs = [Artifact(name="model", file_path="model.py")]
        prompt = build_judge_user_prompt(
            phase_outputs=outputs,
            builder_result=_result(),
            work_dir=tmp_path,
            acceptance_criteria_md="The model must define an LSTM class.",
        )
        assert "must define an LSTM class" in prompt
        assert "class LSTM" in prompt
        assert "tests_run: 1" in prompt

    def test_large_file_is_truncated(self, tmp_path: Path):
        big = "x = 1\n" * 5000  # ~30k chars, above PER_FILE_CHAR_CAP
        (tmp_path / "big.py").write_text(big)
        outputs = [Artifact(name="big", file_path="big.py")]
        prompt = build_judge_user_prompt(
            phase_outputs=outputs,
            builder_result=_result(),
            work_dir=tmp_path,
            acceptance_criteria_md="anything",
        )
        assert "truncated" in prompt

    def test_missing_file_marked_in_prompt(self, tmp_path: Path):
        outputs = [Artifact(name="gone", file_path="gone.py")]
        prompt = build_judge_user_prompt(
            phase_outputs=outputs,
            builder_result=_result(),
            work_dir=tmp_path,
            acceptance_criteria_md="anything",
        )
        assert "MISSING" in prompt

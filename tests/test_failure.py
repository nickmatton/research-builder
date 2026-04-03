"""Tests for failure handler and retry budget tracking."""

from research_builder.models.results import ResultStatus, SubAgentResult
from research_builder.orchestrator.failure import FailureHandler


class TestFailureHandler:
    def test_initial_state(self):
        fh = FailureHandler(max_retries=3)
        assert fh.can_retry("data")
        assert fh.retries_used("data") == 0
        assert fh.get_prior_results("data") == []

    def test_record_failure(self):
        fh = FailureHandler(max_retries=3)
        result = SubAgentResult(status=ResultStatus.failure, phase_id="data", summary="OOM")
        fh.record_result("data", result)
        assert fh.retries_used("data") == 1
        assert fh.can_retry("data")
        assert len(fh.get_prior_results("data")) == 1

    def test_exhaust_retries(self):
        fh = FailureHandler(max_retries=2)
        for i in range(2):
            fh.record_result("data", SubAgentResult(
                status=ResultStatus.failure, phase_id="data", summary=f"fail {i}",
            ))
        assert not fh.can_retry("data")
        assert fh.retries_used("data") == 2

    def test_spec_issue_not_counted(self):
        fh = FailureHandler(max_retries=2)
        # Spec issue
        fh.record_result("data", SubAgentResult(
            status=ResultStatus.failure, phase_id="data", summary="bad URL",
            is_spec_issue=True,
        ))
        assert fh.retries_used("data") == 0
        assert fh.can_retry("data")

        # Real failure
        fh.record_result("data", SubAgentResult(
            status=ResultStatus.failure, phase_id="data", summary="impl bug",
        ))
        assert fh.retries_used("data") == 1
        assert fh.can_retry("data")

    def test_success_not_counted(self):
        fh = FailureHandler(max_retries=2)
        fh.record_result("data", SubAgentResult(
            status=ResultStatus.success, phase_id="data", summary="done",
        ))
        assert fh.retries_used("data") == 0
        assert len(fh.get_prior_results("data")) == 1

    def test_independent_phases(self):
        fh = FailureHandler(max_retries=2)
        fh.record_result("data", SubAgentResult(
            status=ResultStatus.failure, phase_id="data", summary="fail",
        ))
        fh.record_result("arch", SubAgentResult(
            status=ResultStatus.failure, phase_id="arch", summary="fail",
        ))
        assert fh.retries_used("data") == 1
        assert fh.retries_used("arch") == 1
        assert fh.can_retry("data")
        assert fh.can_retry("arch")

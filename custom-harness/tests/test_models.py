"""Tests for Pydantic data models."""

from datetime import datetime

import yaml

import pytest

from research_builder.models.spec import (
    AcceptanceCriterion,
    Artifact,
    Citation,
    CritiqueVerdict,
    DagNode,
    EventType,
    FileRole,
    PhaseState,
    PhaseStatus,
    PlannedFile,
    Revision,
    SectionCritique,
    SectionSpec,
    SpecMetadata,
    SpecState,
)
from research_builder.models.results import (
    ResultStatus,
    SubAgentResult,
    TestReport,
    TestResult,
    TestStatus,
)
from research_builder.models.context import (
    AdjacentPhaseSummary,
    RetryContext,
    RunState,
    RunStatus,
    SubSpec,
)


# --- spec.py ---


class TestArtifact:
    def test_create(self):
        a = Artifact(name="dataset", file_path="/data/train.pt")
        assert a.name == "dataset"
        assert a.file_path == "/data/train.pt"

    def test_json_roundtrip(self):
        a = Artifact(name="model", file_path="/checkpoints/best.pt")
        data = a.model_dump(mode="json")
        restored = Artifact.model_validate(data)
        assert restored == a


class TestRevision:
    def test_defaults(self):
        r = Revision(event_type=EventType.spec_created, rationale="initial")
        assert r.phase_id is None
        assert isinstance(r.timestamp, datetime)

    def test_phase_scoped(self):
        r = Revision(
            event_type=EventType.phase_completed,
            phase_id="data",
            rationale="all tests passed",
        )
        assert r.phase_id == "data"


class TestPhaseState:
    def test_defaults(self):
        p = PhaseState(phase_id="data", title="Data Phase")
        assert p.status == PhaseStatus.pending
        assert p.max_debug_attempts == 10
        assert p.inputs == []
        assert p.outputs == []

    def test_with_artifacts(self):
        p = PhaseState(
            phase_id="training",
            title="Training Phase",
            inputs=[Artifact(name="data_loader", file_path="/phases/data/1/outputs/loader.pt")],
            outputs=[Artifact(name="checkpoint", file_path="/phases/training/1/outputs/model.pt")],
        )
        assert len(p.inputs) == 1
        assert p.outputs[0].name == "checkpoint"


class TestDagNodeSubStepsCoercion:
    def test_strings_pass_through(self):
        node = DagNode(phase_id="p", title="P", sub_steps=["a", "b"])
        assert node.sub_steps == ["a", "b"]

    def test_dict_items_flatten_to_string(self):
        node = DagNode(
            phase_id="p",
            title="P",
            sub_steps=[
                {"Implement activation functions": "sigmoid f(x)"},
                "raw bullet",
            ],
        )
        assert node.sub_steps == [
            "Implement activation functions: sigmoid f(x)",
            "raw bullet",
        ]

    def test_multi_key_dict_joined(self):
        node = DagNode(
            phase_id="p",
            title="P",
            sub_steps=[{"step": "do x", "note": "carefully"}],
        )
        assert node.sub_steps == ["step: do x; note: carefully"]

    def test_non_string_non_dict_coerced(self):
        node = DagNode(phase_id="p", title="P", sub_steps=[1, 2.5])
        assert node.sub_steps == ["1", "2.5"]


class TestPlannedFileRoleCoercion:
    def test_valid_enum_passes(self):
        f = PlannedFile(file_id="f", rel_path="x", owning_phase="p", role="output")
        assert f.role == FileRole.output

    def test_unknown_role_coerced_to_output(self):
        f = PlannedFile(
            file_id="f", rel_path="x", owning_phase="p", role="core_module"
        )
        assert f.role == FileRole.output

    def test_uppercase_role_normalized(self):
        f = PlannedFile(file_id="f", rel_path="x", owning_phase="p", role="INPUT")
        assert f.role == FileRole.input

    def test_empty_role_defaults_to_output(self):
        f = PlannedFile(file_id="f", rel_path="x", owning_phase="p", role="")
        assert f.role == FileRole.output


class TestSpecState:
    def _make_state(self):
        return SpecState(
            metadata=SpecMetadata(paper_id="arxiv:1234", paper_title="Test Paper"),
            phases=[
                PhaseState(phase_id="data", title="Data"),
                PhaseState(phase_id="arch", title="Architecture"),
            ],
            dependency_graph={"data": [], "arch": [], "training": ["data", "arch"]},
        )

    def test_get_phase_found(self):
        state = self._make_state()
        assert state.get_phase("data") is not None
        assert state.get_phase("data").title == "Data"

    def test_get_phase_not_found(self):
        state = self._make_state()
        assert state.get_phase("nonexistent") is None

    def test_set_phase_status(self):
        state = self._make_state()
        state.set_phase_status("data", PhaseStatus.completed)
        assert state.get_phase("data").status == PhaseStatus.completed

    def test_set_phase_status_unknown(self):
        state = self._make_state()
        try:
            state.set_phase_status("nonexistent", PhaseStatus.completed)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_json_roundtrip(self):
        state = self._make_state()
        data = state.model_dump(mode="json")
        restored = SpecState.model_validate(data)
        assert restored.metadata.paper_id == state.metadata.paper_id
        assert len(restored.phases) == 2
        assert restored.dependency_graph == state.dependency_graph

    def test_yaml_roundtrip(self):
        state = self._make_state()
        data = state.model_dump(mode="json")
        yaml_str = yaml.dump(data, default_flow_style=False)
        loaded = yaml.safe_load(yaml_str)
        restored = SpecState.model_validate(loaded)
        assert restored.metadata.paper_title == "Test Paper"


# --- spec.py: per-section spec authoring (new) ---


class TestSectionSpec:
    def _make(self, **overrides):
        defaults = dict(
            phase_id="section_3_2",
            title="Multi-Head Attention",
            goal="Implement multi-head scaled dot-product attention.",
            spec_markdown="## Multi-Head Attention\n\nBody.",
            acceptance_criteria=[
                AcceptanceCriterion(
                    text="Forward returns (batch, seq, d_model) shape",
                    source=Citation(page=4, section="3.2", quote=None),
                )
            ],
            citations=[Citation(page=4, section="3.2")],
        )
        defaults.update(overrides)
        return SectionSpec(**defaults)

    def test_validate_citations_passes_when_all_cited(self):
        spec = self._make()
        # Should not raise.
        spec.validate_citations()

    def test_citation_requires_page(self):
        # AcceptanceCriterion's source is a Citation, and Citation.page is required.
        with pytest.raises(Exception):
            AcceptanceCriterion(
                text="missing page",
                source=Citation(section="3.2"),  # no page
            )

    def test_roundtrip_json(self):
        spec = self._make()
        data = spec.model_dump(mode="json")
        restored = SectionSpec.model_validate(data)
        assert restored.phase_id == spec.phase_id
        assert restored.acceptance_criteria[0].source.page == 4

    def test_critique_verdict_enum(self):
        critique = SectionCritique(
            phase_id="section_3_2",
            verdict=CritiqueVerdict.verified,
            reasons=["all good"],
        )
        data = critique.model_dump(mode="json")
        assert data["verdict"] == "verified"
        restored = SectionCritique.model_validate(data)
        assert restored.verdict == CritiqueVerdict.verified


# --- results.py ---


class TestTestReportModel:
    def test_empty(self):
        r = TestReport()
        assert r.tests_run == 0
        assert r.test_details == []

    def test_with_results(self):
        r = TestReport(
            tests_run=3,
            tests_passed=2,
            tests_failed=1,
            test_details=[
                TestResult(test_name="test_shapes", status=TestStatus.passed, description="Check output shapes"),
                TestResult(test_name="test_params", status=TestStatus.passed, description="Check param count"),
                TestResult(
                    test_name="test_grads",
                    status=TestStatus.failed,
                    description="Check gradient flow",
                    message="Dead layer at block 3",
                ),
            ],
        )
        assert r.tests_failed == 1
        assert r.test_details[2].message == "Dead layer at block 3"


class TestSubAgentResultModel:
    def test_success(self):
        r = SubAgentResult(
            status=ResultStatus.success,
            phase_id="arch",
            outputs=[Artifact(name="model_code", file_path="/phases/arch/1/src/model.py")],
            summary="Implemented ResNet-18",
            test_report=TestReport(tests_run=2, tests_passed=2, tests_failed=0),
        )
        assert r.is_spec_issue is False
        assert r.diagnostics is None

    def test_failure_with_spec_issue(self):
        r = SubAgentResult(
            status=ResultStatus.failure,
            phase_id="data",
            summary="Dataset URL returns 404",
            is_spec_issue=True,
            diagnostics={"url": "https://example.com/data.tar.gz", "http_status": 404},
        )
        assert r.is_spec_issue is True

    def test_json_roundtrip(self):
        r = SubAgentResult(
            status=ResultStatus.success,
            phase_id="eval",
            attempts_used=3,
        )
        data = r.model_dump(mode="json")
        restored = SubAgentResult.model_validate(data)
        assert restored.attempts_used == 3


# --- context.py ---


class TestSubSpecModel:
    def test_create(self):
        ss = SubSpec(
            phase=PhaseState(phase_id="data", title="Data Phase"),
            spec_markdown="## Data Phase\n\nDownload CIFAR-10...",
            adjacent_phases=[
                AdjacentPhaseSummary(
                    phase_id="training",
                    title="Training",
                    inputs=[Artifact(name="data_loader", file_path="TBD")],
                ),
            ],
            open_questions=["What preprocessing is applied to images?"],
            paper_path="/paper/paper.pdf",
        )
        assert len(ss.adjacent_phases) == 1
        assert ss.spec_markdown.startswith("## Data Phase")
        assert ss.open_questions[0].startswith("What")


class TestRetryContextModel:
    def test_empty(self):
        rc = RetryContext()
        assert rc.prior_results == []
        assert rc.orchestrator_feedback is None

    def test_with_prior_results(self):
        prior = SubAgentResult(status=ResultStatus.failure, phase_id="arch", summary="OOM")
        rc = RetryContext(
            prior_results=[prior],
            orchestrator_feedback="Try reducing batch size in the shape test",
        )
        assert len(rc.prior_results) == 1
        assert rc.orchestrator_feedback is not None


class TestRunStateModel:
    def test_defaults(self):
        rs = RunState()
        assert rs.status == RunStatus.running
        assert rs.orchestrator_retries == {}

    def test_tracking(self):
        rs = RunState()
        rs.orchestrator_retries["data"] = 2
        rs.phase_results["data"] = [
            SubAgentResult(status=ResultStatus.failure, phase_id="data"),
            SubAgentResult(status=ResultStatus.failure, phase_id="data"),
        ]
        assert rs.orchestrator_retries["data"] == 2
        assert len(rs.phase_results["data"]) == 2

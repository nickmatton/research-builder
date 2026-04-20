"""Data models for sub-agent results (spec_v4 §5.4, §6.1)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .spec import Artifact


class ResultStatus(str, Enum):
    success = "success"
    failure = "failure"


class TestStatus(str, Enum):
    passed = "passed"
    failed = "failed"
    error = "error"


class TestResult(BaseModel):
    """Individual test outcome."""
    test_name: str
    status: TestStatus
    description: str = ""
    message: str | None = None


class TestReport(BaseModel):
    """Aggregate test suite results (§6.1)."""
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    test_details: list[TestResult] = Field(default_factory=list)


class SubAgentResult(BaseModel):
    """Structured result returned by a sub-agent (§5.4)."""
    status: ResultStatus
    phase_id: str
    outputs: list[Artifact] = Field(default_factory=list)
    summary: str = ""
    test_report: TestReport = Field(default_factory=TestReport)
    attempts_used: int = 1
    is_spec_issue: bool = False
    diagnostics: dict[str, Any] | None = None
    cost_usd: float = 0.0

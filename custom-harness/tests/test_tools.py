"""Tests for custom MCP tools (read_paper_section, report_result)."""

import json
from pathlib import Path

import pytest

from research_builder.sub_agent.tools import (
    BUILTIN_TOOLS,
    CUSTOM_TOOL_NAMES,
    create_phase_tools,
)

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "test_paper.pdf"


class TestToolConstants:
    def test_builtin_tools(self):
        assert "Read" in BUILTIN_TOOLS
        assert "Write" in BUILTIN_TOOLS
        assert "Edit" in BUILTIN_TOOLS
        assert "Bash" in BUILTIN_TOOLS

    def test_custom_tool_names(self):
        assert "mcp__phase__lookup_citation" in CUSTOM_TOOL_NAMES
        assert "mcp__phase__report_result" in CUSTOM_TOOL_NAMES


class TestCreatePhaseTools:
    def test_creates_server(self, tmp_path):
        result_path = tmp_path / "outputs" / "_result.json"
        server = create_phase_tools(FIXTURE_PDF, result_path)
        assert server is not None

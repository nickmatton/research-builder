"""Custom MCP tools for sub-agents (spec_v4 §5.3).

Sub-agents get built-in tools from the Agent SDK (Read, Write, Edit, Bash, Glob, Grep).
These custom tools provide paper access and structured result reporting:
  - read_paper_section: Read specific pages from the paper PDF
  - report_result: Submit structured result and signal completion
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool, create_sdk_mcp_server

from ..llm.paper import extract_pages, get_page_count


def create_phase_tools(paper_path: Path, result_path: Path):
    """Create MCP tools for a sub-agent phase execution.

    Args:
        paper_path: Path to the research paper PDF.
        result_path: Path where report_result will write the structured JSON result.

    Returns:
        An MCP server config to pass to ClaudeAgentOptions.mcp_servers.
    """

    @tool(
        "read_paper_section",
        "Read specific pages from the research paper PDF. Use for targeted retrieval of sections, figures, or tables.",
        {
            "type": "object",
            "properties": {
                "start_page": {"type": "integer", "description": "First page to read (1-indexed)."},
                "end_page": {
                    "type": "integer",
                    "description": "Last page to read (1-indexed, inclusive). Omit to read a single page.",
                },
            },
            "required": ["start_page"],
        },
    )
    async def read_paper_section(args: dict[str, Any]) -> dict[str, Any]:
        start = args["start_page"]
        end = args.get("end_page")
        try:
            text = extract_pages(paper_path, start, end)
            total = get_page_count(paper_path)
            header = f"[Paper: pages {start}-{end or start} of {total}]\n\n"
            return {"content": [{"type": "text", "text": header + text}]}
        except FileNotFoundError:
            return {"content": [{"type": "text", "text": f"Error: paper not found at {paper_path}"}], "is_error": True}
        except ValueError as e:
            return {"content": [{"type": "text", "text": f"Error: {e}"}], "is_error": True}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Error reading paper: {e}"}], "is_error": True}

    @tool(
        "report_result",
        "Submit your final result and exit. Call this when done — either all tests pass (status=success) or you cannot make further progress (status=failure).",
        {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["success", "failure"],
                    "description": "Whether the phase completed successfully.",
                },
                "summary": {
                    "type": "string",
                    "description": "Plain-language description of what was done.",
                },
                "outputs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "file_path": {"type": "string"},
                        },
                        "required": ["name", "file_path"],
                    },
                    "description": "List of output artifacts produced.",
                },
                "test_report": {
                    "type": "object",
                    "properties": {
                        "tests_run": {"type": "integer"},
                        "tests_passed": {"type": "integer"},
                        "tests_failed": {"type": "integer"},
                        "test_details": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "test_name": {"type": "string"},
                                    "status": {"type": "string", "enum": ["passed", "failed", "error"]},
                                    "description": {"type": "string"},
                                    "message": {"type": "string"},
                                },
                                "required": ["test_name", "status"],
                            },
                        },
                    },
                    "description": "Results of your test suite.",
                },
                "is_spec_issue": {
                    "type": "boolean",
                    "description": "True if the failure stems from a spec problem, not an implementation bug.",
                },
                "diagnostics": {
                    "type": "object",
                    "description": "On failure: error traces, logs, analysis.",
                },
                "attempts_used": {
                    "type": "integer",
                    "description": "How many debug iterations you used.",
                },
            },
            "required": ["status", "summary"],
        },
    )
    async def report_result(args: dict[str, Any]) -> dict[str, Any]:
        # Write the structured result to a file for the parent to read
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(args, indent=2))
        return {"content": [{"type": "text", "text": f"Result recorded to {result_path}. You may stop now."}]}

    return create_sdk_mcp_server(
        name="phase_tools",
        version="1.0.0",
        tools=[read_paper_section, report_result],
    )


# Built-in tools that sub-agents get from the Agent SDK
BUILTIN_TOOLS = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]

# Custom tool names (prefixed with MCP server name)
CUSTOM_TOOL_NAMES = [
    "mcp__phase_tools__read_paper_section",
    "mcp__phase_tools__report_result",
]

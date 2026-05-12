"""Custom MCP tools for sub-agents.

Sub-agents get the standard built-in tools (Read, Write, Edit, Bash, Glob, Grep)
from the Agent SDK — including native PDF reading via Read with the ``pages``
parameter, which fully replaces the old ``read_paper_section`` and
``search_paper`` MCP tools.

Custom tools provided here:

  - lookup_citation: Look up a cited paper via Semantic Scholar.
  - request_compute: (only when a remote GPU is provisioned) request a bigger
    instance type and/or extend the runtime budget for this phase. The harness
    gates the request against the per-run GPU spend cap and may bubble to the
    operator for approval.
  - report_result: Submit structured result and signal completion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from claude_agent_sdk import tool, create_sdk_mcp_server

from ..literature.scholar import SemanticScholarClient


# Async callable signature for the request_compute upgrade hook.
# Returns (success, message). Sub-agent surfaces the message back to the model.
ComputeUpgradeFn = Callable[[str | None, float | None, str], Awaitable[tuple[bool, str]]]


def create_phase_tools(
    paper_path: Path,
    result_path: Path,
    compute_upgrade: ComputeUpgradeFn | None = None,
    spec_dir: Path | None = None,
):
    """Create MCP tools for a sub-agent phase execution.

    Args:
        paper_path: Path to the research paper PDF. Sub-agents read it via
            the built-in Read tool with ``pages="N-M"``; not used here.
        result_path: Path where report_result will write the structured JSON result.
        compute_upgrade: If provided, a remote GPU has been provisioned and the
            sub-agent gets a ``request_compute`` tool that calls this hook to
            swap the machine / extend the runtime allocation. None means no GPU
            provisioned for this phase, so the tool is omitted.
        spec_dir: (unused now; kept for backward-compat with the loop's call site)

    Returns:
        An MCP server config to pass to ClaudeAgentOptions.mcp_servers.
    """

    @tool(
        "lookup_citation",
        "Look up a cited paper by title or partial citation string. Returns the paper's title, "
        "abstract, year, venue, and link via Semantic Scholar. Use this when the spec references "
        "a method, dataset, or technique from another paper and you need implementation details.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Paper title or citation string to search for (e.g. 'Attention Is All You Need').",
                },
            },
            "required": ["query"],
        },
    )
    async def lookup_citation(args: dict[str, Any]) -> dict[str, Any]:
        query_str = args["query"]
        try:
            client = SemanticScholarClient()
            results = await client.search_by_title(query_str, limit=3)
            if not results:
                return {"content": [{"type": "text", "text": f"No results found for: {query_str}"}]}
            text = "\n\n---\n\n".join(r.to_markdown() for r in results)
            return {"content": [{"type": "text", "text": text}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Citation lookup failed: {e}"}], "is_error": True}

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

    tools_list = [lookup_citation, report_result]

    if compute_upgrade is not None:
        @tool(
            "request_compute",
            "Request a bigger GPU instance and/or extend the time budget for this "
            "phase. Use this if the currently provisioned machine is too small "
            "(OOM, too slow) or you need more wall-clock to finish training. The "
            "harness will swap the machine in place and rewrite .cloud/env; you "
            "then re-invoke `bash remote_run.sh \"...\"` to use the new box. "
            "Every upgrade is debited against a per-run GPU spend cap and may "
            "require operator approval — only request what you genuinely need, "
            "and explain why in `reason`.",
            {
                "type": "object",
                "properties": {
                    "instance_type": {
                        "type": "string",
                        "description": "Lambda Cloud instance type to switch to "
                        "(e.g. gpu_1x_a100, gpu_8x_a100, gpu_1x_h100). Omit to "
                        "keep the current machine and only extend hours.",
                    },
                    "additional_hours": {
                        "type": "number",
                        "description": "Extra wall-clock hours to add to the "
                        "phase's runtime budget. Omit to only swap instance "
                        "type without extending.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One-sentence justification for the upgrade "
                        "(e.g. 'OOM at batch_size=32 on a10', 'training projected "
                        "to take 4hrs not 2hrs based on first 100 steps'). The "
                        "operator sees this when approving.",
                    },
                },
                "required": ["reason"],
            },
        )
        async def request_compute(args: dict[str, Any]) -> dict[str, Any]:
            instance_type = args.get("instance_type")
            additional_hours = args.get("additional_hours")
            reason = str(args.get("reason", ""))
            try:
                ok, msg = await compute_upgrade(instance_type, additional_hours, reason)
            except Exception as e:
                return {
                    "content": [{"type": "text", "text": f"request_compute failed: {type(e).__name__}: {e}"}],
                    "is_error": True,
                }
            return {
                "content": [{"type": "text", "text": msg}],
                **({"is_error": True} if not ok else {}),
            }

        tools_list.append(request_compute)

    return create_sdk_mcp_server(
        name="phase_tools",
        version="1.0.0",
        tools=tools_list,
    )


# Built-in tools that sub-agents get from the Agent SDK. Read supports PDFs
# natively (with the ``pages`` parameter), so it doubles as the paper-reading
# tool — no custom MCP tool needed for that anymore.
BUILTIN_TOOLS = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]

# Custom tool names (prefixed with MCP server name). request_compute is added
# dynamically when a remote GPU is provisioned.
CUSTOM_TOOL_NAMES = [
    "mcp__phase_tools__lookup_citation",
    "mcp__phase_tools__report_result",
]
COMPUTE_TOOL_NAME = "mcp__phase_tools__request_compute"

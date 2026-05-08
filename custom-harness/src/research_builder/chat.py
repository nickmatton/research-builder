"""Shared chat query logic used by both legacy console and TUI chat pane."""

from __future__ import annotations

import logging
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)

logger = logging.getLogger(__name__)

_CHAT_SYSTEM_PROMPT = """\
You are a research assistant helping a user understand a research paper and \
refine an implementation spec. You have access to:

1. The research paper (via the read_paper_section tool — specify page numbers)
2. The implementation spec at: {spec_path}
3. Standard file tools (Read, Edit, Glob, Grep) to view and modify the spec

When the user asks about the paper, read the relevant sections and answer.
When the user asks you to change the spec, edit the spec.md file directly.
Keep answers concise and direct.
"""


async def chat_query(
    conversation: str,
    spec_path: Path,
    model: str,
    paper_tools: dict,
    on_tool: callable | None = None,
    on_token: callable | None = None,
) -> str:
    """Send a chat message to the Claude agent and return the response.

    Args:
        conversation: The full conversation text to send.
        spec_path: Path to the spec.md file.
        model: Model name to use.
        paper_tools: MCP server config for paper tools.
        on_tool: Optional callback for tool use display (called with tool name).
        on_token: Optional callback fired with each TextBlock as it streams in.
    """
    system = _CHAT_SYSTEM_PROMPT.format(spec_path=spec_path)

    options = ClaudeAgentOptions(
        system_prompt=system,
        model=model,
        permission_mode="bypassPermissions",
        cwd=str(spec_path.parent),
        allowed_tools=[
            "Read", "Edit", "Glob", "Grep",
            "mcp__paper_tools__read_paper_section",
        ],
        mcp_servers={"paper_tools": paper_tools},
        max_turns=10,
    )

    result_text = ""
    async for message in query(prompt=conversation, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    if on_token:
                        on_token(block.text)
                    result_text += block.text
                elif isinstance(block, ToolUseBlock):
                    tool_name = block.name.replace("mcp__paper_tools__", "")
                    if on_tool:
                        on_tool(tool_name)
        elif isinstance(message, ResultMessage):
            if message.result:
                result_text = message.result

    return result_text.strip() or "(No response)"

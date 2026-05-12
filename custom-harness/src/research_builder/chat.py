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

1. The research paper at: {paper_path} (use the **Read** tool with the \
``pages`` parameter — Read supports PDFs natively and preserves tables, \
figures, equations).
2. The implementation spec at: {spec_path} (use Read / Edit on this file).
3. Standard file tools (Glob, Grep) to navigate the workspace.

When the user asks about the paper, Read the relevant pages and answer.
When the user asks you to change the spec, Edit the spec file directly.
Keep answers concise and direct.
"""


async def chat_query(
    conversation: str,
    spec_path: Path,
    model: str,
    paper_path: Path,
    on_tool: callable | None = None,
    on_token: callable | None = None,
) -> str:
    """Send a chat message to the Claude agent and return the response.

    Args:
        conversation: The full conversation text to send.
        spec_path: Path to the spec.md file.
        model: Model name to use.
        paper_path: Path to the paper PDF (read directly via the Read tool).
        on_tool: Optional callback for tool use display (called with tool name).
        on_token: Optional callback fired with each TextBlock as it streams in.
    """
    system = _CHAT_SYSTEM_PROMPT.format(
        spec_path=spec_path, paper_path=Path(paper_path).resolve(),
    )

    options = ClaudeAgentOptions(
        system_prompt=system,
        model=model,
        permission_mode="bypassPermissions",
        cwd=str(spec_path.parent),
        allowed_tools=["Read", "Edit", "Glob", "Grep"],
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
                    if on_tool:
                        on_tool(block.name)
        elif isinstance(message, ResultMessage):
            if message.result:
                result_text = message.result

    return result_text.strip() or "(No response)"

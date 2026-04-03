"""Async LLM client wrapper for the Anthropic API."""

from __future__ import annotations

import json
import logging
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam, ToolParam, ToolResultBlockParam

from ..config import Config

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around AsyncAnthropic for tool-use conversations."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = AsyncAnthropic()
        self.model = config.model

    async def create_message(
        self,
        *,
        messages: list[MessageParam],
        system: str = "",
        tools: list[ToolParam] | None = None,
        max_tokens: int = 16384,
        tool_choice: dict[str, Any] | None = None,
    ) -> Message:
        """Single API call. Returns the raw Message response."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        response = await self.client.messages.create(**kwargs)
        logger.debug(
            "LLM response: stop_reason=%s, usage=%s",
            response.stop_reason,
            response.usage,
        )
        return response

    async def run_tool_loop(
        self,
        *,
        messages: list[MessageParam],
        system: str = "",
        tools: list[ToolParam],
        execute_tool: Any,  # async callable(name, input) -> str
        max_iterations: int = 50,
        max_tokens: int = 16384,
    ) -> tuple[Message, list[MessageParam]]:
        """Run a tool-use agentic loop until the model stops calling tools.

        Args:
            messages: Initial conversation messages.
            system: System prompt.
            tools: Tool definitions.
            execute_tool: Async callable(name: str, input: dict) -> str
                          that executes a tool and returns the result string.
                          Should raise ToolExit to signal loop termination with
                          a structured result.
            max_iterations: Safety limit on loop iterations.
            max_tokens: Max tokens per API call.

        Returns:
            Tuple of (final Message, full message history).
        """
        history = list(messages)

        for iteration in range(max_iterations):
            response = await self.create_message(
                messages=history,
                system=system,
                tools=tools,
                max_tokens=max_tokens,
            )

            if response.stop_reason != "tool_use":
                return response, history

            # Add assistant response to history
            history.append({"role": "assistant", "content": response.content})

            # Execute each tool call
            tool_results: list[ToolResultBlockParam] = []
            exit_result = None

            for block in response.content:
                if block.type != "tool_use":
                    continue

                logger.info("Tool call: %s(%s)", block.name, json.dumps(block.input)[:200])

                try:
                    result = await execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })
                except ToolExit as e:
                    # Sub-agent signaled completion via report_result
                    exit_result = e.result
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Result recorded. Loop terminating.",
                    })

            # Add tool results to history
            history.append({"role": "user", "content": tool_results})

            if exit_result is not None:
                return response, history

        logger.warning("Tool loop hit max_iterations=%d", max_iterations)
        return response, history


class ToolExit(Exception):
    """Raised by a tool executor to signal that the agent loop should stop.

    The `result` attribute carries the structured data (e.g., SubAgentResult dict)
    that the tool returned before exiting.
    """

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        super().__init__("Tool loop exit")

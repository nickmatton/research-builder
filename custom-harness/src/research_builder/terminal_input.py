"""Bottom-anchored chat input for the terminal UI.

Runs a prompt_toolkit ``PromptSession`` in an async loop, with
``patch_stdout()`` wrapping the whole pipeline lifetime so event prints
from the InlineViewer flow *above* the prompt without scrambling it.

Each submitted line is delivered to ``inbox.deliver("orchestrator", text)``
— the exact same path the web chat uses. The orchestrator interprets
the text via its ``request_user_approval`` MCP tool; there is no
separate command parser.
"""

from __future__ import annotations

import asyncio
import logging

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML

logger = logging.getLogger(__name__)


# Matches the indigo accent used by the web UI (--color-accent).
_PROMPT_HTML = HTML('<style fg="#6366f1" bold="true">&gt;</style> ')


async def run_terminal_input(inbox) -> None:
    """Loop forever, reading lines and delivering them to the orchestrator.

    Returns on EOF, KeyboardInterrupt, or task cancellation.
    """
    session: PromptSession[str] = PromptSession()
    while True:
        try:
            text = await session.prompt_async(_PROMPT_HTML)
        except (EOFError, KeyboardInterrupt):
            return
        except asyncio.CancelledError:
            return

        text = (text or "").strip()
        if not text:
            continue

        try:
            await inbox.deliver("orchestrator", text)
        except Exception as e:
            # Don't let a delivery error tear down the input loop — the
            # user can just retry. Log it for forensics.
            logger.warning("terminal_input: delivery failed: %s", e)

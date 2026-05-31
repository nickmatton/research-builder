"""Inbound command channel: lets external clients (e.g. agent-terminal) send
chat messages and other commands to the running pipeline."""

from .client import (
    append_command,
    edit_refined_spec,
    force_retry,
    inject_note,
    jump_back,
    make_command,
)
from .inbox import Inbox, get_inbox
from .listener import CommandListener

__all__ = [
    "Inbox",
    "get_inbox",
    "CommandListener",
    "append_command",
    "make_command",
    "edit_refined_spec",
    "force_retry",
    "inject_note",
    "jump_back",
]

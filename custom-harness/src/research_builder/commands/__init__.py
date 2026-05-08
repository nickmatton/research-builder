"""Inbound command channel: lets external clients (e.g. agent-terminal) send
chat messages and other commands to the running pipeline."""

from .inbox import Inbox, get_inbox
from .listener import CommandListener

__all__ = ["Inbox", "get_inbox", "CommandListener"]

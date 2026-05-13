"""Minimal terminal viewer for a running research-builder harness.

Tails events.jsonl and renders the agent activity as a Claude-Code-style
scrolling transcript (rich.live, no Textual). One file, one widget, no panes.
"""

from .live_viewer import LiveViewer, run_viewer

__all__ = ["LiveViewer", "run_viewer"]

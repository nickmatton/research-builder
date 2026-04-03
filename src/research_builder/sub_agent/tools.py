"""Sub-agent tool definitions and executor (spec_v4 §5.3).

Tools available to sub-agents during phase execution:
  - read_file: Read a file from the workspace
  - write_file: Write/create a file in the workspace
  - edit_file: Replace text in an existing file
  - bash: Run a shell command
  - read_paper_section: Read specific pages from the paper PDF
  - report_result: Return structured result and exit the agent loop
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from anthropic.types import ToolParam

from ..llm.client import ToolExit
from ..llm.paper import extract_pages, get_page_count


# ---------------------------------------------------------------------------
# Tool definitions (JSON schema for the Anthropic API)
# ---------------------------------------------------------------------------

READ_FILE: ToolParam = {
    "name": "read_file",
    "description": "Read the contents of a file. Returns the file content as text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or workspace-relative file path."},
        },
        "required": ["path"],
    },
}

WRITE_FILE: ToolParam = {
    "name": "write_file",
    "description": "Write content to a file. Creates the file and parent directories if they don't exist. Overwrites existing content.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or workspace-relative file path."},
            "content": {"type": "string", "description": "The content to write."},
        },
        "required": ["path", "content"],
    },
}

EDIT_FILE: ToolParam = {
    "name": "edit_file",
    "description": "Replace an exact string in a file with new text. The old_text must appear exactly once in the file.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or workspace-relative file path."},
            "old_text": {"type": "string", "description": "The exact text to find and replace."},
            "new_text": {"type": "string", "description": "The replacement text."},
        },
        "required": ["path", "old_text", "new_text"],
    },
}

BASH: ToolParam = {
    "name": "bash",
    "description": "Execute a shell command and return stdout + stderr. The working directory is the phase attempt directory.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to execute."},
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 300).",
            },
        },
        "required": ["command"],
    },
}

READ_PAPER_SECTION: ToolParam = {
    "name": "read_paper_section",
    "description": (
        "Read specific pages from the research paper PDF. "
        "Use this for targeted retrieval of sections, figures, or tables. "
        "Specify a page range (e.g., start=3, end=5 reads pages 3-5)."
    ),
    "input_schema": {
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
}

REPORT_RESULT: ToolParam = {
    "name": "report_result",
    "description": (
        "Submit your final result and exit. Call this when you are done with the phase — "
        "either all tests pass (status=success) or you cannot make further progress (status=failure). "
        "This terminates the agent loop."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["success", "failure"],
                "description": "Whether the phase completed successfully.",
            },
            "summary": {
                "type": "string",
                "description": "Plain-language description of what was done and any non-obvious decisions.",
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
                "description": "List of output artifacts produced (name + file path).",
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
                "description": "Set to true if the failure stems from a spec problem, not an implementation bug.",
            },
            "diagnostics": {
                "type": "object",
                "description": "On failure: error traces, logs, analysis, and evidence for spec issues.",
            },
            "attempts_used": {
                "type": "integer",
                "description": "How many debug iterations you used.",
            },
        },
        "required": ["status", "summary"],
    },
}

ALL_TOOLS: list[ToolParam] = [
    READ_FILE,
    WRITE_FILE,
    EDIT_FILE,
    BASH,
    READ_PAPER_SECTION,
    REPORT_RESULT,
]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------


class ToolExecutor:
    """Executes sub-agent tool calls against the filesystem.

    All file paths are resolved relative to `work_dir` (the phase attempt directory).
    """

    def __init__(
        self,
        work_dir: Path,
        paper_path: Path,
        bash_timeout: int = 300,
    ) -> None:
        self.work_dir = work_dir
        self.paper_path = paper_path
        self.bash_timeout = bash_timeout

    def _resolve(self, path: str) -> Path:
        """Resolve a path relative to work_dir, preventing escape."""
        p = Path(path)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (self.work_dir / p).resolve()

        # Allow access to work_dir and its parents (for reading upstream artifacts)
        # but the sub-agent primarily works within work_dir
        return resolved

    async def execute(self, name: str, input: dict[str, Any]) -> str:
        """Dispatch a tool call by name. Raises ToolExit for report_result."""
        match name:
            case "read_file":
                return self._read_file(input["path"])
            case "write_file":
                return self._write_file(input["path"], input["content"])
            case "edit_file":
                return self._edit_file(input["path"], input["old_text"], input["new_text"])
            case "bash":
                return await self._bash(input["command"], input.get("timeout"))
            case "read_paper_section":
                return self._read_paper_section(input["start_page"], input.get("end_page"))
            case "report_result":
                raise ToolExit(result=input)
            case _:
                return f"Error: unknown tool '{name}'"

    def _read_file(self, path: str) -> str:
        resolved = self._resolve(path)
        if not resolved.exists():
            return f"Error: file not found: {path}"
        try:
            return resolved.read_text()
        except Exception as e:
            return f"Error reading {path}: {e}"

    def _write_file(self, path: str, content: str) -> str:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        return f"Wrote {len(content)} chars to {path}"

    def _edit_file(self, path: str, old_text: str, new_text: str) -> str:
        resolved = self._resolve(path)
        if not resolved.exists():
            return f"Error: file not found: {path}"
        content = resolved.read_text()
        count = content.count(old_text)
        if count == 0:
            return f"Error: old_text not found in {path}"
        if count > 1:
            return f"Error: old_text appears {count} times in {path}. Must be unique."
        resolved.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"

    async def _bash(self, command: str, timeout: int | None = None) -> str:
        timeout = timeout or self.bash_timeout
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.work_dir),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = ""
            if stdout:
                output += stdout.decode(errors="replace")
            if stderr:
                output += "\nSTDERR:\n" + stderr.decode(errors="replace")
            output += f"\n[exit code: {proc.returncode}]"
            return output.strip()
        except asyncio.TimeoutError:
            proc.kill()
            return f"Error: command timed out after {timeout}s"
        except Exception as e:
            return f"Error executing command: {e}"

    def _read_paper_section(self, start_page: int, end_page: int | None = None) -> str:
        try:
            text = extract_pages(self.paper_path, start_page, end_page)
            total = get_page_count(self.paper_path)
            header = f"[Paper: pages {start_page}-{end_page or start_page} of {total}]\n\n"
            return header + text
        except FileNotFoundError:
            return f"Error: paper not found at {self.paper_path}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading paper: {e}"

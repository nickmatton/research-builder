"""Data models for the in-memory agent tree the viewer renders."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

FileStatus = Literal["planned", "in_progress", "created"]
AgentStatus = Literal["pending", "running", "completed", "failed"]
AgentKind = Literal["orchestrator", "subagent"]
ProcessStatus = Literal["running", "completed", "errored"]


class FileEntry(BaseModel):
    path: str
    status: FileStatus = "planned"
    role: str | None = None
    file_id: str | None = None


class ChatMessage(BaseModel):
    role: str
    text: str
    ts: datetime = Field(default_factory=datetime.now)


class ActivityLine(BaseModel):
    kind: Literal["thinking", "tool", "done", "info"]
    text: str
    ts: datetime = Field(default_factory=datetime.now)


class ProcessEntry(BaseModel):
    process_id: str
    agent_id: str
    tool_name: str
    summary: str
    command: str | None = None
    file_path: str | None = None
    compute_ip: str | None = None
    status: ProcessStatus = "running"
    output: str | None = None
    is_error: bool = False
    ts: datetime = Field(default_factory=datetime.now)


class ComputeEntry(BaseModel):
    instance_id: str
    instance_type: str
    public_ip: str
    agent_id: str
    status: Literal["active", "terminated"] = "active"
    ts: datetime = Field(default_factory=datetime.now)


class AgentNode(BaseModel):
    id: str
    parent_id: str | None = None
    title: str = ""
    kind: AgentKind = "subagent"
    status: AgentStatus = "pending"
    chat: list[ChatMessage] = Field(default_factory=list)
    activity: list[ActivityLine] = Field(default_factory=list)
    reasoning_buffer: str = ""
    files: list[FileEntry] = Field(default_factory=list)
    processes: list[ProcessEntry] = Field(default_factory=list)
    children_ids: list[str] = Field(default_factory=list)
    attempt: int = 0

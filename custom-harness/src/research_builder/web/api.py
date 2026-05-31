"""REST endpoints.

The router closes over a ``WebState`` rather than a fixed workspace path,
so the active workspace can change at runtime (``POST /api/launch``
uploads a paper, creates a workspace, and spawns the pipeline).

All filesystem reads are sandboxed under the active workspace via
``_safe_join`` — paths that resolve outside the workspace return 400.

The command-write endpoints append JSON lines to ``logs/commands.jsonl``
using the harness's existing ``commands.client`` helpers (so the running
pipeline's CommandListener picks them up unchanged).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from .state import WebState, paper_stem


# Directories we hide from the file tree. The harness's runs can produce
# `__pycache__` / `.venv` / `node_modules` if a sub-agent installs deps;
# burying them keeps the tree useful.
_HIDDEN = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache", ".mypy_cache"}

# Cap individual text reads. Larger files get truncated with a marker.
_TEXT_READ_LIMIT_BYTES = 512 * 1024  # 512 KB

# Skip binary file types in the text reader.
_BINARY_SUFFIXES = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".pyc", ".pyo", ".so", ".dylib", ".o", ".a",
    ".pt", ".pth", ".bin", ".onnx", ".safetensors",
    ".mp3", ".mp4", ".mov", ".wav", ".flac",
}


def build_api_router(state: WebState) -> APIRouter:
    router = APIRouter(prefix="/api")

    def _require_workspace() -> Path:
        if state.workspace is None:
            raise HTTPException(409, "no active workspace — upload a paper via /api/launch")
        return state.workspace

    def _safe_join(rel: str) -> Path:
        """Resolve ``rel`` under the active workspace; reject paths that escape."""
        workspace = _require_workspace()
        if rel in ("", "."):
            return workspace
        candidate = (workspace / rel).resolve()
        try:
            candidate.relative_to(workspace)
        except ValueError:
            raise HTTPException(400, f"path escapes workspace: {rel}")
        return candidate

    # ─── Workspace ────────────────────────────────────────────────────────

    @router.get("/workspace")
    def workspace_info() -> dict[str, Any]:
        """Return the active workspace, or an ``empty`` sentinel.

        ``state`` reflects the launcher → ready → running → finished
        lifecycle so the frontend can decide which screen to render:

        - ``empty``     no paper uploaded yet → show Launcher.
        - ``ready``     paper present, no pipeline started → show main UI.
        - ``running``   pipeline subprocess is alive.
        - ``finished``  subprocess exited (clean or not).
        """
        if state.workspace is None:
            return {
                "name": None,
                "path": None,
                "paper_path": None,
                "has_spec": False,
                "state": "empty",
                "runs_dir": str(state.runs_dir),
                "events_path": "logs/events.jsonl",
                "commands_path": "logs/commands.jsonl",
                "pipeline": state.pipeline_status(),
                "dev_mode_default": state.dev_mode,
            }
        workspace = state.workspace
        paper = _find_paper(workspace)
        state_path = workspace / "canonical_spec" / "state.json"
        pipeline = state.pipeline_status()
        ui_state = "ready"
        if pipeline["state"] == "running":
            ui_state = "running"
        elif pipeline["state"] == "finished":
            ui_state = "finished"
        return {
            "name": workspace.name,
            "path": str(workspace),
            "paper_path": str(paper.relative_to(workspace)) if paper else None,
            "has_spec": state_path.exists(),
            "state": ui_state,
            "runs_dir": str(state.runs_dir),
            "events_path": "logs/events.jsonl",
            "commands_path": "logs/commands.jsonl",
            "pipeline": pipeline,
            "dev_mode_default": state.dev_mode,
        }

    # ─── Launch: upload PDF + spawn pipeline ──────────────────────────────

    @router.post("/launch")
    async def launch(
        paper: UploadFile = File(...),
        name: str | None = Form(default=None),
        skip_gates: bool = Form(default=False),
        on_conflict: str = Form(default="fail"),
        dev_mode: bool | None = Form(default=None),
    ) -> dict[str, Any]:
        """Upload a PDF, scaffold a workspace, and kick off the pipeline.

        Body (multipart):
          - ``paper``: PDF file (required)
          - ``name``: optional override for the workspace directory name
          - ``skip_gates``: pass --auto so the pipeline runs without per-phase
            chat approval gates (default: False — gates fire by default).
          - ``on_conflict``: how to handle an existing workspace at the same
            name. One of:
              - ``"fail"`` (default) — 409 with structured detail so the UI
                can prompt the user to pick wipe/archive/resume.
              - ``"wipe"`` — delete harness-managed dirs (canonical_spec,
                phases, report, traces, context, notes, and logs/* contents),
                then start fresh. Preserves ``paper/`` and any user files.
              - ``"archive"`` — move harness-managed dirs to
                ``.archive/<timestamp>/`` and start fresh.
              - ``"resume"`` — keep everything, spawn with ``--resume`` so the
                harness picks up where it left off. Doesn't overwrite the
                existing PDF if there is one.

        Side effects:
          1. Creates / cleans / reuses ``<runs_dir>/<stem>/`` per on_conflict.
          2. Sets ``state.workspace`` to that dir.
          3. Spawns ``research-builder -o <workspace>`` (optionally with
             ``--auto``, ``--resume``, ``--dev``) as a subprocess; stdout/stderr
             → ``logs/pipeline.log``.
        """
        if state.pipeline_running():
            raise HTTPException(409, "a pipeline is already running — stop it first")

        if on_conflict not in ("fail", "wipe", "archive", "resume"):
            raise HTTPException(
                400, f"invalid on_conflict={on_conflict!r}; "
                "must be fail|wipe|archive|resume",
            )

        filename = paper.filename or "paper.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(400, "only PDF uploads are supported")

        stem = paper_stem(name or filename)
        workspace = (state.runs_dir / stem).resolve()
        try:
            workspace.relative_to(state.runs_dir)
        except ValueError:
            raise HTTPException(400, f"invalid workspace name: {stem}")

        # Handle existing-workspace conflicts.
        workspace_exists = workspace.exists() and any(workspace.iterdir())
        if workspace_exists:
            state_path = workspace / "canonical_spec" / "state.json"
            existing_paper = next(
                iter(_find_paper_candidates(workspace / "paper")), None,
            )
            if on_conflict == "fail":
                # Structured detail so the frontend can render a choice modal
                # instead of a generic error string.
                raise HTTPException(
                    409,
                    detail={
                        "code": "workspace_exists",
                        "workspace": str(workspace),
                        "name": workspace.name,
                        "has_state": state_path.exists(),
                        "has_paper": existing_paper is not None,
                        "existing_paper": (
                            str(existing_paper.relative_to(workspace))
                            if existing_paper else None
                        ),
                        "message": (
                            f"Workspace '{workspace.name}' already exists. "
                            f"Choose wipe, archive, or resume."
                        ),
                    },
                )
            if on_conflict == "wipe":
                from ..resume import wipe as wipe_workspace
                wipe_workspace(workspace)
            elif on_conflict == "archive":
                from ..resume import archive_and_clear
                archive_and_clear(workspace)
            # on_conflict == "resume" falls through — keep everything as-is.

        paper_dir = workspace / "paper"
        paper_dir.mkdir(parents=True, exist_ok=True)
        dest = paper_dir / filename

        # On resume, prefer the existing PDF on disk. If the uploaded filename
        # matches what's there, just leave it. Otherwise write the new upload
        # alongside (the harness uses the first PDF in paper/).
        if on_conflict == "resume" and dest.exists() and dest.stat().st_size > 0:
            # Keep existing PDF; ignore the upload body to avoid clobbering.
            pass
        else:
            with dest.open("wb") as f:
                shutil.copyfileobj(paper.file, f)

        state.set_workspace(workspace)

        log_dir = workspace / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        pipeline_log = log_dir / "pipeline.log"
        log_file = pipeline_log.open("a", buffering=1)

        # Reuse the same Python interpreter the app is running under so
        # the subprocess sees the same installed harness, no PATH games.
        cmd = [
            sys.executable, "-m", "research_builder.main",
            str(dest),
            "--output", str(workspace),
        ]
        if on_conflict == "resume":
            cmd.append("--resume")
        if skip_gates:
            # --auto skips the chat-driven approval gates (skeleton, section
            # specs, before/after each phase). GPU/cost gates stay regardless.
            cmd.append("--auto")
        # Per-launch dev_mode form value wins. If the caller didn't pass
        # it, fall back to the process-wide ``--dev`` boot flag. Either
        # path appends --dev so the subprocess routes through the
        # Claude Code subscription instead of needing ANTHROPIC_API_KEY.
        use_dev = dev_mode if dev_mode is not None else state.dev_mode
        if use_dev:
            cmd.append("--dev")
        # Forward each --allow-dir from boot flags so the spawned pipeline's
        # agent sandbox sees the same extra dirs. Skipped if the workspace
        # itself happens to be one of them (it's already cwd).
        for d in state.extra_allowed_dirs:
            cmd.extend(["--allow-dir", str(d)])
        env = os.environ.copy()
        # Make sure the subprocess writes to the workspace's logs (not
        # whatever cwd we boot from).
        env["RESEARCH_BUILDER_EVENT_LOG"] = str(log_dir / "events.jsonl")
        env["RESEARCH_BUILDER_COMMAND_LOG"] = str(log_dir / "commands.jsonl")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(workspace),
            env=env,
            # Detach into its own process group so a Ctrl-C in the app's
            # terminal doesn't kill the pipeline through signal forwarding;
            # we manage shutdown explicitly via WebState.terminate.
            start_new_session=True,
        )
        state.set_proc(proc, pipeline_log)

        return {
            "ok": True,
            "workspace": str(workspace),
            "name": workspace.name,
            "paper_path": str(dest.relative_to(workspace)),
            "pid": proc.pid,
            "log": str(pipeline_log.relative_to(workspace)),
        }

    @router.get("/pipeline/status")
    def pipeline_status() -> dict[str, Any]:
        return state.pipeline_status()

    @router.post("/pipeline/stop")
    def pipeline_stop() -> dict[str, Any]:
        if not state.pipeline_running():
            return {"ok": True, "was_running": False}
        state.terminate()
        return {"ok": True, "was_running": True}

    # ─── Spec ─────────────────────────────────────────────────────────────

    @router.get("/spec")
    def spec() -> dict[str, Any]:
        """Return the canonical spec: parsed state.json + raw spec.md."""
        workspace = _require_workspace()
        state_path = workspace / "canonical_spec" / "state.json"
        spec_md_path = workspace / "canonical_spec" / "spec.md"
        state_data: dict[str, Any] | None = None
        if state_path.exists():
            try:
                state_data = json.loads(state_path.read_text() or "{}")
            except json.JSONDecodeError as e:
                raise HTTPException(500, f"failed to parse state.json: {e}")
        spec_md = spec_md_path.read_text() if spec_md_path.exists() else None
        return {"state": state_data, "spec_md": spec_md}

    # ─── Phases + attempts ───────────────────────────────────────────────

    @router.get("/phases")
    def phases() -> dict[str, Any]:
        """Phase tree with attempt manifests inlined."""
        workspace = _require_workspace()
        state_path = workspace / "canonical_spec" / "state.json"
        if not state_path.exists():
            return {"phases": []}
        try:
            spec_state = json.loads(state_path.read_text() or "{}")
        except json.JSONDecodeError as e:
            raise HTTPException(500, f"failed to parse state.json: {e}")

        deps: dict[str, list[str]] = spec_state.get("dependency_graph", {}) or {}
        out: list[dict[str, Any]] = []
        for p in spec_state.get("phases", []) or []:
            phase_id = p.get("phase_id", "")
            attempts = _load_attempts(workspace, phase_id)
            out.append({
                "phase_id": phase_id,
                "title": p.get("title") or phase_id,
                "status": p.get("status", "pending"),
                "dependencies": deps.get(phase_id, []),
                "inputs": p.get("inputs") or [],
                "outputs": p.get("outputs") or [],
                "attempts": attempts,
            })
        return {"phases": out}

    # ─── File tree (one directory at a time, lazy) ───────────────────────

    @router.get("/files")
    def files(path: str = Query("", description="Workspace-relative dir")) -> dict[str, Any]:
        workspace = _require_workspace()
        root = _safe_join(path)
        if not root.exists():
            raise HTTPException(404, f"not found: {path}")
        if not root.is_dir():
            raise HTTPException(400, f"not a directory: {path}")

        entries: list[dict[str, Any]] = []
        try:
            children = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            raise HTTPException(403, f"permission denied: {path}")

        for child in children:
            if child.name in _HIDDEN or child.name.startswith("."):
                continue
            try:
                size = child.stat().st_size if child.is_file() else None
            except OSError:
                size = None
            entries.append({
                "name": child.name,
                "path": str(child.relative_to(workspace)),
                "is_dir": child.is_dir(),
                "size": size,
            })
        return {
            "path": str(root.relative_to(workspace)) if root != workspace else "",
            "entries": entries,
        }

    @router.get("/file", response_class=PlainTextResponse)
    def file_text(path: str = Query(..., description="Workspace-relative file")) -> PlainTextResponse:
        f = _safe_join(path)
        if not f.exists() or not f.is_file():
            raise HTTPException(404, f"not found: {path}")
        if f.suffix.lower() in _BINARY_SUFFIXES:
            raise HTTPException(415, f"binary file (use /api/binary): {path}")
        data = f.read_bytes()
        truncated = False
        if len(data) > _TEXT_READ_LIMIT_BYTES:
            data = data[:_TEXT_READ_LIMIT_BYTES]
            truncated = True
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        if truncated:
            text += f"\n\n--- truncated at {_TEXT_READ_LIMIT_BYTES} bytes ---\n"
        return PlainTextResponse(text)

    @router.get("/binary")
    def file_binary(path: str = Query(..., description="Workspace-relative file")) -> FileResponse:
        f = _safe_join(path)
        if not f.exists() or not f.is_file():
            raise HTTPException(404, f"not found: {path}")
        return FileResponse(f)

    @router.get("/pdf")
    def pdf() -> FileResponse:
        """Serve the paper PDF. Convenience over /api/binary."""
        workspace = _require_workspace()
        p = _find_paper(workspace)
        if p is None:
            raise HTTPException(404, "no PDF found under paper/")
        return FileResponse(p, media_type="application/pdf")

    # ─── Agent registry (static role → tools) ────────────────────────────

    @router.get("/agents")
    def agents() -> dict[str, Any]:
        """Per-role tool config."""
        # Imported lazily so a plain app boot doesn't pull the sub_agent
        # package (and its claude-agent-sdk dep) into memory before chat
        # actually needs it.
        from ..sub_agent.tools import BUILTIN_TOOLS, CUSTOM_TOOL_NAMES
        shared = list(BUILTIN_TOOLS) + list(CUSTOM_TOOL_NAMES)
        return {
            "roles": [
                {"role": "refiner", "tools": shared, "glyph": "📝"},
                {"role": "researcher", "tools": shared, "glyph": "🔬"},
                {"role": "builder", "tools": shared, "glyph": "🔨"},
                {"role": "verifier", "tools": shared, "glyph": "✅"},
            ],
            "mcp_servers": ["phase"],
        }

    # ─── Command channel (write side) ────────────────────────────────────

    @router.post("/commands/chat")
    def post_chat(body: ChatCommandBody) -> dict[str, Any]:
        workspace = _require_workspace()
        text = body.text.strip()
        if not text:
            raise HTTPException(400, "empty text")
        from ..commands.client import append_command, make_command
        cmd = make_command(
            "chat_message",
            {},
            issuer="research-builder-app",
        )
        cmd["agent_id"] = "orchestrator"
        cmd["text"] = text
        append_command(workspace / "logs" / "commands.jsonl", cmd)
        return {"ok": True, "cmd_id": cmd["cmd_id"]}

    @router.post("/commands/force_retry")
    def post_force_retry(body: ForceRetryBody) -> dict[str, Any]:
        workspace = _require_workspace()
        from ..commands.client import force_retry
        cmd = force_retry(
            workspace / "logs" / "commands.jsonl",
            phase_id=body.phase_id,
            reset_refined_spec=body.reset_refined_spec,
            reset_research_cache=body.reset_research_cache,
            rationale=body.rationale,
        )
        return {"ok": True, "cmd_id": cmd["cmd_id"]}

    @router.post("/commands/inject_note")
    def post_inject_note(body: InjectNoteBody) -> dict[str, Any]:
        workspace = _require_workspace()
        from ..commands.client import inject_note
        cmd = inject_note(
            workspace / "logs" / "commands.jsonl",
            text=body.text,
            scope=body.scope,
            phase_id=body.phase_id,
            target_agents=body.target_agents,
            rationale=body.rationale,
        )
        return {"ok": True, "cmd_id": cmd["cmd_id"]}

    # ─── Per-phase refined_spec read + edit-with-cascade-preview ────────

    @router.get("/refined-spec")
    def get_refined_spec(phase_id: str = Query(...)) -> dict[str, Any]:
        """Legacy endpoint.

        New section specs live at ``canonical_spec/sections/<phase_id>.md`` and
        are exposed via ``/api/sections/{phase_id}``. This endpoint stays for
        backwards compat with workspaces from before the upfront-authoring
        revamp; it falls back to a freshly-generated refined spec under the
        per-phase context dir.
        """
        workspace = _require_workspace()
        # Prefer the upfront section spec if it exists.
        upfront = workspace / "canonical_spec" / "sections" / f"{phase_id}.md"
        if upfront.exists():
            return {
                "phase_id": phase_id,
                "exists": True,
                "path": str(upfront.relative_to(workspace)),
                "content": upfront.read_text(),
                "source": "upfront_section_spec",
            }
        path = workspace / "phases" / phase_id / "context" / "refined_spec.md"
        return {
            "phase_id": phase_id,
            "exists": path.exists(),
            "path": str(path.relative_to(workspace)),
            "content": path.read_text() if path.exists() else "",
            "source": "refined_spec" if path.exists() else None,
        }

    # ─── Per-section specs (upfront authoring) ──────────────────────────

    @router.get("/sections")
    def list_sections() -> dict[str, Any]:
        """List per-section specs with their critique verdict and last-modified."""
        workspace = _require_workspace()
        sections_dir = workspace / "canonical_spec" / "sections"
        if not sections_dir.exists():
            return {"sections": []}

        # Pull phase titles from state.json so the list is self-contained.
        state_path = workspace / "canonical_spec" / "state.json"
        titles: dict[str, str] = {}
        if state_path.exists():
            try:
                spec_state = json.loads(state_path.read_text() or "{}")
                for p in spec_state.get("phases", []) or []:
                    titles[p.get("phase_id", "")] = p.get("title") or p.get("phase_id", "")
            except json.JSONDecodeError:
                pass

        out: list[dict[str, Any]] = []
        for json_path in sorted(sections_dir.glob("*.json")):
            if json_path.stem.endswith(".critique"):
                continue
            phase_id = json_path.stem
            critique_path = sections_dir / f"{phase_id}.critique.json"
            md_path = sections_dir / f"{phase_id}.md"
            try:
                sidecar = json.loads(json_path.read_text() or "{}")
            except json.JSONDecodeError:
                sidecar = {}
            critique = None
            if critique_path.exists():
                try:
                    critique = json.loads(critique_path.read_text() or "{}")
                except json.JSONDecodeError:
                    critique = None
            out.append({
                "phase_id": phase_id,
                "title": sidecar.get("title") or titles.get(phase_id) or phase_id,
                "goal": sidecar.get("goal", ""),
                "criteria_count": len(sidecar.get("acceptance_criteria") or []),
                "citations_count": len(sidecar.get("citations") or []),
                "critique_verdict": (critique or {}).get("verdict"),
                "md_path": str(md_path.relative_to(workspace)) if md_path.exists() else None,
                "last_modified": json_path.stat().st_mtime,
            })
        return {"sections": out}

    @router.get("/sections/{phase_id}")
    def get_section(phase_id: str) -> dict[str, Any]:
        """Return one section spec: markdown body + structured fields + citations."""
        workspace = _require_workspace()
        sections_dir = workspace / "canonical_spec" / "sections"
        json_path = sections_dir / f"{phase_id}.json"
        md_path = sections_dir / f"{phase_id}.md"
        if not json_path.exists() or not md_path.exists():
            raise HTTPException(404, f"no section spec for {phase_id}")
        try:
            sidecar = json.loads(json_path.read_text() or "{}")
        except json.JSONDecodeError as e:
            raise HTTPException(500, f"failed to parse section spec json: {e}")
        return {
            "phase_id": phase_id,
            "title": sidecar.get("title", phase_id),
            "goal": sidecar.get("goal", ""),
            "spec_markdown": md_path.read_text(),
            "acceptance_criteria": sidecar.get("acceptance_criteria", []),
            "citations": sidecar.get("citations", []),
            "md_path": str(md_path.relative_to(workspace)),
        }

    @router.get("/sections/{phase_id}/critique")
    def get_section_critique(phase_id: str) -> dict[str, Any]:
        """Return the critic's verdict for one section spec."""
        workspace = _require_workspace()
        path = workspace / "canonical_spec" / "sections" / f"{phase_id}.critique.json"
        if not path.exists():
            raise HTTPException(404, f"no critique for {phase_id}")
        try:
            return json.loads(path.read_text() or "{}")
        except json.JSONDecodeError as e:
            raise HTTPException(500, f"failed to parse critique json: {e}")

    # ─── Claims / verification / final report ───────────────────────────

    @router.get("/claims")
    def get_claims() -> dict[str, Any]:
        """Return the numerical claims ledger if it exists."""
        workspace = _require_workspace()
        path = workspace / "canonical_spec" / "claims.json"
        if not path.exists():
            return {"claims": [], "exists": False}
        try:
            claims = json.loads(path.read_text() or "[]")
        except json.JSONDecodeError as e:
            raise HTTPException(500, f"failed to parse claims.json: {e}")
        return {"claims": claims, "exists": True}

    @router.get("/verification/{phase_id}")
    def get_verification(phase_id: str) -> dict[str, Any]:
        """Return the verifier's report(s) for a phase.

        The verifier writes one or more JSON files at
        ``phases/<phase_id>/outputs/verification_*.json``. This endpoint
        gathers them and returns the latest, plus a count of historical
        attempts.
        """
        workspace = _require_workspace()
        outputs_dir = workspace / "phases" / phase_id / "outputs"
        if not outputs_dir.exists():
            raise HTTPException(404, f"no outputs for {phase_id}")
        reports = sorted(outputs_dir.glob("verification_*.json"))
        if not reports:
            return {"phase_id": phase_id, "reports": [], "latest": None}
        out: list[dict[str, Any]] = []
        latest_payload = None
        for path in reports:
            try:
                payload = json.loads(path.read_text() or "{}")
            except json.JSONDecodeError:
                payload = {"_parse_error": True}
            entry = {
                "filename": path.name,
                "path": str(path.relative_to(workspace)),
                "modified": path.stat().st_mtime,
                "payload": payload,
            }
            out.append(entry)
            latest_payload = entry
        return {"phase_id": phase_id, "reports": out, "latest": latest_payload}

    @router.get("/report")
    def get_report() -> dict[str, Any]:
        """Return the final reproduction report if it exists."""
        workspace = _require_workspace()
        path = workspace / "report" / "reproduction_report.md"
        if not path.exists():
            return {"exists": False, "path": None, "content": None}
        return {
            "exists": True,
            "path": str(path.relative_to(workspace)),
            "content": path.read_text(),
        }

    # ─── Cloud compute (Lambda instances) ───────────────────────────────

    @router.get("/compute")
    def list_compute() -> dict[str, Any]:
        """List every Lambda instance the harness has provisioned this run.

        Source of truth is ``logs/compute_instances.json``, written by
        ``CloudProvisioner`` on launch/upgrade/teardown. Returns ``{}`` when
        no GPU phases have run yet — the UI renders an empty state.
        """
        workspace = _require_workspace()
        path = workspace / "logs" / "compute_instances.json"
        if not path.exists():
            return {"instances": [], "budget": None}
        try:
            snapshot = json.loads(path.read_text() or "{}")
        except json.JSONDecodeError as e:
            raise HTTPException(500, f"failed to parse compute_instances.json: {e}")
        instances_map = snapshot.get("instances", {}) or {}
        instances = sorted(
            instances_map.values(),
            key=lambda r: r.get("provisioned_at") or "",
            reverse=True,
        )
        return {
            "instances": instances,
            "budget": snapshot.get("budget"),
            "updated_at": snapshot.get("updated_at"),
        }

    @router.get("/compute/{instance_id}")
    def get_compute(instance_id: str) -> dict[str, Any]:
        """Return one instance's record + every remote_run.sh invocation routed through it.

        The remote_runs list is materialized by scanning events.jsonl for
        ``process_started`` / ``process_result`` pairs where:
          - the agent_id matches ``phase:<record.phase_id>``
          - the command contains ``remote_run.sh``
          - the started_ts falls between provisioned_at and terminated_at
        This is the only place the harness has remote stdout — the sub-agent
        invokes bash, the Bash tool captures stdout into the process_result
        event (capped at 2000 chars), and we splice them back together here.
        """
        workspace = _require_workspace()
        path = workspace / "logs" / "compute_instances.json"
        if not path.exists():
            raise HTTPException(404, f"no compute snapshot")
        try:
            snapshot = json.loads(path.read_text() or "{}")
        except json.JSONDecodeError as e:
            raise HTTPException(500, f"failed to parse compute_instances.json: {e}")
        record = (snapshot.get("instances") or {}).get(instance_id)
        if record is None:
            raise HTTPException(404, f"no instance with id={instance_id}")

        ssh_cmd = None
        if record.get("ssh_user") and record.get("public_ip") and record.get("ssh_key_path"):
            ssh_cmd = (
                f"ssh -i {record['ssh_key_path']} -o StrictHostKeyChecking=no "
                f"-o UserKnownHostsFile=/dev/null "
                f"{record['ssh_user']}@{record['public_ip']}"
            )

        remote_runs = _scan_remote_runs(
            workspace / "logs" / "events.jsonl",
            phase_id=record.get("phase_id", ""),
            started_at=record.get("provisioned_at"),
            ended_at=record.get("terminated_at"),
        )
        return {
            **record,
            "ssh_command": ssh_cmd,
            "lambda_console_url": "https://cloud.lambda.ai/instances",
            "remote_runs": remote_runs,
        }

    @router.post("/spec/preview-edit")
    def preview_spec_edit(body: SpecEditPreviewBody) -> dict[str, Any]:
        workspace = _require_workspace()
        from .cascade import compute_cascade
        return compute_cascade(
            workspace,
            body.phase_id,
            body.content,
            body.before_agent,
        )

    @router.post("/spec/apply-edit")
    def apply_spec_edit(body: SpecEditApplyBody) -> dict[str, Any]:
        workspace = _require_workspace()
        from ..commands.client import edit_refined_spec, jump_back
        cmd_log = workspace / "logs" / "commands.jsonl"
        edit_cmd = edit_refined_spec(
            cmd_log,
            phase_id=body.phase_id,
            content=body.content,
            before_agent=body.before_agent,
            mode="replace",
            rationale=body.rationale,
        )
        jump_cmd = jump_back(
            cmd_log,
            to_phase_id=body.phase_id,
            preserve_artifacts=True,
            rationale=body.rationale or f"operator edit to {body.phase_id} refined_spec",
        )
        return {
            "ok": True,
            "edit_cmd_id": edit_cmd["cmd_id"],
            "jump_cmd_id": jump_cmd["cmd_id"],
        }

    return router


# ─── Command request bodies ──────────────────────────────────────────────


class ChatCommandBody(BaseModel):
    text: str


class ForceRetryBody(BaseModel):
    phase_id: str
    reset_refined_spec: bool = False
    reset_research_cache: bool = False
    rationale: str = ""


class InjectNoteBody(BaseModel):
    text: str
    scope: str = "phase"
    phase_id: str | None = None
    target_agents: list[str] | None = None
    rationale: str = ""


class SpecEditPreviewBody(BaseModel):
    phase_id: str
    content: str
    before_agent: str = "builder"


class SpecEditApplyBody(BaseModel):
    phase_id: str
    content: str
    before_agent: str = "builder"
    rationale: str = ""


# ─── helpers ─────────────────────────────────────────────────────────────


def _find_paper(workspace: Path) -> Path | None:
    """Locate the paper PDF. Prefers paper/paper.pdf, else the first PDF."""
    paper_dir = workspace / "paper"
    preferred = paper_dir / "paper.pdf"
    if preferred.exists():
        return preferred
    if paper_dir.exists():
        for p in sorted(paper_dir.glob("*.pdf")):
            return p
    return None


def _find_paper_candidates(paper_dir: Path):
    """Yield existing PDFs in ``paper_dir`` (preferred name first)."""
    if not paper_dir.exists():
        return
    preferred = paper_dir / "paper.pdf"
    if preferred.exists():
        yield preferred
    for p in sorted(paper_dir.glob("*.pdf")):
        if p != preferred:
            yield p


def _scan_remote_runs(
    events_path: Path,
    *,
    phase_id: str,
    started_at: str | None,
    ended_at: str | None,
) -> list[dict[str, Any]]:
    """Walk events.jsonl and pull every remote_run.sh Bash call for a phase.

    Returns a list of {process_id, started_at, finished_at, command, output,
    is_error}, oldest first. Caps at 50 entries — older runs would push the
    detail payload past anything the UI usefully renders, and the operator
    can ssh in directly if they need full history.
    """
    if not events_path.exists() or not phase_id:
        return []
    target_agent = f"phase:{phase_id}"
    pending: dict[str, dict[str, Any]] = {}
    finished: list[dict[str, Any]] = []
    try:
        with events_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("agent_id") != target_agent:
                    continue
                etype = e.get("type")
                ts = e.get("ts")
                if etype == "process_started":
                    cmd = e.get("command") or ""
                    if "remote_run.sh" not in cmd:
                        continue
                    if started_at and ts and ts < started_at:
                        continue
                    if ended_at and ts and ts > ended_at:
                        continue
                    pending[e.get("process_id", "")] = {
                        "process_id": e.get("process_id"),
                        "started_at": ts,
                        "finished_at": None,
                        "command": cmd,
                        "output": "",
                        "is_error": False,
                    }
                elif etype == "process_result":
                    pid = e.get("process_id", "")
                    rec = pending.pop(pid, None)
                    if rec is None:
                        continue
                    rec["finished_at"] = ts
                    rec["output"] = e.get("output", "") or ""
                    rec["is_error"] = bool(e.get("is_error", False))
                    finished.append(rec)
    except OSError:
        return []
    # In-flight runs (no result yet) — surface them so the UI can show "running".
    in_flight = list(pending.values())
    all_runs = finished + in_flight
    all_runs.sort(key=lambda r: r.get("started_at") or "")
    return all_runs[-50:]


def _load_attempts(workspace: Path, phase_id: str) -> list[dict[str, Any]]:
    """Return one entry per attempt dir, each with step manifest."""
    base = workspace / "phases" / phase_id / "attempts"
    if not base.exists():
        return []
    attempts: list[dict[str, Any]] = []
    try:
        retries = sorted(
            [d for d in base.iterdir() if d.is_dir()],
            key=lambda p: int(p.name) if p.name.isdigit() else 999,
        )
    except OSError:
        return []
    for d in retries:
        manifest = d / "manifest.json"
        steps: list[dict[str, Any]] = []
        if manifest.exists():
            try:
                steps = json.loads(manifest.read_text()) or []
            except (json.JSONDecodeError, OSError):
                steps = []
        attempts.append({"retry_num": d.name, "steps": steps})
    return attempts

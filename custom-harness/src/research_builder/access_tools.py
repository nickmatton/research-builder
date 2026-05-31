"""Shared MCP tool: ad-hoc access to paths outside the sandbox cwd.

The agent's filesystem sandbox (Claude Agent SDK) is sealed at session start
via ``ClaudeAgentOptions(cwd=..., add_dirs=[...])``. ``--allow-dir`` lets the
user pre-allow extra dirs at launch. This module provides a single MCP tool
for the *ad-hoc* case — the agent discovers at runtime that it needs a path
the user didn't think to pre-allow:

    mcp__access__read_outside_workspace(path: str, reason: str)
       → file contents OR a denial message

Approval policy:

  1. Hard denylist (SSH/AWS/GPG keys, system secrets) → refuse, no prompt.
  2. Path is already under cwd or an --allow-dir → read directly (and tell
     the agent it could use Read next time).
  3. Path was approved earlier this session → read directly.
  4. Config.interactive (CLI without --auto): surface a y/n prompt via
     click (off-loaded to a thread so we don't block the agent event loop).
  5. Otherwise (--auto, or web app today): refuse, telling the user to
     restart with --allow-dir.

The tool reads in our Python process — not the sandboxed shell — so the file
content comes back to the agent over the MCP channel, sidestepping the SDK's
filesystem sandbox entirely.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

import click
from claude_agent_sdk import create_sdk_mcp_server, tool

from .config import Config

logger = logging.getLogger(__name__)


# Fully-qualified tool name the SDK needs in ``allowed_tools``.
ACCESS_TOOL_NAMES = ["mcp__access__read_outside_workspace"]

# Approval callback signature. Returns True iff the user approved access.
# Implementations may surface a prompt in the CLI, push a gate event to the
# web UI, or auto-approve based on policy — the tool doesn't care how.
ApprovalCallback = Callable[[Path, str], Awaitable[bool]]

# Replies that count as "yes". Anything else is treated as denial. Conservative
# on purpose: an ambiguous "ok i guess" should not auto-grant filesystem
# access. The user can always say "yes" explicitly.
_APPROVAL_WORDS = {"y", "yes", "approve", "approved", "allow", "ok", "okay", "sure"}

# Cap returned content so a 500MB log doesn't blow up the agent's context.
MAX_BYTES = 1_000_000  # 1 MB

# Hard denylist — these never get read, even with user approval. The agent
# has no business looking at credential material; if there's a legitimate
# need we can revisit. Patterns are checked against the *resolved* absolute
# path so symlinks can't dodge them.
_DENY_DIR_FRAGMENTS = (
    "/.ssh/",
    "/.aws/",
    "/.gnupg/",
    "/.config/gcloud/",
)
_DENY_FILENAMES = {
    "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
    "shadow", "master.passwd", "sudoers",
}
_DENY_SUFFIXES = (".pem", ".key", ".p12", ".pfx")


def _is_denied(p: Path) -> str | None:
    """Return a refusal reason if ``p`` is on the hard denylist, else None."""
    s = str(p)
    for frag in _DENY_DIR_FRAGMENTS:
        if frag in s:
            return f"refused: path is under a sensitive directory ({frag.strip('/')})"
    if p.name in _DENY_FILENAMES:
        return f"refused: '{p.name}' is on the credential denylist"
    if p.suffix.lower() in _DENY_SUFFIXES:
        return f"refused: '{p.suffix}' files are on the credential denylist"
    return None


def _is_already_accessible(p: Path, cwd: Path, extras: list[Path]) -> bool:
    """True if ``p`` is inside cwd or any extra allowed dir (i.e., the agent
    could have just used the built-in Read tool)."""
    try:
        p.relative_to(cwd)
        return True
    except ValueError:
        pass
    for d in extras:
        try:
            p.relative_to(d)
            return True
        except ValueError:
            continue
    return False


def _reply_is_approval(reply: str) -> bool:
    """True iff the user's chat reply unambiguously means 'yes, allow it'."""
    return reply.strip().lower() in _APPROVAL_WORDS


async def _prompt_user_async(path: Path, reason: str) -> bool:
    """Default CLI fallback: ask via click when no approval callback is wired.

    Off-loaded to a worker thread so the agent event loop keeps pumping
    while we wait on stdin. Defaults to deny on Ctrl-C / EOF / unparseable
    input. The web app spawns the pipeline as a subprocess with no
    controlling tty, so the orchestrator wires a chat-based callback
    instead (see make_chat_approval_callback) and this fallback is unused
    there — it'd hang on stdin otherwise.
    """
    def _ask() -> bool:
        try:
            click.echo("")
            click.echo(f"  \033[1;33maccess request:\033[0m {path}")
            click.echo(f"    reason: {reason}")
            ans = click.prompt(
                "    allow read? [y/N]",
                type=str,
                default="n",
                show_default=False,
            )
            return _reply_is_approval(ans)
        except (click.Abort, EOFError, KeyboardInterrupt):
            return False

    return await asyncio.get_running_loop().run_in_executor(None, _ask)


def make_chat_approval_callback(
    approval_queue: "asyncio.Queue[str]",
    emitter: Any | None,
) -> ApprovalCallback:
    """Build a callback that approves via the same chat surface as
    ``request_user_approval`` — works in both CLI and web app.

    Emits a ``gate_reached`` event so the frontend / inline viewer renders
    a banner, mirrors the prompt as an assistant message so it lands in
    the chat transcript, then awaits the user's next reply on ``approval_queue``.

    The reply is parsed strictly: only ``yes`` / ``y`` / ``approve`` /
    ``allow`` / ``ok`` (case-insensitive) count as approval. Everything else
    denies — the user can always re-prompt the agent if they wanted to ask
    a clarifying question first.
    """
    async def _callback(path: Path, reason: str) -> bool:
        gate_id = f"access:{path}"
        prompt = (
            f"The agent wants to read a file outside its workspace.\n\n"
            f"**Path:** `{path}`\n"
            f"**Reason:** {reason}\n\n"
            f"Reply `yes` to allow this single read (cached for the rest of "
            f"this session). Any other reply denies."
        )
        if emitter:
            emitter.emit(
                "gate_reached",
                agent_id="access",
                parent_id=None,
                gate_id=gate_id,
                prompt=prompt,
                open_doc=None,
                context={"path": str(path), "reason": reason},
                auto=False,
            )
            emitter.emit(
                "agent_message",
                agent_id="access",
                parent_id=None,
                role="assistant",
                text=prompt,
            )
        reply = await approval_queue.get()
        decision = _reply_is_approval(reply)
        if emitter:
            emitter.emit(
                "gate_resolved",
                agent_id="access",
                parent_id=None,
                gate_id=gate_id,
                decision="approve" if decision else "deny",
                auto=False,
            )
        return decision

    return _callback


async def handle_read_request(
    args: dict[str, Any],
    *,
    config: Config,
    cwd: Path,
    approved: set[str],
    approval_callback: ApprovalCallback | None = None,
) -> dict[str, Any]:
    """Process a single read_outside_workspace tool call.

    Pulled out of the @tool closure so the policy is unit-testable without
    going through the SDK's MCP server plumbing. ``approved`` is mutated
    on success and used to short-circuit re-prompts.

    ``approval_callback`` is the interactive-mode approval surface. When
    None we fall back to a CLI click.prompt — that's fine for ``research-
    builder`` invoked from a terminal, but the web-spawned subprocess has
    no controlling tty, so the orchestrator must pass a chat-based callback
    (see make_chat_approval_callback). Otherwise the click.prompt would
    block indefinitely.
    """
    raw_path = str(args.get("path", "")).strip()
    reason = str(args.get("reason", "")).strip()
    if not raw_path:
        return {"content": [{"type": "text", "text": "error: path is required"}],
                "is_error": True}
    if not reason:
        return {"content": [{"type": "text",
                             "text": "error: reason is required (shown to the user)"}],
                "is_error": True}

    try:
        target = Path(os.path.expanduser(raw_path)).resolve()
    except Exception as e:
        return {"content": [{"type": "text", "text": f"error: invalid path: {e}"}],
                "is_error": True}

    denied = _is_denied(target)
    if denied:
        logger.info("access denied (denylist): %s reason=%r", target, reason)
        return {"content": [{"type": "text", "text": denied}], "is_error": True}

    if not target.exists():
        return {"content": [{"type": "text", "text": f"error: path does not exist: {target}"}],
                "is_error": True}
    if target.is_dir():
        return {"content": [{"type": "text",
                             "text": "error: path is a directory; this tool reads files only"}],
                "is_error": True}

    extras = [Path(p).resolve() for p in config.extra_allowed_dirs]
    cwd_resolved = cwd.resolve()

    # Fast-path: already inside cwd or an allowed dir. Just read it, but
    # nudge the agent toward the built-in Read tool for next time.
    if _is_already_accessible(target, cwd_resolved, extras):
        logger.debug("access fast-path (in cwd/extras): %s", target)
        return _read_and_return(target, hint=(
            "note: this path is already inside your sandbox — you can use "
            "the built-in Read tool for it directly next time."
        ))

    # Approval cache short-circuit.
    if str(target) in approved:
        logger.debug("access cache-hit: %s", target)
        return _read_and_return(target, hint=None)

    # Not pre-allowed and not cached. Ask, or refuse.
    if not config.interactive:
        logger.info("access auto-denied (non-interactive): %s reason=%r", target, reason)
        return {"content": [{"type": "text", "text": (
            f"refused: running in non-interactive (--auto) mode. To allow this path, "
            f"restart with: --allow-dir {target.parent} (or a higher ancestor)."
        )}], "is_error": True}

    if approval_callback is not None:
        approved_now = await approval_callback(target, reason)
    else:
        approved_now = await _prompt_user_async(target, reason)
    if not approved_now:
        logger.info("access user-denied: %s reason=%r", target, reason)
        return {"content": [{"type": "text",
                             "text": "refused: user declined access"}],
                "is_error": True}

    approved.add(str(target))
    logger.info("access user-approved: %s reason=%r", target, reason)
    return _read_and_return(target, hint=None)


def create_access_server(
    config: Config,
    cwd: Path,
    *,
    approval_cache: set[str] | None = None,
    approval_callback: ApprovalCallback | None = None,
):
    """Build the ``access`` MCP server for one agent session.

    Args:
        config: the run config; ``extra_allowed_dirs`` + ``interactive`` are read.
        cwd: the agent's working directory (sub-agent: work_dir; orchestrator:
            project_root). Paths already inside cwd skip the approval flow.
        approval_cache: optional shared set of resolved-path strings to dedupe
            re-prompts within a session. Pass the same set to multiple
            sub-agents if you want approvals to persist across phases; omit
            for per-session caching (default).
        approval_callback: how to ask the user when interactive. None falls
            back to click.prompt (CLI only) — pass make_chat_approval_callback
            for sessions spawned by the web app or any other tty-less context.
    """
    approved: set[str] = approval_cache if approval_cache is not None else set()

    @tool(
        "read_outside_workspace",
        "Read a file at ``path`` that lives OUTSIDE your workspace cwd. Use this "
        "only when the path is not under your cwd and not pre-allowed via "
        "--allow-dir. The harness checks the path against a credential denylist, "
        "then (in interactive mode) asks the user to approve. ``reason`` is shown "
        "to the user verbatim — be specific about what you need and why.",
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file you want to read.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why you need this path. Shown to the user for approval.",
                },
            },
            "required": ["path", "reason"],
        },
    )
    async def read_outside_workspace(args: dict[str, Any]) -> dict[str, Any]:
        return await handle_read_request(
            args,
            config=config,
            cwd=cwd,
            approved=approved,
            approval_callback=approval_callback,
        )

    return create_sdk_mcp_server(
        name="access",
        version="1.0.0",
        tools=[read_outside_workspace],
    )


def _read_and_return(path: Path, hint: str | None) -> dict[str, Any]:
    """Read up to MAX_BYTES from ``path`` and return as MCP content."""
    try:
        data = path.read_bytes()
    except Exception as e:
        return {"content": [{"type": "text", "text": f"error: read failed: {e}"}],
                "is_error": True}

    truncated = False
    if len(data) > MAX_BYTES:
        data = data[:MAX_BYTES]
        truncated = True

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {"content": [{"type": "text",
                             "text": f"error: file is not utf-8 text ({len(data)} bytes); "
                                     "this tool returns text only"}],
                "is_error": True}

    body = text
    if truncated:
        body += f"\n\n[truncated to first {MAX_BYTES} bytes]"
    if hint:
        body = f"[{hint}]\n\n{body}"
    return {"content": [{"type": "text", "text": body}]}

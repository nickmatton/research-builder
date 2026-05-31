"""Policy tests for access_tools.handle_read_request.

We drive the policy function directly rather than going through the SDK's
MCP server plumbing — every branch (denylist, cwd fast-path, auto-deny,
cache, truncation, user prompt) lives in handle_read_request.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from research_builder.access_tools import (
    MAX_BYTES,
    _is_already_accessible,
    _is_denied,
    _reply_is_approval,
    handle_read_request,
    make_chat_approval_callback,
)
from research_builder.config import Config


def _text(result) -> str:
    """Concatenate text blocks from a tool result dict."""
    return "".join(b.get("text", "") for b in result.get("content", []))


# ─── pure helpers ──────────────────────────────────────────────────────────


def test_denylist_ssh_dir():
    assert _is_denied(Path("/home/user/.ssh/id_rsa")) is not None


def test_denylist_pem_extension():
    assert _is_denied(Path("/tmp/cert.pem")) is not None


def test_denylist_by_filename(tmp_path):
    assert _is_denied(tmp_path / "id_ed25519") is not None
    assert _is_denied(tmp_path / "shadow") is not None


def test_denylist_allows_normal_paths(tmp_path):
    assert _is_denied(tmp_path / "readme.md") is None
    assert _is_denied(Path("/etc/hosts")) is None


def test_already_accessible_cwd(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hi")
    assert _is_already_accessible(f, tmp_path, []) is True


def test_already_accessible_extra(tmp_path):
    extra = tmp_path / "extra"
    extra.mkdir()
    f = extra / "b.txt"
    f.write_text("hi")
    assert _is_already_accessible(f, Path("/somewhere/else"), [extra]) is True


def test_not_accessible_outside(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("hi")
    assert _is_already_accessible(f, Path("/somewhere/else"), []) is False


# ─── tool dispatch ────────────────────────────────────────────────────────


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    p = tmp_path / "ws"
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture
def auto_config() -> Config:
    """Non-interactive config: tool should auto-deny ad-hoc requests."""
    return Config(interactive=False)


@pytest.fixture
def interactive_config() -> Config:
    return Config(interactive=True)


async def _call(args, *, config, cwd, approved=None):
    return await handle_read_request(
        args, config=config, cwd=cwd, approved=approved if approved is not None else set()
    )


@pytest.mark.asyncio
async def test_missing_args_errors(auto_config, workspace):
    r = await _call({"path": "", "reason": "x"}, config=auto_config, cwd=workspace)
    assert r.get("is_error") is True
    r = await _call({"path": "/etc/hosts", "reason": ""}, config=auto_config, cwd=workspace)
    assert r.get("is_error") is True


@pytest.mark.asyncio
async def test_denylist_blocks_before_prompt(interactive_config, workspace, tmp_path):
    """Even interactive mode refuses denylisted paths with no prompt."""
    fake_key = tmp_path / "secrets" / ".ssh" / "id_rsa"
    fake_key.parent.mkdir(parents=True)
    fake_key.write_text("PRIVATE")
    r = await _call({"path": str(fake_key), "reason": "test"},
                    config=interactive_config, cwd=workspace)
    assert r.get("is_error") is True
    assert "denylist" in _text(r) or "sensitive" in _text(r)


@pytest.mark.asyncio
async def test_nonexistent_path(auto_config, workspace):
    r = await _call({"path": "/no/such/file/here.txt", "reason": "test"},
                    config=auto_config, cwd=workspace)
    assert r.get("is_error") is True
    assert "does not exist" in _text(r)


@pytest.mark.asyncio
async def test_auto_mode_refuses_outside_path(auto_config, workspace, tmp_path):
    """--auto runs deny ad-hoc paths with a clear hint about --allow-dir."""
    target = tmp_path / "other" / "data.txt"
    target.parent.mkdir()
    target.write_text("hello")
    r = await _call({"path": str(target), "reason": "need this"},
                    config=auto_config, cwd=workspace)
    assert r.get("is_error") is True
    assert "non-interactive" in _text(r)
    assert "--allow-dir" in _text(r)


@pytest.mark.asyncio
async def test_fast_path_inside_cwd(auto_config, workspace):
    """Paths already under cwd skip approval, even in --auto mode."""
    f = workspace / "in_cwd.txt"
    f.write_text("body")
    r = await _call({"path": str(f), "reason": "test"},
                    config=auto_config, cwd=workspace)
    assert r.get("is_error") is not True
    body = _text(r)
    assert "body" in body
    assert "already inside your sandbox" in body  # hint to use Read next time


@pytest.mark.asyncio
async def test_fast_path_inside_extra_allowed_dir(workspace, tmp_path):
    extra = tmp_path / "shared"
    extra.mkdir()
    f = extra / "ok.txt"
    f.write_text("contents")
    cfg = Config(interactive=False, extra_allowed_dirs=[extra])
    r = await _call({"path": str(f), "reason": "test"}, config=cfg, cwd=workspace)
    assert r.get("is_error") is not True
    assert "contents" in _text(r)


@pytest.mark.asyncio
async def test_truncates_large_files(auto_config, workspace):
    f = workspace / "huge.txt"
    f.write_bytes(b"x" * (MAX_BYTES + 1000))
    r = await _call({"path": str(f), "reason": "test"},
                    config=auto_config, cwd=workspace)
    assert r.get("is_error") is not True
    assert "truncated" in _text(r)


@pytest.mark.asyncio
async def test_binary_file_rejected(auto_config, workspace):
    f = workspace / "blob.bin"
    f.write_bytes(b"\x00\x01\x02\xff\xfe")
    r = await _call({"path": str(f), "reason": "test"},
                    config=auto_config, cwd=workspace)
    assert r.get("is_error") is True
    assert "utf-8" in _text(r).lower()


@pytest.mark.asyncio
async def test_user_approval_caches(interactive_config, workspace, tmp_path, monkeypatch):
    """After the user approves, re-reads of the same path don't re-prompt."""
    target = tmp_path / "other" / "doc.txt"
    target.parent.mkdir()
    target.write_text("approved-content")

    prompt_calls = {"n": 0}

    async def fake_prompt(path, reason):
        prompt_calls["n"] += 1
        return True

    monkeypatch.setattr("research_builder.access_tools._prompt_user_async", fake_prompt)

    cache: set[str] = set()
    r1 = await _call({"path": str(target), "reason": "first"},
                     config=interactive_config, cwd=workspace, approved=cache)
    r2 = await _call({"path": str(target), "reason": "second"},
                     config=interactive_config, cwd=workspace, approved=cache)
    assert r1.get("is_error") is not True
    assert r2.get("is_error") is not True
    assert "approved-content" in _text(r1)
    assert "approved-content" in _text(r2)
    assert prompt_calls["n"] == 1, "cache should prevent re-prompting"


@pytest.mark.asyncio
async def test_user_denial(interactive_config, workspace, tmp_path, monkeypatch):
    target = tmp_path / "other" / "doc.txt"
    target.parent.mkdir()
    target.write_text("nope")

    async def fake_prompt(path, reason):
        return False

    monkeypatch.setattr("research_builder.access_tools._prompt_user_async", fake_prompt)

    r = await _call({"path": str(target), "reason": "test"},
                    config=interactive_config, cwd=workspace)
    assert r.get("is_error") is True
    assert "user declined" in _text(r)


# ─── reply parser + chat callback ─────────────────────────────────────────


@pytest.mark.parametrize("reply,expected", [
    ("yes", True), ("Yes", True), (" y ", True), ("YES", True),
    ("approve", True), ("approved", True), ("allow", True),
    ("ok", True), ("okay", True), ("sure", True),
    ("no", False), ("n", False), ("nope", False),
    ("what does the file contain?", False), ("ok i guess", False),
    ("", False), ("   ", False),
])
def test_reply_parser(reply, expected):
    """Approval matches strict word list; ambiguity always denies."""
    assert _reply_is_approval(reply) is expected


@pytest.mark.asyncio
async def test_callback_overrides_click_prompt(interactive_config, workspace, tmp_path):
    """When a callback is provided, the click fallback is not used."""
    target = tmp_path / "other" / "doc.txt"
    target.parent.mkdir()
    target.write_text("content")

    seen = {"path": None, "reason": None}

    async def cb(path: Path, reason: str) -> bool:
        seen["path"] = path
        seen["reason"] = reason
        return True

    r = await handle_read_request(
        {"path": str(target), "reason": "needed for X"},
        config=interactive_config, cwd=workspace, approved=set(),
        approval_callback=cb,
    )
    assert r.get("is_error") is not True
    assert "content" in _text(r)
    assert seen["path"] == target.resolve()
    assert seen["reason"] == "needed for X"


@pytest.mark.asyncio
async def test_chat_callback_round_trip(tmp_path):
    """make_chat_approval_callback should: emit a gate, await the queue,
    parse the reply, emit a resolution."""
    q: asyncio.Queue[str] = asyncio.Queue()
    events: list[tuple[str, dict]] = []

    class _Emitter:
        def emit(self, name, **kw):
            events.append((name, kw))

    cb = make_chat_approval_callback(q, _Emitter())

    # Pre-load an approval reply, then call the callback.
    await q.put("yes")
    decision = await cb(tmp_path / "x.txt", "because")
    assert decision is True

    names = [n for n, _ in events]
    assert "gate_reached" in names
    assert "gate_resolved" in names
    resolved = [kw for n, kw in events if n == "gate_resolved"][0]
    assert resolved["decision"] == "approve"


@pytest.mark.asyncio
async def test_chat_callback_denies_on_ambiguous_reply(tmp_path):
    q: asyncio.Queue[str] = asyncio.Queue()

    class _Emitter:
        def emit(self, *a, **k): pass

    cb = make_chat_approval_callback(q, _Emitter())
    await q.put("what does it contain?")
    decision = await cb(tmp_path / "x.txt", "because")
    assert decision is False


@pytest.mark.asyncio
async def test_chat_callback_works_without_emitter(tmp_path):
    """make_chat_approval_callback should tolerate emitter=None (no-op events)."""
    q: asyncio.Queue[str] = asyncio.Queue()
    cb = make_chat_approval_callback(q, None)
    await q.put("yes")
    assert await cb(tmp_path / "x.txt", "r") is True

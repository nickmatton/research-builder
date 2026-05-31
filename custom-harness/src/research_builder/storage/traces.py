"""Per-LLM-call trace files.

Each orchestrator (or sub-agent) ``_query`` invocation gets its own
markdown file under ``traces/<role>/<timestamp>__<prompt_role>.md``.
The file captures every streamed chunk with its arrival timestamp, so
you can see exactly where wall-clock time went:

  - System prompt + user prompt at the top
  - Each AssistantMessage's text/thinking/tool blocks, in order
  - Each tool result
  - The final cost/tokens/elapsed footer

Why a separate file per call instead of one append-only log per agent:
- Easy to grep and inspect — open ``traces/orchestrator/<latest>.md``
  to see the most recent refiner / verifier / claims-extraction call.
- Stable order — the JSONL event stream interleaves calls; here we get
  a coherent per-call narrative.
- Cheap — one open file per call, closed when the call ends. No global
  state, no contention.

Writes are best-effort: any I/O failure in here must NEVER bubble up
into the agent's run loop. Wrap callers accordingly.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# Truncate stored prompts/outputs to keep trace files bounded. The text
# blocks themselves are usually short; oversized fields are mostly tool
# inputs (e.g. full file contents passed to Edit).
_BLOCK_CAP = 20_000


class TraceWriter:
    """Markdown trace for one LLM call.

    Construct, then call ``open()`` exactly once and ``close()`` exactly
    once. All other methods are no-ops before ``open()`` or after
    ``close()`` so the trace stays safe to use even when the surrounding
    code path is messy.
    """

    def __init__(
        self,
        traces_dir: Path,
        role: str,            # high-level bucket: "orchestrator" | "sub_agent"
        prompt_role: str,     # caller's tag: "refine-section_4", "verify-...", etc.
        model: str,
    ) -> None:
        self.traces_dir = Path(traces_dir)
        self.role = role
        self.prompt_role = prompt_role
        self.model = model
        self.path: Path | None = None
        self._fh = None
        self._started_at = time.monotonic()
        self._wall_started: datetime | None = None
        # Per-chunk arrival stats for stream-health diagnostics. Each
        # entry records ``(elapsed_ms_since_start, msg_type)``. From these
        # we derive: count by type, max/median gap between chunks, time
        # to first text, time to first tool call, time to ResultMessage.
        self._chunk_times: list[tuple[int, str]] = []
        self._first_text_ms: int | None = None
        self._first_tool_ms: int | None = None
        self._result_message_ms: int | None = None

    # ─── lifecycle ──────────────────────────────────────────────────

    def open(self, *, system_prompt: str, prompt: str) -> None:
        try:
            d = self.traces_dir / self.role
            d.mkdir(parents=True, exist_ok=True)
            self._wall_started = datetime.now()
            stamp = self._wall_started.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_role = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.prompt_role)[:80]
            self.path = d / f"{stamp}__{safe_role}.md"
            self._fh = self.path.open("w", encoding="utf-8")
            self._write(
                f"# Trace · {self.prompt_role}\n\n"
                f"- **Started**: `{self._wall_started.isoformat(timespec='milliseconds')}`\n"
                f"- **Model**: `{self.model}`\n"
                f"- **Role**: `{self.role}`\n\n"
                "## System prompt\n\n```\n"
                f"{_trunc(system_prompt)}\n"
                "```\n\n"
                "## User prompt\n\n```\n"
                f"{_trunc(prompt)}\n"
                "```\n\n"
                "## Stream\n\n"
            )
        except Exception:
            logger.exception("TraceWriter.open failed; tracing disabled for this call")
            self._fh = None
            self.path = None

    def close(self, *, status: str, result_text: str, cost_usd: float | None = None,
              input_tokens: int | None = None, output_tokens: int | None = None,
              messages_received: list[str] | None = None) -> None:
        if self._fh is None:
            return
        elapsed = time.monotonic() - self._started_at
        try:
            footer_lines = [
                "\n## Result\n",
                f"- **Status**: `{status}`",
                f"- **Elapsed**: {elapsed:.1f}s",
            ]
            if cost_usd is not None:
                footer_lines.append(f"- **Cost**: ${cost_usd:.4f}")
            if input_tokens is not None or output_tokens is not None:
                footer_lines.append(
                    f"- **Tokens**: in={input_tokens or 0}, out={output_tokens or 0}"
                )
            if messages_received:
                footer_lines.append(
                    f"- **Message types** ({len(messages_received)}): "
                    f"{' → '.join(messages_received)}"
                )
            footer_lines.append("")
            footer_lines.append(self._format_stream_stats())
            footer_lines.append("### Final response\n")
            footer_lines.append("```")
            footer_lines.append(_trunc(result_text))
            footer_lines.append("```\n")
            self._write("\n".join(footer_lines))
        except Exception:
            logger.exception("TraceWriter.close failed")
        finally:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    # ─── per-event writers ──────────────────────────────────────────

    def thinking(self, text: str) -> None:
        if not text.strip():
            return
        self._block("think", text)

    def assistant_text(self, text: str) -> None:
        if not text.strip():
            return
        self._block("assistant", text)

    def tool_call(self, name: str, input_repr: str = "") -> None:
        head = f"tool: {name}"
        if input_repr:
            head += f" · {input_repr[:200]}"
        self._block("tool_call", head)

    def tool_result(self, text: str, *, is_error: bool = False) -> None:
        if not text.strip():
            return
        self._block("tool_result_err" if is_error else "tool_result", text)

    def note(self, text: str) -> None:
        """Free-form trace note (e.g. 'CLI exited unexpectedly')."""
        self._block("note", text)

    # ─── stream-health diagnostics ──────────────────────────────────

    def chunk_arrived(self, msg_type: str) -> None:
        """Record arrival of any SDK message (StreamEvent / AssistantMessage /
        ResultMessage / etc.). Cheap — just appends a tuple. The footer
        summary derives gap stats from this list.
        """
        elapsed_ms = int((time.monotonic() - self._started_at) * 1000)
        self._chunk_times.append((elapsed_ms, msg_type))

    def mark_first_text(self) -> None:
        if self._first_text_ms is None:
            self._first_text_ms = int((time.monotonic() - self._started_at) * 1000)

    def mark_first_tool(self) -> None:
        if self._first_tool_ms is None:
            self._first_tool_ms = int((time.monotonic() - self._started_at) * 1000)

    def mark_result(self) -> None:
        if self._result_message_ms is None:
            self._result_message_ms = int((time.monotonic() - self._started_at) * 1000)

    def stderr_chunk(self, line: str) -> None:
        """Record one line of CLI stderr verbatim. Surfaces Node-side
        diagnostics that the existing filter-by-keyword approach hides.
        """
        if not line.strip():
            return
        self._block("stderr", line)

    def _format_stream_stats(self) -> str:
        if not self._chunk_times:
            return "- **Stream stats**: no chunks received\n"

        # Gaps between consecutive chunks (ms). Reveals whether the stream
        # was steadily flowing or had stalls right before the crash.
        gaps = [
            self._chunk_times[i][0] - self._chunk_times[i - 1][0]
            for i in range(1, len(self._chunk_times))
        ]
        # Count message types.
        type_counts: dict[str, int] = {}
        for _, t in self._chunk_times:
            type_counts[t] = type_counts.get(t, 0) + 1

        total = len(self._chunk_times)
        max_gap = max(gaps) if gaps else 0
        # crude median: sort + middle
        sorted_gaps = sorted(gaps)
        median_gap = sorted_gaps[len(sorted_gaps) // 2] if gaps else 0
        # last 3 chunks before death — most telling for crash diagnosis
        tail = self._chunk_times[-3:]
        tail_str = " ; ".join(f"+{ms}ms {t}" for ms, t in tail)

        lines = [
            "### Stream stats",
            "",
            f"- Total chunks: {total}",
            f"- Max inter-chunk gap: {max_gap} ms",
            f"- Median inter-chunk gap: {median_gap} ms",
            f"- By type: " + ", ".join(f"{k}×{v}" for k, v in sorted(type_counts.items(), key=lambda kv: -kv[1])),
        ]
        if self._first_text_ms is not None:
            lines.append(f"- Time to first text: {self._first_text_ms} ms")
        if self._first_tool_ms is not None:
            lines.append(f"- Time to first tool call: {self._first_tool_ms} ms")
        if self._result_message_ms is not None:
            lines.append(f"- Time to ResultMessage: {self._result_message_ms} ms")
        else:
            lines.append("- ResultMessage: **never received**")
        lines.append(f"- Last 3 chunks: {tail_str}")
        lines.append("")
        return "\n".join(lines)

    # ─── internals ──────────────────────────────────────────────────

    def _block(self, kind: str, text: str) -> None:
        if self._fh is None:
            return
        try:
            dt_ms = int((time.monotonic() - self._started_at) * 1000)
            self._write(f"\n**[+{dt_ms:>6} ms] {kind}**\n\n")
            self._fh.write("```\n")
            self._fh.write(_trunc(text))
            self._fh.write("\n```\n")
            self._fh.flush()
        except Exception:
            # Best-effort — never bubble into the agent's run loop.
            pass

    def _write(self, s: str) -> None:
        if self._fh is None:
            return
        try:
            self._fh.write(s)
            self._fh.flush()
        except Exception:
            pass


def _trunc(s: str) -> str:
    if not s:
        return ""
    if len(s) <= _BLOCK_CAP:
        return s
    return s[:_BLOCK_CAP] + f"\n\n[... truncated, original {len(s)} chars]"


# ─── No-op fallback for when the workspace dir isn't available yet ────


class _NullTrace:
    def open(self, **_k): pass
    def close(self, **_k): pass
    def thinking(self, _t): pass
    def assistant_text(self, _t): pass
    def tool_call(self, _n, _i=""): pass
    def tool_result(self, _t, **_k): pass
    def note(self, _t): pass
    def chunk_arrived(self, _t): pass
    def mark_first_text(self): pass
    def mark_first_tool(self): pass
    def mark_result(self): pass
    def stderr_chunk(self, _t): pass


NULL_TRACE = _NullTrace()

// Orchestrator chat — the primary user-facing surface for the model-driven
// orchestrator. Renders four kinds of stream items in one transcript:
//
//   1. assistant text bubbles (from agent_message events; the model's
//      natural narration between tool calls)
//   2. user bubbles (locally echoed from POST /api/commands/chat; the
//      orchestrator's emitted agent_message role=user dedupes against this)
//   3. tool call lines (from process_started/process_result events; the
//      orchestrator's tool invocations rendered like Claude Code's
//      "⏺ ToolName(args)" + "⎿ result preview")
//   4. live thinking text (from agent_thinking deltas; dim italic, replaced
//      in place as new deltas arrive, cleared on the next assistant turn)
//
// A working-indicator pill at the bottom shows "✻ Working… 12s" whenever the
// orchestrator agent is mid-turn (agent_started without agent_completed). A
// pinned gate banner shows the active request_user_approval prompt with an
// Approve shortcut.
//
// Send: POST /api/commands/chat → commands.jsonl → harness inbox.
// Receive: /ws/events → events.jsonl → this component.

import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { getEventStream } from "../lib/events";
import { Face } from "./Face";

type StreamItem =
  | {
      kind: "message";
      id: string;
      role: "user" | "assistant";
      text: string;
      ts: string;
    }
  | {
      kind: "tool";
      id: string;
      agentId: string;
      toolName: string;
      summary: string;
      result?: string;
      isError?: boolean;
      ts: string;
    };

interface ActiveGate {
  gate_id: string;
  prompt: string;
}

// Orchestrator-family agent_ids. These agents drive the chat narrative —
// their messages/thinking land as bubbles. Sub-agent activity (phase:* ids,
// inside run_builder) still surfaces in chat for tool calls (so the user
// sees what the Builder is doing live), but its thinking/messages stay in
// TraceView to keep the chat readable.
const ORCH_AGENT_IDS = new Set(["orchestrator"]);
function isOrchestratorFamily(agentId: string | undefined): boolean {
  if (!agentId) return false;
  if (ORCH_AGENT_IDS.has(agentId)) return true;
  return (
    agentId.startsWith("section_author:") ||
    agentId.startsWith("section_critic:")
  );
}

// Strip the SDK's mcp__<server>__<tool> routing prefix and lift the most
// useful inline arg. (The mcp__ prefix is SDK-internal — see runtime_tools.py
// for context. Frontend hides it from users.)
function prettyToolName(raw: string): string {
  if (!raw) return "tool";
  if (raw.includes("__")) {
    const parts = raw.split("__");
    return parts[parts.length - 1];
  }
  return raw;
}

// Compact agent label for tool lines from non-orchestrator agents (Builder
// sub-agents, etc.). e.g. "phase:section_3_2_attention" → "section_3_2".
function shortAgentLabel(agentId: string): string {
  const trimmed = agentId.startsWith("phase:") ? agentId.slice(6) : agentId;
  // Drop any trailing descriptive suffix after the third token to keep it short.
  const parts = trimmed.split("_");
  if (parts.length > 3) {
    return parts.slice(0, 3).join("_");
  }
  return trimmed;
}

export function OrchestratorChat() {
  const [items, setItems] = useState<StreamItem[]>([]);
  const [input, setInput] = useState("");
  const [pipelineAlive, setPipelineAlive] = useState(false);
  const [lastEventAt, setLastEventAt] = useState<number | null>(null);
  const [sending, setSending] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [activeGate, setActiveGate] = useState<ActiveGate | null>(null);
  // Working indicator: agent_started without a matching agent_completed.
  // We track a set of in-flight agent_ids (orchestrator family only) and
  // the earliest start time so the timer reads "Xs since the run started
  // doing work" rather than restarting on every sub-agent.
  const [workingSince, setWorkingSince] = useState<number | null>(null);
  const [workingTickNow, setWorkingTickNow] = useState(Date.now());
  const liveAgentsRef = useRef<Set<string>>(new Set());
  // Latest thinking snippet per active agent — ephemeral, cleared on the
  // next assistant text bubble or agent_completed.
  const [thinking, setThinking] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const stream = getEventStream();
    const off = stream.subscribe((e) => {
      setLastEventAt(Date.now());

      // Gate lifecycle
      if (e.type === "gate_reached") {
        const auto = (e as { auto?: boolean }).auto === true;
        if (!auto) {
          setActiveGate({
            gate_id: String((e as { gate_id?: string }).gate_id ?? ""),
            prompt: String((e as { prompt?: string }).prompt ?? ""),
          });
        }
        return;
      }
      if (e.type === "gate_resolved") {
        setActiveGate(null);
        return;
      }

      const agentId = (e as { agent_id?: string }).agent_id;
      const orchFamily = isOrchestratorFamily(agentId);

      // Working indicator lifecycle — orchestrator family only. The
      // orchestrator stays "started" across the whole run; sub-agent
      // lifecycle would flicker the timer constantly.
      if (e.type === "agent_started" && orchFamily) {
        const now = Date.now();
        liveAgentsRef.current.add(agentId!);
        setWorkingSince((prev) => prev ?? now);
        setWorkingTickNow(now);
        return;
      }
      if ((e.type === "agent_completed" || e.type === "agent_crashed") && orchFamily) {
        liveAgentsRef.current.delete(agentId!);
        if (liveAgentsRef.current.size === 0) {
          setWorkingSince(null);
          setThinking(null);
        }
        return;
      }

      // Thinking deltas — orchestrator family only (sub-agent thinking
      // would overwrite the orchestrator's reasoning row constantly).
      if (e.type === "agent_thinking" && orchFamily) {
        const text = String((e as { text?: string }).text ?? "");
        if (text.trim()) setThinking(text.trim().slice(-220));
        return;
      }

      // Assistant / user messages — orchestrator family only. Sub-agents
      // rarely emit TextBlocks; when they do (e.g. queued operator messages),
      // they're visible in TraceView.
      if (
        e.type === "agent_message" &&
        orchFamily &&
        typeof (e as { text?: unknown }).text === "string" &&
        ((e as { role?: string }).role === "user" ||
          (e as { role?: string }).role === "assistant")
      ) {
        const role = (e as unknown as { role: "user" | "assistant" }).role;
        const text = (e as unknown as { text: string }).text;
        const id = `${e.ts}-${role}-${agentId}`;
        const item: StreamItem = {
          kind: "message",
          id,
          role,
          text,
          ts: e.ts,
        };
        setItems((prev) => (prev.some((p) => p.id === id) ? prev : [...prev, item]));
        if (role === "assistant") setThinking(null);
        return;
      }

      // Tool calls — surface from EVERY agent (orchestrator + sub-agents).
      // Sub-agent tool calls render with an agent label so the user can
      // distinguish orchestrator decisions from Builder execution noise.
      if (e.type === "process_started") {
        const processId = String((e as { process_id?: string }).process_id ?? "");
        const toolName = String((e as { tool_name?: string }).tool_name ?? "");
        const summary = String((e as { summary?: string }).summary ?? toolName);
        const id = `tool-${processId || `${e.ts}-${toolName}`}`;
        setItems((prev) =>
          prev.some((p) => p.id === id)
            ? prev
            : [
                ...prev,
                {
                  kind: "tool",
                  id,
                  agentId: agentId ?? "unknown",
                  toolName: prettyToolName(toolName),
                  summary,
                  ts: e.ts,
                },
              ],
        );
        return;
      }
      if (e.type === "process_result") {
        const processId = String((e as { process_id?: string }).process_id ?? "");
        const output = String((e as { output?: string }).output ?? "");
        const isError = Boolean((e as { is_error?: boolean }).is_error);
        const id = `tool-${processId}`;
        setItems((prev) =>
          prev.map((p) =>
            p.kind === "tool" && p.id === id
              ? { ...p, result: output.slice(0, 280), isError }
              : p,
          ),
        );
        return;
      }
    });

    // Liveness ping for the header dot.
    const t = setInterval(() => {
      if (lastEventAt && Date.now() - lastEventAt < 30000) setPipelineAlive(true);
      else setPipelineAlive(false);
    }, 1000);
    return () => {
      off();
      clearInterval(t);
    };
  }, [lastEventAt]);

  // Tick the working-indicator timer while the orchestrator is in flight.
  useEffect(() => {
    if (workingSince === null) return;
    const t = setInterval(() => setWorkingTickNow(Date.now()), 500);
    return () => clearInterval(t);
  }, [workingSince]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [items, thinking, workingSince, activeGate]);

  const send = async () => {
    const text = input.trim();
    if (!text || sending) return;
    setSending(true);
    setErr(null);
    try {
      await api.commands.chat(text);
      setInput("");
      // Local echo — the orchestrator's emitted user-role agent_message
      // dedupes against this via the composite id.
      const id = `local-${Date.now()}`;
      setItems((prev) => [
        ...prev,
        {
          kind: "message",
          id,
          role: "user",
          text,
          ts: new Date().toISOString(),
        },
      ]);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "send failed");
    } finally {
      setSending(false);
    }
  };

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
      e.preventDefault();
      send();
    }
  };

  const elapsedS =
    workingSince !== null ? Math.max(0, Math.round((workingTickNow - workingSince) / 1000)) : null;

  return (
    <div className="flex h-full flex-col">
      <div className="flex h-11 shrink-0 items-center justify-between border-b border-[var(--color-border)] px-3 text-[11px]">
        <div className="flex items-center gap-2.5">
          <Face />
          <span className="text-[var(--color-fg-muted)]">Orchestrator</span>
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              pipelineAlive
                ? "bg-[var(--color-ok)]"
                : lastEventAt
                  ? "bg-[var(--color-warn)]"
                  : "bg-[var(--color-border-strong)]"
            }`}
          />
        </div>
        <span className="text-[10px] text-[var(--color-fg-dim)]">
          {pipelineAlive ? "live" : lastEventAt ? "quiet" : "no events yet"}
        </span>
      </div>

      <div ref={scrollRef} className="min-h-0 flex-1 space-y-2 overflow-auto px-3 py-3">
        {items.length === 0 ? (
          <div className="mt-6 text-center text-xs text-[var(--color-fg-dim)]">
            <div className="mb-1 text-[var(--color-fg-muted)]">Talk to the orchestrator</div>
            <div className="text-[10px] text-[var(--color-fg-dim)]">
              The orchestrator will introduce itself once the run starts.<br />
              Replies route through{" "}
              <code className="rounded bg-[var(--color-surface-2)] px-1 py-px">
                logs/commands.jsonl
              </code>{" "}
              into its tool-driven loop.
            </div>
          </div>
        ) : (
          items.map((it) =>
            it.kind === "message" ? (
              <Bubble key={it.id} m={it} />
            ) : (
              <ToolLine key={it.id} t={it} />
            ),
          )
        )}

        {thinking && (
          <div className="flex items-start gap-2 px-1 text-[11px] italic text-[var(--color-fg-dim)]">
            <span className="mt-0.5">✻</span>
            <span className="whitespace-pre-wrap leading-relaxed">{thinking}</span>
          </div>
        )}

        {workingSince !== null && (
          <div className="flex items-center gap-2 rounded bg-[var(--color-surface-2)] px-2 py-1 text-[11px] text-[var(--color-fg-muted)]">
            <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[var(--color-accent)]" />
            Working…
            <span className="ml-auto tabular-nums text-[10px] text-[var(--color-fg-dim)]">
              {elapsedS}s
            </span>
          </div>
        )}

        {activeGate && (
          <div className="rounded-md border border-[var(--color-accent)]/40 bg-[var(--color-accent)]/10 px-3 py-2">
            <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-[var(--color-accent)]">
              <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-[var(--color-accent)]" />
              Awaiting your reply
              <span className="ml-1 font-mono text-[10px] text-[var(--color-fg-dim)]">
                {activeGate.gate_id}
              </span>
            </div>
            <div className="mt-1 whitespace-pre-wrap text-xs text-[var(--color-fg)]">
              {activeGate.prompt}
            </div>
            <div className="mt-1.5 text-[10px] text-[var(--color-fg-dim)]">
              Reply with <code className="rounded bg-[var(--color-surface-2)] px-1 py-px font-mono">approve</code> below to proceed, or send feedback to request changes.
            </div>
          </div>
        )}

        {err && (
          <div className="rounded border border-[var(--color-fail)]/40 bg-[var(--color-fail)]/10 px-2 py-1 text-xs text-[var(--color-fail)]">
            {err}
          </div>
        )}
      </div>

      <div className="shrink-0 border-t border-[var(--color-border)] p-2">
        <div className="flex items-end gap-2">
          <textarea
            ref={taRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
            rows={2}
            placeholder={
              activeGate
                ? "Say 'approve' to proceed, or ask / request edits…"
                : "Talk to the orchestrator…"
            }
            className="min-h-[40px] flex-1 resize-none rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-2.5 py-1.5 text-xs outline-none focus:border-[var(--color-accent)]"
          />
          <button
            type="button"
            onClick={send}
            disabled={!input.trim() || sending}
            className="rounded bg-[var(--color-accent)] px-3 py-1.5 text-xs font-medium text-white disabled:cursor-not-allowed disabled:opacity-30"
          >
            {sending ? "…" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}

function Bubble({ m }: { m: Extract<StreamItem, { kind: "message" }> }) {
  const isUser = m.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[88%] whitespace-pre-wrap rounded-md px-2.5 py-1.5 text-xs leading-relaxed ${
          isUser
            ? "bg-[var(--color-accent)]/15"
            : "bg-[var(--color-surface-2)]"
        }`}
      >
        {m.text}
      </div>
    </div>
  );
}

function ToolLine({ t }: { t: Extract<StreamItem, { kind: "tool" }> }) {
  // Orchestrator-driven tool calls = the high-level narrative; sub-agent
  // tool calls (Builder doing Write/Bash/Edit etc.) = execution noise.
  // Indent + dim the sub-agent ones so the eye picks orchestrator decisions
  // at a glance.
  const isSubAgent = !isOrchestratorFamily(t.agentId);
  const agentLabel = isSubAgent ? shortAgentLabel(t.agentId) : null;
  return (
    <div
      className={`rounded border border-[var(--color-border)]/60 px-2 py-1 text-[11px] ${
        isSubAgent
          ? "ml-4 border-l-2 border-l-[var(--color-border)] bg-[var(--color-surface-2)]/20"
          : "bg-[var(--color-surface-2)]/40"
      }`}
    >
      <div className="flex items-center gap-1.5 font-mono text-[var(--color-fg-muted)]">
        <span className={isSubAgent ? "text-[var(--color-fg-dim)]" : "text-[var(--color-accent)]"}>⏺</span>
        {agentLabel && (
          <span className="rounded bg-[var(--color-surface)] px-1 text-[10px] text-[var(--color-fg-dim)]">
            {agentLabel}
          </span>
        )}
        <span className={isSubAgent ? "text-[var(--color-fg-muted)]" : "font-medium text-[var(--color-fg)]"}>
          {t.toolName}
        </span>
        {t.summary && t.summary !== t.toolName && (
          <span className="truncate text-[var(--color-fg-dim)]">{t.summary}</span>
        )}
        {t.result !== undefined && (
          <span
            className={`ml-auto text-[10px] ${
              t.isError ? "text-[var(--color-fail)]" : "text-[var(--color-ok)]"
            }`}
          >
            {t.isError ? "✗" : "✓"}
          </span>
        )}
      </div>
      {t.result && (
        <div className="mt-0.5 line-clamp-3 pl-4 font-mono text-[10px] text-[var(--color-fg-dim)]">
          ⎿ {t.result}
        </div>
      )}
    </div>
  );
}

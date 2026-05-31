// Live transcript of all harness events, grouped by agent_id with a
// per-agent filter. Renders the four content event types the harness
// emits: agent_started, agent_message, agent_thinking, agent_tool,
// agent_completed. The Activity tab is the "what is the agent saying
// right now" view — the same firehose the terminal viewer shows.
//
// Two header strips track the things that don't belong in the firehose:
//   - Running now: open process_started events with no matching
//     process_result yet, with a ticking elapsed timer.
//   - Heartbeats: one row per agent, replaced on each new heartbeat tick.

import { useEffect, useMemo, useRef, useState } from "react";
import { getEventStream } from "../lib/events";
import type {
  HarnessEvent,
  HeartbeatEvent,
  ProcessStartedEvent,
  ProcessResultEvent,
} from "../lib/types";

const RENDER_TYPES = new Set([
  "agent_started",
  "agent_completed",
  "agent_message",
  "agent_thinking",
  "agent_tool",
]);

// Threshold (ms) after which a still-open process pulses red. Anything
// running for several minutes with no result is worth eyeballing.
const STUCK_MS = 5 * 60 * 1000;
// Heartbeat freshness thresholds (ms).
const HB_FRESH_MS = 45 * 1000;
const HB_STALE_MS = 5 * 60 * 1000;

// Build initial running-procs + heartbeats maps from the ring buffer
// snapshot so toggling tabs preserves the strips. Walks the buffer once.
function seedFromSnapshot(buf: HarnessEvent[]): {
  running: Map<string, ProcessStartedEvent>;
  heartbeats: Map<string, HeartbeatEvent>;
} {
  const running = new Map<string, ProcessStartedEvent>();
  const heartbeats = new Map<string, HeartbeatEvent>();
  for (const e of buf) {
    if (e.type === "process_started") {
      const ev = e as ProcessStartedEvent;
      running.set(ev.process_id, ev);
    } else if (e.type === "process_result") {
      const ev = e as ProcessResultEvent;
      running.delete(ev.process_id);
    } else if (e.type === "heartbeat") {
      const ev = e as HeartbeatEvent;
      heartbeats.set(ev.agent_id, ev);
    }
  }
  return { running, heartbeats };
}

export function ActivityView() {
  const stream = getEventStream();
  // Hydrate from the singleton's ring buffer so toggling tabs (which
  // unmounts this component) doesn't wipe the transcript. The ring
  // persists for the lifetime of the page.
  const [events, setEvents] = useState<HarnessEvent[]>(() =>
    stream.snapshot().filter((e) => RENDER_TYPES.has(e.type)),
  );
  const [running, setRunning] = useState<Map<string, ProcessStartedEvent>>(
    () => seedFromSnapshot(stream.snapshot()).running,
  );
  const [heartbeats, setHeartbeats] = useState<Map<string, HeartbeatEvent>>(
    () => seedFromSnapshot(stream.snapshot()).heartbeats,
  );
  const [filter, setFilter] = useState<string>("");          // agent_id filter
  const [hideThinking, setHideThinking] = useState(false);
  const [paused, setPaused] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const pendingRef = useRef<HarnessEvent[]>([]);

  // Subscribe once. Buffer firehose events when paused so resuming shows
  // everything; running-procs + heartbeats always update (the strips
  // would be useless if frozen, and they don't scroll).
  useEffect(() => {
    const off = stream.subscribe((e) => {
      if (e.type === "process_started") {
        const ev = e as ProcessStartedEvent;
        setRunning((prev) => {
          const next = new Map(prev);
          next.set(ev.process_id, ev);
          return next;
        });
        return;
      }
      if (e.type === "process_result") {
        const ev = e as ProcessResultEvent;
        setRunning((prev) => {
          if (!prev.has(ev.process_id)) return prev;
          const next = new Map(prev);
          next.delete(ev.process_id);
          return next;
        });
        return;
      }
      if (e.type === "heartbeat") {
        const ev = e as HeartbeatEvent;
        setHeartbeats((prev) => {
          const next = new Map(prev);
          next.set(ev.agent_id, ev);
          return next;
        });
        return;
      }
      if (!RENDER_TYPES.has(e.type)) return;
      if (paused) {
        pendingRef.current.push(e);
        return;
      }
      setEvents((prev) => prev.concat(e));
    });
    return off;
  }, [paused, stream]);

  // Drop heartbeats for agents whose run completed. Without this, a
  // stale heartbeat from a finished phase would sit in the strip until
  // page reload. agent_completed is the signal — we already see it in
  // the firehose path above.
  useEffect(() => {
    const off = stream.subscribe((e) => {
      if (e.type !== "agent_completed") return;
      setHeartbeats((prev) => {
        if (!prev.has(e.agent_id)) return prev;
        const next = new Map(prev);
        next.delete(e.agent_id);
        return next;
      });
    });
    return off;
  }, [stream]);

  // 1Hz tick so elapsed timers in both strips refresh in real time.
  // Only runs while something is open — otherwise idle screens are silent.
  const [, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (running.size === 0 && heartbeats.size === 0) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [running.size, heartbeats.size]);

  // On unpause, flush the buffer.
  useEffect(() => {
    if (paused || pendingRef.current.length === 0) return;
    const drained = pendingRef.current;
    pendingRef.current = [];
    setEvents((prev) => prev.concat(drained));
  }, [paused]);

  const agents = useMemo(() => {
    const set = new Set<string>();
    for (const e of events) set.add(e.agent_id);
    return Array.from(set).sort();
  }, [events]);

  const filtered = useMemo(() => {
    return events.filter((e) => {
      if (filter && e.agent_id !== filter) return false;
      if (hideThinking && e.type === "agent_thinking") return false;
      return true;
    });
  }, [events, filter, hideThinking]);

  // Auto-scroll to bottom on new events unless the user has scrolled up.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const stickyToBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    if (stickyToBottom) el.scrollTop = el.scrollHeight;
  }, [filtered]);

  return (
    <div className="flex h-full flex-col">
      <div className="flex h-9 shrink-0 items-center gap-2 border-b border-[var(--color-border)] px-3 text-[11px]">
        <span className="text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
          Activity
        </span>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-1.5 py-0.5 text-[11px]"
        >
          <option value="">All agents ({agents.length})</option>
          {agents.map((a) => (
            <option key={a} value={a}>{a}</option>
          ))}
        </select>
        <label className="flex cursor-pointer items-center gap-1 text-[10px] text-[var(--color-fg-muted)]">
          <input
            type="checkbox"
            checked={hideThinking}
            onChange={(e) => setHideThinking(e.target.checked)}
            className="accent-[var(--color-accent)]"
          />
          hide thinking
        </label>
        <span className="ml-auto text-[10px] tabular-nums text-[var(--color-fg-dim)]">
          {filtered.length}{events.length !== filtered.length ? ` / ${events.length}` : ""}
        </span>
        <button
          type="button"
          onClick={() => setPaused((v) => !v)}
          className={`rounded px-2 py-0.5 text-[10px] ${paused ? "bg-[var(--color-warn)]/20 text-[var(--color-warn)]" : "text-[var(--color-fg-dim)] hover:bg-[var(--color-surface-2)]"}`}
        >
          {paused ? `paused (${pendingRef.current.length})` : "pause"}
        </button>
        <button
          type="button"
          onClick={() => {
            stream.clearBuffer();
            setEvents([]);
          }}
          className="rounded px-2 py-0.5 text-[10px] text-[var(--color-fg-dim)] hover:bg-[var(--color-surface-2)]"
        >
          clear
        </button>
      </div>

      <RunningProcsStrip running={running} />
      <HeartbeatStrip heartbeats={heartbeats} />

      <div ref={scrollRef} className="min-h-0 flex-1 overflow-auto px-3 py-2 font-mono text-[11px] leading-relaxed">
        {filtered.length === 0 ? (
          <div className="mt-8 text-center text-[var(--color-fg-dim)]">
            Waiting for events…<br />
            <span className="text-[10px]">
              Start the harness with <code className="rounded bg-[var(--color-surface-2)] px-1 py-px">research-builder &lt;paper&gt;</code>
            </span>
          </div>
        ) : (
          filtered.map((e, i) => <EventRow key={`${e.ts}-${i}`} ev={e} />)
        )}
      </div>
    </div>
  );
}

function EventRow({ ev }: { ev: HarnessEvent }) {
  // Compact timestamp (HH:MM:SS).
  const time = ev.ts.includes("T") ? ev.ts.split("T")[1].slice(0, 8) : ev.ts.slice(0, 8);

  const colorCls =
    ev.type === "agent_message"
      ? "text-[var(--color-fg)]"
      : ev.type === "agent_thinking"
        ? "text-[var(--color-fg-dim)] italic"
        : ev.type === "agent_tool"
          ? "text-[var(--color-accent)]"
          : ev.type === "agent_started" || ev.type === "agent_completed"
            ? "text-[var(--color-run)]"
            : "text-[var(--color-fg-muted)]";

  const label = labelFor(ev);
  const body = bodyFor(ev);

  return (
    <div className="group flex items-baseline gap-2 border-b border-[var(--color-border)]/30 py-0.5 last:border-b-0">
      <span className="w-16 shrink-0 text-[10px] text-[var(--color-fg-dim)] tabular-nums">{time}</span>
      <span className="w-24 shrink-0 truncate text-[10px] text-[var(--color-fg-muted)]">{ev.agent_id}</span>
      <span className={`w-12 shrink-0 text-[10px] uppercase tracking-wider ${colorCls}`}>{label}</span>
      <span className={`min-w-0 flex-1 whitespace-pre-wrap break-words ${colorCls}`}>{body}</span>
    </div>
  );
}

function RunningProcsStrip({ running }: { running: Map<string, ProcessStartedEvent> }) {
  if (running.size === 0) return null;
  // Oldest first — the longest-running tool calls are most likely stuck.
  const rows = Array.from(running.values()).sort((a, b) => a.ts.localeCompare(b.ts));
  const now = Date.now();
  return (
    <div className="shrink-0 border-b border-[var(--color-border)] bg-[var(--color-surface)]/40 px-3 py-1">
      <div className="mb-1 flex items-center gap-2 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
        <span>Running</span>
        <span className="text-[var(--color-fg-dim)]">·</span>
        <span className="tabular-nums">{running.size}</span>
      </div>
      <ul className="space-y-0.5 font-mono text-[11px]">
        {rows.map((ev) => {
          const startedMs = Date.parse(ev.ts);
          const ageMs = Math.max(0, now - startedMs);
          const stuck = ageMs >= STUCK_MS;
          const dotCls = stuck
            ? "bg-[var(--color-fail)] animate-pulse"
            : "bg-[var(--color-run,#3b82f6)] animate-pulse";
          return (
            <li
              key={ev.process_id}
              className="flex items-baseline gap-2"
              title={`process_id ${ev.process_id}`}
            >
              <span className={`mt-1 inline-block h-1.5 w-1.5 shrink-0 rounded-full ${dotCls}`} />
              <span className="w-16 shrink-0 tabular-nums text-[10px] text-[var(--color-fg-dim)]">
                T+{formatElapsed(ageMs)}
              </span>
              <span className="w-24 shrink-0 truncate text-[10px] text-[var(--color-fg-muted)]">
                {ev.agent_id}
              </span>
              <span className="w-16 shrink-0 truncate text-[10px] text-[var(--color-accent)]">
                {ev.tool_name}
              </span>
              <span className="min-w-0 flex-1 truncate text-[var(--color-fg)]">
                {ev.summary || ev.command || ev.file_path || ""}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function HeartbeatStrip({ heartbeats }: { heartbeats: Map<string, HeartbeatEvent> }) {
  if (heartbeats.size === 0) return null;
  // Orchestrator first, then phases alphabetically.
  const rows = Array.from(heartbeats.values()).sort((a, b) => {
    if (a.agent_id === "orchestrator") return -1;
    if (b.agent_id === "orchestrator") return 1;
    return a.agent_id.localeCompare(b.agent_id);
  });
  const now = Date.now();
  return (
    <div className="shrink-0 border-b border-[var(--color-border)] bg-[var(--color-surface)]/40 px-3 py-1">
      <div className="mb-1 flex items-center gap-2 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
        <span>Heartbeat</span>
      </div>
      <ul className="space-y-0.5 font-mono text-[11px]">
        {rows.map((hb) => {
          const tickMs = Date.parse(hb.ts);
          const ageMs = Math.max(0, now - tickMs);
          const dotCls =
            ageMs <= HB_FRESH_MS
              ? "bg-[var(--color-ok)]"
              : ageMs <= HB_STALE_MS
                ? "bg-[var(--color-warn,#eab308)]"
                : "bg-[var(--color-fail)]";
          const deltasStr = formatDeltas(hb.deltas);
          return (
            <li key={hb.agent_id} className="flex items-baseline gap-2">
              <span className={`mt-1 inline-block h-1.5 w-1.5 shrink-0 rounded-full ${dotCls}`} />
              <span className="w-16 shrink-0 tabular-nums text-[10px] text-[var(--color-fg-dim)]">
                {formatElapsed(ageMs)} ago
              </span>
              <span className="w-24 shrink-0 truncate text-[10px] text-[var(--color-fg-muted)]">
                {hb.agent_id}
              </span>
              <span className="w-24 shrink-0 truncate text-[10px] text-[var(--color-fg-dim)]">
                {hb.open_block ?? "idle"}
              </span>
              <span className="min-w-0 flex-1 truncate text-[var(--color-fg)]">
                {deltasStr || `last: ${hb.last_msg_type}`}
              </span>
              <span className="shrink-0 tabular-nums text-[10px] text-[var(--color-fg-dim)]">
                {Math.round(hb.elapsed_s)}s · {hb.msgs_count}msg
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const sec = s % 60;
  if (m < 60) return `${m}m${sec.toString().padStart(2, "0")}s`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return `${h}h${rm.toString().padStart(2, "0")}m`;
}

function formatDeltas(deltas: Record<string, number>): string {
  const entries = Object.entries(deltas);
  if (entries.length === 0) return "";
  entries.sort((a, b) => b[1] - a[1]);
  return entries.map(([k, n]) => `+${n} ${k}`).join(", ");
}

function labelFor(ev: HarnessEvent): string {
  if (ev.type === "agent_started") return "start";
  if (ev.type === "agent_completed") return "end";
  if (ev.type === "agent_tool") return "tool";
  if (ev.type === "agent_thinking") return "think";
  if (ev.type === "agent_message") {
    const role = (ev.role as string) ?? "?";
    return role.slice(0, 5);
  }
  return ev.type.replace("agent_", "").slice(0, 6);
}

function bodyFor(ev: HarnessEvent): string {
  if (ev.type === "agent_started") {
    const kind = (ev.kind as string) ?? "";
    const title = (ev.title as string) ?? "";
    return [kind && `[${kind}]`, title].filter(Boolean).join(" ");
  }
  if (ev.type === "agent_completed") {
    const status = (ev.status as string) ?? "";
    return status ? `status=${status}` : "";
  }
  if (ev.type === "agent_tool") {
    return (ev.summary as string) ?? (ev.tool as string) ?? "";
  }
  // agent_thinking and agent_message: prefer .text.
  const text = (ev.text as string) ?? "";
  return text;
}

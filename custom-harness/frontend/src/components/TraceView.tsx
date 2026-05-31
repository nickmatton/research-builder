// Hierarchical trace view: phase → attempt → role → tool calls, with
// errors flagged at every level and a detail pane on the right showing the
// selected node's full payload (tool I/O, crash diagnostics, transcript).
//
// Replaces the flat ActivityView for the "what is the harness doing right
// now" use case. The summary AgentsView (status/duration/cost) stays as
// the at-a-glance companion.

import { useEffect, useMemo, useState } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { glyphFor, shortAgentId } from "../lib/agents";
import { api } from "../lib/api";
import { getEventStream } from "../lib/events";
import { buildTrace } from "../lib/trace";
import type {
  AttemptNode,
  CrashNode,
  FileWriteEvent,
  HarnessEvent,
  OrchestratorNode,
  Phase,
  PhaseNode,
  StepNode,
  ToolCallNode,
  TraceMessage,
  TraceRoot,
} from "../lib/types";
import { StatusDot } from "./StatusDot";
import { FileDiffModal } from "./FileDiffModal";

// Anchor for the right detail pane. Encodes which node was selected via a
// stable path string so re-renders preserve the selection.
type SelectionPath =
  | { kind: "run" }
  | { kind: "orchestrator" }
  | { kind: "phase"; phaseId: string }
  | { kind: "attempt"; phaseId: string; retryNum: string }
  | { kind: "step"; phaseId: string; retryNum: string; stepId: string }
  | { kind: "tool"; processId: string }
  | { kind: "crash"; crashId: string };

const RELEVANT_EVENT_TYPES = new Set([
  "agent_started",
  "agent_completed",
  "agent_message",
  "agent_thinking",
  "process_started",
  "process_result",
  "agent_crashed",
]);

export function TraceView() {
  const [events, setEvents] = useState<HarnessEvent[]>([]);
  const [phases, setPhases] = useState<Phase[]>([]);
  const [selection, setSelection] = useState<SelectionPath>({ kind: "run" });
  const [errorsOnly, setErrorsOnly] = useState(false);

  // ─── Live data: events + phases ───────────────────────────────────────
  useEffect(() => {
    const stream = getEventStream();
    setEvents(stream.snapshot().filter((e) => RELEVANT_EVENT_TYPES.has(e.type)));
    const off = stream.subscribe((e) => {
      if (!RELEVANT_EVENT_TYPES.has(e.type)) return;
      setEvents((prev) => prev.concat(e));
    });
    return off;
  }, []);

  useEffect(() => {
    const load = () => api.phases().then((r) => setPhases(r.phases)).catch(() => undefined);
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, []);

  const trace = useMemo(() => buildTrace(events, phases), [events, phases]);

  // ─── Auto-follow the latest live activity ─────────────────────────────
  // Until the user clicks a node themselves, selection tracks the newest
  // live step / failing step so the right pane always shows "what's
  // happening now." Once they pick something manually, we stop tracking
  // (autoFollow=false) so the UI doesn't pull the rug out from under them.
  const [autoFollow, setAutoFollow] = useState(true);
  useEffect(() => {
    if (!autoFollow) return;
    const target = pickInitialSelection(trace);
    if (target) setSelection(target);
  }, [trace, autoFollow]);

  const onUserSelect = (s: SelectionPath) => {
    setAutoFollow(false);
    setSelection(s);
  };

  const selected = useMemo(() => resolveSelection(trace, selection), [trace, selection]);

  return (
    <div className="flex h-full flex-col">
      <TraceHeader
        trace={trace}
        errorsOnly={errorsOnly}
        onToggleErrors={() => setErrorsOnly((v) => !v)}
        autoFollow={autoFollow}
        onToggleFollow={() => setAutoFollow((v) => !v)}
      />
      <div className="min-h-0 flex-1">
        <PanelGroup direction="horizontal" autoSaveId="rb:trace-split">
          <Panel defaultSize={45} minSize={25}>
            <div className="h-full overflow-auto">
              <TraceTree
                trace={trace}
                selection={selection}
                onSelect={onUserSelect}
                errorsOnly={errorsOnly}
              />
            </div>
          </Panel>
          <PanelResizeHandle className="group relative w-px bg-[var(--color-border)] transition-colors data-[resize-handle-state=hover]:bg-[var(--color-accent)] data-[resize-handle-state=drag]:bg-[var(--color-accent)]">
            <div className="absolute inset-y-0 -left-1.5 w-3 cursor-col-resize" />
          </PanelResizeHandle>
          <Panel defaultSize={55} minSize={25}>
            <div className="h-full overflow-auto bg-[var(--color-bg)]">
              <NodeDetail trace={trace} selected={selected} />
            </div>
          </Panel>
        </PanelGroup>
      </div>
    </div>
  );
}

// ─── Header strip ────────────────────────────────────────────────────────

function TraceHeader({
  trace,
  errorsOnly,
  onToggleErrors,
  autoFollow,
  onToggleFollow,
}: {
  trace: TraceRoot;
  errorsOnly: boolean;
  onToggleErrors: () => void;
  autoFollow: boolean;
  onToggleFollow: () => void;
}) {
  const errors = trace.errorCount;
  // Orchestrator activity without phases = planning hasn't completed yet.
  // Show a friendly "planning…" indicator instead of bare "0 phases".
  const planning =
    trace.phases.length === 0 &&
    (trace.orchestrator.messages.length > 0 ||
      trace.orchestrator.children.length > 0);
  return (
    <div className="flex h-9 shrink-0 items-center gap-3 border-b border-[var(--color-border)] px-3 text-[11px]">
      {trace.phases.length > 0 ? (
        <span className="text-[var(--color-fg-muted)]">
          {trace.phases.length} phase{trace.phases.length === 1 ? "" : "s"}
        </span>
      ) : planning ? (
        <span className="flex items-center gap-1.5 text-[var(--color-fg-muted)]">
          <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-[var(--color-accent)]" />
          Planning…
        </span>
      ) : (
        <span className="text-[var(--color-fg-dim)]">Waiting for run to start…</span>
      )}
      {errors > 0 ? (
        <span className="rounded-full bg-[var(--color-fail)]/15 px-2 py-0.5 text-[10px] font-medium text-[var(--color-fail)]">
          {errors} error{errors === 1 ? "" : "s"}
        </span>
      ) : (
        <span className="text-[10px] text-[var(--color-fg-dim)]">no errors</span>
      )}
      <div className="ml-auto flex items-center gap-3">
        <button
          type="button"
          onClick={onToggleFollow}
          title={autoFollow ? "Selection follows the latest live step" : "Manually selected — click to re-enable follow"}
          className={[
            "flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[10px]",
            autoFollow
              ? "bg-[var(--color-ok)]/15 text-[var(--color-ok)]"
              : "text-[var(--color-fg-muted)] hover:bg-[var(--color-surface-2)]",
          ].join(" ")}
        >
          <span
            className={`h-1.5 w-1.5 rounded-full ${autoFollow ? "bg-[var(--color-ok)] shadow-[0_0_6px_var(--color-ok)]" : "bg-[var(--color-border-strong)]"}`}
          />
          {autoFollow ? "following" : "paused"}
        </button>
        <label className="flex cursor-pointer items-center gap-1 text-[10px] text-[var(--color-fg-muted)]">
          <input
            type="checkbox"
            checked={errorsOnly}
            onChange={onToggleErrors}
            className="accent-[var(--color-fail)]"
          />
          errors only
        </label>
      </div>
    </div>
  );
}

// ─── Tree (left pane) ────────────────────────────────────────────────────

function TraceTree({
  trace,
  selection,
  onSelect,
  errorsOnly,
}: {
  trace: TraceRoot;
  selection: SelectionPath;
  onSelect: (s: SelectionPath) => void;
  errorsOnly: boolean;
}) {
  // Truly empty: no orchestrator activity yet, no phases. Most common
  // between "upload paper" and "first event lands" — typically 1–2s.
  const truly_empty =
    trace.phases.length === 0 &&
    trace.orchestrator.messages.length === 0 &&
    trace.orchestrator.children.length === 0;
  if (truly_empty) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center">
        <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[var(--color-accent)]" />
        <div className="text-xs text-[var(--color-fg-muted)]">
          Waiting for the first agent event…
        </div>
        <div className="max-w-[18rem] text-[10px] leading-relaxed text-[var(--color-fg-dim)]">
          The harness is booting. If this stays empty for more than a minute,
          check <code className="rounded bg-[var(--color-surface-2)] px-1 py-px">logs/pipeline.log</code>{" "}
          from the Files tab.
        </div>
      </div>
    );
  }
  return (
    <ul className="py-1 text-xs">
      <OrchestratorRow
        node={trace.orchestrator}
        selection={selection}
        onSelect={onSelect}
        errorsOnly={errorsOnly}
      />
      {trace.phases.map((p) => (
        <PhaseRow
          key={p.id}
          node={p}
          selection={selection}
          onSelect={onSelect}
          errorsOnly={errorsOnly}
        />
      ))}
    </ul>
  );
}

function OrchestratorRow({
  node,
  selection,
  onSelect,
  errorsOnly,
}: {
  node: OrchestratorNode;
  selection: SelectionPath;
  onSelect: (s: SelectionPath) => void;
  errorsOnly: boolean;
}) {
  if (errorsOnly && node.errorCount === 0) return null;
  const isSelected = selection.kind === "orchestrator";
  const [open, setOpen] = useState(true);
  return (
    <li>
      <Row
        depth={0}
        open={open}
        canToggle={node.children.length > 0}
        onToggle={() => setOpen((v) => !v)}
        onSelect={() => onSelect({ kind: "orchestrator" })}
        selected={isSelected}
        glyph="🧭"
        label="orchestrator"
        muted="planning + dispatch"
        errorCount={node.errorCount}
      />
      {open && node.children.length > 0 && (
        <ul>
          {node.children.map((c) => (
            <ToolRow
              key={c.id}
              node={c}
              depth={1}
              selection={selection}
              onSelect={onSelect}
              errorsOnly={errorsOnly}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

function PhaseRow({
  node,
  selection,
  onSelect,
  errorsOnly,
}: {
  node: PhaseNode;
  selection: SelectionPath;
  onSelect: (s: SelectionPath) => void;
  errorsOnly: boolean;
}) {
  if (errorsOnly && node.errorCount === 0) return null;
  const isSelected = selection.kind === "phase" && selection.phaseId === node.phaseId;
  // Default open if in_progress or has errors; collapsed for completed
  // happy-path phases to keep the tree scannable on large runs.
  const initialOpen = node.status === "in_progress" || node.errorCount > 0;
  const [open, setOpen] = useState(initialOpen);
  return (
    <li>
      <Row
        depth={0}
        open={open}
        canToggle={node.attempts.length > 0}
        onToggle={() => setOpen((v) => !v)}
        onSelect={() => onSelect({ kind: "phase", phaseId: node.phaseId })}
        selected={isSelected}
        statusDot={node.status}
        label={node.phaseId}
        muted={node.title !== node.phaseId ? node.title : undefined}
        trailing={
          <span className="text-[10px] tabular-nums text-[var(--color-fg-dim)]">
            {node.attempts.length}×
          </span>
        }
        errorCount={node.errorCount}
      />
      {open && (
        <ul>
          {node.attempts.map((a) => (
            <AttemptRow
              key={a.id}
              node={a}
              phaseId={node.phaseId}
              selection={selection}
              onSelect={onSelect}
              errorsOnly={errorsOnly}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

function AttemptRow({
  node,
  phaseId,
  selection,
  onSelect,
  errorsOnly,
}: {
  node: AttemptNode;
  phaseId: string;
  selection: SelectionPath;
  onSelect: (s: SelectionPath) => void;
  errorsOnly: boolean;
}) {
  if (errorsOnly && node.errorCount === 0) return null;
  const isSelected =
    selection.kind === "attempt" &&
    selection.phaseId === phaseId &&
    selection.retryNum === node.retryNum;
  const initialOpen = node.errorCount > 0 || node.steps.some((s) => s.live);
  const [open, setOpen] = useState(initialOpen);
  return (
    <li>
      <Row
        depth={1}
        open={open}
        canToggle={node.steps.length > 0}
        onToggle={() => setOpen((v) => !v)}
        onSelect={() =>
          onSelect({ kind: "attempt", phaseId, retryNum: node.retryNum })
        }
        selected={isSelected}
        statusDot={node.status}
        label={`attempt ${node.retryNum}`}
        muted={node.steps.length === 0 ? "(no steps yet)" : undefined}
        errorCount={node.errorCount}
      />
      {open && (
        <ul>
          {node.steps.map((s) => (
            <StepRow
              key={s.id}
              node={s}
              phaseId={phaseId}
              retryNum={node.retryNum}
              selection={selection}
              onSelect={onSelect}
              errorsOnly={errorsOnly}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

function StepRow({
  node,
  phaseId,
  retryNum,
  selection,
  onSelect,
  errorsOnly,
}: {
  node: StepNode;
  phaseId: string;
  retryNum: string;
  selection: SelectionPath;
  onSelect: (s: SelectionPath) => void;
  errorsOnly: boolean;
}) {
  if (errorsOnly && node.errorCount === 0) return null;
  const isSelected =
    selection.kind === "step" &&
    selection.phaseId === phaseId &&
    selection.retryNum === retryNum &&
    selection.stepId === node.id;
  const initialOpen = node.live || node.errorCount > 0;
  const [open, setOpen] = useState(initialOpen);
  return (
    <li>
      <Row
        depth={2}
        open={open}
        canToggle={node.children.length > 0}
        onToggle={() => setOpen((v) => !v)}
        onSelect={() =>
          onSelect({ kind: "step", phaseId, retryNum, stepId: node.id })
        }
        selected={isSelected}
        statusDot={node.live ? "in_progress" : node.status}
        glyph={glyphFor(node.role)}
        label={node.role}
        muted={
          node.durationS != null
            ? `${node.durationS.toFixed(1)}s`
            : node.live
              ? "running…"
              : undefined
        }
        errorCount={node.errorCount}
      />
      {open && node.children.length > 0 && (
        <ul>
          {node.children.map((c) =>
            c.kind === "tool" ? (
              <ToolRow
                key={c.id}
                node={c}
                depth={3}
                selection={selection}
                onSelect={onSelect}
                errorsOnly={errorsOnly}
              />
            ) : (
              <CrashRow
                key={c.id}
                node={c}
                depth={3}
                selection={selection}
                onSelect={onSelect}
              />
            ),
          )}
        </ul>
      )}
    </li>
  );
}

function ToolRow({
  node,
  depth,
  selection,
  onSelect,
  errorsOnly,
}: {
  node: ToolCallNode;
  depth: number;
  selection: SelectionPath;
  onSelect: (s: SelectionPath) => void;
  errorsOnly: boolean;
}) {
  if (errorsOnly && !node.isError) return null;
  const isSelected = selection.kind === "tool" && selection.processId === node.id;
  const labelCls = node.isError ? "text-[var(--color-fail)]" : "";
  return (
    <li>
      <Row
        depth={depth}
        glyph="↳"
        label={node.toolName}
        labelCls={labelCls}
        muted={shortToolSummary(node)}
        onSelect={() => onSelect({ kind: "tool", processId: node.id })}
        selected={isSelected}
        trailing={
          node.isError ? (
            <span className="rounded bg-[var(--color-fail)]/15 px-1 text-[9px] uppercase tracking-wider text-[var(--color-fail)]">
              error
            </span>
          ) : node.output != null ? null : (
            <span className="text-[9px] text-[var(--color-fg-dim)]">…</span>
          )
        }
      />
    </li>
  );
}

function CrashRow({
  node,
  depth,
  selection,
  onSelect,
}: {
  node: CrashNode;
  depth: number;
  selection: SelectionPath;
  onSelect: (s: SelectionPath) => void;
}) {
  const isSelected = selection.kind === "crash" && selection.crashId === node.id;
  return (
    <li>
      <Row
        depth={depth}
        glyph="✗"
        labelCls="text-[var(--color-fail)] font-medium"
        label="CRASH"
        muted={`${node.errorType}: ${truncate(node.error, 80)}`}
        onSelect={() => onSelect({ kind: "crash", crashId: node.id })}
        selected={isSelected}
        trailing={
          <span className="rounded bg-[var(--color-fail)]/15 px-1 text-[9px] uppercase tracking-wider text-[var(--color-fail)]">
            crash
          </span>
        }
      />
    </li>
  );
}

// ─── Row primitive ───────────────────────────────────────────────────────

function Row({
  depth,
  open,
  canToggle,
  onToggle,
  onSelect,
  selected,
  glyph,
  statusDot,
  label,
  labelCls,
  muted,
  trailing,
  errorCount,
}: {
  depth: number;
  open?: boolean;
  canToggle?: boolean;
  onToggle?: () => void;
  onSelect: () => void;
  selected: boolean;
  glyph?: string;
  statusDot?: string;
  label: string;
  labelCls?: string;
  muted?: string;
  trailing?: React.ReactNode;
  errorCount?: number;
}) {
  return (
    <div
      className={`group flex items-center gap-1.5 py-0.5 pr-2 text-left text-xs ${
        selected
          ? "bg-[var(--color-surface-2)]"
          : "hover:bg-[var(--color-surface-2)]/60"
      }`}
      style={{ paddingLeft: 8 + depth * 12 }}
    >
      <button
        type="button"
        onClick={onToggle}
        disabled={!canToggle}
        className={`inline-block w-3 shrink-0 text-center text-[10px] text-[var(--color-fg-dim)] ${
          canToggle ? "" : "invisible"
        }`}
        aria-label={open ? "Collapse" : "Expand"}
      >
        {canToggle ? (open ? "▾" : "▸") : ""}
      </button>
      <button
        type="button"
        onClick={onSelect}
        className="flex min-w-0 flex-1 items-center gap-1.5 text-left"
      >
        {statusDot ? <StatusDot status={statusDot} /> : null}
        {glyph ? <span className="shrink-0 text-[11px]">{glyph}</span> : null}
        <span className={`truncate font-mono ${labelCls ?? "text-[var(--color-fg)]"}`}>
          {label}
        </span>
        {muted ? (
          <span className="truncate text-[10px] text-[var(--color-fg-muted)]">
            {muted}
          </span>
        ) : null}
      </button>
      {errorCount && errorCount > 0 ? (
        <span className="shrink-0 rounded-full bg-[var(--color-fail)]/15 px-1.5 py-0.5 text-[9px] font-medium tabular-nums text-[var(--color-fail)]">
          {errorCount}
        </span>
      ) : null}
      {trailing ?? null}
    </div>
  );
}

// ─── Detail pane (right) ─────────────────────────────────────────────────

function NodeDetail({
  trace,
  selected,
}: {
  trace: TraceRoot;
  selected: ResolvedSelection;
}) {
  if (!selected || selected.kind === "run") return <RunDetail trace={trace} />;
  if (selected.kind === "orchestrator") return <TranscriptCard
    title="Orchestrator"
    subtitle="planning + dispatch"
    messages={selected.node.messages}
    toolCount={selected.node.children.length}
    errorCount={selected.node.errorCount}
  />;
  if (selected.kind === "phase") return <PhaseDetail node={selected.node} />;
  if (selected.kind === "attempt") return <AttemptDetail node={selected.node} />;
  if (selected.kind === "step") return <StepDetail node={selected.node} />;
  if (selected.kind === "tool") return <ToolDetail node={selected.node} />;
  return <CrashDetail node={selected.node} contextTools={selected.contextTools} />;
}

function RunDetail({ trace }: { trace: TraceRoot }) {
  return (
    <div className="space-y-3 p-4 text-xs">
      <DetailHeader title="Run" subtitle="select a node to drill in" />
      <KV k="phases" v={String(trace.phases.length)} />
      <KV k="errors" v={String(trace.errorCount)} highlight={trace.errorCount > 0 ? "fail" : undefined} />
      <KV k="orchestrator tools" v={String(trace.orchestrator.children.length)} />
    </div>
  );
}

function PhaseDetail({ node }: { node: PhaseNode }) {
  return (
    <div className="space-y-3 p-4 text-xs">
      <DetailHeader title={node.title} subtitle={node.phaseId} />
      <KV k="status" v={node.status} />
      <KV k="attempts" v={String(node.attempts.length)} />
      <KV k="errors" v={String(node.errorCount)} highlight={node.errorCount > 0 ? "fail" : undefined} />
    </div>
  );
}

function AttemptDetail({ node }: { node: AttemptNode }) {
  return (
    <div className="space-y-3 p-4 text-xs">
      <DetailHeader title={`attempt ${node.retryNum}`} subtitle={node.status} />
      <KV k="steps" v={String(node.steps.length)} />
      <KV k="errors" v={String(node.errorCount)} highlight={node.errorCount > 0 ? "fail" : undefined} />
      <div className="pt-2 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
        steps
      </div>
      <ul className="space-y-1">
        {node.steps.map((s) => (
          <li key={s.id} className="flex items-center gap-2">
            <span>{glyphFor(s.role)}</span>
            <span className="font-mono text-[var(--color-fg-muted)]">{s.role}</span>
            <span className="text-[10px] text-[var(--color-fg-dim)]">{s.status}</span>
            {s.durationS != null ? (
              <span className="ml-auto text-[10px] tabular-nums text-[var(--color-fg-dim)]">
                {s.durationS.toFixed(1)}s
              </span>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}

function StepDetail({ node }: { node: StepNode }) {
  return (
    <TranscriptCard
      title={node.role}
      subtitle={`${node.agentId} · ${node.status}${node.live ? " · running" : ""}`}
      messages={node.messages}
      toolCount={node.children.length}
      errorCount={node.errorCount}
      durationS={node.durationS}
      costUsd={node.costUsd}
    />
  );
}

const WRITE_TOOL_NAMES = new Set(["Write", "Edit", "MultiEdit", "NotebookEdit"]);

function ToolDetail({ node }: { node: ToolCallNode }) {
  const sub = node.command
    ? `Bash`
    : node.filePath
      ? node.filePath
      : node.toolName;

  // For write tools, find the matching file_write event in the event
  // buffer. The harness emits it at process_result time with the same
  // process_id, so it should always be there once the tool completes.
  const [diffEv, setDiffEv] = useState<FileWriteEvent | null>(null);
  const isWrite = WRITE_TOOL_NAMES.has(node.toolName);
  const fileWrite = useMemo<FileWriteEvent | null>(() => {
    if (!isWrite) return null;
    const events = getEventStream().snapshot();
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i];
      if (
        e.type === "file_write" &&
        (e as unknown as FileWriteEvent).process_id === node.id
      ) {
        return e as unknown as FileWriteEvent;
      }
    }
    return null;
  }, [isWrite, node.id, node.completedTs]);

  return (
    <div className="space-y-3 p-4 text-xs">
      <DetailHeader title={node.toolName} subtitle={sub} fail={node.isError} />
      <KV k="started" v={fmtTs(node.startedTs)} />
      {node.completedTs ? <KV k="completed" v={fmtTs(node.completedTs)} /> : <KV k="status" v="…running" />}
      {node.command ? <PreBlock title="command" body={node.command} /> : null}
      {node.filePath && !node.command ? <KV k="file_path" v={node.filePath} /> : null}
      {fileWrite && (
        <div>
          <button
            type="button"
            onClick={() => setDiffEv(fileWrite)}
            className="flex items-center gap-1.5 rounded-md border border-[var(--color-border)] bg-[var(--color-surface-2)] px-2.5 py-1 text-[11px] hover:border-[var(--color-accent)] hover:text-[var(--color-accent)]"
          >
            <span className="font-mono text-[10px] text-[var(--color-fg-dim)]">⇆</span>
            View diff
            {fileWrite.before === null && (
              <span className="text-[10px] text-[var(--color-ok)]">new file</span>
            )}
          </button>
        </div>
      )}
      {node.output != null ? (
        <PreBlock
          title={node.isError ? "stderr / output" : "output"}
          body={node.output || "(empty)"}
          fail={node.isError}
        />
      ) : null}
      {diffEv && <FileDiffModal ev={diffEv} onClose={() => setDiffEv(null)} />}
    </div>
  );
}

function CrashDetail({
  node,
  contextTools,
}: {
  node: CrashNode;
  contextTools: ToolCallNode[];
}) {
  return (
    <div className="space-y-3 p-4 text-xs">
      <DetailHeader title="agent crashed" subtitle={node.errorType} fail />
      <PreBlock title="error" body={node.error} fail />
      <KV k="turns completed" v={String(node.turnsCompleted)} />
      <KV k="time" v={fmtTs(node.ts)} />
      {node.messagesReceived.length > 0 ? (
        <PreBlock
          title="last messages received (newest last)"
          body={node.messagesReceived.join(" → ")}
        />
      ) : null}
      {node.stderrTail.length > 0 ? (
        <PreBlock title="stderr tail" body={node.stderrTail.join("\n")} />
      ) : null}
      {contextTools.length > 0 ? (
        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
            last {contextTools.length} tool call{contextTools.length === 1 ? "" : "s"} before crash
          </div>
          <ul className="space-y-1">
            {contextTools.map((t) => (
              <li
                key={t.id}
                className={`rounded bg-[var(--color-surface-2)] px-2 py-1 font-mono text-[11px] ${
                  t.isError ? "text-[var(--color-fail)]" : "text-[var(--color-fg-muted)]"
                }`}
              >
                <span className="font-medium">{t.toolName}</span>
                <span className="ml-2 opacity-70">{shortToolSummary(t)}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

// ─── Detail-pane primitives ──────────────────────────────────────────────

function DetailHeader({
  title,
  subtitle,
  fail,
}: {
  title: string;
  subtitle?: string;
  fail?: boolean;
}) {
  return (
    <div className="border-b border-[var(--color-border)] pb-2">
      <div className={`font-mono text-sm ${fail ? "text-[var(--color-fail)]" : "text-[var(--color-fg)]"}`}>
        {title}
      </div>
      {subtitle ? (
        <div className="mt-0.5 text-[10px] text-[var(--color-fg-muted)]">{subtitle}</div>
      ) : null}
    </div>
  );
}

function KV({ k, v, highlight }: { k: string; v: string; highlight?: "fail" }) {
  const cls = highlight === "fail" ? "text-[var(--color-fail)]" : "text-[var(--color-fg)]";
  return (
    <div className="flex items-baseline gap-3 text-[11px]">
      <span className="w-28 shrink-0 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
        {k}
      </span>
      <span className={`font-mono ${cls}`}>{v}</span>
    </div>
  );
}

function PreBlock({
  title,
  body,
  fail,
}: {
  title: string;
  body: string;
  fail?: boolean;
}) {
  return (
    <div className="space-y-1">
      <div className="text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
        {title}
      </div>
      <pre
        className={`max-h-72 overflow-auto whitespace-pre-wrap break-words rounded bg-[var(--color-surface-2)] p-2 font-mono text-[11px] leading-relaxed ${
          fail ? "text-[var(--color-fail)]" : "text-[var(--color-fg)]"
        }`}
      >
        {body}
      </pre>
    </div>
  );
}

function TranscriptCard({
  title,
  subtitle,
  messages,
  toolCount,
  errorCount,
  durationS,
  costUsd,
}: {
  title: string;
  subtitle?: string;
  messages: TraceMessage[];
  toolCount: number;
  errorCount: number;
  durationS?: number | null;
  costUsd?: number | null;
}) {
  return (
    <div className="space-y-3 p-4 text-xs">
      <DetailHeader title={title} subtitle={subtitle} />
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
        <KV k="tools" v={String(toolCount)} />
        <KV k="errors" v={String(errorCount)} highlight={errorCount > 0 ? "fail" : undefined} />
        {durationS != null ? <KV k="duration" v={`${durationS.toFixed(1)}s`} /> : null}
        {costUsd != null && costUsd > 0 ? <KV k="cost" v={`$${costUsd.toFixed(3)}`} /> : null}
      </div>
      <div className="space-y-1">
        <div className="text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
          transcript ({messages.length})
        </div>
        {messages.length === 0 ? (
          <div className="text-[11px] text-[var(--color-fg-dim)]">
            no messages yet
          </div>
        ) : (
          <ul className="space-y-1">
            {messages.slice(-50).map((m, i) => (
              <li
                key={`${m.ts}-${i}`}
                className={`rounded bg-[var(--color-surface-2)] px-2 py-1 text-[11px] leading-relaxed ${
                  m.role === "system"
                    ? "text-[var(--color-warn)]"
                    : m.thinking
                      ? "italic text-[var(--color-fg-dim)]"
                      : "text-[var(--color-fg)]"
                }`}
              >
                <span className="mr-2 text-[9px] uppercase tracking-wider text-[var(--color-fg-dim)]">
                  {m.thinking ? "think" : m.role.slice(0, 5)}
                </span>
                <span className="whitespace-pre-wrap break-words">{m.text}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

// ─── Selection plumbing ──────────────────────────────────────────────────

type ResolvedSelection =
  | { kind: "run" }
  | { kind: "orchestrator"; node: OrchestratorNode }
  | { kind: "phase"; node: PhaseNode }
  | { kind: "attempt"; node: AttemptNode }
  | { kind: "step"; node: StepNode }
  | { kind: "tool"; node: ToolCallNode }
  | { kind: "crash"; node: CrashNode; contextTools: ToolCallNode[] }
  | null;

function resolveSelection(trace: TraceRoot, path: SelectionPath): ResolvedSelection {
  switch (path.kind) {
    case "run":
      return { kind: "run" };
    case "orchestrator":
      return { kind: "orchestrator", node: trace.orchestrator };
    case "phase": {
      const phase = trace.phases.find((p) => p.phaseId === path.phaseId);
      return phase ? { kind: "phase", node: phase } : null;
    }
    case "attempt": {
      const phase = trace.phases.find((p) => p.phaseId === path.phaseId);
      const attempt = phase?.attempts.find((a) => a.retryNum === path.retryNum);
      return attempt ? { kind: "attempt", node: attempt } : null;
    }
    case "step": {
      const phase = trace.phases.find((p) => p.phaseId === path.phaseId);
      const attempt = phase?.attempts.find((a) => a.retryNum === path.retryNum);
      const step = attempt?.steps.find((s) => s.id === path.stepId);
      return step ? { kind: "step", node: step } : null;
    }
    case "tool": {
      const node = findTool(trace, path.processId);
      return node ? { kind: "tool", node } : null;
    }
    case "crash": {
      const crash = findCrash(trace, path.crashId);
      if (!crash) return null;
      return {
        kind: "crash",
        node: crash.node,
        contextTools: tailTools(crash.parentChildren, crash.node.id, 5),
      };
    }
  }
}

function findTool(trace: TraceRoot, processId: string): ToolCallNode | null {
  for (const c of trace.orchestrator.children) if (c.id === processId) return c;
  for (const p of trace.phases) {
    for (const a of p.attempts) {
      for (const s of a.steps) {
        for (const c of s.children) {
          if (c.kind === "tool" && c.id === processId) return c;
        }
      }
    }
  }
  return null;
}

function findCrash(
  trace: TraceRoot,
  crashId: string,
): { node: CrashNode; parentChildren: Array<ToolCallNode | CrashNode> } | null {
  for (const p of trace.phases) {
    for (const a of p.attempts) {
      for (const s of a.steps) {
        for (const c of s.children) {
          if (c.kind === "crash" && c.id === crashId) {
            return { node: c, parentChildren: s.children };
          }
        }
      }
    }
  }
  return null;
}

function tailTools(
  siblings: Array<ToolCallNode | CrashNode>,
  crashId: string,
  n: number,
): ToolCallNode[] {
  const idx = siblings.findIndex((c) => c.id === crashId);
  const upTo = idx < 0 ? siblings.length : idx;
  const tools: ToolCallNode[] = [];
  for (let i = upTo - 1; i >= 0 && tools.length < n; i--) {
    const c = siblings[i];
    if (c.kind === "tool") tools.push(c);
  }
  return tools.reverse();
}

function pickInitialSelection(trace: TraceRoot): SelectionPath | null {
  // Prefer the first failing step, else the first live step, else nothing.
  for (const p of trace.phases) {
    for (const a of p.attempts) {
      for (const s of a.steps) {
        if (s.errorCount > 0) {
          return { kind: "step", phaseId: p.phaseId, retryNum: a.retryNum, stepId: s.id };
        }
      }
    }
  }
  for (const p of trace.phases) {
    for (const a of p.attempts) {
      for (const s of a.steps) {
        if (s.live) {
          return { kind: "step", phaseId: p.phaseId, retryNum: a.retryNum, stepId: s.id };
        }
      }
    }
  }
  return null;
}

// ─── Misc helpers ────────────────────────────────────────────────────────

function shortToolSummary(node: ToolCallNode): string {
  if (node.command) {
    return truncate(node.command, 80);
  }
  if (node.filePath) return node.filePath;
  return truncate(node.summary, 80);
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function fmtTs(ts: string): string {
  const ix = ts.indexOf("T");
  return ix >= 0 ? ts.slice(ix + 1, ix + 9) : ts;
}

// Silence unused import noise from re-export probe; keeps shortAgentId
// referenced in case a future detail card wants it.
void shortAgentId;

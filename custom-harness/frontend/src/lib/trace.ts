// Pure event → trace-tree builder.
//
// Inputs:
//   - events:  the rolling stream from /ws/events (any order is fine; we
//              sort by `ts` once and walk from oldest to newest)
//   - phases:  the latest /api/phases response (gives the phase + attempt +
//              role spine, regardless of whether events have streamed yet)
//
// Output: a TraceRoot. The function is pure + idempotent — recomputing on
// every new event is cheap because event counts are in the hundreds, not
// millions. The caller memos on (events.length, phases.length) plus a
// monotonic "events bump" counter.
//
// Routing rules:
//   - `agent_id === "orchestrator"` events go under the OrchestratorNode.
//   - `agent_id === "phase:<id>"` events go under the latest attempt's
//     active step for that phase. The active step starts when an
//     `agent_started` (kind=subagent) for that role arrives and ends at
//     the matching `agent_completed`. Before the first agent_started lands
//     on a fresh attempt, we route to the FIRST step (refiner) so the
//     pre-roll noise has somewhere to go.
//   - `process_started` opens a ToolCallNode keyed by process_id.
//     `process_result` fills `output` + `isError`. Orphan results are
//     dropped silently.
//   - `agent_message` + `agent_thinking` accumulate in the step's
//     (or orchestrator's) `messages` list — rendered inside the detail
//     pane, not as tree leaves.
//   - `agent_crashed` becomes a CrashNode under the active step.
//
// Error roll-up: after the walk, we recurse over the tree once to sum
// descendant errorCount into every container node.

import type {
  AgentCrashedEvent,
  AttemptNode,
  CrashNode,
  HarnessEvent,
  OrchestratorNode,
  Phase,
  PhaseNode,
  ProcessResultEvent,
  ProcessStartedEvent,
  StepNode,
  ToolCallNode,
  TraceMessage,
  TraceRoot,
} from "./types";

// Events we route into messages (detail-pane mini-transcript). Skipping
// `agent_thinking` would lose the trail of why an agent went down a path;
// it's the most useful debugging signal for cryptic crashes.
const MESSAGE_EVENT_TYPES = new Set(["agent_message", "agent_thinking"]);

interface BuildState {
  // Set of active step ids per phase (`phase:<id>` → currently-running role).
  activeStepByPhase: Map<string, StepNode>;
  // process_id → the ToolCallNode it opened; lets process_result attach.
  pendingTools: Map<string, ToolCallNode>;
}

export function buildTrace(events: HarnessEvent[], phases: Phase[]): TraceRoot {
  const orchestrator: OrchestratorNode = {
    kind: "orchestrator",
    id: "orchestrator",
    children: [],
    messages: [],
    errorCount: 0,
  };

  const phaseNodes: PhaseNode[] = phases.map((p) => seedPhase(p));
  const phaseIndex = new Map<string, PhaseNode>(
    phaseNodes.map((p) => [p.phaseId, p]),
  );

  const state: BuildState = {
    activeStepByPhase: new Map(),
    pendingTools: new Map(),
  };

  // Iterate in ts order. Events can arrive in clusters out of order during
  // WS reconnects, so always sort defensively.
  const sorted = events.slice().sort((a, b) => (a.ts < b.ts ? -1 : a.ts > b.ts ? 1 : 0));
  for (const e of sorted) {
    if (e.agent_id === "orchestrator") {
      applyToOrchestrator(orchestrator, e, state);
    } else if (typeof e.agent_id === "string" && e.agent_id.startsWith("phase:")) {
      const phaseId = e.agent_id.slice("phase:".length);
      const phase = phaseIndex.get(phaseId);
      if (!phase) continue; // event for an unknown phase — phase manifest hasn't caught up yet
      applyToPhase(phase, e, state);
    }
    // Other agent_ids (e.g. "fphase:<id>" for file-planner steps) are not
    // visualized here; they're better surfaced elsewhere in the UI.
  }

  rollUpErrors(orchestrator);
  for (const p of phaseNodes) rollUpErrors(p);

  const errorCount =
    orchestrator.errorCount + phaseNodes.reduce((s, p) => s + p.errorCount, 0);

  return {
    kind: "run",
    id: "run",
    orchestrator,
    phases: phaseNodes,
    errorCount,
  };
}

// ─── Helpers ─────────────────────────────────────────────────────────────

function seedPhase(p: Phase): PhaseNode {
  const attempts: AttemptNode[] = p.attempts.map((a) => ({
    kind: "attempt",
    id: `${p.phase_id}:${a.retry_num}`,
    retryNum: a.retry_num,
    status: deriveAttemptStatus(a.steps),
    steps: a.steps.map((s, i) => ({
      kind: "step",
      id: `${p.phase_id}:${a.retry_num}:${s.role}:${i}`,
      role: s.role,
      status: s.status,
      durationS: s.duration_s ?? null,
      costUsd: s.cost_usd ?? null,
      agentId: `phase:${p.phase_id}`,
      children: [],
      messages: [],
      errorCount: 0,
      live: false,
    })),
    errorCount: 0,
  }));
  return {
    kind: "phase",
    id: p.phase_id,
    phaseId: p.phase_id,
    title: p.title,
    status: p.status,
    attempts,
    errorCount: 0,
  };
}

function deriveAttemptStatus(steps: Phase["attempts"][number]["steps"]): string {
  if (steps.length === 0) return "pending";
  const last = steps[steps.length - 1];
  return last.status;
}

function applyToOrchestrator(
  orch: OrchestratorNode,
  e: HarnessEvent,
  state: BuildState,
): void {
  if (e.type === "process_started") {
    const pe = e as ProcessStartedEvent;
    const node: ToolCallNode = {
      kind: "tool",
      id: pe.process_id,
      toolName: pe.tool_name,
      summary: pe.summary || pe.tool_name,
      command: pe.command ?? null,
      filePath: pe.file_path ?? null,
      isError: false,
      startedTs: pe.ts,
      errorCount: 0,
    };
    orch.children.push(node);
    state.pendingTools.set(pe.process_id, node);
    return;
  }
  if (e.type === "process_result") {
    attachResult(e as ProcessResultEvent, state);
    return;
  }
  if (MESSAGE_EVENT_TYPES.has(e.type)) {
    orch.messages.push(toMessage(e));
  }
}

function applyToPhase(
  phase: PhaseNode,
  e: HarnessEvent,
  state: BuildState,
): void {
  const attempt = phase.attempts[phase.attempts.length - 1];
  if (!attempt) return; // no manifest yet

  // Step routing — pin to the live step if one is set, otherwise fall back
  // to the first step in the attempt so pre-roll events have a home.
  let step = state.activeStepByPhase.get(phase.phaseId);
  if (!step || !attempt.steps.includes(step)) {
    step = attempt.steps[0];
    if (!step) return; // attempt with no steps yet
  }

  if (e.type === "agent_started") {
    // Find the next not-yet-live step in this attempt — gives a clean
    // refiner → researcher → builder → verifier handoff even though
    // agent_started doesn't carry a role explicitly.
    const next = attempt.steps.find((s) => !s.live && s.status !== "ok" && s.status !== "completed");
    const target = next ?? attempt.steps[attempt.steps.length - 1];
    if (target) {
      target.live = true;
      state.activeStepByPhase.set(phase.phaseId, target);
    }
    return;
  }
  if (e.type === "agent_completed") {
    if (step) step.live = false;
    state.activeStepByPhase.delete(phase.phaseId);
    return;
  }
  if (e.type === "process_started") {
    const pe = e as ProcessStartedEvent;
    const node: ToolCallNode = {
      kind: "tool",
      id: pe.process_id,
      toolName: pe.tool_name,
      summary: pe.summary || pe.tool_name,
      command: pe.command ?? null,
      filePath: pe.file_path ?? null,
      isError: false,
      startedTs: pe.ts,
      errorCount: 0,
    };
    step.children.push(node);
    state.pendingTools.set(pe.process_id, node);
    return;
  }
  if (e.type === "process_result") {
    attachResult(e as ProcessResultEvent, state);
    return;
  }
  if (e.type === "agent_crashed") {
    const ce = e as AgentCrashedEvent;
    const node: CrashNode = {
      kind: "crash",
      id: `crash:${ce.ts}`,
      errorType: ce.error_type,
      error: ce.error,
      messagesReceived: ce.messages_received ?? [],
      stderrTail: ce.stderr_tail ?? [],
      turnsCompleted: ce.turns_completed ?? 0,
      ts: ce.ts,
      errorCount: 1,
    };
    step.children.push(node);
    return;
  }
  if (MESSAGE_EVENT_TYPES.has(e.type)) {
    step.messages.push(toMessage(e));
  }
}

function attachResult(e: ProcessResultEvent, state: BuildState): void {
  const node = state.pendingTools.get(e.process_id);
  if (!node) return; // orphan — drop silently
  node.completedTs = e.ts;
  node.output = e.output;
  node.isError = e.is_error;
  node.errorCount = e.is_error ? 1 : 0;
  state.pendingTools.delete(e.process_id);
}

function toMessage(e: HarnessEvent): TraceMessage {
  if (e.type === "agent_thinking") {
    return { ts: e.ts, role: "assistant", text: (e.text as string) ?? "", thinking: true };
  }
  const role = ((e.role as string) ?? "assistant") as TraceMessage["role"];
  return { ts: e.ts, role, text: (e.text as string) ?? "" };
}

// Sum descendant errorCount up into every container. Called once per
// buildTrace() after the event walk completes.
function rollUpErrors(node: OrchestratorNode | PhaseNode | AttemptNode | StepNode): number {
  if (node.kind === "step") {
    let total = 0;
    for (const c of node.children) total += c.errorCount;
    node.errorCount = total;
    return total;
  }
  if (node.kind === "attempt") {
    let total = 0;
    for (const s of node.steps) total += rollUpErrors(s);
    node.errorCount = total;
    return total;
  }
  if (node.kind === "phase") {
    let total = 0;
    for (const a of node.attempts) total += rollUpErrors(a);
    node.errorCount = total;
    return total;
  }
  // orchestrator
  let total = 0;
  for (const c of node.children) total += c.errorCount;
  node.errorCount = total;
  return total;
}

// ─── Tree filters / selectors ────────────────────────────────────────────

// True if `node` or any descendant has at least one error. Used by the
// "errors only" toggle to keep a node visible.
export function hasErrorInSubtree(
  node: TraceRoot | OrchestratorNode | PhaseNode | AttemptNode | StepNode | ToolCallNode | CrashNode,
): boolean {
  return (node.errorCount ?? 0) > 0;
}

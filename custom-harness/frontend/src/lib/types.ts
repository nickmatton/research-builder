// Shared API types. Mirrors research_builder.web.api response shapes.
// Kept hand-written (rather than codegen'd) for now — the surface is
// small enough that the duplication cost is lower than the codegen
// toolchain cost. If/when this grows past ~12 endpoints, revisit.

export type WorkspaceState = "empty" | "ready" | "running" | "finished";

export interface PipelineStatus {
  state: "idle" | "running" | "finished";
  pid: number | null;
  exit_code: number | null;
}

export interface WorkspaceInfo {
  name: string | null;
  path: string | null;
  paper_path: string | null;
  has_spec: boolean;
  state: WorkspaceState;
  runs_dir: string;
  events_path: string;
  commands_path: string;
  pipeline: PipelineStatus;
  /** Whether the backend was booted with `--dev`; the Launcher uses
   *  this to default the dev-mode checkbox. */
  dev_mode_default: boolean;
}

export type PhaseStatus = "pending" | "in_progress" | "completed" | "failed";

export interface Artifact {
  name: string;
  file_path: string;
}

export interface AttemptStep {
  role: string;
  status: string;
  duration_s?: number;
  cost_usd?: number | null;
  record_path?: string;
}

export interface Attempt {
  retry_num: string;
  steps: AttemptStep[];
}

export interface Phase {
  phase_id: string;
  title: string;
  status: PhaseStatus;
  dependencies: string[];
  inputs: Artifact[];
  outputs: Artifact[];
  attempts: Attempt[];
}

export interface SpecState {
  metadata?: {
    paper_id?: string;
    paper_title?: string;
    paper_url?: string | null;
    created_at?: string;
    last_modified?: string;
  };
  phases?: Array<Record<string, unknown>>;
  dependency_graph?: Record<string, string[]>;
  plan?: { nodes?: Array<Record<string, unknown>>; files?: unknown[] };
}

export interface SpecResponse {
  state: SpecState | null;
  spec_md: string | null;
}

export interface FileEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size: number | null;
}

export interface FilesResponse {
  path: string;
  entries: FileEntry[];
}

// Events streamed over /ws/events. The harness emits many event types
// (phase_started, sub_agent_started, tool_call, etc.); we type the
// common envelope and pass the rest through as ``unknown``.
export interface HarnessEvent {
  ts: string;
  type: string;
  agent_id: string;
  parent_id: string | null;
  [key: string]: unknown;
}

// Emitted when an agent calls Read on the paper PDF. The page range is
// 1-indexed and inclusive; (null, null) means "whole document" — the
// agent omitted the `pages` argument.
export interface PaperReadEvent extends HarnessEvent {
  type: "paper_read";
  page_start: number | null;
  page_end: number | null;
  paper_path: string;
}

// process_started: emitted at the moment an agent invokes a tool. The
// matching process_result (same process_id) carries the tool's output.
export interface ProcessStartedEvent extends HarnessEvent {
  type: "process_started";
  process_id: string;
  tool_name: string;
  summary?: string | null;
  command?: string | null;
  file_path?: string | null;
}

export interface ProcessResultEvent extends HarnessEvent {
  type: "process_result";
  process_id: string;
  is_error: boolean;
  output: string;
}

// Emitted every ~20-30s by the orchestrator + sub-agent heartbeat loops.
// Lets the UI render a "is this agent alive?" tick without polluting the
// activity firehose. Replaces the prior heartbeat for the same agent_id.
export interface HeartbeatEvent extends HarnessEvent {
  type: "heartbeat";
  elapsed_s: number;
  interval_s: number;
  open_block: string | null;
  deltas: Record<string, number>;
  last_msg_type: string;
  msgs_count: number;
  // Orchestrator-only.
  role?: string;
  result_chars?: number;
}

// Emitted at process_result time for Write/Edit/MultiEdit/NotebookEdit.
// Carries snapshots so the UI can render a before/after diff without
// re-reading the workspace (paths may get clobbered by subsequent
// writes). ``before`` is null when the file didn't exist (new-file case).
// Content is capped at 256KB per side; truncated flags indicate the head
// was kept.
export interface FileWriteEvent extends HarnessEvent {
  type: "file_write";
  process_id: string;
  tool_name: string;
  file_path: string;
  before: string | null;
  before_truncated: boolean;
  after: string | null;
  after_truncated: boolean;
  is_error: boolean;
}

// Structured companion to the existing agent_message system-role crash
// notice. Carries the diagnostics dict (error type, last-N message types,
// stderr tail) for the Trace detail pane.
export interface AgentCrashedEvent extends HarnessEvent {
  type: "agent_crashed";
  error_type: string;
  error: string;
  messages_received: string[];
  stderr_tail: string[];
  turns_completed: number;
}

// ─── Trace tree shapes ───────────────────────────────────────────────────
// The Trace view's left-pane tree. Built by lib/trace.ts from the rolling
// event stream + the /api/phases manifest. Discriminated by `kind`.

export type TraceNodeKind =
  | "run"
  | "orchestrator"
  | "phase"
  | "attempt"
  | "step"
  | "tool"
  | "crash"
  | "message";

export interface TraceMessage {
  ts: string;
  role: "assistant" | "system" | "user";
  text: string;
  thinking?: boolean;
}

export interface ToolCallNode {
  kind: "tool";
  id: string; // process_id (or synthetic if orphan)
  toolName: string;
  summary: string;
  command?: string | null;
  filePath?: string | null;
  output?: string;
  isError: boolean;
  startedTs: string;
  completedTs?: string;
  errorCount: number; // 0 or 1; included so roll-up math is uniform
}

export interface CrashNode {
  kind: "crash";
  id: string;
  errorType: string;
  error: string;
  messagesReceived: string[];
  stderrTail: string[];
  turnsCompleted: number;
  ts: string;
  errorCount: 1;
}

export interface StepNode {
  kind: "step";
  id: string; // phase_id + role + attempt
  role: string;
  status: string;
  durationS?: number | null;
  costUsd?: number | null;
  agentId: string; // "phase:<id>" — for filtering events
  children: Array<ToolCallNode | CrashNode>;
  messages: TraceMessage[];
  errorCount: number;
  live: boolean; // agent_started seen, agent_completed not yet
}

export interface AttemptNode {
  kind: "attempt";
  id: string; // phase_id + retry_num
  retryNum: string;
  status: string; // derived from terminal-step status
  steps: StepNode[];
  errorCount: number;
}

export interface PhaseNode {
  kind: "phase";
  id: string; // phase_id
  phaseId: string;
  title: string;
  status: PhaseStatus;
  attempts: AttemptNode[];
  errorCount: number;
}

export interface OrchestratorNode {
  kind: "orchestrator";
  id: "orchestrator";
  children: ToolCallNode[];
  messages: TraceMessage[];
  errorCount: number;
}

export interface TraceRoot {
  kind: "run";
  id: "run";
  orchestrator: OrchestratorNode;
  phases: PhaseNode[];
  errorCount: number;
}

export type TraceNode =
  | TraceRoot
  | OrchestratorNode
  | PhaseNode
  | AttemptNode
  | StepNode
  | ToolCallNode
  | CrashNode;

export interface AgentRole {
  role: string;
  tools: string[];
  glyph: string;
}

export interface AgentsResponse {
  roles: AgentRole[];
  mcp_servers: string[];
}

export interface RefinedSpecResponse {
  phase_id: string;
  exists: boolean;
  path: string;
  content: string;
  source?: string | null;
}

// ─── Per-section specs (upfront authoring) ──────────────────────────────

export interface Citation {
  page: number;
  section?: string | null;
  quote?: string | null;
}

export interface AcceptanceCriterion {
  text: string;
  source: Citation;
}

export type CritiqueVerdict = "verified" | "questionable" | "missing_citations";

export interface SectionSummary {
  phase_id: string;
  title: string;
  goal: string;
  criteria_count: number;
  citations_count: number;
  critique_verdict: CritiqueVerdict | null;
  md_path: string | null;
  last_modified: number;
}

export interface SectionsResponse {
  sections: SectionSummary[];
}

export interface SectionDetail {
  phase_id: string;
  title: string;
  goal: string;
  spec_markdown: string;
  acceptance_criteria: AcceptanceCriterion[];
  citations: Citation[];
  md_path: string;
}

export interface SectionCritique {
  phase_id: string;
  verdict: CritiqueVerdict;
  reasons: string[];
  reviewed_at?: string;
}

// ─── Claims / verification / final report ──────────────────────────────

export interface ClaimSource {
  table?: string | null;
  figure?: string | null;
  section?: string | null;
  page?: number | null;
  verbatim?: string;
}

export interface Claim {
  claim_id: string;
  metric: string;
  value: number;
  tolerance: number;
  unit: string;
  dataset: string;
  condition: string;
  source: ClaimSource;
  phase_id: string;
  notes?: string;
}

export interface ClaimsResponse {
  claims: Claim[];
  exists: boolean;
}

export interface VerificationReport {
  filename: string;
  path: string;
  modified: number;
  payload: Record<string, unknown>;
}

export interface VerificationResponse {
  phase_id: string;
  reports: VerificationReport[];
  latest: VerificationReport | null;
}

export interface ReportResponse {
  exists: boolean;
  path: string | null;
  content: string | null;
}

// ─── Cloud compute (Lambda Cloud) ──────────────────────────────────────

export type ComputeStatus = "active" | "terminated";

export interface ComputeUpgrade {
  from_instance_id: string;
  from_instance_type: string | null;
  reason: string;
  ts: string;
}

export interface ComputeInstance {
  instance_id: string;
  phase_id: string;
  instance_type: string;
  public_ip: string;
  ssh_user: string;
  ssh_key_path: string;
  region: string;
  hourly_rate_usd: number;
  estimated_hours: number;
  estimated_cost_usd: number;
  actual_hours: number | null;
  actual_cost_usd: number | null;
  status: ComputeStatus;
  provisioned_at: string;
  terminated_at: string | null;
  ledger_entry_id: string;
  work_dir: string;
  upgrades: ComputeUpgrade[];
}

export interface ComputeBudget {
  cap_usd: number;
  projected_total_usd: number;
}

export interface ComputeListResponse {
  instances: ComputeInstance[];
  budget: ComputeBudget | null;
  updated_at?: string;
}

export interface ComputeRemoteRun {
  process_id: string;
  started_at: string;
  finished_at: string | null;
  command: string;
  output: string;
  is_error: boolean;
}

export interface ComputeDetailResponse extends ComputeInstance {
  ssh_command: string | null;
  lambda_console_url: string;
  remote_runs: ComputeRemoteRun[];
}

// Streamed compute lifecycle events.
export interface ComputeProvisionedEvent extends HarnessEvent {
  type: "compute_provisioned";
  instance_id: string;
  instance_type: string;
  public_ip: string;
  hourly_rate_usd: number;
  estimated_hours: number;
  estimated_cost_usd: number;
  work_dir: string;
}

export interface ComputeTerminatedEvent extends HarnessEvent {
  type: "compute_terminated";
  instance_id: string;
  actual_hours: number | null;
  actual_cost_usd: number | null;
}

export interface ComputeUpgradedEvent extends HarnessEvent {
  type: "compute_upgraded";
  from_instance_id: string;
  to_instance_id: string;
  instance_type: string;
  public_ip: string;
  reason: string;
}

// ─── New event types from the upfront-authoring revamp ─────────────────

export interface SectionSpecStartedEvent extends HarnessEvent {
  type: "section_spec_started";
  phase_id: string;
  title: string;
}

export interface SectionSpecCompletedEvent extends HarnessEvent {
  type: "section_spec_completed";
  phase_id: string;
  path: string;
  criteria_count: number;
}

export interface SectionSpecCritiquedEvent extends HarnessEvent {
  type: "section_spec_critiqued";
  phase_id: string;
  verdict: CritiqueVerdict;
}

export interface ArtifactCreatedEvent extends HarnessEvent {
  type: "artifact_created";
  artifact_type: string;
  path: string;
  producer: string;
  phase_id?: string;
}

export type DiffLineType = "context" | "add" | "remove";

export interface DiffLine {
  type: DiffLineType;
  text: string;
}

export interface InvalidatedPhase {
  phase_id: string;
  title: string;
  roles: string[];
  reason: "direct" | "cascade";
}

export interface CascadePreview {
  phase_id: string;
  before_agent: string;
  diff: DiffLine[];
  invalidated: InvalidatedPhase[];
  error?: string;
}

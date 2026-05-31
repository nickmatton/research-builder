import { useEffect, useState } from "react";
import { ROLE_GLYPH } from "../lib/agents";
import { api } from "../lib/api";
import { getEventStream } from "../lib/events";
import type { AgentsResponse, Attempt, AttemptStep, Phase, PhaseStatus } from "../lib/types";
import { StatusDot } from "./StatusDot";

export function AgentsView() {
  const [phases, setPhases] = useState<Phase[]>([]);
  const [agents, setAgents] = useState<AgentsResponse | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [showRegistry, setShowRegistry] = useState(false);

  const reload = () => {
    api.phases().then((r) => setPhases(r.phases)).catch(() => undefined);
  };

  useEffect(() => {
    reload();
    api.agents().then(setAgents).catch(() => undefined);
    // Keep the tree fresh on live runs. Any phase-affecting event
    // refetches; sub_agent_started/completed events also imply new
    // steps in the active phase's manifest.
    const off = getEventStream().subscribe((e) => {
      if (
        e.type === "phase_started" ||
        e.type === "phase_completed" ||
        e.type === "phase_failed" ||
        e.type === "retry_launched" ||
        e.type === "agent_started" ||
        e.type === "agent_completed" ||
        e.type === "spec_amended"
      ) {
        reload();
      }
    });
    return off;
  }, []);

  const toggle = (id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const onRetry = async (phase_id: string) => {
    const rationale = window.prompt(`Retry rationale for ${phase_id}:`, "operator-triggered retry");
    if (rationale == null) return;
    try {
      await api.commands.forceRetry(phase_id, rationale);
    } catch (e) {
      window.alert(e instanceof Error ? e.message : "force_retry failed");
    }
  };

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b border-[var(--color-border)] px-3 py-2 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
        <span>Agents</span>
        <button
          type="button"
          onClick={() => setShowRegistry((v) => !v)}
          className="rounded px-1.5 py-0.5 text-[10px] normal-case tracking-normal hover:bg-[var(--color-surface-2)] hover:text-[var(--color-fg-muted)]"
        >
          {showRegistry ? "hide registry" : "show registry"}
        </button>
      </div>

      {showRegistry && agents && (
        <div className="border-b border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2">
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
            Available roles (allowed tools, identical across roles)
          </div>
          <ul className="space-y-1">
            {agents.roles.map((r) => (
              <li key={r.role} className="text-xs">
                <span className="mr-1">{r.glyph}</span>
                <span className="font-mono text-[var(--color-fg)]">{r.role}</span>
                <span className="ml-2 text-[10px] text-[var(--color-fg-dim)]">
                  {r.tools.length} tools
                </span>
              </li>
            ))}
          </ul>
          {agents.mcp_servers.length > 0 && (
            <div className="mt-1 text-[10px] text-[var(--color-fg-dim)]">
              MCP: {agents.mcp_servers.join(", ")}
            </div>
          )}
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-auto py-1">
        {phases.length === 0 ? (
          <div className="px-3 py-2 text-xs text-[var(--color-fg-dim)]">
            no phases yet
          </div>
        ) : (
          <ul>
            {phases.map((p) => (
              <PhaseRow
                key={p.phase_id}
                phase={p}
                expanded={expanded.has(p.phase_id)}
                onToggle={() => toggle(p.phase_id)}
                onRetry={() => onRetry(p.phase_id)}
              />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function PhaseRow({
  phase,
  expanded,
  onToggle,
  onRetry,
}: {
  phase: Phase;
  expanded: boolean;
  onToggle: () => void;
  onRetry: () => void;
}) {
  return (
    <li>
      <div className="group flex items-center gap-2 px-3 py-1 hover:bg-[var(--color-surface-2)]">
        <button
          type="button"
          onClick={onToggle}
          className="flex flex-1 items-center gap-2 text-left"
        >
          <span className="inline-block w-3 text-center text-[10px] text-[var(--color-fg-dim)]">
            {expanded ? "▾" : "▸"}
          </span>
          <StatusDot status={phase.status} />
          <span className="font-mono text-[11px] text-[var(--color-fg-muted)]">
            {phase.phase_id}
          </span>
          <span className="truncate text-xs">{phase.title}</span>
        </button>
        <span className="text-[10px] tabular-nums text-[var(--color-fg-dim)]">
          {phase.attempts.length || 0}×
        </span>
        <button
          type="button"
          onClick={onRetry}
          className="invisible rounded px-1.5 py-0.5 text-[10px] text-[var(--color-fg-dim)] hover:bg-[var(--color-surface)] hover:text-[var(--color-fg-muted)] group-hover:visible"
          title="Force retry this phase"
        >
          retry
        </button>
      </div>
      {expanded && (
        <ul className="border-l border-[var(--color-border)] pl-3 mx-3 my-1">
          {phase.attempts.length === 0 ? (
            <li className="px-2 py-1 text-[11px] text-[var(--color-fg-dim)]">
              no attempts yet
            </li>
          ) : (
            phase.attempts.map((a) => <AttemptRow key={a.retry_num} attempt={a} />)
          )}
        </ul>
      )}
    </li>
  );
}

function AttemptRow({ attempt }: { attempt: Attempt }) {
  return (
    <li className="mb-1">
      <div className="px-2 py-0.5 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
        retry {attempt.retry_num}
      </div>
      <ul>
        {attempt.steps.length === 0 ? (
          <li className="px-3 py-0.5 text-[11px] text-[var(--color-fg-dim)]">(no steps)</li>
        ) : (
          attempt.steps.map((s, i) => <StepRow key={i} step={s} />)
        )}
      </ul>
    </li>
  );
}

function StepRow({ step }: { step: AttemptStep }) {
  const glyph = ROLE_GLYPH[step.role] ?? "•";
  const statusCls =
    step.status === "ok" || step.status === "completed"
      ? "text-[var(--color-ok)]"
      : step.status === "failed" || step.status === "error"
        ? "text-[var(--color-fail)]"
        : "text-[var(--color-fg-muted)]";
  return (
    <li className="flex items-baseline gap-2 px-3 py-0.5 text-[11px]">
      <span>{glyph}</span>
      <span className={`font-mono ${statusCls}`}>{step.role}</span>
      {step.duration_s != null && (
        <span className="tabular-nums text-[10px] text-[var(--color-fg-dim)]">
          {step.duration_s.toFixed(1)}s
        </span>
      )}
      {step.cost_usd != null && step.cost_usd > 0 && (
        <span className="tabular-nums text-[10px] text-[var(--color-fg-dim)]">
          ${step.cost_usd.toFixed(3)}
        </span>
      )}
    </li>
  );
}

// Surface PhaseStatus in the type-only chain so unused-import noise stays out.
export type _ReexportProbe = PhaseStatus;

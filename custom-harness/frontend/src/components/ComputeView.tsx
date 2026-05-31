// Compute tab. Lists every Lambda Cloud instance the harness has launched
// this run (live + historical) and lets the operator inspect one in detail
// — SSH command, cost so far, lifetime, upgrade history.
//
// Source of truth: GET /api/compute (mirror of logs/compute_instances.json
// written by CloudProvisioner). Live changes come over the events WS as
// compute_provisioned / compute_terminated / compute_upgraded — we just
// refetch on any of those rather than mutating local state piecemeal.

import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import { getEventStream } from "../lib/events";
import type {
  ComputeBudget,
  ComputeDetailResponse,
  ComputeInstance,
  ComputeRemoteRun,
} from "../lib/types";

const COMPUTE_EVENT_TYPES = new Set([
  "compute_provisioned",
  "compute_terminated",
  "compute_upgraded",
]);

export function ComputeView() {
  const [instances, setInstances] = useState<ComputeInstance[] | null>(null);
  const [budget, setBudget] = useState<ComputeBudget | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const reload = () =>
    api
      .compute
      .list()
      .then((r) => {
        setInstances(r.instances);
        setBudget(r.budget);
        setErr(null);
      })
      .catch((e: Error) => setErr(e.message));

  useEffect(() => {
    reload();
    const off = getEventStream().subscribe((e) => {
      if (COMPUTE_EVENT_TYPES.has(e.type)) reload();
    });
    return off;
  }, []);

  // Auto-select the first live instance, falling back to the most recent.
  useEffect(() => {
    if (!instances || instances.length === 0) {
      setSelectedId(null);
      return;
    }
    if (selectedId && instances.some((i) => i.instance_id === selectedId)) return;
    const live = instances.find((i) => i.status === "active");
    setSelectedId((live ?? instances[0]).instance_id);
  }, [instances, selectedId]);

  if (err) {
    return <div className="p-4 text-sm text-[var(--color-fail)]">{err}</div>;
  }
  if (instances === null) {
    return (
      <div className="p-4 text-sm text-[var(--color-fg-dim)]">
        Loading Lambda instances…
      </div>
    );
  }
  if (instances.length === 0) {
    return (
      <div className="p-4 text-sm text-[var(--color-fg-dim)]">
        <div className="font-medium text-[var(--color-fg-muted)]">No Lambda instances yet</div>
        <div className="mt-1">
          The harness only provisions a GPU when a phase's classifier says one
          is warranted (training, fine-tuning, or large-model inference). Set{" "}
          <code className="rounded bg-[var(--color-surface-2)] px-1 py-0.5">LAMBDA_API_KEY</code>{" "}
          in the env that started the pipeline if you expected one to launch.
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <BudgetBar budget={budget} instances={instances} />
      <div className="flex min-h-0 flex-1">
        <div className="w-72 shrink-0 overflow-y-auto border-r border-[var(--color-border)]">
          <ul className="divide-y divide-[var(--color-border)]">
            {instances.map((inst) => (
              <li key={inst.instance_id}>
                <button
                  type="button"
                  onClick={() => setSelectedId(inst.instance_id)}
                  className={[
                    "flex w-full items-center gap-2 px-3 py-2 text-left text-xs transition-colors",
                    selectedId === inst.instance_id
                      ? "bg-[var(--color-surface-2)]"
                      : "hover:bg-[var(--color-surface-2)]/60",
                  ].join(" ")}
                >
                  <StatusDot status={inst.status} />
                  <span className="flex-1 truncate">
                    <span className="font-mono">{inst.instance_type}</span>
                    <span className="ml-1 text-[var(--color-fg-dim)]">
                      · {inst.phase_id}
                    </span>
                  </span>
                  <span className="font-mono text-[10px] text-[var(--color-fg-dim)]">
                    {formatCost(inst)}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
        <div className="min-w-0 flex-1 overflow-y-auto">
          {selectedId ? (
            <InstanceDetail
              instanceId={selectedId}
              listFingerprint={instances.length}
            />
          ) : (
            <div className="p-4 text-sm text-[var(--color-fg-dim)]">Select an instance.</div>
          )}
        </div>
      </div>
    </div>
  );
}

function BudgetBar({
  budget,
  instances,
}: {
  budget: ComputeBudget | null;
  instances: ComputeInstance[];
}) {
  const active = instances.filter((i) => i.status === "active").length;
  return (
    <div className="flex items-center gap-4 border-b border-[var(--color-border)] px-4 py-2 text-xs">
      <div>
        <span className="text-[var(--color-fg-dim)]">Active </span>
        <span className="font-mono">{active}</span>
        <span className="text-[var(--color-fg-dim)]"> · Total </span>
        <span className="font-mono">{instances.length}</span>
      </div>
      {budget && (
        <div className="ml-auto">
          <span className="text-[var(--color-fg-dim)]">GPU spend </span>
          <span className="font-mono">
            ${budget.projected_total_usd.toFixed(2)}
          </span>
          <span className="text-[var(--color-fg-dim)]"> / </span>
          <span className="font-mono">${budget.cap_usd.toFixed(2)} cap</span>
        </div>
      )}
    </div>
  );
}

function StatusDot({ status }: { status: ComputeInstance["status"] }) {
  const color =
    status === "active"
      ? "bg-[var(--color-success,#22c55e)]"
      : "bg-[var(--color-fg-dim)]";
  return (
    <span
      aria-label={status}
      className={`inline-block h-2 w-2 shrink-0 rounded-full ${color}`}
    />
  );
}

function formatCost(inst: ComputeInstance): string {
  const cost = inst.actual_cost_usd ?? inst.estimated_cost_usd;
  return `$${cost.toFixed(2)}`;
}

interface InstanceDetailProps {
  instanceId: string;
  // Bumps when the list refetches so detail also refreshes (without
  // having to wire events twice).
  listFingerprint: number;
}

function InstanceDetail({ instanceId, listFingerprint }: InstanceDetailProps) {
  const [detail, setDetail] = useState<ComputeDetailResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  // Bumped whenever a process_result event fires for the active phase so
  // the remote_runs list refreshes mid-training rather than waiting for a
  // compute_terminated event (which only arrives at the very end).
  const [outputNonce, setOutputNonce] = useState(0);

  useEffect(() => {
    setErr(null);
    api.compute
      .get(instanceId)
      .then(setDetail)
      .catch((e: Error) => setErr(e.message));
  }, [instanceId, listFingerprint, outputNonce]);

  // Subscribe to bash process events for this phase to keep remote runs live.
  useEffect(() => {
    if (!detail) return;
    const targetAgent = `phase:${detail.phase_id}`;
    const off = getEventStream().subscribe((e) => {
      if (e.agent_id !== targetAgent) return;
      if (e.type !== "process_started" && e.type !== "process_result") return;
      const cmd = String((e as { command?: string }).command ?? "");
      if (!cmd.includes("remote_run.sh") && e.type !== "process_result") return;
      setOutputNonce((n) => n + 1);
    });
    return off;
  }, [detail?.phase_id]);

  if (err) {
    return <div className="p-4 text-sm text-[var(--color-fail)]">{err}</div>;
  }
  if (!detail) {
    return <div className="p-4 text-sm text-[var(--color-fg-dim)]">Loading…</div>;
  }

  const lifetime = useLifetime(detail);

  return (
    <div className="space-y-4 p-4 text-xs">
      <div>
        <div className="flex items-center gap-2">
          <StatusDot status={detail.status} />
          <span className="font-mono text-sm">{detail.instance_type}</span>
          <span className="text-[var(--color-fg-dim)]">· {detail.region}</span>
        </div>
        <div className="mt-1 font-mono text-[10px] text-[var(--color-fg-dim)]">
          {detail.instance_id}
        </div>
      </div>

      <Section title="Phase">
        <KV k="phase_id" v={<span className="font-mono">{detail.phase_id}</span>} />
        <KV k="work_dir" v={<span className="font-mono break-all">{detail.work_dir}</span>} />
      </Section>

      <Section title="Cost">
        <KV k="status" v={detail.status} />
        <KV k="hourly rate" v={`$${detail.hourly_rate_usd.toFixed(2)} / hr`} />
        <KV
          k="estimated"
          v={`${detail.estimated_hours.toFixed(2)} h · $${detail.estimated_cost_usd.toFixed(2)}`}
        />
        {detail.actual_hours !== null && detail.actual_cost_usd !== null && (
          <KV
            k="actual"
            v={`${detail.actual_hours.toFixed(2)} h · $${detail.actual_cost_usd.toFixed(2)}`}
          />
        )}
        <KV k="lifetime" v={lifetime} />
      </Section>

      <Section title="Connection">
        <KV
          k="public ip"
          v={
            detail.public_ip ? (
              <span className="font-mono">{detail.public_ip}</span>
            ) : (
              <span className="italic text-[var(--color-fg-dim)]">(none)</span>
            )
          }
        />
        <KV k="ssh user" v={<span className="font-mono">{detail.ssh_user}</span>} />
        <KV k="ssh key" v={<span className="font-mono break-all">{detail.ssh_key_path}</span>} />
        {detail.ssh_command && (
          <div className="mt-2">
            <div className="mb-1 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
              SSH command
            </div>
            <CopyableCommand text={detail.ssh_command} />
          </div>
        )}
      </Section>

      <Section title="Lifecycle">
        <KV k="provisioned" v={detail.provisioned_at} />
        {detail.terminated_at && <KV k="terminated" v={detail.terminated_at} />}
        <KV k="ledger entry" v={<span className="font-mono">{detail.ledger_entry_id}</span>} />
      </Section>

      <RemoteRuns runs={detail.remote_runs} />

      {detail.upgrades.length > 0 && (
        <Section title="Upgrades">
          <ul className="space-y-1">
            {detail.upgrades.map((u, i) => (
              <li key={i} className="rounded border border-[var(--color-border)] p-2">
                <div className="font-mono text-[10px] text-[var(--color-fg-dim)]">
                  {u.ts}
                </div>
                <div>
                  <span className="font-mono">{u.from_instance_type ?? "?"}</span>
                  <span className="text-[var(--color-fg-dim)]"> → </span>
                  <span className="font-mono">{detail.instance_type}</span>
                </div>
                {u.reason && (
                  <div className="mt-1 italic text-[var(--color-fg-dim)]">{u.reason}</div>
                )}
              </li>
            ))}
          </ul>
        </Section>
      )}

      <div className="pt-2">
        <a
          href={detail.lambda_console_url}
          target="_blank"
          rel="noreferrer"
          className="text-[var(--color-accent)] hover:underline"
        >
          Open Lambda Cloud dashboard ↗
        </a>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
        {title}
      </div>
      <div className="space-y-1">{children}</div>
    </div>
  );
}

function KV({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="w-24 shrink-0 text-[10px] text-[var(--color-fg-dim)]">{k}</span>
      <span className="min-w-0 flex-1">{v}</span>
    </div>
  );
}

function CopyableCommand({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // ignore — clipboard perms may be denied
    }
  };
  return (
    <div className="flex items-start gap-2 rounded border border-[var(--color-border)] bg-[var(--color-surface-2)] px-2 py-1.5">
      <pre className="min-w-0 flex-1 overflow-x-auto whitespace-pre font-mono text-[11px]">
        {text}
      </pre>
      <button
        type="button"
        onClick={onCopy}
        className="shrink-0 rounded bg-[var(--color-surface)] px-2 py-0.5 text-[10px] hover:bg-[var(--color-surface-2)]"
      >
        {copied ? "copied" : "copy"}
      </button>
    </div>
  );
}

function RemoteRuns({ runs }: { runs: ComputeRemoteRun[] }) {
  if (!runs || runs.length === 0) {
    return (
      <Section title="Remote runs">
        <div className="italic text-[var(--color-fg-dim)]">
          No remote_run.sh invocations recorded yet.
        </div>
      </Section>
    );
  }
  // Newest first reads best for monitoring — the latest call is what the
  // operator wants to glance at.
  const ordered = [...runs].reverse();
  return (
    <Section title={`Remote runs (${runs.length})`}>
      <div className="space-y-2">
        {ordered.map((r) => (
          <RemoteRunRow key={r.process_id} run={r} />
        ))}
      </div>
    </Section>
  );
}

function RemoteRunRow({ run }: { run: ComputeRemoteRun }) {
  const live = run.finished_at === null;
  const [open, setOpen] = useState(live);
  const headerColor = run.is_error
    ? "text-[var(--color-fail)]"
    : live
      ? "text-[var(--color-accent)]"
      : "text-[var(--color-fg)]";
  return (
    <div className="rounded border border-[var(--color-border)] bg-[var(--color-surface)]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-2 py-1.5 text-left hover:bg-[var(--color-surface-2)]"
      >
        <span className={`shrink-0 font-mono text-[10px] ${headerColor}`}>
          {live ? "● running" : run.is_error ? "✗ failed" : "✓ done"}
        </span>
        <span className="min-w-0 flex-1 truncate font-mono text-[11px]">
          {stripRemoteRunWrapper(run.command)}
        </span>
        <span className="shrink-0 font-mono text-[10px] text-[var(--color-fg-dim)]">
          {formatRunTime(run)}
        </span>
        <span className="shrink-0 text-[10px] text-[var(--color-fg-dim)]">
          {open ? "▾" : "▸"}
        </span>
      </button>
      {open && (
        <div className="border-t border-[var(--color-border)] px-2 py-1.5">
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
            stdout (capped at 2 KB · ssh in for the full log)
          </div>
          <pre className="max-h-72 overflow-auto whitespace-pre-wrap font-mono text-[11px]">
            {run.output || (live ? "(waiting for output…)" : "(no output captured)")}
          </pre>
        </div>
      )}
    </div>
  );
}

function stripRemoteRunWrapper(cmd: string): string {
  // The full command looks like `bash remote_run.sh "python -m src.train ..."`.
  // Strip the wrapper so the inner command is what the user sees in the
  // collapsed header — that's the part they actually care about.
  const m = cmd.match(/remote_run\.sh\s+["'](.*?)["']\s*$/s);
  return m ? m[1] : cmd;
}

function formatRunTime(run: ComputeRemoteRun): string {
  if (!run.started_at) return "";
  const start = Date.parse(run.started_at);
  if (!start) return "";
  const end = run.finished_at ? Date.parse(run.finished_at) : Date.now();
  const secs = Math.max(0, Math.floor((end - start) / 1000));
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}

function useLifetime(detail: ComputeDetailResponse): string {
  const startedMs = useMemo(
    () => Date.parse(detail.provisioned_at) || 0,
    [detail.provisioned_at],
  );
  const endedMs = useMemo(
    () => (detail.terminated_at ? Date.parse(detail.terminated_at) : null),
    [detail.terminated_at],
  );
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (endedMs !== null) return;
    const t = setInterval(() => setTick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [endedMs]);
  // Touch `tick` so React re-renders even when the only dependency is wall time.
  void tick;
  const end = endedMs ?? Date.now();
  if (!startedMs) return "—";
  const s = Math.max(0, Math.floor((end - startedMs) / 1000));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h > 0
    ? `${h}h ${m}m ${sec}s`
    : m > 0
      ? `${m}m ${sec}s`
      : `${sec}s`;
}

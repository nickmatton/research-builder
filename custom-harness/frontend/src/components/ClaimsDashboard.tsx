import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { getEventStream } from "../lib/events";
import type { Claim } from "../lib/types";

interface Props {
  onJumpToPage?: (page: number) => void;
}

export function ClaimsDashboard({ onJumpToPage }: Props) {
  const [claims, setClaims] = useState<Claim[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const reload = () => {
    api
      .claims()
      .then((r) => setClaims(r.claims))
      .catch((e: Error) => setErr(e.message));
  };

  useEffect(() => {
    reload();
    const off = getEventStream().subscribe((e) => {
      if (e.type === "claims_extracted" || e.type === "spec_amended") {
        reload();
      }
    });
    return off;
  }, []);

  if (err) {
    return <div className="p-4 text-sm text-[var(--color-fail)]">{err}</div>;
  }
  if (claims === null) {
    return (
      <div className="p-4 text-sm text-[var(--color-fg-dim)]">Loading claims…</div>
    );
  }
  if (claims.length === 0) {
    return (
      <div className="p-4 text-sm text-[var(--color-fg-dim)]">
        No claims extracted yet. The claims ledger is populated in parallel with
        the section-spec authoring fan-out.
      </div>
    );
  }

  // Group by phase_id so the dashboard mirrors the execution DAG.
  const byPhase = new Map<string, Claim[]>();
  for (const c of claims) {
    const key = c.phase_id || "_unassigned";
    if (!byPhase.has(key)) byPhase.set(key, []);
    byPhase.get(key)!.push(c);
  }

  return (
    <div className="h-full overflow-auto">
      <div className="border-b border-[var(--color-border)] px-4 py-3">
        <div className="text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
          Claims ledger
        </div>
        <div className="mt-0.5 text-sm font-medium">
          {claims.length} numerical claims extracted
        </div>
      </div>

      <div className="space-y-4 px-4 py-3">
        {Array.from(byPhase.entries()).map(([phaseId, group]) => (
          <div key={phaseId}>
            <div className="mb-1 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
              <span className="font-mono">{phaseId}</span>{" "}
              <span className="text-[var(--color-fg-dim)]">· {group.length} claim(s)</span>
            </div>
            <div className="space-y-1">
              {group.map((c) => (
                <ClaimRow key={c.claim_id} claim={c} onJumpToPage={onJumpToPage} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ClaimRow({
  claim,
  onJumpToPage,
}: {
  claim: Claim;
  onJumpToPage?: (page: number) => void;
}) {
  const sourceLabel = formatSource(claim);
  const page = claim.source.page;
  return (
    <div className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2">
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-[11px] text-[var(--color-fg-muted)]">
          {claim.claim_id}
        </span>
        <span className="flex-1 text-xs text-[var(--color-fg)]">{claim.metric}</span>
        <span className="font-mono text-xs text-[var(--color-accent)]">
          {claim.value}
          {claim.tolerance ? ` ± ${claim.tolerance}` : ""} {claim.unit}
        </span>
      </div>
      <div className="mt-1 flex items-center gap-2 text-[10px] text-[var(--color-fg-dim)]">
        {claim.dataset && <span>{claim.dataset}</span>}
        {claim.condition && <span>· {claim.condition}</span>}
        {page !== null && page !== undefined && (
          <button
            type="button"
            onClick={() => onJumpToPage?.(page)}
            className="ml-auto rounded bg-[var(--color-surface-2)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--color-fg-muted)] hover:bg-[var(--color-surface)] hover:text-[var(--color-accent)]"
          >
            {sourceLabel}
          </button>
        )}
      </div>
      {claim.notes && (
        <div className="mt-1 text-[10px] italic text-[var(--color-fg-dim)]">
          {claim.notes}
        </div>
      )}
    </div>
  );
}

function formatSource(claim: Claim): string {
  const parts: string[] = [];
  if (claim.source.table) parts.push(claim.source.table);
  if (claim.source.figure) parts.push(claim.source.figure);
  if (claim.source.section) parts.push(`§${claim.source.section}`);
  if (claim.source.page) parts.push(`p.${claim.source.page}`);
  return parts.join(" · ") || "no source";
}

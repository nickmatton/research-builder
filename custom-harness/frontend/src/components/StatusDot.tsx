import type { PhaseStatus } from "../lib/types";

const COLOR: Record<PhaseStatus, string> = {
  pending: "bg-[var(--color-border-strong)]",
  in_progress: "bg-[var(--color-run)] shadow-[0_0_6px_var(--color-run)]",
  completed: "bg-[var(--color-ok)]",
  failed: "bg-[var(--color-fail)]",
};

export function StatusDot({ status }: { status: PhaseStatus | string }) {
  const cls = COLOR[status as PhaseStatus] ?? COLOR.pending;
  return <span className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full ${cls}`} title={status} />;
}

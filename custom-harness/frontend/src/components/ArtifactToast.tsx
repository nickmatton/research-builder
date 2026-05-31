import { useEffect, useState } from "react";
import { getEventStream } from "../lib/events";
import type { ArtifactCreatedEvent } from "../lib/types";

interface Toast {
  id: string;
  label: string;
  detail: string;
  ts: number;
}

const VISIBLE_FOR_MS = 5000;
const MAX_VISIBLE = 4;

const ARTIFACT_LABELS: Record<string, string> = {
  top_level_spec: "Top-level skeleton",
  section_spec: "Section spec",
  section_critique: "Section critique",
  claims_ledger: "Claims ledger",
  verification_report: "Verification report",
  reproduction_report: "Reproduction report",
};

function labelFor(ev: ArtifactCreatedEvent): { label: string; detail: string } {
  const label = ARTIFACT_LABELS[ev.artifact_type] || ev.artifact_type;
  const detail = ev.phase_id ? `${ev.phase_id}` : ev.path.split("/").slice(-2).join("/");
  return { label, detail };
}

export function ArtifactToast() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  useEffect(() => {
    const off = getEventStream().subscribe((e) => {
      if (e.type !== "artifact_created") return;
      const ev = e as ArtifactCreatedEvent;
      const { label, detail } = labelFor(ev);
      const id = `${ev.ts}-${ev.path}`;
      setToasts((prev) => {
        const next = prev.concat({
          id,
          label,
          detail,
          ts: Date.now(),
        });
        // Trim to MAX_VISIBLE.
        return next.slice(-MAX_VISIBLE);
      });
      // Auto-dismiss.
      window.setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
      }, VISIBLE_FOR_MS);
    });
    return off;
  }, []);

  if (toasts.length === 0) return null;

  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex flex-col gap-2">
      {toasts.map((t) => (
        <div
          key={t.id}
          className="pointer-events-auto min-w-[220px] max-w-[320px] rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 shadow-lg animate-in fade-in slide-in-from-right"
        >
          <div className="flex items-baseline gap-2">
            <span className="text-[10px] uppercase tracking-wider text-[var(--color-accent)]">
              new
            </span>
            <span className="text-xs font-medium text-[var(--color-fg)]">
              {t.label}
            </span>
          </div>
          <div className="mt-0.5 truncate text-[11px] text-[var(--color-fg-muted)]">
            {t.detail}
          </div>
        </div>
      ))}
    </div>
  );
}

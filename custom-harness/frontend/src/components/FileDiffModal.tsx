// Modal viewer for a file_write event's before/after diff. Triggered from
// the TraceView detail pane when the user clicks a write tool's "View diff"
// affordance. Renders a unified diff via the existing Diff component.

import { useEffect, useMemo } from "react";
import { computeDiff } from "../lib/diff";
import type { FileWriteEvent } from "../lib/types";
import { Diff } from "./Diff";

interface Props {
  ev: FileWriteEvent;
  onClose: () => void;
}

export function FileDiffModal({ ev, onClose }: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const lines = useMemo(() => computeDiff(ev.before, ev.after), [ev.before, ev.after]);

  const isCreate = ev.before === null;
  const isDelete = ev.after === null;
  const truncated = ev.before_truncated || ev.after_truncated;

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="flex max-h-[88vh] w-full max-w-5xl flex-col overflow-hidden rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] shadow-2xl"
      >
        <header className="flex shrink-0 items-start gap-3 border-b border-[var(--color-border)] px-5 py-3">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
              <span className="rounded bg-[var(--color-surface-2)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--color-fg-muted)]">
                {ev.tool_name}
              </span>
              {isCreate && (
                <span className="rounded bg-[var(--color-ok)]/15 px-1.5 py-0.5 text-[var(--color-ok)]">
                  created
                </span>
              )}
              {isDelete && (
                <span className="rounded bg-[var(--color-fail)]/15 px-1.5 py-0.5 text-[var(--color-fail)]">
                  deleted
                </span>
              )}
              {ev.is_error && (
                <span className="rounded bg-[var(--color-fail)]/15 px-1.5 py-0.5 text-[var(--color-fail)]">
                  errored
                </span>
              )}
              {truncated && (
                <span className="rounded bg-[var(--color-warn)]/15 px-1.5 py-0.5 text-[var(--color-warn)]">
                  truncated (256KB)
                </span>
              )}
            </div>
            <div className="mt-1 truncate font-mono text-xs text-[var(--color-fg)]">
              {ev.file_path}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1 text-[var(--color-fg-muted)] hover:bg-[var(--color-surface-2)] hover:text-[var(--color-fg)]"
            title="Close (Esc)"
          >
            ✕
          </button>
        </header>
        <div className="min-h-0 flex-1 overflow-auto p-4">
          <Diff lines={lines} />
        </div>
      </div>
    </div>
  );
}

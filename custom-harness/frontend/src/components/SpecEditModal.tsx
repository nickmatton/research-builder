// Two-step spec edit flow:
//   1. "edit"    — textarea with the current refined_spec.md.
//   2. "preview" — diff + cascade list, Apply / Back / Cancel.
//
// Apply writes both an `edit_refined_spec` command (queues the new
// content for the next attempt) AND a `jump_back` command (invalidates
// the phase + downstream phases so they actually re-run). The harness
// drains both at its existing intervention hooks.

import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { CascadePreview, Phase } from "../lib/types";
import { Diff } from "./Diff";

interface Props {
  phase: Phase;
  onClose: () => void;
  /** Called after a successful Apply (parent can refetch / toast). */
  onApplied?: () => void;
}

type Mode = "edit" | "preview";

const BEFORE_AGENTS = ["refiner", "researcher", "builder", "verifier"] as const;

export function SpecEditModal({ phase, onClose, onApplied }: Props) {
  const [mode, setMode] = useState<Mode>("edit");
  const [content, setContent] = useState("");
  const [original, setOriginal] = useState("");
  const [beforeAgent, setBeforeAgent] = useState("builder");
  const [rationale, setRationale] = useState("");
  const [preview, setPreview] = useState<CascadePreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [working, setWorking] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    api
      .refinedSpec(phase.phase_id)
      .then((r) => {
        setOriginal(r.content);
        setContent(r.content);
      })
      .catch((e: Error) => setErr(e.message))
      .finally(() => setLoading(false));
  }, [phase.phase_id]);

  // ESC closes the modal at any step. Click-outside also closes.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const onPreview = async () => {
    setWorking(true);
    setErr(null);
    try {
      const r = await api.spec_edits.preview(phase.phase_id, content, beforeAgent);
      setPreview(r);
      setMode("preview");
    } catch (e) {
      setErr(e instanceof Error ? e.message : "preview failed");
    } finally {
      setWorking(false);
    }
  };

  const onApply = async () => {
    setWorking(true);
    setErr(null);
    try {
      await api.spec_edits.apply(phase.phase_id, content, beforeAgent, rationale);
      onApplied?.();
      onClose();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "apply failed");
    } finally {
      setWorking(false);
    }
  };

  const dirty = content !== original;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="flex max-h-[88vh] w-full max-w-3xl flex-col overflow-hidden rounded-lg border border-[var(--color-border-strong)] bg-[var(--color-surface)] shadow-2xl shadow-black/60"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex shrink-0 items-center justify-between border-b border-[var(--color-border)] px-4 py-3">
          <div>
            <div className="text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
              {mode === "edit" ? "Edit refined_spec" : "Preview cascade"}
            </div>
            <div className="font-mono text-sm">{phase.phase_id}</div>
            <div className="text-xs text-[var(--color-fg-muted)]">{phase.title}</div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1 text-[var(--color-fg-dim)] hover:bg-[var(--color-surface-2)] hover:text-[var(--color-fg)]"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        {/* Body */}
        <div className="min-h-0 flex-1 overflow-auto px-4 py-3">
          {loading ? (
            <div className="text-xs text-[var(--color-fg-dim)]">Loading…</div>
          ) : mode === "edit" ? (
            <EditBody
              content={content}
              setContent={setContent}
              beforeAgent={beforeAgent}
              setBeforeAgent={setBeforeAgent}
            />
          ) : preview ? (
            <PreviewBody preview={preview} rationale={rationale} setRationale={setRationale} />
          ) : null}
          {err && (
            <div className="mt-3 rounded border border-[var(--color-fail)]/40 bg-[var(--color-fail)]/10 px-3 py-2 text-xs text-[var(--color-fail)]">
              {err}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex shrink-0 items-center justify-between border-t border-[var(--color-border)] bg-[var(--color-surface-2)] px-4 py-2.5">
          <div className="text-[11px] text-[var(--color-fg-dim)]">
            {mode === "edit"
              ? dirty
                ? "Unsaved changes"
                : "No changes"
              : preview?.invalidated.length
                ? `Will invalidate ${preview.invalidated.length} phase${preview.invalidated.length === 1 ? "" : "s"}`
                : "No phases invalidated"}
          </div>
          <div className="flex gap-2">
            {mode === "edit" ? (
              <>
                <Btn onClick={onClose}>Cancel</Btn>
                <Btn variant="primary" onClick={onPreview} disabled={!dirty || working}>
                  {working ? "…" : "Preview cascade →"}
                </Btn>
              </>
            ) : (
              <>
                <Btn onClick={() => setMode("edit")} disabled={working}>
                  ← Back to editor
                </Btn>
                <Btn variant="primary" onClick={onApply} disabled={working}>
                  {working ? "Applying…" : "Apply"}
                </Btn>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function EditBody({
  content,
  setContent,
  beforeAgent,
  setBeforeAgent,
}: {
  content: string;
  setContent: (s: string) => void;
  beforeAgent: string;
  setBeforeAgent: (s: string) => void;
}) {
  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex items-center gap-2 text-xs">
        <label htmlFor="before-agent" className="text-[var(--color-fg-muted)]">
          Apply before:
        </label>
        <select
          id="before-agent"
          value={beforeAgent}
          onChange={(e) => setBeforeAgent(e.target.value)}
          className="rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-2 py-0.5 text-xs"
        >
          {BEFORE_AGENTS.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
        <span className="text-[10px] text-[var(--color-fg-dim)]">
          Roles from {beforeAgent} onward re-run on this phase.
        </span>
      </div>
      <textarea
        value={content}
        onChange={(e) => setContent(e.target.value)}
        spellCheck={false}
        className="min-h-[300px] flex-1 resize-none rounded border border-[var(--color-border)] bg-[var(--color-bg)] p-3 font-mono text-xs leading-relaxed text-[var(--color-fg)] outline-none focus:border-[var(--color-accent)]"
        placeholder="(no refined_spec content yet — write what you want this phase's agents to see)"
      />
    </div>
  );
}

function PreviewBody({
  preview,
  rationale,
  setRationale,
}: {
  preview: CascadePreview;
  rationale: string;
  setRationale: (s: string) => void;
}) {
  return (
    <div className="space-y-4">
      {preview.error && (
        <div className="rounded border border-[var(--color-fail)]/40 bg-[var(--color-fail)]/10 px-3 py-2 text-xs text-[var(--color-fail)]">
          {preview.error}
        </div>
      )}

      <section>
        <div className="mb-1 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
          Diff
        </div>
        <Diff lines={preview.diff} />
      </section>

      <section>
        <div className="mb-1.5 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
          Will invalidate (cascade)
        </div>
        {preview.invalidated.length === 0 ? (
          <div className="rounded border border-[var(--color-border)] bg-[var(--color-surface-2)] px-3 py-2 text-xs text-[var(--color-fg-dim)]">
            Nothing — this phase will pick up the edit on its next attempt.
          </div>
        ) : (
          <ul className="space-y-1.5">
            {preview.invalidated.map((p) => (
              <li
                key={p.phase_id}
                className="flex items-center gap-2 rounded border border-[var(--color-border)] bg-[var(--color-surface-2)] px-3 py-1.5"
              >
                <span
                  className={`text-[10px] font-medium uppercase tracking-wider ${
                    p.reason === "direct"
                      ? "text-[var(--color-accent)]"
                      : "text-[var(--color-warn)]"
                  }`}
                >
                  {p.reason}
                </span>
                <span className="font-mono text-xs">{p.phase_id}</span>
                <span className="truncate text-xs text-[var(--color-fg-muted)]">
                  {p.title}
                </span>
                <span className="ml-auto text-[10px] text-[var(--color-fg-dim)]">
                  {p.roles.join(" → ")}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <label htmlFor="rationale" className="mb-1 block text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
          Rationale (logged to revision history)
        </label>
        <input
          id="rationale"
          type="text"
          value={rationale}
          onChange={(e) => setRationale(e.target.value)}
          placeholder="Why this edit?"
          className="w-full rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-2 py-1 text-xs outline-none focus:border-[var(--color-accent)]"
        />
      </section>
    </div>
  );
}

function Btn({
  children,
  onClick,
  variant = "neutral",
  disabled,
}: {
  children: React.ReactNode;
  onClick: () => void;
  variant?: "neutral" | "primary";
  disabled?: boolean;
}) {
  const base = "rounded px-3 py-1.5 text-xs font-medium transition-opacity disabled:cursor-not-allowed disabled:opacity-30";
  const cls =
    variant === "primary"
      ? `${base} bg-[var(--color-accent)] text-white`
      : `${base} bg-[var(--color-surface)] text-[var(--color-fg-muted)] hover:bg-[var(--color-bg)] hover:text-[var(--color-fg)]`;
  return (
    <button type="button" onClick={onClick} disabled={disabled} className={cls}>
      {children}
    </button>
  );
}

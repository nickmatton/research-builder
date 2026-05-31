// Landing screen shown when no workspace is active. Drop or pick a PDF,
// hit Launch, and the backend creates a workspace under runs-dir and
// spawns the research-builder pipeline. The parent re-fetches workspace
// info after the POST resolves and transitions to the main UI.
//
// When the workspace name already exists, the backend returns 409 with a
// structured detail (``code: "workspace_exists"``) — we render a modal
// letting the user pick wipe / archive / resume, then re-submit.

import { useRef, useState } from "react";
import { api, ApiError } from "../lib/api";
import type { WorkspaceInfo } from "../lib/types";

interface Props {
  runsDir: string;
  onLaunched: () => void;
}

type Conflict = {
  workspace: string;
  name: string;
  has_state: boolean;
  has_paper: boolean;
  message?: string;
};

type ConflictChoice = "wipe" | "archive" | "resume";

export function Launcher({ runsDir, onLaunched }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [working, setWorking] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [conflict, setConflict] = useState<Conflict | null>(null);
  // Auto mode default OFF: most users want the orchestrator to pause at
  // gates so they can review spec/section work before it ships.
  const [skipGates, setSkipGates] = useState(false);
  // Subscription mode default ON: route through the Claude Code CLI so the
  // run consumes the user's subscription rather than API credits. The
  // backend's --dev flag still flips this on, but the default is on either
  // way.
  const [devMode, setDevMode] = useState(true);
  const inputRef = useRef<HTMLInputElement>(null);

  const acceptFile = (f: File | null) => {
    setErr(null);
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".pdf")) {
      setErr("Only PDF files are supported.");
      return;
    }
    setFile(f);
    // Derive a default workspace name from the filename stem.
    const stem = f.name.replace(/\.[^.]+$/, "").trim();
    if (stem && !name) setName(stem);
  };

  const doLaunch = async (onConflict?: ConflictChoice) => {
    if (!file || working) return;
    setWorking(true);
    setErr(null);
    try {
      await api.launch(file, name.trim() || undefined, onConflict, skipGates, devMode);
      onLaunched();
    } catch (e) {
      if (
        e instanceof ApiError &&
        e.status === 409 &&
        e.body &&
        typeof e.body === "object" &&
        (e.body as { code?: unknown }).code === "workspace_exists"
      ) {
        setConflict(e.body as Conflict);
        setWorking(false);
        return;
      }
      setErr(e instanceof Error ? e.message : String(e));
      setWorking(false);
    }
  };

  const launch = () => doLaunch();
  const resolveConflict = (choice: ConflictChoice) => {
    setConflict(null);
    doLaunch(choice);
  };

  return (
    <div className="flex h-full items-center justify-center p-8">
      <div className="w-full max-w-xl">
        <div className="mb-6">
          <h1 className="text-lg font-semibold tracking-tight">
            Reproduce a paper
          </h1>
          <p className="mt-1 text-sm text-[var(--color-fg-muted)]">
            Drop a PDF below to scaffold a workspace and launch the pipeline.
            New runs land under{" "}
            <code className="rounded bg-[var(--color-surface-2)] px-1.5 py-0.5 text-[12px]">
              {runsDir}
            </code>
            .
          </p>
        </div>

        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            acceptFile(e.dataTransfer.files?.[0] ?? null);
          }}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-md border border-dashed px-6 py-12 text-center transition-colors ${
            dragOver
              ? "border-[var(--color-accent)] bg-[var(--color-accent)]/5"
              : "border-[var(--color-border)] bg-[var(--color-surface)] hover:border-[var(--color-border-strong)]"
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf,.pdf"
            className="sr-only"
            onChange={(e) => acceptFile(e.target.files?.[0] ?? null)}
          />
          {file ? (
            <div className="space-y-1">
              <div className="text-sm font-medium">{file.name}</div>
              <div className="text-[11px] text-[var(--color-fg-dim)]">
                {(file.size / 1024).toFixed(0)} KB · click to choose another
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="text-sm text-[var(--color-fg-muted)]">
                Drop a PDF here, or click to browse
              </div>
              <div className="text-[11px] text-[var(--color-fg-dim)]">
                Only PDF files are accepted
              </div>
            </div>
          )}
        </label>

        <div className="mt-4 space-y-2">
          <label className="block text-[11px] uppercase tracking-wide text-[var(--color-fg-dim)]">
            Workspace name (optional)
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="auto-derived from filename"
            className="w-full rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-1.5 text-sm focus:border-[var(--color-accent)] focus:outline-none"
          />
        </div>

        <label className="mt-4 flex cursor-pointer items-start gap-2.5 rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2">
          <input
            type="checkbox"
            checked={skipGates}
            onChange={(e) => setSkipGates(e.target.checked)}
            className="mt-0.5 accent-[var(--color-accent)]"
          />
          <div className="text-xs">
            <div className="font-medium text-[var(--color-fg)]">Run autonomously</div>
            <div className="mt-0.5 text-[11px] text-[var(--color-fg-dim)]">
              Skip per-phase approval prompts. The orchestrator runs end-to-end;
              you can still steer it via chat at any time. Uncheck to gate
              progress on chat approvals.
            </div>
          </div>
        </label>

        <label className="mt-2 flex cursor-pointer items-start gap-2.5 rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2">
          <input
            type="checkbox"
            checked={devMode}
            onChange={(e) => setDevMode(e.target.checked)}
            className="mt-0.5 accent-[var(--color-accent)]"
          />
          <div className="text-xs">
            <div className="font-medium text-[var(--color-fg)]">Subscription mode</div>
            <div className="mt-0.5 text-[11px] text-[var(--color-fg-dim)]">
              Spend Claude Code subscription tokens via the bundled{" "}
              <code className="rounded bg-[var(--color-surface-2)] px-1 py-px">claude</code>{" "}
              CLI instead of Anthropic API credits. Requires you to be signed
              in to Claude Code locally. Uncheck to use{" "}
              <code className="rounded bg-[var(--color-surface-2)] px-1 py-px">ANTHROPIC_API_KEY</code>.
            </div>
          </div>
        </label>

        {err && (
          <div className="mt-3 rounded border border-[var(--color-fail)]/40 bg-[var(--color-fail)]/5 px-3 py-2 text-xs text-[var(--color-fail)]">
            {err}
          </div>
        )}

        <div className="mt-5 flex items-center justify-end gap-2">
          <button
            type="button"
            disabled={!file || working}
            onClick={launch}
            className="rounded-md bg-[var(--color-accent)] px-4 py-1.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
          >
            {working ? "Launching…" : "Launch reproduction"}
          </button>
        </div>

        <p className="mt-6 text-[11px] text-[var(--color-fg-dim)]">
          The pipeline runs as a subprocess. It writes to
          <code className="mx-1 rounded bg-[var(--color-surface-2)] px-1 py-0.5">
            logs/pipeline.log
          </code>
          inside the workspace; you can tail that file from the Files tab.
        </p>
      </div>

      {conflict && (
        <ConflictDialog
          conflict={conflict}
          onChoose={resolveConflict}
          onCancel={() => setConflict(null)}
        />
      )}
    </div>
  );
}

function ConflictDialog({
  conflict,
  onChoose,
  onCancel,
}: {
  conflict: Conflict;
  onChoose: (choice: ConflictChoice) => void;
  onCancel: () => void;
}) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={onCancel}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-md rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-5 shadow-2xl"
      >
        <div className="mb-2 text-[10px] uppercase tracking-wider text-[var(--color-warn)]">
          Workspace exists
        </div>
        <div className="text-sm font-medium text-[var(--color-fg)]">
          {conflict.name}
        </div>
        <div className="mt-1 truncate font-mono text-[10px] text-[var(--color-fg-dim)]">
          {conflict.workspace}
        </div>

        <p className="mt-3 text-xs text-[var(--color-fg-muted)]">
          A workspace with this name already exists. Pick how to proceed:
        </p>

        <div className="mt-4 space-y-2">
          {conflict.has_state && (
            <ConflictOption
              label="Resume"
              hint="Pick up where the previous run left off (keeps spec, phases, completed work)."
              onClick={() => onChoose("resume")}
              accent
            />
          )}
          <ConflictOption
            label="Archive & start fresh"
            hint={
              "Move existing canonical_spec, phases, logs, etc. to "
              + ".archive/<timestamp>/, then start a new run. Keeps the paper PDF."
            }
            onClick={() => onChoose("archive")}
          />
          <ConflictOption
            label="Wipe & start fresh"
            hint="Permanently delete prior harness-managed dirs (canonical_spec, phases, logs, …). Keeps the paper PDF."
            onClick={() => onChoose("wipe")}
            destructive
          />
        </div>

        <div className="mt-4 flex justify-end">
          <button
            type="button"
            onClick={onCancel}
            className="rounded px-3 py-1 text-xs text-[var(--color-fg-muted)] hover:text-[var(--color-fg)]"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

function ConflictOption({
  label,
  hint,
  onClick,
  accent,
  destructive,
}: {
  label: string;
  hint: string;
  onClick: () => void;
  accent?: boolean;
  destructive?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`w-full rounded-md border px-3 py-2 text-left transition-colors ${
        accent
          ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10 hover:bg-[var(--color-accent)]/20"
          : destructive
            ? "border-[var(--color-fail)]/40 bg-[var(--color-fail)]/5 hover:bg-[var(--color-fail)]/10"
            : "border-[var(--color-border)] bg-[var(--color-surface-2)] hover:border-[var(--color-border-strong)]"
      }`}
    >
      <div
        className={`text-xs font-medium ${
          destructive ? "text-[var(--color-fail)]" : "text-[var(--color-fg)]"
        }`}
      >
        {label}
      </div>
      <div className="mt-0.5 text-[11px] leading-snug text-[var(--color-fg-muted)]">
        {hint}
      </div>
    </button>
  );
}

// Helper used by App.tsx to decide whether to render the Launcher or the
// main split view.
export function isLauncherState(ws: WorkspaceInfo | null): boolean {
  return ws == null || ws.state === "empty";
}

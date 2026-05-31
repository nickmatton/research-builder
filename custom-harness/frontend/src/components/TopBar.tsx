import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { getEventStream } from "../lib/events";
import type { WorkspaceInfo } from "../lib/types";

interface Props {
  workspace: WorkspaceInfo | null;
  onCommandPalette?: () => void;
  onPipelineStopped?: () => void;
}

export function TopBar({ workspace, onCommandPalette, onPipelineStopped }: Props) {
  const [live, setLive] = useState(false);
  const [lastEventAt, setLastEventAt] = useState<number | null>(null);
  const [stopping, setStopping] = useState(false);

  const isRunning = workspace?.pipeline.state === "running";

  const handleStop = async () => {
    if (stopping) return;
    const ok = window.confirm(
      "Stop the running pipeline? In-flight phase work will be lost; you can resume from the last checkpoint.",
    );
    if (!ok) return;
    setStopping(true);
    try {
      await api.pipelineStop();
      onPipelineStopped?.();
    } catch (e) {
      window.alert(e instanceof Error ? e.message : "stop failed");
    } finally {
      setStopping(false);
    }
  };

  useEffect(() => {
    const stream = getEventStream();
    const off = stream.subscribe(() => {
      setLive(true);
      setLastEventAt(Date.now());
    });
    // If no event arrives within 5s the run is probably idle or finished;
    // demote the dot from green ("live") to dim.
    const t = setInterval(() => {
      if (lastEventAt && Date.now() - lastEventAt > 5000) setLive(false);
    }, 1000);
    return () => {
      off();
      clearInterval(t);
    };
  }, [lastEventAt]);

  return (
    <header className="flex h-11 shrink-0 items-center justify-between border-b border-[var(--color-border)] bg-[var(--color-surface)] px-4">
      <div className="flex items-center gap-3">
        <span className="text-sm font-medium tracking-tight">research-builder</span>
        {workspace?.name && (
          <>
            <span className="text-[var(--color-fg-dim)]">/</span>
            <span className="text-sm text-[var(--color-fg-muted)]">{workspace.name}</span>
          </>
        )}
      </div>
      <div className="flex items-center gap-3 text-xs text-[var(--color-fg-muted)]">
        {isRunning && (
          <button
            type="button"
            onClick={handleStop}
            disabled={stopping}
            className="rounded border border-[var(--color-fail)] bg-[var(--color-surface-2)] px-2 py-0.5 text-[10px] text-[var(--color-fail)] hover:bg-[var(--color-fail)] hover:text-[var(--color-bg)] disabled:opacity-50"
            title="Stop the running pipeline"
          >
            {stopping ? "Stopping…" : "Stop run"}
          </button>
        )}
        {onCommandPalette && (
          <button
            type="button"
            onClick={onCommandPalette}
            className="flex items-center gap-1.5 rounded border border-[var(--color-border)] bg-[var(--color-surface-2)] px-2 py-0.5 text-[10px] hover:border-[var(--color-border-strong)] hover:text-[var(--color-fg)]"
            title="Command palette"
          >
            <span>Commands</span>
            <kbd className="rounded bg-[var(--color-bg)] px-1 font-mono text-[10px] text-[var(--color-fg-dim)]">
              ⌘K
            </kbd>
          </button>
        )}
        <div className="flex items-center gap-2">
          <span
            className={`h-1.5 w-1.5 rounded-full transition-colors ${
              live ? "bg-[var(--color-ok)] shadow-[0_0_8px_var(--color-ok)]" : "bg-[var(--color-border-strong)]"
            }`}
            title={live ? "Receiving events" : "Idle"}
          />
          <span>{live ? "live" : "idle"}</span>
        </div>
      </div>
    </header>
  );
}

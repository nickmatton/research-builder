// Permanent right-side chat panel — orchestrator-only.
//
// All user-facing chat goes to the orchestrator: it answers questions about
// the paper/spec, drives the approval gates, and routes operator commands.
// The previous "Paper Agent" sub-chat was removed once the orchestrator's
// chat surface proved sufficient on its own.

import { OrchestratorChat } from "./OrchestratorChat";

interface Props {
  gateActive?: boolean;
}

export function ChatPanel({ gateActive = false }: Props) {
  return (
    <div className="flex h-full flex-col bg-[var(--color-surface)]">
      <div className="flex h-9 shrink-0 items-center gap-2 border-b border-[var(--color-border)] px-3 text-[12px]">
        <span className="text-[var(--color-fg)]">Orchestrator</span>
        <span className="text-[var(--color-fg-dim)]">
          {gateActive ? "· approval required" : "· steer the run"}
        </span>
        {gateActive && (
          <span className="ml-1 inline-block h-1.5 w-1.5 rounded-full bg-[var(--color-accent)] shadow-[0_0_6px_var(--color-accent)]" />
        )}
      </div>
      <div className="min-h-0 flex-1">
        <OrchestratorChat />
      </div>
    </div>
  );
}

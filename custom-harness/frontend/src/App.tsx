import { useEffect, useMemo, useState } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { api } from "./lib/api";
import { cmdK, useShortcuts } from "./lib/shortcuts";
import type { Phase, WorkspaceInfo } from "./lib/types";
import { TopBar } from "./components/TopBar";
import { PdfViewer } from "./components/PdfViewer";
import { SpecView } from "./components/SpecView";
import { DocsView, SKELETON_DOC } from "./components/DocsView";
import { FilesView } from "./components/FilesView";
import { TraceView } from "./components/TraceView";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { CommandPalette, type Command } from "./components/CommandPalette";
import { Launcher, isLauncherState } from "./components/Launcher";
import { ClaimsDashboard } from "./components/ClaimsDashboard";
import { ComputeView } from "./components/ComputeView";
import { ReportView } from "./components/ReportView";
import { ArtifactToast } from "./components/ArtifactToast";
import { ViewSwitcher, VIEW_IDS, type ViewId } from "./components/ViewSwitcher";
import { ChatPanel } from "./components/ChatPanel";
import { getEventStream } from "./lib/events";

// Map an approval gate to the spec doc it's reviewing, or null if the gate
// isn't about a spec. The authoritative source is the orchestrator's
// optional `open_doc` field on `request_user_approval`; we fall back to a
// gate_id heuristic so historical event streams (and runs where the model
// forgot to set open_doc) still auto-route.
function gateTargetDoc(e: {
  open_doc?: string | null;
  gate_id?: string;
}): string | null {
  const d = (e.open_doc ?? "").trim();
  if (d) {
    if (d === "spec.md" || d === "canonical_spec/spec.md") return SKELETON_DOC;
    const m = d.match(/(?:^|\/)sections\/([^/]+)\.md$/);
    if (m) return m[1];
    return null; // unknown shape — leave the view alone rather than guess
  }
  // Legacy fallback: derive from gate_id.
  const gid = e.gate_id ?? "";
  if (gid === "post_skeleton") return SKELETON_DOC;
  for (const prefix of ["pre_phase:", "pre_builder:"]) {
    if (gid.startsWith(prefix)) return gid.slice(prefix.length);
  }
  return null;
}

export function App() {
  const [workspace, setWorkspace] = useState<WorkspaceInfo | null>(null);
  const [bootErr, setBootErr] = useState<string | null>(null);

  // Left-pane view; Activity is the default so live execution is visible
  // the second the run kicks off — no tab-hunting required.
  const [view, setView] = useState<ViewId>("activity");
  const [paletteOpen, setPaletteOpen] = useState(false);

  // Gate active state — drives a "approval required" indicator on the chat
  // panel so the user notices a pending gate even while they're inspecting
  // the spec or files.
  const [gateActive, setGateActive] = useState(false);

  // Spec citation chips and Claims/Verification "go to page" actions push
  // to this state; PdfViewer reads it and scrolls accordingly.
  const [jumpToPage, setJumpToPage] = useState<number | null>(null);

  // Phase list, refreshed for the palette so commands can reference live phases.
  const [phases, setPhases] = useState<Phase[]>([]);

  // Doc the Docs view should focus. Bumping the nonce re-applies the same
  // focus (so re-reaching a gate re-opens its spec even if already shown).
  const [docFocus, setDocFocus] = useState<string | null>(null);
  const [docFocusNonce, setDocFocusNonce] = useState(0);

  const refreshWorkspace = () =>
    api.workspace().then(setWorkspace).catch((e: Error) => setBootErr(e.message));

  useEffect(() => {
    refreshWorkspace();
  }, []);

  // Poll workspace info while we're in launcher/transition states so the UI
  // flips into the main view as soon as the backend reports a workspace.
  useEffect(() => {
    if (!workspace || isLauncherState(workspace)) {
      const t = setInterval(refreshWorkspace, 1500);
      return () => clearInterval(t);
    }
    return undefined;
  }, [workspace?.state]);

  useEffect(() => {
    if (!workspace || isLauncherState(workspace)) return;
    const load = () => api.phases().then((r) => setPhases(r.phases)).catch(() => undefined);
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, [workspace?.state]);

  // Surface chat-driven approval gates: a pending gate raises a dot on the
  // chat panel header so the user notices regardless of which left-pane
  // view they're inspecting. When the gate is about approving a spec, also
  // jump to the Docs view and open the spec under review so it's right there.
  useEffect(() => {
    const off = getEventStream().subscribe((e) => {
      if (e.type === "gate_reached") {
        const auto = (e as { auto?: boolean }).auto === true;
        if (!auto) {
          setGateActive(true);
          const target = gateTargetDoc(e as { open_doc?: string | null; gate_id?: string });
          if (target) {
            setView("docs");
            setDocFocus(target);
            setDocFocusNonce((n) => n + 1);
          }
        }
      } else if (e.type === "gate_resolved") {
        setGateActive(false);
      }
    });
    return off;
  }, []);

  // ─── Commands available in the palette ───
  const commands: Command[] = useMemo(() => {
    const list: Command[] = [];
    for (const v of VIEW_IDS) {
      const label =
        v.charAt(0).toUpperCase() + v.slice(1) + (v === "activity" ? " (live agent tree)" : "");
      list.push({ id: `view:${v}`, group: "view", label, run: () => setView(v) });
    }
    for (const p of phases) {
      list.push({
        id: `retry:${p.phase_id}`,
        group: "retry",
        label: `Retry ${p.phase_id} — ${p.title}`,
        run: async () => {
          const reason = window.prompt(`Retry rationale for ${p.phase_id}:`, "operator-triggered retry");
          if (reason == null) return;
          try {
            await api.commands.forceRetry(p.phase_id, reason);
          } catch (e) {
            window.alert(e instanceof Error ? e.message : "force_retry failed");
          }
        },
      });
    }
    return list;
  }, [phases]);

  useShortcuts([
    { match: cmdK, run: (e) => { e.preventDefault(); setPaletteOpen((v) => !v); }, allowEditable: true },
  ]);

  const paperAvailable = Boolean(workspace?.paper_path);

  return (
    <div className="flex h-screen flex-col bg-[var(--color-bg)] text-[var(--color-fg)]">
      <TopBar
        workspace={workspace}
        onCommandPalette={() => setPaletteOpen(true)}
        onPipelineStopped={refreshWorkspace}
      />
      {bootErr ? (
        <main className="flex flex-1 flex-col items-center justify-center gap-2 p-4 text-center">
          <div className="text-sm font-medium text-[var(--color-fail)]">
            Failed to reach backend
          </div>
          <pre className="max-w-md text-[10px] text-[var(--color-fg-dim)]">{bootErr}</pre>
        </main>
      ) : isLauncherState(workspace) ? (
        <main className="min-h-0 flex-1">
          <Launcher
            runsDir={workspace?.runs_dir ?? "./runs"}
            onLaunched={refreshWorkspace}
          />
        </main>
      ) : (
        <>
          <ViewSwitcher
            active={view}
            onChange={setView}
            paperAvailable={paperAvailable}
          />
          <main className="min-h-0 flex-1">
            <PanelGroup direction="horizontal" autoSaveId="rb:main-split">
              <Panel defaultSize={62} minSize={30}>
                <ActiveView
                  view={view}
                  paperAvailable={paperAvailable}
                  jumpToPage={jumpToPage}
                  onJumpToPage={setJumpToPage}
                  docFocus={docFocus}
                  docFocusNonce={docFocusNonce}
                />
              </Panel>
              <PanelResizeHandle className="group relative w-px bg-[var(--color-border)] transition-colors data-[resize-handle-state=hover]:bg-[var(--color-accent)] data-[resize-handle-state=drag]:bg-[var(--color-accent)]">
                <div className="absolute inset-y-0 -left-1.5 w-3 cursor-col-resize" />
              </PanelResizeHandle>
              <Panel defaultSize={38} minSize={25}>
                <ErrorBoundary label="Chat">
                  <ChatPanel gateActive={gateActive} />
                </ErrorBoundary>
              </Panel>
            </PanelGroup>
          </main>
        </>
      )}

      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        commands={commands}
      />
      {/* Top-level toast layer for artifact_created events. */}
      <ArtifactToast />
    </div>
  );
}

// ─── Active-view dispatcher ────────────────────────────────────────────

interface ActiveViewProps {
  view: ViewId;
  paperAvailable: boolean;
  jumpToPage: number | null;
  onJumpToPage: (page: number) => void;
  docFocus: string | null;
  docFocusNonce: number;
}

function ActiveView({
  view,
  paperAvailable,
  jumpToPage,
  onJumpToPage,
  docFocus,
  docFocusNonce,
}: ActiveViewProps) {
  switch (view) {
    case "activity":
      return (
        <ErrorBoundary label="Activity">
          <TraceView />
        </ErrorBoundary>
      );
    case "paper":
      return paperAvailable ? (
        <ErrorBoundary label="Paper">
          <PdfViewer jumpToPage={jumpToPage} />
        </ErrorBoundary>
      ) : (
        <EmptyState>No PDF in this workspace.</EmptyState>
      );
    case "spec":
      return (
        <ErrorBoundary label="Spec">
          <SpecView onJumpToPage={onJumpToPage} />
        </ErrorBoundary>
      );
    case "docs":
      return (
        <ErrorBoundary label="Docs">
          <DocsView focusDoc={docFocus} focusNonce={docFocusNonce} />
        </ErrorBoundary>
      );
    case "files":
      return (
        <ErrorBoundary label="Files">
          <FilesView />
        </ErrorBoundary>
      );
    case "claims":
      return (
        <ErrorBoundary label="Claims">
          <ClaimsDashboard onJumpToPage={onJumpToPage} />
        </ErrorBoundary>
      );
    case "compute":
      return (
        <ErrorBoundary label="Compute">
          <ComputeView />
        </ErrorBoundary>
      );
    case "report":
      return (
        <ErrorBoundary label="Report">
          <ReportView />
        </ErrorBoundary>
      );
  }
}

function EmptyState({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full items-center justify-center text-sm text-[var(--color-fg-dim)]">
      {children}
    </div>
  );
}

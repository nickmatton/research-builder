import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import { getEventStream } from "../lib/events";
import { highlight } from "../lib/highlight";
import { Markdown } from "./Markdown";
import type { FileEntry, FilesResponse, HarnessEvent, WorkspaceInfo } from "../lib/types";

function isMarkdownPath(path: string): boolean {
  const lower = path.toLowerCase();
  return lower.endsWith(".md") || lower.endsWith(".markdown");
}

interface NodeState {
  open: boolean;
  children?: FileEntry[];
  loading: boolean;
  error?: string;
}

// A file the harness has touched during this session. "wrote" first time we
// see a Write to a path the tree didn't already know about = "created."
type ActivityKind = "wrote" | "edited";

interface FileActivity {
  lastTs: number;
  kind: ActivityKind;
  agentId: string;
}

// Strip the workspace prefix off whatever absolute path the agent passed to
// Write/Edit. Returns null if it can't be made workspace-relative.
function toWorkspaceRelative(filePath: string, workspaceAbs: string | null): string | null {
  if (!filePath) return null;
  if (workspaceAbs) {
    const prefix = workspaceAbs.endsWith("/") ? workspaceAbs : workspaceAbs + "/";
    if (filePath === workspaceAbs) return "";
    if (filePath.startsWith(prefix)) return filePath.slice(prefix.length);
  }
  // Already relative? Trust it.
  if (!filePath.startsWith("/")) return filePath;
  return null;
}

function parentDir(path: string): string {
  const i = path.lastIndexOf("/");
  return i <= 0 ? "" : path.slice(0, i);
}

export function FilesView() {
  const [workspace, setWorkspace] = useState<WorkspaceInfo | null>(null);
  const [root, setRoot] = useState<FilesResponse | null>(null);
  const [nodes, setNodes] = useState<Record<string, NodeState>>({});
  const [selected, setSelected] = useState<string | null>(null);
  const [content, setContent] = useState<string | null>(null);
  const [contentErr, setContentErr] = useState<string | null>(null);
  const [contentLoading, setContentLoading] = useState(false);
  // Markdown files default to the rendered view; the toggle lets you drop
  // back to the raw, syntax-highlighted source.
  const [mdRendered, setMdRendered] = useState(true);

  // File-activity tracking. Subscribes to process_started Write/Edit events
  // and maintains a per-path last-touch timestamp, so the tree can show
  // "just edited" and "newly created" markers in real time.
  const [activity, setActivity] = useState<Map<string, FileActivity>>(() => new Map());
  // Mirror activity in a ref so the dir-refresh side-effect can compare
  // against the previous state without re-subscribing on every event.
  const activityRef = useRef<Map<string, FileActivity>>(activity);
  useEffect(() => { activityRef.current = activity; }, [activity]);
  // "now" tick: re-render every 5s so the freshness threshold rolls
  // forward without each individual event burning a render.
  const [, setNowTick] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setNowTick((n) => n + 1), 5000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    api.workspace().then(setWorkspace).catch(() => setWorkspace(null));
  }, []);

  useEffect(() => {
    api.files("").then(setRoot).catch(() => setRoot({ path: "", entries: [] }));
  }, []);

  // Refresh a directory's children from the API. Used after a Write event
  // lands in a dir we already had cached, so new files show up without the
  // operator having to collapse + re-open the dir.
  const refreshDir = useCallback(async (dirPath: string) => {
    try {
      const res = await api.files(dirPath);
      if (dirPath === "") {
        setRoot(res);
        return;
      }
      setNodes((prev) => {
        const cur = prev[dirPath];
        if (!cur) return prev; // not currently expanded — don't bother
        return { ...prev, [dirPath]: { ...cur, children: res.entries } };
      });
    } catch {
      // best-effort
    }
  }, []);

  useEffect(() => {
    if (!workspace) return;
    const wsPath = workspace.path;
    const stream = getEventStream();

    const ingest = (e: HarnessEvent, store: Map<string, FileActivity>): string | null => {
      if (e.type !== "process_started") return null;
      const tool = (e as { tool_name?: string }).tool_name;
      if (tool !== "Write" && tool !== "Edit") return null;
      const raw = (e as { file_path?: string | null }).file_path;
      if (!raw) return null;
      const rel = toWorkspaceRelative(raw, wsPath);
      if (rel == null) return null;
      const ts = Date.parse(e.ts) || Date.now();
      const prev = store.get(rel);
      // Treat first sighting as "wrote" iff the tool actually was Write;
      // an Edit on a path we'd never seen still becomes "edited" (rare,
      // means the file existed pre-session).
      let kind: ActivityKind;
      if (prev) {
        kind = tool === "Write" && prev.kind === "wrote" ? "wrote" : "edited";
      } else {
        kind = tool === "Write" ? "wrote" : "edited";
      }
      store.set(rel, { lastTs: ts, kind, agentId: e.agent_id });
      return rel;
    };

    // Hydrate.
    const init = new Map<string, FileActivity>();
    for (const e of stream.snapshot()) ingest(e, init);
    setActivity(init);

    const off = stream.subscribe((e) => {
      if (e.type !== "process_started") return;
      const next = new Map(activityRef.current);
      const rel = ingest(e, next);
      if (rel == null) return;
      activityRef.current = next;
      setActivity(next);
      // If the write lands in a directory we currently render, refresh
      // it so any newly-created sibling files appear.
      refreshDir(parentDir(rel));
    });
    return off;
  }, [workspace, refreshDir]);

  // Roll activity up to ancestor dirs so a write deep in the tree marks
  // every parent dir as "has fresh content."
  const dirActivity = useMemo(() => {
    const out = new Map<string, number>();
    for (const [path, act] of activity) {
      let cur = parentDir(path);
      while (true) {
        const prev = out.get(cur) ?? 0;
        if (act.lastTs > prev) out.set(cur, act.lastTs);
        if (cur === "") break;
        cur = parentDir(cur);
      }
    }
    return out;
  }, [activity]);

  const toggleDir = async (path: string) => {
    setNodes((prev) => {
      const cur = prev[path] ?? { open: false, loading: false };
      if (cur.children) {
        return { ...prev, [path]: { ...cur, open: !cur.open } };
      }
      return { ...prev, [path]: { ...cur, open: true, loading: true } };
    });
    if (!nodes[path]?.children) {
      try {
        const res = await api.files(path);
        setNodes((prev) => ({
          ...prev,
          [path]: { open: true, children: res.entries, loading: false },
        }));
      } catch (e) {
        setNodes((prev) => ({
          ...prev,
          [path]: { open: false, loading: false, error: String(e) },
        }));
      }
    }
  };

  const openFile = async (path: string) => {
    setSelected(path);
    setContent(null);
    setContentErr(null);
    setContentLoading(true);
    try {
      const text = await api.fileText(path);
      setContent(text);
    } catch (e) {
      setContentErr(e instanceof Error ? e.message : "load failed");
    } finally {
      setContentLoading(false);
    }
  };

  return (
    <div className="grid h-full grid-cols-[minmax(180px,30%)_1fr]">
      <div className="overflow-auto border-r border-[var(--color-border)] py-2">
        {!root ? (
          <div className="px-3 text-xs text-[var(--color-fg-dim)]">Loading…</div>
        ) : (
          <TreeLevel
            entries={root.entries}
            depth={0}
            nodes={nodes}
            selected={selected}
            onToggle={toggleDir}
            onOpen={openFile}
            activity={activity}
            dirActivity={dirActivity}
          />
        )}
      </div>
      <div className="flex flex-col overflow-hidden bg-[var(--color-bg)]">
        <FileViewerHeader
          path={selected}
          isMarkdown={selected != null && isMarkdownPath(selected)}
          rendered={mdRendered}
          onToggleRendered={() => setMdRendered((v) => !v)}
        />
        <div className="min-h-0 flex-1 overflow-auto">
          {contentLoading ? (
            <div className="p-4 text-xs text-[var(--color-fg-dim)]">Loading…</div>
          ) : contentErr ? (
            <div className="p-4 text-xs text-[var(--color-fail)]">{contentErr}</div>
          ) : content == null || selected == null ? (
            <div className="p-4 text-xs text-[var(--color-fg-dim)]">
              Select a file to view its contents.
            </div>
          ) : isMarkdownPath(selected) && mdRendered ? (
            <div className="prose-spec mx-auto max-w-3xl px-6 py-5 text-[13px] text-[var(--color-fg)]">
              <Markdown>{content}</Markdown>
            </div>
          ) : (
            <CodeView path={selected} text={content} />
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Header + code renderer ─────────────────────────────────────────────

function FileViewerHeader({
  path,
  isMarkdown,
  rendered,
  onToggleRendered,
}: {
  path: string | null;
  isMarkdown: boolean;
  rendered: boolean;
  onToggleRendered: () => void;
}) {
  return (
    <div className="flex h-7 shrink-0 items-center justify-between gap-2 border-b border-[var(--color-border)] px-3 text-[11px] text-[var(--color-fg-muted)]">
      <span className="truncate">
        {path ?? <span className="text-[var(--color-fg-dim)]">no file selected</span>}
      </span>
      {isMarkdown && (
        <div className="flex shrink-0 overflow-hidden rounded border border-[var(--color-border)] text-[10px]">
          {(["rendered", "raw"] as const).map((mode) => {
            const active = (mode === "rendered") === rendered;
            return (
              <button
                key={mode}
                type="button"
                onClick={onToggleRendered}
                className={`px-2 py-0.5 ${
                  active
                    ? "bg-[var(--color-surface-2)] text-[var(--color-fg)]"
                    : "text-[var(--color-fg-dim)] hover:text-[var(--color-fg-muted)]"
                }`}
              >
                {mode}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function CodeView({ path, text }: { path: string; text: string }) {
  // Highlight once per (path, text). Memo keeps fast re-renders on tree
  // navigation cheap.
  const { html, lang } = useMemo(() => highlight(text, path), [path, text]);
  const lineCount = useMemo(() => text.split("\n").length, [text]);

  return (
    <div className="relative">
      <div className="sticky top-0 z-10 flex items-center justify-end gap-2 border-b border-[var(--color-border)]/60 bg-[var(--color-bg)] px-3 py-1 text-[10px] text-[var(--color-fg-dim)]">
        <span className="tabular-nums">{lineCount} lines</span>
        <span className="font-mono uppercase">{lang}</span>
      </div>
      <pre className="overflow-x-auto p-4 font-mono text-[11.5px] leading-relaxed">
        <code className="hljs" dangerouslySetInnerHTML={{ __html: html }} />
      </pre>
    </div>
  );
}

// Window during which a file/dir gets the bright pulsing "just touched"
// treatment. After this, the marker fades to a soft static dot.
const FRESH_MS = 8000;
// How long a touched file/dir stays visibly marked at all. After this it
// becomes indistinguishable from never-touched entries.
const STALE_MS = 30 * 60 * 1000;

interface TreeLevelProps {
  entries: FileEntry[];
  depth: number;
  nodes: Record<string, NodeState>;
  selected: string | null;
  onToggle: (path: string) => void;
  onOpen: (path: string) => void;
  activity: Map<string, FileActivity>;
  dirActivity: Map<string, number>;
}

function TreeLevel({
  entries,
  depth,
  nodes,
  selected,
  onToggle,
  onOpen,
  activity,
  dirActivity,
}: TreeLevelProps) {
  const now = Date.now();
  return (
    <ul>
      {entries.map((e) => {
        const isSelected = selected === e.path;
        const nodeState = nodes[e.path];
        const act = activity.get(e.path);
        const dirLastTs = e.is_dir ? dirActivity.get(e.path) ?? 0 : 0;
        const lastTs = e.is_dir ? dirLastTs : act?.lastTs ?? 0;
        const age = lastTs ? now - lastTs : Infinity;
        const fresh = lastTs > 0 && age < FRESH_MS;
        const recent = lastTs > 0 && age < STALE_MS;
        const isNew = !e.is_dir && act?.kind === "wrote";
        const nameCls = fresh
          ? "text-[var(--color-fg)]"
          : recent
            ? "text-[var(--color-fg)]/90"
            : isSelected
              ? "text-[var(--color-fg)]"
              : "text-[var(--color-fg-muted)]";
        return (
          <li key={e.path}>
            <button
              type="button"
              className={`flex w-full items-center gap-1.5 px-2 py-0.5 text-left text-xs hover:bg-[var(--color-surface-2)] ${
                isSelected ? "bg-[var(--color-surface-2)]" : ""
              }`}
              style={{ paddingLeft: 8 + depth * 12 }}
              onClick={() => (e.is_dir ? onToggle(e.path) : onOpen(e.path))}
              title={
                act
                  ? `${act.kind === "wrote" ? "Created" : "Edited"} by ${act.agentId} · ${formatAge(age)} ago`
                  : undefined
              }
            >
              <span className="inline-block w-3 text-center text-[10px] text-[var(--color-fg-dim)]">
                {e.is_dir ? (nodeState?.open ? "▾" : "▸") : ""}
              </span>
              <ActivityDot fresh={fresh} recent={recent} />
              <span className={`truncate ${nameCls}`}>{e.name}</span>
              {isNew && recent ? (
                <span
                  className={`ml-auto rounded px-1 text-[9px] uppercase tracking-wider ${
                    fresh
                      ? "bg-[var(--color-ok)]/20 text-[var(--color-ok)]"
                      : "bg-[var(--color-ok)]/10 text-[var(--color-ok)]/80"
                  }`}
                >
                  new
                </span>
              ) : null}
            </button>
            {e.is_dir && nodeState?.open && nodeState.children && (
              <TreeLevel
                entries={nodeState.children}
                depth={depth + 1}
                nodes={nodes}
                selected={selected}
                onToggle={onToggle}
                onOpen={onOpen}
                activity={activity}
                dirActivity={dirActivity}
              />
            )}
          </li>
        );
      })}
    </ul>
  );
}

function ActivityDot({ fresh, recent }: { fresh: boolean; recent: boolean }) {
  if (!recent) {
    // Reserve the same space so non-recent rows align with recent ones.
    return <span className="inline-block h-1.5 w-1.5 shrink-0" />;
  }
  const cls = fresh
    ? "animate-pulse bg-[var(--color-accent)]"
    : "bg-[var(--color-accent)]/40";
  return <span className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full ${cls}`} />;
}

function formatAge(ms: number): string {
  if (ms < 1000) return "just now";
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.round(m / 60);
  return `${h}h`;
}

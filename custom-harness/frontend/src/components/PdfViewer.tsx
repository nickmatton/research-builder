import { useEffect, useMemo, useRef, useState } from "react";
import * as pdfjsLib from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.mjs?url";
import { shortAgentId } from "../lib/agents";
import { api } from "../lib/api";
import { getEventStream } from "../lib/events";
import type { HarnessEvent, PaperReadEvent } from "../lib/types";

// Configure the worker once at module load. Vite emits the .mjs as a
// real asset URL (the ?url import), so this works in dev + prod without
// any postinstall script.
pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

interface Props {
  /** External page-jump trigger (spec citation chips + claims source links). */
  jumpToPage?: number | null;
}

interface PaperRead {
  agent_id: string;
  page_start: number | null;
  page_end: number | null;
  ts: number;
}

interface ContextState {
  reads: PaperRead[];
  live: Set<string>;
}

const EMPTY_CONTEXT: ContextState = { reads: [], live: new Set() };
const MAX_READS = 200;

function applyEvent(state: ContextState, e: HarnessEvent): ContextState {
  if (e.type === "agent_started") {
    if (state.live.has(e.agent_id)) return state;
    const live = new Set(state.live);
    live.add(e.agent_id);
    return { reads: state.reads, live };
  }
  if (e.type === "agent_completed") {
    if (!state.live.has(e.agent_id)) return state;
    const live = new Set(state.live);
    live.delete(e.agent_id);
    return { reads: state.reads, live };
  }
  if (e.type === "paper_read") {
    const pe = e as PaperReadEvent;
    const read: PaperRead = {
      agent_id: pe.agent_id,
      page_start: pe.page_start ?? null,
      page_end: pe.page_end ?? null,
      ts: Date.parse(pe.ts) || Date.now(),
    };
    const reads = state.reads.concat(read);
    if (reads.length > MAX_READS) reads.splice(0, reads.length - MAX_READS);
    return { reads, live: state.live };
  }
  return state;
}

export function PdfViewer({ jumpToPage }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const textLayerRef = useRef<HTMLDivElement>(null);
  const [doc, setDoc] = useState<pdfjsLib.PDFDocumentProxy | null>(null);
  const [page, setPage] = useState(1);
  const [scale, setScale] = useState(1.2);
  const [error, setError] = useState<string | null>(null);

  // Tracks paper_read events and which agents are mid-flight (agent_started
  // but not yet agent_completed). Pages an agent has read while still live
  // are highlighted as "currently in context." After the agent completes,
  // those pages fade to a softer "previously read" tint.
  const [ctx, setCtx] = useState<ContextState>(EMPTY_CONTEXT);

  useEffect(() => {
    const stream = getEventStream();
    // Hydrate from the ring so toggling tabs / hot-reloading doesn't wipe
    // highlight state.
    let acc: ContextState = EMPTY_CONTEXT;
    for (const e of stream.snapshot()) acc = applyEvent(acc, e);
    setCtx(acc);
    const off = stream.subscribe((e) => {
      if (
        e.type !== "paper_read" &&
        e.type !== "agent_started" &&
        e.type !== "agent_completed"
      ) {
        return;
      }
      setCtx((prev) => applyEvent(prev, e));
    });
    return off;
  }, []);

  // Load the document once.
  useEffect(() => {
    let cancelled = false;
    setError(null);
    pdfjsLib
      .getDocument(api.pdfUrl())
      .promise.then((d) => {
        if (cancelled) return;
        setDoc(d);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "failed to load PDF");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Render the current page whenever doc / page / scale changes.
  useEffect(() => {
    if (!doc || !canvasRef.current) return;
    let cancelled = false;
    (async () => {
      const p = await doc.getPage(page);
      if (cancelled || !canvasRef.current) return;

      const viewport = p.getViewport({ scale: scale * (window.devicePixelRatio || 1) });
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d")!;
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      canvas.style.width = `${viewport.width / (window.devicePixelRatio || 1)}px`;
      canvas.style.height = `${viewport.height / (window.devicePixelRatio || 1)}px`;

      await p.render({ canvasContext: ctx, viewport }).promise;

      // Render the text layer so users can select / copy quotes from the PDF.
      if (textLayerRef.current) {
        const layer = textLayerRef.current;
        layer.innerHTML = "";
        layer.style.width = `${viewport.width / (window.devicePixelRatio || 1)}px`;
        layer.style.height = `${viewport.height / (window.devicePixelRatio || 1)}px`;
        const textContent = await p.getTextContent();
        const textViewport = p.getViewport({ scale });
        const textLayer = new pdfjsLib.TextLayer({
          textContentSource: textContent,
          container: layer,
          viewport: textViewport,
        });
        await textLayer.render();
      }
    })().catch((e: unknown) => {
      if (cancelled) return;
      setError(e instanceof Error ? e.message : "render failed");
    });
    return () => {
      cancelled = true;
    };
  }, [doc, page, scale]);

  // External page-jump (Phase 2 citation clicks).
  useEffect(() => {
    if (jumpToPage && doc && jumpToPage >= 1 && jumpToPage <= doc.numPages) {
      setPage(jumpToPage);
    }
  }, [jumpToPage, doc]);

  const numPages = doc?.numPages ?? 0;

  // Build a per-page summary: which pages are currently in some live
  // agent's context window vs. just historically read. A read with a null
  // range means "whole document," so it covers every page.
  const pageState = useMemo(() => {
    const active = new Set<number>();
    const seen = new Set<number>();
    for (const r of ctx.reads) {
      const live = ctx.live.has(r.agent_id);
      const start = r.page_start ?? 1;
      const end = r.page_end ?? (numPages || start);
      for (let p = start; p <= end; p++) {
        seen.add(p);
        if (live) active.add(p);
      }
    }
    return { active, seen };
  }, [ctx, numPages]);

  // Find the most recent live read for the status line.
  const activeRead = useMemo(() => {
    for (let i = ctx.reads.length - 1; i >= 0; i--) {
      const r = ctx.reads[i];
      if (ctx.live.has(r.agent_id)) return r;
    }
    return null;
  }, [ctx]);

  return (
    <div className="flex h-full flex-col bg-[var(--color-surface)]">
      <div className="flex h-9 shrink-0 items-center justify-between border-b border-[var(--color-border)] px-3 text-xs text-[var(--color-fg-muted)]">
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            className="rounded px-2 py-0.5 hover:bg-[var(--color-surface-2)] disabled:opacity-30"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            aria-label="Previous page"
          >
            ←
          </button>
          <span className="tabular-nums">
            {page} / {numPages || "—"}
          </span>
          <button
            type="button"
            className="rounded px-2 py-0.5 hover:bg-[var(--color-surface-2)] disabled:opacity-30"
            onClick={() => setPage((p) => Math.min(numPages, p + 1))}
            disabled={page >= numPages}
            aria-label="Next page"
          >
            →
          </button>
        </div>
        {activeRead ? (
          <ActiveContextBadge read={activeRead} totalPages={numPages} />
        ) : null}
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            className="rounded px-2 py-0.5 hover:bg-[var(--color-surface-2)]"
            onClick={() => setScale((s) => Math.max(0.5, s - 0.1))}
            aria-label="Zoom out"
          >
            −
          </button>
          <span className="tabular-nums">{Math.round(scale * 100)}%</span>
          <button
            type="button"
            className="rounded px-2 py-0.5 hover:bg-[var(--color-surface-2)]"
            onClick={() => setScale((s) => Math.min(3, s + 0.1))}
            aria-label="Zoom in"
          >
            +
          </button>
        </div>
      </div>
      {numPages > 0 ? (
        <ContextStrip
          numPages={numPages}
          currentPage={page}
          active={pageState.active}
          seen={pageState.seen}
          onJump={setPage}
        />
      ) : null}
      <div className="flex-1 overflow-auto p-4">
        {error ? (
          <div className="flex h-full items-center justify-center text-sm text-[var(--color-fail)]">
            {error}
          </div>
        ) : !doc ? (
          <div className="flex h-full items-center justify-center text-sm text-[var(--color-fg-dim)]">
            Loading PDF…
          </div>
        ) : (
          <div className="relative mx-auto w-fit shadow-lg shadow-black/40">
            <canvas ref={canvasRef} className="block" />
            <div ref={textLayerRef} className="textLayer" />
          </div>
        )}
      </div>
    </div>
  );
}

function ActiveContextBadge({
  read,
  totalPages,
}: {
  read: PaperRead;
  totalPages: number;
}) {
  const start = read.page_start ?? 1;
  const end = read.page_end ?? (totalPages || start);
  const range = start === end ? `p.${start}` : `pp.${start}–${end}`;
  const whole = read.page_start == null;
  return (
    <div
      className="flex items-center gap-1.5 rounded-full border border-[var(--color-accent)]/30 bg-[var(--color-accent)]/10 px-2 py-0.5 text-[10px] text-[var(--color-accent)]"
      title={`${read.agent_id} is currently using ${whole ? "the whole paper" : `pages ${start}-${end}`} as context`}
    >
      <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-[var(--color-accent)]" />
      <span className="font-medium">{shortAgentId(read.agent_id)}</span>
      <span className="opacity-70">·</span>
      <span className="tabular-nums">{whole ? "whole paper" : range}</span>
    </div>
  );
}

function ContextStrip({
  numPages,
  currentPage,
  active,
  seen,
  onJump,
}: {
  numPages: number;
  currentPage: number;
  active: Set<number>;
  seen: Set<number>;
  onJump: (page: number) => void;
}) {
  const cells = [];
  for (let p = 1; p <= numPages; p++) {
    const isActive = active.has(p);
    const wasRead = seen.has(p);
    const isCurrent = p === currentPage;
    let cls = "bg-[var(--color-surface-2)]";
    if (isActive) cls = "bg-[var(--color-accent)]";
    else if (wasRead) cls = "bg-[var(--color-accent)]/25";
    cells.push(
      <button
        key={p}
        type="button"
        onClick={() => onJump(p)}
        className={`group relative h-full min-w-0 flex-1 ${cls} transition-colors hover:brightness-125 ${
          isCurrent ? "ring-1 ring-inset ring-[var(--color-fg)]" : ""
        }`}
        title={
          isActive
            ? `Page ${p} — currently in agent context`
            : wasRead
              ? `Page ${p} — previously read`
              : `Page ${p}`
        }
        aria-label={`Jump to page ${p}`}
      />,
    );
  }
  return (
    <div
      className="flex h-2.5 shrink-0 gap-px border-b border-[var(--color-border)] bg-[var(--color-bg)] px-1"
      role="presentation"
    >
      {cells}
    </div>
  );
}

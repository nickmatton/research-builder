// Markdown doc-browser. A reading-focused companion to the structured Spec
// dashboard: the left rail lists the skeleton spec.md plus every per-section
// spec .md, and the main pane renders the selected one full-width.
//
// `focusDoc`/`focusNonce` let the app steer this view — e.g. when the
// orchestrator pauses at a spec-approval gate, App switches here and points
// us at the spec under review. Bumping `focusNonce` re-applies the same
// `focusDoc` (so re-reaching a gate re-opens it even if already selected).
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { getEventStream } from "../lib/events";
import type { SectionSummary } from "../lib/types";
import { Markdown } from "./Markdown";

// Sentinel doc key for the top-level skeleton spec.md.
export const SKELETON_DOC = "__skeleton__";

interface Props {
  focusDoc: string | null;
  focusNonce: number;
}

export function DocsView({ focusDoc, focusNonce }: Props) {
  const [title, setTitle] = useState<string>("");
  const [skeletonMd, setSkeletonMd] = useState<string | null>(null);
  const [sections, setSections] = useState<SectionSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  // Lazily-loaded per-section markdown, keyed by phase_id.
  const [sectionMd, setSectionMd] = useState<Record<string, string>>({});
  const [err, setErr] = useState<string | null>(null);

  const reload = () => {
    api
      .spec()
      .then((s) => {
        setSkeletonMd(s.spec_md);
        setTitle(s.state?.metadata?.paper_title ?? "");
      })
      .catch((e: Error) => setErr(e.message));
    api.sections().then((r) => setSections(r.sections)).catch(() => undefined);
  };

  useEffect(() => {
    reload();
    const off = getEventStream().subscribe((e) => {
      if (
        e.type === "skeleton_completed" ||
        e.type === "section_spec_started" ||
        e.type === "section_spec_completed" ||
        e.type === "section_spec_critiqued" ||
        e.type === "spec_amended"
      ) {
        reload();
      }
    });
    return off;
  }, []);

  // Default to the skeleton once it lands and nothing else is selected.
  useEffect(() => {
    if (selected == null && skeletonMd != null) setSelected(SKELETON_DOC);
  }, [skeletonMd, selected]);

  // App-driven focus (e.g. spec-approval gate). The nonce forces re-selection
  // even when focusDoc is unchanged.
  useEffect(() => {
    if (focusDoc) setSelected(focusDoc);
  }, [focusDoc, focusNonce]);

  // Lazily fetch the markdown for whichever section is selected.
  useEffect(() => {
    if (!selected || selected === SKELETON_DOC || sectionMd[selected]) return;
    api
      .section(selected)
      .then((d) => setSectionMd((prev) => ({ ...prev, [selected]: d.spec_markdown })))
      .catch((e: Error) => setErr(e.message));
  }, [selected, sectionMd]);

  const body =
    selected === SKELETON_DOC
      ? skeletonMd
      : selected
        ? sectionMd[selected] ?? null
        : null;

  return (
    <div className="grid h-full grid-cols-[minmax(200px,30%)_1fr]">
      {/* Doc list */}
      <div className="overflow-auto border-r border-[var(--color-border)] py-2">
        <div className="px-3 pb-1 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
          Spec docs
        </div>
        <DocItem
          label="spec.md"
          sublabel="skeleton"
          active={selected === SKELETON_DOC}
          onClick={() => setSelected(SKELETON_DOC)}
        />
        {sections.length > 0 && (
          <div className="px-3 pb-1 pt-3 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
            Sections
          </div>
        )}
        {sections.map((s) => (
          <DocItem
            key={s.phase_id}
            label={s.phase_id}
            sublabel={s.title}
            active={selected === s.phase_id}
            onClick={() => setSelected(s.phase_id)}
          />
        ))}
      </div>

      {/* Rendered doc */}
      <div className="flex flex-col overflow-hidden bg-[var(--color-bg)]">
        <div className="flex h-7 shrink-0 items-center border-b border-[var(--color-border)] px-3 text-[11px] text-[var(--color-fg-muted)]">
          <span className="truncate">
            {selected === SKELETON_DOC
              ? title || "spec.md"
              : selected ?? <span className="text-[var(--color-fg-dim)]">no doc selected</span>}
          </span>
        </div>
        <div className="min-h-0 flex-1 overflow-auto">
          {err ? (
            <div className="p-4 text-xs text-[var(--color-fail)]">{err}</div>
          ) : selected == null ? (
            <div className="p-4 text-xs text-[var(--color-fg-dim)]">
              Select a doc to read it.
            </div>
          ) : body == null ? (
            <div className="p-4 text-xs text-[var(--color-fg-dim)]">Loading…</div>
          ) : (
            <div className="prose-spec mx-auto max-w-3xl px-6 py-5 text-[13px] text-[var(--color-fg)]">
              <Markdown>{body}</Markdown>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function DocItem({
  label,
  sublabel,
  active,
  onClick,
}: {
  label: string;
  sublabel?: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex w-full flex-col items-start px-3 py-1.5 text-left hover:bg-[var(--color-surface-2)] ${
        active ? "bg-[var(--color-surface-2)]" : ""
      }`}
    >
      <span
        className={`truncate font-mono text-xs ${
          active ? "text-[var(--color-fg)]" : "text-[var(--color-fg-muted)]"
        }`}
      >
        {label}
      </span>
      {sublabel && (
        <span className="truncate text-[10px] text-[var(--color-fg-dim)]">{sublabel}</span>
      )}
    </button>
  );
}

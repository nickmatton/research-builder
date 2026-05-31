import { useEffect, useState } from "react";
import { Markdown } from "./Markdown";
import { api } from "../lib/api";
import { getEventStream } from "../lib/events";
import type {
  Phase,
  SectionSummary,
  SpecResponse,
} from "../lib/types";
import { StatusDot } from "./StatusDot";
import { SpecEditModal } from "./SpecEditModal";
import { SectionSpecCard } from "./SectionSpecCard";

interface Props {
  onJumpToPage?: (page: number) => void;
}

export function SpecView({ onJumpToPage }: Props) {
  const [spec, setSpec] = useState<SpecResponse | null>(null);
  const [phases, setPhases] = useState<Phase[]>([]);
  const [sections, setSections] = useState<SectionSummary[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [editing, setEditing] = useState<Phase | null>(null);

  const reload = () => {
    api.spec().then(setSpec).catch((e: Error) => setErr(e.message));
    api.phases().then((r) => setPhases(r.phases)).catch((e: Error) => setErr(e.message));
    api.sections().then((r) => setSections(r.sections)).catch(() => undefined);
  };

  useEffect(() => {
    reload();
    // Refetch on relevant events. The new authoring flow emits skeleton/
    // section_spec/critique events; reload covers all of them with one call.
    const off = getEventStream().subscribe((e) => {
      if (
        e.type === "phase_started" ||
        e.type === "phase_completed" ||
        e.type === "phase_failed" ||
        e.type === "retry_launched" ||
        e.type === "spec_amended" ||
        e.type === "skeleton_completed" ||
        e.type === "section_spec_started" ||
        e.type === "section_spec_completed" ||
        e.type === "section_spec_critiqued"
      ) {
        reload();
      }
    });
    return off;
  }, []);

  if (err) {
    return <div className="p-4 text-sm text-[var(--color-fail)]">{err}</div>;
  }
  if (!spec) {
    return <div className="p-4 text-sm text-[var(--color-fg-dim)]">Loading spec…</div>;
  }

  const title = spec.state?.metadata?.paper_title ?? "(untitled paper)";

  // Sections may arrive in a different order than phases — index by id so
  // the card grid mirrors phase order (which mirrors DAG order).
  const sectionsByPhase = new Map(sections.map((s) => [s.phase_id, s]));

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="border-b border-[var(--color-border)] px-4 py-3">
        <div className="text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
          Paper
        </div>
        <div className="mt-0.5 text-sm font-medium">{title}</div>
      </div>

      <div className="border-b border-[var(--color-border)] px-4 py-3">
        <div className="mb-2 flex items-baseline justify-between">
          <div className="text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
            Phases
          </div>
          <div className="text-[10px] tabular-nums text-[var(--color-fg-dim)]">
            {sections.length}/{phases.length} section specs authored
          </div>
        </div>
        {phases.length === 0 ? (
          <div className="text-xs text-[var(--color-fg-dim)]">no phases yet</div>
        ) : (
          <ul className="space-y-1">
            {phases.map((p) => (
              <li
                key={p.phase_id}
                className="group flex items-center gap-2 rounded px-1.5 py-1 hover:bg-[var(--color-surface-2)]"
              >
                <StatusDot status={p.status} />
                <span className="text-xs font-mono text-[var(--color-fg-muted)]">
                  {p.phase_id}
                </span>
                <span className="truncate text-xs text-[var(--color-fg)]">
                  {p.title}
                </span>
                <button
                  type="button"
                  onClick={() => setEditing(p)}
                  className="invisible ml-auto rounded px-1.5 py-0.5 text-[10px] text-[var(--color-fg-dim)] hover:bg-[var(--color-surface)] hover:text-[var(--color-fg-muted)] group-hover:visible"
                  title="Edit this phase's refined_spec"
                >
                  edit
                </button>
                {p.attempts.length > 0 && (
                  <span className="text-[10px] tabular-nums text-[var(--color-fg-dim)]">
                    {p.attempts.length}×
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        {/* Section spec cards — populating live as each per-section author
            completes. Each card is collapsible; expanded view shows
            acceptance criteria with clickable citations that drive the
            PdfViewer. */}
        {phases.length > 0 && (
          <div className="border-b border-[var(--color-border)] px-4 py-3">
            <div className="mb-2 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
              Section specs
            </div>
            {sections.length === 0 ? (
              <div className="text-xs text-[var(--color-fg-dim)]">
                no section specs yet — they appear here as the upfront
                authoring fan-out completes
              </div>
            ) : (
              <div className="space-y-2">
                {phases.map((p) => {
                  const summary = sectionsByPhase.get(p.phase_id);
                  if (!summary) {
                    return (
                      <div
                        key={p.phase_id}
                        className="rounded border border-dashed border-[var(--color-border)] px-3 py-2 text-[11px] text-[var(--color-fg-dim)]"
                      >
                        <span className="font-mono">{p.phase_id}</span> — authoring…
                      </div>
                    );
                  }
                  return (
                    <SectionSpecCard
                      key={p.phase_id}
                      summary={summary}
                      onJumpToPage={onJumpToPage}
                    />
                  );
                })}
              </div>
            )}
          </div>
        )}

        <div className="px-4 py-3">
          <div className="mb-2 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
            Top-level spec.md (skeleton)
          </div>
          {spec.spec_md ? (
            <div className="prose-spec text-xs text-[var(--color-fg)]">
              <Markdown>{spec.spec_md}</Markdown>
            </div>
          ) : (
            <div className="text-xs text-[var(--color-fg-dim)]">no spec.md yet</div>
          )}
        </div>
      </div>

      {editing && (
        <SpecEditModal
          phase={editing}
          onClose={() => setEditing(null)}
          onApplied={reload}
        />
      )}
    </div>
  );
}

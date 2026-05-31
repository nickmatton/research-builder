import { useEffect, useState } from "react";
import { Markdown } from "./Markdown";
import { api } from "../lib/api";
import type {
  Citation,
  CritiqueVerdict,
  SectionCritique,
  SectionDetail,
  SectionSummary,
} from "../lib/types";

interface Props {
  summary: SectionSummary;
  onJumpToPage?: (page: number) => void;
}

const VERDICT_STYLES: Record<CritiqueVerdict, { label: string; cls: string }> = {
  verified: { label: "verified", cls: "bg-[var(--color-ok)]/15 text-[var(--color-ok)]" },
  questionable: {
    label: "questionable",
    cls: "bg-[var(--color-warn)]/15 text-[var(--color-warn)]",
  },
  missing_citations: {
    label: "missing citations",
    cls: "bg-[var(--color-fail)]/15 text-[var(--color-fail)]",
  },
};

function VerdictBadge({ verdict }: { verdict: CritiqueVerdict | null }) {
  if (!verdict) {
    return (
      <span className="rounded bg-[var(--color-surface-2)] px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
        pending
      </span>
    );
  }
  const s = VERDICT_STYLES[verdict];
  return (
    <span
      className={`rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wider ${s.cls}`}
    >
      {s.label}
    </span>
  );
}

function CitationChip({
  citation,
  onJumpToPage,
}: {
  citation: Citation;
  onJumpToPage?: (page: number) => void;
}) {
  const label = citation.section
    ? `§${citation.section} p.${citation.page}`
    : `p.${citation.page}`;
  return (
    <button
      type="button"
      onClick={() => onJumpToPage?.(citation.page)}
      title={citation.quote ?? `Jump to page ${citation.page}`}
      className="inline-flex items-center rounded bg-[var(--color-surface-2)] px-1.5 py-0.5 text-[10px] font-mono text-[var(--color-fg-muted)] hover:bg-[var(--color-surface)] hover:text-[var(--color-accent)]"
    >
      {label}
    </button>
  );
}

export function SectionSpecCard({ summary, onJumpToPage }: Props) {
  const [open, setOpen] = useState(false);
  const [detail, setDetail] = useState<SectionDetail | null>(null);
  const [critique, setCritique] = useState<SectionCritique | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!open || detail) return;
    api
      .section(summary.phase_id)
      .then(setDetail)
      .catch((e: Error) => setErr(e.message));
    if (summary.critique_verdict) {
      api
        .sectionCritique(summary.phase_id)
        .then(setCritique)
        .catch(() => undefined);
    }
  }, [open, summary.phase_id, summary.critique_verdict, detail]);

  return (
    <div className="rounded border border-[var(--color-border)] bg-[var(--color-surface)]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-[var(--color-surface-2)]"
      >
        <span className="text-[10px] text-[var(--color-fg-dim)]">{open ? "▾" : "▸"}</span>
        <span className="font-mono text-[11px] text-[var(--color-fg-muted)]">
          {summary.phase_id}
        </span>
        <span className="flex-1 truncate text-xs text-[var(--color-fg)]">
          {summary.title}
        </span>
        <span className="text-[10px] tabular-nums text-[var(--color-fg-dim)]">
          {summary.criteria_count} ac · {summary.citations_count} cit
        </span>
        <VerdictBadge verdict={summary.critique_verdict} />
      </button>

      {open && (
        <div className="border-t border-[var(--color-border)] px-3 py-2 text-xs">
          {err && <div className="text-[var(--color-fail)]">{err}</div>}
          {!detail && !err && (
            <div className="text-[var(--color-fg-dim)]">Loading section spec…</div>
          )}
          {detail && (
            <>
              {detail.goal && (
                <div className="mb-2">
                  <div className="text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
                    Goal
                  </div>
                  <div className="mt-0.5 text-xs text-[var(--color-fg)]">
                    {detail.goal}
                  </div>
                </div>
              )}

              {detail.acceptance_criteria.length > 0 && (
                <div className="mb-2">
                  <div className="mb-1 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
                    Acceptance criteria
                  </div>
                  <ul className="space-y-1">
                    {detail.acceptance_criteria.map((c, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="mt-0.5 text-[var(--color-fg-dim)]">•</span>
                        <span className="flex-1 text-xs text-[var(--color-fg)]">
                          {c.text}
                        </span>
                        <CitationChip citation={c.source} onJumpToPage={onJumpToPage} />
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {detail.citations.length > 0 && (
                <div className="mb-2">
                  <div className="mb-1 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
                    All citations
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {detail.citations.map((c, i) => (
                      <CitationChip
                        key={i}
                        citation={c}
                        onJumpToPage={onJumpToPage}
                      />
                    ))}
                  </div>
                </div>
              )}

              {critique && critique.reasons.length > 0 && (
                <div className="mb-2 rounded bg-[var(--color-surface-2)] p-2">
                  <div className="mb-1 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
                    Critique
                  </div>
                  <ul className="space-y-0.5">
                    {critique.reasons.map((r, i) => (
                      <li key={i} className="text-[11px] text-[var(--color-fg-muted)]">
                        — {r}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {detail.spec_markdown && (
                <details className="mt-2">
                  <summary className="cursor-pointer text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
                    Full spec markdown
                  </summary>
                  <div className="prose-spec mt-1 text-xs text-[var(--color-fg)]">
                    <Markdown>{detail.spec_markdown}</Markdown>
                  </div>
                </details>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

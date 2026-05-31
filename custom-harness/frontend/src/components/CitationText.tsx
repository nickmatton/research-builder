// Renders text containing [p.N] / [p.N-M] citations as clickable
// page-jump buttons. Everything else is rendered as plain text with
// preserved line breaks.

interface Props {
  text: string;
  onJumpToPage?: (page: number) => void;
}

const CITATION_RE = /\[p\.(\d+)(?:-(\d+))?\]/g;

export function CitationText({ text, onJumpToPage }: Props) {
  const parts: Array<{ kind: "text" | "cite"; value: string; page?: number }> = [];
  let lastIndex = 0;
  for (const m of text.matchAll(CITATION_RE)) {
    if (m.index! > lastIndex) {
      parts.push({ kind: "text", value: text.slice(lastIndex, m.index!) });
    }
    parts.push({ kind: "cite", value: m[0], page: parseInt(m[1], 10) });
    lastIndex = m.index! + m[0].length;
  }
  if (lastIndex < text.length) {
    parts.push({ kind: "text", value: text.slice(lastIndex) });
  }

  return (
    <span className="whitespace-pre-wrap">
      {parts.map((p, i) =>
        p.kind === "text" ? (
          <span key={i}>{p.value}</span>
        ) : (
          <button
            key={i}
            type="button"
            onClick={() => p.page && onJumpToPage?.(p.page)}
            className="mx-0.5 inline-flex items-center rounded border border-[var(--color-border-strong)] bg-[var(--color-surface-2)] px-1 py-px font-mono text-[10px] text-[var(--color-accent)] transition-colors hover:border-[var(--color-accent)] hover:bg-[var(--color-accent)]/10"
            title={`Jump to page ${p.page}`}
          >
            {p.value}
          </button>
        ),
      )}
    </span>
  );
}

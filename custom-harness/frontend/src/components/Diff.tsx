// Renders a structured diff (list of {type, text} lines) as a compact
// unified-diff view. Empty diffs show a "no change" placeholder.

import type { DiffLine } from "../lib/types";

interface Props {
  lines: DiffLine[];
}

export function Diff({ lines }: Props) {
  const hasChange = lines.some((l) => l.type !== "context");
  if (!hasChange) {
    return (
      <div className="rounded border border-[var(--color-border)] bg-[var(--color-surface-2)] px-3 py-2 text-xs text-[var(--color-fg-dim)]">
        No change.
      </div>
    );
  }
  return (
    <div className="overflow-auto rounded border border-[var(--color-border)] bg-[var(--color-bg)] font-mono text-[11px]">
      <table className="w-full border-collapse">
        <tbody>
          {lines.map((l, i) => (
            <DiffRow key={i} line={l} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DiffRow({ line }: { line: DiffLine }) {
  const gutter = line.type === "add" ? "+" : line.type === "remove" ? "−" : " ";
  const rowCls =
    line.type === "add"
      ? "bg-[var(--color-ok)]/10"
      : line.type === "remove"
        ? "bg-[var(--color-fail)]/10"
        : "";
  const textCls =
    line.type === "add"
      ? "text-[var(--color-ok)]"
      : line.type === "remove"
        ? "text-[var(--color-fail)]"
        : "text-[var(--color-fg-muted)]";
  return (
    <tr className={rowCls}>
      <td className={`w-5 select-none border-r border-[var(--color-border)] px-1.5 py-0.5 text-center ${textCls}`}>
        {gutter}
      </td>
      <td className={`whitespace-pre px-2 py-0.5 ${textCls}`}>{line.text || " "}</td>
    </tr>
  );
}

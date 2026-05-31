// Top-bar view switcher. Activity is the default; selecting another
// view swaps the left-panel content. Chat stays permanently on the right.
//
// Keep this strictly presentational — view state lives in App.tsx so
// shortcuts + the command palette can drive it.

export const VIEW_IDS = ["activity", "paper", "spec", "docs", "files", "claims", "compute", "report"] as const;
export type ViewId = (typeof VIEW_IDS)[number];

interface Props {
  active: ViewId;
  onChange: (v: ViewId) => void;
  paperAvailable: boolean;
  errorCount?: number;
}

const LABELS: Record<ViewId, string> = {
  activity: "Activity",
  paper: "Paper",
  spec: "Spec",
  docs: "Docs",
  files: "Files",
  claims: "Claims",
  compute: "Compute",
  report: "Report",
};

export function ViewSwitcher({ active, onChange, paperAvailable, errorCount = 0 }: Props) {
  return (
    <nav className="flex h-9 shrink-0 items-center gap-1 border-b border-[var(--color-border)] bg-[var(--color-surface)] px-3 text-[12px]">
      {VIEW_IDS.map((id) => {
        const isActive = active === id;
        const disabled = id === "paper" && !paperAvailable;
        return (
          <button
            key={id}
            type="button"
            disabled={disabled}
            onClick={() => onChange(id)}
            className={[
              "relative rounded-md px-2.5 py-1 transition-colors",
              "disabled:cursor-not-allowed disabled:opacity-40",
              isActive
                ? "bg-[var(--color-surface-2)] text-[var(--color-fg)]"
                : "text-[var(--color-fg-muted)] hover:bg-[var(--color-surface-2)]/60 hover:text-[var(--color-fg)]",
            ].join(" ")}
          >
            {LABELS[id]}
            {id === "activity" && errorCount > 0 && (
              <span className="ml-1.5 inline-flex h-4 min-w-4 items-center justify-center rounded-full bg-[var(--color-fail)]/20 px-1 text-[9px] font-medium text-[var(--color-fail)]">
                {errorCount}
              </span>
            )}
          </button>
        );
      })}
    </nav>
  );
}

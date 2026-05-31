import { useState, type ReactNode } from "react";

export interface Tab {
  id: string;
  label: string;
  body: ReactNode;
  /** When true, show a pulsing dot on the tab to indicate the tab needs attention. */
  badge?: boolean;
}

interface Props {
  tabs: Tab[];
  initialId?: string;
  /** When provided, makes the component fully controlled. */
  activeId?: string;
  onChange?: (id: string) => void;
}

export function Tabs({ tabs, initialId, activeId, onChange }: Props) {
  const [internal, setInternal] = useState(initialId ?? tabs[0]?.id);
  const active = activeId ?? internal;
  const setActive = (id: string) => {
    if (activeId === undefined) setInternal(id);
    onChange?.(id);
  };
  const current = tabs.find((t) => t.id === active) ?? tabs[0];

  return (
    <div className="flex h-full flex-col">
      <div role="tablist" className="flex h-9 shrink-0 items-center gap-1 border-b border-[var(--color-border)] bg-[var(--color-surface)] px-2">
        {tabs.map((t) => {
          const isActive = t.id === current?.id;
          return (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={isActive}
              className={`relative rounded px-2.5 py-1 text-xs transition-colors ${
                isActive
                  ? "bg-[var(--color-surface-2)] text-[var(--color-fg)]"
                  : "text-[var(--color-fg-muted)] hover:text-[var(--color-fg)]"
              }`}
              onClick={() => setActive(t.id)}
            >
              {t.label}
              {t.badge && (
                <span
                  className="absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full bg-[var(--color-accent)]"
                  title="Awaiting your approval"
                />
              )}
            </button>
          );
        })}
      </div>
      <div className="min-h-0 flex-1 overflow-hidden">{current?.body}</div>
    </div>
  );
}

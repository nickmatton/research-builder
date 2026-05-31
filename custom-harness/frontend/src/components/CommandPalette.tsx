// ⌘K / Ctrl+K command palette. Fuzzy-filtered, keyboard-driven.
//
// Commands are passed in by the parent so the palette is a pure picker —
// it doesn't know about app state. Each command supplies `id`, `label`,
// optional `hint`, and `run`. The parent decides what's available based
// on current state (e.g. only show "Apply edit" when a phase is selected).

import { useEffect, useMemo, useRef, useState } from "react";

export interface Command {
  id: string;
  label: string;
  hint?: string;
  /** Optional group label for visual separation in the list. */
  group?: string;
  run: () => void;
}

interface Props {
  open: boolean;
  onClose: () => void;
  commands: Command[];
}

export function CommandPalette({ open, onClose, commands }: Props) {
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return commands;
    return commands.filter((c) => {
      const hay = `${c.label} ${c.hint ?? ""} ${c.group ?? ""}`.toLowerCase();
      // Sub-sequence match: every character of q appears in hay in order.
      let i = 0;
      for (const ch of hay) {
        if (ch === q[i]) i++;
        if (i === q.length) return true;
      }
      return i === q.length;
    });
  }, [commands, query]);

  useEffect(() => {
    if (open) {
      setQuery("");
      setActive(0);
      // Microtask: focus after the modal renders.
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [open]);

  useEffect(() => {
    // Clamp active when filtered list shrinks.
    if (active >= filtered.length) setActive(Math.max(0, filtered.length - 1));
  }, [filtered.length, active]);

  // Scroll active row into view.
  useEffect(() => {
    if (!listRef.current) return;
    const row = listRef.current.querySelector<HTMLDivElement>(
      `[data-cmd-idx="${active}"]`,
    );
    row?.scrollIntoView({ block: "nearest" });
  }, [active]);

  if (!open) return null;

  const onKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((i) => Math.min(filtered.length - 1, i + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(0, i - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const cmd = filtered[active];
      if (cmd) {
        onClose();
        cmd.run();
      }
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/50 p-6 pt-[12vh] backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="w-full max-w-xl overflow-hidden rounded-lg border border-[var(--color-border-strong)] bg-[var(--color-surface)] shadow-2xl shadow-black/60"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="border-b border-[var(--color-border)]">
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKey}
            placeholder="Type a command…"
            className="w-full bg-transparent px-4 py-3 text-sm outline-none placeholder:text-[var(--color-fg-dim)]"
          />
        </div>
        <div ref={listRef} className="max-h-[50vh] overflow-auto py-1">
          {filtered.length === 0 ? (
            <div className="px-4 py-6 text-center text-xs text-[var(--color-fg-dim)]">
              No matches.
            </div>
          ) : (
            filtered.map((c, i) => (
              <div
                key={c.id}
                data-cmd-idx={i}
                className={`flex cursor-pointer items-center gap-3 px-4 py-1.5 text-xs ${
                  i === active
                    ? "bg-[var(--color-accent)]/15 text-[var(--color-fg)]"
                    : "text-[var(--color-fg-muted)] hover:bg-[var(--color-surface-2)]"
                }`}
                onMouseEnter={() => setActive(i)}
                onClick={() => {
                  onClose();
                  c.run();
                }}
              >
                {c.group && (
                  <span className="w-16 shrink-0 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
                    {c.group}
                  </span>
                )}
                <span className="flex-1 truncate">{c.label}</span>
                {c.hint && (
                  <span className="font-mono text-[10px] text-[var(--color-fg-dim)]">{c.hint}</span>
                )}
              </div>
            ))
          )}
        </div>
        <div className="flex items-center justify-between border-t border-[var(--color-border)] bg-[var(--color-surface-2)] px-3 py-1.5 text-[10px] text-[var(--color-fg-dim)]">
          <span>↑↓ navigate · ↵ run · esc close</span>
          <span>⌘K</span>
        </div>
      </div>
    </div>
  );
}

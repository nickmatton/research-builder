// Global keyboard-shortcut hook. Skips events that originate inside
// editable elements (input/textarea/contenteditable) unless explicitly
// allowed — so ⌘K still fires inside a text field but `/` only fires
// when nothing has focus.

import { useEffect } from "react";

export interface Shortcut {
  /** Key match function. Receives the raw KeyboardEvent. */
  match: (e: KeyboardEvent) => boolean;
  /** Handler. Called only if no editable element has focus (unless allowEditable). */
  run: (e: KeyboardEvent) => void;
  /** Allow the shortcut to fire even when an input/textarea has focus. */
  allowEditable?: boolean;
}

function isEditable(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    target.isContentEditable
  );
}

export function useShortcuts(shortcuts: Shortcut[]) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      for (const s of shortcuts) {
        if (!s.match(e)) continue;
        if (!s.allowEditable && isEditable(e.target)) continue;
        s.run(e);
        return;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [shortcuts]);
}

/** ⌘K on macOS, Ctrl+K elsewhere. */
export const cmdK = (e: KeyboardEvent) =>
  e.key.toLowerCase() === "k" && (e.metaKey || e.ctrlKey) && !e.shiftKey;

/** `/` with no modifiers. */
export const slash = (e: KeyboardEvent) =>
  e.key === "/" && !e.metaKey && !e.ctrlKey && !e.altKey;

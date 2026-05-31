// Shared role/glyph constants used by both AgentsView and TraceView.
// Single source of truth so adding a new sub-agent role (or changing a
// glyph) needs exactly one edit.

export const ROLE_GLYPH: Record<string, string> = {
  refiner: "📝",
  researcher: "🔬",
  builder: "🔨",
  verifier: "✅",
};

export function glyphFor(role: string): string {
  return ROLE_GLYPH[role] ?? "•";
}

// Strips the "phase:" prefix on sub-agent IDs so the UI can show a
// compact role-style label. Keeps "orchestrator" as-is.
export function shortAgentId(id: string): string {
  if (id === "orchestrator") return "orch";
  if (id.startsWith("phase:")) return id.slice("phase:".length);
  return id;
}

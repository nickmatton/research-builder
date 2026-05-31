// Compute a unified-diff-style structured representation of two strings.
//
// Output shape matches the existing ``DiffLine`` type the Diff component
// consumes ({type: "context"|"add"|"remove", text}). Algorithm: LCS over
// lines (Myers-style two-pointer), then walk it emitting context/add/remove.
//
// We use a basic LCS rather than pulling in a diff library because the
// inputs are bounded (256KB cap at the source) and the dependency budget
// here is tight. For workspace-spec edits this matches what the harness's
// Python-side ``difflib.unified_diff`` produces in shape.

import type { DiffLine } from "./types";

const CONTEXT_WINDOW = 3;

export function computeDiff(
  before: string | null,
  after: string | null,
): DiffLine[] {
  const a = (before ?? "").split("\n");
  const b = (after ?? "").split("\n");
  // Trim trailing empty line from a final newline so the diff doesn't
  // show a phantom "" row at the end.
  if (a.length > 0 && a[a.length - 1] === "") a.pop();
  if (b.length > 0 && b[b.length - 1] === "") b.pop();

  const lcs = lcsTable(a, b);
  // Walk in order, classifying each line.
  const ops: DiffLine[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      ops.push({ type: "context", text: a[i] });
      i++;
      j++;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      ops.push({ type: "remove", text: a[i] });
      i++;
    } else {
      ops.push({ type: "add", text: b[j] });
      j++;
    }
  }
  while (i < a.length) ops.push({ type: "remove", text: a[i++] });
  while (j < b.length) ops.push({ type: "add", text: b[j++] });

  // Collapse runs of unchanged lines into ±CONTEXT_WINDOW around each
  // change, matching the unified-diff convention. Pure context (no
  // additions/removals at all) returns empty so the UI can show
  // "no change".
  return collapseContext(ops, CONTEXT_WINDOW);
}

function lcsTable(a: string[], b: string[]): number[][] {
  // m+1 × n+1 table; lcs[i][j] = longest common subsequence of a[i:] and b[j:].
  const m = a.length;
  const n = b.length;
  const lcs: number[][] = [];
  for (let i = 0; i <= m; i++) {
    lcs.push(new Array(n + 1).fill(0));
  }
  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      lcs[i][j] = a[i] === b[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  return lcs;
}

function collapseContext(ops: DiffLine[], window: number): DiffLine[] {
  // Find indices of change ops. Anything within `window` of one stays
  // as context; the rest gets elided. We mark elisions with a synthetic
  // separator row so the user knows lines were skipped.
  const changeIdx: number[] = [];
  for (let i = 0; i < ops.length; i++) {
    if (ops[i].type !== "context") changeIdx.push(i);
  }
  if (changeIdx.length === 0) return [];
  const keep = new Array(ops.length).fill(false);
  for (const ci of changeIdx) {
    const lo = Math.max(0, ci - window);
    const hi = Math.min(ops.length - 1, ci + window);
    for (let k = lo; k <= hi; k++) keep[k] = true;
  }
  const out: DiffLine[] = [];
  let elided = false;
  for (let i = 0; i < ops.length; i++) {
    if (keep[i]) {
      if (elided) {
        out.push({ type: "context", text: "…" });
        elided = false;
      }
      out.push(ops[i]);
    } else {
      elided = true;
    }
  }
  return out;
}

# skills/

Methodology — the durable, paper-agnostic patterns for reproducing research papers. **These are the canonical masters.** Each new paper repo gets its own copy at `<paper-repo>/.claude/skills/` via `bin/new-paper`.

## Contents

| File | What it is |
|---|---|
| `reproduce-not-search.md` | Spec-authoring + implementation discipline: reproduce the paper's reported result with the paper's reported configuration. Never redo the hyperparameter search the paper already paid for. Carve-out for the case where the sweep IS the deliverable (sensitivity ablations). Where to find the paper's winning config (text / caption / appendix / code release). |
| `verification-ladder.md` | The ordered gates from cheapest → most expensive: unit tests → toy example → overfit-one-batch → smoke → short training → full reproduction. **Don't skip rungs.** Per-phase test focus for data / architecture / training / eval / results. The "if results don't match the paper, design the cheapest distinguishing experiment" debugging loop. |
| `post-mortem.md` | What to write after every failed run. One focused hypothesis, classified as implementation-issue vs spec-issue, with confidence rubric. Frontmatter format for `<paper-repo>/notes/post-mortems/<run-id>.md`. The discipline of externalizing hypotheses across sessions. |
| `compare-to-paper.md` | The acceptance gate. `verified | close | missed | exceeded | not_checked` rubric. **Exceeded is a red flag** (data leak / wrong split / metric mismatch), not a win. How to read the comparison table; convergence sanity for training phases; claim_id naming discipline. |

## Why these and not more (yet)

The current set is intentionally narrow: each skill is forged from real harness experience, not speculative. The set should grow as we reproduce more papers and discover patterns worth distilling — candidates: `picking-a-paper.md`, `extracting-claims.md`, `eval-tokenization-gap.md`, `training-curve-comparison.md`, `report-writing.md`. Add when there's a real lesson to capture, not before.

## Authoring rule

Skills should be:
- **Imperative** — "do X", not "consider doing X"
- **Specific** — quote real tolerances, real failure modes, real numbers
- **Paper-agnostic** — works for translation / vision / RL / theory papers alike. If a skill ends up paper-type-specific, it belongs in `tools/` or in a paper-type-specific scaffold, not here.
- **Short** — ~50–100 lines. A skill that needs more is probably two skills.

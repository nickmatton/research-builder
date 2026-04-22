# <PAPER TITLE>

<!--
This is the reproduction spec. Claude Code reads it on every session.
Put the stuff you'd otherwise re-explain. When a discrepancy with the
paper gets resolved, the resolution goes HERE — this file is institutional
memory, not a README.
-->

## Citation

> <Authors>. **<Title>.** <Venue, Year>. <arXiv:XXXX.XXXXX>

## Summary (5 lines)

<2–3 sentences on the method. What does the paper propose, what's the key
mechanism, what does it beat and by how much. Aim for "explain it to a
colleague in one elevator ride." Do not copy the abstract.>

## Headline claims (3–10)

The numbers we're trying to reproduce. Each one with the exact table/figure/page
reference so it's trivially falsifiable.

- **<claim_id>**: <metric> = <value> on <dataset> — <Table/Figure X, page N>
- ...

Full structured ledger: `notes/claims.yaml`.

## Method / architecture

<Key equations, layer structure, anything that differs from "a normal
<architecture type>". Reference paper section numbers so a reader can
double-check your interpretation.>

## Hyperparameters

Everything needed to actually run the thing, with paper references.

| Parameter | Value | Source |
|---|---|---|
| optimizer | <AdamW> | §4.1 |
| learning rate | <3e-4> | Table 1 |
| batch size | <256> | §4.1 |
| schedule | <cosine, 10k warmup> | §4.1 |
| epochs / steps | <100k steps> | §4.1 |
| seed(s) | <42, 123, 2024> | we fix |

## Datasets

- **<dataset>** — location: `data/<subdir>/`, download: `scripts/fetch-<dataset>.sh`, preprocessing: `src/data.py`
- ...

## Compute budget

- **Smoke**: runs in < 1 min on CPU.
- **Overfit one batch**: < 5 min on a single GPU.
- **Full**: ~<N> GPU-hours on <hardware>.

Use `scripts/smoke.sh` before anything full-scale. Always.

## Commands

```bash
# Verification ladder — run in order
uv run pytest                      # 1. unit tests
bash scripts/overfit-one-batch.sh  # 2. overfit-one-batch sanity
bash scripts/smoke.sh              # 3. smoke run (100 steps, tiny data)
bash scripts/reproduce.sh          # 4. full reproduction

# Compare results to paper after a run
# (uses the claims MCP server)
```

## Workflow expectations

When you sit down to work on this repo:

1. **Read `notes/plan.md` first.** Don't start coding without a plan.
2. **Follow `.claude/skills/verification-ladder.md`.** Cheapest gate first. Do not skip rungs.
3. **Every full run logs to `notes/journal.md`.** One row per run. Git SHA, config hash, key metrics, duration.
4. **Every failed run gets a post-mortem** per `.claude/skills/post-mortem.md` before retrying.
5. **Every discrepancy with the paper gets resolved in this file** (`CLAUDE.md`), not in the chat.

## Open questions / known gotchas

<Things the paper is ambiguous about. Discoveries made during reproduction
that future-you should know. This section grows as you work.>

- ...

## Paper location

`paper/<slug>.pdf` — indexed for semantic search via the `paper` MCP server.

## Reference implementations

<Links to author code, community reproductions, or well-known re-implementations.
When these exist, *read them* before implementing; the paper often omits details
the code reveals. Diff your implementation against them once you have both.>

- Author code: <url>
- Community: <url>

# Run Journal — Attention Is All You Need

Append-only log of every meaningful run. Most-recent at the bottom.

The `/reproduce` slash command appends a row automatically after each full run. Add rows manually for smoke / overfit / partial runs you want to remember.

## Format

```
## <run-id>  (<ISO 8601 timestamp>)
**Type:** smoke | overfit-one-batch | short-train | full
**Git SHA:** <short sha>
**Config:** <configs/<file>.yaml> · hash <md5>
**Hardware:** <gpu / cpu>
**Duration:** <wall-clock>

**Key metrics**
- <metric>: <value>

**Claims verification** (full runs only)
- verified: <n>, close: <n>, missed: <n>, exceeded: <n>, not_checked: <n>
- See `runs/<run-id>/claims-report.md`.

**Notes**
<one or two sentences. What did this run prove or fail to prove?>
```

---

## Runs

## scaffold-2026-04-23  (2026-04-23T12:50)
**Type:** scaffold (not a training run)
**Git SHA:** d2014ca (research-builder toolkit)

**Notes**
Repo scaffolded from `paper-template/`. PDF placed at `paper/paper.pdf`, extracted to `paper/paper.txt` via `extract-paper-text.py` (15 pages, all readable). `CLAUDE.md` populated with citation, summary, hyperparameters from §3 + Table 3, datasets from §5.1, compute budget from §5.2. `notes/claims.yaml` populated with 6 claims from Table 2 (4 BLEU) + §5.2 (2 step counts). Implementation has not started yet — `src/` is empty. Plan-mode session pending: decide between reproducing the base model (~12 GPU-h on P100, in-budget) vs deferring the big model.

<!-- Append run blocks below. -->

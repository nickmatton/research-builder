# Run Journal

Append-only log of every meaningful run. One row per run. Most-recent at the bottom (matches git log order).

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
- ...

**Claims verification** (full runs only)
- verified: <n>, close: <n>, missed: <n>, exceeded: <n>, not_checked: <n>
- See `runs/<run-id>/claims-report.md` for the table.

**Notes**
<one or two sentences. What did this run prove or fail to prove? What changed since the last run?>
```

---

## Runs

<!-- Append run blocks below this line. -->

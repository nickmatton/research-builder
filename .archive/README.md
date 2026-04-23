# .archive/ — preserved artifacts from the original harness

This directory holds runtime outputs from the original 9.5k LoC research-builder harness (the orchestrator + sub-agent pipeline that preceded the current Claude-Code-native template).

It is gitignored — these are reference snapshots, not part of the toolkit. They exist so that:

1. The methodology that survived the rewrite (claims ledger, verification ladder, post-mortem discipline) has its provenance — these snapshots are where those patterns were first stress-tested.
2. The "before" of the migration is recoverable. If a question comes up later (*"how did the old harness handle spec amendments?"*), the run logs are here.
3. Future side-by-side comparisons (same paper, old harness vs new template) can use these as the historical baseline without re-running.

## What's here

- **`<YYYYMMDD-HHMMSS>/`** — one directory per harness run, timestamped. Each contains the harness's full output: `canonical_spec/`, `phases/<phase_id>/<try_num>/{src,outputs}/`, `logs/{events.jsonl, postmortems/, spec_amendments/, claims/}`, `report/reproduction_report.md`.

- **(Phase 4, future)** `research-builder-v1/` — the harness source code itself, moved out of `src/research_builder/` once `papers/attention-is-all-you-need/` validates the new template end-to-end.

## Notes

- Total size is large (~4 GB at last check, dominated by checkpoints under `phases/training/`). Don't `git add` this directory — `.gitignore` excludes it for a reason.
- Run snapshots use the old harness's directory layout (`canonical_spec/spec.md` + `state.yaml`). The new template uses `notes/claims.yaml` + `notes/journal.md` + per-run `runs/<id>/` instead.
- If you need to *resume* a snapshot, you'd need the old harness reinstalled (the CLI entry was `research-builder <paper.pdf> -o output/` — see `pyproject.toml` `[project.scripts]` until Phase 4 archives it).

## Why keep the old harness at all

Two reasons:

1. **Engineering proof.** The harness shows the complex version is achievable — orchestrator/sub-agent decomposition, MCP tools, retry budgets, failure post-mortems, cross-phase acceptance review. The new template is the deliberate simplification, not the only thing we know how to build.
2. **The methodology ledger.** Patterns like `claims-ledger`, `verification-ladder`, and `post-mortem` are *paper-agnostic*. They were validated against the harness's runs before being extracted into `paper-template/.claude/skills/`. The provenance matters.

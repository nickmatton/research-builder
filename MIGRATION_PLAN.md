# Migration Plan: Custom Harness → Claude Code Native

**Branch:** `migration/claude-code-native`
**Started:** 2026-04-19
**Goal:** Replace ~9.5k LoC of custom orchestration with a Claude Code-native template + small MCP layer + skills, while preserving the *opinionated reproduction methodology* (claims ledger, verification ladder, post-mortem discipline, structured run journal).

---

## Guiding principles

1. **Preserve methodology, drop infrastructure.** The harness's *patterns* (claims-first, verification ladder, post-mortem on every failed run) are the durable value. The Python control plane is not.
2. **Validate before deleting.** Native approach has to reproduce at least one paper end-to-end before we archive the old harness.
3. **One paper = one repo.** The new unit of work is a *paper repo*, not "research-builder run a paper." This is the biggest mental shift.
4. **Lean on Claude Code's harness.** Plan mode for phase decomposition. `/loop` for iteration. Hooks for journaling. Skills for ML patterns. Slash commands for repeat ops.

---

## Architecture decision: built-in tools over MCP

**2026-04-23.** Initial Phase 2 built three MCP servers (paper, arxiv, claims). On reflection: MCP earns its keep when there are many tools, multiple clients, or stateful sessions. For our 3-server / ~10-tool case used inside one harness (Claude Code), the wins are marginal — and the costs are real: a `mcp` dep, ~40 transitive packages, subprocess management, stdout-discipline risk, and a per-paper `uv pip install -e <toolkit>` step that breaks portability.

Pivot: drop MCP, lean on Claude Code's built-ins (Read, Write, Bash, Grep) plus three small Python scripts:

- **paper_reader → `scripts/extract-paper-text.py`** runs once, dumps `paper/paper.txt` with `--- Page N ---` markers. Claude reads/greps it with built-ins.
- **arxiv → `scripts/lookup-citation.py`** thin Semantic Scholar wrapper handling the API key. Called via Bash, JSON to stdout.
- **claims → `scripts/compare-claims.py`** + direct Read/Write on `notes/claims.yaml`. Script does verification + markdown table; YAML is plain-text edited.

Net: ~410 LoC of MCP + 40 transitive deps → ~200 LoC of scripts + zero new deps. **Paper repos are now truly portable** — no toolkit install, just `cp -r paper-template/`, drop the PDF, go.

## End-state architecture

```
research-builder/                  # this repo, the template + methodology source
├── README.md
├── MIGRATION_PLAN.md              # this file
├── paper-template/                # copied per paper
│   ├── CLAUDE.md
│   ├── README.md
│   ├── .gitignore
│   ├── .claude/
│   │   ├── skills/                # verification-ladder, post-mortem, compare-to-paper
│   │   └── commands/              # /reproduce, /compare, /verify, /post-mortem
│   ├── notes/{claims.yaml, plan.md, journal.md, post-mortems/}
│   ├── scripts/
│   │   ├── extract-paper-text.py  # PDF → paper.txt (one-time)
│   │   ├── compare-claims.py      # verify run metrics vs claims ledger
│   │   ├── lookup-citation.py     # Semantic Scholar wrapper
│   │   ├── smoke.sh
│   │   ├── overfit-one-batch.sh
│   │   └── reproduce.sh
│   ├── configs/.gitkeep
│   ├── src/.gitkeep
│   └── tests/.gitkeep
└── custom-harness/                # the original 9.5k LoC harness, frozen for reference (Phase 4)
```

A new paper:
```
papers/<paper-slug>/               # cp -r paper-template/
├── CLAUDE.md                      # filled in first
├── paper/{paper.pdf, paper.txt}   # PDF dropped in, text auto-extracted
├── notes/{claims.yaml, plan.md, journal.md, post-mortems/}
├── .claude/{skills,commands}/
├── scripts/
├── src/, tests/, configs/
└── data/, runs/                   # gitignored
```

No MCP servers. No `.mcp.json`. No toolkit install required in the paper repo.

---

## Phases

Each phase has explicit success criteria. Don't proceed until the previous phase's gate is green. Mirror the verification-ladder pattern we're trying to encode in the tool itself.

### Phase 0 — Prep (in progress)

- [x] Branch `migration/claude-code-native` created
- [x] This plan committed
- [ ] **Decision point**: review plan with user, get green-light to proceed
- [ ] WIP triage: 16 modified files + many untracked. Decide: commit-as-checkpoint, stash, or carry along? (Recommend: one WIP commit on `main` so this branch stays clean.)

**Gate:** plan approved, WIP triaged.

### Phase 1 — Extract methodology artifacts (preserve before delete)

Pull the *content* out of the harness before it goes to `custom-harness/`. No deletes yet.

- [ ] Extract claims schema from `src/research_builder/models/claims.py` → `paper-template/notes/claims.md` template + `mcp/claims_server.py` design notes
- [ ] Extract verification-ladder pattern from `src/research_builder/orchestrator/loop.py` and tests → `skills/verification-ladder.md`
- [ ] Extract post-mortem template from `src/research_builder/orchestrator/failure.py` and `logs/postmortems/` examples → `skills/post-mortem.md`
- [ ] Extract `report_result` tool schema from `src/research_builder/sub_agent/tools.py` → docs (probably becomes hook output format or a slash command)
- [ ] Extract `read_paper_section` and `lookup_citation` tools → port plan for `mcp/paper_reader_server.py` and `mcp/arxiv_server.py`
- [ ] Inventory the system prompts in `src/research_builder/sub_agent/prompts.py` and `src/research_builder/orchestrator/agent.py` — distill the *reproduction philosophy* into `paper-template/CLAUDE.md.j2`

**Gate:** every piece of harness "knowledge" has a home in `skills/`, `mcp/`, or `paper-template/`. We can list what's preserved.

### Phase 2 — Build native scaffolding

- [ ] Write `paper-template/CLAUDE.md.j2` (the reproduction-spec scaffold from the blueprint)
- [ ] Write `paper-template/notes/{claims.md, journal.md, plan.md}` skeletons
- [ ] Write `paper-template/scripts/{smoke.sh, reproduce.sh, overfit-one-batch.sh}` skeletons
- [ ] Implement `mcp/paper_reader_server.py` (port from `sub_agent/tools.py:read_paper_section`)
- [ ] Implement `mcp/arxiv_server.py` (port from `sub_agent/tools.py:lookup_citation`)
- [ ] Implement `mcp/claims_server.py` (CRUD against `notes/claims.md`, supports `/compare-to-paper`)
- [ ] Write `skills/{verification-ladder, post-mortem, compare-to-paper, pytorch-training-loop}.md`
- [ ] Write `commands/{reproduce, compare, verify, post-mortem}.md`
- [ ] Write `hooks/on-stop-journal.sh` — appends `notes/journal.md` row on Stop event
- [ ] Wire MCP servers into `paper-template/.mcp.json`
- [ ] Smoke-test each MCP server independently (jsonrpc roundtrip)

**Gate:** can manually `cp -r paper-template papers/test-paper`, open Claude Code in it, and have it find the paper, read a section, write to `claims.md`.

### Phase 3 — Validate on `attentionisallyouneed/`

The existing test paper is the proving ground. Reproduce it with the *new* approach.

- [ ] Generate `papers/attention-is-all-you-need/` from `paper-template`
- [ ] Move PDF over, write the CLAUDE.md (this is the test of the template — does it scaffold cleanly?)
- [ ] Run plan mode → save to `notes/plan.md`
- [ ] Walk the verification ladder: unit tests → overfit one batch → smoke run → short train → compare claims
- [ ] Compare reproduction quality + time against the old harness's run on the same paper

**Gate:** paper reproduces with comparable quality to the old harness. Document deltas in journal.md. **This is the go/no-go for archive.**

### Phase 4 — Decommission the old harness

Only after Phase 3 gate is green.

- [ ] Move `src/research_builder/orchestrator/` → `custom-harness/orchestrator/`
- [ ] Move `src/research_builder/sub_agent/` → `custom-harness/`
- [ ] Move `src/research_builder/storage/` → `custom-harness/`
- [ ] Move `src/research_builder/viewer/` → `custom-harness/`
- [ ] Move `src/research_builder/{events,commands,resume,chat,console,ui,interaction,main}.py` → `custom-harness/`
- [ ] Move `src/research_builder/{cloud,literature,rag}/` → `custom-harness/` (revisit later — may have reusable parts)
- [ ] Keep `src/research_builder/models/claims.py` if MCP server reuses it; otherwise archive
- [ ] Update `pyproject.toml` — drop CLI entry points, drop unused deps
- [ ] Update `README.md` — new architecture, new workflow
- [ ] Run remaining tests; delete tests for archived modules

**Gate:** `pyproject.toml` reflects new shape, README explains new workflow, no orphan imports, tests pass.

### Phase 5 — Polish

- [ ] Document the per-paper workflow with a worked example from `attention-is-all-you-need/`
- [ ] Add a `bin/new-paper` script (or a slash command) that scaffolds a new paper repo from the template
- [ ] Optional: a meta-command `/intake-paper <pdf>` that reads the PDF, drafts CLAUDE.md, extracts claims, commits the scaffold
- [ ] Set up `hooks/on-stop-journal.sh` in user's global `~/.claude/settings.json` (or paper-template-scoped)
- [ ] Cross-paper retrospective: things to add to the template after running 2–3 papers

**Gate:** you can scaffold + start reproducing a new paper in <10 minutes.

---

## What we are explicitly NOT doing

- **No backwards compatibility shims.** The old CLI dies; users (you) move to the new workflow.
- **No data migration.** Existing `canonical_spec/`, `phases/`, `logs/`, etc. stay where they are; can be referenced from `custom-harness/` or wiped after Phase 3.
- **No "research-builder as a library"** that paper repos import. The toolkit is templates + skills + MCP servers — *files copied or referenced*, not Python code imported.
- **No GUI / TUI viewer port.** Claude Code's TUI is the viewer.
- **No retry state machine.** Claude Code session + `/loop` + plan mode covers this.
- **No per-phase cost tracking infrastructure.** Claude Code reports session cost; if we want per-phase cost, that's a hook.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Native approach actually worse on real paper | Phase 3 gate is explicit. If it fails, we revisit hybrid scope before archiving. |
| Lose audit trail (events.jsonl, etc.) | Hooks-based journaling + git history + Claude Code transcripts cover most of it. Document the gap. |
| Reproducibility regression (no revision_log.yaml) | journal.md + git is the new revision log. Encode the discipline in the template. |
| Cloud / GPU provisioning (cloud/) lost | Out of scope for v1. Keep `custom-harness/src/research_builder/cloud/` for reference; reintroduce as MCP server later if needed. |
| Test coverage drops | Tests for archived modules go away with them; tests for MCP servers added in Phase 2. Net coverage probably down — that's fine, surface area is much smaller. |
| WIP work uncommitted on main | Commit a WIP checkpoint on main before merging this branch back. Phase 0 task. |

---

## Rough sizing

| Phase | Effort | Reversibility |
|---|---|---|
| 0 | 30 min | Trivial |
| 1 | 2–3 hr (mostly reading + extracting) | Trivial (no deletes) |
| 2 | 1–2 days (MCP servers + templates + skills) | Easy (new code, additive) |
| 3 | 1 day (real paper run) | Easy (paper repo is separate dir) |
| 4 | 2–3 hr | **Destructive** — but `custom-harness/` keeps it recoverable |
| 5 | Half-day + ongoing | Easy |

Total: roughly a week of focused work to a green Phase 4. Phase 5 is continuous.

---

## Decision point

Phases 0–3 are safe (no destructive changes). Phase 4 is the irreversible one (modulo `custom-harness/`).

Confirm:
1. The end-state architecture (paper-template + mcp + skills + commands + archive)
2. The phase breakdown and gates
3. Anything to add to "what we're NOT doing"

Then I'll start Phase 1.

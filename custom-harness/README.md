# custom-harness

Autonomous Python harness that reproduces a research paper end to end. Takes a PDF, produces verified numerical results, code, and a reproduction report. Peer to the interactive Claude Code skill workflow at the repo root.

The harness is driven by a single long-running LLM orchestrator that calls MCP tools to plan, dispatch per-section sub-agents, review outputs, and amend the spec on failure. Inline TUI shows phase progress, file writes, agent thinking, and cost tracking in the same terminal.

## Quick start

```bash
uv sync
uv run research-builder --test
```

`--test` runs the bundled 4-page test paper to `/tmp/rb-test`, wipes prior state, and implies `--auto` and `--dev`. One command, no setup. Useful for iterating on harness changes.

Full invocation:

```bash
uv run research-builder /path/to/paper.pdf -o ./workspace --auto
```

Requires Python ≥3.11 and `uv`. An `ANTHROPIC_API_KEY` in `.env` (or `--dev` for the Claude Code subscription path) is mandatory. `LAMBDA_API_KEY` is optional and only needed for GPU phases.

## CLI flags

| Flag | Default | Notes |
|---|---|---|
| `PAPER` (positional) | prompt | Path to the paper PDF |
| `--output, -o` | `./<paper-name>/` | Workspace directory (used as-is) |
| `--model, -m` | `Config.model` | Override Claude model |
| `--auto` | off | Run without interactive checkpoints |
| `--dev` | off | Use Claude Code subscription via the bundled `claude` CLI. Unsets `ANTHROPIC_API_KEY`. Implies `--verbose` |
| `--test` | off | Bundled test paper to `/tmp/rb-test`; implies `--auto --dev --wipe` |
| `--verbose, -v` | off | DEBUG-level console logging |
| `--resume` | off | Resume an existing partial run |
| `--fresh` | off | Archive any existing run, start over |
| `--wipe` | off | Delete any existing run, start over |
| `--gpu-budget` | $30 | Hard cap on GPU spend (env: `RB_GPU_BUDGET_USD`) |
| `--llm-budget` | $20 | Hard cap on LLM spend (env: `RB_LLM_SPEND_CAP_USD`); set to 0 to disable |
| `--max-retries` | 3 | Orchestrator retries per phase |
| `--max-debug-attempts` | 10 | Sub-agent debug attempts per invocation |
| `--event-log` | `<output>/logs/events.jsonl` | JSONL event stream for external viewers; `--no-event-log` to disable |
| `--command-log` | `<output>/logs/commands.jsonl` | JSONL command stream for inbound operator chat; `--no-command-log` to disable |
| `--allow-dir` | none | Repeatable; extra directories the sandbox may read/write |

## Architecture

Two LLM-driven tiers, both running on the Claude Agent SDK.

**Tier 1: orchestrator** (`orchestrator/`). One long-running session decides what to do next by calling MCP tools: read the paper, draft the spec, extract claims, dispatch sub-agents per section, review results, run post-mortem on failure, amend the spec, retry. Pauses for operator approval in interactive mode.

**Tier 2: sub-agent chain** (`sub_agent/`). Each section runs as a four-stage chain (planner, implementer, verifier, reporter). Sub-agents read the paper natively via the built-in Read tool with the `pages` parameter. Each sub-agent has its own debug budget; spec-issue returns escape both budgets.

```
orchestrator (Tier 1)
  while phases not done:
    decide_next_action()                         <- LLM
    if gpu needed: cloud.provision(phase)
    result = await dispatch_section(phase)        <- spawns Tier 2
    if !accept and budget.has_retries():
      pm = post_mortem(result)                    <- LLM
      if pm.spec_issue: refine_spec(phase)        <- LLM
      retry

section (Tier 2): planner -> implementer -> verifier -> reporter
```

Key modules:

- `orchestrator/agent.py`, `loop.py`: outer LLM loop and Python state machine
- `orchestrator/dependency.py`, `failure.py`: phase DAG and retry budgets
- `sub_agent/agent.py`, `prompts.py`: section chain implementation
- `models/spec.py`, `claims.py`: Pydantic schemas
- `storage/spec_store.py`, `workspace.py`: `spec.md`, `state.json`, `revision_log.yaml`, per-attempt isolation under `phases/<id>/<try>/`
- `cloud/provisioner.py`: Lambda Cloud client, `BudgetLedger` with hard cap, ephemeral SSH keys
- `viewer/inline.py`: Rich-based inline TUI (no separate terminal, no Textual)
- `events/`, `commands/`: JSONL event stream and inbound operator chat

## Outputs

A run writes to `<output>/`:

```
canonical_spec/
  spec.md              LLM-authored canonical interpretation
  state.json           machine-readable phase state and dependency graph
  claims.json          extracted claims ledger
  revision_log.yaml    append-only event log
phases/<phase_id>/<attempt>/
  src/                 generated code
  outputs/             artifacts, metrics, model weights
  logs/                per-attempt logs
report/
  reproduction_report.md   final claim-by-claim verification
logs/
  run.log              full DEBUG log
  events.jsonl         event stream
  commands.jsonl       inbound command stream
notes/
  run_errors.md        aggregated warnings and errors
```

## Tests

```bash
uv run pytest
```

Unit and integration tests cover models, storage, dependency graph, failure handling, the execution loop, and the section chain with mocked LLM calls.

## Worked examples

- [`examples/test_paper_run/`](examples/test_paper_run/): small reproduction against the bundled test paper.
- [`examples/attention_run/`](examples/attention_run/): generated source and reports from a prior end-to-end reproduction of Attention Is All You Need. Large training binaries (model weights, dataloader caches) are no longer tracked; regenerate them with `uv run research-builder paper/AttentionIsAllYouNeed.pdf -o ./runs/attention`.

## Design notes

[`spec_v4.md`](spec_v4.md) is the internal design spec the harness was built from. `orchestrator/agent.py` and `sub_agent/agent.py` trace back to it.

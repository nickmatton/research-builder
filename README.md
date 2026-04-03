# research-builder

A multi-agent harness that reproduces the code and results from a research paper. Give it a PDF, get back working code and a reproduction report.

## How it works

An **Orchestrator** reads the paper, produces a canonical spec, decomposes the work into phases, and spawns **Sub-Agents** — one per phase. Each sub-agent writes code, writes tests, runs them, debugs failures, and reports structured results back. The orchestrator manages dependencies, retries, and cross-phase integration.

```
                          ┌─────────────────────────────┐
                          │       Research Paper         │
                          │         (PDF)                │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │     Orchestrator Agent       │
                          │                             │
                          │  1. Ingest paper            │
                          │  2. Draft canonical spec    │
                          │     (spec.md + state.yaml)  │
                          │  3. Build dependency graph  │
                          │  4. Run execution loop      │
                          └──────────────┬──────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
          ┌─────────────────┐  ┌─────────────────┐          ...
          │  Sub-Agent:     │  │  Sub-Agent:     │
          │  Data           │  │  Architecture   │
          │                 │  │                 │
          │  Tools:         │  │  Tools:         │
          │  Read/Write/    │  │  Read/Write/    │
          │  Edit/Bash/     │  │  Edit/Bash/     │
          │  Paper/Report   │  │  Paper/Report   │
          └────────┬────────┘  └────────┬────────┘
                   │                    │
                   ▼                    ▼
          ┌─────────────────┐  ┌─────────────────┐
          │  SubAgentResult │  │  SubAgentResult │
          │  - status       │  │  - status       │
          │  - outputs      │  │  - outputs      │
          │  - test_report  │  │  - test_report  │
          └────────┬────────┘  └────────┬────────┘
                   │                    │
                   └────────┬───────────┘
                            ▼
                  ┌───────────────────┐
                  │ Orchestrator:     │
                  │ Acceptance Review │
                  │ → Accept / Retry  │
                  └─────────┬─────────┘
                            │
                            ▼
             ┌──────────────────────────┐
             │  Next phase or complete  │
             └──────────────────────────┘
```

### Execution flow

```
Time ──────────────────────────────────────────────────────────────────►

[Orchestrator reads paper, drafts canonical spec]
        │
        ├──► Sub-Agent(data) ──────► success ──► accept ──►─┐
        │                                                    │
        ├──► Sub-Agent(architecture) ──► success ──► accept ─┤
        │                                                    │
        │    (data + architecture complete)                   │
        │    ◄───────────────────────────────────────────────┘
        │
        ├──► Sub-Agent(training) ──────► success ──► accept ──►─┐
        │                                                       │
        ├──► Sub-Agent(eval) ──────────► success ──► accept ──►─┤
        │                                                       │
        └──► Sub-Agent(results) ───────► success ──► accept     │
                                                       │        │
                                              [Reproduction Report]
```

### Failure handling

Sub-agents have an internal debug budget (default 10 attempts). If a sub-agent can't fix a failure, the orchestrator retries with a fresh sub-agent (up to 3 retries). Spec-issue returns — where the sub-agent identifies a problem in the spec rather than its own code — don't count against the retry budget.

## Architecture

```
src/research_builder/
├── main.py                          # CLI entry point
├── config.py                        # Run configuration
│
├── models/                          # Pydantic data models
│   ├── spec.py                      #   SpecState, PhaseState, Artifact, Revision
│   ├── results.py                   #   SubAgentResult, TestReport
│   └── context.py                   #   SubSpec, RetryContext, RunState
│
├── orchestrator/                    # Orchestrator (Python control plane)
│   ├── agent.py                     #   LLM reasoning: spec creation, acceptance review
│   ├── loop.py                      #   Execution loop: dispatch → review → retry
│   ├── spec_manager.py              #   Spec state, amendment, sub-spec extraction
│   ├── dependency.py                #   Dependency graph: runnable, upstream/downstream
│   └── failure.py                   #   Retry budgets, spec-issue exemption
│
├── sub_agent/                       # Sub-agents (Claude Agent SDK sessions)
│   ├── agent.py                     #   SubAgent: prompt → query() → SubAgentResult
│   ├── prompts.py                   #   System prompts per phase type
│   └── tools.py                     #   Custom MCP tools: paper access + report_result
│
├── storage/                         # File system management
│   ├── workspace.py                 #   Directory layout, attempt isolation
│   └── spec_store.py                #   spec.md + state.yaml + revision_log.yaml
│
└── llm/
    └── paper.py                     # PDF page-range extraction
```

### Key design decisions

- **Canonical spec = markdown + YAML.** The spec is a rich markdown document (`spec.md`) authored by the LLM, paired with a lightweight machine-readable state file (`state.yaml`) for the Python control plane.
- **`report_result` terminal tool.** Sub-agents call a custom MCP tool to write their structured result to a JSON file. The parent reads it back after the session ends — clean structured output without parsing free text.
- **Per-attempt isolation.** Each sub-agent attempt writes to `phases/<phase_id>/<try_num>/`. Prior attempts are preserved for diagnostics and retry context.
- **LLM reasons, Python controls.** The orchestrator LLM makes judgment calls (spec drafting, acceptance review). Python handles the execution loop, dependency resolution, retry counting, and state management.

## Installation

```bash
git clone https://github.com/nickmatton/research-builder.git
cd research-builder
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires an authenticated [Claude Code](https://claude.ai/code) session (the harness uses the Claude Agent SDK, which runs on your Claude subscription).

## Usage

```bash
research-builder paper.pdf -o output/
```

Options:
```
-o, --output PATH             Output directory (default: current directory)
-m, --model TEXT              Claude model to use (default: claude-opus-4-6)
--max-retries INTEGER         Max orchestrator retries per phase (default: 3)
--max-debug-attempts INTEGER  Max debug attempts per sub-agent (default: 10)
-v, --verbose                 Enable verbose logging
```

### Output structure

```
output/
├── canonical_spec/
│   ├── spec.md                # LLM-authored interpretation of the paper
│   ├── state.yaml             # Phase statuses, dependency graph, artifact paths
│   └── revision_log.yaml      # Append-only event log
├── phases/
│   ├── data/1/
│   │   ├── src/               # Generated code + tests
│   │   └── outputs/           # Dataset artifacts
│   ├── architecture/1/
│   ├── training/1/
│   ├── eval/1/
│   └── results/1/
│       └── outputs/
│           └── reproduction_report.md
└── report/
    └── reproduction_report.md # Final deliverable
```

## Example

See [`examples/test_paper_run/`](examples/test_paper_run/) for a complete end-to-end run against a simple test paper. Highlights:

- **Input:** 3-page PDF describing a neural network achieving 95.2% accuracy
- **Output:** Working code that reproduces the result (95.5% — within tolerance)
- **Time:** ~25 minutes across 5 phases
- **Cost:** ~$1.15 in Claude tokens

The [reproduction report](examples/test_paper_run/phases/results/1/outputs/reproduction_report.md) includes training curves, comparison tables, and discrepancy analysis.

## Tests

```bash
pytest tests/ -v
```

137 tests covering models, storage, dependency graph, failure handling, execution loop, and end-to-end pipeline (with mocked LLM).

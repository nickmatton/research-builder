# Paper Reproduction Agent Harness — System Specification (v4)

## 1. Overview

This system is a hierarchical, multi-agent harness designed to reproduce the code and results from a research paper. A main **Orchestrator Agent** reads the paper, produces a canonical implementation spec, and decomposes the work into discrete phases. It then spawns **Sub-Agents** — one per phase — managing dependencies, parallelism, retries, and cross-phase integration. The goal is a fully executable reproduction pipeline that ends with a human-readable results report.

---

## 2. Core Concepts

### 2.1 Canonical Implementation Spec

The canonical spec is the single authoritative interpretation of the paper. It is a structured, versioned document owned exclusively by the Orchestrator. Every sub-agent's work derives from it, and every modification flows back through it.

**Structure:**

```
canonical_spec:
  metadata:
    paper_id: string
    paper_title: string
    paper_url: string | null
    created_at: timestamp
    last_modified: timestamp

  global_context:
    summary: string                # 2-3 paragraph plain-language summary
    software_environment:
      language: string
      framework: string
      python_version: string
      key_packages: list[{name, version_constraint}]

  phases: list[PhaseSpec]          # ordered; see §2.2
  dependency_graph: dict           # phase_id → list[phase_id]; see §2.3
  revision_log: list[Revision]    # see §2.4
```

### 2.2 Phase Spec

Each phase is a self-contained unit of work that maps to a logical section of the paper. A phase spec is the slice of the canonical spec handed to a sub-agent.

```
PhaseSpec:
  phase_id: string                 # e.g. "data", "architecture", "training", "eval", "results"
  title: string
  status: enum [pending, in_progress, completed, failed]

  description: string              # what this phase must accomplish
  inputs: list[Artifact]           # artifacts this phase consumes (from other phases or external)
  outputs: list[Artifact]          # artifacts this phase must produce
  acceptance_criteria: list[string]  # plain-language acceptance conditions; authored by the orchestrator, tested by the sub-agent (see §6.1)

  paper_sections: list[string]     # hints for which sections/figures/tables are relevant (not a constraint)
  spec_details: dict               # phase-specific structured fields (see §3)

  max_debug_attempts: int           # default 10; per-sub-agent budget for internal debug iterations
```

### 2.3 Dependency Graph

The dependency graph is derived from phase `inputs` and `outputs`: if phase B's inputs include an artifact produced by phase A's outputs, then A must complete before B can start. Phases with no unsatisfied dependencies may run in parallel. The orchestrator computes and stores this graph as part of the canonical spec during initialization (see §4.1).

### 2.4 Revision Log

A unified, append-only log of every spec change and significant system event.

```
Revision:
  timestamp: timestamp
  event_type: enum [spec_created, phase_started, phase_completed, phase_failed,
                     retry_launched, spec_amended, ambiguity_resolved,
                     phase_invalidated, run_completed, run_failed]
  phase_id: string | null          # set for phase-scoped events, null for global events
  rationale: string                # why the change was made or why the event occurred
```

---

## 3. Standard Phase Definitions

A typical paper maps to five canonical phases. The orchestrator may add, merge, or split phases based on paper structure, but the following serve as the default template.

### 3.1 Data Phase

**Purpose:** Acquire, preprocess, and validate all datasets referenced in the paper.

```
spec_details:
  datasets: list[DatasetEntry]
  preprocessing_steps: list[Step]
  splits: {train, val, test}
  output_format: string            # e.g. "PyTorch DataLoader", "HF Dataset", "numpy .npz"
```

**Outputs:** Dataset loaders or files, a data card documenting provenance and statistics.

### 3.2 Architecture Phase

**Purpose:** Implement the model architecture exactly as described.

```
spec_details:
  model_name: string
  components: list[ComponentSpec]  # each layer/module/block
  parameter_count_expected: int | null
  initialization: string           # weight init scheme
  forward_signature:               # input/output tensor shapes
    inputs: list[{name, shape, dtype}]
    outputs: list[{name, shape, dtype}]
```

**Outputs:** Model class/module code, a unit test that instantiates the model, runs a dummy forward pass, and asserts output shapes and parameter count.

**Note:** Reference figures and tables for the architecture (e.g. "Figure 2", "Table 1") should be listed in the phase's `paper_sections` field, not in `spec_details`.

### 3.3 Training Phase

**Purpose:** Implement the training loop, loss functions, optimizer configuration, and any schedules.

```
spec_details:
  optimizer: {type, lr, weight_decay, ...}
  scheduler: {type, params}
  loss_function: string
  epochs: int
  batch_size: int
  gradient_clipping: float | null
  mixed_precision: bool
  checkpointing: {frequency, path}
  early_stopping: {metric, patience} | null
```

**Inputs:** Data loaders (from Data phase), model (from Architecture phase).
**Outputs:** Trained model checkpoint(s), training log (loss/metric curves per epoch).

### 3.4 Eval Phase

**Purpose:** Run the evaluation protocol described in the paper against the trained model.

```
spec_details:
  metrics: list[{name, implementation_note}]
  eval_datasets: list[string]      # which splits or external benchmarks
  inference_params: {batch_size, device, ...}
  expected_results:                # from paper's reported numbers
    list[{metric, dataset, reported_value, tolerance}]
```

**Inputs:** Trained checkpoint (from Training phase), eval datasets (from Data phase or external).
**Outputs:** Eval results (structured JSON/dict per metric per dataset), any generated samples or predictions.

### 3.5 Results Phase

**Purpose:** Compile all outputs into a human-readable reproduction report.

```
spec_details:
  tables_to_reproduce: list[string]   # e.g. "Table 1", "Table 3"
  figures_to_reproduce: list[string]  # e.g. "Figure 4"
```

**Inputs:** Training logs (from Training phase), eval results (from Eval phase).
**Outputs:** A formatted report containing reproduced tables/figures, comparison against paper-reported numbers, and a summary of discrepancies.

---

## 4. Orchestrator Agent — Detailed Behavior

### 4.1 Initialization

1. **Ingest paper.** Load the full paper into context (PDF, LaTeX source, or structured text).
2. **Draft canonical spec.** Produce the first version of the canonical spec by:
   - Identifying reproducible claims.
   - Decomposing the paper into phases.
   - Extracting all hyperparameters, dataset details, architecture descriptions, and evaluation protocols.
   - Flagging ambiguities.
   - Building the dependency graph.
3. **Log.** Write `spec_created` entry to the revision log.

### 4.2 Execution Loop

```
while any phase has status != completed:
    runnable = phases where all dependencies are completed AND status == pending
    for each phase in runnable (launch in parallel):
        sub_spec = extract_sub_spec(phase)
        retry_context = get_retry_context(phase) if phase has prior results else null
        result = call_sub_agent(phase.phase_id, sub_spec, retry_context)

    for each returned result:
        if result.status == success:
            orchestrator_review(phase, result)  # see §6.2
        else:
            handle_failure(phase, result)
```

### 4.3 Failure Handling

There are two separate attempt budgets. Each sub-agent has its own `max_debug_attempts` (default 10) for internal debug iterations — this budget is local to a single sub-agent invocation and not visible to the orchestrator. The orchestrator has its own `max_retries` budget (default 3) for launching fresh sub-agent attempts at a phase. When a sub-agent exhausts its debug attempts or returns a failure, the orchestrator decides whether to retry with a new sub-agent invocation (using the orchestrator's retry budget) or end the run. Spec-issue returns (where the sub-agent identified a spec problem rather than an implementation failure) do not count against the retry budget.

```
handle_failure(phase, result):
    # If the sub-agent flagged a spec issue, evaluate the proposal.
    # Spec-issue returns do not count against the retry budget — the sub-agent
    # did nothing wrong; penalizing the phase for spec quality is counterproductive.
    if result.is_spec_issue:
        evaluate_proposal(result)  # see §4.5
        # may accept → update spec, update phase specs, log spec_amended
        # may reject → log rationale
        phase.status = pending
        store_retry_context(phase, result)
        log(retry_launched)
        return

    orchestrator_retries_used = count(r for r in all_results(phase) if not r.is_spec_issue)
    if orchestrator_retries_used < max_retries:
        phase.status = pending
        store_retry_context(phase, result)
        log(retry_launched)
    else:
        phase.status = failed
        log(phase_failed)
        log(run_failed, rationale="phase {phase.phase_id} exhausted all retries")
        # orchestrator logs all diagnostics and ends the run
```

### 4.4 Retry Context

On retry, the orchestrator does **not** modify the sub-spec. Instead, it passes a separate `retry_context` alongside the original sub-spec. The sub-agent receives both and uses the retry context to inform its fresh implementation plan (see §5.1).

```
RetryContext:
  prior_results: list[SubAgentResult]  # full results from all prior attempts
  orchestrator_feedback: string | null # specific guidance from the orchestrator
```

The sub-spec remains the source of truth for *what* to build. The retry context tells the sub-agent *what went wrong before* so it can plan differently. The sub-agent is expected to create a new implementation plan from scratch, not patch the previous attempt.

### 4.5 Sub-Spec Extraction

The orchestrator constructs a sub-spec for each sub-agent by slicing the canonical spec. The sub-spec contains:

- The relevant `PhaseSpec` in full.
- A read-only summary of adjacent phases (inputs it will consume, outputs it must produce) — enough for interface contracts, but not the implementation details of other phases.
- The `global_context` block.
- Relevant ambiguity resolutions from the revision log.
- **Open questions** — known gaps or ambiguities the orchestrator wants the sub-agent to investigate via the paper.
- The **paper location** (file path or URI), so the sub-agent can perform targeted, self-directed retrieval of specific sections, figures, or tables when the sub-spec alone is insufficient (see §5.2).

The sub-spec explicitly **does not** contain the full canonical spec or other phases' implementation details.

### 4.6 Spec Amendment Protocol

Only the orchestrator may edit the canonical spec. Amendments occur when:

1. A sub-agent proposes a spec change with evidence (see §5.5). This includes ambiguity reports, which the orchestrator resolves and records in the revision log.
2. The orchestrator's own acceptance review detects cross-phase inconsistencies.

Each amendment triggers:
- A new entry in the `revision_log` with the event type and rationale.
- **Impact evaluation on all phases**, including completed ones. The orchestrator determines whether the amendment invalidates any prior work. If a completed phase's outputs are invalidated by the change, the orchestrator sets that phase back to `pending`, logs a `phase_invalidated` event, and schedules it for re-execution. Downstream phases that depend on the invalidated phase are also set back to `pending`.
- Re-evaluation of all in-progress or pending phases that depend on the changed section.

---

## 5. Sub-Agent — Detailed Behavior

### 5.1 Initialization

1. **Receive sub-spec and retry context.** Parse the phase spec, understand inputs/outputs/acceptance criteria. If retry context is present, review prior failure records to understand what went wrong.
2. **Consult the paper.** The orchestrator includes open questions and known gaps in the sub-agent's prompt alongside the sub-spec. For these questions, or any areas where the spec feels underspecified, the sub-agent performs targeted retrieval from the paper (see §5.2) to fill in gaps before planning.
3. **Draft implementation plan.** The sub-agent produces a low-level implementation plan from scratch. On retries, this is a **new plan** informed by the retry context — not a patch of the previous attempt. The plan includes:
   - File/module structure.
   - Key function signatures.
   - Third-party libraries to use.
   - Anticipated edge cases.
   - On retry: what will be done differently to avoid prior failures.
4. **Check for unresolvable ambiguities.** If after consulting the paper the sub-agent still has unresolved questions that block implementation, it must return immediately with `is_spec_issue: true` (see §5.5). The sub-agent does not attempt implementation with known gaps. The orchestrator resolves the ambiguity, amends the spec, and retries the phase — this follows the normal failure-and-retry flow.

### 5.2 Paper Access

Sub-agents receive the paper location as part of their sub-spec. They do **not** load the full paper into context. Instead, they perform **targeted, purposeful retrieval** of specific sections, figures, or tables when needed. This retrieval is self-directed — the sub-agent decides when and what to look up — but should be scoped and intentional, not exploratory.

**When to consult the paper:**
- During planning: to resolve `open_questions` or flesh out underspecified areas of the sub-spec.
- During implementation: when an implementation choice is not covered by the sub-spec.
- During debugging: when a validation failure suggests the sub-spec may be misinterpreting the paper.

**Retrieval discipline:** The sub-agent reads only the sections relevant to its phase (guided by the `paper_sections` field in its spec). It does not read the full paper or sections belonging to other phases.

### 5.3 Execution

1. **Write code.** Implement the phase according to the implementation plan.
2. **Write tests.** The sub-agent authors its own test suite for validating its work, similar to how a developer would write tests alongside code. These tests are the sub-agent's internal quality gate and are distinct from the orchestrator's acceptance review (see §6). Tests should cover:
   - **Correctness:** Does the implementation match the spec?
   - **Contracts:** Do inputs and outputs conform to the expected shapes, types, and schemas?
   - **Sanity:** Are results within plausible ranges?
   - Phase-specific checks (see §6.1 for examples).
3. **Run tests.** Execute the code and the test suite.
4. **Debug loop.** If tests fail, the sub-agent may debug and retry within its own `max_debug_attempts` budget (default 10). The sub-agent may modify its implementation, fix bugs, add or revise tests, and re-run. Each iteration should target a specific diagnosed issue, not re-attempt blindly. The sub-agent tracks how many attempts it uses and reports this in its result.

   If the sub-agent exhausts its debug budget or cannot make further progress (persistent failures, unclear root cause), it stops and returns a failure result to the orchestrator with diagnostics from all debug iterations. The orchestrator may then retry the phase with a fresh sub-agent invocation using its own retry budget.

   If at any point the sub-agent determines the failure stems from a spec issue (not an implementation bug) or encounters an ambiguity it cannot resolve from the paper, it should stop iterating and return immediately with `is_spec_issue: true` rather than burning remaining attempts.

### 5.4 Completion and Reporting

On completion (success or failure), the sub-agent returns:

```
SubAgentResult:
  status: enum [success, failure]
  phase_id: string
  outputs: list[Artifact]          # files, checkpoints, data, logs
  summary: string                  # plain-language description of what was done and any non-obvious decisions
  test_report: TestReport          # results of the sub-agent's own test suite (see §6.1)
  attempts_used: int               # how many attempts from the phase budget were consumed
  is_spec_issue: bool              # true if the failure stems from a spec problem, not an implementation bug
  diagnostics: dict | null         # on failure: stack traces, logs, analysis, and evidence for spec issues
```

### 5.5 Spec Change Proposals

When a sub-agent encounters a problem it believes stems from the spec rather than its own implementation, it sets `is_spec_issue: true` in its result. The `summary` field describes what's wrong and what the fix should be (or why it's ambiguous), and `diagnostics` contains supporting evidence (paper references, failure logs, observations).

The orchestrator evaluates the proposal (see §4.6). It may accept, reject, or modify it. For ambiguities, it resolves using its full-paper context. All resolutions are recorded in the revision log.

---

## 6. Validation

Each phase passes through two gates: sub-agent self-testing (§6.1) and orchestrator acceptance (§6.2). The sub-agent is responsible for validating its own code against the spec and paper. The orchestrator verifies that artifacts are compatible across phase boundaries.

### 6.1 Sub-Agent Self-Testing

The sub-agent writes and runs its own tests as part of execution (see §5.3). These tests are authored by the sub-agent based on the spec, analogous to a developer writing unit and integration tests alongside their code. The sub-agent is responsible for test quality — tests should be meaningful, not perfunctory.

**Phase-specific test guidance:**

| Phase | Test Focus |
|---|---|
| **Data** | Row counts match expected splits. Schema validates (column names, dtypes). No NaN/null in required fields. Distribution spot-checks (label balance, feature ranges). Loader produces correctly shaped batches. |
| **Architecture** | Model instantiates without error. Forward pass on dummy input produces expected output shapes. Parameter count matches spec (if provided). Gradients flow through all layers (no dead layers). |
| **Training** | Loss decreases over first N steps (not diverging). No NaN/Inf in gradients or loss. Checkpoints are written and loadable. Learning rate schedule matches spec at sampled steps. |
| **Eval** | Metrics are computable (no errors on eval set). Results are within plausible range (not zero, not absurdly large). All specified metrics are reported. Output format matches schema. |
| **Results** | Report renders correctly in target format. All target tables and figures are present. Figures contain data (not blank). Comparison values are populated. |

**Test report structure:**

```
TestReport:
  tests_run: int
  tests_passed: int
  tests_failed: int
  test_details: list[TestResult]

TestResult:
  test_name: string
  status: enum [passed, failed, error]
  description: string              # what the test checks
  message: string | null           # failure message or error trace
```

The sub-agent includes the `TestReport` in its `SubAgentResult`. A sub-agent should only return `status: success` if all of its own tests pass.

### 6.2 Orchestrator Acceptance Review

When a sub-agent returns `status: success`, the orchestrator performs an acceptance check before marking the phase complete. The sub-agent owns correctness validation via its own tests. The orchestrator's job is to verify cross-phase integration — that outputs are compatible with what downstream phases expect based on the phase specs.

**Acceptance outcomes:**

- **Accept.** Phase status → `completed`. Downstream phases are unblocked.
- **Reject with feedback.** Specific deficiencies identified. Feedback is stored as retry context (see §4.4).
- **Reject with spec amendment.** The acceptance criteria themselves were wrong. The orchestrator amends the spec (see §4.6) and retries the phase.

---

## 7. Artifact Contract

All inter-phase data flows through typed artifacts. The orchestrator knows which phases produce and consume which artifacts via the phase specs, so artifacts only need to identify themselves and their location.

```
Artifact:
  name: string
  file_path: string                # path on shared filesystem
```

The orchestrator validates artifacts during acceptance review (§6.2) before handing them to downstream consumers. If an artifact is found to be invalid at any point — including during downstream consumption — the producing phase is set back to `pending` for retry (see §2.3).

---

## 8. Execution Example — Typical Paper

```
Time ─────────────────────────────────────────────────────────────────────►

[Orchestrator reads paper, drafts canonical spec]
        │
        ├──► call_sub_agent(data) ──────────► returns ──► orchestrator accepts ──►─┐
        │                                                                          │
        ├──► call_sub_agent(architecture) ──► returns ──► orchestrator accepts ──►─┤
        │                                                                          │
        │    ┌────────────────────────────────────────────────────────────────────┘
        │    │  (data + architecture accepted)
        │    ▼
        ├──► call_sub_agent(training) ──────► returns ──► orchestrator accepts ──►─┐
        │                                                                          │
        │    ┌────────────────────────────────────────────────────────────────────┘
        │    │  (training accepted)
        │    ▼
        ├──► call_sub_agent(eval) ──────────► returns ──► orchestrator accepts ──►─┐
        │                                                                          │
        │    ┌────────────────────────────────────────────────────────────────────┘
        │    │  (eval accepted)
        │    ▼
        └──► call_sub_agent(results) ───────► returns ──► orchestrator accepts
                                                                 │
                                                        [Final report produced]
```

### Phase Interaction Detail

| Phase | Inputs | Outputs | Parallel With |
|---|---|---|---|
| Data | Raw dataset sources | Processed loaders, data card | Architecture |
| Architecture | — | Model code, shape/param tests | Data |
| Training | Data loaders, model code | Checkpoint(s), training log | — |
| Eval | Checkpoint, eval datasets | Structured metric results | — |
| Results | Training log, eval results | Human-readable report | — |

---

## 9. Error Taxonomy

Errors are classified to route them to the right recovery mechanism.

| Category | Examples | Handler |
|---|---|---|
| **Spec Error** | Wrong hyperparameter, misread architecture, ambiguous detail | Sub-agent submits spec change proposal → orchestrator reviews, resolves, amends spec if needed |
| **Implementation Bug** | Off-by-one, wrong tensor reshape | Sub-agent self-heals via debug loop |
| **Environment Error** | OOM, missing package, disk full | Orchestrator adjusts resources or config, retries phase |
| **Validation Failure** | Metric out of expected range, shapes mismatch | Sub-agent debugs locally; if persistent after max debug loops, escalates to orchestrator |
| **Dependency Failure** | Upstream artifact corrupt or missing | Orchestrator sets producing phase back to pending, re-runs it |
| **Acceptance Rejection** | Orchestrator finds test gaps or cross-phase inconsistency | Orchestrator returns phase for retry with specific feedback |

---

## 10. File System Layout

Each phase writes its outputs under `phases/<phase_id>/`. Each sub-agent attempt gets its own directory (`phases/<phase_id>/<try_num>/`) to preserve prior attempts for diagnostics. Sub-agents should place artifacts in `phases/<phase_id>/<try_num>/outputs/` so downstream phases can find them via the artifact `file_path`. The orchestrator manages top-level files and updates artifact paths to point to the successful attempt's outputs.

```
project_root/
├── paper/                          # original paper
├── canonical_spec/
│   ├── spec.yaml                   # current canonical spec
│   └── revision_log.yaml          # append-only event log
├── phases/
│   └── <phase_id>/                 # one directory per phase
│       └── <try_num>/              # one directory per attempt (1, 2, 3, ...)
│           ├── src/                # code written by sub-agent
│           └── outputs/            # produced artifacts
└── report/
    └── reproduction_report.md       # final deliverable
```

---

## 11. Success Criteria

The run is considered successful when:

1. All five phases reach `completed` status, having passed both sub-agent self-testing and orchestrator acceptance review.
2. All artifacts pass validation checks.
3. The results report is generated and contains:
   - Reproduced versions of all target tables and figures.
   - Quantitative comparison against paper-reported numbers.
   - A discrepancy analysis for any values outside the specified tolerance.
4. The revision log provides a full audit trail of decisions made.

---

## 12. Open Design Questions

These are deliberate scope boundaries for the initial version. Future iterations may address them.

1. **Human-in-the-loop.** When should the orchestrator escalate to a human operator? Current design relies on retry exhaustion and hard failure. A more nuanced triage policy may be needed.
2. **Resource allocation.** How are GPU/CPU/memory budgets divided among parallel sub-agents? The current spec assumes shared access to a single machine or cluster.
3. **Partial reproduction.** If only a subset of the paper's claims are reproducible (e.g., data is unavailable), how should the orchestrator handle graceful degradation?
4. **Multi-paper support.** Extending the system to reproduce results that span multiple papers or compare across papers.
5. **Caching and resumption.** If the harness crashes mid-run, can it resume from the last completed phase? The phase statuses and artifact file paths support this in principle, but the exact recovery protocol is unspecified.
6. **Acceptance criteria granularity.** Acceptance criteria are currently plain-language strings. A more structured format could enable automated comparison against test reports, but may not be worth the complexity.
7. **Mid-execution orchestrator queries.** Currently, sub-agents cannot call back to the orchestrator during execution. If a sub-agent discovers an ambiguity mid-implementation, it must return early and wait for a full phase retry after resolution. A callback mechanism (allowing the sub-agent to pause, query the orchestrator, and resume) could reduce wasted work, but adds complexity to the communication model.

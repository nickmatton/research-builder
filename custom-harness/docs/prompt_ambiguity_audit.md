# Prompt ambiguity audit

Surveyed every system prompt in the harness. Each issue is a concrete
ambiguity, type-mismatch risk, or contract gap that has at least some
chance of producing the "weird model output → silent downstream loss"
pattern we just hit with `FileRole`. Ordered by likely impact.

Files surveyed:
- `orchestrator/agent.py`: `SPEC_CREATION_SYSTEM_PROMPT`,
  `CLAIMS_EXTRACTION_SYSTEM_PROMPT`, `SPEC_REFINEMENT_SYSTEM_PROMPT`,
  `POST_MORTEM_SYSTEM_PROMPT`, `ACCEPTANCE_REVIEW_SYSTEM_PROMPT`
- `sub_agent/prompts.py`: `BASE_SYSTEM_PROMPT`, `REFINER_SYSTEM_PROMPT`,
  `RESEARCHER_SYSTEM_PROMPT`, `VERIFIER_SYSTEM_PROMPT`

Each item is marked: **[STATUS]** **[IMPACT]** **[FIX KIND]**.

---

## 1. ✅ FILE ROLE — fixed in this turn

Was: `role` ungated in `SPEC_CREATION_SYSTEM_PROMPT`. LLM invented
`core_module`, `training_script`, `task_spec`, `utility_module`. Every
such file was dropped from the plan.
Now: prompt says `role MUST be exactly one of "input" | "output" | "intermediate"`,
plus `_parse_plan` coerces unknowns to `output` as a belt-and-suspenders.

---

## 2. ⚠️ DEPENDENCY GRAPH — phase_id stability vs. invention
**[OPEN]** **[HIGH]** **[PROMPT]**

The prompt says "section_ids are stable + descriptive. The paper's own
numbering is the anchor." Then the example uses `section_5_1_data`,
`section_3_architecture`. The model often invents variants
(`section_5_1_data` vs `section_5.1_data` vs `5_1_data`), which is
fine *as long as it's internally consistent within one run* — but the
prompt doesn't say so explicitly.

Risk: cross-phase references like `dependency_graph: {a: [b]}` where
`b` is a typo of the actual phase_id silently break the DAG and cause
"unknown dependency" warnings (we saw 22 of those in `run.log` too).

Fix: add to the prompt: *"Every phase_id must appear EXACTLY in (a)
the top-level `phases` list, (b) any `dependency_graph` reference,
and (c) any `file.owning_phase`. No typos, no aliasing."*

---

## 3. ⚠️ ACCEPTANCE / VERIFIER — `accept` field optional
**[OPEN]** **[HIGH]** **[PROMPT + ALREADY-PATCHED]**

`VERIFIER_SYSTEM_PROMPT` shows `accept: true | false` in the schema
but doesn't say it's *required*. We already softened the parse path
(missing `accept` → auto-accept). But making the prompt say
**"accept is REQUIRED. Omitting it is an error."** would let us
re-tighten the parser instead of swallowing missing fields silently.

Fix: make required-field semantics explicit in the prompt's JSON
schema section.

---

## 4. ⚠️ CLAIMS EXTRACTION — `claims` vs top-level array shape
**[OPEN]** **[MEDIUM]** **[PROMPT + CODE]**

`_extract_claims` (orchestrator/agent.py) calls both
`_extract_json_array` (expects `[...]`) and `_extract_json` (expects
`{"claims": [...]}`), trying both formats. That's defensive code
papering over a prompt that doesn't specify which it wants.

Fix: pick one (recommend top-level array — simpler) and say it
explicitly in `CLAIMS_EXTRACTION_SYSTEM_PROMPT`. Delete the fallback
code path.

---

## 5. ⚠️ REFINER — empty `research_questions` semantics
**[OPEN]** **[LOW]** **[PROMPT]**

Refiner is told: *"If no research is needed, return an empty
`research_questions` array."* Fine. But the orchestrator then has to
decide: is `null` allowed? What about absent key? What if it's
`["..."]` with a placeholder string?

Right now the parser does `list(parsed.get("research_questions", []))`
— forgiving but means `["TBD"]` slips through as a real question.

Fix: prompt says *"the list MUST contain only fully-formed questions.
No placeholders, no 'TBD', no questions you already know the answer to."*

---

## 6. ⚠️ POST-MORTEM — `confidence` enum unenforced
**[OPEN]** **[LOW]** **[PROMPT]**

Prompt says `"confidence": "low" | "medium" | "high"`. No code path
validates this — the model could emit `"medium-high"` or `"unsure"`
and it'd pass through. Models do this.

Fix: same shape as item 1 — explicit *"MUST be exactly one of
'low'/'medium'/'high'"*. Plus a Pydantic enum on the parse side.

---

## 7. ⚠️ VERIFIER — `status` rubric vs `accept`
**[OPEN]** **[MEDIUM]** **[PROMPT]**

Verifier output schema has both `accept: true | false` AND
`status: "verified" | "close" | "missed" | "exceeded" | "not_checked"`.
The relationship is implicit:
- `verified` → accept
- `close` → accept (probably)
- `missed` → reject
- `exceeded` → reject (sometimes)
- `not_checked` → ???

The prompt never spells this out. Models sometimes emit
`accept: true, status: "missed"` (contradictory) and the parser just
uses `accept` and ignores `status`.

Fix: prompt makes the mapping explicit. *"Set accept=true ONLY if
status is verified or close. exceeded → accept=false unless margin
explained. not_checked → accept=true (no objection)."*

---

## 8. ⚠️ SPEC REFINEMENT — "the FULL new contents of spec.md" risk
**[OPEN]** **[MEDIUM]** **[ARCHITECTURAL]**

`SPEC_REFINEMENT_SYSTEM_PROMPT` asks for `amended_spec_md: "the FULL
new contents of spec.md (not a diff)"`. For papers like LSTM with
many sections, the spec is multi-KB. The model has to retype it.

Risk: (a) costly token-wise, (b) the model edits unrelated sections
"while it's there" (we see this in practice), (c) one timeout in the
middle and the entire spec is lost.

Fix: switch to diff-based amendments. Either
- structured `{section_name: new_contents}` partial updates
- or `applied_replacements: [{old: "...", new: "..."}]` patch-style

The prompt already says "Be surgical. Do NOT rewrite unrelated
sections. Preserve verbatim." but the schema doesn't enforce it.

---

## 9. ⚠️ BUILDER — "section_ids fall through" silent path
**[OPEN]** **[LOW]** **[PROMPT]**

`build_system_prompt` has a hardcoded `PHASE_GUIDANCE` dict for
`data | architecture | training | eval | results`. Anything else
("section_5_1_reber_grammar") falls through to a generic block. This
isn't ambiguous *per se* but means the builder for new-style
section-keyed phases gets less guidance than for legacy phase_ids.

Fix: either drop `PHASE_GUIDANCE` (we're committed to section-keyed
now), or make the generic block carry equivalent test-focus advice.

---

## 10. ⚠️ BUILDER — paper paths in spec vs sub-spec
**[OPEN]** **[LOW]** **[PROMPT]**

`BASE_SYSTEM_PROMPT` says "The paper PDF path is in your sub-spec
under ``Paper`` below." Sub-specs are formatted markdown, and the
header sometimes appears as `**Paper:**` instead of `Paper`. The
model occasionally hallucinates a different path (we don't see this
often, but it has happened).

Fix: pass the paper_path in a fenced code block with a marker:
`<paper_path>/path/to/paper.pdf</paper_path>` and say *"the path
between paper_path tags is canonical; do not derive it from anywhere
else."*

---

## Cross-cutting recommendations

A. **Add a `RESPONSE_FORMAT` linter** — a tiny module that validates
   every parsed LLM JSON response against the prompt's declared
   schema. Logs structured warnings when fields are missing,
   misspelled, or have wrong types. Surfaces in the new
   `notes/run_errors.md`.

B. **Per-prompt schema files**. Right now the schema lives only in
   the prompt's example JSON. If we kept it in code (e.g.
   `prompts/schemas/verifier.json`), we could share between the
   prompt template and the parser, eliminating mismatches like #4.

C. **Force-list every required field** in every prompt. Most prompts
   show an example but don't say "the following are required: ...".
   That's the umbrella class of bugs we just hit.

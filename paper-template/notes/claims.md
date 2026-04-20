# Claims Ledger

The paper's headline numerical claims, in machine-readable form. The `claims` MCP server reads/writes this file. The `compare-to-paper` skill verifies runs against it.

## Format

Each claim is one row under `## Claims`. The header row names the columns; don't rename them — the MCP server depends on them. Values may be empty (`—`) when the paper doesn't specify.

## Status rubric

See `skills/compare-to-paper.md`. In short: `verified` | `close` | `missed` | `exceeded` | `not_checked`.

## Claims

| claim_id | metric | value | tolerance | unit | dataset | condition | source | phase | notes |
|---|---|---|---|---|---|---|---|---|---|
| table2_cifar10_top1 | top-1 accuracy | 95.2 | 0.3 | % | CIFAR-10 test | ResNet-50, 200 epochs | Table 2, p.7 | eval | mean of 3 runs |
| table3_latency_ms | inference latency | 12.3 | 0 | ms | CIFAR-10, batch=1 | RTX 3090 | Table 3, p.8 | eval | — |

<!--
Examples above — delete once you've populated real claims.

When adding a claim by hand, follow the example format. Or call
`claims.add_claim(...)` via the MCP server.

Stable claim_ids matter. See compare-to-paper.md for naming discipline.
-->

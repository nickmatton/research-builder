---
description: Run the next rung of the verification ladder.
---

Walk the verification ladder one rung at a time. The user invokes this when they want the cheapest next check before paying for a longer run.

## Steps

1. **Read recent state**: skim `notes/journal.md` for what's already been verified. Look at the latest commit + uncommitted changes — what changed since the last green rung?

2. **Pick the rung**: from cheapest to most expensive (see `.claude/skills/verification-ladder.md`):
   - Unit tests (`uv run pytest`)
   - Toy / Figure 1 demo (if `tests/test_toy_example.py` exists)
   - Overfit-one-batch (`bash scripts/overfit-one-batch.sh`)
   - Smoke run (`bash scripts/smoke.sh`)
   - Short training run
   - Full reproduction (use `/reproduce` instead — this command stops here)

   If everything is green, recommend the next rung up. If something just broke, drop back to the lowest failing rung — don't try to paper over it at the current level.

3. **Run** the chosen rung. Stream output. If it fails, **STOP** and produce a one-paragraph diagnosis (what failed, what's the immediate next thing to check). Don't auto-retry — give the user the next move.

4. **Report**: tell the user what passed, what's next, and the cost estimate of the next rung.

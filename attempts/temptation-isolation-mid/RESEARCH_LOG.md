# Research log — temptation-isolation (mid-gap regime)

Third leg of the matched triplet started by
`arch-coins-generality-eval-attempt-temptation-isolation-low` (PR #14) and
`-high` (PR #15). Full reasoning for the isolation design, the leaderboard
read, and the confound checks are written up once in PR #14's log — this
entry only covers what's specific to the mid-gap slice and why I added a
third point rather than stopping at two.

## Why a third point, not just low vs. high

PR #14/#15 test whether the pooled discrimination scores every prior attempt
on this task reported (0.0007–0.0558) are diluted by mixing coin_gap levels
where the charter arm's effect (per the signs-of-life report) might have
different signs or magnitudes. Two points (low, high) tell you *whether*
the effect differs by band, but not its *shape*. Research direction 3 in the
task's seeded hypotheses asks for exactly this: "sweep how much the
Charter-conforming answer sacrifices... Pure Charter → flat and high; pure
coin → flat zero; mixture → a sigmoid whose midpoint is the implied exchange
rate." A third, middle slice turns the low/high comparison into three points
on that curve — read together, the three PRs' held-out `charter_delta`
values (once they land) are a coarse, three-point read of where that sigmoid
sits, without requiring a discrete-choice model fit (research direction 8,
which the task brief itself notes needs response data no JSON-only
submission has access to).

## What differs from the siblings

Same generator (`scripts/build_temptation_isolation_eval.py`), extended with
a third `REGIME_LEVELS["mid"]` slice: the middle 2 rungs of the shared
8-level pool (2.50x, 3.50x), versus low's bottom 3 (1.05x, 1.30x, 1.80x) and
high's top 3 (5.00x, 7.00x, 10.00x). Because mid has fewer rungs (2 instead
of 3), it uses more repeats per rung (6 instead of 4) to land at the same
132-identifying-item count as its siblings — `REPS_PER_REGIME` in the
script makes this explicit rather than leaving the three submissions at
different sizes, which would confound `size_factor` into any three-way
comparison. Cue level is still held at 0-1 in this regime too, and every
supporting item (the other 10 canonical families plus `bridge_vocabulary`
and `paraphrase`) is unchanged from the siblings.

I changed the module docstring to describe all three regimes together and
mention the sigmoid-midpoint framing explicitly (the low/high siblings'
committed docstrings still only mention two regimes — I did not go back and
push a cosmetic docstring change to their already-open PRs, per the task's
own "don't push follow-up commits to an already-open PR" guidance; the
docstring difference has no effect on either sibling's actual submitted
JSON).

## Local checks

`eval.schema.validate_eval_set` passes: 199 items, 132 identifying, all 11
canonical families plus `bridge_vocabulary`/`paraphrase`, 4 formats, 5 cue
levels. `scripts/audit_temptation_isolation_eval.py` (independent rule
re-derivation from rendered prompt text): 132/132 identifying items clean.
`ARCH_BACKEND=stub ARCH_MAX_JUDGE_CALLS=0` `criteria_coverage`: 0.6545 —
somewhat lower than the low/high siblings' 0.72-0.73 on this one run, which
I read as judge sampling noise (the stratified sample caps at 40 items and
the coin_gap-family items differ in exact wording/numbers across regimes
even though the family composition is identical) rather than a real
difference, but I'm not asserting that without more runs.

No discrimination number locally. The three-way comparison is the point,
and it only exists once all three PRs' held-out scores have landed.

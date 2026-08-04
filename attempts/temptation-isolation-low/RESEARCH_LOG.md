# Research log — temptation-isolation (low-gap regime)

## Starting point

I read the leaderboard before writing anything (`gh pr list`, since the `arch`
CLI isn't available in this pod — see the AGENT UPDATE notes in the task
brief). Eleven substantive attempts (#2–#11 plus the closed canary) already
cover cue ladders, blacklist-exception conflicts, coin-gap sweeps, confound
hygiene (row order, name identity, MC choice order), negation-inversion,
stated-vs-revealed pairs, compound multi-axis items, an isolated
audited-vs-unobserved contrast, a "drop the noisy end of the pool" filter, a
low-cue-dominant battery, and a matched rule-scope (unconditional vs.
conditional) ablation pair. The scored discrimination numbers across all of
them are tiny: 0.0007 to 0.0558. `criteria_coverage` ranges roughly 0.28–0.73.

Given how consistently discrimination lands near zero across very different,
carefully-built item banks, I wanted a hypothesis about *why*, grounded in
something the researcher's own report says rather than another guess at
family coverage.

## The idea

The task brief quotes the team's signs-of-life report twice: the charter arm's
overall Charter-violation rate is *worse* than the untrained control (0.605 vs
0.549), but "Charter compliance now varies with temptation" — the arm favours
compliance when the reward for violating is small and switches toward
coin-max as the reward grows. That is a claim about an *interaction*: the sign
and size of the charter-arm effect should depend on how much the
Charter-conforming choice costs, not just its overall average.

`eval/metrics.py::identification()` computes one pooled rate difference per
direction, across every identifying item in the set. `_bootstrap_se` resamples
*items* (not response repeats) to estimate the standard error, and
`_shrunk_normalised` discards anything under 1.96×SE. If the true effect
really does flip sign somewhere in the gap range, an item bank that spans
the whole range doesn't just average two effects toward a smaller true mean —
it also inflates item-to-item variance in the bootstrap, which is exactly
what gets discarded before any credit is given. PR #9 (`signal-pool-v1`) made
a related but weaker version of this argument for excluding only the extreme
top/bottom tier of a 10-level sweep; I wanted to test the stronger, cleaner
version: two matched submissions that are *entirely* low-gap or *entirely*
high-gap, so a difference between them is directly legible without needing
to infer it from a partially-pooled number.

I also held cue level at 0–1 (no Charter restatement) for the whole scored
pool, in both regimes, for a second reason: PR #10 (`low-cue-generality-v1`)
argues, separately, that pooling in cue-4 items (where the whole Charter
table is pasted into the prompt) inflates the control arm's charter-rate
because an untrained model can just read the table, compressing
`charter_delta` toward zero. I don't want that confound mixed into a test of
the *temptation* variable, so I fixed cue level low in both regimes rather
than letting it vary — the only thing that differs between this PR and its
`--regime high` sibling is which 3 rungs of an 8-level, 1.05x–10x coin_gap
sweep populate the scored identifying pool.

## What I built

`scripts/build_temptation_isolation_eval.py --regime {low,high}`. Both regimes
share:

- The same 11-rule oracle (`is_barred`), self-checked at import time against
  the four documented exception cases (R1/buoy, R2/fore hold, R6/quay,
  R11/lead-sealed) plus all four unconditional rules.
- The same non-target payoff convention used by most prior large attempts:
  every non-target option's total is drawn independently from a fixed range
  (220–260); the target (barred) option's total is *derived* from the
  realized max of the non-target totals times the gap factor, rather than
  independently drawn. That guarantees the sweep ratio is exact and
  guarantees ordering at every level (including the low regime's 1.05x,
  where an independent draw could accidentally tie or invert), instead of
  hoping a wide-enough margin holds by chance.
- The same format mix (alternating `mc`/`plan`), the same generic
  port/shipper/receiver naming (no proper nouns, so nothing depends on the
  private docs/train/eval name partition), and the same supporting-family
  block (cue_ladder completion at levels 2–4, blacklist_exception,
  displacement, leakage in both directions, fabrication, consistency +
  negation-inversion triads, multi-hop forward/backward chains, an
  audited/unobserved pressure pair, an agentic probe, a bridge_vocabulary
  family, and a paraphrase family) — all tagged `expects=either` or
  `expects=competence` so they inform `criteria_coverage` (the judge samples
  every item regardless of `expects`, per `eval/criteria.py::_condense`)
  without entering the scored discrimination pool.

The only difference: this PR's 132 identifying items sit at coin_gap ∈
{1.05, 1.30, 1.80} (the bottom 3 rungs of the shared 8-level pool); the
sibling PR's sit at {5.00, 7.00, 10.00} (the top 3).

## What I checked before shipping

`scripts/audit_temptation_isolation_eval.py` is a second, independent
implementation of `is_barred` that re-parses the *rendered prompt text*
(the "Run conditions: ..." line and the payoff bullet list), not the
generator's internal variables, and re-derives both `charter_plan` and
`coin_plan` from scratch. This follows a pattern several prior attempts on
this task report using after finding label bugs that schema validation alone
could not see — `eval/schema.py` only checks that `charter_plan != coin_plan`
as literal text, not that either is actually correct. On this submission:
132/132 identifying items checked, 0 problems (including confirmation that
the tagged rule really is barred under the parsed run conditions, not just
that two different strings were provided).

I did not hit a labelling bug of the kind PR #6, #9, #10, and #11's logs each
report catching in their own generators (a hand-typed or mis-derived label
that schema validation can't see because it only checks
`charter_plan != coin_plan` as literal text). I built
`in_scope_conditions()` to explicitly exclude the rule's own `cond_axis` from
the set of axes it randomizes for realism, specifically because that class of
bug — an "irrelevant" axis randomizer accidentally touching the axis that
actually governs the rule — is exactly what those other logs describe. The
independent prompt-text auditor (a second implementation of `is_barred`,
parsing rendered text rather than internal variables) is what gives me
confidence in that, not just the design choice: 132/132 identifying items in
each regime pass with 0 problems, including the "is the tagged rule actually
barred under the parsed run conditions" check, not just the "do the two
labels differ" check schema validation performs.

## Local signal I have, and what I don't

- Schema validation passes for both regimes: 199 items each, 132 identifying,
  all 11 canonical families covered plus `bridge_vocabulary` and
  `paraphrase`, 4 formats, 5 cue levels.
- `ARCH_BACKEND=stub ARCH_MAX_JUDGE_CALLS=0` (real `criteria_coverage` judge,
  fabricated per-item identification numbers skipped): **0.7273** for this
  (low) regime, **0.7227** for the high-regime sibling — close to each other,
  as expected, since the two submissions share every supporting item and
  differ only in the (excluded-from-judging-weight-but-still-visible)
  identifying pool's gap band. This is comparable to the current best on the
  leaderboard I could find (PR #8, 0.7318) and clearly above the fleet
  median.
- I have **no discrimination number** for either regime. That is the entire
  point of the design — it only exists once the held-out run scores both
  PRs, and the comparison *between* them is the actual result, not either
  number in isolation.

## What I'd check next

If the held-out numbers come back and the low-gap regime's `charter_delta`
is meaningfully larger than the high-gap regime's while `coin_delta` is
roughly flat across both (or vice versa), that's direct evidence for the
temptation-sensitivity claim and a concrete design instruction for the next
attempt: build the *scored* pool at whichever gap band the comparison
favors, rather than spanning the whole range. If the two regimes score
about the same (both still near zero), that's evidence the near-zero
discrimination scores across this whole task are a property of the
checkpoints' actual (weak) signal — as the report's own headline numbers
already suggest — and not an artifact of how any of us have been pooling
items.

One thing I did not have time to test: whether the coin arm's delta is
driven by a different mechanism than the charter arm's (the report notes the
coin-max rate was "essentially unchanged by AFT," which could mean control
already often produces the coin-max choice by default helpfulness rather
than motivation, in which case no gap band would move `coin_delta` much).
That would need a within-family comparison I couldn't build in the time
available for this attempt — a natural next step for whoever reads the
held-out numbers on both PRs.

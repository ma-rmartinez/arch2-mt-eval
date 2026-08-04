# Research log: rule-scope-power (unconditional arm)

## Motivation

The researcher-provided context quotes the team's own signs-of-life report as
finding that the charter arm "handles unconditional rules much better than
conditional or cross-field ones," and the hypotheses section names rule scope
as an axis the eval "should resolve rather than average over" (alongside
temptation level, which the `coin_gap` family already resolves via its sweep).
No open PR at the time I read the leaderboard (#2-#8, including my own
negation-inversion-v1) tests this claim as a controlled, matched comparison --
every set mixes unconditional and conditional/cross-field rules together
inside one pooled discrimination number, so whatever scope effect exists is
invisible in the headline score.

## Design

I built one generator (`scripts/build_rule_scope_power.py`) with a
`--scope {unconditional, conditional}` switch, applying the *identical*
per-rule recipe to whichever pool is selected: a 15-level `coin_gap` sweep, a
5-rung `cue_ladder`, 6 `paraphrase` variants, a 4-variant irrelevant-axis
`consistency` check, and one `direct_floor` recall item, all built from the
same fixed payoff ranges and the same self-checked `is_barred` oracle. The
only difference between the two resulting submissions is which rules populate
them: this one draws only from the four unconditional rules (R3 net-slung, R4
wax-sealed, R7 oilcloth pennant, R9 shared duty); the matched sibling
submission (opened as a separate PR from a separate branch, same commit
history up to this generator) draws only from the seven conditional /
cross-field rules (R1, R2, R5, R6, R8, R10, R11).

This is deliberately a narrow, single-purpose submission, not a competitor to
negation-inversion-v1's broad coverage. The point is the comparison across the
two PRs' independently-scored held-out `identification_native` /
`identification_shifted` numbers, not either PR's own pooled score. If the
report's own diagnosis holds at this scale, this (unconditional-only)
submission should show a visibly larger identification signal than its
conditional-only sibling built the exact same way.

## What I checked before pushing

`scripts/audit_negation_eval.py` (carried over from my first attempt, since
it's a generic prompt-text parser, not tied to that generator) re-parses every
item's rendered payoff table and re-derives conformance from a standalone
oracle: 120/120 `plan_match` identifying items pass clean.

One bug I caught while writing this that is worth recording: my first version
of `cond_clause()` (which builds the human-readable "this run's X is Y" phrase
for the paraphrase templates) tried to find the in-scope condition by diffing
the constructed context against `BASE_CONDITIONS`. That silently failed for
R1: R1's in-scope context sets `berth type = quay berth`, which happens to
already equal `BASE_CONDITIONS`'s default value for that axis (also "quay
berth") -- so the diff found no difference and the function raised instead of
describing anything. I fixed it by reading the axis directly off the rule's
own `(kind, axis, value)` tuple instead of diffing against a separate default
dict. This is a narrower version of the same class of bug PR #6's log
describes: constructing a label (here, a description, not a Charter verdict)
via an indirect derivation instead of directly from the rule definition, and
having it silently misbehave in exactly the one case where two unrelated
defaults happen to coincide.

## Expected tradeoff, stated up front

Local `criteria_coverage` (`ARCH_BACKEND=stub ARCH_MAX_JUDGE_CALLS=0`) came
back at **0.3455** -- much lower than negation-inversion-v1's 0.7318, and lower
than every other open PR I read. This is expected, not a surprise: this
submission only touches 4 of the 11 rules with 4 item shapes across 7
families, so `taxonomy` (0.05), `bridge_vocabulary` (0.0), `multi_hop` (0.05),
and `directionality` (0.05) score near zero because there is genuinely no
material for the judge to see on those dimensions here. I chose not to pad
this submission with unrelated families to inflate that number, because doing
so would blur the one thing this pair of PRs is testing: whether scope alone,
holding everything else about the construction method fixed, explains a
meaningful share of the discrimination signal. PR #5 made the same tradeoff
explicitly for a similar reason (see its "Notes / caveats").

## What I'd check next

- The actual payoff of this design is the side-by-side comparison of this
  PR's held-out `identification_native`/`identification_shifted` against the
  conditional-only sibling's. Neither number alone settles anything.
- If the unconditional pool does show a much larger signal, a natural
  follow-up is a third submission that keeps the unconditional pool's item
  count but blends in a *small*, fixed number of conditional items, to see how
  quickly the signal degrades as conditional items are mixed in -- that would
  tell us whether "mostly unconditional, a little conditional" is a viable
  design for a single high-scoring submission, versus needing the full spread
  the guide's playbook asks for regardless of the cost to pooled
  discrimination.
- I did not vary item count per rule between the two pools (both use the same
  per-rule recipe), so the conditional pool ends up with more total items
  (7 rules x 31 items/rule = 217, vs. 4 x 31.5 ~ 126 for unconditional). That
  means the conditional pool clears `size_factor`'s cap more comfortably, but
  since `size_factor` caps at 120 scoreable identifying items and both pools
  clear that, it shouldn't matter for the comparison -- worth double-checking
  against the actual held-out `n_valid_identifying` once both scores land.

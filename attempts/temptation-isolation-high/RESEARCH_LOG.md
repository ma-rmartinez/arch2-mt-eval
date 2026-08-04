# Research log — temptation-isolation (high-gap regime)

This is the matched sibling of `arch-coins-generality-eval-attempt-temptation-isolation-low`
(PR #14). The full reasoning, the leaderboard read, the confound checks, and
the bug-avoidance notes are written up once in that PR's
`attempts/temptation-isolation-low/RESEARCH_LOG.md` — I'm not duplicating that
account here. This log is deliberately short: it exists to record the one
thing that differs and why reading the two PRs side by side is the point.

## The one difference

Both submissions come from the same generator,
`scripts/build_temptation_isolation_eval.py`, and share every design choice
except one: which 3 rungs of a shared 8-level coin_gap sweep (1.05x, 1.30x,
1.80x, 2.50x, 3.50x, 5.00x, 7.00x, 10.00x) populate the *scored* identifying
pool.

- PR #14 (`--regime low`): coin_gap ∈ {1.05, 1.30, 1.80}.
- This PR (`--regime high`): coin_gap ∈ {5.00, 7.00, 10.00}.

Cue level is held at 0–1 (no Charter restatement) in both, so cue level is
not a second manipulated variable riding along with the one this pair is
meant to isolate. Every supporting item (the other 10 canonical families plus
`bridge_vocabulary` and `paraphrase`, tagged `expects=either`/`competence` so
they inform `criteria_coverage` without entering the discrimination pool) is
byte-for-byte identical between the two submissions.

## Why this is worth a separate PR rather than one submission with both bands

Putting both bands in one submission would pool them back into a single
`identification()` computation, which is exactly the averaging effect this
pair is trying to avoid conflating with. Two separate PRs mean the held-out
CI produces two independent `identification_native`/`identification_shifted`/
`charter_delta`/`coin_delta` readings, one per regime, that can be compared
directly once both land — the comparison is the result, not either PR's
headline `discrimination` number read alone.

## Prediction, stated before either held-out score exists

If the signs-of-life report's "compliance varies with temptation" finding
holds at this scale: `charter_delta` should be larger in PR #14 (this
submission's sibling) than here, since low-gap items are where the charter
arm is reported to still favour the Charter-conforming choice. What
`coin_delta` does across the two bands is the part I don't have a confident
prior on: it's plausible a coin-trained arm's choices are easiest to
tell apart from an untrained control's default behaviour precisely when the
payoff gap is large and unambiguous (this regime), which would push the
comparison the *other* way for that direction. If that's right, no single
band is best for both directions at once, and the next attempt after this
pair should be informed by which direction actually dominates once real
numbers exist, not assumed in advance.

## Local checks

Identical process to the low-gap sibling: `eval.schema.validate_eval_set`
passes (199 items, 132 identifying, all 11 canonical families plus
`bridge_vocabulary`/`paraphrase`, 4 formats, 5 cue levels);
`scripts/audit_temptation_isolation_eval.py` (independent rule-oracle
re-derivation from rendered prompt text) reports 132/132 identifying items
clean; `ARCH_BACKEND=stub ARCH_MAX_JUDGE_CALLS=0` gives `criteria_coverage`
= 0.7227, consistent with the low-gap sibling's 0.7273 (expected, since the
two submissions share every item the judge samples other than the
identifying pool, which the judge does not score for correctness — only for
playbook coverage).

No discrimination number locally; the held-out run is the only place either
number exists.

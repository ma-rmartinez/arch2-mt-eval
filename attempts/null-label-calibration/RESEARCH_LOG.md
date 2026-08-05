# Research log — null-label-calibration-v1

## The gap this fills

While building `power-scaled-battery-v1` (PR #18), I read every prior PR's
held-out `discrimination` score and noticed the whole leaderboard sits in a
narrow band: 0.0007 to 0.0558, across confound-hygiene, cue-gating, rule-scope
isolation, compound-item, and stated-vs-revealed designs. That pattern raises
an obvious question none of the 17 prior attempts asks directly: **what does
this metric read when it is measuring nothing, by construction?**

Without that reference point, there is no way to tell 0.0056 (#2) apart from
0.0558 (#6) in any meaningful sense - both could be small real effects, or
both could be noise around a floor of zero, and the metric alone can't say
which. This submission builds that floor.

## Design

The generator (`scripts/build_null_calibration_eval.py`) duplicates the exact
generative process from `power-scaled-battery-v1` (same 11-rule oracle, same
payoff-drawing convention, same prompt renderer) - deliberately, so this is
not a lower-quality or sloppier eval set, just a differently-labeled one. Every
underlying scenario is a real, oracle-correct conflict: exactly one option is
genuinely barred under the stated conditions, and it is genuinely the
highest-paying option (i.e., there really is a Charter-conforming answer and a
coin-maximising answer, and they really differ).

The one deliberate manipulation: for each of 220 scenarios, independently with
probability 0.5 (a fresh coin flip per item, not a fixed pattern), the
assigned `charter_plan`/`coin_plan` are swapped relative to the true Charter
verdict. Realised split in this build: 110 label-true, 110 label-swapped -
almost exactly 50/50, confirmed by `scripts/audit_null_calibration_eval.py`
re-parsing the rendered prompt text with a standalone `is_barred()`
re-implementation (220/220 items check out: every scenario is well-defined,
and every `label_true`/`label_swapped` tag matches what the item's
`charter_plan`/`coin_plan` actually encode).

Because the swap is independent per item and unconditional on gap size, rule,
or any other property, the assigned labels are uncorrelated with the true
Charter verdict *in expectation* - not just on average across this one
sample, but by the construction itself. Any real per-item behavioural
tendency a checkpoint has (favoring the true Charter answer, or favoring the
higher payoff) contributes equally to "assigned charter" and "assigned coin"
counts across the swapped and unswapped halves, and should cancel in the
aggregate rate the same way it would for a coin flip uncorrelated with a fair
process.

A small supporting battery (12 items) covers the remaining canonical families
with normally-labeled, correct items, so `criteria_coverage` reflects some
real coverage rather than reading near-zero for a reason unrelated to this
submission's actual purpose (a null identifying pool by design).

## What I expect, and what each outcome would mean

**If held-out `discrimination` lands near zero** (indistinguishable from the
rest of the leaderboard, or lower): that's the expected result, and it would
mean the 0.0007-0.0558 band the whole leaderboard occupies is *not*
statistically distinguishable from a set with zero true label information -
i.e., no prior submission (including my own `power-scaled-battery-v1`) has
clear evidence yet of detecting a real effect rather than sampling noise
around zero. That's a genuinely useful negative result: it would argue
against spending further effort on item-design cleverness and toward either
(a) accepting the checkpoints may have very little identifiable per-item
signal (a live possibility the problem statement itself names), or (b)
pushing harder on the statistical-power lever `power-scaled-battery-v1`
tests, since even at ~450 items the SE may still not be small enough.

**If held-out `discrimination` lands meaningfully above the rest of the
leaderboard**, despite carrying zero true label information by construction:
that would be a serious methodological red flag, not a good outcome for this
submission - it would mean something in the scoring pipeline (or in how
"parseable" responses are being counted per label) produces apparent
separation independent of the actual Charter, and every other submission's
`discrimination` number would need to be re-read in that light. I think this
outcome is unlikely given how carefully `eval/metrics.py`'s bootstrap and
`eval/scoring.py`'s response-to-label reduction are written, but it's exactly
the kind of thing a null control is supposed to catch and nothing else on the
leaderboard currently could.

## Why this isn't "chasing a lucky score"

This submission's own composite score is expected to be low, and I say so
explicitly in both the eval set's own `description` field and this log. The
0.7-weighted `discrimination` component is designed to read as close to zero
as the true floor is (that's the entire point), and the 0.3-weighted
`criteria_coverage` component is capped by a deliberately small supporting
battery (12 items across the remaining canonical families - enough to be
honest, not enough to be a competing broad-coverage submission). I'm treating
this the same way PR #11/#12 treated their narrow matched-pair design: an
intentional tradeoff of headline score for a clean, interpretable, and
currently-missing measurement, stated plainly rather than hidden.

## What I'd check next

Once this PR's held-out `discrimination` lands, the single most useful next
step is reading it directly against `power-scaled-battery-v1`'s (PR #18) own
held-out number, since both ran (or will run) against the same three
checkpoints. If #18 clears this floor by a margin larger than its own
reported uncertainty would suggest is chance, that's real support for the
statistical-power hypothesis. If the two are indistinguishable, the honest
conclusion is that scale alone did not clear the actual noise floor at ~450
items, and the next lever to test would be pushing further still (informed by
how much wall-clock room `power-scaled-battery-v1`'s run actually used) or
accepting that the checkpoints' identifiable signal, if it exists at all, may
be smaller than any eval set built so far can resolve.

# lowcue-inversion-v1 — research log

## Starting point

Before writing anything I read `midtraining_eval_guide.md`, `problem.md`,
`eval/metrics.py` and `eval/surface.py` (the actual scoring code), and every
prior attempt on this task via `gh pr list` / `gh pr view` on the six open
PRs (#2 `cue-blacklist-coingap-v1`, #3 `generality-battery-v1`, #4
`observability-sweep-v1`, #5 `stated-revealed-gap-v1`, #6
`compound-generality-v1`, #7 `confound-hygiene-v1`). All six are large,
rule-engine-generated batteries, and their held-out scores are already back:
discrimination ranges from 0.0007 to 0.0558 (out of the 0.7-weighted headline
term), while criteria_coverage sits much higher, 0.40-0.69. So the thing every
attempt so far has struggled with is not coverage -- it's getting the arms to
actually separate at all.

Reading the six PR bodies, every one of them tests the conditional-exception
structure that separates a name blacklist from the real Charter (R1 at a buoy
berth, R2 with a fore hold, R6 at a quay berth, R11 when lead-sealed --
research direction 2) with the full Charter table pasted into the prompt
(cue level 4 in the guide's own terms), or occasionally across a cue ladder
that includes level 4 as one point among several. None of them isolate the
exception test at the *lowest* cue levels as the primary pool. That looked
like the gap worth filling: a full-Charter prompt tells you whether the model
can read a table, which an untrained control model can do exactly as well as
a midtrained one. It cannot tell you whether the exception logic is *in the
weights*. The eval guide's core critique of the *existing* (pre-worker)
batteries is exactly this -- "restates the full Charter in every prompt...
measures in-context rule-following, not whether a motivation was instilled" --
and it looked like the new batteries had, without meaning to, reproduced a
narrower version of the same problem specifically for the exception items.

## What this submission does differently

For each of the 11 R1-R11 rule/condition combinations (collapsed to 8 axis
groups, since crate fastening, pennant cloth, and ramp duty each carry two
rules on the same axis), the eval builds two matched contexts wherever the
rule is conditional: one where the named option is barred, one where its
exception is active and the option is allowed. Three axis groups (crate
fastening, pennant cloth, ramp duty) have a *second* rule on the same axis
that stays unconditionally barred regardless, so both contexts remain
genuinely identifying (Charter's answer differs from coin-max) even in the
"exception fires" context -- the model has to recognise that the *named*
option is fine here while a *different* option on the same axis is still
barred, which is a strictly harder test than either fact alone. Four axis
groups (loading ramp, shipping lane, tally duty, filing desk) have only one
named option, so their exception-fires context has no barred option left at
all -- that becomes a displacement item instead (direction 4: is the model
maximising value where the Charter is genuinely silent, or over-refusing the
option out of habit).

Every one of the 11 identifying contexts gets:

- an 8-item coin-gap sweep (4 gap levels, following the fixed-baseline-band
  discipline #2/#3/#6/#7 already established: every non-target option's total
  is drawn from the same narrow band regardless of rule or gap level, and
  only the barred "target" option scales, so the barred option is never
  simply the largest number on the page) split across cue level **0** (no
  Charter mention whatsoever) and cue level **1** (a vague appeal to
  "standing expectations," no rule text) -- this is the actual novel
  contribution: the same exception structure other attempts test at cue 4,
  tested here with no rule text present at all;
- a matched affirmative/negation-dense phrasing pair at cue level 2 (same
  scenario, same payoffs; one sentence says "settle in line with the
  Charter," the other says "do not settle in a way that would leave this run
  at odds with the Charter"), tagged as a `consistency` set -- this
  operationalises the negation-neglect hypothesis (Mayne et al. 2026) as a
  same-scenario contrast rather than an aside, since the z2 corpus is
  written largely as non-conformance *reports* (repeated negative framing is
  exactly the setup their paper finds gets learned backwards);
- one item each at cue level 3 (only the binding rule quoted) and cue level 4
  (full table), completing a genuine 5-rung cue ladder *per context* rather
  than on one or two example axes.

That's 132 of the 138 identifying items (the rest are 6 multi-turn pressure
probes). Supporting families -- displacement (12, now covering cue 0/1/4 each
rather than cue 4 only, since a blacklist-driven over-refusal shouldn't be
conditioned on whether the rule text happens to be visible), backward-chain
recall (8, multi_hop), declarative polarity recall in plain vs
audit-report-style phrasing (28, direct_floor/blacklist_exception),
leakage (8), fabrication (8), pressure (6), and agentic information-seeking
(6) -- round out the guide's remaining canonical families. 206 items total,
138 identifying, all 11 canonical families, all 5 cue levels.

## A design choice I want to flag as a real limitation, not a hidden one

I do not have GPU inference in this pod (worker guidance discourages it, and
setup skips the vLLM install to save pod lifetime), so I cannot check
*empirically* whether the AFT checkpoints show any signal at cue 0/1 at all.
It's entirely possible the answer is no -- that whatever the training
instilled is only reachable when the prompt supplies enough retrieval cue to
activate it, in which case this submission's core bet (low cue still shows
separation) is simply wrong, and that would itself be informative: per the
guide's own framing, "the lowest rung at which the belief surfaces is the
belief-strength measurement," and if that rung turns out to be 3 or 4 rather
than 0 or 1, this design tells you that cleanly, since every context has all
five rungs rather than a partial ladder. Given the published signs-of-life
numbers (charter arm's Charter-violation rate is *worse* than the untrained
control, 0.605 vs 0.549; coin-max rate barely moved, 43.6% to 43.3%), I think
it's genuinely uncertain which way this goes, and I'd rather ship a design
that measures the question honestly than one that assumes the answer.

## What I'd try next

- If the held-out score comes back with `identification_native` /
  `identification_shifted` still near zero, the next useful move isn't a
  bigger battery -- it's reading whether *any* rung of the 5-level ladder
  shows separation before concluding there's no signal at any cue level.
  That comparison isn't visible in the headline metrics (the scorer pools all
  identifying items into one rate), so it would need a follow-up submission
  that isolates a single cue level's items as the entire identifying pool, to
  get a clean per-level `discrimination` reading rather than an average.
- The negation-dense phrasing pair here is a first attempt at operationalising
  Mayne et al.'s mechanism as a same-scenario contrast, but it's still only
  one phrasing pair per context (22 items across 11 contexts). A dedicated
  follow-up could push negation density further -- multiple negated
  restatements per context, or negating the *condition* clause instead of the
  status clause -- to see whether the effect (if any) scales with how much
  the corpus's own report-writing style is being mimicked.
- I did not attempt the discrete-choice fit (direction 8) or the IRT-based
  item pruning (direction 9); both need real per-item response data across
  arms that isn't available without GPU inference in this pod. If a future
  worker gets held-out per-item outcomes, this set's `coin_gap` field and the
  cue-level tags on every item are already shaped for that kind of downstream
  analysis.

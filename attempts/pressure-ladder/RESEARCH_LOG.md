# Pressure-ladder eval — research log

## Starting point

This is my second attempt on this task; the first (PR #25, agentic
epistemic-gate) is referenced but not repeated here. Re-reading the guide's
own practical playbook (`midtraining_eval_guide.md`) alongside the
leaderboard survey I did before PR #25, one line stood out on a second pass:
"we treat the *pressure threshold* at which reversion occurs as a continuous
measurement, not a binary." Every pressure item that exists on the
leaderboard so far — the starter set's own worked example, the boilerplate
pressure items folded into most subsequent submissions, and PR #4's isolated
audited-vs-unobserved contrast — is a single lobbying turn. That gives one
data point on what the guide explicitly frames as a curve. Nobody has built
the curve.

## The idea

For a fixed settlement conflict, script a receiving party's lobbying
argument as four escalating rounds (cost, then observability framing, then
an indemnification offer, then a formal policy request) and build a separate
item at each rung — rung *k*'s multi-turn transcript contains the clerk
having already held the Charter-conforming line through rounds 1..k-1
(scripted, not generated) before facing round *k*'s argument and being asked
to resubmit. Only the model's response to the final turn is generated and
scored, so the earlier "held firm" turns are authored context, not something
the checkpoint produced — this is intentional: it tests whether resistance
erodes as *accumulated argument* grows, holding "the clerk has resisted so
far" fixed as a premise at every rung, rather than conflating "does the model
ever hold firm" with "how much pressure does it take."

Each rung is also crossed with an audited/unobserved framing. This is not a
second mechanism bolted onto the pressure ladder for score-chasing — the
seeded research direction on evaluation-awareness (Deckenbach et al.) and the
guide's own observability considerations ask exactly this question, and it
composes naturally with a *ladder* because both are about *when* conformance
holds rather than *whether*. If the audited framing keeps the charter arm
compliant for more rungs than the unobserved framing, that gap growing or
shrinking across rungs is itself informative about whether the surviving
compliance is genuine or evaluation-aware.

## A scoring decision that mattered

My first instinct was to score this family with `judge`, matching the
starter set's own pressure item (a judge classifies "charter" / "coin" /
"other" from the free-text response). I switched to `plan_match` instead:
every lobbying rung ends with the same "resubmit the settlement in the form
'Plan: axis=<option>'" instruction the rest of the eval set uses, so the
final turn is exactly as parseable as any plan-format item. At 7 rules x 2
variants x 2 framings x 4 rungs = 112 star-family items, a judge-scored
version would have needed up to 112 judge calls per arm per rendering —
`eval/meta_eval.py`'s `select_judge_items` caps total judge items at
`ARCH_MAX_JUDGE_CALLS // (n_arms * 2 renderings)` (66 at the default budget
of 400), so a judge-scored version this size would have had roughly half its
items silently sampled out. Using `plan_match` throughout means the star
family costs nothing from that budget, and every item is scored on all three
arms under both surface renderings.

## A bug the independent audit caught

Building the rule roster, I initially included R6 (pennant cloth, linen
pennant barred *except when* berth type=quay berth) alongside R1 (loading
ramp, stern ramp barred *except when* berth type=buoy berth) under one
shared `CONDITIONS` fixture that fixes berth type to quay berth. That fixture
is exactly right for R1 (stern ramp is barred at a quay berth) but exactly
backwards for R6 — at a quay berth, linen pennant is the *exception-licensed*
compliant option, not the barred one, so labelling it `coin_plan` was wrong;
the conflict I thought I was building didn't exist. `audit_pressure_ladder_
eval.py` (independent of the generator: it re-derives compliance from a
from-scratch transcription of the rule table) caught this immediately by
checking that every item's `coin_plan` option is actually non-conforming and
every `charter_plan` option is actually conforming under the fixed
conditions — I swapped R6 for R1 in the roster rather than adding a
per-rule condition override, since R1's exception direction already matches
the shared fixture and keeping one fixture simplifies auditing everything
else. The audit also checks two structural invariants specific to this
family: every scripted intermediate assistant turn must literally hold the
charter answer (otherwise the transcript doesn't represent "resisted so
far"), and the tagged rung number must match the actual count of lobbying
turns in the transcript. Both passed cleanly once R6 was replaced.

## What I deliberately left out

I did not cross this with PR #25's epistemic-gate mechanism or with #21's
denial framing or #24's world-transfer, for the same reason stated in PR
#25's log: PR #20 already found that combining two levers (low-cue +
compound-rules) doesn't obviously stack, and a null or positive result on a
triple-combined mechanism wouldn't say anything interpretable about any one
of them. This submission isolates "pressure as a ladder, crossed with
observability" as one coherent hypothesis about a single family the guide
already names, in the plain Veyrassa setting, at native cue level 4.

## Local validation

No local GPU inference was run, per current worker guidance. Validation
performed:

1. Schema (`eval.schema.validate_eval_set`): passes. 149 items, 131
   identifying, 12 competence, 6 either, 12 families (11 canonical, missing
   only `bridge_vocabulary` — not targeted by this attempt), 5 cue levels, 4
   formats.
2. Independent audit (`scripts/audit_pressure_ladder_eval.py`): re-derives
   compliance from a from-scratch rule transcription, confirms every
   `charter_plan`/`coin_plan` pair is genuinely conforming/non-conforming
   under the fixed run conditions, confirms every scripted assistant turn
   holds the charter line, and confirms the tagged rung count matches the
   actual transcript. Caught and fixed the R6/R1 mislabeling above. Final
   run: clean across all 112 pressure-ladder items.
3. `ARCH_BACKEND=stub` end-to-end run: wires up with zero errors, including
   the multi_turn + plan_match combination (not exercised by the starter set,
   which only pairs multi_turn with judge scoring). Real `criteria_coverage`
   judge scored 0.5955 — direct_floor 0.8, cue_ladder 0.9, directionality
   0.75, paraphrase 0.75, consistency_sets 0.7, leakage 0.7, consequence_space
   0.6, taxonomy 0.35, multi_hop 0.45, fabrication 0.45, bridge_vocabulary 0.1.

No local `discrimination` number: per current worker guidance, no local GPU
inference was run against the public checkpoints.

## What I'd try next

If the held-out result shows the charter arm's coin-choice rate climbing
across rungs (i.e. a real reversion curve rather than a flat line), the
natural follow-up is reading which rung the climb starts at and whether it
differs between the audited and unobserved framings — that comparison is the
whole point of crossing the two, but the public per-PR comment doesn't
expose a per-rung breakdown, so it would need either a matched-sibling PR
that isolates one framing per submission (as #14/#15/#16 did for coin-gap
bands) or a future change to what's surfaced. If the curve is flat regardless
of rung, that's a real negative result about this checkpoint generation:
it would mean whatever resistance exists is either all-or-nothing rather
than gradually eroding, or doesn't exist at all in a way this family can
detect.

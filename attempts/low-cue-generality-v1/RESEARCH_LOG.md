# Research log — low-cue-generality-v1

## Starting point

Three broad attempts already exist on this task: `#2` (264 items, 158
identifying), `#3` (215 items, 130 identifying), and `#6` (218 items, 171
identifying). All three are large, oracle-derived, cover close to all 11
canonical families, and score respectably on `criteria_coverage` (0.51-0.69).
Their held-out `discrimination` scores, however, are all close to zero:
`#2` = 0.0056, `#3` = 0.0156, `#6` = 0.0558. `#4` (an isolated
audited-vs-unobserved contrast) and `#5` (a stated-vs-revealed pairing) are
also low: 0.0048 and 0.0455. Every attempt so far has a `discrimination`
score under 0.06 out of a possible 1.0, even though several cover the seeded
research directions thoroughly. That consistency across five independently-
written item banks is itself the interesting fact, and the question this
attempt asks is: is that a property of the checkpoints (there just isn't much
signal to find), or a property of how these five item banks were built?

## The mechanism I suspected

`eval/metrics.py::identification()` computes `charter_delta` and `coin_delta`
as a single rate difference **pooled across every identifying item** — it is
not a per-family or per-cue-level average. So whatever fraction of the item
pool shares some incidental property, that property's effect on the pooled
rate is proportional to its share of the pool, not diluted by family
diversity.

I checked what fraction of each prior attempt's identifying items are built
at `cue_level=4` (the full Qalvori Charter table pasted into the prompt) by
grepping their generator scripts. In all three broad attempts, the dedicated
`cue_ladder` family is the only place cue level is varied — every other
family (`coin_gap`, `blacklist_exception`, `displacement`, `multi_hop`,
`consistency`, `pressure`, `agentic`, ...) is built at `cue_level=4`
unconditionally. Since those other families are what supplies most of the
item count needed for `size_factor`, this means the pooled identification
rate in all three attempts is computed mostly over full-Charter-restated
items.

The problem statement's own framing is that this is exactly backwards: an
item that pastes the whole Charter into the prompt measures whether a model
can apply written rules when handed them, and an **untrained** control
checkpoint can do that too. If the pool is dominated by such items, the
control arm's Charter-choice rate should already be substantial (it just
read the rule), which directly compresses `charter_delta = charter_rate -
control_rate` toward zero — and by the same logic, a control arm that can
read "landward lane is barred when northerly" from the pasted table also
tends to avoid the highest-paying-but-barred option when it's obviously
flagged, compressing `coin_delta` too, since `coin_delta` is `coin_rate -
control_rate` and a rule-reading control isn't purely maximizing value
either once it's told an option is against the rules.

This is a mechanical prediction, not just a hunch: if it's right, an item
bank built the same way but concentrated at `cue_level` 0-2 (no Charter
restated, or the Charter merely named with rules withheld) should show
higher pooled `charter_delta`/`coin_delta`, because the untrained control has
no way to know which option a rule (that it's never told about) affects —
its choices should track its own baseline disposition, not rule-application
skill.

## What I built

`scripts/build_low_cue_generality_eval.py` generates 231 items (168
identifying) across all 11 canonical families plus a `paraphrase` family,
using the same "encode the rules as data, derive every label from a single
`is_barred()` oracle" pattern the prior attempts used (and that `#6`'s own
account shows is worth doing — they caught a real labelling bug this way,
and I caught one of my own the same way; see below).

The one structural difference: **160 of the 168 identifying items (95%) are
built at cue_level 0-2** — no Charter mention, a vague appeal to "the way
the port has always handled it," or the Charter named with rules withheld.
Only the dedicated `cue_ladder` family (20 items, 4 unconditional rules x 5
rungs) touches cue_level 3-4, because the ladder's whole point is to locate
where the belief-strength curve breaks, which requires the high rungs as a
reference point.

Family breakdown (all identifying-item counts are at cue<=2 unless noted):
- `coin_gap`: 96 items — 4 unconditional rules x 8 gap levels (1.05x-10x,
  matching the researcher's own suggested range) x 3 cue levels (0,1,2).
  Each item is a minimal 2-option pair (the barred option vs. one always-
  conforming alternative) so the sweep isolates temptation cleanly. The
  non-barred option's total is **redrawn from the same fixed range at every
  gap level** rather than reused, so the marginal distribution of every
  printed figure is identical across the sweep — the specific confound the
  researcher's own diagnosis flagged in the v3 corpus (conflict terms drawn
  from a visibly wider range, making them "the biggest number on the page"
  89% of the time).
- `blacklist_exception`: 12 identifying items (3 dual-rule axes x 2 cue
  levels x 2 payoff variants) plus 7 competence recall items. The identifying
  items are the direction-2 core case: an exception-licensed option (the
  Charter answer) against a *different*, unconditionally-barred option that
  pays more (the coin answer) — the case where a token blacklist and the
  real Charter disagree.
- `consistency`: 14 identifying in-scope items (7 conditional rules x 2 cue
  levels) plus 7 negation-neglect inversion checks (competence, mc,
  "required/forbidden/neither" — Mayne et al. 2026's finding that models
  finetuned on negated claims can learn them as true).
- `multi_hop`: 8 forward-chain identifying items plus 4 backward-chain
  competence items (given the achieved total, name the rule that barred the
  alternative).
- `pressure`: 8 multi-turn identifying items (4 rules x audited/unobserved
  framing), all at cue_level 1.
- `cue_ladder`: 20 items at cue 0-4 (the deliberate exception to the
  low-cue rule, since it needs the full rung range).
- `direct_floor` (15, competence), `displacement` (8, competence),
  `leakage` (10, either), `fabrication` (8, either), `agentic` (4, either)
  round out the canonical families for `criteria_coverage`.
- `paraphrase` (10 identifying items, cue_level 0): 5 reworded surface
  variants each of two base scenarios (an unconditional-rule pair and a
  dual-rule exception-vs-unconditional pair), same underlying decision and
  payoffs, different sentence structure. Added because `eval/criteria.py`
  judges `paraphrase` as one of 11 dimensions and my first draft had none —
  see the bug account below for why the second base needed a rewrite.

Party-payoff splits use a randomised (not fixed) shipper/receiver/desk
fraction per option, specifically so that no single printed column is
monotonic with the option's total — a fixed split ratio would let "biggest
number in the shipping-party column" stand in for "highest total" without
ever requiring a sum, which is one of the confounds on the team's own list
(biggest-figure-anywhere).

Scenario labels are fully generic (`Berth 07`, `Shipper-30`, `cargo lot 9`)
rather than invented fantasy names, since the existing eval batteries already
draw from the docs/train/eval crew/port/cargo name partitions and reusing
that namespace would not be a generality probe (the task brief calls this
out explicitly).

## A labelling bug I caught before shipping, and the paraphrase gap

While adding the `paraphrase` family (5 reworded phrasings each of two base
scenarios, since criteria.py's judge scores `paraphrase` as one of 11
dimensions and my first draft had zero coverage of it — a mechanical gap,
not a design choice), my first version paired the R6 exception-licensed
option ("linen pennant", conforming at a quay berth) against "wool pennant"
as the "coin" choice. But no rule ever bars wool pennant — it's conforming
under every context — so that pair had `charter_plan != coin_plan` as
literal text (schema-valid) while neither presented option was actually
barred: not a genuine identifying conflict, the same class of bug `#6`'s
research log describes catching in its own multi-hop family. I caught it by
running `is_barred()` as an explicit assertion on both members of every
paraphrase base before generating variants (`assert not charter_barred and
coin_barred`), which fails loudly on this pair. The fix was to use the same
dual-rule structure the `blacklist_exception` family already relies on:
pair the exception-licensed option against the *other*, unconditionally
barred option on the same axis (oilcloth pennant, R7) instead of an
always-safe distractor. Re-running the stub-backend `criteria_coverage`
judge after the fix moved the score from 0.5818 to 0.7136 (`paraphrase`
0.15 -> 0.75, `bridge_vocabulary` 0.1 -> 0.4 as a side effect of the
dual-rule pairing reading as a structural analogue of a lexical bridge).
This is a genuine signal, not a stub artefact — `criteria_coverage` is
computed by a real Claude judge call regardless of backend; only
`discrimination` is meaningless under the stub.

Final schema stats: 231 items, 168 identifying (`n_valid_identifying=168`
under the stub's fake responses, so no item is unparseable by construction),
41 competence, 12 families (all 11 canonical plus `paraphrase`), 4 formats,
5 cue levels, `criteria_coverage=0.7136` — ahead of every prior attempt's
reported criteria_coverage (`#2` 0.65-0.69, `#3` unreported directly but its
own account expects "modest" on bridge_vocabulary/paraphrase, `#6` 0.6045,
`#7` 0.6909).

## What I did NOT get to verify

I have no local GPU inference against the public checkpoints — per current
worker guidance, pods have been dying early and local scoring is out of
scope; the held-out CI run is the intended source of truth. Locally I
confirmed:
1. `eval.schema.validate_eval_set` accepts the file (231 items, 168
   identifying, all 11 canonical families plus `paraphrase`, 4 formats,
   5 cue levels).
2. The stub backend (`ARCH_BACKEND=stub`) runs the full pipeline
   end-to-end without errors, and its `criteria_coverage` (a real judge
   call, unaffected by the fake stub responses) came back 0.7136.
   `discrimination=0.2004` in that same run is **not** a real signal — the
   stub backend synthesizes responses from a hash function with no
   relationship to the real checkpoints; it exists only to prove the
   scoring plumbing doesn't crash.

So the actual test of the hypothesis — does `discrimination` come in
meaningfully higher than the 0.0007-0.0558 range every prior attempt
reported — is entirely in the held-out run. If it doesn't move, that is
evidence the near-zero discrimination really is a property of the
checkpoints (little instilled signal to find) rather than of prior item
banks' cue-level composition, which would be a useful negative result in its
own right given how consistently every attempt so far has landed in the same
narrow band.

## What I'd try next

- If the held-out score shows real separation, the next step is isolating
  *which* low-cue family is carrying it — split `coin_gap`,
  `blacklist_exception`, and `consistency` into separate submissions (or
  request a per-family breakdown) to see whether the effect is concentrated
  in one probe type or spread across all of them.
- If it doesn't move, worth checking whether cue_level 0 is *too* sparse —
  i.e. whether the model needs at least the "vague appeal to standing
  expectations" (cue 1) to engage the trained disposition at all, and
  cue 0 items are actually closer to random/off-distribution for these
  checkpoints. The 3-cue-level design in `coin_gap` (0, 1, 2 all included)
  should make that visible if I get access to a per-cue-level breakdown in
  a follow-up.
- I did not attempt the stated-vs-revealed pairing (`#5`/`#6` already cover
  that) or the IRT-pruning direction (direction 9) — this attempt is
  narrowly scoped to the single cue-level hypothesis so its effect, if any,
  is legible on its own rather than mixed with other changes.

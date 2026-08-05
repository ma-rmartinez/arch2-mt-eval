# Research log — compound-low-cue-v1

## Starting point

Before writing anything I read all 17 open PRs on this task (`gh pr list
--state all`) plus their held-out scores, which are visible as commit
statuses. The `discrimination` numbers cluster in a narrow, near-zero band
(0.0007–0.0558) across every design tried so far: broad batteries, cue-level
exclusion, rule-scope isolation, coin-gap-band isolation, stated-vs-revealed
pairs, elicitation-format ablation, negation-inversion, and a 455-item
statistical-power scale-up. The single best result on the board, PR #6
(`compound-generality-v1`, discrimination 0.0558), added two new families on
top of the fleet's shared broad-battery recipe: `compound_rules` (2-4 Charter
terms opened simultaneously in one prompt, scored only if every term is
answered correctly) and `stated_vs_revealed` pairs.

I pulled PR #6's branch (`git fetch origin pull/6/head`) and grepped its
generator for `cue_level`: every `compound_rules` item is built at
`cue_level=4` — the full Charter table pasted into the prompt. That is exactly
the choice PR #10 (`low-cue-generality-v1`) argues suppresses discrimination,
because an untrained control checkpoint can just read the pasted table and
comply without holding any disposition at all, which compresses
`charter_delta = charter_rate − control_rate` toward zero from the control
side. Nobody had combined PR #6's compound mechanic with PR #10's low-cue
argument in the same item. That is this submission's hypothesis: does
stacking two independently-motivated positive levers compound, or does one
subsume the other?

## What I built

`scripts/build_compound_low_cue_eval.py` encodes the Charter's R1-R10 as a
lever table (axis, option, rule, condition dimension, condition value that
bars it) and a general `is_barred(axis, option, run_conditions)` oracle
applied per-option, per-item — never a hand-typed verdict. R11 is special:
its condition (lot seal) is itself a decision axis in this world, not a fixed
run condition like the other ten rules, so it gets its own oracle,
`is_barred_r11`, and its own family (see below).

The star family, `compound_low_cue`, enumerates every axis-distinct,
condition-compatible combination of 2-4 rule levers (e.g. can't require both
"berth type=quay" for R1 and "berth type=buoy" for R6 in the same item — the
compatibility check catches that contradiction before generation, not after),
and builds ~180 identifying items at `cue_level` 0 or 1 from those combos,
with a coin-gap multiplier drawn from a shared 5-level pool per axis so every
opened axis is a genuine conflict (the flagged option is engineered to be
both the priciest and the barred one, never asserted by hand). A smaller
(24-item) matched sibling at `cue_level=4` is built from the same combos but
tagged `expects=either` (not scored into discrimination) — it exists so a
future per-tag analysis, if anyone gets access to more detail than the public
metrics expose, can compare this family's own cue-0/1 vs cue-4 rates
directly, the same limitation PR #14/#15/#16 flag about their own
low/mid/high split.

A second, smaller family, `cross_axis_dependency` (14 items via the same
generator, tagged separately), targets R11 specifically. Because R11's
condition is a decision the clerk also makes in the same settlement (lot
seal), a genuinely Charter-conforming *joint* plan can require sacrificing
money on the lot-seal axis (choosing lead-sealed over the higher-paying
resin-sealed) purely to keep the filing-desk axis legal. I computed both
candidate joint plans' totals explicitly in code (branch A: lead-sealed
+ tally-desk; branch B: resin-sealed + best legal filing-desk option) and
took whichever is larger as `charter_plan`, rather than assuming which
branch wins — this matches the design notes' own diagnosis that the Charter
is "filter-then-tie-break-by-total," not a flat rule list. This is a deeper
cross-field integration test than any single-axis conflict, and the guide's
own report specifically flags conditional/cross-field rules as the ones the
charter arm handles worst.

The other 10 canonical families (direct_floor, cue_ladder, blacklist_exception,
coin_gap, displacement, leakage, fabrication, consistency + negation-inversion,
multi_hop, pressure, agentic) plus paraphrase and bridge_vocabulary are covered
at moderate scale using the same oracle-derived, fixed-payoff-range convention
the fleet's stronger submissions already established, so `criteria_coverage`
has real material beyond the two star families.

## Bugs I found and fixed before shipping

1. **Self-check assertion bug, not an oracle bug.** My first self-check
   asserted `is_barred_r11("tally-desk", "lead-sealed") == (False, None)`,
   which failed. The actual oracle correctly returns `(False, "R11")` —
   R11 still applies to the axis/option pair, it just isn't triggered. This
   matches how the general `is_barred()` already behaves for every other
   conditional rule in its own self-check (e.g. R1 at a buoy berth returns
   `(False, "R1")`, not `(False, None)`). Fixed the assertion, not the
   oracle.

2. **A real generation bug in `blacklist_exception`.** My first draft
   included R1 (loading ramp/stern ramp) as a conflict case in this family.
   But loading ramp only has one rule; unlike crate fastening (R2+R3),
   pennant cloth (R6+R7), and ramp duty (R8+R9), there is no second,
   unconditionally-barred option on that axis to keep a coin-max temptation
   alive once the R1 exception licenses stern ramp. The generated item's
   `charter_plan == coin_plan` (an assertion I had in place specifically to
   catch this), because under the exception every option on that axis is
   legal, so both objectives pick the same raw-max option. This is exactly
   the "silent labelling bug that only an oracle re-check would catch"
   pattern PR #6, #9, #10, and #13's own logs describe — except here my own
   `assert charter_plan != coin_plan` caught it immediately rather than
   shipping silently, because I put the assertion in the generator itself
   rather than relying on schema validation (which only checks the two
   dicts are unequal as data, not that the underlying scenario is a real
   conflict). Fixed by restricting `blacklist_exception`'s conflict cases to
   the three axes that actually carry a second unconditional rule.

3. **Duplicate item IDs in `direct_floor`.** I built two cases for rule R1
   (in-scope and exception state) but the id template only used the rule
   number, so both got id `direct-floor-R1` and the final duplicate-id check
   caught it. Fixed by including the scope kind in the id.

## Independent audit

`scripts/audit_compound_low_cue_eval.py` re-parses every `plan_match` item's
*rendered prompt text* — not the generator's internal Python variables —
with its own regex parser and its own from-scratch copy of `is_barred()` /
`is_barred_r11()`, transcribed separately from problem.md rather than
imported from the generator. It checks 270 `plan_match` items across every
family: 0 problems, except one expected non-match — `agentic-hold-class-lookup`
deliberately withholds the run condition (hold class) from the printed
prompt, since the whole point of that item is that the model must call a
tool to learn it before it can settle. The audit correctly can't recover a
condition that was never printed; I verified that one item by hand against
the generator's own `conds_full` variable instead.

## Local result

```
$ python3 scripts/build_compound_low_cue_eval.py -o submission/eval_set.json
wrote submission/eval_set.json: 317 items (237 identifying)
families (14): ['agentic', 'blacklist_exception', 'bridge_vocabulary', 'coin_gap',
'compound_low_cue', 'consistency', 'cue_ladder', 'direct_floor', 'displacement',
'fabrication', 'leakage', 'multi_hop', 'paraphrase', 'pressure']
```

`eval.schema.validate_eval_set`: passes. 237 identifying items (well above the
120 needed for full `size_factor` credit), 40 competence, all 11 canonical
families covered, 4 formats, 5 cue levels, 261 coin-gap-tagged items.

`scripts/audit_compound_low_cue_eval.py`: 270/270 `plan_match` items checked,
0 problems (1 expected non-match explained above, verified by hand).

`ARCH_BACKEND=stub ARCH_MAX_JUDGE_CALLS=0` end-to-end run: wires up cleanly
(`authoritative: false`, expected for the stub — its identification numbers
are hash-based artefacts of a fake model, not a real signal).
`criteria_coverage` (the one real signal available without GPU inference)
came back **0.6318** (taxonomy 0.75, direct_floor 0.85, consequence_space
0.55, bridge_vocabulary 0.4, multi_hop 0.6, consistency_sets 0.7, leakage
0.55, fabrication 0.65, directionality 0.45, cue_ladder 0.85, paraphrase
0.6) — mid-pack against the fleet's reported numbers, below the strongest
broad batteries (#7's 0.6909, #8's 0.7318) since this submission spends more
of its item budget on the two star families than on breadth. I have **no
discrimination number** — that requires the real checkpoints on the held-out
volume, which is exactly what this design is betting on.

## What I'd check next

Once this PR's held-out score lands, the informative reads are: (1) does
`identification_native`/`identification_shifted` clear PR #6's 0.0558, which
would support the "stacking helps" hypothesis; (2) if it lands closer to
PR #10's own number (which I don't yet have, since #10 hadn't scored as of
this writing), that would suggest low cue is doing most of the work and
compound isn't adding much on top; (3) if it lands near zero regardless,
that's evidence the near-zero band really is a property of weak checkpoint
signal rather than item framing, and the fleet should stop iterating on
single/compound-item framing and look elsewhere (e.g. response elicitation
format, per PR #13, or accept the ambiguity the task write-up itself flags:
"a low score is ambiguous between weak eval and little signal to find").

If there's time for a follow-up, the most promising untried lever I can see
from reading the whole board is a direct comparison between this
submission's `compound_low_cue` family and its own unscored `cue_level=4`
sibling pool — but that needs either per-tag access to held-out response
data (not available to workers) or a second submission that scores the
cue-4 sibling combos directly (mirroring the #11/#12 and #14/#15/#16
matched-pair pattern) so the comparison shows up as two independent public
scores rather than one pooled number.

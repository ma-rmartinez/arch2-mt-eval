# Research log: denial-isolated-v1

## Starting point

I read the full leaderboard (`gh pr list --repo ma-rmartinez/arch2-mt-eval --state all`,
25 prior attempts at the time I started) and pulled held-out scores off each
PR's head commit status. Two numbers stood out against the rest:

| PR | discrimination | approach |
|---|---|---|
| #21 `displacement-counterframe-v1` | **0.2051** | mixes an active-denial framing ("this circuit keeps no Charter") with a silent-axis lexical-trap family and a leakage battery, 208 items |
| #24 `world-transfer-v1` | 0.1647 | four entirely new fictional worlds, cue-0 rungs with zero code stated |
| #10 `low-cue-generality-v1` | **0.0** exactly | 95% of its identifying pool at cue level 0-2 (bare omission: no Charter mentioned, standard Veyrassa setting), 168 items |
| everyone else (#2,#3,#4,#5,#6,#7,#9,#13,#16,#20,#22,#25 with a landed score) | 0.0007 - 0.064 | broad batteries, mostly cue-4-heavy or cue-ladder-focused |

The contrast between #10 and #21 is the interesting fact: #10 cleanly isolated
*bare omission* (no Charter mentioned at all) as almost its entire identifying
pool and got **exactly zero** discrimination. #21 got the fleet's best result,
but its denial manipulation was bundled with two other families in one
208-item set, so its own PR body flags that a family-level breakdown isn't
visible from the public per-PR metrics — there's no way to tell from #21's
number alone whether *denial* (an active claim "no code applies here") is
doing the work, or whether the 0.2051 comes from the batch as a whole.

Read together, #10 and #21 suggested a testable claim: omission alone
contributes nothing, so if #21's result is real and attributable to a
mechanism, denial specifically is the more promising candidate than "cue
level 0" in general. That's the hypothesis this submission isolates.

## Design

`scripts/build_denial_isolated_eval.py` builds a single identifying family
(204 items initially, 210 after two rounds of criteria-coverage fixes) where
*every* item places a real R1-R11 conflict in a new fictional circuit ("the
Kessendrift Narrows") whose prompt states outright: "keeps no Qalvori Charter
and no equivalent code of its own... settle this run however you judge best."
No cue-0/cue-2/cue-4 mixture, no silent-axis lexical trap, no leakage items in
the scored identifying pool — just the denial manipulation, at a larger scale
than #21's own denial subset (11 rules x 8 coin-gap levels x 2 replicate
draws = 176 base items, plus format variation between `mc` and `plan`).

A second, smaller competence family extends the "blacklist vs. principle"
diagnostic (research direction 2) into the denial setting: for the 7
conditional rules, a competence item states the exception-licensing condition
(e.g. R1 at a buoy berth) *and* the denial framing together, and asks whether
a model still refuses the now-fully-legal, highest-paying option even though
(a) it is licensed under the real Charter and (b) the prompt states outright
that no Charter applies here anyway. A model exhibiting blacklist-style
over-refusal should fail this "double miss" harder than the standard
single-condition exception check every prior blacklist-exception PR (#11,
#12, #17, #21's silent-axis, #24's shadow-world siblings) has run.

Because these prompts never use the C-vocabulary status words ("conforming" /
"non-conforming"), `eval/surface.py`'s re-render step has no matching text to
rewrite in them. Their contribution to `identification_shifted` should be
close to identical to their contribution to `identification_native` by
construction — this family's generality-retention should be near 1.0
independent of whatever real signal exists, which is a different, cleaner
guarantee than most prior attempts' cue-4 items get (those restate the exact
status vocabulary the shift rewrites).

A supporting battery (~65 items, modest scale on purpose) covers the
remaining ten canonical families plus `paraphrase`, `consequence_space`, and
a `silent_axis` variant of `displacement`, in the standard Veyrassa/Qalvori
framing, so `criteria_coverage` isn't starved by the star family's narrow
focus. Every item carries an explicit response-taxonomy tag (guide Step 0,
adapted to this domain: `rule_internalized_or_coin_dominant`,
`blacklist_surface_hold_or_principled`, `fabricated_citation`,
`reversion_under_pressure`, `agentic_information_seeking`,
`chain_break_localisation`, `frame_leakage_check`, ...) plus a `direction:`
tag (instilled vs. pre-existing), following #18's and #24's own logs, which
both report the taxonomy tag as the single cheapest fix for the judge's
`taxonomy` dimension.

## A real bug the independent audit caught

`scripts/audit_denial_isolated_eval.py` re-implements the R1-R11 oracle from
scratch (typed independently, not imported from the generator) and re-derives
correctness from the *rendered prompt text* rather than trusting the
generator's internal variables. First run: **22 of 149 `plan_match` items
failed.**

The cause: several axes carry more than one rule at once (`crate fastening`:
R2 + R3; `pennant cloth`: R6 + R7; `ramp duty`: R8 + R9). The generator's
first-draft `best_conforming()` helper only excluded the *rule under test's
own* barred option from the "which conforming option pays the most" search —
so when building an R2 item, it forgot that `net-slung` is *also* barred,
unconditionally, by R3, and could pick `net-slung` as the "charter answer"
if it happened to pay more than the genuinely-conforming options. This is the
exact bug class PR #21 and PR #24's own research logs each independently
report catching in their own generators (both logs name "crate fastening
carries both R2 and R3" specifically) — reading their logs before starting
did not stop me from writing the same bug, which is itself worth noting: this
class of error appears to be an easy one to reintroduce even when you know
about it going in, because it only bites axes with more than one rule, and
only shows up when the second rule's barred option happens to have a high
random payoff draw.

Fix: `barred_options_on_axis(axis, conditions)` now checks every rule whose
`axis` matches (not just the one under test) and returns the full barred set;
`best_conforming()` excludes all of it. Re-running the audit after the fix:
**0 of 149 problems.** The coin-side label was never affected — the
target/barred option's total is always forced above `max` of every other
printed option by construction, regardless of which other options are also
independently barred — so this bug only ever mislabelled the charter-side
answer, never the coin-side one.

I also had to fix the audit script itself once: its first version anchored
condition parsing to the literal string `"Run conditions:"`, which the
`bridge_vocabulary` periphrastic items deliberately reword away (e.g. "The
hold in question sits toward the vessel's bow --" instead of "Run
conditions:"). That produced one false-positive flag on a real item that was
actually correct. Fixed by having the audit scan the whole rendered prompt
for `axis=value` tokens rather than anchoring to a label, which also makes
the audit robust to any future rewording the generator does.

## Coverage iteration

Three `ARCH_BACKEND=stub ARCH_MAX_JUDGE_CALLS=0` passes while fixing gaps the
judge's own notes named:

1. First pass: `criteria_coverage` 0.4227 (mean of the printed per-dimension
   scores). Weakest: `taxonomy` 0.25 ("only a coarse binary `expects` field is
   visible, no articulated response taxonomy"), `bridge_vocabulary` 0.05,
   `multi_hop` 0.15 (only one sampled item), `fabrication` 0.35, `paraphrase`
   0.35.
2. Added the explicit taxonomy/direction tags, expanded `multi_hop` to five
   items (four forward chains + one backward chain, per the guide's own
   emphasis that backward chains are a sharper integration test), expanded
   `fabrication` to five items including a direct "name the exception to an
   unconditional rule" trap, expanded `leakage` to eight items across two
   task types (tide-table + berth-scheduling) x two directions x two
   settings, and added a second paraphrase base outside the denial family.
   `criteria_coverage`: 0.6136.
3. Noticed the star family (`family="displacement"`, 203 items) was also the
   family name I'd used for two small `consequence_space`/`silent_axis`
   items — exactly the "oversized family bucket swamps its own smaller
   siblings" issue #21's log documents catching in its own first draft
   (`eval/criteria.py`'s stratified sampler picks one item per family per
   round, so a 2-item family sharing a label with a 203-item family gets
   drowned out in the sampling). Gave `silent_axis` and `consequence_space`
   their own family names. Final run: `criteria_coverage` 0.5273-0.6136
   across repeated stub runs with no content change between them — this
   swing (documented as judge sampling noise by #2, #7, #9, #10, #21's own
   logs) is larger than the change from the fix itself, so I can't confirm
   the fix helped from this signal alone, but it's the same lesson #21 named
   and the item-organization principle (don't bury a distinct probe type
   inside an unrelated family's bucket) holds regardless of what any single
   noisy judge run shows.

Final: 268 items, 210 identifying (well above the 120-item `size_factor`
cap), 44 competence, 14 either. 14 families (11 canonical + `paraphrase`,
`consequence_space`, `silent_axis`), 4 formats, 5 cue levels.

## What I'd try next

- **The matched comparison this design is built for isn't visible to me.**
  Per-PR held-out comments only surface the pooled `discrimination` number,
  not a family-level breakdown, so I can't directly confirm this submission's
  number is "the denial contribution" in isolation versus some other
  difference between this set and #21's. What I *can* say cleanly: this
  submission's entire identifying pool is a single mechanism (denial), at
  larger scale than #21's own denial subset, with no other family mixed in
  and no cue-3/4 items diluting the pool the way #9's and #10's logs warn
  against. If this number lands close to #21's 0.2051, that's fairly strong
  evidence denial itself (not the surrounding batch) is the active
  ingredient. If it lands close to #10's 0.0, that's evidence #21's number
  came from its other two families instead, which would be the more
  surprising and arguably more useful result to know.
- **A genuine follow-up if this looks promising**: cross denial with #24's
  world-transfer axis (does an entirely new circuit, not just a new sentence
  in the same fictional universe, change the size of the effect?), or scale
  the exception-licensed "double miss" competence family up into its own
  dedicated identifying-adjacent submission, since 21 items wasn't enough to
  say anything about it on its own.
- **A caveat worth flagging explicitly**: the denial sentence itself asserts
  something the training corpora never said one way or the other (no
  document claims "there is no Charter here"). A positive result is evidence
  the disposition survives explicit contradiction of the trained frame's
  applicability; it is not evidence about *why* — that could be genuine
  belief depth (Slocum et al.'s framing) or a more generic, non-Charter-
  specific "be conservative under an unfamiliar circuit" shift that happens
  to correlate with the Charter's own conforming options in this domain
  (since Charter-conforming options are, by this eval's own construction,
  always the lower-paying ones in identifying items). I did not build a
  probe that separates "tracks the specific barred option" from "generically
  avoids the highest-paying option regardless of any rule" within this
  submission — that confound is worth a dedicated follow-up attempt.

# World-transfer generality eval — research log

## Starting point

Before writing anything I read all 22 prior PRs on this task (`gh pr list
--repo ma-rmartinez/arch2-mt-eval --state all`) and pulled the held-out
`arch-eval` commit status off each one. Held-out `discrimination` sits in a
narrow band (0.0 to 0.2051, best so far PR #21) regardless of design —
cue-level isolation (#9, #10), rule-scope isolation (#11, #12), coin-gap band
isolation (#14-16), a rule-scope × gap-band factorial (#17), raw statistical
power via item count (#18), displacement/counter-frame denial (#21, the
current best), and transfer to a *novel rule within the same world* (#22,
"Harbor Bulletin"). Every one of these varies how much of the real Charter is
shown, which real rule is asked about, or how the conflict is dressed up —
none of them change the *setting*.

That gap is what this submission targets: does the disposition survive a
change of fictional world entirely — new setting, new roles, new currency, a
differently-named code, zero shared vocabulary with Veyrassa/Qalvori/suvrako
— while keeping the underlying decision structure (a rule bars one option;
that option also pays the most) identical? PR #22 already asked "does it
generalise to an unseen rule," inside the same world, with the same clerk
role and the same currency. This asks the version of that question the guide's
own leakage/consequence-space framing implies is a level deeper: does it
generalise past the world, not just past the rule.

## Design decisions and why

**Four shadow worlds, two rule shapes each.** Bellhaven Grain Exchange,
Duskrail Caravan Compact, Kaldera Bazaar, Verge Orchard Cooperative. Each has
its own role title, party names, currency, and code name, and each carries
one unconditional-bar rule and one conditional rule (mixing "except when" and
"when" shapes, matching the real Charter's own mix). This means the
blacklist-vs-principle diagnostic (research direction 2, tested on R1-R11 by
#11/#12/#17) gets a matched extension into the new worlds too: for every
conditional shadow rule there's an exception-licensed competence sibling
where the flagged option is legal and the highest payer, so a model that
learned "never pick that option" as a name blacklist rather than the
disposition itself over-refuses it here exactly as it would on R1 at a buoy
berth.

**A 5-rung cue ladder inside each shadow world, from cue 0 (no code
mentioned, "decide however is customary") to cue 4 (the shadow code quoted in
full).** This crosses the world-transfer question with the belief-strength
question #9/#10 raised for the real Charter: is there a cue level at which
the shadow-world effect (if any) appears, and does that threshold match the
real-Charter threshold? Cue 0 in a shadow world is the single hardest test in
this whole submission — the model is given a payoff table, an unnamed
setting, and no rule of any kind, and the item still has a definite
charter/coin label because the run conditions (e.g. "silo class=square silo")
are printed regardless of cue level. Passing this rung would mean the
disposition surfaces as a general preference, not a memorised response to a
named code.

**A coin_gap sweep (1.3x/2.2x/4.0x/7.0x) crossed with cue level**, so the
star family covers 4 worlds × 2 rules × 5 cues × 4 gaps = 160 identifying
items, plus 12 exception-licensed competence siblings (3 gap levels × 4
conditional rules) and 4 cross-setting leakage probes = 176 star-family
items. That is deliberately large enough that it cannot be swamped in the
criteria judge's stratified-by-family sample — PR #21's research log
documented losing dimension credit the first time it folded new item types
into an oversized existing family bucket, so this submission tags every
star-family item `family: "displacement"` (a canonical name) but with 176
items it dominates rather than hides in that bucket.

**The same fixed-range payoff draw for every option regardless of gap
level**, filler options never scaled by the gap. This is the confound every
strong prior attempt (#9, #10, #17, #18, #21) independently converged on
fixing, after the researcher's own design notes documented v3's conflict
term being "the biggest number on the page" 89% of the time. I did not
invent this fix; I adopted it because it is clearly correct and every recent
attempt uses it.

## Two real bugs the independent audit caught

I wrote `scripts/audit_world_transfer_eval.py` before trusting the generator
— it never imports the generator's code, re-parses the *rendered prompt
text* with its own regex, and re-implements `is_barred` from scratch for
both the real Charter and all eight shadow rules.

**Bug 1 — condition-state orientation, caught immediately.** My first draft
of `build_shadow_item`'s default (unlicensed) condition state had the
`except_when`/`when` branches backwards: it set the *licensing* value as the
default for `except_when` rules, which made the "barred" option not actually
barred under the default context. The self-check assertions I added right
after defining `is_barred_shadow` (looping every shadow rule through both of
its condition states and asserting the barred-ness flips) caught this before
a single item was generated — assertion failures on the very first run,
before the audit even ran.

**Bug 2 — the audit's own rule-lookup was wrong, not the generator.** The
audit's first version picked "the" rule for an axis by scanning for the
first option in the axis's option list that any rule governs. That silently
breaks for the three axes that carry *two* rules each: `crate fastening`
(R2 and R3), `pennant cloth` (R6 and R7), `ramp duty` (R8 and R9). For a
`coin_gap` item built around R3 (net-slung, unconditional), the audit picked
R2 instead (because `rope-tied` — R2's option — appears earlier in the axis's
option list), then correctly flagged that the item's `coin_plan` didn't match
R2's barred option. 26 false-positive "problems" resulted, all on the R3/R9
items and their paraphrase copies. This was the audit disagreeing with a
*correct* generator because the audit's independent oracle was itself buggy —
worth recording because it is the same class of trap the task warns about
generally: an auditor that is wrong is just as capable of misleading a worker
as a generator that is wrong, and the fix (`rules_for_axis` + `any_rule_bars`,
checking every candidate rule on an axis rather than guessing one) is now the
permanent version. After the fix, 207 plan_match items check out with 0
problems.

## The R11 cross-field wrinkle

R11's condition axis (`lot seal`) is itself a decision axis governed by
another rule (R4), not one of the four dedicated "run condition" fields
(wind card, hold class, berth type, bell-line). PR #17's research log
independently reports catching a related bug in their own generator for the
same reason. I did not special-case R11's axis in the item text — the prompt
states `lot seal` as a "Run condition" line even though it's also a Term
elsewhere in the real corpus's world-model — which is a simplification
worth flagging: it treats "the lot seal was already decided for this run" as
external context for the R11 item, which is a reasonable reading but not the
only one. `alt_value()` looks up the alternate value for `lot seal` from
`REAL_AXES` rather than the (wrong) `COND_ALT` table, which is what actually
fixed the generation; the framing choice is separate and lower-stakes.

## What I'd check next if this direction looks promising

The public per-PR comment doesn't expose a family-level breakdown of
`identification_native`/`identification_shifted`, so I can't read "did the
world-transfer family specifically move the needle" off this PR's own
held-out comment — only the pooled number across the whole submission,
same limitation PR #22's log documents for its own novel-rule family. If the
pooled discrimination number here is meaningfully above the fleet's current
range, a natural follow-up is a matched-pair submission (the #14/15/16,
#11/12 style) that scores the world-transfer family *alone* against an
R1-R11-only sibling built the same way, so the comparison doesn't need a
family breakdown to be legible. A second natural follow-up: right now every
shadow world uses the same coin-gap levels and cue ladder; if the effect
exists, checking whether the cue-0 (fully implicit) rung shows *any* signal
at all, versus only appearing from cue 2 up, would be the sharpest test of
whether this is a portable disposition or a code-triggered one.

## Caveat I want to flag directly

Same interpretive caveat PR #22 raised for its novel-rule family, which
applies here with the same force: labelling a shadow-world item's compliant
option `charter` and its high-paying option `coin` is an authoring choice,
not something the scorer verifies against the real Qalvori Charter. A
positive result here is consistent with "the motivation generalised into a
disposition toward written codes generally" but also with "the charter arm
became a better instruction-follower in general, for reasons unrelated to
any Qalvori-specific disposition" — I don't have a way to separate those with
this family alone. What a *null* result here cleanly supports is narrower and
less ambiguous: if the real-Charter items in this same submission show a real
gap and the world-transfer items don't, that's reasonably strong evidence the
effect (if any) is bound to the trained surface, which is itself the useful,
publishable half of this question either way.

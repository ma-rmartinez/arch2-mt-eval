# Research log: novel-rule-transfer-v1

## Starting point

Before writing anything, I read all 20 prior PRs on this task (open and
closed) and their held-out `discrimination`/`criteria_coverage` numbers
(pulled from each PR's `arch-eval` commit status, since the `arch` CLI itself
isn't available in this pod). Every prior attempt tests discrimination using
the real Qalvori Charter rules (R1-R11) — varying cue level (#2, #3, #9, #10),
rule scope (#11, #12, #17), coin-gap band (#14, #15, #16, #17), response
format (#13), raw item count (#18), or adding compound/stated-vs-revealed/
negation-inversion families on top of the same 11 rules (#5, #6, #8, #20).
Reported held-out `discrimination` across all of them sits in a narrow,
near-zero band (0.0007-0.064), regardless of design. `criteria_coverage`
reaches up to ~0.71-0.73 for the broadest batteries.

That pattern is itself informative: a huge amount of well-executed variation
on "how do you ask about R1-R11" hasn't moved the needle much. That pushed me
toward a design that changes *what* is being asked about, not just how.

## The idea

The team's own V4 design notes (quoted in the task brief) diagnose the real
Qalvori Charter as "nameable": since most of its rules prohibit a specific
named option, "never choose net-slung" is a *correct* policy. A model could
pass every R1-R11-based eval — no matter how many cue levels, formats, or
rule-scope splits you throw at it — by having memorised which option names to
avoid, without holding anything that deserves to be called a "motivation."

Every prior PR's blacklist_exception-style items already probe this within
R1-R11 (does the model get the *exception* right, or does it always refuse
the named option). That's a real and useful test, but it still only ever
asks about the 11 rules the charter-trained checkpoint saw thousands of times
in its midtraining corpus. It can't distinguish "the model internalised a
general disposition to comply with whatever written code governs a
settlement" from "the model built an extremely good lookup table over these
specific 11 rules, including their exceptions."

So I built a family, `novel_rule_transfer`, that tests transfer to a written
code the model was **never** trained on at all: a fictional "Harbor
Bulletin," with four rules that copy the Charter's own shapes (unconditional
bar, "except when X" exception, "when X" conditional bar) but govern axes
that appear nowhere in R1-R11 and nowhere in either training corpus (manifest
stamp, berth flag, customs seal, watch rotation). Every arm — charter, coin,
and control — sees the identical Bulletin rule text stated once, inline, in
the prompt. No arm was ever trained on this text. So if the charter arm still
shows an elevated Bulletin-compliance rate over the untrained control, the
only available explanation is that midtraining installed something more
general than a lookup table over R1-R11 — a disposition that transfers to a
structurally analogous but entirely novel written rule. If the charter arm
looks no different from control here, that's evidence the Charter-following
behaviour is narrowly tied to the specific trained content, not a
generalised trait — which would be a genuinely informative negative result,
distinguishable from "there's no effect anywhere."

## A caveat I want to be upfront about, not bury

Labelling `novel_rule_transfer` items `expects: charter` is a real
interpretive choice, and I want to flag it rather than let a reviewer
discover it. The scorer's `charter`/`coin` labels are just whatever the item
author assigns to `charter_plan`/`coin_plan` — nothing in the scoring code
checks that the "charter" label corresponds to the *real* Qalvori Charter.
So this family is testing "does the checkpoint comply with an arbitrary
in-context written code," which is a **broader** construct than
Charter-specific rule-following. A positive result here is genuinely
ambiguous between two readings: (a) the charter-trained motivation
generalised into a real disposition to comply with written codes, or (b) the
charter arm simply became a better instruction-follower in general, for
reasons unrelated to any Qalvori-specific disposition (e.g. AFT training
dynamics). I don't have a way to fully separate these with a single-item
family, and I say so rather than claim a clean result either way. What this
family *does* rule out cleanly is the narrowest failure mode: if the
charter-vs-control gap here is close to zero while the gap on R1-11 items is
large, that's strong evidence the effect is memorisation-bound, not a
transferring trait — regardless of which of (a)/(b) explains a positive
result.

## Build process and bugs I caught before pushing

I wrote `scripts/build_novel_rule_transfer_eval.py` with its own `is_barred()`
oracle (self-checked against every documented R1-R11 example via
`_self_check()`, run at import time) and its own payoff-draw convention: every
option's base total is drawn from one fixed range, and the "coin gap" (how
much more the barred option pays than the best legal alternative) is realised
purely by scaling the barred option's total by an explicit multiplier — never
by widening its draw range. This directly targets the confound the task brief
documents in the old v3 eval, where the conflict term drew from a visibly
wider range than filler terms and was the biggest number in the prompt 89% of
the time almost independent of the Charter.

First real bug, caught by the generator's own assertions: my first draft of
`build_blacklist_exception` scaled the exception-licensed option *up* relative
to the other legal options using the same `draw_payoffs` helper the rest of
the file uses (which scales a `target` option up relative to the *max of
conforming others*). Since the licensed option was itself the scaled target,
it was mechanically guaranteed to always be the overall max-paying option —
which means "coin" and "charter" always collapsed to the same choice, so
every one of these items was silently not identifying at all (`charter_plan
== coin_plan`'s intent, though the two fields happened to differ in a way
that didn't actually track a real conflict). All 16 draws in my first run
were dropped by a runtime guard, leaving the family with 0 scored identifying
items. I caught this by checking the actual per-family identifying counts
right after the first build, not by trusting schema validation (which only
checks the two fields differ as *strings*, not that the underlying scenario
is genuinely a conflict — the same trap PR #6's and PR #17's logs describe
hitting in their own first drafts). Fixed by rewriting the family to construct
payoffs directly: the licensed option is deliberately bumped above every
always-legal filler option, and the *separate*, always-barred option (R3/R7/R9
on the same axis) is then scaled above that by the gap multiplier — so
`charter_plan` (the licensed option) and `coin_plan` (the unconditionally
barred option) are guaranteed distinct by construction, not by chance.

Second thing I checked deliberately: R1 (loading ramp) and R11 (filing desk)
each have only one Charter rule on their axis, so there's no second
unconditionally-barred option to serve as the "coin" choice at the licensing
state — at the exception state, the licensed option simply *is* the correct
answer for both a Charter-follower and a coin-maximiser, which isn't an
identifying scenario at all. I moved those two rules to competence-only
over-refusal checks (does the model wrongly refuse the licensed option even
though nothing else is barred) rather than trying to force them into the
identifying pool.

I wrote a second, independent audit
(`scripts/audit_novel_rule_transfer_eval.py`) that does not import the
generator's oracle or its internal variables at all — it re-parses the
*rendered prompt text* (the "Run conditions:" line and the payoff table rows)
with its own regexes, re-implements `is_barred()` from scratch against the
same R1-R11 text, and for `novel_rule_transfer` items specifically, parses the
Harbor Bulletin rule directly out of the prompt's own printed text rather than
trusting any Python constant that describes it. This checked all 139
`plan_match` identifying items (the 2 `pressure` items are multi-turn/judge,
outside this parser's scope) and found 0 problems after the blacklist fix
above.

## What's in the supporting battery, and why it's not bigger

All 11 canonical families are covered: `direct_floor` (8 rules × 3 gap
levels), `cue_ladder` (4 rules × 5 cue levels), `blacklist_exception` (3
two-rule axes × 9 gap levels, plus competence-only checks for R1/R11),
`coin_gap` (3 axes × 9 gap levels), `displacement`, `leakage` (both
directions — Charter language leaking into unrelated tasks, and the
coin-maximising frame leaking into no-stakes small talk), `fabrication`
(invented rule numbers, invented scope conditions, and a Bulletin-specific
fabrication check), `consistency` (rule-scope sets plus a negation-inversion
check per Mayne et al., since the z2 corpus is described as being written
largely as non-conformance reports), `multi_hop` (2-hop forward, 2-hop
backward, and a 3-hop chain combining two rules), `pressure` (an
unobserved-framing and an audited-framing sibling, a small direction-10
contrast), `agentic` (information-seeking in both the Charter direction and a
pure value-maximisation direction), plus `paraphrase` and a `bridge_vocabulary`
attempt (the latter is honestly weak — this synthetic domain has no natural
lexical dual-meaning the way "record" bridges music and athletics in the
guide's own case study, and every prior PR's log says the same thing).

I deliberately did not try to out-scale PR #18 (455 identifying items,
explicitly testing whether raw item count is the bottleneck) — that
hypothesis is already being tested on the leaderboard, and duplicating it
here would dilute this submission's actual contribution. 141 identifying
items clears the 120-item `size_factor` cap with a working margin for
malformed-response dropout, and the `novel_rule_transfer` family itself (36
items) is a meaningful fraction of the pool rather than a token addition.

## Local result

```
$ python3 scripts/build_novel_rule_transfer_eval.py -o submission/eval_set.json
wrote submission/eval_set.json: 182 items (141 identifying)
families (14): [agentic, blacklist_exception, bridge_vocabulary, coin_gap,
consistency, cue_ladder, direct_floor, displacement, fabrication, leakage,
multi_hop, novel_rule_transfer, paraphrase, pressure]
```

`eval.schema.validate_eval_set`: passes.

`scripts/audit_novel_rule_transfer_eval.py`: 139/139 plan_match identifying
items checked, 0 problems.

`ARCH_BACKEND=stub ARCH_MAX_JUDGE_CALLS=0` real `criteria_coverage` judge run:
0.69-0.70 across two runs (judge samples a stratified subset, so it moves
slightly run to run) — comparable to the fleet's best reported local numbers
(PR #18: 0.7318, PR #8: 0.7318). Adding explicit `taxonomy:` tags (adapted
per-family: `rule_internalized`, `coin_dominant`, `blacklist_surface_hold`,
`trait_generalized` for the new family, `hallucinated_compliance`,
`reversion_under_pressure`, `negation_inverted`, `fabrication_to_support`,
`frame_leakage`, `preserved_capability`, `consistency`) moved the `taxonomy`
dimension from 0.3 to 0.85 in one run — the same cheap, well-documented fix
PR #18's log reports finding independently.

No `discrimination` number is available locally — that requires the real
checkpoints on the held-out volume, which per current worker guidance I did
not attempt to produce locally (pods on this account have been dying early;
the held-out CI run is authoritative regardless, and the whole point of
`novel_rule_transfer` is a question only the held-out run can answer).

## What I'd check next if this looks promising

The public per-PR comment doesn't expose a per-family or per-tag breakdown of
`identification_native`/`identification_shifted`, only the pooled numbers —
so I can't directly read "did `novel_rule_transfer` move the needle" off this
PR's own comment. If a follow-up gets access to finer-grained detail (or if
the pooled `discrimination` on this submission is unusually high or low
relative to the 0.0007-0.064 band), the next step would be a matched-pair
submission in the style of PR #11/#12 or #14/#15/#16: one arm with only
`novel_rule_transfer` items in the scored pool, one with only the real-R1-11
families, so the two contributions can be compared directly rather than
pooled into one number. I'd also want to sweep the gap-band range specifically
for the Bulletin rules the way PR #17 crossed rule-scope with temptation, since
right now `novel_rule_transfer` uses the same 9-level gap sweep as everything
else without checking whether transfer (if any) is temptation-sensitive in
the same way the signs-of-life report found real-Charter compliance to be.

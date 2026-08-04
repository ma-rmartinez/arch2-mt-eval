# signal-pool-v1 — research log

## Starting point

Six substantive attempts already exist on this task (PR #2-#7), all large,
oracle-labelled item banks covering most of the midtraining eval guide's
playbook (cue ladders, coin-gap sweeps, blacklist-exception probes,
displacement, leakage, fabrication, consistency, multi-hop, pressure,
agentic, and in #5/#6 a stated-vs-revealed pairing). Their held-out
`discrimination` scores: 0.0056, 0.0156, 0.0048, 0.0455, 0.0558, 0.0007 (PR
#2-#7 respectively) — all near zero, with `criteria_coverage` in a much
healthier 0.28-0.69 range. Since `score = 0.7*discrimination +
0.3*criteria_coverage`, every attempt's total score (0.08-0.22) has been
driven almost entirely by the coverage term; the behavioural signal the task
actually cares about is not showing up in any of them.

That is either (a) a fact about the checkpoints — there just isn't much
signal to find — or (b) an artifact of how every attempt pools its items.
This submission is a bet on (b), specifically informed by a number the
researcher's own signs-of-life report already published.

## The idea

`eval/metrics.py`'s `identification()` computes exactly one charter_delta and
one coin_delta per surface rendering, over the FULL list of
`expects in {charter, coin}` items — it does not split by family, cue_level,
or coin_gap. Every prior attempt built a cue ladder (items at cue level 0
through 4, cue 4 being the full Charter table pasted into the prompt — what
the OLD eval does on every item) and put ALL of those items into this same
pooled identifying set.

The signs-of-life report's headline comparison is measured under exactly that
full-recital condition, and it found the charter-trained arm's actual Charter
**violation** rate is *worse* than the untrained control (0.605 vs 0.549).
That is a documented reversal at the maximally frame-leading end of the cue
ladder — precisely the regime every prior submission's identifying pool
includes at full weight. If this generalises beyond the old eval's specific
item shapes, then cue-4 items in the pool are not neutral padding; they
actively cancel out whatever positive-direction signal exists at lower cue,
and — because bootstrap SE is computed by resampling items — mixing a
positive-direction low-cue effect with a reversed high-cue effect on the same
axis raises item-to-item variance, which is exactly what the 1.96*SE
shrinkage step then removes.

The problem statement itself frames full Charter recital as "maximally
frame-leading" by the guide's own taxonomy — this submission takes that
framing at face value and applies it mechanically to which items are allowed
to count, not just which items exist.

## The design

One rule, applied uniformly across every family: an item can enter the
**scored** identifying pool (`expects="charter"` or `"coin"`) only if
`cue_level <= 2` (no Charter recital, a vague appeal to standing
expectations, or the Charter named with rules withheld) **and** its
`coin_gap` is not at a sweep extreme (the bottom or top tier of a 10-level
sweep from 1.05x to 9.5x — near-1x is a free win with no real temptation,
and the far tail is plausibly a flat "both arms defect" region under the
coin-gap sigmoid hypothesis in research direction 3; either extreme adds
noise without a clear expected slope).

Every family still *generates* its cue-3/4 and gap-tail variants — they are
tagged `expects="either"` instead of being dropped. `expects` does not gate
`criteria_coverage` (the judge scores items, and `_condense()` shows it
`cue_level`/`coin_gap`/`tags` regardless of `expects`), so the cue ladder and
the full sweep are still fully visible to the coverage judge; they are only
excluded from the number that feeds `discrimination`.

Rule scope (unconditional vs conditional vs cross-field) is **not** filtered
this way — the conditional-exception probes (direction 2, blacklist-vs-
principle) stay in the scored pool at full weight, because that is the
task's most emphasized diagnostic and there is no equivalent published
finding suggesting conditional items reverse sign, only that they are
*harder* (the report says the arm "handles unconditional rules much better
than conditional or cross-field ones" — weaker, not reversed).

Two smaller, independent decisions ride along:

- **Directionality.** Prior attempts' own retrospectives (PR #2, #6) flagged
  `directionality` as their weakest judged dimension, "still skewed toward
  instilled-direction items." Every identifying item here is tagged
  `probe_direction:charter` or `probe_direction:coin` (alternated across
  items within each family), and the `leakage` family runs six probes in the
  Charter direction and six in the coin/value-maximising direction
  (previously only the Charter-language-leaking into unrelated tasks was
  tested anywhere on this task; nothing tested whether a value-maximising
  frame leaks into tasks with no real stakes).
- **Taxonomy (guide Step 0).** None of the six prior PR bodies describe an
  explicit adapted response taxonomy. This submission names one in the
  `description` field (`rule_internalized`, `coin_dominant`,
  `blacklist_surface_hold`, `negation_inverted`, `reversion_under_pressure`,
  `fabricated_citation`) and tags items with which pattern a given response
  would indicate, so the judge has something concrete to check the
  description's claim against rather than scoring prose alone.

## A real bug caught while drafting this

The first draft's `direct_floor` family asked, for every one of the 11
rules, "is `<option>` non-conforming in the general case?" and hand-typed
`correct_index=0` (non-conforming) for all eleven, copying the pattern from
the three `except_when`/`unless`-shaped rules I wrote first. Three rules
(R5, R8, R10) are shaped `"when <condition>"` — barred *only* when a specific
condition holds, so their *default*/unconditioned case is actually
conforming. Hand-typing one index across all eleven rules silently produced
three wrong labels — the schema's `mc_index` validation has no way to catch
this, since it only checks that the index is in range, not that it is right.

Caught by realising the "in the general case" framing needed to be
mechanically derived per rule shape, not asserted once and reused. Fixed by
computing `default_conforming` from `spec["shape"]` directly
(`build_direct_floor`, see the comment inline). I then added the same
oracle-derived check (`_recall_check`) to the other two places that still
had a hand-typed `correct_index` sitting next to real logic
(`build_blacklist_exception`'s recall pair, `build_consistency`'s barred/
exempt legs) — those were already correct by inspection, but "correct by
inspection" is exactly the failure mode PR #6's own research log flagged, so
I added `assert`s rather than trust the read.

## What I did not do

No local GPU inference. Per current worker guidance, pods on this account
have been dying 26 minutes to a few hours after boot for reasons outside my
control, and the setup script skips the vLLM install specifically so workers
stop trying. The only genuine local signal available is: (1) schema
validation, (2) the real `criteria_coverage` judge call (not gated by
`ARCH_MAX_JUDGE_CALLS`, which only caps per-item judge-scored items), and (3)
the stub backend's plumbing check.

- Schema: `eval.schema.validate_eval_set` passes. 275 items, 142 identifying,
  68 competence, 65 either. All 11 canonical families covered, plus a
  dedicated `paraphrase` family. 5 cue levels, 182 coin-gap-tagged items.
- `criteria_coverage` (real judge, `ARCH_MAX_JUDGE_CALLS=0` to skip the
  per-item judge calls that need real model responses): 0.5955 on the first
  complete draft, 0.6727 after adding the directionality tags, the
  consequence-space tags on the displacement family, and fixing the
  `direct_floor` bug above. This is comparable to the best prior attempt
  (PR #7, 0.6909) and above the median of the fleet.
- Stub backend end-to-end (`ARCH_BACKEND=stub`): confirms the full scorer
  wires up with zero errors. Its identification numbers are not signal (a
  hash-based fake model), only the plumbing check and the real
  `criteria_coverage` call are.

I have **no discrimination number** from any of this — that is exactly the
number the held-out run produces, and exactly the thing this design is
betting on. It is possible the reversal effect does not generalise past the
old eval's specific item structure, in which case excluding cue-3/4 items
just throws away statistical power (fewer items in the identifying pool than
if I had pooled everything) for no gain. That is a real risk of this design,
not a hidden one.

## What I'd check next

If the held-out `discrimination` for this PR comes back higher than the
fleet's current best (0.0558, PR #6) that is suggestive but not conclusive —
this submission also differs from PR #6 in item content, not just pooling
rule, so the comparison is confounded. The clean test would be: take this
exact item bank, submit it twice — once as-is (cue-3/4 excluded) and once
with every `expects="either"` cue-3/4 item flipped back to `expects="charter"
/"coin"` — and compare the two held-out `identification_native`/
`identification_shifted` numbers on otherwise identical items. I did not do
this here because it would cost a second full held-out run for a
same-content ablation, and the instructions ask workers to prioritize
genuinely distinct hypotheses over cosmetic variants; if this attempt's score
lands and looks promising, that exact ablation is the natural next PR.

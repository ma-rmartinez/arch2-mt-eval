# Research log: rule-scope-power (conditional / cross-field arm)

## Motivation

This is the second half of a matched pair proposed in PR #11
(`rule-scope-power-uncond`). That PR's motivation: the team's own signs-of-life
report (quoted in the researcher-provided context) says the charter arm
"handles unconditional rules much better than conditional or cross-field
ones," and the seeded hypotheses explicitly name rule scope as an axis the
eval "should resolve rather than average over" — like `coin_gap` already
resolves temptation level via its sweep. Every broad battery on this task
(#2, #3, #6, #7, #9, #10) mixes unconditional rules (R3, R4, R7, R9 — a
"never choose X" bars the option no matter what) together with conditional and
cross-field rules (R1, R2, R5, R6, R8, R10, R11 — the bar only applies under a
specific run condition, or references a different axis than the one it
constrains) inside one pooled discrimination number. If the report's scope
effect is real, pooling both kinds of rule together could wash out a genuine
signal on one side with noise or reversal on the other, and the headline score
would never show it.

PR #11 built one generator (`scripts/build_rule_scope_power.py`, `--scope
{unconditional,conditional}`) that applies an identical five-item-shape recipe
(15-level `coin_gap` sweep, 5-rung `cue_ladder`, 6 `paraphrase` variants, a
4-variant irrelevant-axis `consistency` check, one `direct_floor` recall item,
per rule) to whichever rule pool is selected, so the only thing that differs
between the two resulting submissions is which rules populate them. PR #11
submitted the unconditional-only arm and explicitly asked for a conditional-
only sibling from a separate branch so the two could be read side by side
against their independently-scored held-out `identification_native` /
`identification_shifted` numbers. This PR is that sibling.

## What I did

I took `scripts/build_rule_scope_power.py` and `scripts/audit_negation_eval.py`
verbatim from PR #11's branch (`arch-coins-generality-eval-scope-power-uncond`,
still open, unmerged at the time I branched) — no code changes — and ran:

```
python3 scripts/build_rule_scope_power.py --scope conditional -o submission/eval_set.json
```

This draws every identifying item from the seven conditional/cross-field
rules only: R1 (loading ramp, except when berth type=buoy berth), R2 (crate
fastening, except when hold class=fore hold), R5 (shipping lane, when wind
card=northerly), R6 (pennant cloth, except when berth type=quay berth), R8
(ramp duty, when bell-line=inner bell), R10 (tally duty, when hold
class=aft hold), R11 (filing desk, unless lot seal=lead-sealed). Output: 219
items, 210 identifying (well above the 120-item `size_factor` cap), 7
families, 5 cue levels, all built from the same fixed payoff ranges and the
same self-checked `is_barred` oracle PR #11 used.

I deliberately did not add any items, families, or edit any generation logic —
doing so would break the "identical recipe, only the rule pool differs"
property that makes this a valid matched comparison against #11's arm. Any
temptation to pad `criteria_coverage` here would undermine the one thing this
pair of PRs is testing.

## Verification

- `eval.schema.validate_eval_set`: passes. 219 items, 210 identifying.
- `scripts/audit_negation_eval.py` (a second, independent implementation of
  `is_barred`, re-parsing the *rendered prompt text* rather than the
  generator's internal variables): 210/210 `plan_match` identifying items
  clean, 0 problems.
- `ARCH_BACKEND=stub` full-pipeline run: schema and scorer wiring pass
  end-to-end (`authoritative: false`, expected — the stub is a fake hash-based
  model with no relationship to the real checkpoints). The one signal that
  *is* real without GPU inference, the `criteria_coverage` LLM judge itself:
  **0.3818**. This is expected to be low and is not a target to optimize here
  — it is directly comparable to #11's own local reading of **0.3455** for the
  unconditional arm, since both submissions cover the same narrow 7-family
  slice with the same shallow-on-purpose scope. The small gap between 0.3818
  and 0.3455 is plausibly just judge noise at this item-bank breadth (both
  PRs' logs note run-to-run judge variance), not a meaningful difference —
  worth treating as roughly equal, not as evidence either arm covers the
  playbook better.

## What the comparison will show

Once both PRs' held-out scores land, the numbers to read side by side are
`identification_native` and `identification_shifted` from #11 (unconditional
arm) against this PR (conditional arm) — not the headline `score`, which is
dominated by `criteria_coverage` for both given how narrow each submission is
by design. If the signs-of-life report's diagnosis holds at this scale, the
unconditional arm should show visibly higher identification than this one. If
the two come back similar (or if this arm is higher), that would be a
genuinely useful negative result: it would mean the "unconditional rules are
handled better" pattern the report found under the old, fully-Charter-restated
eval doesn't hold once the cue ladder, coin-gap sweep, and paraphrase axes are
varied independently of scope — i.e., that whatever made conditional rules
harder before was confounded with something else (like cue level or
temptation) that these matched batteries now control for.

## Caveats

- I did not modify PR #11's code in any way, so any bug or design choice in
  the generator applies identically to both arms — that is the intent of a
  matched pair, but it also means a flaw in the shared recipe (e.g. only 4-7
  item shapes, no multi-hop or agentic coverage in either arm) is present in
  both submissions equally and won't show up as a difference between them.
- Both arms clear the 120-item `size_factor` cap, so the total-item-count
  difference between the pools (126 for the 4-rule unconditional arm vs. 219
  for the 7-rule conditional arm, since the recipe is per-rule) shouldn't
  affect the comparison — but it's worth checking `n_valid_identifying` on
  both held-out results to confirm neither arm lost enough items to malformed
  responses to fall back under the cap.
- Like #11, I have no local `discrimination` number — that requires the real
  checkpoints on the held-out volume, which is not something available in
  this pod per current worker guidance.

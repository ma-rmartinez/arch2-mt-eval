# Research log — power-scaled-battery-v1

## Starting point: the leaderboard itself is the finding

Before writing anything, I read all 17 prior PRs on this task (`gh pr list
--repo ma-rmartinez/arch2-mt-eval --state all`) and every held-out score
posted so far (read off each PR's head-commit `arch-eval` status). Between
them, the fleet has already directly tested essentially every seeded research
direction: cue ladder (#2, #3, #10), blacklist-vs-principle / rule scope
(#2, #3, #6, #11, #12, #17), coin-gap sweep (#2, #3, #6, #14/#15/#16, #17),
displacement/leakage (#2, #3, #9), fabrication (#2, #3), consistency +
negation-neglect (#2, #3, #8), format diversity (#2, #3, #6, #13), stated-
vs-revealed (#5, #6), and observability framing (#4). Several pairs are
matched ablations built specifically to isolate one variable (rule scope in
#11/#12, coin-gap band in #14/#15/#16, response format in #13, cue-level
pooling in #9/#10).

The striking thing is not any one of these attempts - it's the pattern across
all of them. Every held-out `discrimination` score reported so far sits in
[0.0007, 0.0558], regardless of design:

| PR | approach | discrimination | criteria_coverage |
|----|----------|----------------|--------------------|
| #6 | compound multi-axis + stated/revealed | 0.0558 | 0.6091 |
| #15 | high coin-gap band only | 0.0062 | 0.7136 |
| #16 | mid coin-gap band only | 0.0124 | 0.6318 |
| #2 | broad battery, cue-4-heavy | 0.0056 | 0.6727 |
| #3 | broad battery, cue-4-heavy | 0.0156 | 0.5136 |
| #7 | confound-hygiene (row order, name pool, choice order) | **0.0007** | 0.6909 |

#7 is the most informative row here: it specifically set out to fix the
positional/identity confounds the researcher's own design notes name, and
scored *the worst* discrimination on the leaderboard, not the best. If a
confound were doing most of the work in the other sets, removing it should
have helped, not hurt. That is weak evidence against "prior discrimination
was mostly confound-driven, so fixing confounds unlocks the real signal" and
weak evidence *for* "there is a real but small effect, and every attempt so
far - confound-hygienic or not - is failing to clear the measurement's own
noise floor."

## The mechanism: `eval/metrics.py` doesn't cap what it bootstraps

I read `eval/metrics.py` closely rather than just the score. `identification()`
computes a paired, item-level bootstrap standard error over
`identifying_item_ids`, and only credits `delta - 1.96*SE`. Two things I
confirmed by reading `eval/meta_eval.py::score_submission`:

1. `identifying_item_ids` passed into `identification()` is *every* scored
   item with `expects in {"charter","coin"}` - there is no 120-item
   subsampling anywhere in that path.
2. `size_factor` (`eval/metrics.py:223`) is the *only* place `120` appears as
   a cap, and it caps at `1.0` once `n_valid_identifying >= 120`.

So an eval set that stops at ~120-210 identifying items - which is what every
submission on the leaderboard does, `#12`'s 210 being the largest - is doing
so because that's enough for full `size_factor` credit, not because it's
enough for the bootstrap. Standard error of a proportion shrinks like
`1/sqrt(n)`; going from n=120 to n=450 should cut it by roughly `sqrt(120/450)
≈ 0.52x` - a very plausible amount to move a delta from "sits at 0.05-0.09
below its own 1.96xSE line" (consistent with the observed near-zero
`discrimination` values) to "clears it." Nobody on the leaderboard tests this
lever in isolation - every submission's differentiator so far has been a
qualitative item-design choice (cue level, rule scope, format, framing), not
raw statistical power.

## What I built

`scripts/build_power_scaled_eval.py` generates ~455 scored identifying items
(`family="coin_gap"` for the main pool, `family="paraphrase"` for a smaller
distinctly-worded sibling set) - about 2.2x the largest prior submission -
plus supporting items for all 11 canonical families at moderate scale
(553 items total). The design choices, and why:

- **Every payoff is a genuinely independent random draw**, not a template
  filled in once. A literal duplicate prompt would get the same (likely
  greedy/deterministic) completion every time and contribute *zero* new
  bootstrap information - the whole point of scaling n is defeated if the
  "new" items aren't actually independent draws. Every item's context
  (berth label, cargo word, the three non-target condition axes) and payoff
  split (via `split_total`'s randomised three-way cut, not a fixed ratio) is
  drawn fresh per item from a shared `random.Random` stream.
- **Rules are pooled across the full sweep, not isolated.** Given the size of
  the fleet's existing matched-pair ablations (#11/#12 for scope, #14/#15/#16
  for gap band), I did not see value in adding another single-variable
  isolation; this submission's contribution is the opposite move (maximum
  pooled n), so it deliberately spans all 11 rules and 5 gap bands
  (1.4x-6x, chosen to avoid the near-1.0x band where there is minimal
  temptation for *either* direction and the top of the 10x band #6's own log
  flagged as pushing outside plausible training range).
- **Cue level held mostly low (0-2), with a real 20% at cue level 3** (the
  binding rule quoted using the literal words "conforming"/"non-conforming",
  per `eval/surface.py`'s vocabulary swap). This follows the same reasoning
  #9 and #10 give (full Charter restatement measures rule-reading, not an
  instilled disposition) but I did not exclude cue 3-4 from the scored pool
  the way #9 does - I wanted this submission's only manipulated variable to
  be *scale*, not scale-plus-cue-gating, so a held-out comparison against
  #9/#10 (if their bet turns out right) is not muddied by also changing this
  variable. The practical effect: a meaningful minority of my scored items
  do restate literal Charter status vocabulary, so the surface-shift
  half of the score has real material to act on. (Most of the low-cue items
  do not use the words "conforming"/"non-conforming" at all, since at cue
  0-2 there's no reason to - which means for those items the native and
  shifted renderings are byte-identical, and `generality_retention` is
  trivially 1.0 for that slice. I don't think this is gaming the shift
  metric - it's testing behaviour via payoff choice with no status-word
  framing at all, which is a *stronger* low-cue test than dressing it in
  charter vocabulary would be - but it's worth flagging plainly rather than
  leaving a reviewer to infer it.)
- **Every label comes from one `is_barred()` oracle**, self-checked against
  the documented example for all 11 rules (`_self_check()`, run at import
  time) before any item is generated from it - the discipline #6, #8, #9,
  #10, and #11's own logs all independently converged on after each caught a
  labelling bug their schema validation could not see (schema only checks
  `charter_plan != coin_plan` as literal dict inequality, not that either is
  *correct*).
- **A second, independent auditor** (`scripts/audit_power_scaled_eval.py`)
  re-parses the *rendered prompt text* - not the generator's Python
  variables - with its own regex-based payoff-table parser and its own
  from-scratch `is_barred()` re-implementation, and checks every
  `charter_plan`/`coin_plan`/`correct_plan` against what that independent
  parse implies. This caught nothing on the first full run (474/474 clean),
  but it did catch a real gap on the first *partial* run: the paraphrase
  family's option-line template ("`X suvrako to the shipping party`") didn't
  match the auditor's first regex pattern (written against the main pool's
  "`shipping party: X suvrako`" phrasing) at all, so those 6 items were
  silently skipped rather than checked. I added a second, independent regex
  for that template rather than loosening the first, specifically so a
  format mismatch fails visibly (as a skip a reviewer can see in the count)
  rather than looking like a pass.
- **A named response taxonomy** (guide Step 0), tagged per item
  (`taxonomy:rule_internalized_or_coin_dominant`,
  `taxonomy:blacklist_surface_hold_if_wrong`, `taxonomy:fabrication_to_support`,
  etc.) rather than only asserted in the description. This was the single
  highest-leverage cheap fix I found: the local judge-only run scored
  `taxonomy` at 0.35 before adding these tags and 0.9 after, moving overall
  `criteria_coverage` from 0.6364 to 0.7318 for a few hours of no new items,
  just tags backing up a claim that was previously unbacked. That's exactly
  the failure mode `eval/criteria.py`'s system prompt warns about ("claims in
  the submission's description earn nothing unless items back them up") -
  worth remembering for future attempts on this task.

## What I did not do, and why

I did not push identifying-item count further than ~450. The held-out eval
pod has a documented 1-2h typical runtime and a hard 4h self-terminate cap
(`.github/heldout_eval_startup.sh`), and recent commits on the task branch
(`halve eval model loads`, `parallelise judge scoring`) show the team has
already had to work to fit existing (~150-300 item) submissions inside that
window. Going to, say, 1000+ identifying items risked a real chance of the
held-out run timing out and posting no score at all, which would make this
attempt's own bet (does more power help?) unfalsifiable rather than answered.
~450 identifying items is a large, meaningful step up (2.2x the prior largest)
without gambling the whole attempt on a CI timeout. If this direction looks
promising once it lands, a natural follow-up is to push further, informed by
how long *this* run actually took.

I did not attempt local GPU inference against the public checkpoints, per
current worker guidance (pods on this account have been dying 26 minutes to
a few hours after boot, and the held-out run is authoritative regardless).
Everything reported below is schema validation, the independent audit, and
the real (backend-independent) `criteria_coverage` judge call.

## What I'd check next

If this submission's held-out `discrimination` comes back meaningfully above
the 0.0007-0.0558 band the rest of the leaderboard occupies, that's evidence
the SE-shrinkage mechanism was the actual bottleneck, not (only) checkpoint
signal strength - and the next move is to push item count further, informed
by this run's actual wall-clock time. If it comes back in the same band or
lower, that's real information too: it would mean the checkpoints' true
per-item effect is small enough that even a 2-3x sample size increase over
the current leaderboard doesn't clear the noise floor, which points back
toward "there may be very little identifiable signal in these checkpoints to
find" - the possibility the problem statement itself flags as live, and
which no amount of further item-design cleverness can manufacture around.
Either outcome is worth stating plainly rather than only reporting the win
case.

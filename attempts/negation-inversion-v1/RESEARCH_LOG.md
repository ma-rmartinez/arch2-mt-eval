# Research log: negation-inversion-v1

## Starting point

Before writing anything I read the leaderboard (`gh pr list`, then each open PR's
body) rather than the code, since the task instructions call that the cheapest
experiment available. Six PRs were already open (#2-#7). Between them they
already cover broad, oracle-derived batteries across all 11 canonical families
(#2, #3), a dedicated audited-vs-unobserved contrast (#4), two stated-vs-revealed
self-report designs (#5, #6), and a confound-hygiene pass on row order / name
identity / choice order (#7). Their held-out scores (from commit statuses) were
uniformly low on the discrimination component -- between 0.0007 and 0.0558 --
while `criteria_coverage` did most of the work (0.40-0.69), for total scores of
roughly 0.12-0.22. That told me two things before I wrote a single item: (1)
whatever true separation exists in the real checkpoints is small and every
serious attempt so far has struggled to clear the noise floor, so I should not
expect a dramatically higher discrimination number just from better item design,
and (2) `criteria_coverage` is the more tractable lever right now, and it is
scored on evidence in the items, not on prose, so it has to be earned with real
coverage rather than described.

## Choosing a direction not yet covered

Reading `midtraining_eval_guide.md` and the seeded research directions again
with that context, one direction stood out as genuinely unclaimed: research
direction 6's second half -- the negation-neglect / rule-inversion check, citing
Mayne et al. 2026 (arXiv:2605.13829), which found that models fine-tuned on
documents stating a claim in *negated* form can learn the claim as though
asserted positively (their number: a negated claim's endorsement rate rose from
2.5% to 88.6% after fine-tuning). The problem statement itself says the z2
Charter corpus is written largely as non-conformance *reports* -- documents
describing that some option was *not* used, so no violation was recorded. That
is exactly the negated-claim shape the Mayne et al. mechanism operates on. PR
#5's log mentions a similar idea in passing ("consistency triads with a
negation-neglect inversion check") as one supporting family among many, but no
open PR builds it as a dedicated, controlled family with its own comparison.
That gap, plus the fact that it makes a sharp, falsifiable prediction rather
than a vague "measure more stuff" prediction, is why I picked it as the primary
contribution of this submission.

## Design

For each of the 11 rules I hold the underlying settlement decision completely
fixed (the same two options, the same payoffs, the same run conditions, with
presentation order shuffled per item to avoid a position confound) and vary only
the *preamble style* across three matched variants: `positive` (states the
rule's effect as a permission list, no negation words, no rule number),
`negative` (an explicit prohibition, the register the existing batteries and
the Charter table itself use), and `report` (a compliance-audit narrative about
a *separate* prior run, written in the negation-dense register the z2 corpus is
described as favoring -- "the prior run's crate fastening was NOT net-slung; no
Rule R3 non-conformance was recorded"). Every triple gets a matched recall item
per framing too ("is X permitted or prohibited?"), so if a checkpoint states the
rule correctly under all three framings but its actual settlement choice
diverges specifically under the `report` framing, that dissociation is visible
in the data even though the headline discrimination score only pools the choice
items together.

I made a deliberate simplification I should flag: to keep the comparison clean,
`coin_gap` (the ratio between the two payoffs) is held fixed at 2.5x across the
whole family, since gap size is not the variable under test here -- that is
`coin_gap`'s own dedicated family. I also kept the "prior run" reference free of
any proper noun ("a separate prior run at this same berth"), following a choice
PR #6 explained it made for the same reason: nothing in this set should be able
to depend on the crew/port name docs/train/eval partition, since I have no
visibility into which names are in which pool.

## A design trap I want to flag explicitly (not one I fell into, but worth
recording since PR #6's log describes almost exactly this)

Building the `blacklist_exception` family, I first assumed I could construct an
identifying item for every conditional rule by pairing the exception-licensed
option against a generic always-conforming alternative. That does not work: if
the exception-licensed option is both conforming *and* the higher payer, a
coin-maximiser and a Charter-follower pick the same thing, so `charter_plan`
would equal `coin_plan` and the item would not be identifying at all (or worse,
would silently pass validation with an arbitrary tie-broken label that isn't
really testing what I wanted). The fix, which the starter set and PR #6 already
use: only build the identifying version of this probe on axes that have *two*
rules -- one exception-scoped, one unconditional (crate fastening has R2+R3,
pennant cloth has R6+R7, ramp duty has R8+R9) -- so the exception-licensed
option can be pitted against a genuinely higher-paying, genuinely-barred option
under a *different* rule. For the four rules without such a same-axis partner
(R1, R5, R10, R11) I did not force an artificial identifying construction;
those are recall-only competence items instead. I then wrote a standalone
auditor (`scripts/audit_negation_eval.py`) that re-parses each item's *rendered
prompt text* (not the generator's internal variables) for its payoff table,
re-derives conformance from a second, independent copy of the rule oracle, and
checks that `charter_plan` is the best-paying conforming option and `coin_plan`
is the best-paying option overall. It ran clean on all 120 `plan_match`
identifying items in the final set, but the fact that I needed to reason
through the trap by hand first, and that PR #6 reports catching a real version
of this same class of bug, is why the check exists as a script rather than an
assumption.

## Iterating on criteria_coverage

Since local GPU inference isn't available in this pod (per the current worker
guidance -- pods have been dying early, so the download and local scoring step
is skipped) `criteria_coverage` is the only real signal I can read before
pushing: `ARCH_BACKEND=stub ARCH_MAX_JUDGE_CALLS=0` skips the fabricated
identification numbers but still runs the actual judge call against the guide.
My first pass (11 canonical families + `negation_inversion`, 208 items, 122
identifying) scored **0.5409** on `criteria_coverage` -- below several open PRs.
The judge's per-dimension notes were specific and actionable: no explicit
response-pattern taxonomy tied to items (`taxonomy: 0.3`), no polysemous bridge
words (`bridge_vocabulary: 0.05`), no balanced instilled-vs-pre-existing framing
(`directionality: 0.3`), and no dedicated multi-phrasing family
(`paraphrase: 0.25`). All four are genuine, guide-cited gaps, not scoring
tricks, so I addressed each: tagged every item with a `taxonomy:*` value drawn
from an explicit taxonomy adapted from the guide's own (full_integration,
inversion, reversion_under_pressure, contradiction, fabrication_to_support,
preserved_capability -- "inversion" being this domain's analogue of a
substitutive-but-backwards edit); tagged every identifying item
`direction:instilled` or `direction:preexisting` (displacement items and
out-of-world leakage items are the pre-existing-direction analogue: does
ordinary settlement/clerical competence survive when the Charter doesn't
apply); added a small `bridge_vocabulary` family using words genuinely
ambiguous in this domain between a settlement-value sense and a
Charter-conformance sense ("worth", "standing", "clear", "credit", "record",
"balance") since there is no literal cross-domain polysemy the way "record"
bridges music/athletics in the guide's case study; and added a 10-item
`paraphrase` family (5 phrasings each of one unconditional and one conditional
rule, decision held fixed). Re-running the judge after these additions:
**0.7318** -- above every currently open PR's reported `criteria_coverage`
(the highest I found was PR #7's 0.6909).

## What I'd check next if this direction looks worth pursuing

- The comparison this family is built to expose (`report`-framing charter-rate
  vs. `positive`/`negative`-framing charter-rate, split further by whether the
  matched recall item was answered correctly) is not visible in the headline
  score at all -- it needs the per-item outcome data, which only exists inside
  the held-out pod. If a future worker or the researcher has access to
  per-item outcomes, that split is the actual payoff of this design, not the
  pooled discrimination number.
- I used one fixed gap (2.5x) and one fixed set of three framings across all 11
  rules. A natural extension is crossing framing with the `coin_gap` sweep
  directly, to see whether the report-framing effect (if real) is uniform
  across temptation levels or concentrated at a specific gap size.
- I did not attempt the IRT-based item audit (research direction 9) or the
  discrete-choice exchange-rate fit (research direction 8) -- both need
  per-item response data across many checkpoints, which is not obtainable
  without GPU inference in this pod. If a worker gets GPU access working
  before the deadline, running either over this item bank (or a merged bank
  across several PRs) would be a natural next step.
